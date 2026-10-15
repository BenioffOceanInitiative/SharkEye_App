import multiprocessing
import sys
import os
import time
import argparse
import cv2
import torch
import json
from ultralytics import YOLO
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import csv
from tqdm import tqdm
import re
from utility import resource_path, get_results_dir, select_torch_device
from log_config import get_logger, install_crash_handlers

logger = get_logger("sharkeye.headless")
from frame_sampling import (iter_sampled_frames, parse_detections, format_sampling_stats,
                            format_sampling_timeline)
try:
    # PyAV-backed keyframe sampling, on by default; falls back to grab-through when
    # unavailable or disabled (SHARKEYE_KEYFRAME_SAMPLING=0). See keyframe_sampling.
    from keyframe_sampling import try_keyframe_sampler
except Exception:  # pragma: no cover - PyAV missing / import failure
    def try_keyframe_sampler(*_args, **_kwargs):
        return None
import signal
import json
import requests
import zipfile
import shutil
import tempfile
import io
import math
from pathlib import Path
from segmentation.segmentation_model import (
    run_prediction, calculate_shark_length_from_pixel, find_pixel_length, draw_mask,
    get_sam_predictor,
)
# CustomTracker + length calibration now come from the shared `tracking` module (previously
# a diverged copy lived here with the old ~3x-inflated bbox length calc and no best-frame
# segmentation). One source of truth across the GUI, mass_prediction, and this CLI.
from tracking import (
    CustomTracker, resolve_fov_radians, load_drone_settings,
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT,
)
from segment_anything import sam_model_registry, SamPredictor 


# Use a constant for the model path
MODEL_PATH = resource_path('model_weights/runs-detect-train-weights-best.pt')

# Use a constant for file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = {'.mp4', '.mov'}


    
class HeadlessVideoProcessor():
    progress_update = 0
    processing_complete = {}
    
    def __init__(self, video_path, model, output_dir, sam_model_path,
                 drone_type="Air 2S", altitude=40.0, drone_settings=None,
                 group_name=None):
        super().__init__()
        self.video_path = video_path
        self.model = model
        # Base run folder (holds output.csv). frames/masks live under
        # {output_dir}/{group_name}/ — one group per input folder (e.g. sample_data, 2023).
        self.output_dir = output_dir
        self.group_name = group_name or Path(video_path).parent.name or "videos"
        self.group_output_dir = os.path.join(output_dir, self.group_name)
        self.sam_model_path = sam_model_path
        self.drone_type = drone_type
        self.altitude = float(altitude)
        self.drone_settings = drone_settings if drone_settings is not None else load_drone_settings()

    def frame_image_relpath(self, track_id):
        """Path to a track's saved frame image, relative to the base output_dir."""
        filename = f"{Path(self.video_path).name}_{track_id}.jpg"
        return os.path.join(self.group_name, 'frames', filename)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # used for keyframe-mode timestamps
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        custom_tracker = CustomTracker(sam_model_path=self.sam_model_path)
        # Per-video FOV so length math matches the GUI (previously defaulted).
        fov = resolve_fov_radians(self.drone_type, video_width, video_height, self.drone_settings)
        if fov is not None:
            custom_tracker.fov_radians = fov
            custom_tracker.drone_altitude = self.altitude
        else:
            logger.warning(f"[gsd] {Path(self.video_path).name}: no FOV for drone={self.drone_type!r} "
                           f"@ {video_width}x{video_height}; using default {custom_tracker.fov_radians:.4f}rad")

        os.makedirs(os.path.join(self.group_output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.group_output_dir, 'masks'), exist_ok=True)

        detection_threshold = 0.4

        # Sequential forward sampling. Keyframe-scan when enabled + decodable, else
        # grab-through; no preview needed here.
        sampler = try_keyframe_sampler(self.video_path, logger)
        use_keyframe = sampler is not None
        if not use_keyframe:
            sampler = iter_sampled_frames(cap)
        had_detection = None
        sampling_stats = {}
        infer_start = time.perf_counter()
        try:
            while True:
                frame_num, frame = sampler.send(had_detection)

                results = self.model(frame, classes=[0], verbose=False)
                detections = parse_detections(results, detection_threshold)
                had_detection = bool(detections)

                if had_detection:
                    timestamp = (frame_num / fps * 1000.0) if use_keyframe else cap.get(cv2.CAP_PROP_POS_MSEC)
                    custom_tracker.update(detections, frame, timestamp)

                self.progress_update = int((frame_num + 1) / total_frames * 100)
        except StopIteration as stop:
            sampling_stats = stop.value or {}

        infer_time = time.perf_counter() - infer_start
        cap.release()

        # Adaptive frame-sampling analytics for troubleshooting/throughput tuning.
        video_name = Path(self.video_path).name
        logger.info(format_sampling_stats(video_name, infer_time, sampling_stats))
        timeline = format_sampling_timeline(video_name, sampling_stats)
        if timeline:
            logger.info(timeline)
        # Group folder: {output_dir}/{group_name}/frames|masks/ (shared across videos in the batch)
        custom_tracker.save_best_frames(self.group_output_dir, self.video_path)
        self.write_frame_labels(custom_tracker)

        all_track_info = [] 

        for track_id, track in custom_tracker.tracks.items():
            meets_thresholds = (len(track['confidences']) >= 10 and 
                                np.mean(track['confidences']) > 0.4)
            # Only significant tracks get a frames/ image (same gate as save_best_frames).
            frame_path = ''
            if custom_tracker.is_significant_track(track) and track.get('best_frame') is not None:
                frame_path = self.frame_image_relpath(track_id)

            # Longest-box length from the longest_* frame/box (not best_pos). Recompute
            # feet from longest_pos because save_best_frames overwrites longest_length with SAM.
            longest_pos = track.get('longest_pos') or (track['positions'][0] if track['positions'] else None)
            longest_frame = track.get('longest_frame')
            if longest_pos is not None and longest_frame is not None:
                _, _, lw, lh = longest_pos
                longest_length_px = math.hypot(lw, lh)
                longest_length_ft = custom_tracker._bbox_length_ft(
                    longest_pos, longest_frame.shape[1], longest_frame.shape[0])
            else:
                longest_length_px = ''
                longest_length_ft = ''

            seg_px = track.get('segmentation_length_px')
            seg_ft = track.get('segmentation_length_ft')
            # Empty cells when SAM did not run (sub-threshold tracks).
            segmentation_length_px = '' if seg_px is None else seg_px
            segmentation_length_ft = '' if seg_ft is None else seg_ft
            
            track_info = {
                'Video name': self.video_path,
                'Drone': self.drone_type,
                'Altitude': self.altitude,
                'Track Id': track_id,
                'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                'Highest Confidence': max(track['confidences']),
                'Average Confidence': np.mean(track['confidences']),
                'Lowest Confidence': min(track['confidences']),
                'Number of Detections': len(track['confidences']),
                'Meets Thresholds': meets_thresholds,
                'longest_length_px': longest_length_px,
                'longest_length_ft': longest_length_ft,
                'segmentation_length_px': segmentation_length_px,
                'segmentation_length_ft': segmentation_length_ft,
                'Frame Path': frame_path,
                'manual_length_px': '',
                'manual_length_ft': '',
            }

            all_track_info.append(track_info)
        
        return all_track_info 

    def write_frame_labels(self, tracker, class_id=0):
        """Write a YOLO ``.txt`` next to each image saved under the group's ``frames/``.

        Uses the same naming as ``CustomTracker.save_best_frames``
        (``{video_basename}_{track_id}.jpg``) and the track's ``best_pos`` box
        (highest-confidence detection on that frame), normalized to 0–1.
        """
        frames_dir = os.path.join(self.group_output_dir, 'frames')
        video_basename = Path(self.video_path).name
        labels_written = 0

        for track_id, track in tracker.tracks.items():
            if not tracker.is_significant_track(track):
                continue
            seg_frame = track.get('best_frame')
            seg_bbox = track.get('best_pos') or (track['positions'][0] if track['positions'] else None)
            if seg_frame is None or seg_bbox is None:
                continue

            x, y, w, h = seg_bbox
            height, width = seg_frame.shape[:2]
            if width <= 0 or height <= 0:
                continue

            label_path = os.path.join(frames_dir, f"{video_basename}_{track_id}.txt")
            line = (
                f"{int(class_id)} "
                f"{x / width:.6f} {y / height:.6f} "
                f"{w / width:.6f} {h / height:.6f}\n"
            )
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(line)
            labels_written += 1

        logger.info(f"[labels] wrote {labels_written} YOLO label(s) under {self.group_name}/frames/")
    
    @staticmethod
    def draw_bounding_boxes(frame, detections):
        frame_with_boxes = frame.copy()
        for x, y, w, h, confidence in detections:
            cv2.rectangle(frame_with_boxes, 
                          (int(x - w/2), int(y - h/2)), 
                          (int(x + w/2), int(y + h/2)), 
                          (0, 255, 0), 2)
            label = f"Shark: {confidence:.2f}"
            cv2.putText(frame_with_boxes, label, (int(x - w/2), int(y - h/2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        return frame_with_boxes
    
def warmup_models(model, sam_model_path=None):
    """One-shot YOLO (+ optional SAM) warmup so the first video doesn't pay compile cost.

    Mirrors ``ModelLoader.run`` in ``sharkeye_app.py``.
    """
    try:
        warmup_frame = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), dtype=np.uint8)
        model(warmup_frame, classes=[0], verbose=False)
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup skipped: {e}")

    try:
        if sam_model_path:
            predictor = get_sam_predictor(Path(sam_model_path))
        else:
            predictor = get_sam_predictor()
        predictor.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
        predictor.reset_image()  # free the warmup embedding; keep the loaded weights
        logger.info("SAM warmup complete")
    except Exception as e:
        logger.warning(f"SAM warmup skipped: {e}")


def mass_prediction(video_paths, current_output_dir, sam_model_path, drone_type="Air 2S",
                    altitude=40.0, group_name=None, model=None, drone_settings=None):
    """Run inference on ``video_paths``, writing frames/masks under ``{output}/{group_name}/``.

    ``group_name`` is the input-folder label (e.g. ``sample_data``, ``2023``). When omitted,
    it falls back to each video's parent directory name.
    If ``model`` is provided it is reused (cloud-bucket path loads once across year folders).
    """
    if model is None:
        device = select_torch_device()
        logger.info(f"Using device: {device}")
        model = YOLO(MODEL_PATH).to(device)
        warmup_models(model, sam_model_path=sam_model_path)
    if drone_settings is None:
        drone_settings = load_drone_settings()

    videos_tqdm = tqdm(video_paths)
    all_track_results = []
    for path in videos_tqdm:
        videos_tqdm.set_description(f"Processing {path}")
        processor = HeadlessVideoProcessor(
            path, model, current_output_dir, sam_model_path=sam_model_path,
            drone_type=drone_type, altitude=altitude, drone_settings=drone_settings,
            group_name=group_name,
        )
        all_track_results.extend(processor.run())

    return all_track_results

# Cloud Bucket Inference Operations
def scan_bucket_for_videos(
    video_paths: str = "video_paths.json",
    refresh_all: bool = False,
    subfolders: list[str] | None = None,
) -> dict[str, list[str]]:
    """Scan video directories and update cached video paths.

    Behavior:
        refresh_all=True:
            Rescan all valid subdirectories.

        subfolders provided:
            Rescan only specified subdirectories.

        Otherwise:
            Rescan only the last valid subdirectory.

    Existing cached paths for unscanned directories are preserved.
    """

    valid_subdirs = [
        "2019", "2020", "2021", "2022", "2023",
        "2023_Missed_Sharks_Trimmed",
        "2024", "2025", "2026"
    ]

    base_path = Path("./sharkeye_videos")
    json_path = Path(video_paths)

    # Determine which folders to scan
    if refresh_all:
        folders_to_scan = valid_subdirs

    elif subfolders is not None:
        invalid_folders = set(subfolders) - set(valid_subdirs)

        if invalid_folders:
            raise ValueError(
                f"Invalid subfolders: {sorted(invalid_folders)}"
            )

        folders_to_scan = subfolders

    else:
        folders_to_scan = [valid_subdirs[-1]]

    # Load existing paths unless performing a full refresh
    if not refresh_all and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            videos_dict = json.load(f)
    else:
        videos_dict = {}

    # Scan selected directories
    for subdir in folders_to_scan:
        full_subdir_path = base_path / subdir

        if not full_subdir_path.is_dir():
            raise FileNotFoundError(
                f"Video directory not found: {full_subdir_path}"
            )

        paths = [
            str(path)
            for path in full_subdir_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
        ]

        videos_dict[subdir] = sorted(paths)

        logger.info(
            f"Scanned {subdir}: found {len(paths)} videos"
        )
    # Save updated paths to JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(videos_dict, f, indent=4)

    return videos_dict

def save_video_paths():
    videos_dict = scan_bucket_for_videos()
    videos_list = [item for videos in videos_dict.values() for item in videos]
    print(f'Found {len(videos_list)} videos across {len(videos_dict.keys())} subfolders')
    for subdir, paths in videos_dict.items():
        print(f'{subdir}: {len(paths)} videos found')

    with open("video_paths.json", "w") as file:
        json.dump(videos_dict, file)
    return videos_dict

def read_video_paths():
    with open("video_paths.json", "r") as file:
        videos_dict = json.load(file)
    return videos_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--sam_model_path', type=str, required=True, help="Path to segment anything model")
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Local video file (.mp4/.mov) or directory of videos. Takes precedence over --cloud_bucket')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
    parser.add_argument('--drone', type=str, default='Air 2S', help='Drone model, for per-video FOV / length calibration')
    parser.add_argument('--altitude', type=float, default=40.0, help='Flight altitude in meters, for length calibration')
    parser.add_argument('--max_videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--cloud_bucket', action='store_true', default=False,
                        help='Scan ./sharkeye_videos cloud-bucket mirror (used when --input_dir is omitted)')
    return parser.parse_args()


def collect_local_videos(input_path: Path) -> list[Path]:
    """Resolve ``input_path`` to a list of video files.

    Accepts a single ``.mp4``/``.mov`` file or a directory (recursive scan).
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video file: {input_path} (expected .mp4/.mov)"
            )
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return sorted({
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    })


def write_results_csv(output_dir: Path, results) -> None:
    if results:
        csv_path = output_dir / "output.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results saved to {csv_path}")
    else:
        logger.warning("No valid tracks were found.")


def cloud_bucket_inference():
    """Entry point: local file/dir via ``--input_dir``, else cloud-bucket scan."""
    install_crash_handlers()
    args = parse_args()

    sam_model_path = Path(args.sam_model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_videos = args.max_videos
    drone_type = args.drone
    altitude = args.altitude

    # Local mode: --input_dir may be a single video file or a folder of videos.
    # Output group folder is the input directory name (e.g. sample_data/), not per-video.
    if args.input_dir:
        input_path = Path(args.input_dir)
        try:
            video_paths = collect_local_videos(input_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return
        if max_videos is not None:
            video_paths = video_paths[:max_videos]
        group_name = input_path.name if input_path.is_dir() else input_path.parent.name
        if not group_name or group_name in (".", ".."):
            group_name = "videos"
        logger.info(f"Found {len(video_paths)} video(s) under {args.input_dir} "
                    f"(output group: {group_name})")
        results = mass_prediction(
            video_paths=video_paths,
            current_output_dir=output_dir,
            sam_model_path=sam_model_path,
            drone_type=drone_type,
            altitude=altitude,
            group_name=group_name,
        )
        write_results_csv(output_dir, results)
        return

    # Cloud-bucket mirror under ./sharkeye_videos (default when --input_dir omitted).
    # One output group per year/subdir: headless_predictions/2023/, /2024/, ...
    videos_dict = scan_bucket_for_videos()
    all_track_results = []
    remaining_videos = max_videos

    device = select_torch_device()
    logger.info(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)
    warmup_models(model, sam_model_path=sam_model_path)
    drone_settings = load_drone_settings()

    for subdir, paths in videos_dict.items():
        if remaining_videos is not None and remaining_videos <= 0:
            break
        if remaining_videos is not None and len(paths) > remaining_videos:
            paths = paths[:remaining_videos]
        all_track_results.extend(
            mass_prediction(
                video_paths=paths,
                current_output_dir=output_dir,
                sam_model_path=sam_model_path,
                drone_type=drone_type,
                altitude=altitude,
                group_name=subdir,
                model=model,
                drone_settings=drone_settings,
            )
        )
        if remaining_videos is not None:
            remaining_videos -= len(paths)

    write_results_csv(output_dir, all_track_results)


def main():
    """Alias for the shared entry point (local path or cloud bucket)."""
    cloud_bucket_inference()


if __name__ == '__main__':
    cloud_bucket_inference()