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
from log_config import get_logger, install_crash_handlers, ROOT_NAME
from logging.handlers import RotatingFileHandler
import logging

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

# Parseable completion marker written after each successful video (resume skips these).
VIDEO_COMPLETE_PREFIX = "[video_complete]"
_VIDEO_COMPLETE_RE = re.compile(r"\[video_complete\]\s+(.+)\s*$")


def normalize_video_key(path) -> str:
    """Stable absolute path key for resume matching across log/CSV/CLI inputs."""
    try:
        return str(Path(path).resolve())
    except Exception:
        return os.path.normcase(os.path.normpath(str(path)))


def attach_output_dir_logging(output_dir: Path) -> Path | None:
    """Also write logs under ``{output_dir}/logs/`` (in addition to results/logs)."""
    try:
        logs_dir = Path(output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        name = f"sharkeye_{datetime.now().strftime('%m%d%Y_%H%M%S')}_{os.getpid()}.log"
        path = logs_dir / name
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setLevel(logging.getLogger(ROOT_NAME).level)
        handler.setFormatter(fmt)
        logging.getLogger(ROOT_NAME).addHandler(handler)
        logger.info("Logging to %s", path)
        return path
    except Exception as e:
        logger.warning("Could not attach output-dir log handler: %s", e)
        return None


def load_completed_videos_from_logs(output_dir: Path) -> set[str]:
    """Return video path keys marked ``[video_complete]`` in prior output-dir logs."""
    completed: set[str] = set()
    logs_dir = Path(output_dir) / "logs"
    if not logs_dir.is_dir():
        return completed
    for log_file in sorted(logs_dir.glob("*.log*")):
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Could not read log %s: %s", log_file, e)
            continue
        for line in text.splitlines():
            match = _VIDEO_COMPLETE_RE.search(line)
            if match:
                completed.add(normalize_video_key(match.group(1).strip()))
    return completed


def filter_incomplete_videos(video_paths, completed: set[str]) -> list:
    """Drop videos already marked complete in prior logs."""
    remaining = []
    skipped = 0
    for path in video_paths:
        key = normalize_video_key(path)
        if key in completed:
            skipped += 1
            logger.info("Skipping already completed video: %s", path)
            continue
        remaining.append(path)
    if skipped:
        logger.info("Resume: skipping %d completed video(s); %d remaining",
                    skipped, len(remaining))
    return remaining


def upsert_results_csv(output_dir: Path, video_key: str, new_rows: list[dict]) -> None:
    """Merge ``new_rows`` into ``output.csv``, replacing any prior rows for ``video_key``.

    Preserves rows from other videos so continued runs update the same CSV in place.
    Writes even when ``new_rows`` is empty (clears stale rows for a zero-detection video).
    """
    csv_path = Path(output_dir) / "output.csv"
    existing: list[dict] = []
    fieldnames: list[str] = []

    if csv_path.exists():
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                for row in reader:
                    row_key = normalize_video_key(row.get("Video name") or row.get("video_name") or "")
                    if row_key and row_key == video_key:
                        continue  # drop prior rows for this video; replaced below
                    existing.append(row)
        except OSError as e:
            logger.warning("Could not read existing CSV %s: %s", csv_path, e)

    if new_rows:
        for key in new_rows[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)
        existing.extend(new_rows)
    elif not fieldnames:
        # Nothing to write and no prior schema — skip creating an empty header-only file.
        logger.info("No tracks for %s; CSV unchanged (no prior schema)", video_key)
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
    logger.info("Updated %s (%d row(s) total) after %s",
                csv_path, len(existing), Path(video_key).name)


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
                    altitude=40.0, group_name=None, model=None, drone_settings=None,
                    completed_videos: set[str] | None = None):
    """Run inference on ``video_paths``, writing frames/masks under ``{output}/{group_name}/``.

    ``group_name`` is the input-folder label (e.g. ``sample_data``, ``2023``). When omitted,
    it falls back to each video's parent directory name.
    If ``model`` is provided it is reused (cloud-bucket path loads once across year folders).

    After each successful video: append/upsert rows into ``{output}/output.csv`` and log
    ``[video_complete]`` so later runs can skip it. Failures are logged and do not mark complete.
    """
    if model is None:
        device = select_torch_device()
        logger.info(f"Using device: {device}")
        model = YOLO(MODEL_PATH).to(device)
        warmup_models(model, sam_model_path=sam_model_path)
    if drone_settings is None:
        drone_settings = load_drone_settings()

    completed = completed_videos if completed_videos is not None else set()
    output_dir = Path(current_output_dir)

    videos_tqdm = tqdm(video_paths)
    all_track_results = []
    for path in videos_tqdm:
        video_key = normalize_video_key(path)
        if video_key in completed:
            logger.info("Skipping already completed video: %s", path)
            continue

        videos_tqdm.set_description(f"Processing {path}")
        try:
            processor = HeadlessVideoProcessor(
                path, model, current_output_dir, sam_model_path=sam_model_path,
                drone_type=drone_type, altitude=altitude, drone_settings=drone_settings,
                group_name=group_name,
            )
            rows = processor.run()
            # Normalize Video name to the resolved key so CSV resume matching is stable.
            for row in rows:
                row["Video name"] = video_key
            upsert_results_csv(output_dir, video_key, rows)
            logger.info("%s %s", VIDEO_COMPLETE_PREFIX, video_key)
            completed.add(video_key)
            all_track_results.extend(rows)
        except Exception:
            logger.exception("Failed processing %s — will retry on a later run", path)

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


def cloud_bucket_inference():
    """Entry point: local file/dir via ``--input_dir``, else cloud-bucket scan."""
    install_crash_handlers()
    args = parse_args()

    sam_model_path = Path(args.sam_model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attach_output_dir_logging(output_dir)

    # Resume: skip videos that prior runs in this output_dir already finished successfully.
    completed_videos = load_completed_videos_from_logs(output_dir)
    if completed_videos:
        logger.info("Found %d completed video(s) in prior logs under %s",
                    len(completed_videos), output_dir / "logs")

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
        video_paths = filter_incomplete_videos(video_paths, completed_videos)
        if max_videos is not None:
            video_paths = video_paths[:max_videos]
        group_name = input_path.name if input_path.is_dir() else input_path.parent.name
        if not group_name or group_name in (".", ".."):
            group_name = "videos"
        logger.info(f"Found {len(video_paths)} video(s) to process under {args.input_dir} "
                    f"(output group: {group_name})")
        if not video_paths:
            logger.info("Nothing left to process.")
            return
        mass_prediction(
            video_paths=video_paths,
            current_output_dir=output_dir,
            sam_model_path=sam_model_path,
            drone_type=drone_type,
            altitude=altitude,
            group_name=group_name,
            completed_videos=completed_videos,
        )
        return

    # Cloud-bucket mirror under ./sharkeye_videos (default when --input_dir omitted).
    # One output group per year/subdir: headless_predictions/2023/, /2024/, ...
    videos_dict = scan_bucket_for_videos()
    remaining_videos = max_videos

    device = select_torch_device()
    logger.info(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)
    warmup_models(model, sam_model_path=sam_model_path)
    drone_settings = load_drone_settings()

    for subdir, paths in videos_dict.items():
        if remaining_videos is not None and remaining_videos <= 0:
            break
        paths = filter_incomplete_videos(paths, completed_videos)
        if remaining_videos is not None and len(paths) > remaining_videos:
            paths = paths[:remaining_videos]
        if not paths:
            continue
        mass_prediction(
            video_paths=paths,
            current_output_dir=output_dir,
            sam_model_path=sam_model_path,
            drone_type=drone_type,
            altitude=altitude,
            group_name=subdir,
            model=model,
            drone_settings=drone_settings,
            completed_videos=completed_videos,
        )
        if remaining_videos is not None:
            remaining_videos -= len(paths)


def main():
    """Alias for the shared entry point (local path or cloud bucket)."""
    cloud_bucket_inference()


if __name__ == '__main__':
    cloud_bucket_inference()