import multiprocessing
import sys
import os
import time
import argparse
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import csv
from tqdm import tqdm
import re
from utility import resource_path, get_results_dir
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
from segmentation.segmentation_model import run_prediction, calculate_shark_length_from_pixel, find_pixel_length, draw_mask
# CustomTracker + length calibration now come from the shared `tracking` module (previously
# a diverged copy lived here with the old ~3x-inflated bbox length calc and no best-frame
# segmentation). One source of truth across the GUI, mass_prediction, and this CLI.
from tracking import CustomTracker, resolve_fov_radians, load_drone_settings
from segment_anything import sam_model_registry, SamPredictor 


# Use a constant for the model path
MODEL_PATH = resource_path('model_weights/runs-detect-train-weights-best.pt')

# Use a constant for file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


    
class HeadlessVideoProcessor():
    progress_update = 0
    processing_complete = {}
    
    def __init__(self, video_path, model, output_dir, sam_model_path,
                 drone_type="Air 2S", altitude=40.0, drone_settings=None):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.sam_model_path = sam_model_path
        self.drone_type = drone_type
        self.altitude = float(altitude)
        self.drone_settings = drone_settings if drone_settings is not None else load_drone_settings()

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

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'false_positives'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'masks'), exist_ok=True)

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
        custom_tracker.save_best_frames(self.output_dir, self.video_path)

        all_track_info = [] 

        for track_id, track in custom_tracker.tracks.items():
            meets_thresholds = (len(track['confidences']) >= 10 and 
                                np.mean(track['confidences']) > 0.4)
            
            track_info = {
                'Video name': self.video_path,
                'Track Id': track_id,
                'Length (ft)': track['longest_length'],   # canonical length (SAM, or bbox if sub-threshold)
                'Highest Conf Timestamp': CustomTracker._format_timestamp(track['longest_timestamp']),
                'Highest Confidence': max(track['confidences']),
                'Average Confidence': np.mean(track['confidences']),
                'Lowest Confidence': min(track['confidences']),
                'Highest Confidence Length': track['longest_length'],
                'Number of Detections': len(track['confidences']),
                'Meets Thresholds': meets_thresholds,
                'manual_length_px': '',
                'manual_length_ft': '',
            }

            all_track_info.append(track_info)
        
        return all_track_info 
    
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
    
def mass_prediction(video_paths, current_output_dir, sam_model_path, drone_type="Air 2S", altitude=40.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)
    drone_settings = load_drone_settings()

    videos_tqdm = tqdm(video_paths)
    all_track_results = []
    for path in videos_tqdm:
        videos_tqdm.set_description(f"Processing {path}")
        processor = HeadlessVideoProcessor(path, model, current_output_dir, sam_model_path=sam_model_path,
                                           drone_type=drone_type, altitude=altitude, drone_settings=drone_settings)
        all_track_results.extend(processor.run())

    return all_track_results

def parse_args():
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--sam_model_path', type=str, required=True, help="Path to segment anything model")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing videos to process (.mp4/.mov, case-insensitive)')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
    parser.add_argument('--drone', type=str, default='Air 2S', help='Drone model, for per-video FOV / length calibration')
    parser.add_argument('--altitude', type=float, default=40.0, help='Flight altitude in meters, for length calibration')
    return parser.parse_args()

def main():
    install_crash_handlers()
    args = parse_args()

    sam_model_path = Path(args.sam_model_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Recursively find every video under input_dir, matching the documented
    # "--input_dir <dir_of_mp4s>" contract (case-insensitive extension, de-duplicated
    # and sorted for a stable order). Previously this was hardcoded to a specific
    # "*/Transect/*" research layout, so pointing the tool at an arbitrary folder of
    # .mp4s silently found nothing.
    video_exts = {".mp4", ".mov"}
    video_paths = sorted({p for p in input_dir.rglob("*") if p.suffix.lower() in video_exts})
    logger.info(f"Found {len(video_paths)} video(s) under {input_dir}")

    # Run prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    results = mass_prediction(video_paths=video_paths, current_output_dir=output_dir, sam_model_path=sam_model_path,
                              drone_type=args.drone, altitude=args.altitude)

    # Save results to CSV
    if results:
        csv_path = output_dir / "output.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results saved to {csv_path}")
    else:
        logger.warning("No valid tracks were found.")

if __name__ == '__main__':
    main()       