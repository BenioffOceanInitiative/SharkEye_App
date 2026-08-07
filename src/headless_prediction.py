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
from segment_anything import sam_model_registry, SamPredictor 

# Add these constants for length calculation
DRONE_ALTITUDE_M = 40
SENSOR_WIDTH_MM = 13.2
FOCAL_LENGTH_MM = 28
MODEL_WIDTH = MODEL_HEIGHT = 640
# ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 3840, 2160
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512
ASPECT_RATIO = ORIGINAL_WIDTH / ORIGINAL_HEIGHT

# Use a constant for the model path
MODEL_PATH = resource_path('model_weights/runs-detect-train-weights-best.pt')

# Use a constant for file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def calculate_gsd(altitude, sensor_width, focal_length, image_width):
    """Calculate Ground Sample Distance (GSD)"""
    return (altitude * sensor_width) / (focal_length * image_width)

GSD = calculate_gsd(DRONE_ALTITUDE_M, SENSOR_WIDTH_MM, FOCAL_LENGTH_MM, MODEL_WIDTH)

def calculate_shark_length(bbox):
    """Calculate shark length in feet based on bounding box"""
    _, _, _, height = bbox
    adjusted_height = height * (MODEL_HEIGHT / MODEL_WIDTH)
    length_m = adjusted_height * GSD
    # depth_correction_factor = (1 + DRONE_ALTITUDE_M) / DRONE_ALTITUDE_M
    return length_m * 3.28084 # * depth_correction_factor  # Convert meters to feet

class CustomTracker:
    # Cost returned for a track whose last sighting is older than the grace period:
    # large enough to always exceed distance_threshold (so the Hungarian pairing is
    # rejected and the detection starts a new id), but finite — linear_sum_assignment
    # rejects inf/NaN.
    _UNMATCHABLE_COST = 1e9

    # The match gate (distance_threshold) grows by this many pixels per second of dropout:
    # position uncertainty and extrapolation error both increase with the gap, so a longer
    # dropout should tolerate a larger jump before the detection is ruled a different
    # object. See _match_threshold.
    _GATE_GROWTH_PX_PER_S = 60

    def __init__(self, sam_model_path, distance_threshold=250, min_frames=5, confidence_threshold=0.4):
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        # Re-association grace period, in video time (ms). A shark's detection often
        # blinks out for a moment and resurfaces after moving a short distance; without a
        # grace window the tracker spawns a fresh id each time and fragments one animal
        # into several. If a detection reappears within this window of a track's last
        # sighting we re-link it to that track — matching against a velocity-extrapolated
        # prediction (see _predict_new_position); past the window the track is treated as
        # gone and a new id is started (see _calculate_cost). Measured in video time, not
        # frame count, because a dropout advances no frame counter — update() only runs on
        # frames that had a detection — so only the timestamp reflects how long the object
        # was actually missing. Sized to cover real shark dropouts (glint, a side-on turn,
        # a brief submersion), which routinely run past 2s — the keyframe sampler already
        # keeps looking densely for ~2.5s (see keyframe_sampling._DENSE_HOLD_SECONDS), so a
        # shorter grace here re-splits a shark the sampler was still tracking.
        self.reassociation_grace_ms = 4000
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.unique_sharks = 0
        self.last_reported_sharks = 0
        self.sam_model_path = sam_model_path

    def update(self, detections, frame, timestamp):
        active_tracks = set()
        new_unique_shark = False

        if not self.tracks:
            for detection in detections:
                self._create_new_track(detection, frame, timestamp)
                active_tracks.add(self.next_id - 1)
            new_unique_shark = True
            self.unique_sharks = 1
        else:
            predicted_positions = {track_id: self._predict_new_position(track, timestamp)
                                   for track_id, track in self.tracks.items()}

            cost_matrix = np.array([[self._calculate_cost(track, det, predicted_positions[track_id], timestamp)
                                     for det in detections]
                                    for track_id, track in self.tracks.items()])
            
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            track_ids = list(self.tracks.keys())
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                track_id = track_ids[track_idx]
                elapsed_ms = timestamp - self.tracks[track_id]['timestamps'][-1]
                if cost_matrix[track_idx, detection_idx] < self._match_threshold(elapsed_ms):
                    self._update_track(track_id, detections[detection_idx], frame, timestamp)
                    active_tracks.add(track_id)
                else:
                    self._create_new_track(detections[detection_idx], frame, timestamp)
                    active_tracks.add(self.next_id - 1)

            unassigned_detections = set(range(len(detections))) - set(detection_indices)
            for i in unassigned_detections:
                self._create_new_track(detections[i], frame, timestamp)
                active_tracks.add(self.next_id - 1)

        current_unique_sharks = self._count_significant_tracks()
        if current_unique_sharks > self.unique_sharks:
            new_unique_shark = True
            self.unique_sharks = current_unique_sharks

        for track_id in self.tracks:
            self.tracks[track_id]['frames_since_last_detection'] = 0 if track_id in active_tracks else self.tracks[track_id]['frames_since_last_detection'] + 1

        if self.unique_sharks != self.last_reported_sharks:
            logger.info("Shark detected — unique shark count: %d", self.unique_sharks)
            self.last_reported_sharks = self.unique_sharks

        return active_tracks

    def _create_new_track(self, detection, frame, timestamp):
        x, y, w, h, confidence = detection
        length = (calculate_shark_length((x, y, w, h)))
        self.tracks[self.next_id] = {
            'id': self.next_id,
            'unique_id': self.next_id,
            'positions': deque([(x, y, w, h)], maxlen=100),
            'confidences': deque([confidence], maxlen=100),
            # Fresh un-mutated buffer per retrieve(); store references, not full-frame copies.
            'frames': deque([frame], maxlen=100),
            'timestamps': deque([timestamp], maxlen=100),
            'lengths': deque([length], maxlen=100),
            'best_frame': frame,
            'best_conf': confidence,
            'best_timestamp': timestamp,
            'best_length': length,
            'longest_frame': frame,
            'longest_conf': confidence, 
            'longest_timestamp': timestamp,
            'longest_length': length,         
            'longest_position': (x, y, w, h),   
            'frames_since_last_detection': 0,
            'velocity': np.array([0, 0]),
            'label': 'Shark',
            'track_frames': []
        }
        self.next_id += 1

    def _update_track(self, track_id, detection, frame, timestamp):
        x, y, w, h, confidence = detection
        length = (calculate_shark_length((x, y, w, h)))
        track = self.tracks[track_id]
        
        # Store frame with bounding box
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, 
                     (int(x - w/2), int(y - h/2)), 
                     (int(x + w/2), int(y + h/2)), 
                     (0, 255, 0), 2)
        track['track_frames'].append(frame_with_box)
        
        track['positions'].append((x, y, w, h))
        track['confidences'].append(confidence)
        track['frames'].append(frame)
        track['timestamps'].append(timestamp)
        track['lengths'].append(length)

        if confidence > track['best_conf']:
            track['best_conf'] = confidence
            track['best_frame'] = frame  # fresh un-mutated buffer; no copy needed
            track['best_timestamp'] = timestamp
            track['best_length'] = length

        if confidence > .8 and length > track['longest_length']:
            track['longest_conf'] = confidence
            track['longest_frame'] = frame  # fresh un-mutated buffer; no copy needed
            track['longest_timestamp'] = timestamp
            track['longest_length'] = length
            track['longest_position'] = (x, y, w, h)

        if len(track['positions']) > 1:
            prev_pos = np.array(track['positions'][-2][:2])
            curr_pos = np.array([x, y])
            # Velocity in pixels-per-ms so _predict_new_position can extrapolate over the
            # real elapsed time of a dropout (frame counts don't advance while a shark is
            # undetected). Guard duplicate / zero-dt samples: keep the prior velocity
            # rather than dividing by zero.
            dt_ms = track['timestamps'][-1] - track['timestamps'][-2]
            if dt_ms > 0:
                track['velocity'] = (curr_pos - prev_pos) / dt_ms

    @staticmethod
    def _format_timestamp(milliseconds):
        """Format timestamp in MM:SS format for CSV"""
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%M:%S")

    @staticmethod
    def _format_timestamp_filename(milliseconds):
        """Format timestamp in MMSS format for filename"""
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%M%S")

    def save_best_frames(self, output_dir, video_path):
        """Save best frames for each significant track"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        images_saved = 0
        
        for track_id, track in self.tracks.items():
            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])

            # SAM is the expensive per-track step; run it only on significant tracks.
            # Sub-threshold tracks (likely false positives) keep their bbox-estimated
            # length and get no mask.
            is_significant = num_frames >= self.min_frames and avg_confidence > self.confidence_threshold
            if not is_significant:
                logger.debug('Track detected below threshold')

            longest_frame = track['longest_frame']
            longest_timestamp = track['longest_timestamp']
            longest_confidence = track['longest_conf']
            longest_length = track['longest_length']
            longest_position = track['longest_position']

            if longest_frame is None:
                continue

            timestamp_str = self._format_timestamp_filename(longest_timestamp)
            x, y, w, h = longest_position

            if is_significant:
                # Use segmentation model to generate lengths (SAM/draw_mask expect RGB;
                # longest_frame is BGR from the decoder).
                rgb_frame = cv2.cvtColor(longest_frame, cv2.COLOR_BGR2RGB)
                mask = run_prediction(rgb_frame, (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)), checkpoint_path=self.sam_model_path)
                pixel_length = find_pixel_length(mask, draw_line=False, viz_name = f'{video_name}-viz')
                segmentation_length = calculate_shark_length_from_pixel(pixel_length, original_width=longest_frame.shape[1], original_height=longest_frame.shape[0])
                track['longest_length'] = segmentation_length
                longest_length = track['longest_length']

                mask_overlay = draw_mask(mask, rgb_frame)
                track['mask_overlay'] = mask_overlay

            feet, inches = divmod(longest_length, 1)
            length_str = f"{int(feet)}ft{int(inches * 12)}in"

            avg_conf_int = int(avg_confidence * 100)
            longest_conf_int = int(longest_confidence * 100)

            filename = f"{video_name}_shark{track_id}_time{timestamp_str}_det{num_frames}_avgConf{avg_conf_int}_bestConf{longest_conf_int}_len{length_str}.jpg"

            # Save the raw best-confidence frame. The annotated "bounding_boxes/" copy is no
            # longer written: a box reconstructable from the label/CSV, never read back.
            cv2.imwrite(os.path.join(output_dir, 'frames', filename), longest_frame)

            # Mask only exists for segmented (significant) tracks.
            if is_significant:
                cv2.imwrite(os.path.join(output_dir, 'masks', filename), mask_overlay)

            images_saved += 1

        logger.info(f"[segmentation] saved {images_saved} track image(s)")

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1
        self.unique_sharks = 0

    def _predict_new_position(self, track, timestamp):
        """Predict where the track's object should be at `timestamp`.

        Velocity is stored in pixels-per-ms (see _update_track), so we extrapolate the
        last known position by the *elapsed video time* since the last detection. That is
        what lets a shark which dropped out for a moment and kept swimming re-link to its
        existing track: over a ~1s gap the prediction moves with the animal instead of
        sitting at its stale last position. Time-based (not frame count) because the
        tracker only sees frames that had a detection — a dropout advances no frame
        counter, so only the timestamp reflects how long the object was missing.
        """
        if not track['positions']:
            return np.array([0, 0])  # Default prediction if no positions available
        last_pos = np.array(track['positions'][-1][:2])
        elapsed_ms = timestamp - track['timestamps'][-1]
        return last_pos + track['velocity'] * elapsed_ms

    def _match_threshold(self, elapsed_ms):
        """Distance gate for re-linking a detection to a track, widening with the dropout.

        Near-continuous detections use the base distance_threshold; as the gap grows, both
        the animal's possible travel and the extrapolation error grow, so the gate expands
        linearly with elapsed video time. Kept well below _UNMATCHABLE_COST so a
        past-grace track (see _calculate_cost) is still rejected."""
        return self.distance_threshold + self._GATE_GROWTH_PX_PER_S * (elapsed_ms / 1000.0)

    def _calculate_cost(self, track, detection, predicted_position, timestamp):
        """Cost for the Hungarian assignment.

        Within the re-association grace period the cost is just the distance from the
        (time-extrapolated) predicted position to the detection, so a shark that briefly
        dropped out re-links to its existing id. Past the grace period the track is
        considered gone and made effectively unmatchable, so the detection starts a fresh
        id rather than resurrecting a stale track."""
        elapsed_ms = timestamp - track['timestamps'][-1]
        if elapsed_ms > self.reassociation_grace_ms:
            return self._UNMATCHABLE_COST
        return np.linalg.norm(predicted_position - np.array(detection[:2]))

    def _count_significant_tracks(self):
        """Count tracks that meet the criteria for being a significant detection"""
        return sum(1 for track in self.tracks.values() 
                   if len(track['positions']) >= self.min_frames 
                   and np.mean(track['confidences']) > self.confidence_threshold)
    
class HeadlessVideoProcessor():
    progress_update = 0
    processing_complete = {}
    
    def __init__(self, video_path, model, output_dir, sam_model_path):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.sam_model_path = sam_model_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # used for keyframe-mode timestamps

        custom_tracker = CustomTracker(sam_model_path = self.sam_model_path)
        
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
    
def mass_prediction(video_paths, current_output_dir, sam_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)
    
    videos_tqdm = tqdm(video_paths)
    all_track_results = []
    for path in videos_tqdm:
        videos_tqdm.set_description(f"Processing {path}")
        processor = HeadlessVideoProcessor(path, model, current_output_dir, sam_model_path=sam_model_path)
        all_track_results.extend(processor.run())
    
    return all_track_results

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--sam_model_path', type=str, required=True, help="Path to segment anything model")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .mp4 videos to process')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
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
    results = mass_prediction(video_paths=video_paths, current_output_dir=output_dir, sam_model_path=sam_model_path)

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