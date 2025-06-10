import multiprocessing
import sys
import os
import argparse
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
import csv
from tqdm import tqdm
import re
from utility import resource_path, get_results_dir
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
from sharkeye_app import calculate_gsd, calculate_shark_length, CustomTracker

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

class HeadlessVideoProcessor():
    progress_update = 0
    processing_complete = {}
    
    def __init__(self, video_path, model, output_dir):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        custom_tracker = CustomTracker()
        
        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'bounding_boxes'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'false_positives'), exist_ok=True)

        min_frame_skip, max_frame_skip = 10, 60
        frame_skip = min_frame_skip
        consecutive_empty_frames = 0
        max_empty_frames = 1 * fps
        detection_threshold = 0.4

        frame_num = 0
        while frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, classes=[0], verbose=False)

            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu()
                confidences = results[0].boxes.conf.cpu().tolist()

                detections = [(float(x), float(y), float(w), float(h), confidence) 
                               for (x, y, w, h), confidence in zip(boxes, confidences) 
                               if confidence > detection_threshold]

            has_detection = bool(detections)
            if has_detection:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                active_tracks = custom_tracker.update(detections, frame, timestamp)
                
                # Draw bounding boxes on the frame
                frame_with_boxes = self.draw_bounding_boxes(frame, detections)
                
                # Emit the processed frame with bounding boxes
                frame_processed = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                
                consecutive_empty_frames = 0
                frame_skip = min_frame_skip
            else:
                # Emit the frame without bounding boxes
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                consecutive_empty_frames += frame_skip

            # Increase frame skip more aggressively
            if consecutive_empty_frames >= max_empty_frames:
                frame_skip = min(max_frame_skip, frame_skip * 2)

            frame_num += frame_skip
            self.progress_update = int((frame_num + 1) / total_frames * 100) 

        cap.release()
        custom_tracker.save_best_frames(self.output_dir, self.video_path)

        all_track_info = [] 

        for track_id, track in custom_tracker.tracks.items():
            meets_thresholds = (len(track['confidences']) >= 10 and 
                                np.mean(track['confidences']) > 0.4)
            
            track_info = {   
                'Video name': self.video_path.name, 
                'Track Id': track_id,
                'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                'Highest Confidence': max(track['confidences']),
                'Average Confidence': np.mean(track['confidences']),
                'Lowest Confidence': min(track['confidences']),
                'Longest Length': max(track['lengths']),
                'Highest Confidence Length': track['best_length'],
                'Number of Detections': len(track['confidences']),
                'Meets Thresholds': meets_thresholds
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
    
def mass_prediction(video_path, current_output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)
    
    videos_tqdm = tqdm(video_path)
    all_track_results = []
    for path in videos_tqdm:
        videos_tqdm.set_description(f"Processing {path}")
        processor = HeadlessVideoProcessor(path, model, current_output_dir)
        all_track_results.extend(processor.run())
    
    return all_track_results

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .mp4 videos to process')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
    return parser.parse_args()

def main():
    args = parse_args()  

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    video_paths = input_dir.rglob("*.mp4")
    if not video_paths:
        print(f"No .mp4 videos found in {input_dir}")
        exit(1)

    # Run prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    results = mass_prediction(video_path=video_paths, current_output_dir=output_dir)

    # Save results to CSV
    if results:
        csv_path = output_dir / "output.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")
    else:
        print("No valid tracks were found.")

if __name__ == '__main__':
    main()        