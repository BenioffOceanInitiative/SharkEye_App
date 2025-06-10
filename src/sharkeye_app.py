import multiprocessing
import sys
import os
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel, QComboBox, 
                             QProgressBar, QStackedWidget, QSpacerItem, QSizePolicy, QScrollArea, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QDir, QTimer, QDateTime, QObject
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QIcon

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
from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtCore import QThread, pyqtSignal
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

def calculate_bbox_area(bbox):
    """Calculate area of bbox detection"""
    _, _, width, height = bbox
    return width * height

# def calculate_shark_length(bbox):
#     """Calculate shark length in feet based on bounding box"""
#     ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512
#     ASPECT_RATIO = ORIGINAL_WIDTH / ORIGINAL_HEIGHT
#     DRONE_ALTITUDE_M = 40
#     FOV_RADIANS = 1.274090354 # From estimate of 73 degrees 

#     long_side = (2 * ASPECT_RATIO * DRONE_ALTITUDE_M * math.tan(FOV_RADIANS / 2))/ np.sqrt(1 + ASPECT_RATIO ** 2) 
#     pixel_size_m = long_side / ORIGINAL_WIDTH

#     _, _, width, height = bbox

#     shark_ratio = 1.5
#     side_a = min(width, height)
#     side_b = max(width, height)

#     print(f'Short side: {side_a}')
#     print(f'Long side: {side_b}')
#     print(f'Long to short {(side_b / side_a)}')
#     if (side_b / side_a) < shark_ratio:
#         shark_pixel_length = np.sqrt((side_a ** 2) + (side_b ** 2))
#     else:
#         shark_pixel_length = side_a

#     length_m = shark_pixel_length * pixel_size_m 
#     return length_m * 3.28084  # Convert meters to feet

def calculate_adjusted_shark_length(length_raw):
    """Calculate adjusted shark length in feet using correction factors"""
    asl_correction_factor = 1
    depth_correction_factor = (1 + DRONE_ALTITUDE_M)/DRONE_ALTITUDE_M
    length_adj = length_raw * asl_correction_factor * depth_correction_factor
    return length_adj

class CustomTracker:
    def __init__(self, distance_threshold=250, min_frames=5, confidence_threshold=0.4):
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.unique_sharks = 0
        self.last_reported_sharks = 0

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
            predicted_positions = {track_id: self._predict_new_position(track) 
                                   for track_id, track in self.tracks.items()}

            cost_matrix = np.array([[self._calculate_cost(track, det, predicted_positions[track_id]) 
                                     for det in detections] 
                                    for track_id, track in self.tracks.items()])
            
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            for track_idx, detection_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, detection_idx] < self.distance_threshold:
                    track_id = list(self.tracks.keys())[track_idx]
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
            tqdm.write("Shark Detected: Shark Count: {}".format(self.unique_sharks))
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
            'frames': deque([frame.copy()], maxlen=100),
            'timestamps': deque([timestamp], maxlen=100),
            'lengths': deque([length], maxlen=100),
            'best_frame': frame.copy(),
            'best_conf': confidence,
            'best_timestamp': timestamp,
            'best_length': length,
            'longest_frame': frame.copy(),
            'longest_conf': confidence, 
            'longest_timestamp': timestamp,
            'longest_length': length,            
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
            track['best_frame'] = frame.copy()
            track['best_timestamp'] = timestamp
            track['best_length'] = length

        if confidence > .8 and length > track['longest_length']:
            track['longest_conf'] = confidence
            track['longest_frame'] = frame.copy()
            track['longest_timestamp'] = timestamp
            track['longest_length'] = length

        if len(track['positions']) > 1:
            prev_pos = np.array(track['positions'][-2][:2])
            curr_pos = np.array([x, y])
            track['velocity'] = curr_pos - prev_pos

    @staticmethod
    def _format_timestamp(milliseconds):
        """Format timestamp in MM:SS format for CSV"""
        return datetime.utcfromtimestamp(milliseconds / 1000).strftime("%M:%S")

    @staticmethod
    def _format_timestamp_filename(milliseconds):
        """Format timestamp in MMSS format for filename"""
        return datetime.utcfromtimestamp(milliseconds / 1000).strftime("%M%S")

    def save_best_frames(self, output_dir, video_path):
        """Save best frames for each significant track"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        images_saved = 0
        
        for track_id, track in self.tracks.items():
            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])
            
            if num_frames >= self.min_frames and avg_confidence > self.confidence_threshold:
                pass
            else:
                print('Track detected below threshold')

            longest_frame = track['longest_frame']
            longest_timestamp = track['longest_timestamp']
            longest_confidence = track['longest_conf']
            longest_length = track['longest_length']
            
            if longest_frame is not None:
                timestamp_str = self._format_timestamp_filename(longest_timestamp)
                
                x, y, w, h = track['positions'][track['confidences'].index(longest_confidence)]
                
                # Use segmentation model to generate lengths
                mask = run_prediction(longest_frame, (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
                pixel_length = find_pixel_length(mask, draw_line=False, viz_name = f'{video_name}-viz')
                segmentation_length = calculate_shark_length_from_pixel(pixel_length, original_width=longest_frame.shape[1], original_height=longest_frame.shape[0])
                track['longest_length'] = segmentation_length
                longest_length = track['longest_length']

                mask_overlay = draw_mask(mask, longest_frame)
                track['mask_overlay'] = mask_overlay

                feet, inches = divmod(longest_length, 1)
                length_str = f"{int(feet)}ft{int(inches * 12)}in"
                
                avg_conf_int = int(avg_confidence * 100)
                longest_conf_int = int(longest_confidence * 100)
                
                filename = f"{video_name}_shark{track_id}_time{timestamp_str}_det{num_frames}_avgConf{avg_conf_int}_bestConf{longest_conf_int}_len{length_str}.jpg"
                
                # Save original frame
                cv2.imwrite(os.path.join(output_dir, 'frames', filename), longest_frame)
                
                # Save frame with bounding box
                boxed_frame = longest_frame.copy()
                cv2.rectangle(boxed_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                label = f"ID: {track_id}, Conf: {longest_confidence:.2f}, Length: {length_str}"
                cv2.putText(boxed_frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                bounding_box_path = os.path.join(output_dir, 'bounding_boxes', filename)
                cv2.imwrite(bounding_box_path, boxed_frame)
                
                # Update the track with the path to the bounding box image
                track['image_path'] = bounding_box_path
                
                images_saved += 1

        print(f"Shark Images Saved: {images_saved}")

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1
        self.unique_sharks = 0

    def _predict_new_position(self, track):
        """Predict new position based on previous positions and velocity"""
        if len(track['positions']) > 0:
            return np.array(track['positions'][-1][:2]) + track['velocity']
        else:
            return np.array([0, 0])  # Default prediction if no positions available

    def _calculate_cost(self, track, detection, predicted_position):
        """Calculate cost for Hungarian algorithm"""
        position_cost = np.linalg.norm(predicted_position - np.array(detection[:2]))
        time_since_last_detection = track['frames_since_last_detection']
        return position_cost + time_since_last_detection * 10  # Penalize tracks that haven't been detected recently

    def _count_significant_tracks(self):
        """Count tracks that meet the criteria for being a significant detection"""
        return sum(1 for track in self.tracks.values() 
                   if len(track['positions']) >= self.min_frames 
                   and np.mean(track['confidences']) > self.confidence_threshold)

class VideoProcessingWorker(QObject):
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(dict, str)
    frame_processed = pyqtSignal(np.ndarray)  # Add a boolean flag for detection

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
        
        self.detection_threshold = 0.4

        frame_num = 0
        while frame_num < total_frames:
            if QThread.currentThread().isInterruptionRequested():
                print("Processing interrupted")
                break

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
                               if confidence > self.detection_threshold]

            has_detection = bool(detections)
            if has_detection:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                active_tracks = custom_tracker.update(detections, frame, timestamp)
                
                # Draw bounding boxes on the frame
                frame_with_boxes = self.draw_bounding_boxes(frame, detections)
                
                # Emit the processed frame with bounding boxes
                self.frame_processed.emit(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))
                
                consecutive_empty_frames = 0
                frame_skip = min_frame_skip
            else:
                # Emit the frame without bounding boxes
                self.frame_processed.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                consecutive_empty_frames += frame_skip

            # Increase frame skip more aggressively
            if consecutive_empty_frames >= max_empty_frames:
                frame_skip = min(max_frame_skip, frame_skip * 2)

            frame_num += frame_skip
            self.progress_update.emit(int((frame_num + 1) / total_frames * 100))

        cap.release()
        
        if not QThread.currentThread().isInterruptionRequested():
            # Only save results if not interrupted
            custom_tracker.save_best_frames(self.output_dir, self.video_path)
            self.save_detections_csv(custom_tracker.tracks, self.output_dir)
            self.processing_finished(custom_tracker.tracks)

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

    def processing_finished(self, tracks):
        self.progress_update.emit(100)
        self.processing_complete.emit(tracks, os.path.basename(self.video_path))

    def save_detections_csv(self, tracks, output_dir):
        csv_path = os.path.join(output_dir, 'shark_detections.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Track Id', 'Highest Conf Timestamp', 'Highest Confidence', 'Average Confidence', 
                          'Lowest Confidence', 'Longest Length', 'Highest Confidence Length',
                          'Number of Detections', 'Meets Thresholds']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for track_id, track in tracks.items():
                meets_thresholds = (len(track['confidences']) >= 10 and 
                                    np.mean(track['confidences']) > 0.4)
                
                csv_writer.writerow({
                    'Track Id': track_id,
                    'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                    'Highest Confidence': max(track['confidences']),
                    'Average Confidence': np.mean(track['confidences']),
                    'Lowest Confidence': min(track['confidences']),
                    'Longest Length': max(track['lengths']),
                    'Highest Confidence Length': track['best_length'],
                    'Number of Detections': len(track['confidences']),
                    'Meets Thresholds': meets_thresholds
                })

class DraggableListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dropEvent(self, event):
        super().dropEvent(event)
        self.updateInternalOrder()

    def updateInternalOrder(self):
        # This method will be implemented in the MainWindow class
        pass

# Add this function at the top of the file, outside any class
def format_time(seconds: float) -> str:
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 120:
        return f"1 minute {seconds % 60:.0f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} minutes {remaining_seconds} seconds"

class MainWindow(QMainWindow):
    upload_finished = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1000, 800)

        self.init_ui()
        self.init_attributes()
        self.setup_model()
        self.setup_signal_handlers()

        # Connect the upload_finished signal
        self.upload_finished.connect(self.on_upload_finished)

    def init_attributes(self):
        self.is_processing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.start_time = None
        self.elapsed_time = 0
        self.current_video = ""
        self.tracks = {}
        self.sorted_tracks = []
        self.current_detection_index = 0
        self.video_queue = []
        self.current_video_index = 0
        self.total_videos = 0
        self.processed_videos = 0
        self.processing_thread = None
        self.processing_worker = None
        self.api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload"
        self.is_uploading = False
        self.upload_thread = None
        self.progress_dialog = None
        self.confidence_threshold = .4 

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setup_banner()
        self.setup_content_widget()
        self.setup_stack_widget()
        self.setup_home_page()
        self.setup_review_widget()

    def setup_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.model = YOLO(MODEL_PATH).to(device)

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def setup_banner(self):
        # Add banner
        self.banner = QLabel()
        logo_path = resource_path('assets/images/logo-white.png')
        banner_pixmap = QPixmap(logo_path).scaledToHeight(40, Qt.TransformationMode.SmoothTransformation)
        self.banner.setPixmap(banner_pixmap)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet("background-color: #1d2633;")
        self.banner.setFixedHeight(60)  # Adjust height as needed
        self.layout.addWidget(self.banner)

    def setup_content_widget(self):
        # Create a container for the rest of the content
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 0, 20, 20)

    def setup_stack_widget(self):
        self.stack_widget = QStackedWidget()
        self.content_layout.addWidget(self.stack_widget)

        # Add the content widget to the main layout
        self.layout.addWidget(self.content_widget)

        self.home_widget = QWidget()
        self.review_widget = QWidget()

        self.stack_widget.addWidget(self.home_widget)
        self.stack_widget.addWidget(self.review_widget)

    def setup_home_page(self):
        layout = QVBoxLayout(self.home_widget)

        # Select Video(s) button
        self.select_videos_button = QPushButton("Select Video(s)")
        self.select_videos_button.clicked.connect(self.select_videos)
        layout.addWidget(self.select_videos_button)

        # Remove buttons in horizontal layout
        remove_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected Video(s)")
        self.remove_button.clicked.connect(self.remove_selected_videos)
        self.remove_button.setEnabled(False)  # Initially disabled
        remove_layout.addWidget(self.remove_button)

        self.remove_all_button = QPushButton("Remove All Videos")
        self.remove_all_button.clicked.connect(self.remove_all_videos)
        self.remove_all_button.setEnabled(False)  # Initially disabled
        remove_layout.addWidget(self.remove_all_button)
        layout.addLayout(remove_layout)

        # Video list
        self.video_list = DraggableListWidget()
        self.video_list.setMaximumHeight(100)
        self.video_list.itemSelectionChanged.connect(self.update_remove_buttons)
        self.video_list.updateInternalOrder = self.update_video_order
        layout.addWidget(self.video_list)

        # Process Videos button
        self.process_button = QPushButton("Process Videos")
        self.process_button.clicked.connect(self.toggle_processing)
        self.process_button.setEnabled(False)  # Initially disabled
        layout.addWidget(self.process_button)

        # Frame display
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_display.setMinimumSize(720, 480)
        self.frame_display.hide()
        layout.addWidget(self.frame_display)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Timer label (initially hidden)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.hide()
        layout.addWidget(self.timer_label)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # # Review History button
        # self.to_review_history_button = QPushButton("Review History")
        # self.to_review_history_button.clicked.connect(self.go_to_review_history)
        # layout.addWidget(self.to_review_history_button)

    def toggle_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.confirm_cancel_processing()

    def start_processing(self):
        self.is_processing = True
        self.process_button.setText("Cancel")
        self.process_button.setEnabled(True)
        self.remove_button.setEnabled(False)
        self.remove_all_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.timer_label.show()

        self.start_time = QDateTime.currentDateTime()
        self.timer.start(1000)
        self.elapsed_time = 0
        self.update_timer()

        # Reset processing state
        self.tracks = {}
        self.current_video_index = 0
        self.processed_videos = 0
        self.total_videos = len(self.video_queue)
        
        # Reset any prefixed emojis
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            item.setText(item.text().replace('🔎 ', '').replace('✅ ', ''))

        self.video_queue = [self.video_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.count())]
        self.current_video_index = 0
        self.total_videos = len(self.video_queue)
        self.processed_videos = 0
        
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.current_output_dir = os.path.join(get_results_dir(), timestamp)
        os.makedirs(self.current_output_dir, exist_ok=True)
        
        self.process_next_video()

    def process_next_video(self):
        if self.current_video_index < len(self.video_queue):
            self.process_video(self.video_queue[self.current_video_index])
        else:
            self.finish_processing()

    def process_video(self, video_path):
        self.current_video = video_path
        
        # Reset all video list items to remove any existing emojis
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            clean_text = item.text().replace('🔎 ', '').replace('✅ ', '')
            if item.data(Qt.ItemDataRole.UserRole) == video_path:
                item.setText(f"🔎 {clean_text}")  # Current video gets magnifying glass
            elif item.data(Qt.ItemDataRole.UserRole) in [self.video_queue[j] for j in range(self.current_video_index)]:
                item.setText(f"✅ {clean_text}")  # Completed videos get checkmark
            else:
                item.setText(clean_text)  # Pending videos have no emoji
        
        self.cleanup_previous_processing()
        
        self.processing_thread = QThread()
        self.processing_worker = VideoProcessingWorker(video_path, self.model, self.current_output_dir)
        self.processing_worker.moveToThread(self.processing_thread)
        
        self.connect_worker_signals()
        
        self.processing_thread.start()

        self.update_video_list_emoji()
        self.prepare_frame_display()

        self.bounding_boxes_dir = os.path.join(self.current_output_dir, 'bounding_boxes')
        self.false_positives_dir = os.path.join(self.current_output_dir, 'false_positives')
        
        os.makedirs(os.path.join(self.current_output_dir, 'frames'), exist_ok=True)
        os.makedirs(self.bounding_boxes_dir, exist_ok=True)
        os.makedirs(self.false_positives_dir, exist_ok=True)

    def cleanup_previous_processing(self):
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.processing_thread.deleteLater()
            self.processing_thread = None
        if self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

    def connect_worker_signals(self):
        self.processing_worker.frame_processed.connect(self.update_frame_display, Qt.ConnectionType.QueuedConnection)
        self.processing_worker.progress_update.connect(self.update_progress, Qt.ConnectionType.QueuedConnection)
        self.processing_worker.processing_complete.connect(self.processing_complete, Qt.ConnectionType.QueuedConnection)
        self.processing_thread.started.connect(self.processing_worker.run)

    def update_video_list_emoji(self):
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_video:
                item.setText(f"🔎 {item.text().replace('🔎 ', '').replace('✅ ', '')}")
                break

    def prepare_frame_display(self):
        self.frame_display.clear()
        self.frame_display.show()

    def on_video_complete(self, tracks, video_filename):
        self.tracks[video_filename] = tracks
        self.current_detection_index = 0
        
        # Update video list item with checkmark
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_video:
                clean_text = item.text().replace('🔎 ', '').replace('✅ ', '')
                item.setText(f"✅ {clean_text}")
                break
        
        # Move to the next video
        self.processed_videos += 1
        self.current_video_index += 1

        # Clean up the current thread and worker
        self.processing_thread.quit()
        self.processing_thread.wait()
        self.processing_thread.deleteLater()
        self.processing_worker.deleteLater()
        self.processing_worker = None

        # Process the next video or finish
        QTimer.singleShot(0, self.process_next_video)

    def confirm_cancel_processing(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.NoIcon)
        msg_box.setText("Are you sure you want to cancel?")
        msg_box.setWindowTitle("Confirm Cancellation")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        if msg_box.exec() == QMessageBox.StandardButton.Yes:
            self.cancel_processing()

    def cancel_processing(self):
        self.is_processing = False
        self.process_button.setText("Process Videos")
        self.process_button.setEnabled(True)
        self.remove_button.setEnabled(len(self.video_list.selectedItems()) > 0)
        self.remove_all_button.setEnabled(self.video_list.count() > 0)

        if self.processing_thread:
            # Disconnect all signals from the worker
            if self.processing_worker:
                self.processing_worker.progress_update.disconnect()
                self.processing_worker.processing_complete.disconnect()
                self.processing_worker.frame_processed.disconnect()

            # Request the thread to stop
            self.processing_thread.requestInterruption()
            
            # Wait for a short time for the thread to finish
            if not self.processing_thread.wait(1000):  # Wait for 1 seconds
                print("Thread did not finish in time, forcefully terminating")
                self.processing_thread.terminate()
                self.processing_thread.wait()

            self.processing_thread.deleteLater()
            self.processing_thread = None

        if self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

        self.progress_bar.hide()
        self.timer_label.hide()
        self.timer.stop()
        self.frame_display.hide()

        # Reset video list items to clean state (no emojis)
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            clean_text = item.text().replace('🔎 ', '').replace('✅ ', '')
            item.setText(clean_text)

        # Reset processing state
        self.current_video_index = 0
        self.processed_videos = 0
        self.tracks = {}

        print("Processing cancelled")

    def update_remove_buttons(self):
        has_selected_items = len(self.video_list.selectedItems()) > 0
        has_any_items = self.video_list.count() > 0
        self.remove_button.setEnabled(has_selected_items and not self.is_processing)
        self.remove_all_button.setEnabled(has_any_items and not self.is_processing)
        self.process_button.setEnabled(has_any_items and not self.is_processing)

    def select_videos(self):
        file_dialog = QFileDialog()
        video_files, _ = file_dialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        
        # Get the current list of file paths
        current_files = set(self.video_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.count()))
        
        new_files_added = 0
        for file_path in video_files:
            if file_path not in current_files:
                file_name = os.path.basename(file_path)
                item = QListWidgetItem(file_name)  # No emoji for new items
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.video_list.addItem(item)
                current_files.add(file_path)
                new_files_added += 1
        
        self.update_remove_buttons()

    def remove_selected_videos(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        self.update_remove_buttons()

    def remove_all_videos(self):
        self.video_list.clear()
        self.update_remove_buttons()

    def update_progress(self, value):
        # Adjust progress calculation to account for post-processing
        video_progress = value * 0.9  # Assume video processing is 90% of total work
        post_processing_progress = 10 if self.processed_videos == self.total_videos else 0
        overall_progress = int((self.processed_videos * 100 + video_progress + post_processing_progress) / self.total_videos)
        self.progress_bar.setValue(overall_progress)

    def processing_complete(self, tracks, video_filename):
        self.tracks[video_filename] = tracks
        self.current_detection_index = 0
        
        # Move to the next video
        self.processed_videos += 1
        self.current_video_index += 1

        if self.current_video_index < len(self.video_queue):
            self.process_video(self.video_queue[self.current_video_index])
        else:
            # Sort tracks before showing review widget
            self.sort_tracks()
            # Update detection list
            self.update_detection_list()
            # Show first detection if available
            if self.sorted_tracks:
                self.show_detection(0)
            self.finish_processing()
            # Automatically show review widget after processing
            self.stack_widget.setCurrentWidget(self.review_widget)

    def finish_processing(self):
        self.is_processing = False
        self.timer.stop()
        self.process_button.setEnabled(True)  # Re-enable the process button
        
        # Calculate total time using the standalone function
        time_str = format_time(self.elapsed_time)
        
        # Calculate total detections
        total_detections = sum(len(tracks) for tracks in self.tracks.values())
        
        # Show completion popup with both time and detections
        msg = QMessageBox()
        msg.setWindowTitle("Processing Complete")
        msg.setText(f"Processing completed!\n\nTotal detections: {total_detections}\nTime taken: {time_str}")
        msg.exec()

    def go_to_review_from_popup(self, popup):
        popup.accept()
        self.go_to_review_history()

    def show_detection(self, index):
        if 0 <= index < len(self.sorted_tracks):
            self.current_detection_index = index
            key, track = self.sorted_tracks[index]
            
            # Create frames with bounding boxes for the entire track
            track_frames = []
            for pos, frame in zip(track['positions'], track['frames']):
                x, y, w, h = pos
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box, 
                             (int(x - w/2), int(y - h/2)), 
                             (int(x + w/2), int(y + h/2)), 
                             (0, 255, 0), 2)
                track_frames.append(frame_with_box)
            
            # Show track frames in the player
            self.frame_player.set_frames(track_frames)
            self.show_confidence_warning()
            
            self.label_combo.setCurrentText(track.get('label', 'Shark'))
            self.prev_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self.sorted_tracks) - 1)
            self.highlight_current_detection()
        else:
            print(f"Error: Invalid detection index: {index}")
            self.frame_player.clear()
            self.frame_player.setText("No detections available")

    def show_no_detections_message(self):
        self.frame_player.clear()
        self.frame_player.setText("No detections available")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.label_combo.setEnabled(False)

    def update_detection_list(self): # Handle images that wouldn't get saved?
        self.detection_list.clear()
        
        print(f"Updating detection list with {len(self.sorted_tracks)} tracks")
        
        for index, (key, track) in enumerate(self.sorted_tracks):
            try:
                timestamp = track['timestamps'][0]  # Get first timestamp
                time_str = datetime.utcfromtimestamp(timestamp / 1000).strftime("%M%S")
                formatted_time = f"{time_str[:2]}:{time_str[2:]}"
                item_text = f"Video: {track['video_name']} - ID: {track['unique_id']} - Time: {formatted_time} - Confidence: {track['longest_conf']:.2f} - Length: {track['longest_length']:.1f}ft - Label: {track['label']}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, index)
                if track['longest_conf'] < 0.65:
                    item.setBackground(QColor("yellow"))   
                self.detection_list.addItem(item)
            except KeyError as e:
                print(f"Missing key in track data: {e}")
            except Exception as e:
                print(f"Error creating list item for track: {str(e)}")

        print(f"Updated detection list with {self.detection_list.count()} items")
        self.highlight_current_detection()

    def show_selected_detection(self):
        selected_items = self.detection_list.selectedItems()
        if selected_items:
            index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if index != self.current_detection_index:
                self.show_detection(index)
            self.show_confidence_warning()

    def highlight_current_detection(self):
        for i in range(self.detection_list.count()):
            item = self.detection_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_detection_index:
                item.setSelected(True)
                self.detection_list.scrollToItem(item)
            else:
                item.setSelected(False)

    def prev_detection(self):
        self.show_detection(self.current_detection_index - 1)

    def next_detection(self):
        self.show_detection(self.current_detection_index + 1)

    def update_label(self, index):
        if not self.sorted_tracks:
            print("Error: No sorted tracks available. Cannot update label.")
            return

        new_label = self.label_combo.currentText()
        key, track = self.sorted_tracks[self.current_detection_index]
        old_label = track['label']
        
        print(f"Updating label for track: {key}")
        
        # Simply update the label in memory
        track['label'] = new_label
        
        # Update the detection list to reflect the new label
        self.update_detection_list()
        
        # Ensure the current detection remains selected
        self.show_detection(self.current_detection_index)
        
        print(f"Label updated from {old_label} to {new_label} for track {key}")

    def sort_tracks(self):
        print("Sorting tracks...")
        print(f"Number of tracks before sorting: {len(self.tracks)}")
        
        # Flatten all tracks from all videos into a single list
        all_tracks = []
        for video_name, video_tracks in self.tracks.items():
            for track_id, track in video_tracks.items():
                track_info = {
                    'video_name': video_name,
                    'track_id': track_id,
                    **track  # Include all track information
                }
                all_tracks.append((f"{video_name}_{track_id}", track_info))
        
        self.sorted_tracks = sorted(
            all_tracks,
            key=lambda x: (x[1]['video_name'], x[1]['timestamps'][0], x[1]['id'])
        )
        
        print(f"Number of sorted tracks: {len(self.sorted_tracks)}")
        for key, track in self.sorted_tracks:
            print(f"Sorted track: {key}")

    def go_to_review_history(self):
        self.stack_widget.setCurrentWidget(self.review_widget)

    def toggle_display_mode(self):
        if self.frame_player.timer.isActive():
            self.frame_player.timer.stop()

            current_track = self.sorted_tracks[self.current_detection_index]
            if 'mask_overlay' not in current_track[1]:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Alert")
                dlg.setText("Error: No mask drawn for this track")
                button = dlg.exec()

                if button == QMessageBox.StandardButton.Ok:
                    print("OK!")

            else:
                mask_overlay = current_track[1]['mask_overlay']
                frame = mask_overlay
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.frame_player.size(), Qt.AspectRatioMode.KeepAspectRatio)
                
                self.frame_player.setPixmap(scaled_pixmap)
        else:
            self.frame_player.timer.start()

    def setup_review_widget(self):
        layout = QVBoxLayout(self.review_widget)
        
        # Frame player container with horizontal centering
        frame_player_container = QVBoxLayout()
        frame_player_container.addStretch()  # Add stretch before frame player
        
        self.frame_player = FramePlayer()
        self.frame_player.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_player.setMinimumSize(int(.93 * 720), int(.93 * 480))
        frame_player_container.addWidget(self.frame_player)

        # Add warning when detection falls before 
        self.low_confidence_warning = QLabel("⚠️ Warning: Low confidence in this detection. Please double check the image to make sure the boxed area is a shark!")
        self.low_confidence_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_confidence_warning.setMinimumHeight(30)
        frame_player_container.addWidget(self.low_confidence_warning)

        frame_player_container.addStretch()  # Add stretch after frame player
        layout.addLayout(frame_player_container)

        # Button to toggle display to show gif/segmentation mask
        self.toggle_display_mode_button = QPushButton("Toggle Mask/Bounding Box Display")
        self.toggle_display_mode_button.clicked.connect(self.toggle_display_mode)
        layout.addWidget(self.toggle_display_mode_button)
        
        # Label combo
        self.label_combo = QComboBox()
        self.label_combo.addItems(["Shark", "Kelp", "Dolphin", "Surfer", "Boat", "Bird", "Other"])
        self.label_combo.currentIndexChanged.connect(self.update_label)
        layout.addWidget(self.label_combo)
        
        # Navigation controls
        controls_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_detection)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_detection)
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.next_button)
        layout.addLayout(controls_layout)

        # Detection list
        self.detection_list = QListWidget()
        self.detection_list.itemSelectionChanged.connect(self.show_selected_detection)
        self.detection_list.setMaximumHeight(100)
        layout.addWidget(self.detection_list)

        # Export/Upload buttons
        button_layout = QHBoxLayout()
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        upload_button = QPushButton("Upload Results")
        upload_button.clicked.connect(self.upload_images)
        button_layout.addWidget(export_button)
        button_layout.addWidget(upload_button)
        layout.addLayout(button_layout)

        # Home button
        home_button = QPushButton("Home")
        home_button.clicked.connect(self.go_to_home)
        layout.addWidget(home_button)

    def go_to_home(self):
        # Clean up generated files and folders
        if hasattr(self, 'current_output_dir') and self.current_output_dir:
            try:
                if os.path.exists(self.current_output_dir):
                    shutil.rmtree(self.current_output_dir)
                    print(f"Cleaned up output directory: {self.current_output_dir}")
            except Exception as e:
                print(f"Error cleaning up output directory: {str(e)}")
        
        # Reset processing state
        self.is_processing = False
        self.process_button.setText("Process Videos")
        
        # Reset progress indicators
        self.progress_bar.hide()
        self.timer_label.hide()
        self.timer.stop()
        self.elapsed_time = 0
        
        # Hide frame display
        self.frame_display.hide()
        self.frame_display.clear()
        
        # Clear video list and reset buttons
        self.video_list.clear()
        self.video_queue = []
        self.current_video_index = 0
        self.processed_videos = 0
        
        # Reset tracking data
        self.tracks = {}
        self.sorted_tracks = []
        self.current_detection_index = 0
        
        # Reset button states
        self.process_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.remove_all_button.setEnabled(False)
        
        # Switch to home widget
        self.stack_widget.setCurrentWidget(self.home_widget)
        
    def show_confidence_warning(self):
        _, track = self.sorted_tracks[self.current_detection_index]
        if track['longest_conf'] < .65:
            self.low_confidence_warning.setVisible(True)
        else:
            self.low_confidence_warning.setVisible(False)

    def update_timer(self):
        self.elapsed_time += 1
        hours, remainder = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def generate_filename(self, track, new_label):
        return f"{track['video_name']}_{new_label.lower()}{track['unique_id']}_time{track['time']}_det{track['detections']}_avgConf{int(track['avg_conf']*100):02d}_bestConf{int(track['best_conf']*100):02d}_len{track['length'].replace('ft', 'ft').replace('in', 'in')}.jpg"

    def update_frame_display(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.frame_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.frame_display.setPixmap(scaled_pixmap)
        self.frame_display.show()

    def closeEvent(self, event):
        # Clean up generated files and folders
        if hasattr(self, 'current_output_dir') and self.current_output_dir:
            try:
                if os.path.exists(self.current_output_dir):
                    shutil.rmtree(self.current_output_dir)
                    print(f"Cleaned up output directory: {self.current_output_dir}")
            except Exception as e:
                print(f"Error cleaning up output directory: {str(e)}")
        
        # Ensure threads are properly closed
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        event.accept()

    def update_video_order(self):
        # Update the internal order of videos after drag and drop
        self.video_queue = [self.video_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.count())]
        print("Video order updated:", self.video_queue)

    def export_results(self):
        if not self.sorted_tracks:
            QMessageBox.warning(self, "No Data", "There are no results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        
        if not file_path:
            return  # User cancelled the dialog

        try:
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['video_name', 'track_id', 'label', 'timestamp', 'confidence', 'length_ft']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for _, track in self.sorted_tracks:
                    timestamp = track['longest_timestamp']
                    time_str = datetime.utcfromtimestamp(timestamp / 1000).strftime('%M:%S')
                    
                    writer.writerow({
                        'video_name': track['video_name'],
                        'track_id': track['unique_id'],
                        'label': track['label'],
                        'timestamp': time_str,
                        'confidence': track['longest_conf'],
                        'length_ft': track['longest_length']
                    })
            
            QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")

    def upload_images(self):
        if self.is_uploading:
            QMessageBox.warning(self, "Upload in Progress", "An upload is already in progress.")
            return

        if not self.sorted_tracks:
            QMessageBox.warning(self, "No Data", "There are no results to upload.")
            return

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Upload Data")
        msg_box.setText("Do you want to upload the current data?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        if msg_box.exec() == QMessageBox.StandardButton.Yes:
            self.is_uploading = True
            self.upload_to_gcs()

    def upload_to_gcs(self):
        if self.progress_dialog:
            self.progress_dialog.close()
        
        self.progress_dialog = QProgressDialog("Preparing and uploading files...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.show()

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        try:
            # Create required directories
            for folder in ['bounding_boxes', 'false_positives', 'frames']:
                os.makedirs(os.path.join(temp_dir, folder))

            # Save frames and bounding boxes
            for _, track in self.sorted_tracks:
                video_name = track['video_name']
                track_id = track['unique_id']
                label = track['label'].lower()
                
                # Save best frame with bounding box
                frame_with_box = track['frames'][0].copy()  # Use first frame
                x, y, w, h = track['positions'][0]
                cv2.rectangle(frame_with_box, 
                             (int(x - w/2), int(y - h/2)), 
                             (int(x + w/2), int(y + h/2)), 
                             (0, 255, 0), 2)
                
                frame_filename = f"{video_name}_{label}{track_id}_conf{int(track['best_conf']*100):02d}_len{track['longest_length']:.1f}ft.jpg"
                
                if label == 'shark':
                    cv2.imwrite(os.path.join(temp_dir, 'bounding_boxes', frame_filename), frame_with_box)
                else:
                    cv2.imwrite(os.path.join(temp_dir, 'false_positives', frame_filename), frame_with_box)
                
                # Save original frame
                cv2.imwrite(os.path.join(temp_dir, 'frames', frame_filename), track['frames'][0])

            # Create zip file
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as zipf:
                for folder in ['bounding_boxes', 'false_positives', 'frames']:
                    folder_path = os.path.join(temp_dir, folder)
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        arcname = os.path.join(folder, file)
                        zipf.write(file_path, arcname)

            buffer.seek(0)
            files = {'file': ('upload.zip', buffer, 'application/zip')}
            response = requests.post(self.api_url, files=files)
            response.raise_for_status()

            self.upload_finished.emit(True, "Data uploaded successfully")
        except Exception as e:
            self.upload_finished.emit(False, f"Upload failed: {str(e)}")
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def on_upload_finished(self, success, message):
        self.is_uploading = False
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        if success:
            QMessageBox.information(self, "Upload Complete", message)
        else:
            QMessageBox.critical(self, "Upload Failed", message)

        if self.upload_thread:
            self.upload_thread.wait()
            self.upload_thread = None

        self.is_uploading = False

    def ensure_track_consistency(self):
        if len(self.tracks) != len(self.sorted_tracks):
            print("Warning: Inconsistency detected between tracks and sorted_tracks")
            self.tracks = dict(self.sorted_tracks)
        
        for key, track in self.sorted_tracks:
            if key not in self.tracks:
                print(f"Warning: Track {key} found in sorted_tracks but not in tracks")
                self.tracks[key] = track

        print(f"Tracks consistency check complete. Total tracks: {len(self.tracks)}")

class UploadThread(QThread):
    progress_updated = pyqtSignal(int)
    upload_finished = pyqtSignal(bool, str)

    def __init__(self, api_url, experiment_dir):
        super().__init__()
        self.api_url = api_url
        self.experiment_dir = experiment_dir

    def run(self):
        try:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as zipf:
                for folder in ['bounding_boxes', 'false_positives', 'frames']:
                    folder_path = os.path.join(self.experiment_dir, folder)
                    if os.path.exists(folder_path):
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.experiment_dir)
                                zipf.write(file_path, arcname)

            buffer.seek(0)
            files = {'file': ('upload.zip', buffer, 'application/zip')}
            response = requests.post(self.api_url, files=files)
            response.raise_for_status()

            self.upload_finished.emit(True, "Folder uploaded successfully")
        except requests.RequestException as e:
            self.upload_finished.emit(False, "Upload failed: {}".format(str(e)))
        except Exception as e:
            self.upload_finished.emit(False, "An unexpected error occurred: {}".format(str(e)))

def signal_handler(signum, frame):
    print(f"Received signal {signum}")
    QApplication.quit()

class FramePlayer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = []
        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(100)  # 10 FPS
        
    def set_frames(self, frames):
        self.frames = frames
        self.current_frame = 0
        if frames:
            self.show_frame(0)
            self.timer.start()
        else:
            self.clear()
            self.timer.stop()
            
    def show_frame(self, index):
        if 0 <= index < len(self.frames):
            frame = self.frames[index]
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(scaled_pixmap)
            
    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.show_frame(self.current_frame)

    def format_time(self, seconds: float) -> str:
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 120:
            return f"1 minute {seconds % 60:.0f} seconds"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes} minutes {remaining_seconds} seconds"

    def format_time(self, seconds: float) -> str:
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 120:
            return f"1 minute {seconds % 60:.0f} seconds"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes} minutes {remaining_seconds} seconds"

    def finish_processing(self):
        self.is_processing = False
        self.timer.stop()
        
        # Calculate total time
        time_str = self.format_time(self.elapsed_time)
        
        # Calculate total detections
        total_detections = sum(len(tracks) for tracks in self.tracks.values())
        
        # Show completion popup with both time and detections
        self.show_completion_popup(time_str, total_detections)

class HeadlessVideoProcessor(VideoProcessingWorker):
    progress_update = 0
    processing_complete = {}

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

# if __name__ == '__main__':
    # main()        
# if __name__ == '__main__':
#     #video_path = [Path("./TRIMMED_2023-04-23_Transect_DJI_0502.mp4")]
#     video_path = [Path(path) for path in Path('C:/Users/legop/Downloads/videos/videos').glob("*.mp4")]

#     output_dir = Path("./headless_predictions")
#     results = mass_prediction(video_path=video_path, current_output_dir=output_dir)
    # main()        
if __name__ == '__main__':
    #video_path = [Path("./TRIMMED_2023-04-23_Transect_DJI_0502.mp4")]
    # video_path = [Path(path) for path in Path('C:/Users/legop/Downloads/videos/videos').glob("*.mp4")]

    # output_dir = Path("./headless_predictions")
    # results = mass_prediction(video_path=video_path, current_output_dir=output_dir)

    # with open(output_dir / "output.csv", mode="w", newline="", encoding="utf-8") as file:
    #     writer = csv.DictWriter(file, fieldnames=results[0].keys())
    #     writer.writeheader()
    #     writer.writerows(results)

    # multiprocessing.freeze_support()
    # app = QApplication(sys.argv)
    # app.setQuitOnLastWindowClosed(True)
    # with open(output_dir / "output.csv", mode="w", newline="", encoding="utf-8") as file:
    #     writer = csv.DictWriter(file, fieldnames=results[0].keys())
    #     writer.writeheader()
    #     writer.writerows(results)

    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    app_icon_path = {
        'win32': 'assets/logo/SharkEye.ico',
        'darwin': 'assets/logo/SharkEye.icns'
    }.get(sys.platform, 'assets/logo/SharkEye.iconset/icon_32x32.png')
    app_icon_path = {
        'win32': 'assets/logo/SharkEye.ico',
        'darwin': 'assets/logo/SharkEye.icns'
    }.get(sys.platform, 'assets/logo/SharkEye.iconset/icon_32x32.png')
    
    app.setWindowIcon(QIcon(resource_path(app_icon_path)))
    app.setWindowIcon(QIcon(resource_path(app_icon_path)))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    #  python SharkEye_App/src/sharkeye_app.py --input_dir "sharkeye/2023" --output_dir "2023/predictions"