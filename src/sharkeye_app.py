import multiprocessing
import sys
import os
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel, QComboBox, 
                             QProgressBar, QStackedWidget, QSpacerItem, QSizePolicy, QScrollArea, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QDialogButtonBox, QTextEdit, QLineEdit, QTreeWidget, QTreeWidgetItem, QInputDialog, QFormLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QDir, QTimer, QDateTime, QObject, QSettings, QByteArray
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QIcon, QDoubleValidator, QFont, QMovie

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
import imageio
import pandas as pd
import math
from pathlib import Path
from segmentation.segmentation_model import run_prediction, calculate_shark_length_from_pixel, find_pixel_length, draw_mask
from segment_anything import sam_model_registry, SamPredictor 
import ast

# Add these constants for length calculation
DRONE_ALTITUDE_M = 40
SENSOR_WIDTH_MM = 13.2
FOCAL_LENGTH_MM = 28
MODEL_WIDTH = MODEL_HEIGHT = 640
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

def calculate_adjusted_shark_length(length_raw):
    """Calculate adjusted shark length in feet using correction factors"""
    asl_correction_factor = 1
    depth_correction_factor = (1 + DRONE_ALTITUDE_M)/DRONE_ALTITUDE_M
    length_adj = length_raw * asl_correction_factor * depth_correction_factor
    return length_adj

class EditDroneDialog(QDialog):
    resolution_deleted = pyqtSignal()  # 🔔 Custom signal

    def __init__(self, drone_name: str, width: str, height: str, fov_rad: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Drone")
        self.delete_requested = False  # Fallback if you prefer checking flags

        self.drone_name_input = QLineEdit(drone_name)
        self.resolution_width_input = QLineEdit(str(width))
        self.resolution_height_input = QLineEdit(str(height))
        self.fov_input = QLineEdit(f"{fov_rad:.5f}")

        layout = QFormLayout(self)
        layout.addRow("Drone Name:", self.drone_name_input)
        layout.addRow("Resolution Width:", self.resolution_width_input)
        layout.addRow("Resolution Height:", self.resolution_height_input)
        layout.addRow("FOV (radians):", self.fov_input)

        # === Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # === Delete resolution button
        self.delete_button = QPushButton("Delete Resolution")
        self.delete_button.clicked.connect(self.handle_delete)

        # Add both buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.buttons)
        button_layout.addWidget(self.delete_button)
        layout.addRow(button_layout)

        self.drone_name_input.setDisabled(True)

    def get_inputs(self):
        return (
            self.resolution_width_input.text().strip(),
            self.resolution_height_input.text().strip(),
            self.fov_input.text().strip()
        )

    def handle_delete(self):
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to delete this resolution?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.resolution_deleted.emit()  # 🔔 Notify parent
            self.done(2)  # Custom dialog code for delete

class NewDroneDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Drone")

        self.name_input = QLineEdit()
        self.resolution_width_input = QLineEdit()
        self.resolution_height_input = QLineEdit()
        self.fov_input = QLineEdit()

        layout = QFormLayout(self)
        layout.addRow("Drone Name:", self.name_input)
        layout.addRow("Resolution Width:", self.resolution_width_input)
        layout.addRow("Resolution Height:", self.resolution_height_input)
        layout.addRow("FOV (radians):", self.fov_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_inputs(self):
        return (
            self.name_input.text().strip(),
            self.resolution_width_input.text().strip(),
            self.resolution_height_input.text().strip(),
            self.fov_input.text().strip()
        )

class SettingsDialog(QDialog):
    settings_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 800, 500)
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        
        main_layout = QHBoxLayout(self)

        # Left: category list
        self.category_list = QListWidget()
        self.category_list.addItem("Drone Settings")
        self.category_list.setFixedWidth(150)
        self.category_list.currentRowChanged.connect(self.switch_category)
        main_layout.addWidget(self.category_list)

        # Right: stacked settings pages
        self.pages = QStackedWidget()
        self.drone_settings_page = DroneSettingsPage(self.settings_obj, self)
        self.pages.addWidget(self.drone_settings_page)

        main_layout.addWidget(self.pages)
        self.setLayout(main_layout)

        self.category_list.setCurrentRow(0)

    def switch_category(self, index):
        self.pages.setCurrentIndex(index)

    def closeEvent(self, event):
        self.drone_settings_page.save_settings()
        self.settings_updated.emit()  # 🚀 Notify parent to refresh drone list
        super().closeEvent(event)

class DroneSettingsPage(QWidget):
    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj
        self.settings = {}

        layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Drones", ""])
        layout.addWidget(self.tree)

        # Buttons
        button_layout = QHBoxLayout()
        self.edit_button = QPushButton("Edit")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self.edit_button_state)

        self.new_drone_button = QPushButton("Add New Drone")
        self.new_drone_button.clicked.connect(self.add_new_drone)

        self.delete_button = QPushButton("Delete Drone")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_drone)

        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.new_drone_button)
        button_layout.addWidget(self.delete_button)
        layout.addLayout(button_layout)

        # Initialize tree and signals
        self.load_settings()
        self.tree.itemClicked.connect(self.enable_action_buttons)

    def load_settings(self):
        default_drones = {
            "Mavic 2 Pro": {
                "Resolution": {
                    "(2688, 1512)": math.radians(73)
                }
            },
            "Air 2S": {
                "Resolution": {
                    "(2688, 1512)": math.radians(63.5),
                    "(5472, 3078)": math.radians(82.9)
                }
            }
        }

        value = self.settings_obj.value("drone_settings")
        if not value:
            self.settings_obj.setValue("drone_settings", json.dumps(default_drones))
            value = self.settings_obj.value("drone_settings")
        self.settings = json.loads(value)

        self.populate_tree()

    def populate_tree(self):
        self.tree.clear()
        for drone_name, drone_settings in self.settings.items():
            drone_item = QTreeWidgetItem([drone_name])
            self.tree.addTopLevelItem(drone_item)

            for resolution, fov in drone_settings["Resolution"].items():
                width, height = eval(resolution)
                resolution_label = f"{width}x{height}"
                resolution_item = QTreeWidgetItem([resolution_label])
                drone_item.addChild(resolution_item)

                fov_item = QTreeWidgetItem(["FOV (radians):", f"{fov:.5f}"])
                resolution_item.addChild(fov_item)

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)

        self.edit_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def enable_action_buttons(self):
        item = self.tree.currentItem()
        if not item:
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            return

        parent = item.parent()
        grandparent = parent.parent() if parent else None

        is_resolution = parent is not None and grandparent is None
        is_fov = parent is not None and grandparent is not None
        self.edit_button.setEnabled(is_resolution or is_fov)
        self.delete_button.setEnabled(parent is None)

    def edit_button_state(self):
        item = self.tree.currentItem()
        if not item:
            return

        parent = item.parent()
        grandparent = parent.parent() if parent else None

        if parent and grandparent is None:
            drone_name = parent.text(0)
            width_str, height_str = item.text(0).split("x")
            key = f"({width_str.strip()}, {height_str.strip()})"
            current_fov_rad = self.settings[drone_name]["Resolution"][key]

        elif parent and grandparent:
            drone_name = grandparent.text(0)
            width_str, height_str = parent.text(0).split("x")
            key = f"({width_str.strip()}, {height_str.strip()})"
            current_fov_rad = self.settings[drone_name]["Resolution"][key]
        else:
            return

        dialog = EditDroneDialog(drone_name, width_str, height_str, current_fov_rad, self)

        # Connect delete signal
        def on_resolution_deleted():
            res_dict = self.settings[drone_name]["Resolution"]
            key = f"({width_str.strip()}, {height_str.strip()})"
            if key in res_dict:
                del res_dict[key]

                # If drone has no more resolutions, remove it entirely
                if not res_dict:
                    del self.settings[drone_name]

                self.save_settings()

        dialog.resolution_deleted.connect(on_resolution_deleted)

        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            new_width, new_height, new_fov_input = dialog.get_inputs()

            if not new_width or not new_height or not new_fov_input:
                QMessageBox.warning(self, "Incomplete Input", "All fields must be filled.")
                return

            if not new_width.isdigit() or not new_height.isdigit():
                QMessageBox.warning(self, "Invalid Input", "Width and Height must be positive integers.")
                return

            try:
                new_fov_rad = float(new_fov_input)
                if new_fov_rad <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "FOV must be a positive number (in radians).")
                return

            res_dict = self.settings[drone_name]["Resolution"]
            old_key = f"({width_str.strip()}, {height_str.strip()})"
            new_key = f"({new_width}, {new_height})"

            if old_key != new_key:
                res_dict.pop(old_key, None)

            res_dict[new_key] = new_fov_rad
            self.save_settings()

        elif result == 2:  # Custom dialog code for "Delete Resolution"
            return  # Deletion already handled via signal


    def add_new_drone(self):
        dialog = NewDroneDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            drone_name, width, height, fov_input = dialog.get_inputs()

            if not drone_name or not width or not height or not fov_input:
                QMessageBox.warning(self, "Incomplete Input", "All fields must be filled.")
                return

            if not width.isdigit() or not height.isdigit():
                QMessageBox.warning(self, "Invalid Input", "Width and Height must be positive integers.")
                return

            try:
                fov_rad = float(fov_input)
                if fov_rad <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "FOV must be a positive number (in radians).")
                return

            res_key = f"({width}, {height})"
            self.settings.setdefault(drone_name, {}).setdefault("Resolution", {})[res_key] = fov_rad
            self.save_settings()

    def delete_drone(self):
        item = self.tree.currentItem()
        if not item:
            return

        drone_name = item.text(0)

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the drone '{drone_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if drone_name in self.settings:
                del self.settings[drone_name]
                self.save_settings()

    def save_settings(self):
        self.settings_obj.setValue("drone_settings", json.dumps(self.settings))
        self.populate_tree()
        
class CustomTracker:
    def __init__(self, distance_threshold=250, min_frames=5, confidence_threshold=0.4):
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.confidence_threshold = confidence_threshold
        self.unique_sharks = 0
        self.last_reported_sharks = 0
        self.fov_radians = 1.274090354
        self.drone_altitude = DRONE_ALTITUDE_M

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
                segmentation_length = calculate_shark_length_from_pixel(pixel_length,
                                                                         original_width=longest_frame.shape[1], original_height=longest_frame.shape[0],
                                                                         drone_altitude=self.drone_altitude,
                                                                         fov_radians=self.fov_radians)
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

    def __init__(self, video_path, model, output_dir, drone_type, altitude):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.drone_type = drone_type
        self.altitude = altitude
        
        # Read settings 
        settings = QSettings("BOSL", "SharkEye_App")
        value = settings.value("drone_settings")
        if not value:
            # seed safe defaults (match your SettingsDialog defaults)
            value = json.dumps({
                "Mavic 2 Pro": {"Resolution": {"(2688, 1512)": math.radians(73)}},
                "Air 2S": {"Resolution": {"(2688, 1512)": math.radians(63.5),
                                        "(5472, 3078)": math.radians(82.9)}}
            })
            settings.setValue("drone_settings", value)
        self.settings = json.loads(value)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        custom_tracker = CustomTracker()
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        custom_tracker.fov_radians = self.settings[self.drone_type]["Resolution"][f"({video_width}, {video_height})"]
        custom_tracker.drone_altitude = self.altitude

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'bounding_boxes'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'false_positives'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'detection_results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "tracking_gifs"), exist_ok=True)

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
            self.save_detections_csv(custom_tracker.tracks, os.path.join(self.output_dir, 'detection_results'))
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
        csv_path = os.path.join(output_dir, f'{Path(self.video_path).stem}.csv')
        print(f"Starting save to {csv_path}")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['video_name', 'Track Id', 'Highest Conf Timestamp', 'Highest Confidence', 'Average Confidence', 
                        'Lowest Confidence', 'Longest Length', 'Highest Confidence Length',
                        'Number of Detections', 'Meets Thresholds', 'Confidence of Longest Length', 'Label']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for track_id, track in tracks.items():
                meets_thresholds = (len(track['confidences']) >= 10 and 
                                    np.mean(track['confidences']) > 0.4)
                
                csv_writer.writerow({
                    'video_name': self.video_path,
                    'Track Id': track_id,
                    'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                    'Highest Confidence': max(track['confidences']),
                    'Average Confidence': np.mean(track['confidences']),
                    'Lowest Confidence': min(track['confidences']),
                    'Longest Length': max(track['lengths']),  
                    'Highest Confidence Length': track['longest_length'], # 
                    'Number of Detections': len(track['confidences']),
                    'Meets Thresholds': meets_thresholds,
                    'Confidence of Longest Length': track['longest_conf'],
                    'Label': 'Shark',
                })
            print("Done saving csv")

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
    
    def load_drone_settings(self):
        settings = QSettings("BOSL", "SharkEye_App")
        settings_dialog = SettingsDialog()
        settings_dialog.settings_updated.connect(self.update_available_drones)
        settings_dialog.exec()

    def save_setting(self, key, value):
        self.settings.setValue(key, value)
    
    def load_setting(self, key, default_value=None):
        return self.settings.value(key, default_value)
    
    def save_complex_setting(self, key, value):
        serialized_value = json.dumps(value)
        self.settings.setValue(key, serialized_value)
    
    def load_complex_setting(self, key, default_value=None):
        value = self.settings.value(key, default_value)
        return json.loads(value) if value else default_value
        
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
        self.cleanup_trees = False
        self.reviewing_history = False
        self.historical_label_changes = {}  # key: (experiment, csv_name, track_id) -> new_label


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
        device = torch.device('cpu') if getattr(sys, 'frozen', False) else \
         torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
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
    
    def update_available_drones(self):
        settings = QSettings("BOSL", "SharkEye_App")
        value = settings.value("drone_settings")
        
        if not value:
            return  # No drones saved yet

        try:
            drone_data = json.loads(value)
            drone_names = list(drone_data.keys())
        except (json.JSONDecodeError, TypeError):
            drone_names = []

        self.drone_select.clear()
        self.drone_select.addItems(drone_names)

    def setup_home_page(self):
        layout = QVBoxLayout(self.home_widget)

        # Select Video(s) button
        self.select_videos_button = QPushButton("Select Video(s)")
        self.select_videos_button.clicked.connect(self.select_videos)
        # self.select_videos_button.setAlignment(Qt.AlignmentFlag.AlignTop)
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

        # Select Drone and Altitude Dropdown
        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("Select Drone Model:"), 0, 0)
        self.drone_select = QComboBox()
        self.update_available_drones()
        self.drone_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Settings button
        self.settings_button = QPushButton("⚙️")
        self.settings_button.clicked.connect(self.load_drone_settings)
        
        form_layout.addWidget(self.drone_select, 0, 1)
        form_layout.addWidget(self.settings_button, 0, 2)

        form_layout.addWidget(QLabel("Enter Drone Altitude:"), 1, 0)
        self.altitude_input = QLineEdit('40')
        self.altitude_input.setValidator(QDoubleValidator(0, 999, 2))
        self.altitude_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form_layout.addWidget(self.altitude_input, 1, 1)

        layout.addLayout(form_layout)

        # Review History button
        self.review_history_button = QPushButton("Review Previous Experiments")
        self.review_history_button.clicked.connect(self.go_to_review_history)
        self.review_history_button.clicked.connect(lambda: setattr(self, "reviewing_history", False))
        self.review_history_button.clicked.connect(self.toggle_historical_experiments)
        layout.addWidget(self.review_history_button)

        # Process Videos button
        process_layout = QVBoxLayout()
        self.process_button = QPushButton("Process Videos")
        self.process_button.clicked.connect(self.toggle_processing)
        self.process_button.setEnabled(False)  # Initially disabled
        process_layout.addWidget(self.process_button)
        layout.addLayout(process_layout)
        layout.addStretch()

        # Frame display
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self.frame_display.setMinimumSize(720, 480)
        self.frame_display.hide()
        layout.addWidget(self.frame_display, stretch=1)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.progress_bar)

        # Timer label (initially hidden)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.timer_label.hide()
        layout.addWidget(self.timer_label)
        
        # Review History button
        # self.to_review_history_button = QPushButton("Review History")
        # self.to_review_history_button.clicked.connect(self.go_to_review_history)
        # layout.addWidget(self.to_review_history_button)

    def toggle_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.confirm_cancel_processing()

    def get_valid_resolutions_for_drone(self, drone_name):
        settings = QSettings("BOSL", "SharkEye_App")
        value = settings.value("drone_settings")
        
        if not value:
            return []

        try:
            drone_data = json.loads(value)
            if drone_name in drone_data and "Resolution" in drone_data[drone_name]:
                return [eval(res) for res in drone_data[drone_name]["Resolution"].keys()]
        except Exception as e:
            print(f"Error loading drone resolutions: {e}")
            return []
        
        return []

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
        
        # Validate resolution before processing
        if self.video_queue:
            first_video = self.video_queue[0]
            cap = cv2.VideoCapture(first_video)
            if not cap.isOpened():
                QMessageBox.critical(self, "Video Error", f"Could not open video: {first_video}")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            valid_resolutions = self.get_valid_resolutions_for_drone(self.drone_select.currentText())
            if (width, height) not in valid_resolutions:
                QMessageBox.warning(
                    self,
                    "Resolution Mismatch",
                    f"The selected video has resolution {width}x{height}, which is not valid for the selected drone.\n\n"
                    f"Valid resolutions for '{self.drone_select.currentText()}':\n" +
                    "\n".join([f"{w}x{h}" for w, h in valid_resolutions])
                )
                # Reset UI
                self.is_processing = False
                self.process_button.setText("Process Videos")
                self.process_button.setEnabled(True)
                self.remove_button.setEnabled(len(self.video_list.selectedItems()) > 0)
                self.remove_all_button.setEnabled(self.video_list.count() > 0)
                return
    
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
        self.processing_worker = VideoProcessingWorker(video_path, self.model, self.current_output_dir, drone_type=self.drone_select.currentText(), altitude=float(self.altitude_input.text()))
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
            # Save GIFs for each detection
            self.save_detection_gif(self.current_output_dir)
            # Show first detection if available
            if self.sorted_tracks:
                self.show_detection(0)
            self.finish_processing()
            # Automatically show review widget after processing
            self.stack_widget.setCurrentWidget(self.review_widget)
    
    def save_detection_gif(self, output_dir):
        print("Saving Track results as GIFs")
        gifs_dir = os.path.join(output_dir, "tracking_gifs")
        for key, track in self.sorted_tracks:
            # Use the bounding box frames for the track
            track_frames = []
            for pos, frame in zip(track['positions'], track['frames']):
                x, y, w, h = pos
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box,
                            (int(x - w/2), int(y - h/2)),
                            (int(x + w/2), int(y + h/2)),
                            (0, 255, 0), 2)
                track_frames.append(cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB))  # Convert to RGB for imageio

            if track_frames:
                gif_filename = f"{key}.gif"
                gif_path = os.path.join(gifs_dir, gif_filename)
                imageio.mimsave(gif_path, track_frames, fps=10)
                print(f"Saved GIF: {gif_path}")

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
        self.reviewing_history = False
        self.switch_detection_list(show_historical=False)
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
            self.show_no_detections_message()

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
                    item.setText( f"⚠️ Video: {track['video_name']} - ID: {track['unique_id']} - Time: {formatted_time} - Confidence: {track['longest_conf']:.2f} - Length: {track['longest_length']:.1f}ft - Label: {track['label']}")
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
        if self.reviewing_history:
            self.historical_items.setCurrentRow(max(0, self.historical_items.currentRow() - 1))
        else:
            self.show_detection(self.current_detection_index - 1)

    def next_detection(self):
        if self.reviewing_history:
            self.historical_items.setCurrentRow(min(self.historical_items.count() - 1, self.historical_items.currentRow() + 1))
        else:
            self.show_detection(self.current_detection_index + 1)

    def update_label(self, index):
        new_label = self.label_combo.currentText()

        if self.reviewing_history:
            selected = self.historical_items.selectedItems()
            if not selected:
                return
            item = selected[0]
            meta = self._parse_historical_item_text(item.text())
            experiment = meta["experiment"]
            csv_name = meta["csv_name"]
            track_id = meta["track_id"]

            key = (experiment, csv_name, track_id)
            self.historical_label_changes[key] = new_label

            # Update the visible line’s label text immediately
            txt = item.text()
            if "Label:" in txt:
                item.setText(re.sub(r"Label:.*$", f"Label: {new_label}", txt))
            else:
                item.setText(txt + f" - Label: {new_label}")
            return
        
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
        self.low_confidence_warning.setVisible(False)
        frame_player_container.addWidget(self.low_confidence_warning)

        frame_player_container.addStretch()  # Add stretch after frame player
        layout.addLayout(frame_player_container)

        # Button to toggle display to show gif/segmentation mask
        display_mode_layout = QHBoxLayout()
        self.toggle_display_mode_button = QPushButton("Toggle Mask/Bounding Box Display")
        self.toggle_display_mode_button.clicked.connect(self.toggle_display_mode)

        self.toggle_historical_experiments_button = QPushButton("Toggle Historical Experiments Button")
        self.toggle_historical_experiments_button.clicked.connect(self.toggle_historical_experiments)

        display_mode_layout.addWidget(self.toggle_display_mode_button)
        display_mode_layout.addWidget(self.toggle_historical_experiments_button)
        layout.addLayout(display_mode_layout)
        
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

        # Historical items list (initially hidden)
        self.historical_items = QListWidget()
        self.historical_items.setMaximumHeight(100)
        self.historical_items.hide()
        self.historical_items.itemSelectionChanged.connect(self.show_historical_gif)
        layout.addWidget(self.historical_items)

        # Export/Upload buttons
        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results")  # text will change in historical mode
        self.export_button.clicked.connect(self.export_results)
        self.upload_button = QPushButton("Upload Results")
        self.upload_button.clicked.connect(self.upload_images)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.upload_button)
        layout.addLayout(button_layout)

        # Home button
        home_button = QPushButton("Return to Home Screen")
        home_button.clicked.connect(self.go_to_home)
        layout.addWidget(home_button)

    def switch_detection_list(self, show_historical=False):
        current_list = self.historical_items if show_historical else self.detection_list
        other_list = self.detection_list if show_historical else self.historical_items

        other_list.hide()
        current_list.show()

        if current_list.count() > 0:
            current_list.setCurrentRow(0)
            if not show_historical:
                self.show_detection(self.current_detection_index)
            else:
                self.update_detection_list()
        else:
            self.show_no_detections_message()

    def show_historical_gif(self):
        if self.frame_player.timer.isActive():
            self.frame_player.timer.stop()

        selected = self.historical_items.selectedItems()
        if not selected:
            return

        meta = self._parse_historical_item_text(selected[0].text())
        experiment = meta["experiment"]
        video_basename = meta["video_basename"]
        track_id = meta["track_id"]

        gif_dir = Path(get_results_dir()) / experiment / "tracking_gifs"
        gif_name = f"{video_basename}_{track_id}.gif"
        gif_path = gif_dir / gif_name

        if gif_path.exists():
            self.frame_player.set_gif(str(gif_path))
        else:
            alt = gif_dir / f"{Path(video_basename).stem}_{track_id}.gif"
            if alt.exists():
                self.frame_player.set_gif(str(alt))
            else:
                self.frame_player.clear()
                self.frame_player.setText(f"GIF not found:\n{gif_name}")

        self.prev_button.setEnabled(self.historical_items.currentRow() > 0)
        self.next_button.setEnabled(self.historical_items.currentRow() < (self.historical_items.count() - 1))

    def toggle_historical_experiments(self):
        experiments_root = get_results_dir()
        self.historical_items.clear()

        if not self.reviewing_history:
            try:
                # newest-first
                for experiment in sorted(os.listdir(experiments_root), reverse=True):
                    exp_dir = Path(experiments_root) / experiment
                    det_dir = exp_dir / "detection_results"
                    gif_dir = exp_dir / "tracking_gifs"

                    if not (det_dir.exists() and gif_dir.exists()):
                        continue

                    # each CSV can contain multiple tracks (rows) → iterate rows!
                    for csv_name in os.listdir(det_dir):
                        csv_path = det_dir / csv_name
                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as e:
                            print(f"Error reading {csv_path}: {e}")
                            continue

                        # Create one item per track (row)
                        for _, row in df.iterrows():
                            try:
                                # CSV writer used these column names in save_detections_csv()
                                video_path_str = str(row.get('video_name', ''))
                                video_basename = Path(video_path_str).name  # e.g., "clip.mp4"
                                track_id = int(row.get('Track Id'))
                                time_str = str(row.get('Highest Conf Timestamp', ''))
                                conf_longest = float(row.get('Confidence of Longest Length', 0.0))
                                len_high_conf = float(row.get('Highest Confidence Length', 0.0))
                                label = str(row.get('Label', 'Shark'))

                                # Pretty experiment timestamp for the list label
                                try:
                                    exp_dt = datetime.strptime(experiment, "%m%d%Y_%H%M%S")
                                    exp_disp = exp_dt.strftime("%Y/%-m/%-d %I:%M:%S %p")  # mac/linux
                                except Exception:
                                    # Windows compatibility (no %-m / %-d)
                                    try:
                                        exp_disp = datetime.strptime(experiment, "%m%d%Y_%H%M%S").strftime("%Y/%#m/%#d %I:%M:%S %p")
                                    except Exception:
                                        exp_disp = experiment

                                item_text = (
                                    f"Experiment: {exp_disp} - "
                                    f"Video: {video_basename} - "
                                    f"ID: {track_id} - "
                                    f"Time: {time_str} - "
                                    f"Confidence: {conf_longest:.2f} - "
                                    f"Length: {len_high_conf:.1f}ft - "
                                    f"Label: {label}"
                                )

                                item = QListWidgetItem(item_text)
                                # Store everything we need to resolve the GIF
                                item.setData(Qt.ItemDataRole.UserRole, (experiment, video_basename, track_id))
                                # Optional: visually flag low confidence
                                if conf_longest < 0.65:
                                    item.setText("⚠️ " + item.text())
                                self.historical_items.addItem(item)
                            except Exception as e:
                                print(f"Error creating historical row item from {csv_path}: {e}")

                self.switch_detection_list(show_historical=True)
                self.reviewing_history = True
                self.toggle_display_mode_button.setEnabled(False)
                self.label_combo.setEnabled(True)
                
                # editor mode UI
                self.export_button.setText("Save Changes")
                self.upload_button.setEnabled(False)
                self.historical_label_changes.clear()

            except Exception as e:
                print(f"Error while building historical list: {e}")
                # Fall back to current detections list
                self.switch_detection_list(show_historical=False)
                self.reviewing_history = False
                self.toggle_display_mode_button.setEnabled(True)
        else:
            # toggling back to current run
            self.switch_detection_list(show_historical=False)
            self.reviewing_history = False
            self.toggle_display_mode_button.setEnabled(True)
            
            # Restore normal UI
            self.export_button.setText("Export Results")
            self.upload_button.setEnabled(True)

    def go_to_home(self):
        # Clean up generated files and folders
        if self.cleanup_trees:
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
        if self.cleanup_trees:
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
        if self.reviewing_history:
            self._save_historical_label_changes()
            return
    
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

    def _parse_historical_item_text(self, text: str):
        """
        Parses a historical list row like and returns dict with experiment folder name, video_basename, track_id.
        """
        # strip optional warning prefix
        text = re.sub(r'^⚠️\s*', '', text).strip()

        # Pull out Experiment, Video, ID (robust to spaces / dashes in video name)
        m = re.search(
            r'Experiment:\s*(?P<exp>.*?)\s*-\s*Video:\s*(?P<video>.*?)\s*-\s*ID:\s*(?P<id>\d+)',
            text
        )
        if not m:
            raise ValueError(f"Could not parse historical item text: {text}")

        exp_disp = m.group('exp').strip()            # e.g. 2025/8/14 3:05:22 PM
        video_basename = m.group('video').strip()    # e.g. clip.mp4
        track_id = int(m.group('id'))

        # Parse the human-readable timestamp back to datetime, then to folder format
        # %m/%d handles unpadded numbers too.
        try:
            dt = datetime.strptime(exp_disp, "%Y/%m/%d %I:%M:%S %p")
        except ValueError:
            # As a last resort, parse with a regex (no external deps)
            m2 = re.match(
                r'(?P<Y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})\s+(?P<h>\d{1,2}):(?P<M>\d{2}):(?P<S>\d{2})\s+(?P<ampm>AM|PM)',
                exp_disp
            )
            if not m2:
                raise
            Y = int(m2.group('Y'))
            m_ = int(m2.group('m'))
            d_ = int(m2.group('d'))
            h  = int(m2.group('h'))
            M  = int(m2.group('M'))
            S  = int(m2.group('S'))
            ampm = m2.group('ampm')
            if ampm == 'PM' and h != 12:
                h += 12
            if ampm == 'AM' and h == 12:
                h = 0
            dt = datetime(Y, m_, d_, h, M, S)

        experiment_folder = dt.strftime("%m%d%Y_%H%M%S")  # e.g. 08142025_150522
        csv_name = f"{Path(video_basename).stem}.csv"

        return {
            "experiment": experiment_folder,
            "video_basename": video_basename,
            "csv_name": csv_name,
            "track_id": track_id,
        }

    def _save_historical_label_changes(self):
        """
        Persist queued label changes into their corresponding historical CSV files.
        Keys in self.historical_label_changes are (experiment, csv_name, track_id).
        """
        if not self.historical_label_changes:
            QMessageBox.information(self, "No Changes", "There are no label changes to save.")
            return

        failures = []
        updated_files = set()

        for (experiment, csv_name, track_id), new_label in list(self.historical_label_changes.items()):
            try:
                exp_dir = Path(get_results_dir()) / experiment / "detection_results"
                csv_path = exp_dir / csv_name
                if not csv_path.exists():
                    failures.append(f"Missing CSV: {csv_path}")
                    continue

                # Load, edit, save
                df = pd.read_csv(csv_path)
                if 'Track Id' not in df.columns or 'Label' not in df.columns:
                    failures.append(f"CSV missing columns in {csv_path}")
                    continue

                # Update all rows matching this track id (normally one)
                mask = df['Track Id'].astype(int) == int(track_id)
                if mask.any():
                    df.loc[mask, 'Label'] = new_label
                    # Persist
                    df.to_csv(csv_path, index=False)
                    updated_files.add(str(csv_path))
                    # Remove from pending
                    del self.historical_label_changes[(experiment, csv_name, track_id)]
                else:
                    failures.append(f"Track {track_id} not found in {csv_path}")

            except Exception as e:
                failures.append(f"{csv_name} (Track {track_id}): {e}")

        # Feedback
        if failures and updated_files:
            QMessageBox.warning(
                self,
                "Partial Save",
                "Some changes were saved, but a few failed:\n\n" + "\n".join(failures[:10])
                + ("\n..." if len(failures) > 10 else "")
            )
        elif failures:
            QMessageBox.critical(
                self,
                "Save Failed",
                "Could not save changes:\n\n" + "\n".join(failures[:15])
                + ("\n..." if len(failures) > 15 else "")
            )
        else:
            QMessageBox.information(
                self,
                "Changes Saved",
                "All label changes were saved back to their CSV files."
            )

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
        self._movie = None

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
    
     # --- NEW: apply scaled size while keeping aspect ratio
    def _apply_movie_scaled_size(self):
        if self._movie:
            img = self._movie.currentImage()
            if not img.isNull():
                target = self.size()
                if not target.isValid() or target.isEmpty():
                    # fallback if first layout hasn't run yet
                    target = self.parent().size() if self.parent() and self.parent().size().isValid() else QSize(720, 480)

                scaled = img.size().scaled(target, Qt.AspectRatioMode.KeepAspectRatio)
                self._movie.setScaledSize(scaled)

    def set_gif(self, path: str):
        self.timer.stop()  # pause any slideshow frames

        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheMode.CacheAll)

        # Determine target size (fallback if widget not laid out yet)
        target = self.size()
        if not target.isValid() or target.isEmpty():
            if self.parent() and self.parent().size().isValid():
                target = self.parent().size()
            else:
                target = QSize(720, 480)

        # PRE-SCALE FIRST FRAME to avoid initial flash/jump
        if movie.isValid() and movie.jumpToFrame(0):
            img = movie.currentImage()
            if not img.isNull():
                scaled_size = img.size().scaled(target, Qt.AspectRatioMode.KeepAspectRatio)
                movie.setScaledSize(scaled_size)

                # Paint first frame at final size BEFORE attaching the movie
                first_pix = QPixmap.fromImage(img).scaled(
                    scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(first_pix)

        # Attach and wire signals AFTER scaled size & prepaint
        self._movie = movie

        # Maintain aspect ratio as frames change or widget resizes
        self._movie.frameChanged.connect(lambda _=None: self._apply_movie_scaled_size())

        # Seamless loop WITHOUT setLoopCount: restart at end without recreating movie
        def _restart():
            # keep the same scaled size; just rewind and continue
            self._movie.stop()
            self._movie.jumpToFrame(0)
            self._apply_movie_scaled_size()
            self._movie.start()

        self._movie.finished.connect(_restart)

        self.setMovie(self._movie)
        self._movie.start()

    def resizeEvent(self, event):
        self._apply_movie_scaled_size()
        super().resizeEvent(event)

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

if __name__ == '__main__':

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
    