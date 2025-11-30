import multiprocessing
import sys
import os
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel, QComboBox, 
                             QProgressBar, QStackedWidget, QSizePolicy, QMessageBox, QDialog, QLayout, 
                             QTableWidget, QTableWidgetItem, QDialogButtonBox, QLineEdit, QTreeWidget, 
                             QTreeWidgetItem, QFormLayout, QHeaderView, QCheckBox, QStackedLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QObject, QSettings, QSize, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor, QIcon, QDoubleValidator, QIntValidator, QMovie, QPainter
from PyQt6.QtSvgWidgets import QSvgWidget

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

class QComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.previous_text = self.currentText()
    #     self.currentIndexChanged.connect(self._save_previous_text)

    # def _save_previous_text(self):
    #     self.previous_text = self.currentText()

class SettingsDialog(QDialog):
    settings_updated = pyqtSignal()

    def __init__(self, settings_obj):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 800, 500)
        self.settings_obj = settings_obj
        
        main_layout = QHBoxLayout(self)

        # Left: category list
        self.category_list = QListWidget()
        self.category_list.addItem("Drone Settings")
        self.category_list.addItem("Past Experiments")
        self.category_list.addItem("Confidence Threshold")
        self.category_list.addItem("Cloud Features")
        self.category_list.setFixedWidth(150)
        self.category_list.currentRowChanged.connect(self.switch_category)
        main_layout.addWidget(self.category_list)

        # Right: stacked settings pages
        self.pages = QStackedWidget()
        self.drone_settings_page = DroneSettingsPage(self.settings_obj, self)
        self.historical_settings_page = HistoricalExperimentsPage()
        self.confidence_settings_page = ConfidencePage(self.settings_obj)
        self.cloud_feature_page = CloudUploadPage(self.settings_obj)
        self.pages.addWidget(self.drone_settings_page)
        self.pages.addWidget(self.historical_settings_page)
        self.pages.addWidget(self.confidence_settings_page)
        self.pages.addWidget(self.cloud_feature_page)

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

class HistoricalExperimentsPage(QWidget):
    """Minimal settings page: single button to remove all previous experiments."""
    results_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        if type(self) == HistoricalExperimentsPage:
            # Setup Layout
            layout = QVBoxLayout(self)

            self.checked = set()
            self.export_sharks_only = QCheckBox("Export only sharks to CSV")
            self.historical_experiments_settings = QTableWidget()
            self.historical_experiments_settings.setColumnCount(2)
            self.historical_experiments_settings.verticalHeader().setVisible(False)
            self.historical_experiments_settings.horizontalHeader().setVisible(False)
            self.historical_experiments_settings.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

            # Allow the table to expand vertically to fill available space
            self.historical_experiments_settings.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.historical_experiments_settings.setMinimumHeight(200)

            self.populate_experiment_table()

            experiment_table = QVBoxLayout()
            experiment_table.addWidget(QLabel("Past Experiments"))
            experiment_table.addWidget(self.historical_experiments_settings)
            experiment_table.addWidget(self.export_sharks_only)

            experiment_buttons = QHBoxLayout()
            experiment_buttons.setContentsMargins(0, 0, 0, 0)
            experiment_buttons.setSpacing(8)

            export_selected = QPushButton("Export Selected Results")
            deleted_selected = QPushButton("Delete Selected Results")
            # upload_selected = QPushButton("Upload Selected Results to Cloud")
            select_all = QPushButton("Select All")

            # Make buttons expand horizontally and share space equally
            for btn in (export_selected, deleted_selected, select_all):
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            export_selected.clicked.connect(self.on_export)
            deleted_selected.clicked.connect(self.on_delete)
            # upload_selected.clicked.connect(self.on_upload)
            select_all.clicked.connect(self.on_select_all)

            experiment_buttons.addWidget(select_all)
            experiment_buttons.addWidget(export_selected)
            experiment_buttons.addWidget(deleted_selected)
            # experiment_buttons.addWidget(upload_selected)

            experiment_table.addLayout(experiment_buttons)

            layout.addLayout(experiment_table)
            # Give the experiment_table (index 0) a stretch so the table area grows vertically
            layout.setStretch(0, 1)
            layout.addStretch(0)
    
    def find_checked_boxes(self):
        checked = set()
        for r in range(self.historical_experiments_settings.rowCount()):
            if self.historical_experiments_settings.cellWidget(r, 0).isChecked():
                checked.add(self.historical_experiments_settings.item(r, 1).data(Qt.ItemDataRole.UserRole))
        return checked
            
    def on_export(self):
        selected = self.find_checked_boxes()
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select at least one experiment to export.")
            return

        combined_dfs = []
        failures = []
        found_any = False

        for exp in selected:
            try:
                exp_path = Path(exp) if not isinstance(exp, Path) else exp
                det_dir = exp_path / "detection_results"
                if not det_dir.exists() or not det_dir.is_dir():
                    continue

                for csv_file in sorted(det_dir.glob("*.csv")):
                    try:
                        df = pd.read_csv(csv_file)
                        # Add provenance columns so user knows origin
                        df.insert(0, "experiment_folder", exp_path.name)
                        df.insert(1, "csv_source", csv_file.name)
                        combined_dfs.append(df)
                        found_any = True
                    except Exception as e:
                        failures.append(f"Failed to read {csv_file}: {e}")
            except Exception as e:
                failures.append(f"Error processing {exp}: {e}")

        if not found_any:
            QMessageBox.information(self, "No CSVs Found", "No CSV files were found in the selected experiments' detection_results folders.")
            return

        try:
            final_df = pd.concat(combined_dfs, ignore_index=True)
            if self.export_sharks_only.isChecked():
                    final_df = final_df.query('Label == "Shark"')
        except Exception as e:
            QMessageBox.critical(self, "Combine Error", f"Failed to combine CSVs: {e}")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Combined CSV", "", "CSV Files (*.csv)")
        if not save_path:
            return

        try:
            final_df.to_csv(save_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save combined CSV: {e}")
            return

        msg = f"Combined CSV saved to:\n{save_path}"
        if failures:
            msg += "\n\nSome files failed to be processed. See console for details."
        QMessageBox.information(self, "Export Complete", msg)
        
    
    def on_delete(self):
        selected = self.find_checked_boxes()
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select at least one experiment to delete.")
            return
        try:
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"This will permanently delete the results of {len(selected)} historical experiment{'s' * (len(selected) > 1)}. This action cannot be undone.\n\nDo you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            removed = 0
            for exp in selected:
                shutil.rmtree(exp)
                removed += 1
            QMessageBox.information(self, "Experiments Deleted", f"{removed} experiment{'s' * (removed > 1)} deleted.")
            
            # Refresh Experiment Table
            self.populate_experiment_table()
        except Exception as e:
            QMessageBox.critical(self, "Deletion Error", f"Failed to delete some experiment: {e}")
            return
        
    def on_select_all(self):
        count_checked = len(self.find_checked_boxes())
        at_least_one_checked = (count_checked > 0 and count_checked < self.historical_experiments_settings.rowCount())
            
        for r in range(self.historical_experiments_settings.rowCount()):
            self.historical_experiments_settings.cellWidget(r, 0).setChecked(
                (not at_least_one_checked) * (abs(self.historical_experiments_settings.cellWidget(r, 0).isChecked() - 1))
            )
    
    def on_upload(self):
        checked = self.find_checked_boxes()
        api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload"
        for experiment_dir in checked: 
            try:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zipf:
                    for folder in ['bounding_boxes', 'detection_results', 'false_positives', 'frames', 'masks']:
                        folder_path = os.path.join(experiment_dir, folder)
                        if os.path.exists(folder_path):
                            for root, _, files in os.walk(folder_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, experiment_dir)
                                    zipf.write(file_path, arcname)

                buffer.seek(0)
                files = {'file': ('upload.zip', buffer, 'application/zip')}
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                upload_status, message = "Upload Finished", "Folder uploaded successfully"
            except requests.RequestException as e:
                upload_status, message = "Upload Error", "Failed to Upload folder to cloud storage: {}".format(str(e))
            except Exception as e:
                upload_status, message = "Upload Error", "An unexpected error occurred: {}".format(str(e))
            QMessageBox.information(self, upload_status, message)


    def populate_experiment_table(self):
        experiments_root = get_results_dir()
        # newest-first
        
        self.historical_experiments_settings.clearContents()
        self.historical_experiments_settings.setRowCount(0)

        for experiment in sorted(os.listdir(experiments_root), reverse=True):
            if validate_experiment_date(experiment):
                # First Column: Checkbox
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left:9%; margin-right:2.5%;")

                # Second Column: Experiments
                row_position = self.historical_experiments_settings.rowCount()
                exp_dir = Path(experiments_root) / experiment
                exp_date = format_experiment_date(experiment, to_human=True)
                exp_disp = exp_date + " " + add_experiment_info(exp_dir)

                item = QTableWidgetItem(exp_disp)
                item.setData(Qt.ItemDataRole.UserRole, exp_dir)
                self.historical_experiments_settings.insertRow(row_position)
                self.historical_experiments_settings.setItem(row_position, 1, item)
                self.historical_experiments_settings.setCellWidget(row_position, 0, checkbox)
                
        
        self.historical_experiments_settings.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.historical_experiments_settings.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.historical_experiments_settings.setShowGrid(False)
        # self.historical_experiments_settings.setStyleSheet("""
        #     QTableWidget {
        #         border: none;
        #         background-color: #fafafa;
        #         alternate-background-color: #f0f0f0;
        #     }
        #     QHeaderView::section {
        #         background-color: #e6e6e6;
        #         border: none;
        #         padding: 4px;
        #         font-weight: bold;
        #     }
        # """)

    def clear_all_results(self):
        root = Path(get_results_dir())
        if not root.exists():
            QMessageBox.information(self, "Nothing to Clear", "No historical results directory found.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "This will permanently delete ALL historical experiment results. This action cannot be undone.\n\nDo you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        for child in root.iterdir():
            try:
                if child.is_dir():
                    # remove directory (only within results dir)
                    shutil.rmtree(child)
                    removed += 1
            except Exception:
                # ignore failures for individual items
                pass

        QMessageBox.information(self, "Cleared", f"Removed {removed} experiment folder(s).")
        self.results_cleared.emit()

class ConfidencePage(QWidget):
    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj
        self.confidence_threshold = self.settings_obj.value("confidence_threshold")
        self.min_frames = self.settings_obj.value("min_frames")

        form_layout = QGridLayout(self)

        # Tighten spacing and margins
        form_layout.setContentsMargins(10, 10, 10, 10)  # (left, top, right, bottom)
        form_layout.setVerticalSpacing(4)
        form_layout.setHorizontalSpacing(6)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align the layout to the top

        # Input fields
        self.confidence_input = QLineEdit(self.confidence_threshold)
        self.confidence_input.setValidator(QDoubleValidator(0, 1, 2))

        self.min_frames_input = QLineEdit(str(self.min_frames))
        self.min_frames_input.setValidator(QIntValidator(1, 10000))

        # Buttons
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset to Default")

        save_btn.clicked.connect(self.on_save)
        reset_btn.clicked.connect(self.on_reset)

        # Add widgets
        form_layout.addWidget(QLabel("Enter Confidence Threshold:"), 0, 0)
        form_layout.addWidget(self.confidence_input, 0, 1)
        form_layout.addWidget(QLabel("Minimum Frames:"), 1, 0)
        form_layout.addWidget(self.min_frames_input, 1, 1)
        form_layout.addWidget(reset_btn, 2, 0)
        form_layout.addWidget(save_btn, 2, 1)

        # Optional: shrink to fit contents
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def on_save(self):
        conf_text = self.confidence_input.text().strip()
        min_text = self.min_frames_input.text().strip()

        # Validate confidence
        try:
            conf_val = float(conf_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a numeric confidence value between 0 and 1.")
            return
        if conf_val < 0 or conf_val > 1:
            QMessageBox.warning(self, "Invalid Range", "Confidence must be between 0 and 1.")
            return

        # Validate min tracks
        try:
            min_val = int(min_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter an integer for Minimum Frames.")
            return
        if min_val < 1:
            QMessageBox.warning(self, "Invalid Range", "Minimum Frames must be >= 1.")
            return

        self.settings_obj.setValue("confidence_threshold", f"{conf_val:.2f}")
        self.settings_obj.setValue("min_frames", str(min_val))
        QMessageBox.information(self, "Saved", f"Settings saved: confidence={conf_val:.2f}, min_frames={min_val}")

    def on_reset(self):
        self.settings_obj.setValue("confidence_threshold", "0.40")
        self.settings_obj.setValue("min_frames", "5")
        self.confidence_input.setText(self.settings_obj.value("confidence_threshold"))
        self.min_frames_input.setText(self.settings_obj.value("min_frames"))
        QMessageBox.information(self, "Reset", "Confidence threshold reset to 0.40 and Minimum Frames reset to 5")

class CloudUploadPage(HistoricalExperimentsPage):
    """ Page containing settings related to uploading experiments to Google Cloud Bucket"""
    def __init__(self, settings_obj, parent=None):
        super().__init__()
        self.settings_obj = settings_obj
        # Convert setting string to a boolean
        enable_auto_upload_bool = str(self.settings_obj.value("enable_auto_upload")).lower() == "true"

        # --- Layout setup ---
        layout = QVBoxLayout()

        self.checked = set()

        # --- Auto-upload checkbox ---
        self.auto_upload_checkbox = QCheckBox("Enable automatic Cloud upload when saving")
        self.auto_upload_checkbox.setChecked(enable_auto_upload_bool)

        # Save setting when checkbox is toggled
        self.auto_upload_checkbox.stateChanged.connect(
            lambda state: self.settings_obj.setValue("enable_auto_upload", str(bool(state)))
        )

        # --- Historical experiments table ---
        self.historical_experiments_settings = QTableWidget()
        self.historical_experiments_settings.setColumnCount(2)
        self.historical_experiments_settings.verticalHeader().setVisible(False)
        self.historical_experiments_settings.horizontalHeader().setVisible(False)
        self.historical_experiments_settings.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        # Allow the table to expand vertically
        self.historical_experiments_settings.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.historical_experiments_settings.setMinimumHeight(200)

        self.populate_experiment_table()

        # --- Experiment layout ---
        experiment_table = QVBoxLayout()
        experiment_table.addWidget(QLabel("Past Experiments"))
        experiment_table.addWidget(self.historical_experiments_settings)
        experiment_table.addWidget(self.auto_upload_checkbox)

        # --- Buttons ---
        experiment_buttons = QHBoxLayout()
        experiment_buttons.setContentsMargins(0, 0, 0, 0)
        experiment_buttons.setSpacing(8)

        upload_selected = QPushButton("Upload Selected Results to Cloud")
        select_all = QPushButton("Select All")

        upload_selected.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        select_all.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        upload_selected.clicked.connect(self.on_upload)
        select_all.clicked.connect(self.on_select_all)

        experiment_buttons.addWidget(select_all)
        experiment_buttons.addWidget(upload_selected)

        experiment_table.addLayout(experiment_buttons)

        # --- Final layout assembly ---
        layout.addLayout(experiment_table)
        layout.setStretch(0, 1)
        layout.addStretch(0)
        self.setLayout(layout)

class CustomTracker:
    def __init__(self, distance_threshold=250):
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        self.min_frames = int(self.settings_obj.value("min_frames"))
        self.confidence_threshold = float(self.settings_obj.value("confidence_threshold"))
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
            print("Starting new track")
            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])

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
                print("Running Segmentation")
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
                
                filename = f"{Path(video_path).name}_{track_id}.jpg"
                
                # Save original frame
                cv2.imwrite(os.path.join(output_dir, 'frames', filename), longest_frame)
                
                # Save frame with bounding box
                boxed_frame = longest_frame.copy()
                cv2.rectangle(boxed_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                label = f"ID: {track_id}, Conf: {longest_confidence:.2f}, Length: {length_str}"
                cv2.putText(boxed_frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                bounding_box_path = os.path.join(output_dir, 'bounding_boxes', filename)
                cv2.imwrite(bounding_box_path, boxed_frame)
                mask_path = os.path.join(output_dir, 'masks', filename)
                cv2.imwrite(mask_path, mask_overlay)
                
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
        # Read settings 
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.drone_type = drone_type
        self.altitude = altitude
        self.detection_threshold = float(self.settings_obj.value("confidence_threshold"))
        self.drone_settings = json.loads(self.settings_obj.value("drone_settings"))
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        custom_tracker = CustomTracker()
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        custom_tracker.fov_radians = self.drone_settings[self.drone_type]["Resolution"][f"({video_width}, {video_height})"]
        custom_tracker.drone_altitude = self.altitude

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'bounding_boxes'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'false_positives'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'detection_results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "tracking_gifs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

        min_frame_skip, max_frame_skip = 10, 60
        frame_skip = min_frame_skip
        consecutive_empty_frames = 0
        max_empty_frames = 1 * fps
        
        # self.detection_threshold = 0.4

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
                
            #     passes_requirements = (num_frames >= self.min_frames) and (avg_confidence > self.confidence_threshold)
            # if not passes_requirements:
            #     print(f"Track detected below minimum requirements: num_frames = {num_frames}, avg_confidence = {avg_confidence}")
            #     continue                
            # print("This should only appear for valid tracks")

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
        csv_path = os.path.join(output_dir, f'{Path(self.video_path).name}.csv')
        print(f"Starting save to {csv_path}")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['video_name', 'Track Id', 'Highest Conf Timestamp', 'Highest Confidence', 'Average Confidence', 
                        'Lowest Confidence', 'Longest Length', 'Highest Confidence Length',
                        'Number of Detections', 'Meets Thresholds', 'Confidence of Longest Length', 'Label']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for track_id, track in tracks.items():
                # meets_thresholds = (len(track['confidences']) >= 10 and 
                #                     np.mean(track['confidences']) > 0.4)
                meets_thresholds = True
                
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

def validate_experiment_date(date_str):
    """
    Returns True if date_str matches the format <mmddYYYY_HHMMSS>, else False.
    """
    try:
        datetime.strptime(date_str, "%m%d%Y_%H%M%S")
        return True
    except Exception:
        return False

def add_experiment_info(experiment_path: Path): 
    """
    Given an experiment folder name (e.g. "09252025_110653"), returns a string like "(1 video, 3 sharks)".
    Counts videos by unique video basenames in the masks folder, and sharks by number of mask images.
    """
    gif_dir = os.path.join(experiment_path, "tracking_gifs")
    if not os.path.isdir(gif_dir):
        return "(0 video, 0 sharks)"

    gif_files = [f for f in os.listdir(gif_dir) if f.lower().endswith(".gif")]
    video_names = set()
    for f in gif_files:
        # Example gif filename: "clip.mp4_1.gif" or "TRIMMED_2023-05-05_Transect_DJI_0516.mp4_1.gif"
        # Split at last underscore to get video name (handles underscores in video name)
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            video_names.add(parts[0])
        else:
            # fallback: just use the whole name minus extension
            video_names.add(os.path.splitext(f)[0])

    num_videos = len(video_names)
    num_sharks = len(gif_files)
    return f"({num_videos} video{'s' if num_videos != 1 else ''}, {num_sharks} detection{'s' if num_sharks != 1 else ''})"

def format_experiment_date(date_str, to_human=True):
    """
    Convert experiment folder name <mmddYYYY_HHMMSS> to human-readable <YYYY/m/d h:m:s AM/PM> and vice versa.
    If to_human is True, convert folder name to human-readable.
    If False, convert human-readable back to folder name.
    """
    if to_human:
        # Folder name to human-readable
        try:
            dt = datetime.strptime(date_str, "%m%d%Y_%H%M%S")
            return dt.strftime("%Y/%-m/%-d %I:%M:%S %p")  # mac/linux
        except Exception:
            try:
                return datetime.strptime(date_str, "%m%d%Y_%H%M%S").strftime("%Y/%#m/%#d %I:%M:%S %p")  # windows
            except Exception:
                return date_str
    else:
        # Human-readable to folder name
        try:
            dt = datetime.strptime(date_str, "%Y/%m/%d %I:%M:%S %p")
            return dt.strftime("%m%d%Y_%H%M%S")
        except Exception:
            # Try to parse with regex fallback
            m = re.match(
                r'(?P<Y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})\s+(?P<h>\d{1,2}):(?P<M>\d{2}):(?P<S>\d{2})\s+(?P<ampm>AM|PM)',
                date_str
            )
            if m:
                Y = int(m.group('Y'))
                m_ = int(m.group('m'))
                d_ = int(m.group('d'))
                h  = int(m.group('h'))
                M  = int(m.group('M'))
                S  = int(m.group('S'))
                ampm = m.group('ampm')
                if ampm == 'PM' and h != 12:
                    h += 12
                if ampm == 'AM' and h == 12:
                    h = 0
                dt = datetime(Y, m_, d_, h, M, S)
                return dt.strftime("%m%d%Y_%H%M%S")
            return date_str

class MainWindow(QMainWindow):
    upload_finished = pyqtSignal(bool, str)  # (success, message)
    resized = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1000, 800)

        self.initialize_settings()
        self.init_ui()
        self.init_attributes()  
        self.setup_model()
        self.setup_signal_handlers()

        # Connect the upload_finished signal
        self.upload_finished.connect(self.on_upload_finished)
    
    def initialize_settings(self):
        # Drone Settings
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
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

        if not self.settings_obj.value("drone_settings"):
            self.settings_obj.setValue("drone_settings", json.dumps(default_drones))
        
        # Confidence
        if not self.settings_obj.value("confidence_threshold"):
            self.settings_obj.setValue("confidence_threshold", ".40")
        if not self.settings_obj.value("min_frames"):
            self.settings_obj.setValue("min_frames", "5")

        # Cloud Settings
        if not self.settings_obj.value("enable_auto_upload"):
            self.settings_obj.setValue("enable_auto_upload", "false")
            
    def load_drone_settings(self):
        settings_dialog = SettingsDialog(self.settings_obj)
        settings_dialog.settings_updated.connect(self.update_available_drones)
        settings_dialog.exec()
        
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
        self.experiments = []
        self.gif_active = False

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setup_home_banner()
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

    def setup_home_banner(self):
        # Banner container with left/right buttons and centered logo
        banner_widget = QWidget()
        banner_widget.setStyleSheet("background-color: #1d2633;")
        banner_layout = QHBoxLayout(banner_widget)
        banner_layout.setContentsMargins(20, 8, 20, 8)
        banner_layout.setSpacing(8)

        # Left button (exposed as attribute for later connections)
        self.banner_left_button = QPushButton()
        self.banner_left_button.setIcon(QIcon(resource_path("assets/images/clock-history.svg")))
        self.banner_left_button.setFixedSize(40, 40)
        self.banner_left_button.setFlat(True)
        self.banner_left_button.setStyleSheet(
            "color: white; background: transparent; border: none; font-size: 18px;"
        )
        self.banner_left_button.setToolTip("Previous Experiments")
        
        self.banner_left_button.clicked.connect(self.go_to_review_history) # sets top widget as review
        self.banner_left_button.clicked.connect(lambda: setattr(self, "reviewing_history", False))

        # Center logo
        logo_label = QLabel()
        logo_path = resource_path('assets/images/logo-white.png')
        banner_pixmap = QPixmap(logo_path)
        if not banner_pixmap.isNull():
            banner_pixmap = banner_pixmap.scaledToHeight(40, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(banner_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Right button (exposed as attribute for later connections)
        self.banner_right_button = QPushButton()
        self.banner_right_button.setIcon(QIcon(resource_path("assets/images/gear-fill.svg")))
        self.banner_right_button.clicked.connect(self.load_drone_settings)
        self.banner_right_button.setFixedSize(40, 40)
        self.banner_right_button.setFlat(True)
        self.banner_right_button.setStyleSheet(
            "color: white; background: transparent; border: none; font-size: 18px;"
        )
        self.banner_right_button.setToolTip("Settings")

        # Layout: left button, spacer, logo, spacer, right button
        banner_layout.addWidget(self.banner_left_button, 0, Qt.AlignmentFlag.AlignLeft)
        banner_layout.addStretch(1)
        banner_layout.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignCenter)
        banner_layout.addStretch(1)
        banner_layout.addWidget(self.banner_right_button, 0, Qt.AlignmentFlag.AlignRight)

        banner_widget.setFixedHeight(60)
        self.layout.addWidget(banner_widget)

        # keep reference for tests/other code
        self.banner = banner_widget
    
    def toggle_banner_buttons(self, review=True):
        self.banner_left_button.clicked.disconnect()
        self.banner_right_button.clicked.disconnect()
        self.banner_left_button.setEnabled(True)
        self.banner_right_button.setEnabled(True)
        self.banner_left_button.show()
        self.banner_right_button.show()
        
        if review == True:
            # Review Window
            self.banner_left_button.setText("")
            self.banner_left_button.setIcon(QIcon(resource_path("assets/images/house-fill.svg")))
            self.banner_left_button.setToolTip("Go to Home")
            self.banner_left_button.clicked.connect(self.go_to_home)
            
            if self.review_dropdown.count() == 0:
                self.banner_right_button.setEnabled(False)
                # self.banner_right_button.hide()

            self.banner_right_button.setText("")
            self.banner_right_button.setIcon(QIcon(resource_path("assets/images/pencil-fill.svg")))
            self.banner_right_button.setToolTip("Edit Results")
            self.banner_right_button.clicked.connect(self.toggle_edit_state)
        else:
            # Home Screen
            self.banner_left_button.setIcon(QIcon(resource_path("assets/images/clock-history.svg")))
            self.banner_left_button.setFlat(True)
            self.banner_left_button.setStyleSheet(
                "color: white; background: transparent; border: none; font-size: 18px;"
            )
            self.banner_left_button.setToolTip("Previous Experiments")
            
            self.banner_left_button.clicked.connect(self.go_to_review_history) # sets top widget as review
            self.banner_left_button.clicked.connect(lambda: setattr(self, "reviewing_history", False))

            self.banner_right_button.setIcon(QIcon(resource_path("assets/images/gear-fill.svg")))
            self.banner_right_button.clicked.connect(self.load_drone_settings)
            self.banner_right_button.setFlat(True)
            self.banner_right_button.setStyleSheet(
                "color: white; background: transparent; border: none; font-size: 18px;"
            )
            self.banner_right_button.setToolTip("Settings")

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
        value = self.settings_obj.value("drone_settings")
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
        self.select_videos_button.clicked.connect(self.select_videos_2)
        # self.select_videos_button.clicked.connect(self.update_remove_buttons)
        # self.select_videos_button.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.select_videos_button)

        # Remove buttons in horizontal layout
        remove_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected Video(s)")
        self.remove_button.clicked.connect(self.remove_selected_videos)
        self.remove_button.setEnabled(False)  # Initially disabled
        # remove_layout.addWidget(self.remove_button)

        self.remove_all_button = QPushButton("Remove All Videos")
        self.remove_all_button.clicked.connect(self.remove_all_videos)
        self.remove_all_button.setEnabled(False)  # Initially disabled
        remove_layout.addWidget(self.remove_all_button)
        layout.addLayout(remove_layout)

        self.video_list = QTableWidget()
        self.video_list.setShowGrid(False)
        self.video_list.setMaximumHeight(250)
        self.video_list.verticalHeader().setVisible(False)
        self.video_list.horizontalHeader().setVisible(False)
        self.video_list.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        # Set table headers
        self.video_list.setColumnCount(2)
        
        # self.video_list.rowCountChanged.connect(self.update_remove_buttons)
        self.video_list.updateInternalOrder = self.update_video_order
        layout.addWidget(self.video_list)
        # Video added
        
        # buttons
        # add_video_button = QPushButton()
        # remove_video_button = QPushButton() 
        # self.video_list.setCellWidget(row_position, 2, combo)
        # self.historical_items.setItem(row_position, col, cell)

        # Select Drone and Altitude Dropdown
        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("Select Drone Model:"), 0, 0)
        self.drone_select = QComboBox()
        self.update_available_drones()
        self.drone_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Settings button
        
        form_layout.addWidget(self.drone_select, 0, 1)

        form_layout.addWidget(QLabel("Enter Drone Altitude:"), 1, 0)
        self.altitude_input = QLineEdit('40')
        self.altitude_input.setValidator(QDoubleValidator(0, 999, 2))
        self.altitude_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form_layout.addWidget(self.altitude_input, 1, 1)

        layout.addLayout(form_layout)
        # Process Videos button
        process_layout = QVBoxLayout()
        self.process_button = QPushButton("Process Videos")
        self.process_button.clicked.connect(self.toggle_processing)
        self.process_button.setEnabled(False)  # Initially disabled
        process_layout.addWidget(self.process_button)
        layout.addLayout(process_layout)
        layout.addStretch()

    def toggle_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.confirm_cancel_processing()
        
    def create_progress_dialog(self):
        """Create a non-modal dialog that displays the frame, progress bar, and timer."""
        # Close existing dialog if present
        if getattr(self, "progress_display_dialog", None):
            try:
                self.progress_display_dialog.close()
            except Exception:
                pass

        dlg = QDialog(self)
        dlg.setWindowTitle("Processing Preview")
        dlg.setModal(False)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

         # Frame display
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_display.hide()

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignBottom)

        # Timer label (initially hidden)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.timer_label.hide()

        # Reparent the existing widgets into the dialog so live updates continue
        self.cancel_processsing_button = QPushButton("Cancel Processing")
        self.cancel_processsing_button.clicked.connect(self.toggle_processing)
        
        layout.addWidget(self.frame_display)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.timer_label)
        layout.addWidget(self.cancel_processsing_button)

        dlg.setLayout(layout)
        dlg.resize(800, 600)
        self.frame_display.show()
        self.progress_bar.show()
        self.timer_label.show()
        
        self.progress_bar.setValue(0)
        self.start_time = QDateTime.currentDateTime()
        self.timer.start(1000)
        self.elapsed_time = 0
        self.update_timer()

        # Disable Homescreen Buttons While Processing
        self.progress_display_dialog = dlg
        self.progress_display_dialog.show()
        
    def get_valid_resolutions_for_drone(self, drone_name):
        value = self.settings_obj.value("drone_settings")
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
        self.remove_button.setEnabled(False)
        self.remove_all_button.setEnabled(False)
        self.video_list.setEnabled(False)
        self.select_videos_button.setEnabled(False)
        self.drone_select.setEnabled(False)
        self.altitude_input.setEnabled(False)
        self.process_button.setEnabled(False)

        self.create_progress_dialog()

        # Reset processing state
        self.tracks = {}
        self.current_video_index = 0
        self.processed_videos = 0
        self.total_videos = len(self.video_queue)
        
        # Reset any prefixed emojis
        for i in range(self.video_list.rowCount()): 
            item = self.video_list.item(i, 0)
            item.setText(item.text().replace('🔎 ', '').replace('✅ ', ''))

        self.video_queue = [self.video_list.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.rowCount())]
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
                self.remove_all_button.setEnabled(self.video_list.rowCount() > 0)
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
        for i in range(self.video_list.rowCount()):
            item = self.video_list.item(i, 0)
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
        for i in range(self.video_list.rowCount()):
            item = self.video_list.item(i, 0)
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
        for i in range(self.video_list.rowCount()):
            item = self.video_list.item(i, 0)
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
        self.progress_display_dialog.close()
        
        # Reset button states
        self.remove_button.setEnabled(True)
        self.remove_all_button.setEnabled(True)
        self.video_list.setEnabled(True)
        self.select_videos_button.setEnabled(True)
        self.drone_select.setEnabled(True)
        self.altitude_input.setEnabled(True)
        self.process_button.setEnabled(True)

        self.remove_all_button.setEnabled(self.video_list.rowCount() > 0)

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
        for i in range(self.video_list.rowCount()):
            item = self.video_list.item(i, 0)
            clean_text = item.text().replace('🔎 ', '').replace('✅ ', '')
            item.setText(clean_text)

        # Delete Experiment Directory
        shutil.rmtree(self.current_output_dir)

        # Reset processing state
        self.current_video_index = 0
        self.processed_videos = 0
        self.tracks = {}

        print("Processing cancelled")

    def update_remove_buttons(self):    
        has_any_items = self.video_list.rowCount() > 0
        self.remove_all_button.setEnabled(has_any_items and not self.is_processing)
        self.process_button.setEnabled(has_any_items and not self.is_processing)

    def select_videos_2(self):
        file_dialog = QFileDialog()
        video_files, _ = file_dialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")

        # Get the current list of file paths in the table
        current_files = set()
        for row in range(self.video_list.rowCount()):
            item = self.video_list.item(row, 0)
            if item:
                current_files.add(item.data(Qt.ItemDataRole.UserRole))

        for file_path in video_files:
            if file_path not in current_files:
                file_name = os.path.basename(file_path)
                row_position = self.video_list.rowCount()
                self.video_list.insertRow(row_position)

                # First column: filename
                item = QTableWidgetItem(file_name)
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.video_list.setItem(row_position, 0, item)
                # Stretch the video column to fill available space
                self.video_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
                self.video_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
                # Second column: delete button
                delete_btn = QPushButton("")
                delete_btn.setIcon(QIcon(resource_path("assets/images/x-lg.svg")))
                delete_btn.setStyleSheet("background: transparent; border: none;")
                def delete_row():
                    button = self.sender()
                    if button:
                        index = self.video_list.indexAt(button.pos())
                        self.video_list.removeRow(index.row())

                delete_btn.clicked.connect(delete_row)
                self.video_list.setCellWidget(row_position, 1, delete_btn)
        self.update_remove_buttons()

    def select_videos(self):
        file_dialog = QFileDialog()
        video_files, _ = file_dialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        
        # Get the current list of file paths
        current_files = set(self.video_list.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.rowCount()))
        
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
        self.video_list.setRowCount(0)
        # self.video_list.setHorizontalHeaderLabels(["Video",  ""])
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
            # Display most recent detections
            self.toggle_banner_buttons()
            self.switch_detection_list(show_historical=True)
            self.setup_review_dropdown()
            self.render_historical_experiments()
            self.toggle_edit_state(set_state=True)
    
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
        
        # Close processing window and show completion popup with both time and detections
        self.progress_display_dialog.close()
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
            # self.show_confidence_warning()
            self.highlight_current_detection()
        else:
            print(f"Error: Invalid detection index: {index}")
            self.show_no_detections_message()

    def show_no_detections_message(self):
        self.frame_player.clear()
        self.frame_player.setText("No detections available")

    def update_detection_list(self):
        # Use a table format for the detection list, matching the historical items table
        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']
        self.detection_list.setRowCount(0)
        self.detection_list.clearContents()
        self.detection_list.setHorizontalHeaderLabels(labels)

        print(f"Updating detection list with {len(self.sorted_tracks)} tracks")

        for index, (key, track) in enumerate(self.sorted_tracks):
            try:
                experiment_disp = getattr(self, "current_experiment", "")
                if experiment_disp:
                    experiment_disp = format_experiment_date(experiment_disp, to_human=True)
                else:
                    experiment_disp = ""

                video_basename = Path(track['video_name']).name
                track_id = track['unique_id']
                timestamp = track['longest_timestamp'] if 'longest_timestamp' in track else track['timestamps'][0]
                time_str = datetime.utcfromtimestamp(timestamp / 1000).strftime("%M:%S")
                conf_longest = track.get('longest_conf', 0.0)
                len_high_conf = track.get('longest_length', 0.0)
                label = track.get('label', 'Shark')

                row_position = self.detection_list.rowCount()
                self.detection_list.insertRow(row_position)
                values = [
                    experiment_disp,
                    video_basename,
                    str(track_id),
                    time_str,
                    f"{conf_longest:.2f}",
                    f"{len_high_conf:.1f}ft",
                    label
                ]
                for col, value in enumerate(values):
                    if col == 6:
                        combo = QComboBox()
                        combo.addItems(["Shark", "Kelp", "Dolphin", "Surfer", "Boat", "Bird", "Duplicate", "None", "Other"])
                        combo.setCurrentText(label)
                        combo.currentIndexChanged.connect(lambda _, idx=index, c=combo: self._update_label_from_table(idx, c))
                        self.detection_list.setCellWidget(row_position, col, combo)
                    else:
                        cell = QTableWidgetItem(value)
                        if col == 4 and conf_longest < 0.65:
                            cell.setForeground(QColor('red'))
                        self.detection_list.setItem(row_position, col, cell)

                self.detection_list.item(row_position, 0).setData(Qt.ItemDataRole.UserRole, index)
            except KeyError as e:
                print(f"Missing key in track data: {e}")
            except Exception as e:
                print(f"Error creating table row for track: {str(e)}")

        self.detection_list.resizeColumnsToContents()
        self.detection_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.detection_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        print(f"Updated detection list with {self.detection_list.rowCount()} items")
        self.highlight_current_detection()

    def _update_label_from_table(self, index, combo):
        # Update label in sorted_tracks when changed from table dropdown
        if 0 <= index < len(self.sorted_tracks):
            key, track = self.sorted_tracks[index]
            new_label = combo.currentText()
            track['label'] = new_label
            print(f"Label updated for track {key} to {new_label}")
    
    def mark_for_deletion(self):
        if self.reviewing_history:
            row = self.historical_items.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Mark for Deletion", "No track selected.")
                return

            experiment_disp = self.historical_items.item(row, 0).text()
            experiment = format_experiment_date(experiment_disp, to_human=False)
            video_basename = self.historical_items.item(row, 1).text()
            track_id = self.historical_items.item(row, 2).text()
            csv_name = f"{Path(video_basename)}.csv"

            key = (experiment, csv_name, int(track_id))
            self.historical_label_changes[key] = "Delete"

            # Remove the row from the QTableWidget
            self.historical_items.removeRow(row)
            self.historical_items.selectRow(max(0, row - 1))
            self.show_historical_gif()
        else:
            return
    
    def delete_track(self, experiment, csv_name, track_id):
        """
        Delete a track given (experiment, csv_name, track_id).
        Removes images and CSV row for the specified track.
        Before deleting files, disables the GIF display if showing the deleted track.
        """
        # Detach files from player 
        self.frame_player.clear_frame()

        exp_dir = Path(get_results_dir()) / experiment

        # Remove images from bounding_boxes, frames, masks, tracking_gifs
        folders = ["bounding_boxes", "frames", "masks", "tracking_gifs"]

        # Check if this is the last track in the experiment
        det_dir = exp_dir / "detection_results"
        total_tracks = 0
        if det_dir.exists():
            for csv_file in det_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    total_tracks += len(df)
                except Exception:
                    pass

        # If this is the last track, delete the entire experiment directory
        if total_tracks == 1:
            try:
                shutil.rmtree(exp_dir)
                print(f"Deleted experiment directory: {exp_dir}")
                self.sort_tracks()
                self.setup_review_dropdown()
                self.render_historical_experiments()
                return
            except Exception as e:
                print(f"Error deleting experiment directory {exp_dir}: {e}")

        # Delete track-specific files
        for folder in folders:
            folder_path = exp_dir / folder
            print(csv_name)
            # video_name = str(csv_name).split('/')[-1].strip('.csv')
            pattern = f"{Path(csv_name).stem}_{track_id}.*"
            print(pattern)
            for file in folder_path.glob(pattern):
                try:
                    file.unlink()
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")

        # Remove row from CSV in detection_results
        csv_path = det_dir / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if len(df) == 1:
                    print(f"Removing last detection for video. Deleting CSV")
                    csv_path.unlink()
                else:
                    df = df[df['Track Id'].astype(int) != int(track_id)]
                    df.to_csv(csv_path, index=False)
                    print(f"Removed track {track_id} from {csv_path}")
            except Exception as e:
                print(f"Error updating CSV {csv_path}: {e}")

        # Remove from in-memory tracks and update detection list
        for video_name in list(self.tracks.keys()):
            self.tracks[video_name] = {
                tid: t for tid, t in self.tracks[video_name].items()
                if int(tid) != int(track_id)
            }
            if not self.tracks[video_name]:
                del self.tracks[video_name]

        self.sort_tracks()
        self.setup_review_dropdown()
        self.render_historical_experiments()

    def show_selected_detection(self):
        # For QTableWidget, get the selected row and show that detection
        if isinstance(self.detection_list, QTableWidget):
            selected_ranges = self.detection_list.selectedRanges()
            if selected_ranges:
                row = selected_ranges[0].topRow()
                index = self.detection_list.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if index != self.current_detection_index:
                    self.show_detection(index)
                # self.show_confidence_warning()
        else:
            # Fallback for QListWidget (shouldn't be used anymore)
            selected_items = self.detection_list.selectedItems()
            if selected_items:
                index = selected_items[0].data(Qt.ItemDataRole.UserRole)
                if index != self.current_detection_index:
                    self.show_detection(index)
                # self.show_confidence_warning()

    def highlight_current_detection(self):
        # For QTableWidget, select the row corresponding to current_detection_index
        for row in range(self.detection_list.rowCount()):
            index = self.detection_list.item(row, 0).data(Qt.ItemDataRole.UserRole) + 1
            if index == self.current_detection_index:
                self.detection_list.selectRow(row)
                self.detection_list.scrollToItem(self.detection_list.item(row, 0))
            # else:
            #     self.detection_list.selectRow(row)
            #     #self.detection_list.setRangeSelected(
            #     #     QTableWidgetSelectionRange(0, row, self.detection_list.columnCount(), row))
            #         # self.detection_list.visualItemRect(self.detection_list.item(row, 0)), False

    def update_label(self):
        print(self.label_combo.previous_text)
        if self.reviewing_history:
            row = self.historical_items.currentRow()
            if row < 0:
                return

            # Get experiment, video, track_id from the cells
            experiment_disp = self.historical_items.item(row, 0).text()
            
            experiment = format_experiment_date(experiment_disp, to_human=False)
            video_basename = self.historical_items.item(row, 1).text()
            track_id = self.historical_items.item(row, 2).text()
            csv_name = f"{Path(video_basename)}.csv"
            print(csv_name)

            key = (experiment, csv_name, int(track_id))
            new_label = self.label_combo.currentText()
            print(new_label)
            print(self.label_combo.currentIndex()) 
            self.historical_label_changes[key] = new_label 
            print(self.historical_label_changes)
            
            if self.historical_label_changes[key] == None:
                pass
            if self.historical_label_changes[key] == self.label_combo.previous_text:
                del self.historical_label_changes[key]
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
        self.setup_review_dropdown()
        self.render_historical_experiments()
        self.toggle_banner_buttons(review=True)        

    def toggle_edit_state(self, set_state=None):
        if set_state:
            self.save_changes_button.setEnabled(set_state)
            # self.delete_track_button.setEnabled(set_state)
            for r in range(self.historical_items.rowCount()):
                self.historical_items.cellWidget(r, 6).setEnabled(set_state)
                self.historical_items.cellWidget(r, 7).setEnabled(set_state)
        else:
            self.save_changes_button.setEnabled(not self.save_changes_button.isEnabled())
            # self.delete_track_button.setEnabled(not self.delete_track_button.isEnabled())
            for r in range(self.historical_items.rowCount()):
                self.historical_items.cellWidget(r, 6).setEnabled(not self.historical_items.cellWidget(r, 6).isEnabled())
                self.historical_items.cellWidget(r, 7).setEnabled(not self.historical_items.cellWidget(r, 7).isEnabled())

    def toggle_review_buttons(self, enable):
        self.historical_items.setEnabled(enable)
        self.save_changes_button.setEnabled(enable)
        self.toggle_display_mode_button.setEnabled(enable)

        if enable == False:
            self.toggle_display_mode_button.hide()
        else:
            self.toggle_display_mode_button.show()
        
    def toggle_display_mode(self):
        # Historical mode: always show mask overlay if available
        if self.reviewing_history:
            if self.gif_active:
                row = self.historical_items.currentRow()  # Toggle should be disabled in historical mode
                if row < 0:
                    return
                experiment_disp = self.historical_items.item(row, 0).text()
                experiment = format_experiment_date(experiment_disp, to_human=False)
                video_basename = self.historical_items.item(row, 1).text()
                track_id = self.historical_items.item(row, 2).text()
                mask_dir = Path(get_results_dir()) / experiment / "masks"
                mask_filename = f"{video_basename}_{track_id}.jpg"
                mask_path = mask_dir / mask_filename

                if mask_path.exists():
                    mask_overlay = cv2.imread(str(mask_path))
                    frame_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(self.frame_player.size(), Qt.AspectRatioMode.KeepAspectRatio)
                    self.frame_player.set_static_pixmap(scaled_pixmap)
                    self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFinOutline.svg")))
                else:
                    dlg = QMessageBox(self)
                    dlg.setWindowTitle("Alert")
                    dlg.setText("Error: No mask drawn for this track")
                    dlg.exec()
                # Do NOT start/stop timer in historical mode
                self.gif_active = False
                return
            else:
                self.show_historical_gif()
                return
        # Non-historical mode: toggle between mask overlay and animation
        if self.frame_player.timer.isActive():
            self.frame_player.timer.stop()
            current_track = self.sorted_tracks[self.current_detection_index]
            if 'mask_overlay' not in current_track[1]:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Alert")
                dlg.setText("Error: No mask drawn for this track")
                dlg.exec()
            else:
                mask_overlay = current_track[1]['mask_overlay']
                if mask_overlay is not None:
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
        
        # Add dropdown for historical review
        self.review_dropdown = QComboBox()
        layout.addWidget(self.review_dropdown)
        
        # Frame player container with horizontal centering
        # frame_player_container = QGridLayout()
        frame_player_container = QVBoxLayout()
        # frame_player_container.addStretch()  # Add stretch before frame player
        
        self.frame_player = FramePlayer()
        self.frame_player.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.frame_player.setA
        self.frame_player.setMinimumSize(int(720), int(480))
        self.frame_player.setBaseSize(int(720), int(480))
        #self.frame_player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_player_container.addWidget(self.frame_player)
        # frame_player_container.addWidget(self.frame_player, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add warning when detection falls before 
        self.low_confidence_warning = QLabel("⚠️ Warning: Low confidence in this detection. Please double check the image to make sure the boxed area is a shark!")
        self.low_confidence_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.low_confidence_warning.setMinimumHeight(30)
        self.low_confidence_warning.setVisible(False)
        frame_player_container.addWidget(self.low_confidence_warning)

        # frame_player_container.addStretch()  # Add stretch after frame player
        layout.addLayout(frame_player_container)
        # layout.addWidget(self.low_confidence_warning)

        # Button to toggle display to show gif/segmentation mask
        self.toggle_display_mode_button = QPushButton("", self.frame_player)
        self.toggle_display_mode_button.clicked.connect(self.toggle_display_mode)
        self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
        self.toggle_display_mode_button.setFixedSize(40, 40)
        self.toggle_display_mode_button.setIconSize(QSize(40, 40))
        self.toggle_display_mode_button.setStyleSheet(
            "color: white; background: transparent; border: none; font-size: 18px;"
        )

        def update_button_position():
            rect = self.frame_player.content_rect()
            if rect.isNull():
                return

            # Convert from frame_player-local coords → global → back to parent coords
            top_left = self.frame_player.mapToParent(rect.topLeft())
            
            # Position button near bottom-right inside the actual content rect
            btn_x = top_left.x() + rect.width() - self.toggle_display_mode_button.width() - 13
            btn_y = top_left.y() + rect.height() - self.toggle_display_mode_button.height() - 38

            self.toggle_display_mode_button.move(btn_x, btn_y)

        self.frame_player.resized.connect(self.update_button_position)
        # update_button_position(self.frame_player._movie.scaledSize().width(), self.frame_player._movie.scaledSize().height())

        # layout.addWidget(self.toggle_display_mode_button)

        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']

        self.detection_list = QTableWidget()
        self.detection_list.setColumnCount(len(labels))
        self.detection_list.setMaximumHeight(100)
        self.detection_list.hide()
        self.detection_list.itemSelectionChanged.connect(self.show_selected_detection)
        layout.addWidget(self.detection_list)

        # Historical items list (initially hidden)
        self.historical_items = QTableWidget()
        self.historical_items.setColumnCount(len(labels))
        self.historical_items.setMaximumHeight(100)
        self.historical_items.hide()
        self.historical_items.itemSelectionChanged.connect(self.show_historical_gif)
        self.historical_items.itemSelectionChanged.connect(self.set_current_label_combo)
        self.historical_items.setHorizontalHeaderLabels(labels)
        layout.addWidget(self.historical_items)

        # Export/Upload buttons
        button_layout = QHBoxLayout()
        self.save_changes_button = QPushButton("Save Changes")  # text will change in historical mode
        self.save_changes_button.clicked.connect(self.export_results)
        # self.delete_track_button = QPushButton("Delete Track")
        # self.delete_track_button.clicked.connect(self.mark_for_deletion)
        
        button_layout.addWidget(self.save_changes_button)
        # button_layout.addWidget(self.delete_track_button)
        layout.addLayout(button_layout)

        # Finish Review Setup
        self.setup_review_dropdown()
        self.review_dropdown.currentIndexChanged.connect(self.render_historical_experiments)

    def update_button_position(self):
        if self.frame_player and self.toggle_display_mode_button:
            rect = self.frame_player.content_rect()
            if rect.isNull():
                return

            # Convert from frame_player-local coords → global → back to parent coords
            top_left = self.frame_player.mapToParent(rect.topLeft())
            
            # Position button near bottom-right inside the actual content rect
            btn_x = top_left.x() + rect.width() - self.toggle_display_mode_button.width() - 13
            btn_y = top_left.y() + rect.height() - self.toggle_display_mode_button.height() - 38

            self.toggle_display_mode_button.move(btn_x, btn_y)

        # layout.addWidget(self.toggle_display_mode_button)
    def switch_detection_list(self, show_historical=False):
        current_list = self.historical_items if show_historical else self.detection_list
        other_list = self.detection_list if show_historical else self.historical_items

        other_list.hide()
        current_list.show()

        if current_list.rowCount() > 0:
            current_list.setCurrentCell(0, 0)
            if not show_historical:
                self.toggle_review_buttons(enable=True)
                self.show_detection(self.current_detection_index)
                self.review_dropdown.hide()
            else:
                self.toggle_review_buttons(enable=True)
                self.update_detection_list()
                self.review_dropdown.show()
        else:
            self.toggle_review_buttons(enable=False)
            self.show_no_detections_message()

    def set_current_label_combo(self):
        """
        Set self.label_combo to the QComboBox found in the currently selected row.
        Works for both historical (historical_items) and current detections (detection_list).
        """
        table = self.historical_items if getattr(self, "reviewing_history", False) else self.detection_list
        if table is None:
            print("No table")
            return

        row = table.currentRow()
        print(row)
        if row < 0:
            print("No rows")
            return

        combo = table.cellWidget(row, 6)
        if combo is None:
            print("No combo")
            return

        # Assign and preserve previous text for change detection
        self.label_combo = combo # 

    def show_historical_gif(self):
        self.gif_active = True
        if self.frame_player.timer.isActive():
            self.frame_player.timer.stop()

        selected = self.historical_items.selectedItems()
        if not selected and self.historical_items.rowCount() == 0:
            self.show_no_detections_message()
            return
        if not selected:
            return
        # Get the selected row index
        row = self.historical_items.currentRow()
        if row < 0:
            return

        experiment = format_experiment_date(self.historical_items.item(row, 0).text(), to_human=False)  # first column
        video_basename = self.historical_items.item(row, 1).text()  # second column
        track_id = self.historical_items.item(row, 2).text()  # third column
        gif_dir = Path(get_results_dir()) / experiment / "tracking_gifs"
        gif_name = f"{video_basename}_{track_id}.gif"
        gif_path = gif_dir / gif_name
        if gif_path.exists():
            self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
            self.update_button_position()
            self.frame_player.set_gif(str(gif_path))
        else:
            alt = gif_dir / f"{Path(video_basename).stem}_{track_id}.gif"
            if alt.exists():
                self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
                self.update_button_position()
                self.frame_player.set_gif(str(alt))
            else:
                self.frame_player.clear()
                self.frame_player.setText(f"GIF not found:\n{gif_name}")

    def render_historical_experiments(self):
        # Render Historical Experiments and add to List
        self.historical_items.setRowCount(0)
        self.historical_items.clearContents()
        self.reviewing_history = True

        if not self.current_experiment:
            self.show_no_detections_message()
            self.toggle_review_buttons(enable=False)
            return

        exp_disp = self.review_dropdown.currentText()
        # Extract only the date part (before the first parenthesis, if present)
        exp_date = exp_disp.split(' (')[0].strip()
        self.current_experiment = format_experiment_date(exp_date, to_human=False)

        experiments_root = get_results_dir()
        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']

        try:
            # newest-first
            exp_dir = Path(experiments_root) / self.current_experiment
            det_dir = exp_dir / "detection_results"
            gif_dir = exp_dir / "tracking_gifs"

            if not (det_dir.exists() and gif_dir.exists()):
                print("Error")

            # each CSV can contain multiple tracks (rows) → iterate rows!
            for csv_name in os.listdir(det_dir):
                csv_path = det_dir / csv_name
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue

                # Create one item per track (row)
                for index, row in df.iterrows():
                    try:
                        video_path_str = str(row.get('video_name', ''))
                        video_basename = Path(video_path_str).name
                        track_id = int(row.get('Track Id'))
                        time_str = str(row.get('Highest Conf Timestamp', ''))
                        conf_longest = float(row.get('Confidence of Longest Length', 0.0))
                        len_high_conf = float(row.get('Highest Confidence Length', 0.0))
                        label = str(row.get('Label', 'Shark'))

                        pending_change = (self.current_experiment, csv_name, track_id) 
                        row_position = self.historical_items.rowCount()
                        self.historical_items.insertRow(row_position)
                        values = [
                            exp_date,
                            video_basename,
                            str(track_id),
                            time_str,
                            f"{conf_longest:.2f}",
                            f"{len_high_conf:.1f}ft",
                            label # if not self.historical_label_changes.get(pending_change) else self.historical_label_changes.get(pending_change)
                        ]

                        for col, value in enumerate(values):
                            if col == 6:
                                # Creates dropdown for label
                                combo = QComboBox()
                                combo.addItems(["Shark", "Kelp", "Dolphin", "Surfer", "Boat", "Bird", "Duplicate", "None", "Other"])
                                combo.setCurrentText(label)
                                combo.previous_text = label
                                # combo.currentIndexChanged.connect(lambda: setattr(self, "label_combo", combo))
                                combo.currentIndexChanged.connect(self.update_label)
                                self.historical_items.setCellWidget(row_position, col, combo)
                            else:
                                cell = QTableWidgetItem(value)
                                # Make cell selectable but not editable (prevents editing on double-click)
                                cell.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                                if col == 4 and conf_longest < 0.65:
                                    cell.setForeground(QColor('red'))
                                self.historical_items.setItem(row_position, col, cell)

                        # Create delete button
                        del_button = QPushButton("")
                        del_button.setIcon(QIcon(resource_path("assets/images/x-lg.svg")))
                        del_button.setStyleSheet("background: transparent; border: none;")
                        del_button.clicked.connect(self.mark_for_deletion)
                        self.historical_items.setCellWidget(row_position, 7, del_button)

                        self.historical_items.item(row_position, 0).setData(
                            Qt.ItemDataRole.UserRole, (self.current_experiment, video_basename, track_id)
                        )
                    except Exception as e:
                        print(f"Error creating historical row item from {csv_path}: {e}")
                

            # configure headers and resizing once
            self.historical_items.setHorizontalHeaderLabels(labels)
            self.historical_items.resizeColumnsToContents()
            self.historical_items.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.historical_items.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.historical_items.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.historical_items.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)

            self.switch_detection_list(show_historical=True)
            self.reviewing_history = True

            detections_present = self.historical_items.rowCount() > 0
            
            # self.toggle_review_buttons(enable=detections_present)
            self.toggle_edit_state(set_state=False)

        except Exception as e:
            print(f"Error while building historical list: {e}")
            self.switch_detection_list(show_historical=False)
            self.reviewing_history = False

            self.switch_detection_list(show_historical=True)
            self.reviewing_history = True

    def setup_review_dropdown(self):
        experiments_root = get_results_dir()
        # newest-first
        self.review_dropdown.clear()
        for experiment in sorted(os.listdir(experiments_root), reverse=True):
            if validate_experiment_date(experiment):
                exp_dir = Path(experiments_root) / experiment
                exp_date = format_experiment_date(experiment, to_human=True)
                exp_disp = exp_date + " " + add_experiment_info(exp_dir)
                self.review_dropdown.addItem(exp_disp)
            else:
                continue
        self.review_dropdown.setCurrentIndex(0)
        self.current_experiment = format_experiment_date(self.review_dropdown.currentText(), to_human=False)
    
    def toggle_dropdown_display(self):
        if self.reviewing_history:  
            self.review_dropdown.show()
        else:
            self.review_dropdown.hide()

    def go_to_home(self):
        if len(self.historical_label_changes) > 0:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Unsaved Label Changes")
            msg_box.setText("You have unsaved label changes. Save changes before returning home?")
            
            save_button = msg_box.addButton("Save and Return", QMessageBox.ButtonRole.YesRole)
            discard_button = msg_box.addButton("Return Without Saving", QMessageBox.ButtonRole.NoRole)
            cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.layout().setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == save_button:
                self._save_historical_label_changes()
            elif clicked_button == discard_button:
                pass
            else:
                return  # cancel pressed            

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
        self.timer.stop()
        self.elapsed_time = 0
        
        # Clear frame player
        self.frame_player.clear_frame()

        # Clear video list and reset buttons
        self.video_list.clear()
        # self.video_list.setHorizontalHeaderLabels(["Video",  ""])
        self.video_list.setRowCount(0)

        self.video_queue = []
        self.current_video_index = 0
        self.processed_videos = 0
        
        # Reset tracking data
        self.tracks = {}
        self.sorted_tracks = []
        self.current_detection_index = 0
        
        # Reset button states
        self.remove_button.setEnabled(True)
        self.remove_all_button.setEnabled(False)
        self.video_list.setEnabled(True)
        self.select_videos_button.setEnabled(True)
        self.drone_select.setEnabled(True)
        self.altitude_input.setEnabled(True)
        self.process_button.setEnabled(False)
        
        # Switch to home widget
        self.stack_widget.setCurrentWidget(self.home_widget)
        self.toggle_banner_buttons(review=False)
        
    def show_confidence_warning(self):
        print(self.current_detection_index)
        print(self.sorted_tracks)
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
        self.video_queue = [self.video_list.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.rowCount())]
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

        reply = QMessageBox.question(
            self,
            "Save Changes",
            "Save changes to experiment results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return
        
        failures = []
        updated_files = set()
        experiments_with_changes = set()

        for (experiment, csv_name, track_id), new_label in list(self.historical_label_changes.items()):
            if new_label == "Delete":
                self.delete_track(experiment, csv_name, track_id)
                del self.historical_label_changes[(experiment, csv_name, track_id)]
                continue
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
                experiments_with_changes.add(Path(get_results_dir()) / experiment)
            except Exception as e:
                failures.append(f"{csv_name} (Track {track_id}): {e}")
        
        if str(self.settings_obj.value("enable_auto_upload").lower()) == "true":
            for exp in list(experiments_with_changes):
                print(f"{len(list(experiments_with_changes))} experiments being uploaded")
                experiment_upload = UploadThread(api_url=self.api_url, experiment_dir=exp)
                experiment_upload.run()
                
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
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

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
                for folder in ['bounding_boxes', 'detection_results', 'false_positives', 'frames', 'masks']:
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
    resized = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._movie = None
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frames = []
        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(100)  # 10 FPS
        self._static_pixmap = None

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

    def set_static_pixmap(self, pixmap: QPixmap):
        """Display a still image. Detach the movie if one is active."""
        if self._movie:
            self._movie.stop()
            self.setMovie(None)
            self._movie = None

        self._static_pixmap = pixmap
        self.update()

    def set_gif(self, path: str):
        self._static_pixmap = None

        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        self._movie = movie
        self.setMovie(self._movie)
        self._movie.start()
        self._movie.finished.connect(lambda: self._movie.start())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        widget_size = self.size()

        # 1) Static Image Mode
        if self._static_pixmap:
            frame = self._static_pixmap
            scaled = frame.size().scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
            x = (widget_size.width() - scaled.width()) // 2
            y = (widget_size.height() - scaled.height()) // 2
            painter.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), frame)
            return

        # 2) Movie Mode
        if self._movie:
            frame = self._movie.currentPixmap()
            if not frame.isNull():
                scaled = frame.size().scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
                x = (widget_size.width() - scaled.width()) // 2
                y = (widget_size.height() - scaled.height()) // 2
                painter.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), frame)
            return

        # 3) Fallback (no movie, no pixmap)
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
        self.update()

    def clear_frame(self):
        self._static_pixmap = None
        if self._movie:
            self._movie.stop()
            self.setMovie(None)
            self._movie = None
        self.update()
    
    def content_rect(self):
        """
        Returns the QRect of the actually displayed pixmap/movie frame
        inside the widget, after aspect-ratio scaling and centering.
        """
        if self._static_pixmap:
            frame_size = self._static_pixmap.size()
        elif self._movie:
            frame = self._movie.currentPixmap()
            if frame.isNull():
                return QRect()
            frame_size = frame.size()
        else:
            return QRect()

        widget_size = self.size()
        scaled = frame_size.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)

        x = (widget_size.width() - scaled.width()) // 2
        y = (widget_size.height() - scaled.height()) // 2

        return QRect(x, y, scaled.width(), scaled.height())

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
