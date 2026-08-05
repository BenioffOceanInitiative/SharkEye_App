import multiprocessing
# In a PyInstaller-frozen app, multiprocessing helper subprocesses (e.g. the
# resource_tracker) re-launch this executable with interpreter flags + `-c "..."`.
# PyInstaller overloads freeze_support() to detect those helpers, run them, and exit.
# It MUST be called before argparse/Qt see the helper's arguments — and placing it
# above the torch/Qt imports also keeps the helper child lightweight.
if __name__ == "__main__":
    multiprocessing.freeze_support()
import sys
import os
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel, QComboBox, 
                             QProgressBar, QStackedWidget, QSizePolicy, QMessageBox, QDialog, QLayout, 
                             QTableWidget, QTableWidgetItem, QDialogButtonBox, QLineEdit, QTreeWidget, 
                             QTreeWidgetItem, QFormLayout, QHeaderView, QCheckBox, QStackedLayout, QColorDialog,
                             QSlider, QMenuBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QObject, QSettings, QSize, QRect, QPoint, QRunnable, QThreadPool, QEventLoop, qInstallMessageHandler, QUrl
from PyQt6.QtGui import QImage, QPixmap, QColor, QIcon, QDoubleValidator, QIntValidator, QMovie, QPainter, QPalette, QDesktopServices  # TODO: remove QPalette — unused (moved to theme.py)
from PyQt6.QtSvg import QSvgRenderer  # TODO: remove — unused (moved to theme.py colored_svg_icon)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6_SwitchControl import SwitchControl
    
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
from log_config import get_logger

logger = get_logger("sharkeye.app")
from help_docs_window import HelpDocsWindow
from frame_line_editor import FrameLineEditorWidget
import signal
import json
import requests
import zipfile
from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtCore import QThread, pyqtSignal
import shutil
import tempfile
import io
import time
import imageio
from PIL import Image
import pandas as pd
import math
from pathlib import Path
from segmentation.segmentation_model import (
    run_prediction,
    calculate_shark_length_from_pixel,
    find_pixel_length,
    draw_mask,
    release_sam_model,
    get_sam_predictor,
)
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


# Frame sampling / detection parsing (shared with the headless processors).
from frame_sampling import (iter_sampled_frames, parse_detections, downscale_for_preview,
                            format_sampling_stats, format_sampling_timeline)
try:
    # Keyframe-scan sampling needs PyAV; keep it optional so the app still runs (on
    # grab-through) if PyAV is unavailable. try_keyframe_sampler itself returns None
    # unless SHARKEYE_KEYFRAME_SAMPLING=1, so importing it changes no default behavior.
    from keyframe_sampling import try_keyframe_sampler
except Exception:  # pragma: no cover - PyAV missing / import failure
    def try_keyframe_sampler(*_args, **_kwargs):
        return None

# Theming: colors, icon tints, and reusable style snippets live in theme.py so styling
# decisions stay in one place and adapt to the OS light/dark palette.
from theme import (
    apply_theme,
    banner_icon,
    banner_surface_style,
    colored_svg_icon,
    colored_svg_icon_fit,
    is_dark_mode,
    theme_icon_color,
    warning_text_color,
    BANNER_BUTTON,
    FLAT_ICON_BUTTON,
)


DEFAULT_DETECTION_LABELS = [
    "Shark", "Kelp", "Dolphin", "Surfer", "Boat", "Bird", "Duplicate", "Glare", "None", "Other"
]


def get_detection_labels(settings_obj):
    value = settings_obj.value("detection_labels")
    if not value:
        return list(DEFAULT_DETECTION_LABELS)
    try:
        labels = json.loads(value)
        if isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels):
            return labels
    except (json.JSONDecodeError, TypeError):
        pass
    return list(DEFAULT_DETECTION_LABELS)


def save_detection_labels(settings_obj, labels):
    settings_obj.setValue("detection_labels", json.dumps(labels))


# Trailing entry in the home screen's drone dropdown that opens the Add New Drone dialog.
ADD_DRONE_ITEM_TEXT = "➕ Add New Drone…"
ADD_DRONE_SENTINEL = "__add_drone__"

# Review-clip playback rates, as multiples of the clips' native 10 fps. Detection clips
# are built from sparsely sampled frames, so even 1x replays far faster than real time;
# 0.5x is the default because reviewers need time to judge each detection.
PLAYBACK_SPEEDS = (0.5, 1.0)
DEFAULT_PLAYBACK_SPEED = 0.5

# Accessibility annotation preview. Drawn at PREVIEW_RENDER_* with the user's literal
# cv2 parameters (so a thickness of 2 really is 2 px, as on an exported frame), then
# downscaled to PREVIEW_DISPLAY_* to fit the settings dialog. Rendering at the true
# 2688x1512 export size instead would cost ~37 ms per keystroke — too slow to feel live.
PREVIEW_RENDER_WIDTH, PREVIEW_RENDER_HEIGHT = 640, 360
PREVIEW_DISPLAY_WIDTH, PREVIEW_DISPLAY_HEIGHT = 360, 203


def add_drone_to_settings(settings_obj, drone_name, width, height, fov_input):
    """Validate and persist one drone/resolution entry into the ``drone_settings`` JSON.

    Shared by the Settings page's "Add New Drone" button and the home screen's
    "Add New Drone…" dropdown entry so the validation rules live in one place.
    Returns an error message on rejection, or ``None`` on success.
    """
    drone_name = (drone_name or "").strip()
    width = (width or "").strip()
    height = (height or "").strip()
    fov_input = (fov_input or "").strip()

    if not drone_name or not width or not height or not fov_input:
        return "All fields must be filled."
    if not width.isdigit() or not height.isdigit():
        return "Width and Height must be positive integers."
    try:
        fov_rad = float(fov_input)
        if fov_rad <= 0:
            raise ValueError
    except ValueError:
        return "FOV must be a positive number (in radians)."

    value = settings_obj.value("drone_settings")
    try:
        drone_settings = json.loads(value) if value else {}
    except (json.JSONDecodeError, TypeError):
        drone_settings = {}

    res_key = f"({width}, {height})"
    drone_settings.setdefault(drone_name, {}).setdefault("Resolution", {})[res_key] = fov_rad
    settings_obj.setValue("drone_settings", json.dumps(drone_settings))
    return None


# Cloud Function that serves version checks + build downloads (same endpoint the
# model download uses). `?request=check_version` compares commits; the default
# request (`?user_os=<os>`) redirects to a signed URL for the latest build.
UPDATE_ENDPOINT = "https://us-central1-sharkeye-329715.cloudfunctions.net/sign-up"


def get_build_info():
    """Return the bundled build identifier (os/commit/committed_at) written by SharkEye.spec.

    Returns None for dev runs or unstamped builds where version.json is absent, so the
    caller can skip the update check (there is nothing to compare against).
    """
    try:
        with open(resource_path("version.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("commit") and data.get("os"):
        return data
    return None


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

class SwitchControl(SwitchControl):
    """
    Child class of SwitchControl that:
    - Removes dragging behavior
    - Allows clicking anywhere (including the circle) to toggle
    - Preserves animations and appearance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The base widget is a QCheckBox custom-painted as a pill switch. On macOS the
        # native style still draws a focus ring around the (invisible) checkbox indicator,
        # which surfaces as a stray square over the switch handle. The switch is toggled by
        # clicking, so suppress that focus ring; harmless/no-op on other platforms.
        self.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)

        #
        # Disable dragging on the circle
        # but allow its click to toggle the parent switch
        #
        self.__circle.mousePressEvent = self._circle_click
        self.__circle.mouseMoveEvent  = lambda e: None    # no dragging
        self.__circle.mouseReleaseEvent = lambda e: None  # no drag logic

    # --------------------------
    # Circle click handler
    # --------------------------
    def _circle_click(self, event):
        # Clicking the circle should toggle exactly like clicking the background
        new_state = not self.isChecked()
        self.start_animation(new_state)
        self.toggled.emit(new_state)
        self.clicked.emit(new_state)
        event.accept()

    # --------------------------
    # Override parent drag logic
    # --------------------------
    def mousePressEvent(self, event):
        # No drag detection → act like a normal checkbox
        event.accept()

    def mouseMoveEvent(self, event):
        # Ignore movement completely
        event.ignore()

    def mouseReleaseEvent(self, event):
        # Always toggle on release
        new_state = not self.isChecked()
        self.start_animation(new_state)
        self.toggled.emit(new_state)
        self.clicked.emit(new_state)
        event.accept()
    
    def reset_position(self, checked=False, animate=False):
        if animate:
            self.start_animation(checked)
        else:
            if checked:
                self.__circle.move(self.width() - 26, 3)
                self.setChecked(True)
            else:
                self.__circle.move(3, 3)
                self.setChecked(False)
        self.update()


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
        self.category_list.addItem("Detection Labels")
        self.category_list.addItem("Cloud Features")
        self.category_list.addItem("Accessibility")
        self.category_list.addItem("Playback Settings")
        self.category_list.setFixedWidth(150)
        self.category_list.currentRowChanged.connect(self.switch_category)
        main_layout.addWidget(self.category_list)

        # Right: stacked settings pages
        self.pages = QStackedWidget()
        self.drone_settings_page = DroneSettingsPage(self.settings_obj, self)
        self.historical_settings_page = HistoricalExperimentsPage()
        self.confidence_settings_page = ConfidencePage(self.settings_obj)
        self.detection_labels_page = DetectionLabelsPage(self.settings_obj)
        self.cloud_feature_page = CloudUploadPage(self.settings_obj)
        self.accessibility_page = AccessibilityPage(self.settings_obj)
        self.playback_settings_page = PlaybackSettingsPage(self.settings_obj)
        self.pages.addWidget(self.drone_settings_page)
        self.pages.addWidget(self.historical_settings_page)
        self.pages.addWidget(self.confidence_settings_page)
        self.pages.addWidget(self.detection_labels_page)
        self.pages.addWidget(self.cloud_feature_page)
        self.pages.addWidget(self.accessibility_page)
        self.pages.addWidget(self.playback_settings_page)

        main_layout.addWidget(self.pages)
        self.setLayout(main_layout)

        self.category_list.setCurrentRow(0)

    def switch_category(self, index):
        self.pages.setCurrentIndex(index)
        if self.pages.currentWidget() in (self.cloud_feature_page, self.historical_settings_page):
            self.pages.currentWidget().populate_experiment_table()

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
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        error = add_drone_to_settings(self.settings_obj, *dialog.get_inputs())
        if error:
            QMessageBox.warning(self, "Invalid Input", error)
            return

        # The helper wrote straight to QSettings; refresh our in-memory copy so the
        # closeEvent save_settings() doesn't write the stale dict back over it.
        self.load_settings()

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
        logger.info(f"[upload] Manual upload requested for {len(checked)} selected experiment(s)")
        for experiment_dir in checked:
            zip_name = f'{Path(experiment_dir).name}.zip'
            logger.info(f"[upload] Zipping experiment '{experiment_dir}' -> {zip_name}")
            try:
                file_count = 0
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
                                    file_count += 1

                zip_size = buffer.tell()
                buffer.seek(0)
                logger.info(f"[upload] {zip_name}: {file_count} file(s), {zip_size / 1024:.1f} KB; "
                      f"POST -> {api_url}")
                files = {'file': (zip_name, buffer, 'application/zip')}
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                logger.info(f"[upload] {zip_name}: SUCCESS (HTTP {response.status_code})")
                upload_status, message = "Upload Finished", "Folder uploaded successfully"
            except requests.RequestException as e:
                logger.error(f"[upload] {zip_name}: FAILED (request error): {e}")
                upload_status, message = "Upload Error", "Failed to Upload folder to cloud storage: {}".format(str(e))
            except Exception as e:
                logger.error(f"[upload] {zip_name}: FAILED (unexpected error): {e}")
                upload_status, message = "Upload Error", "An unexpected error occurred: {}".format(str(e))
            QMessageBox.information(self, upload_status, message)


    def populate_experiment_table(self):
        experiments_root = get_results_dir()
        # newest-first
        
        self.historical_experiments_settings.clearContents()
        self.historical_experiments_settings.setRowCount(0)

        for experiment in sorted(os.listdir(experiments_root), reverse=True):
            if validate_experiment_date(experiment) and validate_experiment_folder(Path(experiments_root) / experiment):
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
        QMessageBox.information(self, "Saved", f"Settings saved")

    def on_reset(self):
        self.settings_obj.setValue("confidence_threshold", "0.40")
        self.settings_obj.setValue("min_frames", "5")
        self.confidence_input.setText(self.settings_obj.value("confidence_threshold"))
        self.min_frames_input.setText(self.settings_obj.value("min_frames"))
        QMessageBox.information(self, "Reset", "Confidence threshold reset to 0.40 and Minimum Frames reset to 5")


class PlaybackSettingsPage(QWidget):
    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj
        self.playback_min_frames = self.settings_obj.value("playback_min_frames", "5")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        description = QLabel(
            "Minimum frame count for playback: animations with fewer frames than this value "
            "display a single center frame instead of playing, to avoid jittery playback.\n\n"
            "Default speed is the playback rate the Review screen starts at. Detection clips "
            "are built from sparsely sampled frames, so they replay much faster than real "
            "time — 0.5x gives you longer to judge each detection."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form_layout = QGridLayout()
        form_layout.setVerticalSpacing(4)
        form_layout.setHorizontalSpacing(6)

        self.playback_min_frames_input = QLineEdit(str(self.playback_min_frames))
        self.playback_min_frames_input.setValidator(QIntValidator(1, 10000))

        self.playback_speed_combo = QComboBox()
        for multiplier in PLAYBACK_SPEEDS:
            self.playback_speed_combo.addItem(f"{multiplier:g}x", multiplier)
        self._select_speed(self.settings_obj.value("playback_speed", str(DEFAULT_PLAYBACK_SPEED)))

        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset to Default")
        save_btn.clicked.connect(self.on_save)
        reset_btn.clicked.connect(self.on_reset)

        form_layout.addWidget(QLabel("Minimum Frames:"), 0, 0)
        form_layout.addWidget(self.playback_min_frames_input, 0, 1)
        form_layout.addWidget(QLabel("Default Speed:"), 1, 0)
        form_layout.addWidget(self.playback_speed_combo, 1, 1)
        form_layout.addWidget(reset_btn, 2, 0)
        form_layout.addWidget(save_btn, 2, 1)
        layout.addLayout(form_layout)

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def _select_speed(self, value):
        try:
            speed = float(value)
        except (TypeError, ValueError):
            speed = DEFAULT_PLAYBACK_SPEED
        index = self.playback_speed_combo.findData(speed)
        self.playback_speed_combo.setCurrentIndex(index if index >= 0 else 0)

    def on_save(self):
        text = self.playback_min_frames_input.text().strip()
        try:
            min_val = int(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter an integer for Minimum Frames.")
            return
        if min_val < 1:
            QMessageBox.warning(self, "Invalid Range", "Minimum Frames must be >= 1.")
            return

        self.settings_obj.setValue("playback_min_frames", str(min_val))
        self.settings_obj.setValue("playback_speed", str(self.playback_speed_combo.currentData()))
        QMessageBox.information(self, "Saved", "Playback settings saved")

    def on_reset(self):
        self.settings_obj.setValue("playback_min_frames", "5")
        self.settings_obj.setValue("playback_speed", str(DEFAULT_PLAYBACK_SPEED))
        self.playback_min_frames_input.setText(self.settings_obj.value("playback_min_frames"))
        self._select_speed(DEFAULT_PLAYBACK_SPEED)
        QMessageBox.information(
            self, "Reset", f"Playback settings reset (5 frames, {DEFAULT_PLAYBACK_SPEED:g}x)")


class DetectionLabelsPage(QWidget):
    labels_updated = pyqtSignal()

    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QLabel("Available Labels"))

        self.label_table = QTableWidget()
        self.label_table.setShowGrid(False)
        self.label_table.verticalHeader().setVisible(False)
        self.label_table.horizontalHeader().setVisible(False)
        self.label_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.label_table.setColumnCount(2)
        self.label_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label_table.setMinimumHeight(200)
        layout.addWidget(self.label_table, 1)

        add_row = QHBoxLayout()
        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("New label name")
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_label)
        self.new_label_input.returnPressed.connect(self.add_label)
        add_row.addWidget(self.new_label_input)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_to_default)
        layout.addWidget(reset_btn)

        self.refresh_list()

    def refresh_list(self):
        self.label_table.setRowCount(0)
        for label in get_detection_labels(self.settings_obj):
            self._append_label_row(label)
        self.label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

    def _append_label_row(self, label):
        row_position = self.label_table.rowCount()
        self.label_table.insertRow(row_position)

        item = QTableWidgetItem(label)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.label_table.setItem(row_position, 0, item)

        delete_btn = QPushButton("")
        # x-lg.svg is fill="currentColor", which QSvgRenderer resolves to black — tint it
        # so the glyph stays visible in dark mode.
        delete_btn.setIcon(colored_svg_icon(resource_path("assets/images/x-lg.svg"), theme_icon_color()))
        delete_btn.setStyleSheet(FLAT_ICON_BUTTON)
        delete_btn.clicked.connect(self.delete_label_row)
        self.label_table.setCellWidget(row_position, 1, delete_btn)

    def delete_label_row(self):
        button = self.sender()
        if not button:
            return
        index = self.label_table.indexAt(button.pos())
        row = index.row()
        if row < 0:
            return

        labels = get_detection_labels(self.settings_obj)
        if len(labels) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "At least one label must remain.")
            return

        item = self.label_table.item(row, 0)
        if item is None:
            return
        label = item.text()
        if label in labels:
            labels.remove(label)
            save_detection_labels(self.settings_obj, labels)
        self.label_table.removeRow(row)
        self.labels_updated.emit()

    def add_label(self):
        name = self.new_label_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Input", "Enter a label name.")
            return
        labels = get_detection_labels(self.settings_obj)
        if name in labels:
            QMessageBox.warning(self, "Duplicate", f'"{name}" is already in the list.')
            return
        labels.append(name)
        save_detection_labels(self.settings_obj, labels)
        self.new_label_input.clear()
        self._append_label_row(name)
        self.label_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.labels_updated.emit()

    def reset_to_default(self):
        reply = QMessageBox.question(
            self,
            "Reset Detection Labels",
            "Reset all detection labels to the defaults? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        save_detection_labels(self.settings_obj, list(DEFAULT_DETECTION_LABELS))
        self.refresh_list()
        self.labels_updated.emit()
        QMessageBox.information(self, "Reset", "Detection labels reset to defaults.")


class AccessibilityPage(QWidget):
    """Page containing accessibility settings for bounding box and text display"""
    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj
        
        # Load current settings or use defaults
        default_color = (255, 96, 31)  # Neon orange in RGB
        color_str = self.settings_obj.value("annotation_color", f"{default_color[0]},{default_color[1]},{default_color[2]}")
        color_parts = color_str.split(",")
        self.annotation_color = tuple(int(c.strip()) for c in color_parts) if len(color_parts) == 3 else default_color
        
        self.box_thickness = int(self.settings_obj.value("box_thickness", "2"))
        self.text_thickness = int(self.settings_obj.value("text_thickness", "2"))
        self.text_scale = float(self.settings_obj.value("text_scale", "2.0"))

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(16)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        form_layout = QGridLayout()
        form_layout.setVerticalSpacing(10)
        form_layout.setHorizontalSpacing(6)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Color picker for annotation color
        color_label = QLabel("Annotation Color (RGB):")
        self.color_button = QPushButton()
        self.color_button.setFixedSize(80, 30)
        self.update_color_button()
        self.color_button.clicked.connect(self.pick_color)
        form_layout.addWidget(color_label, 0, 0)
        form_layout.addWidget(self.color_button, 0, 1)

        # Box thickness
        box_thickness_label = QLabel("Box Thickness:")
        self.box_thickness_input = QLineEdit(str(self.box_thickness))
        self.box_thickness_input.setValidator(QIntValidator(1, 20))
        form_layout.addWidget(box_thickness_label, 1, 0)
        form_layout.addWidget(self.box_thickness_input, 1, 1)

        # Text thickness
        text_thickness_label = QLabel("Text Thickness:")
        self.text_thickness_input = QLineEdit(str(self.text_thickness))
        self.text_thickness_input.setValidator(QIntValidator(1, 20))
        form_layout.addWidget(text_thickness_label, 2, 0)
        form_layout.addWidget(self.text_thickness_input, 2, 1)

        # Text scale
        text_scale_label = QLabel("Text Scale:")
        self.text_scale_input = QLineEdit(str(self.text_scale))
        self.text_scale_input.setValidator(QDoubleValidator(0.1, 10.0, 1))
        form_layout.addWidget(text_scale_label, 3, 0)
        form_layout.addWidget(self.text_scale_input, 3, 1)

        # Buttons
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset to Default")

        save_btn.clicked.connect(self.on_save)
        reset_btn.clicked.connect(self.on_reset)

        form_layout.addWidget(reset_btn, 4, 0)
        form_layout.addWidget(save_btn, 4, 1)
        outer_layout.addLayout(form_layout)

        # Live preview so the numeric fields stop being a guessing game — it is drawn
        # with the same cv2 calls the clip exporter uses, so what you see here is what
        # lands on the exported frames.
        preview_column = QVBoxLayout()
        preview_column.setSpacing(4)
        preview_column.setAlignment(Qt.AlignmentFlag.AlignTop)
        preview_caption = QLabel("Preview")
        preview_column.addWidget(preview_caption)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(PREVIEW_DISPLAY_WIDTH, PREVIEW_DISPLAY_HEIGHT)
        preview_column.addWidget(self.preview_label)
        outer_layout.addLayout(preview_column)

        # Preview tracks the *unsaved* field values, so the effect is visible while typing.
        self.box_thickness_input.textChanged.connect(self.update_preview)
        self.text_thickness_input.textChanged.connect(self.update_preview)
        self.text_scale_input.textChanged.connect(self.update_preview)
        self.update_preview()

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def _current_preview_values(self):
        """Read the live field values, falling back to the last saved ones while a field
        is mid-edit (empty or out of range)."""
        def as_int(text, fallback, low, high):
            try:
                value = int(text.strip())
            except (AttributeError, ValueError):
                return fallback
            return value if low <= value <= high else fallback

        try:
            scale = float(self.text_scale_input.text().strip())
            if not 0.1 <= scale <= 10.0:
                scale = self.text_scale
        except (AttributeError, ValueError):
            scale = self.text_scale

        return (as_int(self.box_thickness_input.text(), self.box_thickness, 1, 20),
                as_int(self.text_thickness_input.text(), self.text_thickness, 1, 20),
                scale)

    def update_preview(self):
        """Redraw the annotation sample with the currently entered values."""
        box_thickness, text_thickness, text_scale = self._current_preview_values()
        image = render_annotation_preview(
            self.annotation_color, box_thickness, text_thickness, text_scale)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        q_image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_image.copy()))

    def update_color_button(self):
        """Update the color button to show the current color"""
        color = QColor(self.annotation_color[0], self.annotation_color[1], self.annotation_color[2])
        self.color_button.setStyleSheet(f"background-color: rgb({self.annotation_color[0]}, {self.annotation_color[1]}, {self.annotation_color[2]});")

    def pick_color(self):
        """Open color picker dialog"""
        current_color = QColor(self.annotation_color[0], self.annotation_color[1], self.annotation_color[2])
        color = QColorDialog.getColor(current_color, self, "Select Annotation Color")
        if color.isValid():
            self.annotation_color = (color.red(), color.green(), color.blue())
            self.update_color_button()
            self.update_preview()

    def on_save(self):
        """Save accessibility settings"""
        # Validate box thickness
        try:
            box_thick = int(self.box_thickness_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter an integer for Box Thickness.")
            return
        if box_thick < 1 or box_thick > 20:
            QMessageBox.warning(self, "Invalid Range", "Box Thickness must be between 1 and 20.")
            return

        # Validate text thickness
        try:
            text_thick = int(self.text_thickness_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter an integer for Text Thickness.")
            return
        if text_thick < 1 or text_thick > 20:
            QMessageBox.warning(self, "Invalid Range", "Text Thickness must be between 1 and 20.")
            return

        # Validate text scale
        try:
            text_scale_val = float(self.text_scale_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a number for Text Scale.")
            return
        if text_scale_val < 0.1 or text_scale_val > 10.0:
            QMessageBox.warning(self, "Invalid Range", "Text Scale must be between 0.1 and 10.0.")
            return

        # Save settings
        color_str = f"{self.annotation_color[0]},{self.annotation_color[1]},{self.annotation_color[2]}"
        self.settings_obj.setValue("annotation_color", color_str)
        self.settings_obj.setValue("box_thickness", str(box_thick))
        self.settings_obj.setValue("text_thickness", str(text_thick))
        self.settings_obj.setValue("text_scale", str(text_scale_val))
        
        self.box_thickness = box_thick
        self.text_thickness = text_thick
        self.text_scale = text_scale_val
        
        QMessageBox.information(self, "Saved", "Accessibility settings saved successfully.")

    def on_reset(self):
        """Reset to default values"""
        default_color = (255, 96, 31)  # Neon orange
        self.annotation_color = default_color
        self.box_thickness = 2
        self.text_thickness = 2
        self.text_scale = 2.0
        
        self.update_color_button()
        self.box_thickness_input.setText("2")
        self.text_thickness_input.setText("2")
        self.text_scale_input.setText("2.0")
        self.update_preview()

        QMessageBox.information(self, "Reset", "Accessibility settings reset to defaults.")

class CloudUploadPage(HistoricalExperimentsPage):
    """ Page containing settings related to uploading experiments to Google Cloud Bucket"""
    def __init__(self, settings_obj, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj
        # Convert setting string to a boolean
        enable_auto_upload_bool = str(self.settings_obj.value("enable_auto_upload")).lower() == "true"

        # --- Layout setup ---
        layout = QVBoxLayout(self)

        self.checked = set()
        
        # --- Cloud Info Disclaimer ---
        cloud_info_disclaimer = QLabel("<i>Uploading via cloud will share selected experiments with the SharkEye development team</i>")
        layout.addWidget(cloud_info_disclaimer)

        # --- Auto-upload checkbox ---
        self.auto_upload_checkbox = QCheckBox("Enable automatic Cloud upload when saving")
        self.auto_upload_checkbox.setChecked(enable_auto_upload_bool)
        self.auto_upload_checkbox.clicked.connect(self._on_auto_upload_clicked)

        # --- Update check checkbox ---
        # Stored as "ignore_update"; the checkbox is the positive phrasing (checked = check).
        check_updates_bool = str(self.settings_obj.value("ignore_update")).lower() != "true"
        self.check_updates_checkbox = QCheckBox("Check for app updates on startup")
        self.check_updates_checkbox.setChecked(check_updates_bool)
        self.check_updates_checkbox.clicked.connect(self._on_check_updates_clicked)

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
        experiment_table.addWidget(self.check_updates_checkbox)

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
        layout.addLayout(experiment_table, 1)
        experiment_table.setStretch(1, 1)

    def _on_auto_upload_clicked(self, checked):
        if not checked:
            logger.info("[upload] Auto-upload disabled by user")
            self.settings_obj.setValue("enable_auto_upload", "false")
            return
        reply = QMessageBox.question(
            self,
            "Enable Automatic Cloud Upload",
            "When enabled, experiments are automatically shared with the development team after you "
            "save label changes in review mode.\n\nEnable automatic cloud upload when saving?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            logger.info("[upload] Auto-upload enable declined at confirmation dialog")
            self.auto_upload_checkbox.blockSignals(True)
            self.auto_upload_checkbox.setChecked(False)
            self.auto_upload_checkbox.blockSignals(False)
            return
        logger.info("[upload] Auto-upload enabled by user")
        self.settings_obj.setValue("enable_auto_upload", "true")

    def _on_check_updates_clicked(self, checked):
        # Checkbox is positive ("check on startup"); the stored flag is the inverse.
        self.settings_obj.setValue("ignore_update", "false" if checked else "true")

class CustomTracker:
    def __init__(self, distance_threshold=250):
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        
        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        self.min_frames = int(self.settings_obj.value("min_frames", "5"))
        self.confidence_threshold = float(self.settings_obj.value("confidence_threshold", "0.40"))
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

            # Snapshot the key order once; cost-matrix row indices map to it. (Previously
            # rebuilt list(self.tracks.keys()) for every matched pair — O(n^2) per frame.)
            track_ids = list(self.tracks.keys())
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, detection_idx] < self.distance_threshold:
                    track_id = track_ids[track_idx]
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
            # Each cap.retrieve() returns a fresh buffer that is never mutated in place
            # (all consumers copy before drawing), so store references rather than paying
            # for a full-frame copy per detection.
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
        # frame_with_box = frame.copy()
        # cv2.rectangle(frame_with_box, 
        #              (int(x - w/2), int(y - h/2)), 
        #              (int(x + w/2), int(y + h/2)), 
        #              (0, 255, 0), 2)
        # track['track_frames'].append(frame_with_box)
        
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

        if len(track['positions']) > 1:
            prev_pos = np.array(track['positions'][-2][:2])
            curr_pos = np.array([x, y])
            track['velocity'] = curr_pos - prev_pos

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
            if not self.is_significant_track(track):
                continue

            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])

            longest_frame = track['longest_frame']
            longest_confidence = track['longest_conf']
            longest_length = track['longest_length']

            if longest_frame is None:
                continue

            x, y, w, h = track['positions'][track['confidences'].index(longest_confidence)]

            # SAM is the expensive post-processing step and it runs once per track. Only
            # run it on significant tracks (same criteria as _count_significant_tracks);
            # sub-threshold tracks are almost always false positives, keep their
            # bbox-estimated length, and get no mask. Significant tracks are segmented
            # exactly as before, so their reported lengths are unchanged.
            is_significant = (num_frames >= self.min_frames
                              and avg_confidence > self.confidence_threshold)
            if is_significant:
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

            filename = f"{Path(video_path).name}_{track_id}.jpg"

            # Save original frame + bounding-box frame for every track (cheap, keeps the
            # review display populated regardless of segmentation).
            cv2.imwrite(os.path.join(output_dir, 'frames', filename), longest_frame)

            boxed_frame = longest_frame.copy()
            annotation_color, box_thickness, text_thickness, text_scale = get_annotation_settings(self.settings_obj)
            annotation_color_bgr = (annotation_color[2], annotation_color[1], annotation_color[0])

            cv2.rectangle(boxed_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), annotation_color_bgr, box_thickness)
            label = f"ID: {track_id}, Conf: {longest_confidence:.2f}, Length: {length_str}"
            cv2.putText(boxed_frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, annotation_color_bgr, text_thickness)
            bounding_box_path = os.path.join(output_dir, 'bounding_boxes', filename)
            cv2.imwrite(bounding_box_path, boxed_frame)

            # Mask image only exists for segmented (significant) tracks.
            if is_significant:
                mask_path = os.path.join(output_dir, 'masks', filename)
                cv2.imwrite(mask_path, mask_overlay)

            # Update the track with the path to the bounding box image
            track['image_path'] = bounding_box_path

            images_saved += 1

        logger.info(f"[segmentation] saved {images_saved} track image(s)")

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

    def is_significant_track(self, track):
        """Return True if a track meets the confidence and minimum-frame settings."""
        return (len(track['positions']) >= self.min_frames
                and np.mean(track['confidences']) > self.confidence_threshold)

    def get_significant_tracks(self):
        """Return only tracks that meet the confidence and minimum-frame settings."""
        return {track_id: track for track_id, track in self.tracks.items()
                if self.is_significant_track(track)}

    def _count_significant_tracks(self):
        """Count tracks that meet the criteria for being a significant detection"""
        return sum(1 for track in self.tracks.values() if self.is_significant_track(track))

def get_annotation_settings(settings_obj):
    """Get annotation color, thickness, and scale from settings"""
    default_color = (255, 96, 31)  # Neon orange in RGB
    color_str = settings_obj.value("annotation_color", f"{default_color[0]},{default_color[1]},{default_color[2]}")
    color_parts = color_str.split(",")
    annotation_color = tuple(int(c.strip()) for c in color_parts) if len(color_parts) == 3 else default_color
    
    box_thickness = int(settings_obj.value("box_thickness", "2"))
    text_thickness = int(settings_obj.value("text_thickness", "2"))
    text_scale = float(settings_obj.value("text_scale", "2.0"))
    
    return annotation_color, box_thickness, text_thickness, text_scale


def render_annotation_preview(annotation_color, box_thickness, text_thickness, text_scale):
    """Draw a sample detection box for the Accessibility settings preview.

    Deliberately uses the same cv2.rectangle / cv2.putText calls (and the same RGB->BGR
    flip) as encode_track_clips, so the preview is a faithful rendering of what will be
    burned into the exported clips rather than an approximation.
    """
    # Muted ocean backdrop so the annotation color is judged against something
    # representative rather than flat white.
    gradient = np.linspace(0, 1, PREVIEW_RENDER_HEIGHT, dtype=np.float32).reshape(-1, 1)
    base = np.array([110, 84, 46], dtype=np.float32)
    image = np.repeat((base + gradient * np.array([28, 22, 16], dtype=np.float32))[:, None, :],
                      PREVIEW_RENDER_WIDTH, axis=1).astype(np.uint8)

    annotation_color_bgr = (annotation_color[2], annotation_color[1], annotation_color[0])
    # Box sits low-left so a large text scale has room to render before it clips —
    # clipping here is a genuine signal that the label is oversized.
    x1, y1, x2, y2 = 40, 150, 380, 290
    cv2.rectangle(image, (x1, y1), (x2, y2), annotation_color_bgr, box_thickness)
    cv2.putText(image, "Shark: 0.87", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, annotation_color_bgr, text_thickness)
    return cv2.resize(image, (PREVIEW_DISPLAY_WIDTH, PREVIEW_DISPLAY_HEIGHT),
                      interpolation=cv2.INTER_AREA)


def encode_track_clips(payload, output_dir, video_name, annotation_color,
                       box_thickness, text_thickness, text_scale, fps=10):
    """Encode per-track detection clips as MP4 from a self-contained payload.

    MP4 encoding via cv2.VideoWriter is C-level and releases the GIL, so it's far
    faster than the old per-frame PIL palette quantization and doesn't starve the
    concurrent inference thread. `payload` maps track key ->
    {'frames', 'positions', 'lengths', 'confidences'}; the payload owns its own frame
    buffers (no shared state with the UI's track dicts). Output filenames match what
    the review player looks for: "<video_name>_<key>.mp4".
    """
    clips_dir = os.path.join(output_dir, "tracking_gifs")
    os.makedirs(clips_dir, exist_ok=True)

    # cv2 works in BGR, so no RGB conversion is needed before writing.
    annotation_color_bgr = (annotation_color[2], annotation_color[1], annotation_color[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for key, track in payload.items():
        positions = track.get('positions') or []
        frames = track.get('frames') or []
        confidences = track.get('confidences') or []

        writer = None
        frame_size = None
        clip_path = os.path.join(clips_dir, f"{video_name}_{key}.mp4")
        try:
            for frame_idx, (pos, frame) in enumerate(zip(positions, frames)):
                if frame is None:
                    continue
                x, y, w, h = pos
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box,
                              (int(x - w/2), int(y - h/2)),
                              (int(x + w/2), int(y + h/2)),
                              annotation_color_bgr, box_thickness)

                conf = confidences[frame_idx] if frame_idx < len(confidences) else 0.0
                label = f"Shark: {conf:.2f}"
                cv2.putText(frame_with_box, label, (int(x - w/2), int(y - h/2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, annotation_color_bgr, text_thickness)

                if writer is None:
                    h_px, w_px = frame_with_box.shape[:2]
                    frame_size = (w_px, h_px)
                    writer = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
                    if not writer.isOpened():
                        logger.error(f"Could not open video writer for {clip_path}; skipping track {key}")
                        writer = None
                        break

                # Guard against a stray frame of a different size
                if (frame_with_box.shape[1], frame_with_box.shape[0]) != frame_size:
                    frame_with_box = cv2.resize(frame_with_box, frame_size)

                writer.write(frame_with_box)

            if writer is not None:
                writer.release()
                logger.info(f"Saved clip: {clip_path}")
        finally:
            # Free the frame buffers as we go
            track['frames'] = None


def export_training_frames_locally(payload, video_stem, annotation_format="yolo"):
    """Bundle per-track frames + annotations into a training zip under the results dir.

    Extracted from the old `VideoProcessingWorker.upload_frames_for_training` (local
    export path) so it can run off the worker/UI thread. Operates on the self-contained
    post-processing payload (track key -> {'frames', 'positions', ...}), so it shares no
    mutable state with the tracks handed to the UI.
    """
    fmt = (annotation_format or "yolo").strip().lower()
    if fmt not in ("coco", "yolo"):
        logger.warning(f"Unsupported annotation_format {annotation_format!r}; skipping training export")
        return

    category_names = [
        "great white shark", "kelp", "human", "surfer", "dolphin",
        "bat ray", "bird", "boat", "seal", "kayaker",
    ]
    num_classes = len(category_names)

    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "",
                 "url": "", "version": "", "year": ""},
        "categories": [{"id": i + 1, "name": name, "supercategory": ""}
                       for i, name in enumerate(category_names)],
        "images": [],
        "annotations": [],
    }
    image_id = 1
    annotation_id = 1
    buffer = io.BytesIO()
    yolo_data_dir = "obj_train_data"
    yolo_train_paths = []

    try:
        with zipfile.ZipFile(buffer, "w") as zipf:
            for track_id, track in payload.items():
                positions = track.get("positions")
                frames = track.get("frames")
                if positions is None or frames is None:
                    continue

                for frame_idx, (pos, frame) in enumerate(zip(positions, frames)):
                    x, y, w, h = pos
                    if frame is None:
                        continue
                    try:
                        height, width = frame.shape[:2]
                    except Exception:
                        continue

                    x_min = max(0, int(x - w / 2))
                    y_min = max(0, int(y - h / 2))
                    box_w = int(w)
                    box_h = int(h)
                    if x_min >= width or y_min >= height:
                        continue
                    box_w = min(box_w, width - x_min)
                    box_h = min(box_h, height - y_min)
                    if box_w <= 0 or box_h <= 0:
                        continue

                    image_basename = f"{video_stem}_track{track_id}_frame{frame_idx:04d}"
                    success, encoded = cv2.imencode(".jpg", frame)
                    if not success:
                        continue

                    if fmt == "yolo":
                        image_path_in_zip = os.path.join(yolo_data_dir, image_basename + ".jpg")
                        label_path_in_zip = os.path.join(yolo_data_dir, image_basename + ".txt")
                        zipf.writestr(image_path_in_zip, encoded.tobytes())
                        yolo_train_paths.append(image_path_in_zip)
                        cx_norm = x / width
                        cy_norm = y / height
                        rw = w / width
                        rh = h / height
                        zipf.writestr(label_path_in_zip,
                                      f"0 {cx_norm:.6f} {cy_norm:.6f} {rw:.6f} {rh:.6f}\n")
                    else:
                        image_filename = image_basename + ".jpg"
                        zipf.writestr(os.path.join("images", image_filename), encoded.tobytes())
                        coco["images"].append({
                            "id": image_id, "width": int(width), "height": int(height),
                            "file_name": image_filename, "license": 0, "flickr_url": "",
                            "coco_url": "", "date_captured": 0,
                        })
                        coco["annotations"].append({
                            "id": annotation_id, "image_id": image_id, "category_id": 1,
                            "segmentation": [], "area": float(box_w * box_h),
                            "bbox": [float(x_min), float(y_min), float(box_w), float(box_h)],
                            "iscrowd": 0,
                            "attributes": {"occluded": False, "rotation": 0.0,
                                           "track_id": track_id, "keyframe": True},
                        })
                        image_id += 1
                        annotation_id += 1

            if fmt == "coco":
                zipf.writestr("instances_default.json", json.dumps(coco))
            else:
                zipf.writestr("train.txt", "\n".join(yolo_train_paths) + ("\n" if yolo_train_paths else ""))
                zipf.writestr("obj.names", "\n".join(category_names) + "\n")
                zipf.writestr("obj.data", f"classes = {num_classes}\nnames = obj.names\ntrain = train.txt\n")

        export_dir = get_results_dir()
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"{video_stem}_training_frames.zip")
        with open(export_path, "wb") as f:
            f.write(buffer.getvalue())
        logger.info(f"Training frames zip saved to {export_path}")
    except Exception as e:
        logger.error(f"Training frame export failed: {e}")


class PostProcessJob(QRunnable):
    """Runs the per-video CPU/IO post-processing (training-frame export + MP4 clip
    encoding) on a background thread pool so the next video's inference isn't blocked."""

    def __init__(self, payload, output_dir, video_name, annotation_color,
                 box_thickness, text_thickness, text_scale):
        super().__init__()
        self.payload = payload
        self.output_dir = output_dir
        self.video_name = video_name
        self.annotation_color = annotation_color
        self.box_thickness = box_thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale

    def run(self):
        video_stem = Path(self.video_name).stem
        t_export = t_clip = -1.0
        try:
            t0 = time.perf_counter()
            export_training_frames_locally(self.payload, video_stem, annotation_format="yolo")
            t_export = time.perf_counter() - t0
        except Exception as e:
            logger.error(f"Async training-frame export failed: {e}")
        try:
            t0 = time.perf_counter()
            encode_track_clips(self.payload, self.output_dir, self.video_name,
                               self.annotation_color, self.box_thickness,
                               self.text_thickness, self.text_scale)
            t_clip = time.perf_counter() - t0
        except Exception as e:
            logger.error(f"Async clip encoding failed: {e}")
        logger.info(f"[timing] {self.video_name}: (background) export={t_export:.2f}s clip={t_clip:.2f}s")


class VideoProcessingWorker(QObject):
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(dict, str)
    frame_processed = pyqtSignal(np.ndarray)  # Add a boolean flag for detection
    progress_status_changed = pyqtSignal(str)  # current process summary for MainWindow.progress_status
    postproc_ready = pyqtSignal(dict, str, str)  # (payload, output_dir, video_name) for async export + clip encoding
    video_timing_ready = pyqtSignal(dict)  # per-video phase timing for the batch summary

    def __init__(self, video_path, model, output_dir, drone_type, altitude, flight_location):
        super().__init__()
        # Read settings 
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.drone_type = drone_type
        self.altitude = altitude
        self.flight_location = flight_location
        self.detection_threshold = float(self.settings_obj.value("confidence_threshold", "0.40"))
        self.drone_settings = json.loads(self.settings_obj.value("drone_settings"))
        
    def run(self):
        self.progress_status_changed.emit("Running Inference")
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        frames_sampled = 0
        total_detections = 0
        infer_start = time.perf_counter()
        # Split the loop wall-time into its real phases. On 4K/5.3K H.265 footage the
        # dominant cost is *video decode* (the sampler grab()s through every skipped
        # frame to advance), not YOLO inference — decode is ~30fps-bound while inference
        # is ~40ms/frame. Timing them separately keeps the [timing] line honest instead
        # of attributing the whole wall to "inference".
        decode_time = 0.0   # time inside the sampler (grab/retrieve = H.265 decode)
        model_time = 0.0     # time inside YOLO inference on sampled frames

        # Live preview is a courtesy view, not the output. Cap empty-frame updates at
        # ~20 fps so we don't color-convert, copy across the thread boundary, and
        # re-scale a pixmap on the UI thread faster than a human can see. Detection
        # frames always emit so no detection flashes by unseen.
        preview_interval = 1.0 / 20.0
        last_preview = 0.0

        # Sequential forward sampling. Default: grab-through with an adaptive stride.
        # When SHARKEYE_KEYFRAME_SAMPLING=1 (and the file decodes cleanly), swap in the
        # keyframe-scan sampler: decode only keyframes over empty water, go dense around
        # detections — ~5x faster on long-GOP HEVC (decode-bound) footage with equal
        # recall. try_keyframe_sampler returns None on any problem, so we transparently
        # fall back to grab-through and never regress. In keyframe mode cap stays open
        # for metadata only; timestamps come from the frame index since the capture is
        # no longer advanced by reads.
        sampler = try_keyframe_sampler(self.video_path, logger)
        use_keyframe = sampler is not None
        if use_keyframe:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
        else:
            sampler = iter_sampled_frames(cap)
        had_detection = None
        sampling_stats = {}
        try:
            while True:
                _t_decode = time.perf_counter()
                frame_num, frame = sampler.send(had_detection)
                decode_time += time.perf_counter() - _t_decode

                if QThread.currentThread().isInterruptionRequested():
                    logger.warning("Processing interrupted")
                    break

                _t_model = time.perf_counter()
                results = self.model(frame, classes=[0], verbose=False)
                model_time += time.perf_counter() - _t_model
                frames_sampled += 1

                detections = parse_detections(results, self.detection_threshold)
                had_detection = bool(detections)

                if had_detection:
                    timestamp = (frame_num / fps * 1000.0) if use_keyframe else cap.get(cv2.CAP_PROP_POS_MSEC)
                    total_detections += len(detections)
                    custom_tracker.update(detections, frame, timestamp)

                now = time.perf_counter()
                if had_detection or (now - last_preview) >= preview_interval:
                    # Draw boxes + downscale only when we actually emit: the preview
                    # widget is far smaller than a source frame, so shipping full 4K wastes CPU.
                    preview = self.draw_bounding_boxes(frame, detections) if had_detection else frame
                    preview = downscale_for_preview(preview)
                    self.frame_processed.emit(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    last_preview = now

                self.progress_update.emit(int((frame_num + 1) / total_frames * 100))
        except StopIteration as stop:
            sampling_stats = stop.value or {}

        infer_time = time.perf_counter() - infer_start
        cap.release()

        if not QThread.currentThread().isInterruptionRequested():
            significant_tracks = custom_tracker.get_significant_tracks()
            filtered_count = len(custom_tracker.tracks) - len(significant_tracks)
            if filtered_count:
                logger.info(f"Filtered out {filtered_count} track(s) below confidence/minimum-frame thresholds")

            # Only save results if not interrupted
            self.progress_status_changed.emit("Running Segmentation")
            seg_start = time.perf_counter()
            custom_tracker.save_best_frames(self.output_dir, self.video_path)
            seg_time = time.perf_counter() - seg_start

            self.progress_status_changed.emit("Saving detection results")
            csv_start = time.perf_counter()
            self.save_detections_csv(
                significant_tracks,
                os.path.join(self.output_dir, 'detection_results'),
                custom_tracker,
            )
            csv_time = time.perf_counter() - csv_start

            # Defer the CPU/IO post-processing (training-frame export + GIF encoding) to
            # a background pool so the next video's inference can start immediately. Move
            # the frame buffers out of the tracks first so the UI's copy and the
            # background job never share mutable state.
            self.progress_status_changed.emit("Finalizing")
            payload = self._extract_postproc_payload(significant_tracks)
            self.postproc_ready.emit(payload, self.output_dir, Path(self.video_path).name)

            # Per-phase timing breakdown (export + clip are timed in the background job).
            # `infer_time` is the whole sampling-loop wall; break it into decode vs. YOLO
            # so the real bottleneck is visible. `other` is preview/tracking/progress
            # overhead (the remainder of the loop).
            track_count = len(significant_tracks)
            other_time = max(0.0, infer_time - decode_time - model_time)
            decode_pct = (decode_time / infer_time * 100) if infer_time > 0 else 0.0
            ms_per_infer = (model_time / frames_sampled * 1000) if frames_sampled else 0.0
            logger.info(f"[timing] {Path(self.video_path).name}: "
                  f"loop={infer_time:.1f}s [decode={decode_time:.1f}s ({decode_pct:.0f}%), "
                  f"yolo={model_time:.1f}s ({frames_sampled} frames, {ms_per_infer:.0f}ms/f), "
                  f"other={other_time:.1f}s] {total_detections} dets | "
                  f"segmentation={seg_time:.1f}s csv={csv_time:.2f}s "
                  f"tracks={track_count} (export+clip deferred to background)")

            # Per-track discovery line: one row per significant track so a run's actual
            # findings (when, how confident, how long, how many detections) are legible
            # from the log instead of only a bare count.
            for tid, tr in significant_tracks.items():
                confs = tr.get('confidences') or [0.0]
                logger.info(f"[track {tid}] t={CustomTracker._format_timestamp(tr.get('best_timestamp', 0))} "
                      f"peak_conf={max(confs):.2f} avg_conf={np.mean(confs):.2f} "
                      f"dets={len(confs)} length={tr.get('longest_length', 0):.1f}ft")

            # Adaptive frame-sampling analytics: how much source-video time the
            # acceleration skipped vs. the wall time spent on inference, then a timeline of
            # when acceleration was engaged vs. when inference ran full-rate.
            video_name = Path(self.video_path).name
            logger.info(format_sampling_stats(video_name, infer_time, sampling_stats))
            timeline = format_sampling_timeline(video_name, sampling_stats)
            if timeline:
                logger.info(timeline)

            fps_eff = sampling_stats.get('fps') or 30
            accel_skipped_frames = sampling_stats.get('accelerated_skipped_frames', 0)
            skipped_frames = sampling_stats.get('baseline_skipped_frames', 0) + accel_skipped_frames
            self.video_timing_ready.emit({
                'video': Path(self.video_path).name,
                'inference': infer_time,
                'segmentation': seg_time,
                'csv': csv_time,
                'frames_sampled': frames_sampled,
                'total_frames': total_frames,
                'detections': total_detections,
                'tracks': track_count,
                'skipped_frames': skipped_frames,
                'accelerated_skipped_seconds': accel_skipped_frames / fps_eff,
            })
            self.processing_finished(significant_tracks)

    def _extract_postproc_payload(self, tracks):
        """Move frame buffers out of `tracks` into a standalone payload for async
        post-processing (training-frame export + GIF encoding).

        `frames` is removed from each track (post-processing consumed/deleted it anyway);
        positions/lengths/confidences are copied so the returned payload shares no
        mutable state with the tracks handed to the UI thread.
        """
        payload = {}
        for key, track in tracks.items():
            frames = track.pop('frames', None)
            if frames is None or len(frames) == 0:
                continue
            payload[key] = {
                'frames': frames,
                'positions': list(track.get('positions', [])),
                'lengths': list(track.get('lengths', [])),
                'confidences': list(track.get('confidences', [])),
            }
        return payload

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
        self.progress_status_changed.emit("")
        self.progress_update.emit(100)
        self.processing_complete.emit(tracks, os.path.basename(self.video_path))

    def save_detections_csv(self, tracks, output_dir, tracker=None):
        csv_path = os.path.join(output_dir, f'{Path(self.video_path).name}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['video_name', 'Flight Location', 'Track Id', 'Highest Conf Timestamp', 'Highest Confidence', 'Average Confidence', 
                        'Lowest Confidence', 'Longest Length', 'Highest Confidence Length',
                        'Number of Detections', 'Meets Thresholds', 'Confidence of Longest Length', 'Label',
                        'manual_length_px', 'manual_length_ft']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for track_id, track in tracks.items():
                meets_thresholds = tracker.is_significant_track(track) if tracker else True
                
                csv_writer.writerow({
                    'video_name': self.video_path,
                    'Flight Location': self.flight_location,
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
                    'manual_length_px': '',
                    'manual_length_ft': '',
                })
            logger.info(f"[csv] wrote {len(tracks)} track(s) -> {csv_path}")


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

def _csv_value_is_empty(value) -> bool:
    """True if a CSV cell is missing, NaN, or blank (used for optional manual lengths)."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def validate_experiment_folder(experiment_folder):
    """
    Returns True if experiment folder contains nonempty detection results folder, else False.
    """
    detection_results_path  = Path(experiment_folder) / "detection_results"
    if not detection_results_path.exists() or len(os.listdir(detection_results_path)) == 0:
        return False
    return True

def validate_experiment_date(date_str):
    """
    Returns True if date_str matches the format <mmddYYYY_HHMMSS>, else False.
    """
    try:
        datetime.strptime(date_str, "%m%d%Y_%H%M%S")
        return True
    except Exception:
        return False

def get_experiment_video_names(experiment_path: Path) -> list[str]:
    """Return sorted unique video basenames for an experiment folder."""
    video_names = set()
    gif_dir = experiment_path / "tracking_gifs"
    if gif_dir.is_dir():
        for f in gif_dir.iterdir():
            if f.name.lower().endswith((".mp4", ".gif")):
                # Example: "clip.mp4_1.gif" or "TRIMMED_2023-05-05_Transect_DJI_0516.mp4_1.gif"
                parts = f.name.rsplit("_", 1)
                if len(parts) == 2:
                    video_names.add(parts[0])
                else:
                    video_names.add(f.stem)

    if not video_names:
        results_dir = experiment_path / "detection_results"
        if results_dir.is_dir():
            for f in results_dir.iterdir():
                if f.suffix.lower() == ".csv":
                    video_names.add(f.name.removesuffix(".csv"))

    return sorted(video_names)


def add_experiment_info(experiment_path: Path): 
    """
    Given an experiment folder name (e.g. "09252025_110653"), returns a string like "(1 video, 3 sharks)".
    Counts videos by unique video basenames in the masks folder, and sharks by number of mask images.
    """
    gif_dir = os.path.join(experiment_path, "tracking_gifs")
    if not os.path.isdir(gif_dir):
        return "(0 video, 0 sharks)"

    gif_files = [f for f in os.listdir(gif_dir) if f.lower().endswith((".mp4", ".gif"))]
    video_names = get_experiment_video_names(experiment_path)
    num_videos = len(video_names)
    num_sharks = len(gif_files)
    return f"({num_videos} video{'s' if num_videos != 1 else ''}, {num_sharks} detection{'s' if num_sharks != 1 else ''})"

EXPERIMENT_NOTE_FILENAME = "experiment_note.txt"


def get_experiment_note_path(exp_dir: Path) -> Path:
    return exp_dir / EXPERIMENT_NOTE_FILENAME


def default_experiment_note(video_names: list[str]) -> str:
    return ", ".join(video_names)


def read_experiment_note(exp_dir: Path) -> str:
    path = get_experiment_note_path(exp_dir)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    video_names = get_experiment_video_names(exp_dir)
    if video_names:
        return default_experiment_note(video_names)
    return ""


def write_experiment_note(exp_dir: Path, note: str) -> None:
    path = get_experiment_note_path(exp_dir)
    note = note.strip()
    if note:
        path.write_text(note, encoding="utf-8")
    elif path.exists():
        path.write_text("", encoding="utf-8")


def build_experiment_display_name(experiment_folder: str, exp_dir: Path) -> str:
    exp_date = format_experiment_date(experiment_folder, to_human=True)
    display = exp_date + " " + add_experiment_info(exp_dir)
    note = read_experiment_note(exp_dir)
    if note:
        display = f"{display} — {note}"
    return display


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


class ModelLoader(QObject):
    """Loads + warms up YOLO and SAM on a background thread so app launch is not
    blocked by the ~375MB SAM checkpoint load and one-time kernel compilation.

    Emits `finished` with the ready YOLO model (or None on failure). SAM is left
    warmed and cached inside SamPredictorCache for the first segmentation call.
    """
    finished = pyqtSignal(object)  # emits the loaded YOLO model, or None on failure

    def run(self):
        model = None
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            model = YOLO(MODEL_PATH).to(device)
            # Warm up the model with one dummy inference so the first real video doesn't
            # pay the one-time kernel-compilation cost mid-run (which showed up as a
            # multi-second blank preview on the very first video).
            try:
                warmup_frame = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), dtype=np.uint8)
                model(warmup_frame, classes=[0], verbose=False)
                logger.info("Model warmup complete")
            except Exception as e:
                logger.warning(f"Model warmup skipped: {e}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

        # Preload + warm up SAM here too. Loading the ~375MB checkpoint (and compiling the
        # first image-encoder kernels) otherwise happened inside the first video's
        # segmentation phase (visible as a ~2s longer first-video "Running Segmentation").
        try:
            predictor = get_sam_predictor()
            predictor.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
            predictor.reset_image()  # free the warmup embedding; keep the loaded weights
            logger.info("SAM warmup complete")
        except Exception as e:
            logger.warning(f"SAM warmup skipped: {e}")

        self.finished.emit(model)


class MainWindow(QMainWindow):
    resized = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1000, 800)

        self.initialize_settings()
        self.init_ui()
        self.init_attributes()
        self.setup_signal_handlers()

        # Load + warm up the models on a background thread AFTER the window is shown,
        # so app launch is instant. The user has to select videos before any inference
        # runs, which almost always outlasts the load; `toggle_processing` gates on
        # `self.model_ready` in the rare case they click Process first.
        self.model = None
        self.model_ready = False
        self._pending_start = False
        QTimer.singleShot(0, self._start_model_loading)

        # Check for a newer build once the window is up (skipped if the user opted out).
        self.version_check_thread = None
        QTimer.singleShot(0, self.check_for_update)

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
        if not self.settings_obj.value("playback_min_frames"):
            self.settings_obj.setValue("playback_min_frames", "5")
        if not self.settings_obj.value("playback_speed"):
            self.settings_obj.setValue("playback_speed", str(DEFAULT_PLAYBACK_SPEED))

        # Cloud Settings
        if not self.settings_obj.value("enable_auto_upload"):
            self.settings_obj.setValue("enable_auto_upload", "false")

        # Update check: when "true", the app skips the startup version check.
        if not self.settings_obj.value("ignore_update"):
            self.settings_obj.setValue("ignore_update", "false")

        if not self.settings_obj.value("detection_labels"):
            save_detection_labels(self.settings_obj, list(DEFAULT_DETECTION_LABELS))

    def populate_label_combo(self, combo, current_label):
        labels = get_detection_labels(self.settings_obj)
        combo.clear()
        items = list(labels)
        if current_label and current_label not in items:
            items.append(current_label)
        combo.addItems(items)
        if current_label:
            combo.setCurrentText(current_label)
        elif items:
            combo.setCurrentIndex(0)

    def refresh_label_combos(self):
        for table in (self.detection_list, self.historical_items):
            if table is None:
                continue
            for row in range(table.rowCount()):
                combo = table.cellWidget(row, 6)
                if combo is None:
                    continue
                current = combo.currentText()
                combo.blockSignals(True)
                self.populate_label_combo(combo, current)
                combo.blockSignals(False)

    def load_drone_settings(self):
        settings_dialog = SettingsDialog(self.settings_obj)
        settings_dialog.settings_updated.connect(self.update_available_drones)
        settings_dialog.detection_labels_page.labels_updated.connect(self.refresh_label_combos)
        settings_dialog.exec()
        # Pick up a default-speed change made in Playback Settings without needing a restart.
        if hasattr(self, "speed_cycle_button"):
            self.frame_player.set_speed(self.playback_speed)
            self._check_speed_button(self.playback_speed)

    def check_for_update(self):
        """Kick off a background version check against the Cloud Function on startup."""
        if str(self.settings_obj.value("ignore_update", "false")).lower() == "true":
            return

        build_info = get_build_info()
        if not build_info:
            # Dev run or unstamped build — no commit to compare, so nothing to check.
            return

        os_key = build_info.get("os", "")
        commit = build_info.get("commit", "")
        if not os_key or not commit:
            return

        self.version_check_thread = VersionCheckThread(UPDATE_ENDPOINT, os_key, commit)
        self.version_check_thread.check_finished.connect(self.on_version_check_finished)
        self.version_check_thread.start()

    def on_version_check_finished(self, update_available, os_key, error):
        if error:
            # Never interrupt the user over a failed/unreachable update check.
            logger.error(f"Version check failed: {error}")
            return
        if update_available:
            self.show_update_dialog(os_key)

    def show_update_dialog(self, os_key):
        box = QMessageBox(self)
        box.setWindowTitle("Update Available")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("A new version of SharkEye is available.")
        box.setInformativeText(
            "Download the latest version?"
        )

        download_btn = box.addButton("Update", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)

        dont_ask_checkbox = QCheckBox("Don't check for updates again")
        box.setCheckBox(dont_ask_checkbox)

        box.exec()

        if dont_ask_checkbox.isChecked():
            self.settings_obj.setValue("ignore_update", "true")

        if box.clickedButton() is download_btn:
            download_url = f"{UPDATE_ENDPOINT}?user_os={os_key}"
            QDesktopServices.openUrl(QUrl(download_url))

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
        # Background pool for per-video post-processing (training-frame export + GIF
        # encoding); capped at 1 so it doesn't starve inference.
        self.postproc_pool = QThreadPool(self)
        self.postproc_pool.setMaxThreadCount(1)
        self._awaiting_first_frame = False
        self.batch_timings = []  # per-video phase timings for the batch summary
        self.api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload"
        self.progress_dialog = None
        self.confidence_threshold = .4 
        self.cleanup_trees = False
        self.reviewing_history = False
        self.confirming_detections = False
        self.edit_mode = False
        self.historical_label_changes = {}  # key: (experiment, video_name, csv_name, track_id) -> new_label
        self.experiments = []
        self.gif_active = False
        self.mask_active = False
        self.current_flight_location = None
        self.low_confidence_threshold = .65
        self.progress_status = None  # string summarizing current process (Running Inference, Uploading Frames, etc.)

    def set_progress_status(self, status: str):
        """Update progress_status and the progress status label when present."""
        self.progress_status = status if status else None
        if getattr(self, "progress_status_label", None) is not None:
            self.progress_status_label.setText(status or "")

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

    def show_help_docs(self):
        guide_path = resource_path("docs/USER_GUIDE_VISUAL.md")
        if not os.path.exists(guide_path):
            QMessageBox.warning(
                self,
                "Help Unavailable",
                f"Help guide not found:\n{guide_path}",
            )
            return

        help_window = getattr(self, "_help_docs_window", None)
        if help_window is None or not help_window.isVisible():
            self._help_docs_window = HelpDocsWindow(guide_path, parent=self)
            self._help_docs_window.show()
            return

        self._help_docs_window.raise_()
        self._help_docs_window.activateWindow()

    def _start_model_loading(self):
        """Kick off model load + warmup on a background QThread (see ModelLoader)."""
        self._model_thread = QThread()
        self._model_loader = ModelLoader()
        self._model_loader.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_loader.run)
        self._model_loader.finished.connect(self._on_models_ready)
        self._model_loader.finished.connect(self._model_thread.quit)
        self._model_loader.finished.connect(self._model_loader.deleteLater)
        self._model_thread.finished.connect(self._model_thread.deleteLater)
        self._model_thread.start()

    def _on_models_ready(self, model):
        """Runs on the main thread once background loading finishes."""
        self.model = model
        self.model_ready = model is not None
        if not self.model_ready:
            QMessageBox.critical(
                self,
                "Model Load Failed",
                "The detection model could not be loaded. Please restart the app.",
            )
            self._pending_start = False
            return
        logger.info("Models ready")
        # If the user already clicked Process while the model was loading, start now.
        if self._pending_start:
            self._pending_start = False
            self.process_button.setText("Process Videos")
            self.start_processing()

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def setup_home_banner(self):
        # Banner container with left/right buttons and centered logo
        banner_widget = QWidget()
        banner_widget.setStyleSheet(banner_surface_style())
        # A grid whose single cell holds three overlapping full-width layers (left group,
        # right group, logo). The logo therefore centers on the *whole* banner width, so it
        # never drifts when the leading/trailing button groups differ in width.
        banner_layout = QGridLayout(banner_widget)
        banner_layout.setContentsMargins(20, 8, 20, 8)

        # Left button (exposed as attribute for later connections)
        self.banner_left_button = self._make_banner_button(
            "Review Previous Experiments", "clock-history.svg", "Previous Experiments")
        self.banner_left_button.clicked.connect(lambda: setattr(self, "reviewing_history", True))
        self.banner_left_button.clicked.connect(self.go_to_review_history) # sets top widget as review

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setScaledContents(False)
        logo_label.setFixedHeight(40)
    
        logo_path = resource_path('assets/images/logo-white.png')
        
        pixmap = QPixmap(logo_path)
        dpr = logo_label.devicePixelRatioF()

        pixmap = pixmap.scaledToHeight(
            int(40 * dpr),
            Qt.TransformationMode.SmoothTransformation
        )
        pixmap.setDevicePixelRatio(dpr)

        logo_label.setPixmap(pixmap)

        # Right button (exposed as attribute for later connections)
        self.banner_right_button = self._make_banner_button(
            "Settings", "gear-fill.svg", "Settings")
        self.banner_right_button.clicked.connect(self.load_drone_settings)

        # Help used to live in a bottom QMenuBar as a bare top-level QAction, which macOS
        # simply does not render — so it was invisible in the Mac build. It's a banner
        # button now, available from both the home and review screens.
        self.banner_help_button = self._make_banner_button(
            "Help", "question-circle-fill.svg", "Open the user guide")
        self.banner_help_button.clicked.connect(self.show_help_docs)

        # Trailing group: settings + help, sized to their content so it hugs the right
        # edge without spanning the cell (a full-width sibling would repaint over — and
        # erase — the left button beneath it).
        right_group = QWidget()
        right_row = QHBoxLayout(right_group)
        right_row.setContentsMargins(0, 0, 0, 0)
        right_row.setSpacing(8)
        right_row.addWidget(self.banner_right_button)
        right_row.addWidget(self.banner_help_button)

        # All three occupy cell (0, 0). Each is content-sized and edge/center-aligned, so
        # they sit at the left edge, right edge, and true center without overlapping — the
        # logo is centered on the full banner width regardless of the side groups' widths.
        v_center = Qt.AlignmentFlag.AlignVCenter
        banner_layout.addWidget(self.banner_left_button, 0, 0, Qt.AlignmentFlag.AlignLeft | v_center)
        banner_layout.addWidget(right_group, 0, 0, Qt.AlignmentFlag.AlignRight | v_center)
        banner_layout.addWidget(logo_label, 0, 0, Qt.AlignmentFlag.AlignCenter)

        banner_widget.setFixedHeight(60)
        self.layout.addWidget(banner_widget)

        # keep reference for tests/other code
        self.banner = banner_widget

    def _make_banner_button(self, text, icon_name, tooltip):
        """Build a labelled banner button (icon + text) on the fixed navy brand surface.

        Icon-only chrome was unlabelled and unguessable; every banner control now carries
        its name. Width follows the label instead of the old fixed 40x40.
        """
        button = QPushButton(f" {text}")
        button.setIcon(banner_icon(resource_path(f"assets/images/{icon_name}")))
        button.setMinimumHeight(32)
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        button.setFlat(True)
        button.setStyleSheet(BANNER_BUTTON)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def toggle_banner_buttons(self, review=True):
        self.banner_left_button.clicked.disconnect()
        # self.banner_right_button.clicked.disconnect()
        self.banner_left_button.setEnabled(True)
        self.banner_right_button.setEnabled(True)
        self.banner_left_button.show()
        self.banner_right_button.show()

        # Help is reachable from every screen — it is deliberately never disabled here.
        self.banner_help_button.setEnabled(True)
        self.banner_help_button.show()

        if review == True:
            # Review Window
            self.banner_left_button.setText(" Home")
            self.banner_left_button.setIcon(banner_icon(resource_path("assets/images/house-fill.svg")))
            self.banner_left_button.setToolTip("Go to Home")
            self.banner_left_button.clicked.connect(self.go_to_home)

            # Allow home when there is nothing to confirm (no detections)
            show_home = (not self.confirming_detections) or (not self.sorted_tracks)
            self.banner_left_button.setVisible(show_home)
            self.banner_left_button.setEnabled(show_home)

            # Settings is unavailable mid-review; hide it rather than leaving a greyed
            # label sitting next to Help.
            self.banner_right_button.setEnabled(False)
            self.banner_right_button.hide()
        else:
            # Home Screen
            self.banner_left_button.setText(" Review Previous Experiments")
            self.banner_left_button.setIcon(banner_icon(resource_path("assets/images/clock-history.svg")))
            self.banner_left_button.setFlat(True)
            self.banner_left_button.setStyleSheet(
                BANNER_BUTTON
            )
            self.banner_left_button.setToolTip("Previous Experiments")

            self.banner_left_button.clicked.connect(lambda: setattr(self, "reviewing_history", True))
            self.banner_left_button.clicked.connect(self.go_to_review_history) # sets top widget as review

            self.banner_right_button.setText(" Settings")
            self.banner_right_button.setIcon(banner_icon(resource_path("assets/images/gear-fill.svg")))
            self.banner_right_button.setEnabled(True)
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
        self.layout.addWidget(self.content_widget, 1)

        self.home_widget = QWidget()
        self.review_widget = QWidget()

        self.stack_widget.addWidget(self.home_widget)
        self.stack_widget.addWidget(self.review_widget)
    
    def update_available_drones(self, select_drone=None):
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

        # Trailing shortcut into the Add New Drone dialog, so users don't have to dig
        # through Settings just to register the drone they're about to process with.
        self.drone_select.addItem(ADD_DRONE_ITEM_TEXT)
        self.drone_select.setItemData(
            self.drone_select.count() - 1, ADD_DRONE_SENTINEL, Qt.ItemDataRole.UserRole
        )

        # Never restore onto the sentinel — findText only matches real drone names.
        preferred = select_drone or self.settings_obj.value("last_drone_type")
        if preferred:
            idx = self.drone_select.findText(str(preferred))
            if idx >= 0:
                self.drone_select.setCurrentIndex(idx)

    def on_drone_selected(self, index):
        """Handle the trailing "Add New Drone…" entry in the drone dropdown.

        Wired to `activated` (user-initiated only) rather than `currentIndexChanged`, so
        repopulating the combo programmatically can never pop the dialog.
        """
        if self.drone_select.itemData(index, Qt.ItemDataRole.UserRole) != ADD_DRONE_SENTINEL:
            self._last_drone_index = index
            return

        dialog = NewDroneDialog(self)
        added_name = None
        if dialog.exec() == QDialog.DialogCode.Accepted:
            drone_name, width, height, fov_input = dialog.get_inputs()
            error = add_drone_to_settings(self.settings_obj, drone_name, width, height, fov_input)
            if error:
                QMessageBox.warning(self, "Invalid Input", error)
            else:
                added_name = drone_name.strip()

        if added_name:
            self.update_available_drones(select_drone=added_name)
        else:
            # Cancelled or rejected: fall back to whatever was selected before.
            self.drone_select.setCurrentIndex(min(getattr(self, "_last_drone_index", 0),
                                                  max(self.drone_select.count() - 2, 0)))
        self._last_drone_index = self.drone_select.currentIndex()

    def setup_home_page(self):
        layout = QVBoxLayout(self.home_widget)

        # Select Video(s) button
        self.select_videos_button = QPushButton("Select Video(s)")
        self.select_videos_button.clicked.connect(self.select_videos)
        # self.select_videos_button.clicked.connect(self.update_remove_buttons)
        # self.select_videos_button.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.select_videos_button)

        # Remove buttons in horizontal layout
        remove_layout = QHBoxLayout()
        self.remove_all_button = QPushButton("Remove All Videos")
        self.remove_all_button.clicked.connect(self.remove_all_videos)
        self.remove_all_button.setEnabled(False)  # Initially disabled
        self.remove_all_button.setVisible(False)  # Only shown once videos are selected
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

        # Drone Dropdown
        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("Select Drone Model:"), 0, 0)
        self.drone_select = QComboBox()
        self.update_available_drones()
        self._last_drone_index = self.drone_select.currentIndex()
        self.drone_select.activated.connect(self.on_drone_selected)
        self.drone_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        form_layout.addWidget(self.drone_select, 0, 1)

        # Altitude Entry
        form_layout.addWidget(QLabel("Enter Drone Altitude (m):"), 1, 0)
        self.altitude_input = QLineEdit('40')
        self.altitude_input.setValidator(QDoubleValidator(0, 999, 2))
        self.altitude_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        form_layout.addWidget(self.altitude_input, 1, 1)

        # Flight Location
        
        form_layout.addWidget(QLabel("Enter Flight Location:"), 2, 0)
        last_flight_location = self.settings_obj.value("last_flight_location")
        self.flight_location_input = QLineEdit(last_flight_location) 
        self.flight_location_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        form_layout.addWidget(self.flight_location_input, 2, 1)

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
        if not self.flight_location_input.text():
            QMessageBox.warning(self, "Failed to Start Processing", "Please enter the flight location for the selected videos")
            return
        if not self.altitude_input.text():
            QMessageBox.warning(self, "Failed to Start Processing", "Please enter the flight altitude for the selected videos")
            return
        if self.drone_select.currentData(Qt.ItemDataRole.UserRole) == ADD_DRONE_SENTINEL:
            QMessageBox.warning(self, "Failed to Start Processing", "Please select a drone model for the selected videos")
            return
        if not self.is_processing:
            if not self.model_ready:
                # Background model load hasn't finished; auto-start once it does.
                self._pending_start = True
                self.process_button.setEnabled(False)
                self.process_button.setText("Loading model…")
                return
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

        # Progress status bar
        self.progress_status_label = QLabel(self.progress_status or "")
        self.progress_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.progress_status_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.progress_status_label.hide()

        # Timer label (initially hidden)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        # self.timer_label.setAlignment()
        self.timer_label.hide()

        # Layout for progress information
        self.progress_info = QHBoxLayout()
        self.progress_info.addWidget(self.progress_status_label)
        self.progress_info.addWidget(self.timer_label)
        # self.progress_info.setSizePolicy(QSizePolicy.Policy.Expanding)

        # Reparent the existing widgets into the dialog so live updates continue
        self.cancel_processsing_button = QPushButton("Cancel Processing")
        self.cancel_processsing_button.clicked.connect(self.toggle_processing)
        
        layout.addWidget(self.frame_display)
        layout.addWidget(self.progress_bar)
        layout.addLayout(self.progress_info)
        # layout.addWidget(self.timer_label)
        layout.addWidget(self.cancel_processsing_button)

        dlg.setLayout(layout)
        dlg.resize(800, 600)

        def progress_dialog_close_event(event):
            if self.is_processing:
                event.ignore()
                self.confirm_cancel_processing()
            else:
                QDialog.closeEvent(dlg, event)

        dlg.closeEvent = progress_dialog_close_event

        self.frame_display.show()
        self.progress_bar.show()
        self.progress_status_label.show()
        self.timer_label.show()
        
        self.progress_bar.setValue(0)
        self.start_time = QDateTime.currentDateTime()
        self.timer.start(1000)
        self.elapsed_time = 0
        self.update_timer()
        self.set_progress_status("Running Inference")

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
            logger.error(f"Error loading drone resolutions: {e}")
            return []
        
        return []

    def start_processing(self):
        self.is_processing = True
        self.batch_timings = []
        self.remove_all_button.setEnabled(False)
        self.video_list.setEnabled(False)
        self.select_videos_button.setEnabled(False)
        self.drone_select.setEnabled(False)
        self.altitude_input.setEnabled(False)
        self.flight_location_input.setEnabled(False)
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

        # Save last flight location and last selected drone model
        self.settings_obj.setValue("last_flight_location", self.flight_location_input.text())
        self.settings_obj.setValue("last_drone_type", self.drone_select.currentText())

        self.video_queue = [self.video_list.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.rowCount())]
        self.current_video_index = 0
        self.total_videos = len(self.video_queue)
        self.processed_videos = 0
        
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.current_output_dir = os.path.join(get_results_dir(), timestamp)
        os.makedirs(self.current_output_dir, exist_ok=True)
        video_names = [os.path.basename(v) for v in self.video_queue]
        write_experiment_note(Path(self.current_output_dir), default_experiment_note(video_names))
        
        # Validate resolution before processing
        if self.video_queue:
            valid_resolutions = self.get_valid_resolutions_for_drone(self.drone_select.currentText())
            open_errors = []
            resolution_mismatches = []

            for video_path in self.video_queue:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    open_errors.append(video_path)
                    continue

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if (width, height) not in valid_resolutions:
                    resolution_mismatches.append((os.path.basename(video_path), width, height))

            if open_errors:
                QMessageBox.critical(
                    self,
                    "Video Error",
                    "Could not open video(s):\n" + "\n".join(os.path.basename(v) for v in open_errors),
                )
                return

            if resolution_mismatches:
                mismatch_lines = "\n".join(f"{name}: {w}x{h}" for name, w, h in resolution_mismatches)
                QMessageBox.warning(
                    self,
                    "Resolution Mismatch",
                    f"The following video(s) have resolutions not valid for the selected drone:\n\n"
                    f"{mismatch_lines}\n\n"
                    f"Valid resolutions for '{self.drone_select.currentText()}':\n"
                    + "\n".join([f"{w}x{h}" for w, h in valid_resolutions]),
                )
                self.cancel_processing()
                return
    
        self.process_next_video()

    def process_next_video(self):
        if self.current_video_index < len(self.video_queue):
            self.process_video(self.video_queue[self.current_video_index])
        else:
            self.finish_processing()

    def process_video(self, video_path):
        self.current_video = video_path
        logger.info(f"[{self.current_video_index + 1}/{self.total_videos}] Processing {os.path.basename(video_path)}")

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
        self.processing_worker = VideoProcessingWorker(video_path, self.model, self.current_output_dir, drone_type=self.drone_select.currentText(), altitude=float(self.altitude_input.text()), flight_location=self.flight_location_input.text())
        self.processing_worker.moveToThread(self.processing_thread)

        self.connect_worker_signals()
        
        self.processing_thread.start()

        self.update_video_list_emoji()
        self.prepare_frame_display()

        # Show an indeterminate (busy) bar until this video's first frame arrives.
        # This covers the model-warmup / video-open gap where the timer was running
        # but no frame had been displayed yet.
        self._awaiting_first_frame = True
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.setRange(0, 0)

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
        self.processing_worker.progress_status_changed.connect(self.set_progress_status, Qt.ConnectionType.QueuedConnection)
        self.processing_worker.postproc_ready.connect(self.dispatch_postproc_job, Qt.ConnectionType.QueuedConnection)
        self.processing_worker.video_timing_ready.connect(self.record_video_timing, Qt.ConnectionType.QueuedConnection)
        self.processing_thread.started.connect(self.processing_worker.run)

    def record_video_timing(self, timing):
        """Accumulate per-video phase timings for the end-of-batch summary."""
        self.batch_timings.append(timing)

    def dispatch_postproc_job(self, payload, output_dir, video_name):
        """Queue per-video post-processing (export + GIF) on the background pool so the next video can start inference."""
        if not payload:
            return
        annotation_color, box_thickness, text_thickness, text_scale = get_annotation_settings(self.settings_obj)
        job = PostProcessJob(payload, output_dir, video_name, annotation_color,
                             box_thickness, text_thickness, text_scale)
        self.postproc_pool.start(job)

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
        release_sam_model()
        self.is_processing = False
        self.progress_display_dialog.close()
        
        # Reset button states
        self.video_list.setEnabled(True)
        self.select_videos_button.setEnabled(True)
        self.drone_select.setEnabled(True)
        self.altitude_input.setEnabled(True)
        self.flight_location_input.setEnabled(True)
        self.update_remove_buttons()  # re-derives Remove All / Process from the list

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
                logger.warning("Thread did not finish in time, forcefully terminating")
                self.processing_thread.terminate()
                self.processing_thread.wait()

            self.processing_thread.deleteLater()
            self.processing_thread = None

        if self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

        self.progress_bar.hide()
        self.timer_label.hide()
        self.progress_status_label.hide()
        self.timer.stop()
        self.frame_display.hide()
        self.set_progress_status("")

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

        logger.info("Processing cancelled")

    def update_remove_buttons(self):
        has_any_items = self.video_list.rowCount() > 0
        # "Remove All Videos" is meaningless with an empty list, so hide it outright
        # rather than leaving a disabled button sitting there.
        self.remove_all_button.setVisible(has_any_items)
        self.remove_all_button.setEnabled(has_any_items and not self.is_processing)
        self.process_button.setEnabled(has_any_items and not self.is_processing)

    def select_videos(self):
        file_dialog = QFileDialog()
        video_files, _ = file_dialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        self.add_video_paths(video_files)

    def add_video_paths(self, video_files):
        """Add one or more video file paths to the queue table, skipping duplicates.

        Shared by the QFileDialog flow (`select_videos`) and any programmatic caller
        (e.g. the headless automation harness), so injecting videos never has to poke
        at the QTableWidget internals directly.
        """
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
                # x-lg.svg is fill="currentColor", which QSvgRenderer resolves to black — tint it
                # so the glyph stays visible in dark mode.
                delete_btn.setIcon(colored_svg_icon(resource_path("assets/images/x-lg.svg"), theme_icon_color()))
                delete_btn.setStyleSheet(FLAT_ICON_BUTTON)
                def delete_row():
                    button = self.sender()
                    if not button:
                        return
                    # Find the row by identity rather than indexAt(button.pos()): the
                    # cell-widget position can resolve to a neighbouring row.
                    for row in range(self.video_list.rowCount()):
                        if self.video_list.cellWidget(row, 1) is button:
                            self.video_list.removeRow(row)
                            break
                    self.update_remove_buttons()

                delete_btn.clicked.connect(delete_row)
                self.video_list.setCellWidget(row_position, 1, delete_btn)
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
            # Let the last video's background post-processing (export + GIF) finish
            # writing to disk before building the review page (which loads the GIFs).
            self.set_progress_status("Finalizing")
            self.postproc_pool.waitForDone()
            # Sort tracks before showing review widget
            self.sort_tracks()
            # Update detection list
            self.update_detection_list()
            # Show first detection if available
            self.finish_processing()
            # Automatically show review widget after processing
            self.stack_widget.setCurrentWidget(self.review_widget)
            self.confirming_detections = True
            self.reviewing_history = False
            self.toggle_banner_buttons()
            self.switch_detection_list(show_historical=True)
            self.setup_review_dropdown(
                select_experiment=os.path.basename(self.current_output_dir)
            )
            self.edit_mode = False
            self.render_historical_experiments()

    def finish_processing(self):
        release_sam_model()
        self.is_processing = False
        self.timer.stop()
        self.process_button.setEnabled(True)  # Re-enable the process button
        
        # Calculate total time using the standalone function
        time_str = format_time(self.elapsed_time)

        # Calculate total detections
        total_detections = sum(len(tracks) for tracks in self.tracks.values())

        # Batch analysis summary: critical-path phase totals across all videos vs.
        # the actual wall clock (the gap shows how much the async pipeline hid).
        if self.batch_timings:
            n = len(self.batch_timings)
            tot_inf = sum(t.get('inference', 0) for t in self.batch_timings)
            tot_seg = sum(t.get('segmentation', 0) for t in self.batch_timings)
            tot_csv = sum(t.get('csv', 0) for t in self.batch_timings)
            tot_frames = sum(t.get('frames_sampled', 0) for t in self.batch_timings)
            tot_tracks = sum(t.get('tracks', 0) for t in self.batch_timings)
            tot_dets = sum(t.get('detections', 0) for t in self.batch_timings)
            tot_accel_skip = sum(t.get('accelerated_skipped_seconds', 0) for t in self.batch_timings)
            logger.info(f"[batch] {n} videos | {tot_tracks} tracks, {tot_dets} detections, "
                  f"{tot_frames} frames sampled | processing(loop)={tot_inf:.1f}s "
                  f"segmentation={tot_seg:.1f}s csv={tot_csv:.2f}s | "
                  f"video time skipped by acceleration={tot_accel_skip:.1f}s | wall clock={time_str}")

        # Close processing window and show completion popup with both time and detections
        self.progress_display_dialog.close()
        msg = QMessageBox()
        msg.setWindowTitle("Processing Complete")
        msg.setText(f"Processing completed!\n\nTotal detections: {total_detections}\nTime taken: {time_str}")
        msg.exec()

    def go_to_review_from_popup(self, popup):
        popup.accept()
        self.reviewing_history = False
        self.switch_detection_list(show_historical=True)
        self.toggle_dropdown_display()
        self.go_to_review_history()

    def show_detection(self, index): # Not in Use
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
            self.update_frame_elements()
            QTimer.singleShot(0, self.update_frame_elements)
            self.show_confidence_warning()
            self.highlight_current_detection()
        else:
            logger.error(f"Error: Invalid detection index: {index}")
            self.show_no_detections_message()

    def show_no_detections_message(self):
        self.frame_player.clear()
        self.frame_player.setText("No detections available")

    def update_detection_list(self):
        # Use a table format for the detection list, matching the historical items table
        labels = ['Experiment', 'Video', 'ID', 'Timestamp', 'Confidence', 'Length', 'Label', '']
        self.detection_list.setRowCount(0)
        self.detection_list.clearContents()
        self.detection_list.setHorizontalHeaderLabels(labels)

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
                time_str = datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%M:%S")
                conf_longest = track.get('longest_conf', 0.0)
                len_high_conf = track.get('longest_length', 0.0)
                label = self.historical_label_changes.get(
                    (getattr(self, "current_experiment", ""), track['video_name'], track['csv_name'], track_id),
                    track.get('label', 'Shark'),
                )

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
                        self.populate_label_combo(combo, label)
                        combo.currentIndexChanged.connect(
                            lambda _index, idx=index, combo=combo: self._update_label_from_table(idx, combo)
                        )
                        self.detection_list.setCellWidget(row_position, col, combo)
                    else:
                        cell = QTableWidgetItem(value)
                        if col == 4 and conf_longest < 0.65:
                            cell.setForeground(QColor('red'))
                        self.detection_list.setItem(row_position, col, cell)

                self.detection_list.item(row_position, 0).setData(Qt.ItemDataRole.UserRole, index)
            except KeyError as e:
                logger.warning(f"Missing key in track data: {e}")
            except Exception as e:
                logger.error(f"Error creating table row for track: {str(e)}")

        self.detection_list.resizeColumnsToContents()
        self.detection_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.detection_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self.highlight_current_detection()

    def _update_label_from_table(self, index, combo):
        # Update label in sorted_tracks when changed from table dropdown
        if 0 <= index < len(self.sorted_tracks):
            key, track = self.sorted_tracks[index]
            new_label = combo.currentText()
            track['label'] = new_label
            logger.info(f"Label updated for track {key} to {new_label}")
    
    def mark_for_deletion(self):
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete this detection?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            row = self.historical_items.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Mark for Deletion", "No track selected.")
                return

            experiment, video_name, csv_name, track_id = self.historical_items.item(row, 0).data(Qt.ItemDataRole.UserRole)
            key = (experiment, video_name, csv_name, int(track_id))
            self.historical_label_changes[key] = "Delete"

            # Remove the row from the QTableWidget
            self.historical_items.removeRow(row)
            self.historical_items.selectRow(max(0, row - 1))
            self.show_historical_gif()
    
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
                logger.info(f"Deleted experiment directory: {exp_dir}")
                self.sort_tracks()
                self.setup_review_dropdown()
                self.render_historical_experiments()
                return
            except Exception as e:
                logger.error(f"Error deleting experiment directory {exp_dir}: {e}")

        # Delete track-specific files
        deleted_files = 0
        for folder in folders:
            folder_path = exp_dir / folder
            pattern = f"{Path(csv_name).stem}_{track_id}.*"
            for file in folder_path.glob(pattern):
                try:
                    file.unlink()
                    deleted_files += 1
                except Exception as e:
                    logger.error(f"Error deleting {file}: {e}")
        logger.info(f"Deleted track {track_id} from {csv_name} ({deleted_files} files removed)")

        # Remove row from CSV in detection_results
        csv_path = det_dir / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if len(df) == 1:
                    logger.info(f"Removing last detection for video. Deleting CSV")
                    csv_path.unlink()
                else:
                    df = df[df['Track Id'].astype(int) != int(track_id)]
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Removed track {track_id} from {csv_path}")
            except Exception as e:
                logger.error(f"Error updating CSV {csv_path}: {e}")

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
                    self.show_confidence_warning()
        else:
            # Fallback for QListWidget (shouldn't be used anymore)
            selected_items = self.detection_list.selectedItems()
            if selected_items:
                index = selected_items[0].data(Qt.ItemDataRole.UserRole)
                if index != self.current_detection_index:
                    self.show_detection(index)
                    self.show_confidence_warning()

    def highlight_current_detection(self):
        # For QTableWidget, select the row corresponding to current_detection_index
        for row in range(self.detection_list.rowCount()):
            item = self.detection_list.item(row, 0)
            if item is None:
                continue
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                continue
            if index == self.current_detection_index:
                self.detection_list.selectRow(row)
                self.detection_list.scrollToItem(item)
            # else:
            #     self.detection_list.selectRow(row)
            #     #self.detection_list.setRangeSelected(
            #     #     QTableWidgetSelectionRange(0, row, self.detection_list.columnCount(), row))
            #         # self.detection_list.visualItemRect(self.detection_list.item(row, 0)), False

    def update_label(self, combo, row):
        if row < 0:
            return
        experiment, video_name, csv_name, track_id = self.historical_items.item(row, 0).data(Qt.ItemDataRole.UserRole)
        key = (experiment, video_name, csv_name, int(track_id))
        new_label = combo.currentText()
        self.historical_label_changes[key] = new_label

        if self.historical_label_changes[key] is None:
            pass
        if self.historical_label_changes[key] == combo.previous_text:
            del self.historical_label_changes[key]

        
        # if not self.sorted_tracks:
        #     print("Error: No sorted tracks available. Cannot update label.")
        #     return

        # Refresh the preview for the newly-labeled selection. The visible historical
        # table already reflects the change (the user edited its combo directly), so
        # we only refresh the GIF/MP4 preview here. We deliberately do NOT rebuild the
        # hidden detection_list from sorted_tracks — that was pointless (the table is
        # hidden), noisy, and dropped to "0 tracks" when viewing history without a
        # fresh inference. Guarded so a relabel can never take down the app.
        try:
            self.show_historical_gif()
        except Exception as e:
            logger.error(f"Error refreshing preview after label change: {e}")
        
    def sort_tracks(self):
        # Flatten all tracks from all videos into a single list
        all_tracks = []
        for video_name, video_tracks in self.tracks.items():
            csv_name = f"{Path(video_name).name}.csv"
            full_video_name = video_name
            for video_path in self.video_queue:
                if Path(video_path).name == Path(video_name).name:
                    full_video_name = video_path
                    break
            for track_id, track in video_tracks.items():
                track_info = {
                    'video_name': full_video_name,
                    'csv_name': csv_name,
                    'track_id': track_id,
                    **track  # Include all track information
                }
                all_tracks.append((f"{video_name}_{track_id}", track_info))
        
        self.sorted_tracks = sorted(
            all_tracks,
            key=lambda x: (x[1]['video_name'], x[1]['timestamps'][0], x[1]['id'])
        )

        logger.info(f"Sorted {len(self.sorted_tracks)} tracks across {len(self.tracks)} videos")

    def go_to_review_history(self):
        self.confirming_detections = False
        self.reviewing_history = True
        self.stack_widget.setCurrentWidget(self.review_widget)
        self.setup_review_dropdown(select_newest=True)
        self.edit_mode = False
        self.render_historical_experiments()
        self.toggle_banner_buttons(review=True)        

    def _apply_review_ui_state(self):
        confirming = self.confirming_detections
        has_items = self.historical_items.rowCount() > 0

        self.confirm_detections_button.setVisible(confirming)
        self.edit_tracks_button.setVisible(not confirming)
        self.save_changes_button.setVisible(not confirming)

        if confirming:
            self.confirm_detections_button.setEnabled(has_items)
        else:
            self.edit_tracks_button.setEnabled(True)
            self.edit_tracks_button.setText("Cancel Changes" if self.edit_mode else "Edit Tracks")
            self.save_changes_button.setEnabled(self.edit_mode and has_items)

        for r in range(self.historical_items.rowCount()):
            label_combo = self.historical_items.cellWidget(r, 6)
            delete_button = self.historical_items.cellWidget(r, 7)
            if label_combo:
                label_combo.setEnabled(self.edit_mode or confirming)
            if delete_button:
                delete_button.setEnabled(self.edit_mode and not confirming)

        self.toggle_dropdown_display()

        if self.stack_widget.currentWidget() == self.review_widget:
            # Allow home when there is nothing to confirm (no detections)
            show_home_button = (not confirming) or (not self.sorted_tracks)
            self.banner_left_button.setVisible(show_home_button)
            self.banner_left_button.setEnabled(show_home_button)

    def _is_edit_mode(self):
        return self.edit_mode

    def _set_edit_state(self, enabled):
        self.edit_mode = enabled
        self._apply_review_ui_state()

    def _cancel_edit_changes(self):
        self.historical_label_changes = {}
        self.edit_mode = False
        self.render_historical_experiments()
        self.update_detection_list()

    def toggle_edit_state(self, set_state=None):
        if set_state is True:
            self._set_edit_state(True)
        elif set_state is False:
            self._set_edit_state(False)
        elif self._is_edit_mode():
            self._cancel_edit_changes()
        else:
            self._set_edit_state(True)

    def toggle_review_buttons(self, enable):
        self.historical_items.setEnabled(enable)
        self.toggle_display_switch.setEnabled(enable)
        self.toggle_display_switch.setChecked(enable)
        self.toggle_display_switch.setVisible(enable)
        self.mask_icon.setVisible(enable)
        self.box_icon.setVisible(enable)
        if not enable:
            self.mask_active = False
        self._update_edit_frame_button()
        self._apply_review_ui_state()

    def confirm_detections(self):
        if not self._save_historical_label_changes(confirm_always=True):
            return
        self._enter_normal_review_mode()

    def _enter_normal_review_mode(self):
        self.confirming_detections = False
        self.reviewing_history = True
        self.edit_mode = False
        experiment = (
            os.path.basename(self.current_output_dir)
            if getattr(self, "current_output_dir", None)
            else None
        )
        if experiment:
            self.setup_review_dropdown(select_experiment=experiment)
        else:
            self.setup_review_dropdown(select_newest=True)
        self.render_historical_experiments()
        
    def toggle_display_mode(self):
        # Historical mode: always show mask overlay if available
        # if self.reviewing_history:
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
                # self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFinOutline.svg")))
                self.mask_active = True
            else:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Alert")
                dlg.setText("Error: No mask drawn for this track")
                dlg.exec()
                self.mask_active = False
            # Do NOT start/stop timer in historical mode
            self.gif_active = False
            self.update_frame_elements()
            self._update_edit_frame_button()
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

        self.review_select_widget = QWidget()
        review_select_row = QHBoxLayout(self.review_select_widget)
        review_select_row.setContentsMargins(0, 0, 0, 0)
        review_select_row.setSpacing(6)
        
        self.review_dropdown = QComboBox()
        self.review_dropdown.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.review_dropdown.setMinimumContentsLength(35)
        self.review_dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        review_select_row.addWidget(self.review_dropdown, 1)
        layout.addWidget(self.review_select_widget)
        self.review_select_widget.hide()

        self.experiment_note_button = QPushButton()
        self.experiment_note_button.setIcon(colored_svg_icon(
            resource_path("assets/images/pencil-fill.svg"),
            theme_icon_color(),
        ))
        self.experiment_note_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.experiment_note_button.setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.experiment_note_button.setToolTip("Add or edit experiment note")
        self.experiment_note_button.clicked.connect(self.edit_experiment_note)
        review_select_row.addWidget(self.experiment_note_button)


        
        # Frame player / in-place frame editor share one stack slot so Edit Frame
        # can replace the active mask view without opening a separate window.
        self.frame_stack = QStackedWidget()
        self.frame_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_stack.setMinimumWidth(720)       
        self.frame_stack.setMinimumHeight(405)

        self.frame_player = FramePlayer(self.settings_obj)
        self.frame_player.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Match ZoomableFrameView: fill the stack; image is KeepAspectRatio-centered inside.
        self.frame_player.setMinimumSize(0, 0)
        self.frame_player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_stack.addWidget(self.frame_player)

        self.frame_editor = FrameLineEditorWidget(
            parent=self.frame_stack,
            settings_obj=self.settings_obj,
        )
        self.frame_editor.changes_confirmed.connect(self._on_frame_editor_result)
        self.frame_stack.addWidget(self.frame_editor)
        self.frame_stack.setCurrentWidget(self.frame_player)

        layout.addWidget(self.frame_stack)

        self.setup_playback_controls(layout)

        self.box_icon = QSvgWidget(resource_path("assets/images/MdiSharkFinOutline.svg"), parent=self.frame_player)
        self.box_icon.setFixedSize(30, 30)
        self.box_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.mask_icon = QSvgWidget(resource_path("assets/images/MdiSharkFin.svg"), parent=self.frame_player)
        self.mask_icon.setFixedSize(30, 30)
        self.mask_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Overlay button (bottom-left of the frame, opposite the mask toggle) that
        # opens the frame line editor for the currently selected mask frame.
        edit_icon, edit_icon_size = colored_svg_icon_fit(
            resource_path("assets/images/draw-line.svg"),
            QColor("white"),
            max_edge=30,
        )
        self.edit_frame_button = QPushButton(parent=self.frame_player)
        self.edit_frame_button.setIcon(edit_icon)
        self.edit_frame_button.setIconSize(edit_icon_size)
        self.edit_frame_button.setFixedSize(edit_icon_size.width() + 4, edit_icon_size.height() + 4)
        self.edit_frame_button.setStyleSheet(FLAT_ICON_BUTTON)
        self.edit_frame_button.setToolTip("Edit Frame")
        self.edit_frame_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_frame_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.edit_frame_button.clicked.connect(self.open_frame_editor)
        self.edit_frame_button.setVisible(False)

        self.toggle_display_switch = SwitchControl(
            bg_color="#777777",
            circle_color="#DDD",
            active_color="#777777",
            animation_duration=100,
            checked=False,
            change_cursor=True,
            parent=self.frame_player
        )
        self.toggle_display_switch.clicked.connect(self.toggle_display_mode)

        # Add warning when detection falls before 
        self.low_confidence_warning = QLabel(
            "⚠️ Low confidence in this detection. Please review before saving!",
            parent=self.frame_player)
        self.low_confidence_warning.setStyleSheet(f"color: {warning_text_color()};")
        self.low_confidence_warning.setScaledContents(True)
        self.low_confidence_warning.setVisible(False)

        self.frame_player.resized.connect(self.update_frame_elements)
        self.update_frame_elements()

        labels = ['Experiment', 'Video', 'ID', 'Timestamp', 'Confidence', 'Length', 'Label', '']

        self.detection_list = QTableWidget()
        self.detection_list.setColumnCount(len(labels))
        self.detection_list.setMaximumHeight(120)
        self.detection_list.hide()
        self.detection_list.itemSelectionChanged.connect(self.show_selected_detection)
        layout.addWidget(self.detection_list)

        # Historical items list (initially hidden)
        self.historical_items = QTableWidget()
        self.historical_items.setColumnCount(len(labels))
        self.historical_items.setMaximumHeight(120)
        self.historical_items.hide()
        self.historical_items.itemSelectionChanged.connect(self.show_historical_gif)
        self.historical_items.itemSelectionChanged.connect(self.set_current_label_combo)
        self.historical_items.itemSelectionChanged.connect(lambda: setattr(self, 'current_detection_index', self.historical_items.currentRow()))
        self.historical_items.setHorizontalHeaderLabels(labels)
        layout.addWidget(self.historical_items)

        # Export/Upload buttons
        button_layout = QHBoxLayout()
        self.save_changes_button = QPushButton("Save Changes")  # text will change in historical mode
        self.confirm_detections_button = QPushButton("Confirm Detections")
        self.confirm_detections_button.hide()
        self.confirm_detections_button.clicked.connect(self.confirm_detections)
        self.edit_tracks_button = QPushButton("Edit Tracks")
        self.edit_tracks_button.clicked.connect(lambda: self.toggle_edit_state())
        self.save_changes_button.clicked.connect(self.export_results)
        
        button_layout.addWidget(self.edit_tracks_button)
        button_layout.addWidget(self.save_changes_button)
        button_layout.addWidget(self.confirm_detections_button)
        layout.addLayout(button_layout)

        # Finish Review Setup
        self.setup_review_dropdown()
        self.review_dropdown.currentIndexChanged.connect(self.render_historical_experiments)
        self.review_dropdown.currentIndexChanged.connect(self._update_review_dropdown_tooltip)

    def setup_playback_controls(self, layout):
        """Play/pause, speed and scrub controls for the detection clip.

        Clips are built from frames sampled every 10-60 source frames (see
        iter_sampled_frames) and written at 10 fps, so they replay tens of times faster
        than real time and loop immediately — too fast to judge a detection. These
        controls let a reviewer stop on a frame and step through it.
        """
        self.playback_controls = QWidget()
        row = QHBoxLayout(self.playback_controls)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(8)

        self.play_pause_button = QPushButton()
        self.play_pause_button.setToolTip("Play or pause the detection clip (Space)")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_pause_button)

        row.addWidget(QLabel("Speed:"))
        # A single button that cycles through PLAYBACK_SPEEDS; its label is the active rate.
        self.speed_cycle_button = QPushButton()
        self.speed_cycle_button.setMaximumWidth(56)
        self.speed_cycle_button.setToolTip("Click to cycle playback speed")
        self.speed_cycle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.speed_cycle_button.clicked.connect(self.cycle_playback_speed)
        row.addWidget(self.speed_cycle_button)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setToolTip("Scrub through the clip frame by frame")
        # Dragging implies "I want to look at this frame", so stop playback first.
        self.frame_slider.sliderPressed.connect(self.frame_player.pause)
        self.frame_slider.sliderPressed.connect(self._sync_play_pause_button)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        row.addWidget(self.frame_slider, 1)

        self.frame_counter_label = QLabel("0 / 0")
        self.frame_counter_label.setMinimumWidth(64)
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.frame_counter_label)

        layout.addWidget(self.playback_controls)

        self.frame_player.frame_changed.connect(self._on_player_frame_changed)
        self.frame_player.playback_mode_changed.connect(self._on_playback_mode_changed)

        # Apply the saved default speed (0.5x out of the box — 1x is too fast to review).
        self.frame_player.set_speed(self.playback_speed)
        self._check_speed_button(self.playback_speed)
        self._sync_play_pause_button()
        self._on_playback_mode_changed("empty")

    @property
    def playback_speed(self):
        try:
            return float(self.settings_obj.value("playback_speed", str(DEFAULT_PLAYBACK_SPEED)))
        except (TypeError, ValueError):
            return DEFAULT_PLAYBACK_SPEED

    def set_playback_speed(self, multiplier):
        """Apply and persist a playback rate so it sticks across detections and sessions."""
        self.settings_obj.setValue("playback_speed", str(multiplier))
        self.frame_player.set_speed(multiplier)

    def cycle_playback_speed(self):
        """Advance to the next configured playback rate, wrapping back to the first."""
        speeds = list(PLAYBACK_SPEEDS)
        try:
            i = speeds.index(self.playback_speed)
        except ValueError:
            i = -1
        nxt = speeds[(i + 1) % len(speeds)]
        self.set_playback_speed(nxt)
        self._check_speed_button(nxt)

    def _check_speed_button(self, multiplier):
        # The cycle button's label reflects the active rate.
        if hasattr(self, "speed_cycle_button"):
            self.speed_cycle_button.setText(f"{multiplier:g}x")

    def toggle_playback(self):
        self.frame_player.toggle_play_pause()
        self._sync_play_pause_button()

    def _sync_play_pause_button(self):
        playing = self.frame_player.is_playing()
        icon_name = "pause-button-svg.svg" if playing else "play-button-svg.svg"
        self.play_pause_button.setIcon(
            colored_svg_icon(resource_path(f"assets/images/{icon_name}"), theme_icon_color()))
        self.play_pause_button.setText(" Pause" if playing else " Play")
        # Drawing on a playback frame requires it to be paused, so keep the draw-line
        # button's enabled state in sync with play/pause.
        self._update_edit_frame_button()

    def _on_player_frame_changed(self, index, total):
        self.frame_counter_label.setText(f"{index + 1} / {total}")
        if self.frame_slider.maximum() != total - 1:
            self.frame_slider.setMaximum(max(0, total - 1))
        # Guard so echoing the player's position back doesn't re-trigger a seek.
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)

    def _on_slider_moved(self, value):
        self.frame_player.seek(value)

    def _on_playback_mode_changed(self, mode):
        """Show controls only for the backend that can actually use them.

        "frames" (MP4 clips) supports everything; "movie" (legacy GIFs) supports
        play/pause and speed but not reliable frame seeking; "static"/"empty" (clips
        below playback_min_frames, the mask overlay, no selection) support nothing.
        """
        self._playback_mode = mode
        self.playback_controls.setVisible(mode in ("frames", "movie"))
        scrubbable = mode == "frames"
        self.frame_slider.setEnabled(scrubbable)
        self.frame_counter_label.setVisible(scrubbable)
        if not scrubbable:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(0)
            self.frame_slider.setMaximum(0)
            self.frame_slider.blockSignals(False)
        self._sync_play_pause_button()
        self._update_edit_frame_button()

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

    def update_frame_elements(self):
        if not hasattr(self, "frame_player"):
            return

        rect = self.frame_player.content_rect()
        if rect.isNull():
            return

        self.low_confidence_warning.adjustSize()

        # Keep overlays inset inside the displayed video area (not the full widget).
        margin = 7
        btn_y = rect.y() + rect.height() - self.mask_icon.height() - 4
        btn_x = rect.x() + rect.width() - self.mask_icon.width() - margin
        switch_x = btn_x - self.toggle_display_switch.width() - 9
        box_x = switch_x - self.box_icon.width() - 9

        # Clamp so the control strip never leaves the video content rect.
        min_x = rect.x() + margin
        if box_x < min_x:
            shift = min_x - box_x
            box_x += shift
            switch_x += shift
            btn_x += shift

        self.mask_icon.move(btn_x, btn_y)
        self.toggle_display_switch.move(switch_x, btn_y)
        self.box_icon.move(box_x, btn_y)
        self.box_icon.raise_()
        self.toggle_display_switch.raise_()
        self.mask_icon.raise_()

        # Mirror the mask toggle's margins onto the opposite (left) side.
        edit_y = rect.y() + rect.height() - self.edit_frame_button.height() - 4
        edit_x = rect.x() + margin
        self.edit_frame_button.move(edit_x, edit_y)
        self.edit_frame_button.raise_()

        # Position warning to bottom center of video.
        warning_x = rect.x() + (rect.width() - self.low_confidence_warning.width()) // 2
        warning_y = btn_y - self.low_confidence_warning.height() - 4
        warning_x = max(rect.x() + margin, min(warning_x, rect.x() + rect.width() - self.low_confidence_warning.width() - margin))
        warning_y = max(rect.y() + margin, warning_y)
        self.low_confidence_warning.move(warning_x, warning_y)
        self.low_confidence_warning.raise_()

    def _update_edit_frame_button(self):
        """Show the draw-line overlay whenever there's a measurable frame to edit.

        Two sources: the mask overlay (edits the saved best frame) and clip playback
        (edits the frame currently shown). A playback frame must be paused before it
        can be drawn on, so the button is disabled while the clip is playing.
        """
        if not hasattr(self, "edit_frame_button"):
            return
        editing = (
            hasattr(self, "frame_stack")
            and hasattr(self, "frame_editor")
            and self.frame_stack.currentWidget() is self.frame_editor
        )
        mask_showing = bool(getattr(self, "mask_active", False))
        playback_frames = getattr(self, "_playback_mode", None) == "frames"
        visible = (mask_showing or playback_frames) and not editing
        self.edit_frame_button.setVisible(visible)
        if not visible:
            return
        if playback_frames and not mask_showing:
            paused = not self.frame_player.is_playing()
            self.edit_frame_button.setEnabled(paused)
            self.edit_frame_button.setToolTip(
                "Draw a measurement line on this frame" if paused
                else "Pause the clip to draw on a frame"
            )
        else:
            self.edit_frame_button.setEnabled(True)
            self.edit_frame_button.setToolTip("Draw a measurement line on this frame")
        self.edit_frame_button.raise_()

    def _current_frame_image_path(self):
        """Resolve the original frame image path for the currently selected track."""
        row = self.historical_items.currentRow()
        if row < 0:
            return None
        experiment = format_experiment_date(
            self.historical_items.item(row, 0).text(), to_human=False
        )
        video_basename = self.historical_items.item(row, 1).text()
        track_id = self.historical_items.item(row, 2).text()
        frame_path = (
            Path(get_results_dir()) / experiment / "frames"
            / f"{video_basename}_{track_id}.jpg"
        )
        return str(frame_path) if frame_path.exists() else None

    def open_frame_editor(self):
        """Replace the active frame view with the in-place line editor.

        From the mask overlay this edits the saved best frame; during clip playback it
        edits the frame currently paused in the player (captured full-res from memory).
        """
        self.frame_editor._update_drone_settings()
        initial_drone = self.settings_obj.value("last_drone_type") or None

        if getattr(self, "mask_active", False):
            frame_path = self._current_frame_image_path()
            if not frame_path:
                self._frame_editor_error("Error: No frame available to edit")
                return
            loaded = self.frame_editor.load_image(frame_path, initial_drone=initial_drone)
        else:
            # Playback frame — only reachable while paused (button is disabled otherwise).
            pixmap = self.frame_player.current_frame_pixmap()
            if pixmap is None:
                self._frame_editor_error("Error: No frame available to edit")
                return
            loaded = self.frame_editor.load_pixmap(pixmap, initial_drone=initial_drone)

        if not loaded:
            self._frame_editor_error("Error: Failed to load frame for editing")
            return

        self.frame_stack.setCurrentWidget(self.frame_editor)
        self.edit_frame_button.setVisible(False)

    def _frame_editor_error(self, message):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Alert")
        dlg.setText(message)
        dlg.exec()

    def close_frame_editor(self):
        """Return from the in-place editor to the normal frame/mask view."""
        self.frame_stack.setCurrentWidget(self.frame_player)
        self._update_edit_frame_button()
        self.update_frame_elements()

    def _on_frame_editor_result(self, result):
        """Persist manual line lengths from the frame editor into the track's CSV row."""
        if not result:
            self.close_frame_editor()
            return

        length_px = result.get("length_pixels")
        length_ft = result.get("length_feet")
        if length_px is None:
            self.close_frame_editor()
            return

        row = self.historical_items.currentRow()
        if row < 0:
            self.close_frame_editor()
            return

        meta = self.historical_items.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if not meta:
            self.close_frame_editor()
            return

        experiment, _video_name, csv_name, track_id = meta
        csv_path = Path(get_results_dir()) / experiment / "detection_results" / csv_name
        try:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            for col in ("manual_length_px", "manual_length_ft"):
                if col not in df.columns:
                    df[col] = ""

            mask = df["Track Id"].astype(int) == int(track_id)
            if not mask.any():
                raise ValueError(f"Track {track_id} not found in {csv_path}")

            df.loc[mask, "manual_length_px"] = length_px
            df.loc[mask, "manual_length_ft"] = length_ft if length_ft is not None else ""
            df.to_csv(csv_path, index=False)

            if length_ft is not None:
                length_item = self.historical_items.item(row, 5)
                if length_item is not None:
                    length_item.setText(f"{float(length_ft):.1f}ft")

            QMessageBox.information(
                self,
                "Length Saved",
                "Length correction saved to detection results.",
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Could not save manual length to CSV:\n{e}",
            )

        self.close_frame_editor()

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
                self.toggle_dropdown_display()
        else:
            self.toggle_review_buttons(enable=False)
            self.show_no_detections_message()

    def set_current_label_combo(self):
        """
        Set self.label_combo to the QComboBox found in the currently selected row.
        Works for both historical (historical_items) and current detections (detection_list).
        """
        table = self.historical_items # if getattr(self, "reviewing_history", False) else self.detection_list
        if table is None:
            return

        row = table.currentRow()
        if row < 0:
            return

        combo = table.cellWidget(row, 6)
        if combo is None:
            return

        # Assign and preserve previous text for change detection
        self.label_combo = combo # 

    def show_historical_gif(self):
        if (
            hasattr(self, "frame_stack")
            and hasattr(self, "frame_editor")
            and self.frame_stack.currentWidget() is self.frame_editor
        ):
            self.close_frame_editor()
        self.gif_active = True
        self.mask_active = False
        self._update_edit_frame_button()
        if self.frame_player.timer.isActive():
            self.frame_player.timer.stop()

        selected = self.historical_items.selectedItems()
        if not selected and self.historical_items.rowCount() == 0:
            self.show_no_detections_message()
            return
        if not selected:
            row = self.current_detection_index
        else:
            # Get the selected row index
            row = self.historical_items.currentRow()
        if row < 0:
            return

        experiment = format_experiment_date(self.historical_items.item(row, 0).text(), to_human=False)  # first column
        video_basename = self.historical_items.item(row, 1).text()  # second column
        track_id = self.historical_items.item(row, 2).text()  # third column
        gif_dir = Path(get_results_dir()) / experiment / "tracking_gifs"
        
        # Try MP4 first (new format), then fall back to GIF (legacy format)
        mp4_name = f"{video_basename}_{track_id}.mp4"
        mp4_path = gif_dir / mp4_name
        gif_name = f"{video_basename}_{track_id}.gif"
        gif_path = gif_dir / gif_name
        
        if mp4_path.exists():
            # self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
            self.toggle_display_switch.reset_position()
            self.toggle_display_switch.update()
            self.frame_player.set_video(str(mp4_path))
            self.update_frame_elements()
            QTimer.singleShot(0, self.update_frame_elements)
        elif gif_path.exists():
            # self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
            self.toggle_display_switch.reset_position()
            self.toggle_display_switch.update()
            self.frame_player.set_gif(str(gif_path))
            self.update_frame_elements()
            QTimer.singleShot(0, self.update_frame_elements)
        else:
            # Try alternative naming (without extension in basename)
            alt_mp4 = gif_dir / f"{Path(video_basename).stem}_{track_id}.mp4"
            alt_gif = gif_dir / f"{Path(video_basename).stem}_{track_id}.gif"
            if alt_mp4.exists():
                self.frame_player.set_video(str(alt_mp4))
                self.update_frame_elements()
                QTimer.singleShot(0, self.update_frame_elements)
            elif alt_gif.exists():
                self.frame_player.set_gif(str(alt_gif))
                self.update_frame_elements()
                QTimer.singleShot(0, self.update_frame_elements)
            else:
                self.frame_player.clear()
                self.frame_player.setText(f"Video not found:\n{mp4_name} or {gif_name}")

    def render_historical_experiments(self):
        # Render Historical Experiments and add to List
        self.historical_items.setRowCount(0)
        self.historical_items.clearContents()
        
        experiment_folder = self.review_dropdown.currentData(Qt.ItemDataRole.UserRole)
        if not experiment_folder:
            self.toggle_review_buttons(enable=False)
            return

        self.current_experiment = experiment_folder
        exp_date = format_experiment_date(self.current_experiment, to_human=True)

        experiments_root = get_results_dir()
        labels = ['Experiment', 'Video', 'ID', 'Timestamp', 'Confidence', 'Length', 'Label', '']

        try:
            # newest-first
            exp_dir = Path(experiments_root) / self.current_experiment
            det_dir = exp_dir / "detection_results"
            gif_dir = exp_dir / "tracking_gifs"

            if not (det_dir.exists() and gif_dir.exists()):
                logger.error("Error")

            # each CSV can contain multiple tracks (rows) → iterate rows!
            for csv_name in os.listdir(det_dir):
                csv_path = det_dir / csv_name
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    logger.error(f"Error reading {csv_path}: {e}")
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
                        manual_ft = row.get('manual_length_ft', '')
                        if not _csv_value_is_empty(manual_ft):
                            display_length = float(manual_ft)
                        else:
                            display_length = len_high_conf
                        label = self.historical_label_changes.get(
                            (self.current_experiment, video_path_str, csv_name, track_id),
                            row.get('Label', 'Shark'),
                        )

                        row_position = self.historical_items.rowCount()
                        self.historical_items.insertRow(row_position)
                        values = [
                            exp_date,
                            video_basename,
                            str(track_id),
                            time_str,
                            f"{conf_longest:.2f}",
                            f"{display_length:.1f}ft",
                            label
                        ]

                        for col, value in enumerate(values):
                            if col == 6:
                                # Creates dropdown for label
                                combo = QComboBox()
                                self.populate_label_combo(combo, label)
                                combo.previous_text = label
                                # combo.currentIndexChanged.connect(lambda: setattr(self, "label_combo", combo))
                                combo.currentIndexChanged.connect(
                                    lambda _index, combo=combo, row=row_position: self.update_label(combo, row)
                                )
                                self.historical_items.setCellWidget(row_position, col, combo)
                            else:
                                cell = QTableWidgetItem(value)
                                # Make cell selectable but not editable (prevents editing on double-click)
                                cell.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                                if col == 4 and conf_longest < 0.65:
                                    cell.setForeground(QColor('red'))
                                self.historical_items.setItem(row_position, col, cell)
                        
                        # Hide "Experiment" and "ID" Columns
                        self.historical_items.setColumnHidden(0, True)
                        self.historical_items.setColumnHidden(2, True)

                        # Create delete button
                        del_button = QPushButton("")
                        del_button.setIcon(colored_svg_icon(resource_path("assets/images/trash-fill.svg"), theme_icon_color()))
                        del_button.setStyleSheet(FLAT_ICON_BUTTON)
                        del_button.clicked.connect(self.mark_for_deletion)
                        self.historical_items.setCellWidget(row_position, 7, del_button)

                        self.historical_items.item(row_position, 0).setData(
                            Qt.ItemDataRole.UserRole, (self.current_experiment, video_path_str, csv_name, track_id)
                        )

                    except Exception as e:
                        logger.error(f"Error creating historical row item from {csv_path}: {e}")
                

            # configure headers and resizing once
            self.historical_items.setHorizontalHeaderLabels(labels)
            self.historical_items.resizeColumnsToContents()
            self.historical_items.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.historical_items.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.historical_items.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.historical_items.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self.historical_items.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)

            self.switch_detection_list(show_historical=True)
            # self.reviewing_history = True

            detections_present = self.historical_items.rowCount() > 0
            
            # self.toggle_review_buttons(enable=detections_pres6ent)
            self.show_confidence_warning()
            self.toggle_dropdown_display()
            self._set_edit_state(self.edit_mode)
            self._select_first_historical_row()

        except Exception as e:
            logger.error(f"Error while building historical list: {e}")
            # self.switch_detection_list(show_historical=True)
            # self.reviewing_history = False

            # self.switch_detection_list(show_historical=True)
            # self.reviewing_history = True

    def _select_first_historical_row(self):
        if self.historical_items.rowCount() > 0:
            self.historical_items.setCurrentCell(0, 0)
            self.current_detection_index = 0
            self.show_historical_gif()

    def setup_review_dropdown(self, select_experiment=None, select_newest=False):
        experiments_root = get_results_dir()
        preserve_folder = None
        if not select_newest and select_experiment is None:
            preserve_folder = self.review_dropdown.currentData(Qt.ItemDataRole.UserRole)

        self.review_dropdown.blockSignals(True)
        self.review_dropdown.clear()
        for experiment in sorted(os.listdir(experiments_root), reverse=True):
            if validate_experiment_date(experiment):
                exp_dir = Path(experiments_root) / experiment
                if not validate_experiment_folder(exp_dir):
                    continue
                exp_disp = build_experiment_display_name(experiment, exp_dir)
                self.review_dropdown.addItem(exp_disp, experiment)
                idx = self.review_dropdown.count() - 1
                self.review_dropdown.setItemData(idx, exp_disp, Qt.ItemDataRole.ToolTipRole)
            else:
                continue

        if select_experiment is not None:
            idx = self.review_dropdown.findData(select_experiment, Qt.ItemDataRole.UserRole)
            if idx >= 0:
                self.review_dropdown.setCurrentIndex(idx)
        elif select_newest and self.review_dropdown.count() > 0:
            self.review_dropdown.setCurrentIndex(0)
        elif preserve_folder is not None:
            idx = self.review_dropdown.findData(preserve_folder, Qt.ItemDataRole.UserRole)
            if idx >= 0:
                self.review_dropdown.setCurrentIndex(idx)

        if self.review_dropdown.count() > 0 and self.review_dropdown.currentIndex() < 0:
            self.review_dropdown.setCurrentIndex(0)

        self.current_experiment = self.review_dropdown.currentData(Qt.ItemDataRole.UserRole)
        self.review_dropdown.blockSignals(False)
        self._update_review_dropdown_tooltip()

    def refresh_review_dropdown_item(self, index):
        experiment_folder = self.review_dropdown.itemData(index, Qt.ItemDataRole.UserRole)
        if not experiment_folder:
            return
        exp_dir = Path(get_results_dir()) / experiment_folder
        exp_disp = build_experiment_display_name(experiment_folder, exp_dir)
        self.review_dropdown.setItemText(index, exp_disp)
        self.review_dropdown.setItemData(index, exp_disp, Qt.ItemDataRole.ToolTipRole)
        if index == self.review_dropdown.currentIndex():
            self._update_review_dropdown_tooltip()

    def _update_review_dropdown_tooltip(self):
        text = self.review_dropdown.currentText()
        self.review_dropdown.setToolTip(text if text else "")

    def edit_experiment_note(self):
        exp_dir = self._current_experiment_dir()
        if exp_dir is None:
            QMessageBox.warning(self, "Experiment Note", "No experiment selected.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Add Experiment Note")
        dlg_layout = QVBoxLayout(dlg)

        note_edit = QLineEdit()
        note_edit.setText(read_experiment_note(exp_dir))
        note_edit.setPlaceholderText("Enter a note for this experiment...")
        dlg_layout.addWidget(note_edit)

        def refresh_dropdown_note():
            idx = self.review_dropdown.currentIndex()
            if idx >= 0:
                self.review_dropdown.blockSignals(True)
                self.refresh_review_dropdown_item(idx)
                self.review_dropdown.blockSignals(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        clear_btn = buttons.addButton("Clear", QDialogButtonBox.ButtonRole.ResetRole)

        def clear_note():
            note_edit.clear()
            write_experiment_note(exp_dir, "")
            refresh_dropdown_note()

        clear_btn.clicked.connect(clear_note)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        dlg.adjustSize()
        dlg.setFixedHeight(dlg.height())
        dlg.setFixedWidth(420)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        write_experiment_note(exp_dir, note_edit.text())
        refresh_dropdown_note()

    def _current_experiment_dir(self):
        folder = None
        if hasattr(self, "review_dropdown"):
            folder = self.review_dropdown.currentData(Qt.ItemDataRole.UserRole)
        if not folder:
            folder = getattr(self, "current_experiment", None)
        if not folder:
            return None
        return Path(get_results_dir()) / folder

    def toggle_dropdown_display(self):
        self.review_select_widget.setVisible(self.reviewing_history and not self.confirming_detections)

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
                self.historical_label_changes = {}
            else:
                return  # cancel pressed            

        # Clean up generated files and folders
        if self.cleanup_trees:
            if hasattr(self, 'current_output_dir') and self.current_output_dir:
                try:
                    if os.path.exists(self.current_output_dir):
                        shutil.rmtree(self.current_output_dir)
                        logger.info(f"Cleaned up output directory: {self.current_output_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up output directory: {str(e)}")
        
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
        self.video_list.setEnabled(True)
        self.select_videos_button.setEnabled(True)
        self.drone_select.setEnabled(True)
        self.altitude_input.setEnabled(True)
        self.flight_location_input.setEnabled(True)
        self.update_remove_buttons()  # list is now empty: hides Remove All, disables Process

        self.confirming_detections = False
        self.reviewing_history = False
        self.edit_mode = False
        
        # Switch to home widget
        self.stack_widget.setCurrentWidget(self.home_widget)
        self.toggle_banner_buttons(review=False)
        
    def show_confidence_warning(self):
        # if self.reviewing_history:
        row = self.historical_items.currentRow()
        # When an experiment has no detections the table is empty, currentRow()
        # is -1 and item(-1, 4) is None -> guard so we don't crash building the
        # historical list (previously: "'NoneType' object has no attribute 'text'").
        conf_item = self.historical_items.item(row, 4) if row >= 0 else None
        if conf_item is None:
            self.low_confidence_warning.setVisible(False)
            return
        try:
            conf = float(conf_item.text())
        except (TypeError, ValueError):
            self.low_confidence_warning.setVisible(False)
            return
        self.low_confidence_warning.setVisible(conf < self.low_confidence_threshold)
        # else:
        #     _, track = self.sorted_tracks[self.current_detection_index]
        #     if track['longest_conf'] < self.low_confidence_threshold:
        #         self.low_confidence_warning.setVisible(True)
        #     else:
        #         self.low_confidence_warning.setVisible(False)

    def update_timer(self):
        if self.timer_label:
            self.elapsed_time += 1
            hours, remainder = divmod(self.elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def generate_filename(self, track, new_label):
        return f"{track['video_name']}_{new_label.lower()}{track['unique_id']}_time{track['time']}_det{track['detections']}_avgConf{int(track['avg_conf']*100):02d}_bestConf{int(track['best_conf']*100):02d}_len{track['length'].replace('ft', 'ft').replace('in', 'in')}.jpg"

    def update_frame_display(self, frame):
        # First frame of this video arrived — switch the busy bar back to determinate.
        if getattr(self, "_awaiting_first_frame", False):
            self._awaiting_first_frame = False
            if getattr(self, "progress_bar", None) is not None:
                self.progress_bar.setRange(0, 100)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.frame_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.frame_display.setPixmap(scaled_pixmap)
        self.frame_display.show()

    def closeEvent(self, event):
        if self.is_processing:
            event.ignore()
            self.confirm_cancel_processing()
            return

        release_sam_model()
        # Clean up generated files and folders
        if self.cleanup_trees:
            if hasattr(self, 'current_output_dir') and self.current_output_dir:
                try:
                    if os.path.exists(self.current_output_dir):
                        shutil.rmtree(self.current_output_dir)
                        logger.info(f"Cleaned up output directory: {self.current_output_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up output directory: {str(e)}")
            
        # Ensure threads are properly closed
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        event.accept()

    def update_video_order(self):
        # Update the internal order of videos after drag and drop
        self.video_queue = [self.video_list.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(self.video_list.rowCount())]

    def export_results(self):
        # if self.reviewing_history:
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
                    time_str = datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime('%M:%S')
                    
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

    def ensure_track_consistency(self):
        if len(self.tracks) != len(self.sorted_tracks):
            logger.warning("Warning: Inconsistency detected between tracks and sorted_tracks")
            self.tracks = dict(self.sorted_tracks)
        
        for key, track in self.sorted_tracks:
            if key not in self.tracks:
                logger.warning(f"Warning: Track {key} found in sorted_tracks but not in tracks")
                self.tracks[key] = track

        logger.info(f"Tracks consistency check complete. Total tracks: {len(self.tracks)}")

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
        csv_name = f"{Path(video_basename).name}.csv"

        return {
            "experiment": experiment_folder,
            "video_basename": video_basename,
            "csv_name": csv_name,
            "track_id": track_id,
        }

    def _save_historical_label_changes(self, allow_no_changes=False, confirm_always=False):
        """
        Persist queued label changes into their corresponding historical CSV files.
        Keys in self.historical_label_changes are (experiment, video_name, csv_name, track_id).
        Returns True on success (including no-op when allow_no_changes), False on cancel or failure.
        """
        if confirm_always:
            reply = QMessageBox.question(
                self,
                "Confirm Detections",
                "Save changes to experiment results?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False
            if not self.historical_label_changes:
                QMessageBox.information(
                    self,
                    "Changes Saved",
                    "All label changes were saved back to their CSV files."
                )
                return True
        elif not self.historical_label_changes:
            if allow_no_changes:
                return True
            QMessageBox.information(self, "No Changes", "There are no label changes to save.")
            return False

        if not confirm_always:
            reply = QMessageBox.question(
                self,
                "Save Changes",
                "Save changes to experiment results?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return False
        
        failures = []
        updated_files = set()
        experiments_with_changes = set()

        for key, new_label in list(self.historical_label_changes.items()):
            experiment, video_name, csv_name, track_id = key
            if new_label == "Delete":
                self.delete_track(experiment, csv_name, track_id)
                del self.historical_label_changes[key]
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
                    del self.historical_label_changes[key]
                else:
                    failures.append(f"Track {track_id} not found in {csv_path}")
                experiments_with_changes.add(Path(get_results_dir()) / experiment)
            except Exception as e:
                failures.append(f"{csv_name} (Track {track_id}): {e}")
        
        if str(self.settings_obj.value("enable_auto_upload").lower()) == "true":
            exps = list(experiments_with_changes)
            logger.info(f"[upload] Auto-upload enabled; {len(exps)} experiment(s) with changes to upload")
            if exps:
                dlg = QDialog(self)
                dlg.setWindowTitle("Upload in Progress")
                dlg.setModal(False)

                layout = QVBoxLayout()
                layout.addWidget(QLabel(f"Uploading {len(exps)} experiment{'s' * (len(exps) > 1)}"))
                dlg.setLayout(layout)
                dlg.show()

                # Run each upload on its own background QThread so the GUI stays
                # responsive, then wait for all of them via a local event loop
                # (which keeps the event loop pumping, unlike thread.wait()).
                errors = {}
                threads = []
                loop = QEventLoop()
                remaining = len(exps)

                def _on_upload_done(success, message, exp):
                    nonlocal remaining
                    if not success:
                        errors[exp] = message
                    remaining -= 1
                    logger.info(f"[upload] Finished '{Path(exp).name}': "
                          f"{'OK' if success else 'ERROR - ' + message}; {remaining} remaining")
                    if remaining == 0:
                        loop.quit()

                for exp in exps:
                    thread = UploadThread(api_url=self.api_url, experiment_dir=exp)
                    thread.upload_finished.connect(
                        lambda success, message, exp=exp: _on_upload_done(success, message, exp))
                    threads.append(thread)

                for thread in threads:
                    thread.start()

                loop.exec()

                for thread in threads:
                    thread.wait()

                logger.info(f"[upload] Auto-upload batch complete: "
                      f"{len(exps) - len(errors)} succeeded, {len(errors)} failed")

                dlg.hide()
                result_box = QMessageBox(self)
                result_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                if not errors:
                    result_box.setWindowTitle("Upload Complete")
                    result_box.setText(f"Successfully uploaded {len(exps)} experiment{'s' * (len(exps) > 1)}")
                else:
                    result_box.setWindowTitle("Upload Error")
                    message = "Error uploading the following:\n"
                    for exp, err in errors.items():
                        message += f"{exp}: {err}\n"
                    result_box.setText(message)
                result_box.exec()

        # Feedback
        if failures and updated_files:
            QMessageBox.warning(
                self,
                "Partial Save",
                "Some changes were saved, but a few failed:\n\n" + "\n".join(failures[:10])
                + ("\n..." if len(failures) > 10 else "")
            )
            return False
        elif failures:
            QMessageBox.critical(
                self,
                "Save Failed",
                "Could not save changes:\n\n" + "\n".join(failures[:15])
                + ("\n..." if len(failures) > 15 else "")
            )
            return False

        if confirm_always:
            QMessageBox.information(
                self,
                "Changes Saved",
                "All label changes were saved back to their CSV files."
            )
            self.update_detection_list()
        elif not allow_no_changes:
            QMessageBox.information(
                self,
                "Changes Saved",
                "All label changes were saved back to their CSV files."
            )
            self.update_detection_list()
            self._set_edit_state(False)
        elif updated_files:
            self.update_detection_list()
        return True
    def auto_upload_experiments(self):
        pass
        
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
        zip_name = f'{Path(self.experiment_dir).name}.zip'
        logger.info(f"[upload] Zipping experiment '{self.experiment_dir}' -> {zip_name}")
        try:
            file_count = 0
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
                                file_count += 1
                    else:
                        logger.warning(f"[upload]   skipping missing folder: {folder}")

            zip_size = buffer.tell()
            buffer.seek(0)
            logger.info(f"[upload] {zip_name}: {file_count} file(s), {zip_size / 1024:.1f} KB; "
                  f"POST -> {self.api_url}")
            files = {'file': (zip_name, buffer, 'application/zip')}
            response = requests.post(self.api_url, files=files)
            response.raise_for_status()

            logger.info(f"[upload] {zip_name}: SUCCESS (HTTP {response.status_code})")
            self.upload_finished.emit(True, "Folder uploaded successfully")
        except requests.RequestException as e:
            logger.error(f"[upload] {zip_name}: FAILED (request error): {e}")
            self.upload_finished.emit(False, "Upload failed: {}".format(str(e)))
        except Exception as e:
            logger.error(f"[upload] {zip_name}: FAILED (unexpected error): {e}")
            self.upload_finished.emit(False, "An unexpected error occurred: {}".format(str(e)))

    # def run(self):
    #     try:
    #         buffer = io.BytesIO()
    #         with zipfile.ZipFile(buffer, 'w') as zipf:
    #             for folder in ['bounding_boxes', 'detection_results', 'false_positives', 'frames', 'masks']:
    #                 folder_path = os.path.join(self.experiment_dir, folder)
    #                 if os.path.exists(folder_path):
    #                     for root, _, files in os.walk(folder_path):
    #                         for file in files:
    #                             file_path = os.path.join(root, file)
    #                             arcname = os.path.relpath(file_path, self.experiment_dir)
    #                             zipf.write(file_path, arcname)

    #         buffer.seek(0)
    #         files = {'file': ('upload.zip', buffer, 'application/zip')}
    #         response = requests.post(self.api_url, files=files)
    #         response.raise_for_status()

    #         self.upload_finished.emit(True, "Folder uploaded successfully")
    #     except requests.RequestException as e:
    #         self.upload_finished.emit(False, "Upload failed: {}".format(str(e)))
    #     except Exception as e:
    #         self.upload_finished.emit(False, "An unexpected error occurred: {}".format(str(e)))

class VersionCheckThread(QThread):
    """Ask the Cloud Function whether the running build is the latest for its OS.

    Runs off the UI thread so a slow/unreachable network never delays app launch.
    The endpoint returns a JSON boolean: true == up to date, false == update available.
    """
    # (update_available, os_key, error_message)
    check_finished = pyqtSignal(bool, str, str)

    def __init__(self, endpoint, os_key, commit):
        super().__init__()
        self.endpoint = endpoint
        self.os_key = os_key
        self.commit = commit

    def run(self):
        try:
            response = requests.get(
                self.endpoint,
                params={
                    "request": "check_version",
                    "user_os": self.os_key,
                    "latest_commit": self.commit,
                },
                timeout=10,
            )
            response.raise_for_status()
            up_to_date = bool(response.json())
            self.check_finished.emit(not up_to_date, self.os_key, "")
        except Exception as e:
            self.check_finished.emit(False, self.os_key, str(e))


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    QApplication.quit()

class FramePlayer(QLabel):
    resized = pyqtSignal()
    # (current index, total frames) — drives the review scrubber and frame counter.
    frame_changed = pyqtSignal(int, int)
    # "frames" | "movie" | "static" | "empty" — tells the review controls which of the
    # three playback backends is live so they can enable/disable themselves.
    playback_mode_changed = pyqtSignal(str)

    # Clips are written at 10 fps by encode_track_clips(), so one frame per 100 ms is 1x.
    BASE_INTERVAL_MS = 100

    def __init__(self, settings_obj=None, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
        self._movie = None
        self.setScaledContents(False)
        # Fill the parent stack like ZoomableFrameView; image fits inside via KeepAspectRatio.
        # Do not inherit QLabel's pixmap/movie sizeHint — native drone frames are huge and
        # would force the window wider than the screen once frame_stack is height-capped.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frames = []
        self.current_frame = 0
        self._speed = 1.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(self.BASE_INTERVAL_MS)
        self._static_pixmap = None

    # --- playback control -------------------------------------------------
    # Two backends sit behind these: a QTimer stepping self.frames (MP4 clips, the
    # current format) and a QMovie (legacy .gif clips). Both must honour every call.

    def set_speed(self, multiplier):
        """Set playback rate as a multiple of the clip's native 10 fps."""
        multiplier = float(multiplier) or 1.0
        self._speed = multiplier
        self.timer.setInterval(max(1, int(round(self.BASE_INTERVAL_MS / multiplier))))
        if self._movie:
            self._movie.setSpeed(int(round(multiplier * 100)))

    def is_playing(self):
        if self._movie:
            return self._movie.state() == QMovie.MovieState.Running
        return self.timer.isActive()

    def play(self):
        if self._movie:
            self._movie.setPaused(False)
        elif self.frames:
            self.timer.start()

    def pause(self):
        if self._movie:
            self._movie.setPaused(True)
        else:
            self.timer.stop()

    def toggle_play_pause(self):
        self.pause() if self.is_playing() else self.play()

    def seek(self, index):
        """Jump to a frame without changing play/pause state (frame-list mode only)."""
        if not self.frames:
            return
        self.current_frame = max(0, min(int(index), len(self.frames) - 1))
        self.show_frame(self.current_frame)

    def current_frame_pixmap(self):
        """Return the currently displayed clip frame as a full-resolution QPixmap.

        Frame-list ("frames") mode only — the frames are the native-resolution clip
        frames, so a line drawn on one measures at the original scale. Returns None when
        no such frame is available (empty player, movie/static backend).
        """
        if not self.frames or not (0 <= self.current_frame < len(self.frames)):
            return None
        frame = self.frames[self.current_frame]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        # .copy() detaches the QImage from the transient numpy buffer before it's freed.
        return QPixmap.fromImage(qimg.copy())

    def sizeHint(self):
        # Prefer filling the stack, not growing the window to native clip resolution.
        return QSize(640, 360)

    def minimumSizeHint(self):
        return QSize(0, 0)

    def _clear_height_constraints(self):
        """Ensure the player can expand to the full stack (no leftover fixed height)."""
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)

    def _detach_movie(self):
        if self._movie:
            on_frame = getattr(self, "_on_movie_frame", None)
            if on_frame is not None:
                try:
                    self._movie.frameChanged.disconnect(on_frame)
                except TypeError:
                    pass
            self._movie.stop()
            self._movie = None
        # Avoid QLabel::setMovie sizeHint = native frame size.
        self.setMovie(None)

    def set_frames(self, frames):
        self._clear_height_constraints()
        self.frames = frames
        self.current_frame = 0
        if frames:
            self.show_frame(0)
            self.set_speed(self._speed)  # re-apply the interval for the new clip
            self.timer.start()
            self.playback_mode_changed.emit("frames")
        else:
            self.clear()
            self.timer.stop()
            self.playback_mode_changed.emit("empty")
        self.resized.emit()

    def show_frame(self, index):
        if 0 <= index < len(self.frames):
            self.current_frame = index
            # Painting fits KeepAspectRatio to the current widget size (like ZoomableFrameView).
            self.update()
            self.frame_changed.emit(index, len(self.frames))

    def next_frame(self):
        if not self.frames:
            self.timer.stop()
            return
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
        self._detach_movie()
        self.timer.stop()
        self.frames = []
        self._clear_height_constraints()
        self._static_pixmap = pixmap
        self.clear()  # drop any QLabel pixmap/text so sizeHint stays small
        self.update()
        self.playback_mode_changed.emit("static")
        self.resized.emit()

    def set_video(self, path: str):
        """Play an MP4 clip by decoding it into frames and animating them via the
        frame timer. QMovie can't decode MP4, so this is the MP4 counterpart to
        set_gif(); it reuses the existing set_frames() playback path."""
        self._detach_movie()
        self._static_pixmap = None
        self.timer.stop()
        self.frames = []
        self._clear_height_constraints()
        self.clear()

        frames = []
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            self.clear()
            self.setText(f"Clip not found or empty:\n{os.path.basename(path)}")
            self.playback_mode_changed.emit("empty")
            return

        # Very short clips look jittery animated; show the center frame instead
        # (mirrors set_gif's playback_min_frames behavior).
        playback_min_frames = int(self.settings_obj.value("playback_min_frames"))
        if 0 < len(frames) < playback_min_frames:
            mid = frames[len(frames) // 2]
            frame_rgb = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            q_image = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.set_static_pixmap(QPixmap.fromImage(q_image))
            return

        self.set_frames(frames)

    def set_gif(self, path: str):
        self._static_pixmap = None
        self._clear_height_constraints()
        self._detach_movie()
        self.clear()

        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        frame_count = movie.frameCount()

        # Very short animations look jittery; show only the center frame instead.
        playback_min_frames = int(self.settings_obj.value("playback_min_frames"))
        if 0 < frame_count < playback_min_frames:
            middle_index = frame_count // 2
            if movie.jumpToFrame(middle_index):
                self.set_static_pixmap(movie.currentPixmap())
                return

        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        self._movie = movie
        # Paint via paintEvent — do not setMovie on QLabel (native size blows out window width).
        self._movie.frameChanged.connect(self._on_movie_frame)
        self._movie.setSpeed(int(round(self._speed * 100)))
        self._movie.start()
        self._movie.finished.connect(lambda: self._movie.start())
        self.update()
        self.playback_mode_changed.emit("movie")
        self.resized.emit()

    def _on_movie_frame(self, _frame_number=0):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        widget_size = self.size()

        # Same as ZoomableFrameView: KeepAspectRatio, centered in the full widget.
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

        # 3) Frame-sequence Mode (MP4 path) — paint from native frames so resize
        #    refits like the editor instead of using a stale pre-scaled QLabel pixmap.
        if self.frames:
            frame = self.frames[self.current_frame]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            q_image = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.size().scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
            x = (widget_size.width() - scaled.width()) // 2
            y = (widget_size.height() - scaled.height()) // 2
            painter.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), pixmap)
            return

        # 4) Fallback
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Do not setFixedHeight — fill the stack; content_rect / paint handle fit.
        self.resized.emit()
        self.update()

    def clear_frame(self):
        self._static_pixmap = None
        self._detach_movie()
        self.timer.stop()
        self.frames = []
        self._clear_height_constraints()
        self.clear()
        self.update()
        self.playback_mode_changed.emit("empty")
    
    def content_rect(self):
        """
        Returns the QRect of the actually displayed pixmap/movie/frame
        inside the widget (KeepAspectRatio, centered), matching paintEvent.
        """
        frame_size = None

        if self._static_pixmap and not self._static_pixmap.isNull():
            frame_size = self._static_pixmap.size()
        elif self._movie:
            frame = self._movie.currentPixmap()
            if not frame.isNull():
                frame_size = frame.size()
        elif self.frames:
            height, width = self.frames[0].shape[:2]
            frame_size = QSize(width, height)
        else:
            pixmap = self.pixmap()
            if pixmap is not None and not pixmap.isNull():
                frame_size = pixmap.size()

        if frame_size is None or frame_size.isEmpty():
            return QRect()

        widget_size = self.size()
        scaled = frame_size.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
        x = (widget_size.width() - scaled.width()) // 2
        y = (widget_size.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

class HeadlessVideoProcessor(VideoProcessingWorker):
    def __init__(self, video_path, model, output_dir):
        self.settings_obj = QSettings("BOSL", "SharkEye_App")
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.detection_threshold = float(self.settings_obj.value("confidence_threshold", "0.40"))
        self.drone_settings = json.loads(self.settings_obj.value("drone_settings"))
    
    progress_update = 0
    processing_complete = {}

    @staticmethod
    def save_best_frames(output_dir, video_path, tracks, tracker=None):
        """Save best frames for each significant track"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Same significance thresholds the tracker uses (static method has no self).
        settings_obj = QSettings("BOSL", "SharkEye_App")
        min_frames = int(settings_obj.value("min_frames"))
        confidence_threshold = float(settings_obj.value("confidence_threshold"))

        images_saved = 0

        for track_id, track in tracks.items():
            if tracker and not tracker.is_significant_track(track):
                continue

            logger.debug("Starting new track")
            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])

            longest_frame = track['longest_frame']
            longest_confidence = track['longest_conf']
            longest_length = track['longest_length']

            if longest_frame is None:
                continue

            x, y, w, h = track['positions'][track['confidences'].index(longest_confidence)]

            # SAM is the expensive per-track step; run it only on significant tracks.
            is_significant = num_frames >= min_frames and avg_confidence > confidence_threshold
            if is_significant:
                # Use segmentation model to generate lengths
                mask = run_prediction(longest_frame, (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
                pixel_length = find_pixel_length(mask, draw_line=False, viz_name = f'{video_name}-viz')

                track['longest_length'] = pixel_length
                longest_length = track['longest_length']

                mask_overlay = draw_mask(mask, longest_frame)
                track['mask_overlay'] = mask_overlay

            length_str = f"{longest_length:.2f}px"

            filename = f"{Path(video_path).name}_{track_id}.jpg"

            # Save original frame + bounding-box frame for every track.
            cv2.imwrite(os.path.join(output_dir, 'frames', filename), longest_frame)

            boxed_frame = longest_frame.copy()
            cv2.rectangle(boxed_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            label = f"ID: {track_id}, Conf: {longest_confidence:.2f}, Length: {length_str}"
            cv2.putText(boxed_frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bounding_box_path = os.path.join(output_dir, 'bounding_boxes', filename)
            cv2.imwrite(bounding_box_path, boxed_frame)

            # Mask only exists for segmented (significant) tracks.
            if is_significant:
                mask_path = os.path.join(output_dir, 'masks', filename)
                cv2.imwrite(mask_path, mask_overlay)

            # Update the track with the path to the bounding box image
            track['image_path'] = bounding_box_path

            images_saved += 1

        logger.info(f"Shark Images Saved: {images_saved}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        custom_tracker = CustomTracker()
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'bounding_boxes'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'false_positives'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'detection_results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "tracking_gifs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

        # Sequential forward sampling. Keyframe-scan when enabled + decodable, else
        # grab-through (see the main worker for the rationale). No preview is emitted
        # here, so no color conversion is done.
        sampler = try_keyframe_sampler(self.video_path, logger)
        use_keyframe = sampler is not None
        if use_keyframe:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
        else:
            sampler = iter_sampled_frames(cap)
        had_detection = None
        try:
            while True:
                frame_num, frame = sampler.send(had_detection)

                results = self.model(frame, classes=[0], verbose=False)
                detections = parse_detections(results, self.detection_threshold)
                had_detection = bool(detections)

                if had_detection:
                    timestamp = (frame_num / fps * 1000.0) if use_keyframe else cap.get(cv2.CAP_PROP_POS_MSEC)
                    custom_tracker.update(detections, frame, timestamp)

                self.progress_update = int((frame_num + 1) / total_frames * 100)
        except StopIteration:
            pass

        cap.release()
        significant_tracks = custom_tracker.get_significant_tracks()
        self.save_best_frames(
            self.output_dir,
            self.video_path,
            tracks=significant_tracks,
            tracker=custom_tracker,
        )

        all_track_info = []

        for track_id, track in significant_tracks.items():
            meets_thresholds = custom_tracker.is_significant_track(track)
    
            track_info = {
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
                'manual_length_px': '',
                'manual_length_ft': '',
            }

            all_track_info.append(track_info)
    
        return all_track_info            

def mass_prediction(video_path, current_output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
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
    parser.add_argument('--testing', action='store_true', help='Enables testing for app in headless environment')
    parser.add_argument('--input_dir', type=str, required=False, help='Directory containing .mp4 videos to process')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
    return parser.parse_args()

def _install_qt_message_filter():
    """Route Qt's own log messages through Python, dropping one known-benign
    warning that clutters startup logs on macOS.

    During UI construction Qt emits 'QImage::QImage(), XPM is not supported'
    (from a bundled icon / the third-party SwitchControl widget). It has no
    functional effect -- the widgets still render -- so we suppress just that
    line and forward every other Qt message to stderr unchanged.
    """
    def handler(msg_type, context, message):
        if "XPM is not supported" in message:
            return
        print(message, file=sys.stderr)
    qInstallMessageHandler(handler)


if __name__ == '__main__':
    _install_qt_message_filter()
    args = parse_args()
    if args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        video_paths = input_dir.rglob("*.mp4")
        if not video_paths:
            logger.warning(f"No .mp4 videos found in {input_dir}")
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
            logger.info(f"Results saved to {csv_path}")
        else:
            logger.warning("No valid tracks were found.")
    else:
        if args.testing:
            os.environ["QT_DEBUG_PLUGINS"] = "1"
            os.environ["QT_QPA_PLATFORM"] = "minimal"
        # freeze_support() already handled at import time (top of file).
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        apply_theme(app)

        app_icon_path = {
            'win32': 'assets/logo/SharkEye.ico',
            'darwin': 'assets/logo/SharkEye.icns'
        }.get(sys.platform, 'assets/logo/SharkEye.iconset/icon_32x32.png')

        app.setWindowIcon(QIcon(resource_path(app_icon_path)))

        window = MainWindow()
        window.show()
        sys.exit(app.exec())
