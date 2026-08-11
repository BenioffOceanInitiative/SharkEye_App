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
from PyQt6.QtGui import QImage, QPixmap, QColor, QIcon, QDoubleValidator, QIntValidator, QMovie, QPainter, QPen, QPalette, QDesktopServices, QShortcut, QKeySequence  # TODO: remove QPalette — unused (moved to theme.py)
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
from utility import (
    resource_path,
    get_results_dir,
    select_torch_device,
    get_writable_docs_dir,
    resolve_help_guide_path,
    read_local_doc_version,
    write_local_doc_version,
    local_help_docs_present,
)
from log_config import get_logger, install_crash_handlers

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

# Windowed (console=False) PyInstaller builds set stdout/stderr to None on Windows.
# tqdm.write/print then raise AttributeError ('NoneType' has no attribute 'write').
def _ensure_stdio():
    if sys.stdout is None or sys.stderr is None:
        try:
            log_path = os.path.join(tempfile.gettempdir(), "sharkeye_console.log")
            log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            log_fh = io.StringIO()
        if sys.stdout is None:
            sys.stdout = log_fh
        if sys.stderr is None:
            sys.stderr = log_fh

_ensure_stdio()

# Add these constants for length calculation
# Length calibration + CustomTracker now live in the shared `tracking` module so the GUI,
# the mass_prediction batch, and the headless_prediction CLI share ONE implementation
# (previously duplicated and drifted). Re-exported into this namespace so existing
# references (CustomTracker, GSD, ORIGINAL_WIDTH, ...) keep working unchanged.
from tracking import (
    CustomTracker, calculate_gsd, GSD, calculate_shark_length, calculate_bbox_area,
    calculate_adjusted_shark_length, DRONE_ALTITUDE_M, SENSOR_WIDTH_MM, FOCAL_LENGTH_MM,
    MODEL_WIDTH, MODEL_HEIGHT, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, ASPECT_RATIO,
    DEFAULT_DRONE_SETTINGS, resolve_fov_radians,
)

# Use a constant for the model path
MODEL_PATH = resource_path('model_weights/runs-detect-train-weights-best.pt')


# Frame sampling / detection parsing (shared with the headless processors).
from frame_sampling import (iter_sampled_frames, parse_detections, downscale_for_preview,
                            format_sampling_stats, format_sampling_timeline)
try:
    # Keyframe-scan sampling needs PyAV; keep it optional so the app still runs (on
    # grab-through) if PyAV is unavailable. It is on by default; try_keyframe_sampler
    # validates each file and returns None on any problem, falling back to grab-through.
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

# --- YOLO annotation format ---------------------------------------------------
# Detections are stored on disk as YOLO — the single format for the review overlay,
# the upload, and retraining. Each sampled frame gets a `frame_<NNNN>.txt` with a
# line "class cx cy w h" (normalized 0-1, so it survives the <=1080p downscale at
# upload), alongside a `meta.json` holding the per-frame data YOLO can't carry
# (confidence, length, timestamp, length-source frame). The class map is fixed:
# Shark -> 0, Kelp -> 1, everything else -> 2.
YOLO_CLASS_NAMES = ["shark", "kelp", "other"]


def label_to_yolo_class(label):
    """Map a review label to its YOLO class id (Shark->0, Kelp->1, everything else->2)."""
    key = (label or "").strip().lower()
    if key == "shark":
        return 0
    if key == "kelp":
        return 1
    return 2

# DEFAULT_DRONE_SETTINGS now lives in `tracking` (imported above) so the GUI and the
# headless CLIs resolve the same per-video FOV.


def ensure_app_settings(settings_obj=None):
    """Seed QSettings defaults when missing (fresh CI runners, first launch, etc.)."""
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    if not settings_obj.value("drone_settings"):
        settings_obj.setValue("drone_settings", json.dumps(DEFAULT_DRONE_SETTINGS))
    if not settings_obj.value("confidence_threshold"):
        settings_obj.setValue("confidence_threshold", "0.40")
    if not settings_obj.value("min_frames"):
        settings_obj.setValue("min_frames", "5")
    if not settings_obj.value("playback_min_frames"):
        settings_obj.setValue("playback_min_frames", "5")
    if not settings_obj.value("playback_speed"):
        settings_obj.setValue("playback_speed", str(DEFAULT_PLAYBACK_SPEED))
    if not settings_obj.value("enable_auto_upload"):
        settings_obj.setValue("enable_auto_upload", "false")
    if not settings_obj.value("ignore_update"):
        settings_obj.setValue("ignore_update", "false")
    if not settings_obj.value("detection_labels"):
        save_detection_labels(settings_obj, list(DEFAULT_DETECTION_LABELS))
    return settings_obj


def get_drone_settings_dict(settings_obj=None):
    """Return the drone settings dict, seeding defaults if needed."""
    settings_obj = ensure_app_settings(settings_obj)
    return json.loads(settings_obj.value("drone_settings"))


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
# model download uses). `?request=check_version` compares commits; `?request=check_docs`
# syncs help docs in-app; the default request (`?user_os=<os>`) redirects to a signed
# URL for the latest build.
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


# calculate_gsd / GSD / calculate_shark_length / calculate_bbox_area /
# calculate_adjusted_shark_length now live in `tracking` (imported near the top of this
# file) so the GUI, mass_prediction, and headless CLI share one length calibration.

def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)

    # Get total number of frames and frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate duration in seconds
    duration = frame_count / fps
    video.release()
    return duration


_libav_check_done = False


def warn_on_libav_collision():
    """Best-effort: warn once if OpenCV and PyAV bundle mismatched libav builds.

    cv2 and PyAV each ship their own ``libav*`` dylibs; when their major versions differ,
    duplicate native symbols can cause crashes — the ``objc[...] implemented in both`` lines
    at startup are exactly this condition. Benign in dev, a real hazard in the frozen build
    (a native abort leaves nothing but a crash code). This records the mismatch in the log
    instead of leaving it buried in objc noise. Idempotent; never raises.
    """
    global _libav_check_done
    if _libav_check_done:
        return
    _libav_check_done = True
    try:
        import re
        import av
        av_ver = av.library_versions.get('libavcodec')
        av_major = av_ver[0] if isinstance(av_ver, (tuple, list)) else None
        m = re.search(r'avcodec[^0-9]*([0-9]+)\.', cv2.getBuildInformation())
        cv_major = int(m.group(1)) if m else None
        if av_major and cv_major and av_major != cv_major:
            logger.warning("[env] OpenCV (libavcodec %s) and PyAV (libavcodec %s) bundle different "
                           "libav builds; duplicate native symbols can crash the frozen build. "
                           "Consider excluding one libav copy in SharkEye.spec.", cv_major, av_major)
    except Exception:
        pass  # a diagnostic check must never break startup

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
                # Bake the reviewer's corrected labels into the YOLO class column first.
                refresh_yolo_labels_from_csv(experiment_dir)
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zipf:
                    # 'shark_frames' = every sampled frame per shark + YOLO labels; images
                    # are downscaled to <=1080p to keep the upload under the size limit.
                    file_count = add_experiment_to_zip(zipf, experiment_dir)

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


def _downscale_frame_to_fit(frame, max_w, max_h):
    """Resize ``frame`` down to fit within ``(max_w, max_h)`` keeping aspect; return it
    unchanged if already within bounds.

    INTER_LINEAR, not INTER_AREA: on a real 5.3K drone frame the whole "resize + write a
    1080p JPG" costs ~10ms with LINEAR vs ~34ms with AREA (and ~58ms writing the full-res
    JPG). These are the training/upload frames feeding a 640-input model, so 1080p LINEAR
    is far more resolution than needed and the aliasing difference is irrelevant."""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def encode_track_clips(payload, output_dir, video_name, annotation_color,
                       box_thickness, text_thickness, text_scale, fps=10):
    """Persist per-track review/upload artifacts from a self-contained payload.

    For each track this writes three things, all from the same in-memory frames:
      * tracking_gifs/<video_name>_<key>.mp4 — a RAW clip (no baked bounding box). The
        review player (FramePlayer) draws the box as a live, toggleable/recolorable
        overlay from the sidecar below, so it must not be burned into the pixels.
      * shark_frames/<video_name>_<key>/frame_<NNNN>.jpg — every sampled frame of the
        shark at full resolution, for upload ("every frame per shark").
      * shark_frames/<video_name>_<key>/frame_<NNNN>.txt — a YOLO label per frame
        (class cx cy w h, normalized), parallel to the JPG sequence. Class is 0 (shark)
        here; the reviewer's corrected label is baked in at upload time (see
        refresh_yolo_labels_from_csv). This is the upload / retraining annotation.
      * shark_frames/<video_name>_<key>/meta.json — the per-frame metadata YOLO can't
        carry (confidence, length, timestamp) plus the length-source (longest) frame
        index. Drives the review overlay's confidence / length-source display.

    MP4 encoding via cv2.VideoWriter is C-level and releases the GIL, so it doesn't
    starve the concurrent inference thread. `payload` maps track key -> {'frames',
    'positions', 'lengths', 'confidences', 'timestamps', 'longest_timestamp'}; the
    payload owns its own frame buffers (no shared state with the UI's track dicts).
    The `annotation_*` args are retained for signature stability but no longer used to
    draw — the box is an overlay now, not baked in.
    """
    clips_dir = os.path.join(output_dir, "tracking_gifs")
    os.makedirs(clips_dir, exist_ok=True)
    frames_root = os.path.join(output_dir, "shark_frames")
    os.makedirs(frames_root, exist_ok=True)
    # Class-name manifest for the YOLO dataset rooted at shark_frames/.
    with open(os.path.join(frames_root, "classes.txt"), "w") as f:
        f.write("\n".join(YOLO_CLASS_NAMES) + "\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for key, track in payload.items():
        positions = track.get('positions') or []
        frames = track.get('frames') or []
        confidences = track.get('confidences') or []
        lengths = track.get('lengths') or []
        timestamps = track.get('timestamps') or []
        longest_ts = track.get('longest_timestamp')

        writer = None
        frame_size = None   # the CLIP/JPG/meta size (downscaled to <=1080p)
        orig_size = None    # the source frame size, for normalizing box coords
        clip_path = os.path.join(clips_dir, f"{video_name}_{key}.mp4")
        track_frames_dir = os.path.join(frames_root, f"{video_name}_{key}")
        os.makedirs(track_frames_dir, exist_ok=True)

        meta_frames = []      # per-frame metadata (aligned with the clip/JPGs/.txt files)
        longest_index = None  # which written frame is the length-source frame
        seq = 0
        try:
            for frame_idx, frame in enumerate(frames):
                if frame is None:
                    continue

                if writer is None:
                    oh, ow = frame.shape[:2]
                    orig_size = (ow, oh)
                    # The clip + per-frame JPGs are written at <=1080p: the review player
                    # decodes the whole clip on the UI thread (5.3K was slow to load) and
                    # the uploader downscales to 1080p anyway. Writing full 5.3K here was
                    # the dominant cost of this background stage (~58ms JPG + ~38ms MP4 per
                    # frame vs ~10ms + a few ms at 1080p). meta.json records the CLIP dims,
                    # so the review overlay maps the normalized boxes onto it correctly.
                    cw, ch = _downscale_frame_to_fit(frame, UPLOAD_IMAGE_MAX_W,
                                                     UPLOAD_IMAGE_MAX_H).shape[1::-1]
                    frame_size = (cw, ch)
                    writer = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
                    if not writer.isOpened():
                        logger.error(f"Could not open video writer for {clip_path}; skipping track {key}")
                        writer = None
                        break

                # One downscale serves both the clip and the JPG.
                small = _downscale_frame_to_fit(frame, UPLOAD_IMAGE_MAX_W, UPLOAD_IMAGE_MAX_H)
                if (small.shape[1], small.shape[0]) != frame_size:
                    small = cv2.resize(small, frame_size)   # guard a stray odd-sized frame
                writer.write(small)
                cv2.imwrite(os.path.join(track_frames_dir, f"frame_{seq:04d}.jpg"), small)

                pos = positions[frame_idx] if frame_idx < len(positions) else None
                conf = confidences[frame_idx] if frame_idx < len(confidences) else None
                length = lengths[frame_idx] if frame_idx < len(lengths) else None
                t_ms = timestamps[frame_idx] if frame_idx < len(timestamps) else None

                # YOLO label, parallel to frame_<seq>.jpg. Normalized by the ORIGINAL
                # source dims (pos is in source pixels); normalized coords are scale-free,
                # so they map correctly onto the downscaled JPG/clip and survive the upload
                # downscale. Class is 0 (shark) now; refresh_yolo_labels_from_csv() rewrites
                # it from the reviewer's corrected label. An empty file = a negative frame.
                label_path = os.path.join(track_frames_dir, f"frame_{seq:04d}.txt")
                if pos is not None and orig_size:
                    x, y, w, h = pos
                    ow, oh = orig_size
                    with open(label_path, "w") as lf:
                        lf.write(f"0 {x / ow:.6f} {y / oh:.6f} {w / ow:.6f} {h / oh:.6f}\n")
                else:
                    open(label_path, "w").close()

                meta_frames.append(None if pos is None else {
                    'conf': (float(conf) if conf is not None else None),
                    'length': (float(length) if length is not None else None),
                    't_ms': (float(t_ms) if t_ms is not None else None),
                })

                if (longest_index is None and longest_ts is not None
                        and t_ms is not None and abs(t_ms - longest_ts) < 1e-6):
                    longest_index = seq
                seq += 1

            if writer is not None:
                writer.release()
                logger.info(f"Saved clip: {clip_path}")

            meta = {
                'video': video_name, 'track_id': key, 'fps': fps,
                # The .txt coords are normalized; these are the CLIP/JPG dims (<=1080p) the
                # review overlay multiplies them by to map boxes onto the clip it plays.
                'frame_width': (frame_size[0] if frame_size else None),
                'frame_height': (frame_size[1] if frame_size else None),
                'longest_index': longest_index,
                'class_names': YOLO_CLASS_NAMES,
                'frames': meta_frames,
            }
            with open(os.path.join(track_frames_dir, "meta.json"), 'w') as f:
                json.dump(meta, f)
            logger.info(f"[frames] saved {seq} frame(s) + YOLO labels -> {track_frames_dir}")
        except Exception as e:
            logger.error(f"Clip/frame export failed for track {key}: {e}")
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

    # Same class scheme as the shark_frames YOLO labels (Shark->0, Kelp->1, else->2).
    category_names = list(YOLO_CLASS_NAMES)
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
    frame_processed = pyqtSignal(QImage)  # owned image — do not queue raw ndarray across threads
    progress_status_changed = pyqtSignal(str)  # current process summary for MainWindow.progress_status
    postproc_ready = pyqtSignal(dict, str, str)  # (payload, output_dir, video_name) for async export + clip encoding
    video_timing_ready = pyqtSignal(dict)  # per-video phase timing for the batch summary

    def __init__(self, video_path, model, output_dir, drone_type, altitude, flight_location):
        super().__init__()
        # Read settings 
        self.settings_obj = ensure_app_settings()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.drone_type = drone_type
        self.altitude = altitude
        self.flight_location = flight_location
        self.detection_threshold = float(self.settings_obj.value("confidence_threshold", "0.40"))
        self.drone_settings = get_drone_settings_dict(self.settings_obj)
        
    def run(self):
        # File log for frozen builds (console=False) so inference crashes are diagnosable.
        _log_path = os.path.join(tempfile.gettempdir(), "sharkeye_infer.log")

        def _log(msg):
            try:
                with open(_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            except Exception:
                pass

        try:
            self._run_inference(_log)
        except Exception:
            import traceback
            _log("FATAL:\n" + traceback.format_exc())
            # Also send the full traceback to the durable log so a user can retrieve
            # it after a crash without digging the temp file out of the OS temp dir.
            logger.exception("Inference worker crashed on %s",
                             os.path.basename(self.video_path))
            # Don't re-raise: an uncaught worker exception can abort the frozen
            # Qt process (BEX64 / 0xc0000409). Let the UI tear down cleanly.
            try:
                self.progress_status_changed.emit("")
                self.processing_complete.emit({}, os.path.basename(self.video_path))
            except Exception:
                pass

    def _run_inference(self, _log):
        self.progress_status_changed.emit("Running Inference")
        _log(f"start video={self.video_path}")
        # Re-bind the model on this worker thread. It was loaded/warmed on ModelLoader's
        # QThread; sharing a CUDA module across QThreads is a common frozen-Windows abort.
        try:
            device = select_torch_device()
            self.model.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _log(f"model on {device}")
        except Exception as e:
            _log(f"model rebind failed: {e}")

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Read fps up front: it's used in the log line below and, in keyframe mode, to
        # derive per-frame timestamps. `or 30` guards the occasional container that
        # reports 0 fps. (grab-through mode reads POS_MSEC directly and ignores this.)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        custom_tracker = CustomTracker()
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log(f"opened {video_width}x{video_height} frames={total_frames} fps={fps}")

        custom_tracker.fov_radians = self.drone_settings[self.drone_type]["Resolution"][f"({video_width}, {video_height})"]
        custom_tracker.drone_altitude = self.altitude

        # Calibration provenance: every length number derives from these inputs, but they
        # were never echoed, so a wrong altitude/FOV (or the module-level GSD the bbox
        # estimator uses) was invisible in the log. Print once per video.
        logger.info(f"[gsd] {Path(self.video_path).name}: drone={self.drone_type!r} "
                    f"altitude={custom_tracker.drone_altitude}m fov={custom_tracker.fov_radians:.4f}rad "
                    f"resolution={video_width}x{video_height} | module_GSD={GSD:.5f}m/px (bbox estimator)")

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
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
        # Split decode further by what the sampler was doing: scanning empty water
        # (keyframe-only skip, cheap) vs. densely stepping through a shark region (every
        # frame, expensive). The feedback value `had_detection` fed into send() is the
        # signal — True means we're in/entering a dense window. Answers "how much time is
        # spent skipping empty water vs. actually processing sharks."
        decode_scan_time = 0.0
        decode_dense_time = 0.0
        scan_frames = 0
        dense_frames = 0
        # Per-frame samples (ms) for p50/p95 — a mean hides tail stalls (e.g. the decoder
        # flush after a keyframe seek). Bounded: a few hundred floats per video.
        scan_decode_ms = []
        dense_decode_ms = []
        yolo_ms = []

        # Live preview is a courtesy view, not the output. Cap empty-frame updates at
        # ~20 fps so we don't color-convert, copy across the thread boundary, and
        # re-scale a pixmap on the UI thread faster than a human can see. Detection
        # frames always emit so no detection flashes by unseen.
        preview_interval = 1.0 / 20.0
        last_preview = 0.0

        # Sequential forward sampling. Default (unless SHARKEYE_KEYFRAME_SAMPLING=0):
        # when the file decodes cleanly, use the keyframe-scan sampler; otherwise fall
        # back to grab-through with an adaptive stride.
        # The keyframe-scan sampler decodes only keyframes over empty water, goes dense around
        # detections — ~5x faster on long-GOP HEVC (decode-bound) footage with equal
        # recall. try_keyframe_sampler returns None on any problem, so we transparently
        # fall back to grab-through and never regress. In keyframe mode cap stays open
        # for metadata only; timestamps come from the frame index since the capture is
        # no longer advanced by reads.
        sampler = try_keyframe_sampler(self.video_path, logger)
        use_keyframe = sampler is not None
        if not use_keyframe:
            sampler = iter_sampled_frames(cap)
        had_detection = None
        sampling_stats = {}
        try:
            while True:
                _t_decode = time.perf_counter()
                frame_num, frame = sampler.send(had_detection)
                _dt = time.perf_counter() - _t_decode
                decode_time += _dt
                if had_detection:      # last frame had a shark -> this decode is a dense step
                    decode_dense_time += _dt
                    dense_frames += 1
                    dense_decode_ms.append(_dt * 1000)
                else:                  # scanning empty water (or the very first frame)
                    decode_scan_time += _dt
                    scan_frames += 1
                    scan_decode_ms.append(_dt * 1000)

                if QThread.currentThread().isInterruptionRequested():
                    logger.warning("Processing interrupted")
                    break

                _t_model = time.perf_counter()
                results = self.model(frame, classes=[0], verbose=False)
                _mt = time.perf_counter() - _t_model
                model_time += _mt
                yolo_ms.append(_mt * 1000)
                frames_sampled += 1
                if frames_sampled == 1:
                    _log(f"first infer ok frame={frame_num} shape={getattr(frame, 'shape', None)}")

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
                    # Build an owned QImage on the worker thread. Queuing raw np.ndarray
                    # via pyqtSignal has aborted frozen Windows builds in Qt6Core
                    # (0xc0000409) before any frame appeared.
                    rgb = np.ascontiguousarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    h, w, _ = rgb.shape
                    qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
                    self.frame_processed.emit(qimg)
                    if frames_sampled == 1:
                        _log(f"first preview emitted {w}x{h}")
                    last_preview = now

                self.progress_update.emit(int((frame_num + 1) / total_frames * 100))
        except StopIteration as stop:
            sampling_stats = stop.value or {}

        infer_time = time.perf_counter() - infer_start
        cap.release()
        _log(f"infer loop done sampled={frames_sampled} dets={total_detections} t={infer_time:.2f}s")

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
            scan_ms = (decode_scan_time / scan_frames * 1000) if scan_frames else 0.0
            dense_ms = (decode_dense_time / dense_frames * 1000) if dense_frames else 0.0
            logger.info(f"[timing] {Path(self.video_path).name}: "
                  f"loop={infer_time:.1f}s [decode={decode_time:.1f}s ({decode_pct:.0f}%): "
                  f"scan={decode_scan_time:.1f}s ({scan_frames}f {scan_ms:.0f}ms/f), "
                  f"dense={decode_dense_time:.1f}s ({dense_frames}f {dense_ms:.0f}ms/f) | "
                  f"yolo={model_time:.1f}s ({frames_sampled} frames, {ms_per_infer:.0f}ms/f), "
                  f"other={other_time:.1f}s] {total_detections} dets | "
                  f"segmentation={seg_time:.1f}s csv={csv_time:.2f}s "
                  f"tracks={track_count} (export+clip deferred to background)")

            # Per-frame timing distribution. The [timing] line reports means; a p95 >> p50
            # exposes tail stalls a mean hides — most usefully the decoder flush after each
            # keyframe seek (correlate with the [stats] seeks/mode_switches count).
            def _p(samples):
                if not samples:
                    return (0.0, 0.0)
                return (float(np.percentile(samples, 50)), float(np.percentile(samples, 95)))
            scan_p50, scan_p95 = _p(scan_decode_ms)
            dense_p50, dense_p95 = _p(dense_decode_ms)
            yolo_p50, yolo_p95 = _p(yolo_ms)
            logger.info(f"[timing-dist] {Path(self.video_path).name} ms/frame p50/p95: "
                  f"scan-decode={scan_p50:.0f}/{scan_p95:.0f} "
                  f"dense-decode={dense_p50:.0f}/{dense_p95:.0f} "
                  f"yolo={yolo_p50:.0f}/{yolo_p95:.0f}")

            # Association census: how the tracker spent its detections and, critically, why
            # new ids were opened (unassigned by the Hungarian step vs. rejected by the
            # re-association gate). A run where new_ids spikes inside a single shark window
            # is fragmentation; this line surfaces it without diffing on-disk labels. Also
            # reports created/significant/filtered ALWAYS (the "Filtered out N" line above
            # only fires when N>0), so "no sub-threshold tracks existed" is distinguishable
            # from "the line didn't print".
            a = custom_tracker.assoc_stats
            logger.info(f"[assoc] {Path(self.video_path).name}: "
                  f"frames_with_dets={a['frames_with_dets']} detections={a['detections']} "
                  f"matched={a['matched']} | new_ids: first_frame={a['new_first_frame']} "
                  f"unassigned={a['new_unassigned']} gate_rejected={a['new_from_gate']} | "
                  f"tracks_created={len(custom_tracker.tracks)} significant={track_count} "
                  f"filtered={filtered_count}")

            # Per-track discovery line: one row per significant track so a run's actual
            # findings (when, how confident, how long, how many detections) are legible
            # from the log instead of only a bare count. The spatial signature (time span,
            # box-center start->end, and x/y travel) is what tells two co-swimming sharks
            # apart from one fragmented track: two tracks alive at once in disjoint x-bands
            # are two animals; abutting spans that share a location are one animal split.
            # Positions/timestamps are the retained tail (deque maxlen=100), so on a >100-
            # sample track the span covers the last ~100 samples, not the whole life.
            for tid, tr in significant_tracks.items():
                confs = tr.get('confidences') or [0.0]
                positions = list(tr.get('positions') or [])
                timestamps = list(tr.get('timestamps') or [])
                spatial = ""
                if positions and timestamps:
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    t0, t1 = timestamps[0], timestamps[-1]
                    spatial = (
                        f" | t={CustomTracker._format_timestamp(t0)}"
                        f"-{CustomTracker._format_timestamp(t1)} ({(t1 - t0) / 1000:.1f}s) "
                        f"center=({xs[0]:.0f},{ys[0]:.0f})->({xs[-1]:.0f},{ys[-1]:.0f}) "
                        f"x_span={max(xs) - min(xs):.0f}px y_span={max(ys) - min(ys):.0f}px")
                logger.info(f"[track {tid}] t={CustomTracker._format_timestamp(tr.get('best_timestamp', 0))} "
                      f"peak_conf={max(confs):.2f} avg_conf={np.mean(confs):.2f} "
                      f"dets={len(confs)} length={tr.get('longest_length', 0):.1f}ft{spatial}")

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
        else:
            _log("interrupted before save")
            self.progress_status_changed.emit("")
            self.processing_complete.emit({}, os.path.basename(self.video_path))

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
                'timestamps': list(track.get('timestamps', [])),
                'longest_timestamp': track.get('longest_timestamp'),
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
            # 'Length (ft)' is the canonical length every downstream consumer should read:
            # it resolves the precedence manual > SAM at write time (SAM here; the review
            # editor overwrites it with the manual value when a human draws a line). The other
            # length columns are provenance/diagnostics: 'Highest Confidence Length' is the SAM
            # mask measurement; 'Longest Length' is now the bbox estimate at the best-confidence
            # frame (a coarse cross-check), not the old outlier-prone max over every frame.
            fieldnames = ['video_name', 'Flight Location', 'Drone', 'Altitude', 'Track Id', 'Length (ft)',
                        'Highest Conf Timestamp',
                        'Longest Length Timestamp', 'Highest Confidence', 'Average Confidence',
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
                    'Drone': self.drone_type,        # persisted so the review editor auto-resolves FOV
                    'Altitude': self.altitude,
                    'Track Id': track_id,
                    'Length (ft)': track['longest_length'],   # canonical: SAM now, manual overrides in review
                    'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                    'Longest Length Timestamp': CustomTracker._format_timestamp(track['longest_timestamp']),
                    'Highest Confidence': max(track['confidences']),
                    'Average Confidence': np.mean(track['confidences']),
                    'Lowest Confidence': min(track['confidences']),
                    'Longest Length': track['best_length'],   # bbox at best-conf frame (diagnostic, not max)
                    'Highest Confidence Length': track['longest_length'], # SAM mask measurement
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
            device = select_torch_device()
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

        warn_on_libav_collision()
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

        # Silently refresh help docs from the cloud when a newer version is available.
        self.docs_sync_thread = None
        QTimer.singleShot(0, self.sync_help_docs)

    def initialize_settings(self):
        self.settings_obj = ensure_app_settings()

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

    def sync_help_docs(self):
        """Kick off a background help-docs sync against the Cloud Function on startup."""
        self.docs_sync_thread = DocsSyncThread(UPDATE_ENDPOINT)
        self.docs_sync_thread.sync_finished.connect(self.on_docs_sync_finished)
        self.docs_sync_thread.start()

    def on_docs_sync_finished(self, updated, error):
        if error:
            logger.error(f"Help docs sync failed: {error}")
            return
        if updated:
            logger.info("Help docs updated from cloud")

    def show_help_docs(self):
        guide_path = resolve_help_guide_path()
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

        self.false_positives_dir = os.path.join(self.current_output_dir, 'false_positives')

        os.makedirs(os.path.join(self.current_output_dir, 'frames'), exist_ok=True)
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
        # total_detections is the number of tracks (one per shark), not raw detections —
        # label it as such. The [batch] log line reports both track and detection counts.
        msg.setText(f"Processing completed!\n\nSharks detected: {total_detections}\nTime taken: {time_str}")
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
        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']
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
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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
        count_reliable = True
        if det_dir.exists():
            for csv_file in det_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    total_tracks += len(df)
                except Exception as e:
                    # A CSV we can't read (corrupt / locked) means the count is a
                    # lower bound. Deleting the whole experiment on an undercount
                    # would wipe other videos' results, so mark the count untrusted
                    # and fall through to per-track deletion instead.
                    logger.warning(f"Could not read {csv_file} while counting tracks; "
                                   f"will not delete whole experiment: {e}")
                    count_reliable = False

        # Only wipe the entire experiment when we're certain the track being deleted
        # is genuinely the last one across every video in it.
        if count_reliable and total_tracks == 1:
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
        """Toggle the segmentation mask overlay.

        The mask corresponds to one existing clip frame (the length-source frame), so
        rather than replacing the player with a static image, we seek the scrubber to
        that frame and paint the mask there — the media buttons and scrubber stay usable.
        The mask was preloaded for the selected track in _apply_overlay_boxes."""
        if not self.frame_player.has_mask():
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Alert")
            dlg.setText("Error: No mask drawn for this track")
            dlg.exec()
            self.mask_active = False
            self.toggle_display_switch.reset_position()
            self.toggle_display_switch.update()
            self._update_edit_frame_button()
            return

        show_mask = not self.frame_player.mask_visible()
        self.frame_player.set_mask_visible(show_mask)
        self.mask_active = show_mask
        self.gif_active = not show_mask
        self._sync_play_pause_button()
        self.update_frame_elements()
        self._update_edit_frame_button()

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

        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']

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
        self.setup_review_shortcuts()

    def setup_review_shortcuts(self):
        """Keyboard shortcuts for the Review screen so a reviewer can work without the mouse.

        All shortcuts are scoped to the review widget (WidgetWithChildrenShortcut), so they
        never fire on the Home or processing screens. Reviewing hundreds of detections is the
        app's most repetitive task; play/pause + frame-step + next/prev-detection off the
        keyboard is the biggest single throughput win.

            Space         play / pause the clip
            ← / →         step one frame back / forward (pauses first)
            J / K         previous / next detection in the list
        """
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut

        def bind(key, handler):
            sc = QShortcut(QKeySequence(key), self.review_widget)
            sc.setContext(ctx)
            sc.activated.connect(handler)
            return sc

        bind(Qt.Key.Key_Space, self.toggle_playback)
        bind(Qt.Key.Key_Left, lambda: self._step_review_frame(-1))
        bind(Qt.Key.Key_Right, lambda: self._step_review_frame(1))
        bind(Qt.Key.Key_J, lambda: self._select_adjacent_detection(-1))
        bind(Qt.Key.Key_K, lambda: self._select_adjacent_detection(1))

    def _step_review_frame(self, delta):
        """Pause and move the clip by ``delta`` frames (clamped). Only meaningful in
        frame-sequence (MP4) mode, where the slider is scrubbable."""
        if getattr(self, "_playback_mode", None) != "frames":
            return
        self.frame_player.pause()
        self._sync_play_pause_button()
        new_val = min(max(0, self.frame_slider.value() + delta), self.frame_slider.maximum())
        self.frame_slider.setValue(new_val)  # triggers _on_slider_moved -> player.seek

    def _select_adjacent_detection(self, delta):
        """Select the previous/next row in the detection list, which loads that clip via
        the existing itemSelectionChanged wiring."""
        table = getattr(self, "historical_items", None)
        if table is None:
            return
        count = table.rowCount()
        if count == 0:
            return
        cur = table.currentRow()
        if cur < 0:
            cur = 0
        new = min(max(0, cur + delta), count - 1)
        if new != cur:
            table.selectRow(new)

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
        self.play_pause_button.setToolTip(
            "Play or pause the detection clip (Space)\n"
            "←/→ step a frame · J/K previous/next detection")
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

        # Bounding-box overlay prefs. The box (and its confidence label) is drawn live
        # over the raw clip (see FramePlayer), so it can be hidden or recolored without
        # re-encoding anything. Visibility choices persist across sessions. All three
        # (show boxes, show confidence, box color) live behind the gear button added at
        # the right end of this row (see _open_overlay_settings).
        self.show_boxes = self.settings_obj.value("review_show_boxes", "1") == "1"
        self.show_confidence = self.settings_obj.value("review_show_confidence", "1") == "1"

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
        # Left-align so the count sits right against the scrubber's end rather than
        # drifting toward the gear button at the row's right edge.
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.frame_counter_label)

        # Overlay display settings live behind this gear button at the right end of the
        # row: show/hide boxes, show/hide confidence, and box color (see
        # _open_overlay_settings). Keeps the playback row uncluttered.
        self.overlay_settings_button = QPushButton()
        self.overlay_settings_button.setIcon(
            colored_svg_icon(resource_path("assets/images/gear-fill.svg"), theme_icon_color()))
        self.overlay_settings_button.setToolTip(
            "Overlay display settings — show boxes, confidence value, and box color")
        self.overlay_settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.overlay_settings_button.clicked.connect(self._open_overlay_settings)
        row.addWidget(self.overlay_settings_button)

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

    # --- bounding-box overlay controls -----------------------------------

    def _review_box_color(self):
        """The color to draw review bounding boxes in.

        Persisted per-user under "review_box_color"; defaults to the app's annotation
        color so the review overlay matches exported annotations out of the box."""
        stored = self.settings_obj.value("review_box_color", None)
        if stored:
            try:
                r, g, b = (int(v) for v in str(stored).split(","))
                return QColor(r, g, b)
            except (TypeError, ValueError):
                pass
        annotation_color, *_ = get_annotation_settings(self.settings_obj)
        return QColor(annotation_color[0], annotation_color[1], annotation_color[2])

    def _style_color_swatch(self, button):
        """Paint `button` as a color chip showing the current review box color."""
        c = self._review_box_color()
        button.setStyleSheet(
            f"background-color: rgb({c.red()},{c.green()},{c.blue()}); border: 1px solid #888;")

    def _open_overlay_settings(self):
        """Modal dialog for the box-overlay display prefs (confidence + color).

        Both prefs persist immediately through their existing handlers, so the dialog
        needs no OK/Cancel bookkeeping — Close just dismisses it."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Overlay Settings")
        form = QFormLayout(dlg)

        boxes_checkbox = QCheckBox("Show bounding boxes")
        boxes_checkbox.setChecked(self.show_boxes)
        boxes_checkbox.toggled.connect(self._on_show_boxes_toggled)
        form.addRow(boxes_checkbox)

        conf_checkbox = QCheckBox("Show confidence value on the box")
        conf_checkbox.setChecked(self.show_confidence)
        conf_checkbox.toggled.connect(self._on_show_confidence_toggled)
        form.addRow(conf_checkbox)

        color_button = QPushButton()
        color_button.setFixedSize(80, 24)
        color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_color_swatch(color_button)
        color_button.clicked.connect(lambda: self._pick_box_color(color_button))
        form.addRow("Bounding box color:", color_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        form.addRow(buttons)

        dlg.exec()

    def _on_show_boxes_toggled(self, checked):
        self.show_boxes = bool(checked)
        self.settings_obj.setValue("review_show_boxes", "1" if checked else "0")
        self.frame_player.set_boxes_visible(self.show_boxes)

    def _on_show_confidence_toggled(self, checked):
        self.show_confidence = bool(checked)
        self.settings_obj.setValue("review_show_confidence", "1" if checked else "0")
        self.frame_player.set_confidence_visible(self.show_confidence)

    def _pick_box_color(self, swatch=None):
        color = QColorDialog.getColor(self._review_box_color(), self, "Bounding Box Color")
        if color.isValid():
            self.settings_obj.setValue("review_box_color", f"{color.red()},{color.green()},{color.blue()}")
            if swatch is not None:
                self._style_color_swatch(swatch)
            self.frame_player.set_box_color(color)

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
        # The box overlay is only drawn in frame-sequence (MP4) mode and only when the
        # player actually has box data, so gate its controls on both.
        if hasattr(self, "overlay_settings_button"):
            overlay_ok = mode == "frames" and self.frame_player.has_boxes()
            self.overlay_settings_button.setEnabled(overlay_ok)
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

    def _current_experiment_drone_altitude(self):
        """Read the Drone + Altitude the current experiment was processed with, from its CSV,
        so the editor auto-selects the drone whose FOV profile matches the footage (and fills
        altitude). Returns (drone_or_None, altitude_or_None); legacy CSVs without the columns,
        or a non-historical context, return (None, None) so the caller falls back."""
        try:
            row = self.historical_items.currentRow()
            if row < 0:
                return (None, None)
            item = self.historical_items.item(row, 0)
            meta = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not meta:
                return (None, None)
            experiment, _video_name, csv_name, track_id = meta
            csv_path = Path(get_results_dir()) / experiment / "detection_results" / csv_name
            if not csv_path.exists():
                return (None, None)
            df = pd.read_csv(csv_path)
            m = df["Track Id"].astype(int) == int(track_id)
            if not m.any():
                return (None, None)
            rowdata = df[m].iloc[0]
            drone = rowdata["Drone"] if "Drone" in df.columns else None
            alt = rowdata["Altitude"] if "Altitude" in df.columns else None
            drone = str(drone) if not _csv_value_is_empty(drone) else None
            try:
                alt = float(alt) if not _csv_value_is_empty(alt) else None
            except (TypeError, ValueError):
                alt = None
            return (drone, alt)
        except Exception:
            return (None, None)

    def open_frame_editor(self):
        """Replace the active frame view with the in-place line editor.

        From the mask overlay this edits the saved best frame; during clip playback it
        edits the frame currently paused in the player (captured full-res from memory).
        """
        self.frame_editor._update_drone_settings()
        # Prefer the drone/altitude this experiment was processed with (persisted in the CSV),
        # so feet resolve without the user re-picking the drone; fall back to the global
        # last_drone_type for legacy experiments written before the columns existed.
        exp_drone, exp_alt = self._current_experiment_drone_altitude()
        initial_drone = exp_drone or (self.settings_obj.value("last_drone_type") or None)

        if getattr(self, "mask_active", False):
            frame_path = self._current_frame_image_path()
            if not frame_path:
                self._frame_editor_error("Error: No frame available to edit")
                return
            loaded = self.frame_editor.load_image(frame_path, drone_altitude=exp_alt,
                                                  initial_drone=initial_drone)
        else:
            # Playback frame — only reachable while paused (button is disabled otherwise).
            pixmap = self.frame_player.current_frame_pixmap()
            if pixmap is None:
                self._frame_editor_error("Error: No frame available to edit")
                return
            loaded = self.frame_editor.load_pixmap(pixmap, drone_altitude=exp_alt,
                                                   initial_drone=initial_drone)

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
            # An all-blank column reads back as float64 (all-NaN), and pandas then rejects a
            # "" write into it ("Invalid value '' for dtype 'float64'"). Coerce the columns we
            # touch to object dtype so both blanks and floats are accepted.
            for col in ("manual_length_px", "manual_length_ft", "Length (ft)"):
                if col not in df.columns:
                    df[col] = ""
                df[col] = df[col].astype(object)

            mask = df["Track Id"].astype(int) == int(track_id)
            if not mask.any():
                raise ValueError(f"Track {track_id} not found in {csv_path}")

            df.loc[mask, "manual_length_px"] = length_px
            df.loc[mask, "manual_length_ft"] = length_ft if length_ft is not None else ""
            # Re-resolve the canonical 'Length (ft)' with precedence manual > SAM, so every
            # CSV consumer gets the human's correction without reimplementing the fallback.
            if length_ft is not None:
                df.loc[mask, "Length (ft)"] = length_ft
            elif "Highest Confidence Length" in df.columns:
                df.loc[mask, "Length (ft)"] = df.loc[mask, "Highest Confidence Length"]
            df.to_csv(csv_path, index=False)

            if length_ft is not None:
                length_item = self.historical_items.item(row, 5)
                if length_item is not None:
                    length_item.setText(f"{float(length_ft):.1f}ft")
                QMessageBox.information(
                    self,
                    "Length Saved",
                    f"Length correction saved: {float(length_ft):.1f} ft.",
                )
            else:
                # Pixels were saved but feet couldn't be computed — the editor's drone
                # doesn't have a FOV profile for this frame's resolution, or altitude is blank.
                QMessageBox.warning(
                    self,
                    "Pixel Length Saved (no feet)",
                    "Saved the pixel length, but couldn't convert to feet: pick the drone that "
                    "matches this footage and enter an altitude in the editor, then draw again.",
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
        
        frames_dir = Path(get_results_dir()) / experiment / "shark_frames"

        if mp4_path.exists():
            # self.toggle_display_mode_button.setIcon(QIcon(resource_path("assets/images/MdiSharkFin.svg")))
            self.toggle_display_switch.reset_position()
            self.toggle_display_switch.update()
            self.frame_player.set_video(str(mp4_path))
            self._apply_overlay_boxes(frames_dir, video_basename, track_id)
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
                self._apply_overlay_boxes(frames_dir, Path(video_basename).stem, track_id)
                self.update_frame_elements()
                QTimer.singleShot(0, self.update_frame_elements)
            elif alt_gif.exists():
                self.frame_player.set_gif(str(alt_gif))
                self.update_frame_elements()
                QTimer.singleShot(0, self.update_frame_elements)
            else:
                self.frame_player.clear()
                self.frame_player.setText(f"Video not found:\n{mp4_name} or {gif_name}")

    def _read_overlay_annotations(self, track_dir):
        """Return (boxes, longest_index) for a track's frame dir.

        Prefers the YOLO format — frame_<NNNN>.txt (class cx cy w h, normalized) plus
        meta.json (per-frame confidence + length-source index) — and denormalizes the
        coords to native pixels using meta's frame dims. Falls back to the legacy
        boxes.json (full-res pixel coords) so pre-migration experiments still render.
        Each returned box is (x_center, y_center, w, h, conf) in native pixels, or None
        for a frame with no detection. Returns ([], None) if nothing is readable."""
        track_dir = Path(track_dir)
        meta_path = track_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                fw, fh = meta.get('frame_width'), meta.get('frame_height')
                boxes = []
                for i, fm in enumerate(meta.get('frames') or []):
                    box = None
                    txt = track_dir / f"frame_{i:04d}.txt"
                    if fw and fh and txt.exists():
                        line = txt.read_text().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                cx, cy, nw, nh = (float(v) for v in parts[1:5])
                                conf = (fm or {}).get('conf')
                                box = (cx * fw, cy * fh, nw * fw, nh * fh, conf)
                    boxes.append(box)
                return boxes, meta.get('longest_index')
            except Exception as e:
                logger.warning(f"Could not read YOLO overlay in {track_dir}: {e}")

        sidecar = track_dir / "boxes.json"  # legacy format
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                boxes = [None if b is None
                         else (b['x'], b['y'], b['w'], b['h'], b.get('conf'))
                         for b in data.get('boxes', [])]
                return boxes, data.get('longest_index')
            except Exception as e:
                logger.warning(f"Could not read box sidecar {sidecar}: {e}")
        return [], None

    def _apply_overlay_boxes(self, frames_dir, video_basename, track_id):
        """Feed a clip's per-frame boxes into the player as a live overlay.

        Reads the YOLO annotation for the track (frame_<NNNN>.txt + meta.json, written by
        encode_track_clips), converting the normalized coords back to native pixels.
        Legacy experiments have a single boxes.json instead — _read_overlay_annotations
        falls back to it so old runs still render."""
        track_dir = Path(frames_dir) / f"{video_basename}_{track_id}"
        boxes, longest_index = self._read_overlay_annotations(track_dir)

        self.frame_player.set_overlay_boxes(boxes, longest_index)

        # Preload the segmentation mask for the length-source frame so the mask toggle can
        # display it in place (keeping the scrubber) rather than as a static image.
        mask_img = None
        try:
            mask_path = Path(frames_dir).parent / "masks" / f"{video_basename}_{track_id}.jpg"
            if mask_path.exists():
                mask_img = cv2.imread(str(mask_path))
        except Exception as e:
            logger.warning(f"Could not read mask for {video_basename}_{track_id}: {e}")
        self.frame_player.set_mask(mask_img, longest_index)

        has = any(b is not None for b in boxes)
        self.frame_player.set_box_color(self._review_box_color())
        self.frame_player.set_boxes_visible(self.show_boxes if has else False)
        self.frame_player.set_confidence_visible(self.show_confidence if has else False)
        if hasattr(self, "overlay_settings_button"):
            self.overlay_settings_button.setEnabled(has)

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
        labels = ['Experiment', 'Video', 'ID', 'Time', 'Confidence', 'Length', 'Label', '']

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
                        # Show the timestamp of the frame the LENGTH was measured on (the
                        # "longest" frame), so Length Time / Confidence / Length all refer
                        # to the same frame. Fall back to the highest-conf timestamp for
                        # legacy CSVs written before this column existed.
                        length_ts = row.get('Longest Length Timestamp', '')
                        if _csv_value_is_empty(length_ts):
                            length_ts = row.get('Highest Conf Timestamp', '')
                        time_str = str(length_ts)
                        conf_longest = float(row.get('Confidence of Longest Length', 0.0))
                        len_high_conf = float(row.get('Highest Confidence Length', 0.0))
                        # Prefer the canonical 'Length (ft)' column (already resolves
                        # manual > SAM). Fall back to the manual/SAM precedence for legacy
                        # CSVs written before the column existed.
                        canonical_len = row.get('Length (ft)', '')
                        manual_ft = row.get('manual_length_ft', '')
                        if not _csv_value_is_empty(canonical_len):
                            display_length = float(canonical_len)
                        elif not _csv_value_is_empty(manual_ft):
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
                                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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

    def update_frame_display(self, q_image):
        # First frame of this video arrived — switch the busy bar back to determinate.
        if getattr(self, "_awaiting_first_frame", False):
            self._awaiting_first_frame = False
            if getattr(self, "progress_bar", None) is not None:
                self.progress_bar.setRange(0, 100)
        if q_image is None or q_image.isNull():
            return
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.frame_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
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
        """Persist the reviewer's queued label/deletion changes back to the per-video CSVs.

        NOTE: this is wired to the "Save Changes" button. It intentionally only flushes
        `historical_label_changes` — the older "export all tracks to a user-chosen CSV
        file" feature was gated behind a `reviewing_history` flag that no longer exists
        and its (unreachable) code was removed. If a standalone CSV export is wanted
        again, reintroduce it as its own action rather than overloading Save Changes.
        """
        self._save_historical_label_changes()

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

# Experiment uploads go to a Cloud Function whose HTTP request is size-limited, so every
# image is downscaled to <=1080p before zipping. 1080p JPGs are ~150-300 KB each. The
# length-measurement artifacts (frames/, masks/) stay full-res on disk; shark_frames/ (the
# per-shark training set) is now written at <=1080p directly (see encode_track_clips) since
# that's all the uploader would keep anyway, so the re-encode below is usually a no-op for
# those. bounding_boxes/ is intentionally excluded: it was a burned-in duplicate of frames/
# reconstructable from the YOLO label + CSV, and is no longer generated.
UPLOAD_FOLDERS = ['detection_results', 'false_positives',
                  'frames', 'masks', 'shark_frames']
UPLOAD_IMAGE_MAX_W = 1920
UPLOAD_IMAGE_MAX_H = 1080


def _downscale_image_bytes(path, max_w=UPLOAD_IMAGE_MAX_W, max_h=UPLOAD_IMAGE_MAX_H):
    """Return JPG bytes for `path` scaled to fit (max_w, max_h) keeping aspect.

    Returns None on any failure so the caller can fall back to shipping the file
    verbatim. Images already within bounds are still re-encoded (harmless, keeps one
    code path)."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return enc.tobytes() if ok else None
    except Exception:
        return None


def refresh_yolo_labels_from_csv(experiment_dir):
    """Bake the reviewer's corrected labels into the YOLO .txt class column before upload.

    Frames are written at inference time as class 0 (shark). The reviewer may then relabel
    a track (Kelp, Boat, …); those corrections are persisted to detection_results/*.csv.
    Here we map each track's current label -> class id (label_to_yolo_class) and rewrite
    the class token of every frame_*.txt for that track, leaving the geometry untouched, so
    the uploaded dataset trains on the corrected classes. Best-effort per track."""
    det_dir = os.path.join(experiment_dir, "detection_results")
    frames_root = os.path.join(experiment_dir, "shark_frames")
    if not (os.path.isdir(det_dir) and os.path.isdir(frames_root)):
        return
    for csv_name in os.listdir(det_dir):
        if not csv_name.lower().endswith(".csv"):
            continue
        video_base = csv_name[:-4]  # "DJI_0033.MP4.csv" -> "DJI_0033.MP4"
        try:
            with open(os.path.join(det_dir, csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            logger.warning(f"[upload] YOLO relabel skipped for {csv_name}: {e}")
            continue
        for row in rows:
            track_id = (row.get("Track Id") or "").strip()
            if not track_id:
                continue
            cls = label_to_yolo_class(row.get("Label"))
            track_dir = os.path.join(frames_root, f"{video_base}_{track_id}")
            if not os.path.isdir(track_dir):
                continue
            for fn in os.listdir(track_dir):
                if not fn.endswith(".txt"):
                    continue
                p = os.path.join(track_dir, fn)
                try:
                    with open(p) as tf:
                        line = tf.read().strip()
                    if not line:
                        continue  # negative frame — no box to reclass
                    parts = line.split()
                    parts[0] = str(cls)
                    with open(p, "w") as tf:
                        tf.write(" ".join(parts) + "\n")
                except Exception as e:
                    logger.warning(f"[upload] could not relabel {p}: {e}")


def add_experiment_to_zip(zipf, experiment_dir, folders=UPLOAD_FOLDERS):
    """Add an experiment's upload folders to an open ZipFile, downscaling images.

    Any .jpg/.jpeg/.png is downscaled to <=1080p (see _downscale_image_bytes); everything
    else — CSVs, the YOLO frame_*.txt labels, meta.json, classes.txt — is added verbatim.
    Returns the number of files written."""
    count = 0
    for folder in folders:
        folder_path = os.path.join(experiment_dir, folder)
        if not os.path.exists(folder_path):
            logger.warning(f"[upload]   skipping missing folder: {folder}")
            continue
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, experiment_dir)
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data = _downscale_image_bytes(file_path)
                    if data is not None:
                        zipf.writestr(arcname, data)
                    else:
                        zipf.write(file_path, arcname)
                else:
                    zipf.write(file_path, arcname)
                count += 1
    return count


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
            # Bake the reviewer's corrected labels into the YOLO class column first.
            refresh_yolo_labels_from_csv(self.experiment_dir)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as zipf:
                # 'shark_frames' carries every sampled frame of each shark (+ per-frame
                # YOLO labels); add_experiment_to_zip downscales images to <=1080p so the
                # full detection sequence fits under the upload size limit.
                file_count = add_experiment_to_zip(zipf, self.experiment_dir)

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


class DocsSyncThread(QThread):
    """Compare local help-docs version to the cloud and download updates in-app.

    Never opens a browser. Failures are reported via sync_finished so the UI can
    log them without blocking launch.
    """
    # (updated, error_message)
    sync_finished = pyqtSignal(bool, str)

    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint

    def run(self):
        try:
            docs_dir = get_writable_docs_dir()
            docs_present = local_help_docs_present(docs_dir)
            # Missing guide/images must force a download even if a version stamp
            # already matches the cloud (e.g. files deleted or first install).
            local_version = read_local_doc_version(docs_dir) if docs_present else 0

            response = requests.get(
                self.endpoint,
                params={
                    "request": "check_docs",
                    "doc_version": local_version,
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected check_docs response shape")

            if payload.get("up_to_date", False):
                # Cloud says current, but files may still be missing if the stamp
                # somehow matched — treat that as needing a forced re-fetch.
                if docs_present:
                    self.sync_finished.emit(False, "")
                    return
                response = requests.get(
                    self.endpoint,
                    params={"request": "check_docs", "doc_version": 0},
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("up_to_date", False):
                    raise ValueError("Help docs missing locally and cloud returned no files")

            files = payload.get("files") or {}
            latest_version = payload.get("doc_version")
            if not files or latest_version is None:
                raise ValueError("check_docs response missing files or doc_version")

            for relative_name, signed_url in files.items():
                # Reject path traversal from unexpected blob keys.
                relative_name = relative_name.replace("\\", "/").lstrip("/")
                if ".." in relative_name.split("/"):
                    raise ValueError(f"Invalid docs path: {relative_name}")

                dest_path = os.path.join(docs_dir, *relative_name.split("/"))
                parent = os.path.dirname(dest_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                file_response = requests.get(signed_url, timeout=60)
                file_response.raise_for_status()

                # Write via temp file then replace so partial downloads don't corrupt.
                tmp_path = dest_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(file_response.content)
                os.replace(tmp_path, dest_path)

            # Only bump the stamp after every file lands so a failed sync retries.
            write_local_doc_version(int(latest_version), docs_dir=docs_dir)
            self.sync_finished.emit(True, "")
        except Exception as e:
            self.sync_finished.emit(False, str(e))


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

        # Bounding-box overlay, drawn at paint time rather than baked into the frames, so
        # it can be toggled on/off and recolored live. self._boxes[i] is the box for
        # self.frames[i] as (x_center, y_center, w, h) in native frame pixels, or None.
        self._boxes = []
        self._show_boxes = True
        self._show_confidence = True
        self._box_color = QColor(255, 96, 31)
        self._longest_index = None  # index of the frame the length was measured on
        # A clip shorter than playback_min_frames is shown as a single static center frame
        # (see set_video). self._static_box is that frame's box so the overlay still draws
        # on the still; self._static_frame_index is which clip frame is on screen, used to
        # pick the matching box once set_overlay_boxes supplies them (which happens *after*
        # set_video has already rendered the static image).
        self._static_box = None
        self._static_frame_index = None

        # Segmentation mask overlay. The mask corresponds to one existing clip frame (the
        # length-source frame), so it's painted in place on that frame rather than
        # swapping the player into a static image — the scrubber and media controls stay
        # live. self._mask_frame is a full-res BGR array; self._mask_index is its frame.
        self._mask_frame = None
        self._mask_index = None
        self._show_mask = False

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

    # --- bounding-box overlay --------------------------------------------
    # Boxes are drawn on top of the frame in paintEvent (frame-sequence mode only), not
    # baked into the pixels, so visibility and color are live. Callers supply a list
    # parallel to self.frames via set_overlay_boxes().

    def set_overlay_boxes(self, boxes, longest_index=None):
        """Provide per-frame boxes (parallel to self.frames) for the paint-time overlay.

        Each entry is (x_center, y_center, w, h) in native frame pixels, or None for a
        frame with no box. Pass [] to clear the overlay."""
        self._boxes = list(boxes) if boxes else []
        self._longest_index = longest_index
        # If a short clip is being shown as a static still (set_video ran before these
        # boxes arrived), grab the box for the on-screen frame so paintEvent can draw it.
        if self._static_pixmap is not None and self._static_frame_index is not None:
            idx = self._static_frame_index
            self._static_box = self._boxes[idx] if 0 <= idx < len(self._boxes) else None
        self.update()

    def has_boxes(self):
        return any(b is not None for b in self._boxes)

    def set_boxes_visible(self, visible):
        self._show_boxes = bool(visible)
        self.update()

    def boxes_visible(self):
        return self._show_boxes

    def set_confidence_visible(self, visible):
        """Show/hide the per-frame confidence value next to the box."""
        self._show_confidence = bool(visible)
        self.update()

    def set_box_color(self, color):
        if color is not None:
            self._box_color = QColor(color)
            self.update()

    def set_mask(self, mask_bgr, index):
        """Register the segmentation mask overlay (a full-res BGR frame) and the clip
        frame index it belongs to. Pass mask_bgr=None to clear."""
        self._mask_frame = mask_bgr
        self._mask_index = index

    def has_mask(self):
        return self._mask_frame is not None and self._mask_index is not None

    def mask_visible(self):
        return self._show_mask

    def set_mask_visible(self, visible):
        """Toggle the mask overlay. When enabling, pause and seek to the mask's frame so
        it's actually on screen; the scrubber/media controls remain usable throughout."""
        self._show_mask = bool(visible)
        if self._show_mask and self._mask_index is not None and self.frames:
            self.pause()
            self.seek(self._mask_index)
        self.update()

    def _paint_box_overlay(self, painter, dx, dy, dw, dh, frame_w, frame_h, box=None):
        """Draw a box (and optionally its confidence), mapping native pixel coords into the
        displayed (KeepAspectRatio-scaled, centered) rect. Matches content_rect()'s
        geometry. With box=None, draws the current frame's box from self._boxes (frame-
        sequence mode); pass an explicit box to draw over a static still."""
        if not self._show_boxes or frame_w <= 0 or frame_h <= 0:
            return
        if box is None:
            if not self._boxes or not (0 <= self.current_frame < len(self._boxes)):
                return
            box = self._boxes[self.current_frame]
        if box is None:
            return
        cx, cy, bw, bh = box[0], box[1], box[2], box[3]
        conf = box[4] if len(box) > 4 else None
        sx = dw / frame_w
        sy = dh / frame_h
        rx = dx + (cx - bw / 2) * sx
        ry = dy + (cy - bh / 2) * sy
        painter.save()
        pen = QPen(self._box_color)
        pen.setWidth(max(2, round(min(dw, dh) / 250)))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(int(rx), int(ry), int(bw * sx), int(bh * sy))
        if self._show_confidence and conf is not None:
            font = painter.font()
            font.setPointSize(max(9, round(min(dw, dh) / 45)))
            painter.setFont(font)
            painter.drawText(int(rx), max(int(ry) - 5, 14), f"{conf:.2f}")
        painter.restore()

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
        self._boxes = []
        # Reset the static-frame box; set_video re-arms _static_frame_index for a short
        # clip so set_overlay_boxes can populate _static_box afterwards. Other callers
        # (legacy gif, mask) leave it None, so they draw no static box.
        self._static_box = None
        self._static_frame_index = None
        self._mask_frame = None
        self._mask_index = None
        self._show_mask = False
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
        self._boxes = []
        self._mask_frame = None
        self._mask_index = None
        self._show_mask = False
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
            mid_index = len(frames) // 2
            mid = frames[mid_index]
            frame_rgb = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            q_image = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.set_static_pixmap(QPixmap.fromImage(q_image))
            # Remember which clip frame is on screen so set_overlay_boxes (called next by
            # the review flow, after this returns) can pick the matching box to draw.
            self._static_frame_index = mid_index
            return

        self.set_frames(frames)

    def set_gif(self, path: str):
        self._static_pixmap = None
        self._boxes = []
        self._mask_frame = None
        self._mask_index = None
        self._show_mask = False
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
            # Short-clip stills still get their detection box drawn (a 1–few-frame track
            # would otherwise show no box at all, hiding what was detected from review).
            if self._static_box is not None:
                self._paint_box_overlay(painter, x, y, scaled.width(), scaled.height(),
                                        frame.width(), frame.height(), box=self._static_box)
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
            # On the mask's own frame, paint the mask overlay instead of the raw frame
            # (and skip the box — the mask already shows the shark). Every other frame is
            # normal, so scrubbing off the mask frame returns to the clip seamlessly.
            showing_mask = (self._show_mask and self._mask_frame is not None
                            and self.current_frame == self._mask_index)
            frame = self._mask_frame if showing_mask else self.frames[self.current_frame]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            q_image = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.size().scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
            x = (widget_size.width() - scaled.width()) // 2
            y = (widget_size.height() - scaled.height()) // 2
            painter.drawPixmap(QRect(x, y, scaled.width(), scaled.height()), pixmap)
            if not showing_mask:
                self._paint_box_overlay(painter, x, y, scaled.width(), scaled.height(), w, h)
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
        self._boxes = []
        self._mask_frame = None
        self._mask_index = None
        self._show_mask = False
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
    def __init__(self, video_path, model, output_dir, drone_type="Air 2S", altitude=40.0):
        self.settings_obj = ensure_app_settings()
        self.video_path = video_path
        self.model = model
        self.output_dir = output_dir
        self.detection_threshold = float(self.settings_obj.value("confidence_threshold", "0.40"))
        self.drone_settings = get_drone_settings_dict(self.settings_obj)
        self.drone_type = drone_type
        self.altitude = float(altitude)

    progress_update = 0
    processing_complete = {}

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        custom_tracker = CustomTracker()
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Per-video FOV so length math matches the GUI (previously this path silently used
        # the default FOV). Falls back to the default + a warning when the drone/resolution
        # isn't in the map.
        fov = resolve_fov_radians(self.drone_type, video_width, video_height, self.drone_settings)
        if fov is not None:
            custom_tracker.fov_radians = fov
            custom_tracker.drone_altitude = self.altitude
        else:
            logger.warning(f"[gsd] {Path(self.video_path).name}: no FOV for drone={self.drone_type!r} "
                           f"@ {video_width}x{video_height}; using default {custom_tracker.fov_radians:.4f}rad")

        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
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
        # Shared implementation (Priority 2: segments the best-confidence frame; converts
        # SAM pixels -> feet, which this path previously failed to do). FOV was resolved
        # per-video above from --drone/--altitude.
        custom_tracker.save_best_frames(self.output_dir, self.video_path)

        all_track_info = []

        for track_id, track in significant_tracks.items():
            meets_thresholds = custom_tracker.is_significant_track(track)
    
            track_info = {
                'video_name': self.video_path,
                'Drone': self.drone_type,
                'Altitude': self.altitude,
                'Track Id': track_id,
                'Length (ft)': track['longest_length'],   # canonical: SAM (manual overrides in review)
                'Highest Conf Timestamp': CustomTracker._format_timestamp(track['best_timestamp']),
                'Highest Confidence': max(track['confidences']),
                'Average Confidence': np.mean(track['confidences']),
                'Lowest Confidence': min(track['confidences']),
                'Longest Length': track['best_length'],   # bbox at best-conf frame (diagnostic, not max)
                'Highest Confidence Length': track['longest_length'], # SAM mask measurement
                'Number of Detections': len(track['confidences']),
                'Meets Thresholds': meets_thresholds,
                'Confidence of Longest Length': track['longest_conf'],
                'Label': 'Shark',
                'manual_length_px': '',
                'manual_length_ft': '',
                'Segmentation Duration': track['segmentation_duration'],
            }

            all_track_info.append(track_info)
    
        return all_track_info            

def mass_prediction(video_paths, current_output_dir, drone_type="Air 2S", altitude=40.0):
    device = select_torch_device()
    logger.info(f"Using device: {device}")
    model = YOLO(MODEL_PATH).to(device)

    videos_tqdm = tqdm(video_paths)
    all_track_results = []
    processing_logs = {}
    for path in videos_tqdm:
        videos_tqdm.set_description(f"Processing {path}")
        path_start = time.perf_counter()
        processor = HeadlessVideoProcessor(path, model, current_output_dir,
                                           drone_type=drone_type, altitude=altitude)
        path_results = processor.run()
        path_end = time.perf_counter()

        total_processing_duration = path_end - path_start
        total_tracks = len(path_results)
        total_segmentation_duration = 0
        video_length = get_video_length(path)

        for track in path_results:
            total_segmentation_duration += track['Segmentation Duration']

        processing_logs[str(path.name)] = {
            'video_length': video_length,
            'total_tracks': total_tracks,
            'total_processing_duration': total_processing_duration,
            'total_segmentation_duration': total_segmentation_duration,    
        }

        all_track_results.extend(path_results)

    log_path = current_output_dir / "processing_logs.json"
    with open(log_path, "w") as f:
        json.dump(processing_logs, f)
    
    return all_track_results

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run headless object tracking on videos.")
    parser.add_argument('--testing', action='store_true', help='Enables testing for app in headless environment')
    parser.add_argument('--input_dir', type=str, required=False, help='Directory containing videos to process (.mp4/.mov, case-insensitive)')
    parser.add_argument('--output_dir', type=str, default='./headless_predictions', help='Directory to store output predictions and CSV')
    parser.add_argument('--drone', type=str, default='Air 2S', help='Drone model, for per-video FOV / length calibration')
    parser.add_argument('--altitude', type=float, default=40.0, help='Flight altitude in meters, for length calibration')
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
    install_crash_handlers()
    _install_qt_message_filter()
    args = parse_args()
    if args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        # Case-insensitive, de-duped, sorted list (not a generator) so real ".MP4" drone
        # files match and the `if not video_paths` guard actually works.
        video_exts = {".mp4", ".mov"}
        video_paths = sorted({p for p in input_dir.rglob("*") if p.suffix.lower() in video_exts})
        if not video_paths:
            logger.warning(f"No videos found under {input_dir}")
            exit(1)

        # Run prediction
        output_dir.mkdir(parents=True, exist_ok=True)
        results = mass_prediction(video_paths=video_paths, current_output_dir=output_dir,
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
