from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QProgressBar, QLabel, QDialog, QDialogButtonBox, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QPixmap
import os
import sys
import tempfile

from shark_detector import SharkDetector

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int, int, int)  # current_video, total_videos, video_progress
    frame_updated = pyqtSignal(QPixmap)
    detection_finished = pyqtSignal(int, float, str)  # total_detections, total_time, results_dir
    error_occurred = pyqtSignal(str)

    def __init__(self, video_paths):
        super().__init__()
        self.video_paths = video_paths
        self.shark_detector = SharkDetector()

    def run(self):
        try:
            def progress_callback(current_video, total_videos, video_progress):
                self.progress_updated.emit(current_video, total_videos, video_progress)

            def frame_callback(pixmap):
                self.frame_updated.emit(pixmap)

            total_detections, total_time, results_dir = self.shark_detector.process_videos(
                self.video_paths, 
                progress_callback,
                frame_callback
            )
            self.detection_finished.emit(total_detections, total_time, results_dir)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self.shark_detector.cancel()
        
class DetectionsScreen(QMainWindow):
    go_to_verification = pyqtSignal(str)  # Modified to pass the results directory
    go_to_video_selection = pyqtSignal()

    def __init__(self, video_paths):
        super().__init__()
        self.video_paths = video_paths
        self.init_ui()
        self.start_detection()

    def init_ui(self):
        self.setWindowTitle("SharkEye - Detection in Progress")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("Shark Detection in Progress")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        # Overall progress bar
        self.overall_progress_bar = QProgressBar()
        main_layout.addWidget(self.overall_progress_bar)

        # Overall progress label
        self.overall_progress_label = QLabel("Processing video 0 of {}".format(len(self.video_paths)))
        self.overall_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.overall_progress_label)

        # Current video progress bar
        self.video_progress_bar = QProgressBar()
        main_layout.addWidget(self.video_progress_bar)

        # Current video label
        self.current_video_label = QLabel("Preparing...")
        self.current_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.current_video_label)

        # Frame display
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setMinimumSize(640, 480)
        main_layout.addWidget(self.frame_display)

        # Cancel button
        self.cancel_button = QPushButton("Cancel Detection")
        self.cancel_button.setIcon(QIcon("icons/cancel.png"))
        self.cancel_button.clicked.connect(self.cancel_detection)
        main_layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_detection(self):
        self.detection_thread = DetectionThread(self.video_paths)
        self.detection_thread.progress_updated.connect(self.update_progress)
        self.detection_thread.frame_updated.connect(self.update_frame)
        self.detection_thread.detection_finished.connect(self.show_results)
        self.detection_thread.error_occurred.connect(self.handle_error)
        self.detection_thread.start()

    def update_progress(self, current_video, total_videos, video_progress):
        overall_progress = int(((current_video - 1) * 100 + video_progress) / total_videos)
        self.overall_progress_bar.setValue(overall_progress)
        self.overall_progress_label.setText(f"Processing video {current_video} of {total_videos}")

        # Update current video progress
        self.video_progress_bar.setValue(video_progress)
        self.current_video_label.setText(f"Current video progress: {video_progress}%")

    def update_frame(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.frame_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.frame_display.setPixmap(scaled_pixmap)
        self.frame_display.repaint()  # Force immediate repaint

    def cancel_detection(self):
        reply = QMessageBox.question(self, "Cancel Detection", "Are you sure you want to cancel the detection process?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.detection_thread.shark_detector.cancel()
            self.detection_thread.wait()
            self.go_to_video_selection.emit()

    def show_results(self, total_detections, total_time, results_dir):
        if not os.access(results_dir, os.W_OK):
            QMessageBox.warning(self, "Read-Only File System", "The default results directory is read-only. Please choose a writable location.")
            new_results_dir = self.get_writable_directory()
            if not new_results_dir:
                self.go_to_video_selection.emit()
                return
            results_dir = new_results_dir

        dialog = ResultsDialog(total_detections, total_time, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result == "verify":
                self.go_to_verification.emit(results_dir)
            else:
                self.go_to_video_selection.emit()

    def get_writable_directory(self):
        while True:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory for Results")
            if not directory:  # User cancelled
                return None
            if os.access(directory, os.W_OK):
                return directory
            else:
                QMessageBox.warning(self, "Invalid Directory", "Selected directory is not writable. Please choose another.")

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred during detection: {error_message}")
        self.go_to_video_selection.emit()
        
class ResultsDialog(QDialog):
    def __init__(self, total_detections, total_time, parent=None):
        super().__init__(parent)
        self.total_detections = total_detections
        self.total_time = total_time
        self.result = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Detection Results")
        layout = QVBoxLayout(self)

        results_label = QLabel(f"Total Detections: {self.total_detections}\nTotal Time: {self.total_time:.2f} seconds")
        results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(results_label)

        button_box = QDialogButtonBox()
        verify_button = self.create_custom_button("Verify Result", QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(verify_button, QDialogButtonBox.ButtonRole.AcceptRole)
        return_button = button_box.addButton("Return to Video Selection", QDialogButtonBox.ButtonRole.RejectRole)

        verify_button.clicked.connect(self.verify_clicked)
        return_button.clicked.connect(self.return_clicked)

        layout.addWidget(button_box)
    
    def create_custom_button(self, text, role):
        button = QPushButton(text)
        button.setStyleSheet("""
            QPushButton {
                background-color: #1d2633;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
        """)
        return button

    def verify_clicked(self):
        self.result = "verify"
        self.accept()

    def return_clicked(self):
        self.result = "return"
        self.accept()