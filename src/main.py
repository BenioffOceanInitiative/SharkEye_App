import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QFileDialog, QListWidget, QLabel, QProgressBar, QListWidgetItem, QMessageBox, 
                             QSizePolicy, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QResizeEvent, QPixmap
import logging

from shark_detector import SharkDetector

class SharkEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_logging()
        self.init_ui()
        self.init_variables()
        logging.info("SharkEyeApp Initialization Complete")

    # Initialization methods
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def init_ui(self):
        """Initialize the user interface."""
        logging.info("Initializing SharkEyeApp UI")
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1280, 720)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout with zero margins
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.create_logo()

        # Content layout with margins
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)  # Set margins for content
        
        self.main_layout.addWidget(self.content_widget)
        
        self.setup_ui_components()

    def init_variables(self):
        """Initialize instance variables."""
        self.video_paths = []
        self.processing_thread = None
        self.progress_bar = None
        self.video_name_label = None

    def setup_ui_components(self):
        """Set up individual UI components."""
        self.create_video_selection_area()
        self.create_file_list()
        self.create_action_buttons()
        self.create_frame_display()
        self.create_results_label()

    def create_logo(self):
        """Create and add the logo to the layout."""
        self.logo_widget = QWidget()
        self.logo_widget.setStyleSheet("background-color: #1d2633;")
        logo_layout = QVBoxLayout(self.logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)  # Set zero margins
        
        self.logo_label = QLabel()
        logo_pixmap = self.load_logo()
        if logo_pixmap:
            self.logo_label.setPixmap(logo_pixmap)
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.logo_label.setText("SharkEye")
        
        logo_layout.addWidget(self.logo_label)
        self.main_layout.addWidget(self.logo_widget)

    def load_logo(self):
        """Load and scale the logo image."""
        logo_path = './assets/images/logo-white.png'
        original_pixmap = QPixmap(logo_path)
        if not original_pixmap.isNull():
            desired_width = 300
            aspect_ratio = original_pixmap.width() / original_pixmap.height()
            desired_height = int(desired_width / aspect_ratio)
            return original_pixmap.scaled(desired_width, desired_height, 
                                          Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
        return None

    def create_video_selection_area(self):
        """Create and add the video selection buttons to the layout."""
        selection_layout = QVBoxLayout()
        self.select_button = QPushButton("Select Videos")
        self.select_button.clicked.connect(self.select_videos)
        selection_layout.addWidget(self.select_button)

        button_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected_video)
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        selection_layout.addLayout(button_layout)
        self.content_layout.addLayout(selection_layout)

    def create_file_list(self):
        """Create and add the file list widget to the layout."""
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self.toggle_remove_button)
        self.file_list.setMaximumHeight(100)
        self.content_layout.addWidget(self.file_list)

    def create_action_buttons(self):
        """Create and add the start and cancel buttons to the layout."""
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setEnabled(False)
        action_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_detection)
        self.cancel_button.setEnabled(False)
        action_layout.addWidget(self.cancel_button)

        self.content_layout.addLayout(action_layout)

    def create_frame_display(self):
        """Create and add the frame display label to the layout."""
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setText("Select video(s) and Start Tracking!")
        self.content_layout.addWidget(self.frame_label)

    def create_results_label(self):
        """Create and add the results label to the layout."""
        self.results_label = QLabel()
        self.content_layout.addWidget(self.results_label)

    def create_progress_area(self):
        """Create and add the video name label and progress bar to the layout."""
        self.video_name_label = QLabel()
        self.video_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.video_name_label)

        self.progress_bar = QProgressBar()
        self.content_layout.addWidget(self.progress_bar)

    # Event handling methods
    def select_videos(self):
        """Open a file dialog for the user to select video files."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)")
        self.video_paths.extend(files)
        self.update_file_list()

    def toggle_remove_button(self):
        """Enable or disable the remove button based on file list selection."""
        self.remove_button.setEnabled(bool(self.file_list.selectedItems()))

    def remove_selected_video(self):
        """Remove the selected video from the list of video paths and update the UI."""
        selected_items = self.file_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            path = item.data(Qt.ItemDataRole.UserRole)
            self.video_paths.remove(path)
            self.file_list.takeItem(self.file_list.row(item))
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(bool(self.video_paths))
        self.start_button.setEnabled(bool(self.video_paths))

    def clear_selection(self):
        """Clear all selected video paths and update the file list."""
        self.video_paths.clear()
        self.update_file_list()

    def start_detection(self):
        """Initiate the shark detection process for the selected videos."""
        if not self.video_paths:
            return

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.create_progress_area()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        self.video_name_label.setText("Preparing...")

        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)

        try:
            logging.info("Creating VideoProcessingThread")
            self.init_shark_detector()
            self.processing_thread = VideoProcessingThread(self.shark_detector, self.video_paths, output_dir)
            self.connect_signals()
            self.processing_thread.start()
        except Exception as e:
            logging.error(f"Error Starting Detection: {str(e)}")
            self.show_error_message(f"Error Starting Detection: {str(e)}")

    def cancel_detection(self):
        """Cancel the ongoing detection process."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.shark_detector.cancel()
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.remove_progress_area()
            self.reset_ui()
            logging.info("Detection cancelled by user")

    # UI update methods
    def update_file_list(self):
        """Update the QListWidget with the current list of selected video files."""
        self.file_list.clear()
        for path in self.video_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(bool(self.video_paths))
        self.start_button.setEnabled(bool(self.video_paths))

    def update_progress(self, value):
        """Update the progress bar value if it exists."""
        if self.progress_bar is not None:
            self.progress_bar.setValue(value)
            QApplication.processEvents()

    def update_frame(self, pixmap):
        """Update the frame display with the latest processed frame."""
        scaled_pixmap = self.scale_pixmap(pixmap)
        self.frame_label.setPixmap(scaled_pixmap)

    def update_video_name(self, video_name):
        """Update the video name label with the current video being processed."""
        self.video_name_label.setText(f"Processing: {os.path.basename(video_name)}")

    def processing_finished(self, total_detections, total_time):
        """Handle the completion of the video processing."""
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        self.remove_progress_area()

        dialog = ResultsDialog(total_detections, total_time, self)
        dialog.exec()

    # Helper methods
    def init_shark_detector(self):
        """Initialize the SharkDetector object and load the model."""
        try:
            self.shark_detector = SharkDetector()
        except Exception as e:
            logging.error(f"Error initializing SharkEye: {str(e)}")
            self.show_error_message(f"Error initializing SharkEye: {str(e)}")

    def connect_signals(self):
        """Connect signals from SharkDetector and processing thread."""
        self.shark_detector.update_progress.connect(self.update_progress)
        self.shark_detector.update_frame.connect(self.update_frame)
        self.shark_detector.processing_finished.connect(self.processing_finished)
        self.shark_detector.error_occurred.connect(self.handle_error)
        self.shark_detector.current_video_changed.connect(self.update_video_name)
        self.processing_thread.error_occurred.connect(self.handle_error)

    def remove_progress_area(self):
        """Remove the progress bar and video name label from the layout."""
        if self.progress_bar:
            self.content_layout.removeWidget(self.progress_bar)
            self.progress_bar.deleteLater()
            self.progress_bar = None

        if self.video_name_label:
            self.content_layout.removeWidget(self.video_name_label)
            self.video_name_label.deleteLater()
            self.video_name_label = None

    def reset_ui(self):
        """Reset the UI to its initial state."""
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.frame_label.setText("Select video(s) and Start Tracking!")

    def handle_error(self, error_message):
        """Handle errors that occur during processing."""
        logging.error(f"Error occurred: {error_message}")
        self.show_error_message(error_message)
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def show_error_message(self, message):
        """Display an error message in a popup dialog."""
        QMessageBox.critical(self, "Error", message)

    def scale_pixmap(self, pixmap):
        """Scale the given pixmap to fit the frame label while maintaining aspect ratio."""
        label_size = self.frame_label.size()
        return pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def resizeEvent(self, event: QResizeEvent):
        """Handle the window resize event."""
        super().resizeEvent(event)
        if self.frame_label.pixmap() and not self.frame_label.pixmap().isNull():
            scaled_pixmap = self.scale_pixmap(self.frame_label.pixmap())
            self.frame_label.setPixmap(scaled_pixmap)

class ResultsDialog(QDialog):
    def __init__(self, total_detections, total_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results")
        self.setup_ui(total_detections, total_time)

    def setup_ui(self, total_detections, total_time):
        """Set up the user interface for the results dialog."""
        layout = QVBoxLayout()

        formatted_time = self.format_time(total_time)
        results_label = QLabel(f"Total Detections: {total_detections}\nTotal Time: {formatted_time}")
        layout.addWidget(results_label)

        button_box = QDialogButtonBox()
        run_additional = button_box.addButton("Run Additional Inference", QDialogButtonBox.ButtonRole.ActionRole)
        verify_detections = button_box.addButton("Verify Detections", QDialogButtonBox.ButtonRole.ActionRole)

        run_additional.clicked.connect(self.run_additional_inference)
        verify_detections.clicked.connect(self.verify_detections)

        layout.addWidget(button_box)
        self.setLayout(layout)
        
    def format_time(self, seconds: float) -> str:
        """
        Format the given number of seconds into a human-readable string.

        :param seconds: Number of seconds to format
        :return: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 120:
            return f"1 minute {seconds % 60:.0f} seconds"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes} minutes {remaining_seconds} seconds"

    def run_additional_inference(self):
        """
        Placeholder for running additional inference.
        This method should be implemented with the actual logic for additional inference.
        """
        logging.info("Running additional inference")
        self.accept()

    def verify_detections(self):
        """
        Placeholder for verifying detections.
        This method should be implemented with the actual logic for verifying detections.
        """
        logging.info("Verifying detections")
        self.accept()

class VideoProcessingThread(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, shark_detector, video_paths, output_dir):
        super().__init__()
        self.shark_detector = shark_detector
        self.video_paths = video_paths
        self.output_dir = output_dir

    def run(self):
        try:
            logging.info("Starting VideoProcessingThread")
            self.shark_detector.process_videos(self.video_paths, self.output_dir)
        except Exception as e:
            logging.error(f"Error in Video Processing Thread: {str(e)}")
            self.error_occurred.emit(str(e))

if __name__ == "__main__":
    try:
        logging.info("Starting Application")
        app = QApplication(sys.argv)
        window = SharkEyeApp()
        logging.info("Showing Main Window")
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")