import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QFileDialog, QListWidget, QLabel, QProgressBar, QListWidgetItem, QMessageBox, 
                             QSizePolicy, QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent, QPixmap
import logging

from video_selection_area import VideoSelectionArea
from video_processing_thread import VideoProcessingThread
from action_buttons import ActionButtons
from results_dialog import ResultsDialog
from verification_window import VerificationWindow
from shark_detector import SharkDetector

def setup_logging():
    try:
        log_file = os.path.join(os.getcwd(), "sharkeye_log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging initialized")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")

setup_logging()
logging.info(f"Python version: {sys.version}")
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"sys.executable: {sys.executable}")
logging.info(f"sys._MEIPASS: {getattr(sys, '_MEIPASS', 'Not running from PyInstaller')}")
logging.info(f"sys.argv: {sys.argv}")

def get_base_dir():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, use the sys._MEIPASS
        return sys._MEIPASS
    else:
        # If running in development, use the directory of the script
        return os.path.dirname(os.path.abspath(__file__))

def get_writable_dir():
    if getattr(sys, 'frozen', False):
        # We are running in a bundle
        app_dir = Path(sys.executable).parent.parent.parent.parent if sys.platform == 'darwin' else Path(sys.executable).parent
    else:
        # We are running in a normal Python environment
        app_dir = Path(__file__).parent

    results_dir = app_dir / "results"
    
    try:
        results_dir.mkdir(exist_ok=True)
        # Test if we can write to this directory
        test_file = results_dir / 'test_write.txt'
        test_file.write_text('test')
        test_file.unlink()
    except (PermissionError, OSError):
        # If we can't write to the app directory, fall back to a user-writable location
        fallback_dir = Path.home() / "results"
        fallback_dir.mkdir(exist_ok=True)
        results_dir = fallback_dir
        print(f"Warning: Could not write to {app_dir}. Using {fallback_dir} instead.")

    return str(results_dir)

class SharkEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_variables()
        self.init_ui()
        logging.info("SharkEyeApp Initialization Complete")
        
    def init_variables(self):
        """Initialize instance variables."""
        self.processing_thread = None
        self.progress_bar = None
        self.video_name_label = QLabel()
        self.video_selection_area = None
        self.action_buttons = None
        self.frame_label = None
        self.results_label = None

    def init_ui(self):
        """Initialize the user interface."""
        logging.info("Initializing SharkEyeApp UI")
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1280, 720)
        self.central_widget = QWidget()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.central_widget)         
        self.setCentralWidget(self.stack)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.create_logo()

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        
        self.main_layout.addWidget(self.content_widget)
        
        self.setup_ui_components()

    def setup_ui_components(self):
        """Set up individual UI components."""
        self.video_selection_area = VideoSelectionArea()
        self.content_layout.addWidget(self.video_selection_area)

        self.action_buttons = ActionButtons()
        self.content_layout.addWidget(self.action_buttons)

        self.create_frame_display()
        self.create_results_label()

        # Connect signals
        self.video_selection_area.videos_selected.connect(self.on_videos_selected)
        self.video_selection_area.selection_cleared.connect(self.on_selection_cleared)
        self.action_buttons.start_clicked.connect(self.start_detection)
        self.action_buttons.cancel_clicked.connect(self.cancel_detection)
    
    def on_videos_selected(self, video_paths):
        self.video_paths = video_paths
        self.action_buttons.set_start_enabled(bool(video_paths))
        logging.info(f"Selected videos: {video_paths}")

    def on_selection_cleared(self):
        self.video_paths = []
        self.action_buttons.set_start_enabled(False)
        logging.info("Video selection cleared")

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

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def load_logo(self):
        """Load and scale the logo image."""
        logo_path = self.resource_path('assets/images/logo-white.png')
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
        logging.info("Start Detection button clicked")
        if not self.video_paths:
            logging.warning("No video paths selected")
            return

        try:
            logging.info("Disabling start button and enabling cancel button")
            self.action_buttons.set_start_enabled(False)
            self.action_buttons.set_cancel_enabled(True)

            logging.info("Creating progress area")
            self.create_progress_area()
            self.progress_bar.setValue(0)
            self.progress_bar.setRange(0, 100)
            self.video_name_label.setText("Preparing...")
            self.video_name_label.show()

            output_dir = get_writable_dir()
            logging.info(f"Attempting to create output directory: {output_dir}")
            try:
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Output directory created successfully: {output_dir}")
            except Exception as e:
                logging.exception(f"Error creating output directory: {str(e)}")
                raise

            logging.info("Initializing SharkDetector")
            self.init_shark_detector()

            logging.info("Creating VideoProcessingThread")
            self.processing_thread = VideoProcessingThread(self.shark_detector, self.video_paths, output_dir)

            logging.info("Connecting signals")
            self.connect_signals()

            logging.info("Starting processing thread")
            self.processing_thread.start()

        except Exception as e:
            logging.exception(f"Error in start_detection: {str(e)}")
            self.show_error_message(f"Error Starting Detection: {str(e)}")
            self.reset_ui()

    def cancel_detection(self):
        """Cancel the ongoing detection process."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.shark_detector.cancel()
            self.processing_thread.quit()
            self.processing_thread.wait()
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
        self.action_buttons.set_start_enabled(bool(self.video_paths))

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
        if self.video_name_label and self.video_name_label.isVisible():
            self.video_name_label.setText(f"Processing: {os.path.basename(video_name)}")
        else:
            logging.warning("Video name label is not visible or not initialized")

    # Helper methods
    def init_shark_detector(self):
        try:
            logging.info("Creating SharkDetector instance")
            self.shark_detector = SharkDetector()
            logging.info("SharkDetector instance created successfully")
        except Exception as e:
            logging.exception(f"Error initializing SharkDetector: {str(e)}")
            raise  # Re-raise the exception to be caught in start_detection

    def connect_signals(self):
        """Connect signals from SharkDetector and processing thread."""
        self.shark_detector.update_progress.connect(self.update_progress)
        self.shark_detector.update_frame.connect(self.update_frame)
        self.shark_detector.processing_finished.connect(self.on_video_processed)
        self.shark_detector.all_videos_processed.connect(self.on_all_videos_processed)
        self.shark_detector.error_occurred.connect(self.handle_error)
        self.shark_detector.current_video_changed.connect(self.update_video_name)
        self.processing_thread.error_occurred.connect(self.handle_error)
        
    def on_video_processed(self, detections, processing_time):
        """Handle completion of a single video."""
        logging.info(f"Video processed with {detections} detections in {processing_time:.2f} seconds")

    def on_all_videos_processed(self, total_detections, total_time):
        """Handle completion of all videos."""
        self.action_buttons.set_start_enabled(True)
        self.action_buttons.set_cancel_enabled(False)
        
        self.reset_ui()
        
        dialog = ResultsDialog(total_detections, total_time, self)
        dialog.run_additional_inference.connect(self.run_additional_inference)
        dialog.verify_detections.connect(self.verify_detections)
        dialog.exec()

    def remove_progress_area(self):
        """Remove the progress bar and video name label from the layout."""
        if self.progress_bar:
            self.content_layout.removeWidget(self.progress_bar)
            self.progress_bar.deleteLater()
            self.progress_bar = None

        if self.video_name_label:
            self.video_name_label.hide()
            self.content_layout.removeWidget(self.video_name_label)

    def reset_ui(self):
        """Reset the UI to its initial state."""
        self.action_buttons.set_start_enabled(True)
        self.action_buttons.set_cancel_enabled(False)
        self.frame_label.setText("Select video(s) and Start Tracking!")
        self.remove_progress_area()

    def handle_error(self, error_message):
        """Handle errors that occur during processing."""
        logging.error(f"Error occurred: {error_message}")
        self.show_error_message(error_message)
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        self.action_buttons.set_start_enabled(True)
        self.action_buttons.set_cancel_enabled(False)

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
    
    def run_additional_inference(self):
        # Implement additional inference logic here
        logging.info("Running additional inference")
        self.reset_ui()

    def verify_detections(self):
        """Opens Verification Window Widget"""
        logging.info("Verifying detections")
        results_dir = get_writable_dir()
        self.verification_window = VerificationWindow(results_dir)
        self.stack.addWidget(self.verification_window)
        self.stack.setCurrentWidget(self.verification_window)

if __name__ == "__main__":
    try:
        logging.info("Starting Application")
        app = QApplication(sys.argv)
        window = SharkEyeApp()
        logging.info("Showing Main Window")
        window.show()
        exit_code = app.exec()
        logging.info(f"Application exiting with code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logging.exception(f"Unhandled exception: {str(e)}")
        sys.exit(1)
