import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QFileDialog, QListWidget, QLabel, QProgressBar, QListWidgetItem, QMessageBox, 
                             QSizePolicy, QDialog, QDialogButtonBox, QSlider, QStackedWidget, QComboBox)
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
        self.stack = QStackedWidget()
        self.stack.addWidget(self.central_widget)         
        self.setCentralWidget(self.stack)
        
        # Main layout with zero margins
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.create_logo()

        # Content layout with margins
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        
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

    def processing_finished(self, total_detections, total_time):
        """
        Handle the completion of the video processing.
        This method updates the UI and displays the results dialog.

        :param total_detections: Total number of shark detections across all videos
        :param total_time: Total processing time in seconds
        """ 
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        dialog = ResultsDialog(total_detections, total_time, self)
        dialog.exec()
    
    def run_additional_inference(self):
        # Implement additional inference logic here
        logging.info("Running additional inference")

    def verify_detections(self):
        """Opens Verification Window Widget"""
        logging.info("Verifying detections")
        self.verification_window = VerificationWindow()
        self.stack.addWidget(self.verification_window)
        self.stack.setCurrentWidget(self.verification_window)

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

        self.button_box = QDialogButtonBox()
        run_additional = self.button_box.addButton("Run Additional Inference", QDialogButtonBox.ButtonRole.ActionRole)
        verify_detections = self.button_box.addButton("Verify Detections", QDialogButtonBox.ButtonRole.ActionRole)

        run_additional.clicked.connect(self.reject)
        verify_detections.clicked.connect(self.accept)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.accepted.connect(self.parent().verify_detections)
        self.rejected.connect(self.parent().run_additional_inference)
        
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

class VerificationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Detection Verification")
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.resize(self.initial_width, self.initial_height)
        
        self.experiments = os.listdir("./results/")
        self.experiments.sort()
        self.experiments.reverse()
    
        # Shark Buttons 
        marking_layout = QHBoxLayout()

        self.mark_shark_button = QPushButton("Shark")
        self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.mark_shark_button.clicked.connect(self.mark_as_shark)

        self.unmark_shark_button = QPushButton("No Shark")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px;height: 30px;")
        self.unmark_shark_button.clicked.connect(self.unmark_shark) 

        marking_layout.addWidget(self.mark_shark_button)
        marking_layout.addWidget(self.unmark_shark_button)      

        self.finish_verification_button = QPushButton("Finish Verification")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.finish_verification_button.clicked.connect(self.finish_verifications)
        
        self.load_experiment_frames(0)
        
        # Slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)

        # Experiment Selection
        self.experiment_label = QComboBox()
        self.experiment_label.addItems(self.experiments)
        self.experiment_label.currentIndexChanged.connect(self.select_experiment)

        # Display Frame
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setMinimumSize(640, 480)
        self.frame_display.resize(self.disply_width, self.display_height)
        
        self.file_path = QLabel()
        self.file_path.setStyleSheet("color: black;background-color: white; border-radius: 4px")
            
        self.verified_sharks = [True for x in range(len(self.frames))]

        # Main Layout 
        tracker_layout = QVBoxLayout()
        tracker_layout.addWidget(self.experiment_label)
        tracker_layout.addWidget(self.frame_display)        
        tracker_layout.addWidget(self.frame_slider)
        tracker_layout.addLayout(marking_layout)
        tracker_layout.addWidget(self.finish_verification_button)
        self.frame_slider.valueChanged.connect(self.value_change)
        self.setLayout(tracker_layout)
        self.select_experiment(0)
    
    def load_experiment_frames(self, index):
        self.last_run = self.experiments[index]
        self.frames = ["/".join((f"./results/{self.last_run}/bounding_boxes", f)) for f in os.listdir(f"./results/{self.last_run}/bounding_boxes")]

    def finish_verifications(self):
        """Deletes images from 'frames' and 'bounding_boxes' folders marked as having no sharks"""
        for index, x in enumerate(self.verified_sharks):
            if x == False:
                os.remove(self.frames[index])
                os.remove(self.frames[index].replace("bounding_boxes", "frames"))
        self.parent().setCurrentWidget(self.parent().parent().central_widget)
        self.parent().removeWidget(self.parent().parent().verification_window)
    
    def mark_as_shark(self):
        """Marks frame as having a shark."""
        if self.verified_sharks[self.current_frame] != True:
            self.verified_sharks[self.current_frame] = True
            self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; width: 100px;height: 30px;")
            self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px;height: 30px;")

    def unmark_shark(self):
        """Marks frame as having no shark. Frame will be deleted with finish_verification call"""
        if self.verified_sharks[self.current_frame] == True:
            self.verified_sharks[self.current_frame] = False
            self.mark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px;height: 30px;")
            self.unmark_shark_button.setStyleSheet("background-color: red; color: white; border-radius: 4px; width: 100px;height: 30px;")

    def value_change(self):
        """Handles display of frames and colors of verification buttons"""
        index = self.frame_slider.value()
        frame = QPixmap(self.frames[index]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_frame = index

        self.frame_display.setPixmap(frame)
        self.file_path.setText(self.frames[index])
        self.update_button_styles()

    def update_button_styles(self):
        """Changes colors of marking buttons based on selection"""
        if len(self.verified_sharks) > 0:
            if self.verified_sharks[self.current_frame] == True:
                self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; width: 100px; height: 30px;")
                self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px; height: 30px;")
            else:
                self.mark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px; height: 30px;")
                self.unmark_shark_button.setStyleSheet("background-color: red; color: white; border-radius: 4px; width: 100px; height: 30px;")

    def hide_ui_elements(self):
        """Makes display frame, verification buttons and slider disappear"""
        self.frame_slider.hide()
        
        self.frame_display.setText("Select an experiment to start verifying detections")
        # self.placeholder_pixmap = QPixmap(self.disply_width, self.display_height).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        # self.placeholder_pixmap.fill(Qt.GlobalColor.transparent)
        # self.frame_display.setPixmap(self.placeholder_pixmap)
        
        self.mark_shark_button.setDisabled(True)
        self.unmark_shark_button.setDisabled(True)
        self.finish_verification_button.setDisabled(True)

        self.mark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")

    def show_ui_elements(self):
        """Makes display frame, verification buttons and slider appear"""
        self.frame_slider.show()
        self.frame_display.show()
        
        self.mark_shark_button.setEnabled(True)
        self.unmark_shark_button.setEnabled(True)
        self.finish_verification_button.setEnabled(True)
        
        self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; width: 100px;height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; width: 100px;height: 30px;")

    def select_experiment(self, index):
        """Selects experiment to run verification on""" 
        self.load_experiment_frames(index)

        if len(self.frames) > 0: 
            self.current_frame = 0
            self.frame_slider.setValue(0)
            frame = QPixmap(self.frames[self.current_frame]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            self.frame_display.setPixmap(frame)

            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(len(self.frames) - 1)

            self.verified_sharks = [True for x in range(len(self.frames))]

            self.show_ui_elements()
        else:
            self.hide_ui_elements()

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
