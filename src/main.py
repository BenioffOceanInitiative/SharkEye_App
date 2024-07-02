import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QFileDialog, QListWidget, QLabel, QProgressBar, QListWidgetItem, QMessageBox, 
                             QSizePolicy, QDialog, QDialogButtonBox, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QResizeEvent, QIcon, QPixmap

from shark_detector import SharkDetector

class SharkEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing SharkEyeApp")
        self.setWindowTitle("SharkEye")
        # self.setWindowIcon(QIcon('./assets/images/logo-white.png'))
        self.setGeometry(100, 100, 1280, 720)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.setup_ui()

        self.video_paths = []
        self.processing_thread = None
        
        print("SharkEyeApp Initialization Complete")
    
    def init_shark_detector(self):
        try:
            self.shark_detector = SharkDetector()
            self.shark_detector.load_model('./model_weights/train6-weights-best.pt')
        except Exception as e:
            print(f"Error initializing SharkEye: {str(e)}")
            self.show_error_message(f"Error initializing SharkEye: {str(e)}")

    def handle_error(self, error_message):
        print(f"Error occurred: {error_message}")
        self.show_error_message(error_message)
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def setup_ui(self):
        """
        Set up the user interface components of the application.
        This method creates and arranges all the widgets in the main window.
        """
        self.logo_label = QLabel()
        logo_pixmap = self.create_logo()
        if logo_pixmap:
            self.logo_label.setPixmap(logo_pixmap)
            self.logo_label.setStyleSheet("background-color: #1d2633;")
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.logo_label.setText("SharkEye")
        self.layout.addWidget(self.logo_label)

        # Select Videos button (spans across)
        self.select_button = QPushButton("Select Videos")
        self.select_button.clicked.connect(self.select_videos)
        self.layout.addWidget(self.select_button)

        # Remove Selected and Clear All buttons
        button_layout = QHBoxLayout()
        
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected_video)
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        self.layout.addLayout(button_layout)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self.toggle_remove_button)
        self.file_list.setMaximumHeight(100)
        self.layout.addWidget(self.file_list)

        # Start Detection button
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setEnabled(False)  # Initially disabled
        self.layout.addWidget(self.start_button)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_detection)
        self.cancel_button.setEnabled(False)
        self.layout.addWidget(self.cancel_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setText("Select video(s) and Start Tracking!")
        self.layout.addWidget(self.frame_label)

        # Results label
        self.results_label = QLabel()
        self.layout.addWidget(self.results_label)
        
    def create_logo(self):
        logo_path = './assets/images/logo-white.png'
        original_pixmap = QPixmap(logo_path)
        if not original_pixmap.isNull():
            # Define the desired width, and scale height proportionally
            desired_width = 200
            aspect_ratio = original_pixmap.width() / original_pixmap.height()
            desired_height = int(desired_width / aspect_ratio)

            scaled_pixmap = original_pixmap.scaled(desired_width, desired_height, 
                                                Qt.AspectRatioMode.KeepAspectRatio, 
                                                Qt.TransformationMode.SmoothTransformation)
            return scaled_pixmap
        return None

    def select_videos(self):
        """
        Open a file dialog for the user to select video files.
        The selected video paths are added to the video_paths list and the file list is updated.
        """
        
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)")
        self.video_paths.extend(files)
        self.update_file_list()

    def clear_selection(self):
        self.video_paths.clear()
        self.update_file_list()

    def update_file_list(self):
        """
        Update the QListWidget with the current list of selected video files.
        This method also enables/disables buttons based on the selection state.
        """
        
        self.file_list.clear()
        for path in self.video_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(bool(self.video_paths))
        self.start_button.setEnabled(bool(self.video_paths))

    def toggle_remove_button(self):
        self.remove_button.setEnabled(bool(self.file_list.selectedItems()))

    def remove_selected_video(self):
        selected_items = self.file_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            path = item.data(Qt.ItemDataRole.UserRole)
            self.video_paths.remove(path)
            self.file_list.takeItem(self.file_list.row(item))
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(bool(self.video_paths))
        self.start_button.setEnabled(bool(self.video_paths))

    def start_detection(self):
        """
        Initiate the shark detection process for the selected videos.
        This method sets up the processing thread and connects the necessary signals.
        """
        
        if not self.video_paths:
            return

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)

        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)

        try:
            print("Creating VideoProcessingThread")
            self.init_shark_detector()
            self.processing_thread = VideoProcessingThread(self.shark_detector, self.video_paths, output_dir)
            self.shark_detector.update_progress.connect(self.update_progress)
            self.shark_detector.update_frame.connect(self.update_frame)
            self.shark_detector.processing_finished.connect(self.processing_finished)
            self.shark_detector.error_occurred.connect(self.handle_error)
            self.processing_thread.error_occurred.connect(self.handle_error)
            self.processing_thread.start()
        except Exception as e:
            print(f"Error Starting Detection: {str(e)}")
            self.show_error_message(f"Error Starting Detection: {str(e)}")

    def cancel_detection(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.shark_detector.cancel()
            self.processing_thread.wait()
            self.processing_finished(0, 0)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_frame(self, pixmap):
        """
        Update the frame display with the latest processed frame.
        
        :param pixmap: QPixmap containing the latest processed frame
        """
        
        scaled_pixmap = self.scale_pixmap(pixmap)
        self.frame_label.setPixmap(scaled_pixmap)

    def scale_pixmap(self, pixmap):
        # Get the size of the frame_label
        label_size = self.frame_label.size()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        return scaled_pixmap

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        # If there's a pixmap, rescale it when the window is resized
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

class ResultsDialog(QDialog):
    def __init__(self, total_detections, total_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results")
        layout = QVBoxLayout()

        results_label = QLabel(f"Total Detections: {total_detections}\nTotal Time: {total_time:.2f} seconds")
        layout.addWidget(results_label)

        button_box = QDialogButtonBox()
        run_additional = button_box.addButton("Run Additional Inference", QDialogButtonBox.ButtonRole.ActionRole)
        verify_detections = button_box.addButton("Verify Detections", QDialogButtonBox.ButtonRole.ActionRole)

        run_additional.clicked.connect(self.run_additional_inference)
        verify_detections.clicked.connect(self.verify_detections)

        layout.addWidget(button_box)
        self.setLayout(layout)

    def run_additional_inference(self):
        # Implement additional inference logic here
        print("Running additional inference")
        self.accept()

    def verify_detections(self):
        # Implement detection verification logic here
        print("Verifying detections")
        self.review_window = ReviewWindow()
        self.review_window.show()
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
            print("Starting VideoProcessingThread")
            self.shark_detector.process_videos(self.video_paths, self.output_dir)
        except Exception as e:
            print(f"Error in Video Processing Thread: {str(e)}")
            self.error_occurred.emit(str(e))

class ReviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Tracking Results")
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.resize(self.initial_width, self.initial_height)
        
        experiments = os.listdir("./results/")
        experiments.sort()

        self.false_flags = []

        # does this need logic to handle no experiments?

        last_run = experiments[-1]

        self.frames = f"./results/{last_run}/frames"

        # Slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)

        # Display Frame
        self.frame_display = QLabel()
        self.frame_display.resize(self.disply_width, self.display_height)
        
        self.file_path = QLabel()
        self.file_path.setStyleSheet("color: white")

        if len(self.frames) > 0:
            self.frame_display.setPixmap(QPixmap(self.frames[0]).scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio))
            self.file_path.setText(self.frames[0])
            self.current_frame = self.frames[0]

        # Bbox Info
        self.bbox_list = QListWidget()
        self.bbox_list.setStyleSheet("background-color: black; color: white")
        self.bbox_list.setMaximumHeight(100)

        # buttons to add
        add_remove_layout = QHBoxLayout()

        self.add_frame_button = QPushButton("Flag as false positive")
        self.add_frame_button.setStyleSheet("background-color: #082f54; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.add_frame_button.clicked.connect(self.flag_false_positive)

        self.remove_frame_button = QPushButton("Remove flag")
        self.remove_frame_button.setStyleSheet("background-color: #f22613; color: white; border-radius: 4px; width: 100px;height: 30px;")
        self.remove_frame_button.clicked.connect(self.remove_false_positive) 

        add_remove_layout.addWidget(self.add_frame_button)
        add_remove_layout.addWidget(self.remove_frame_button)      

        frame_review_layout = QVBoxLayout()      

        # self.delete_annotation
        
        # Layout 
        tracker_layout = QVBoxLayout()
        tracker_layout.addWidget(self.file_path)
        tracker_layout.addWidget(self.frame_display)        
        tracker_layout.addWidget(self.frame_slider)
        tracker_layout.addLayout(add_remove_layout)
        tracker_layout.addWidget(self.bbox_list)
        self.frame_slider.valueChanged.connect(self.valuechange)
        self.setLayout(tracker_layout)
        
    def flag_false_positive(self):
        if self.current_frame not in self.false_flags:
            self.bbox_list.addItem(self.current_frame)
            self.false_flags.append(self.current_frame)

    def remove_false_positive(self):
        if self.current_frame in self.false_flags:
            print(self.bbox_list.selectedItems())
            self.bbox_list.takeItem(self.bbox_list.row(self.current_frame))
            self.false_flags.remove(self.current_frame)

    def valuechange(self):
        index = self.frame_slider.value()
        frame = QPixmap(self.frames[index]).scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        
        self.current_frame = self.frames[index]

        self.frame_display.setPixmap(frame)
        self.file_path.setText(self.frames[index])

if __name__ == "__main__":
    try:
        print("Starting Application")
        app = QApplication(sys.argv)
        window = SharkEyeApp()
        print("Showing Main Window")
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")

