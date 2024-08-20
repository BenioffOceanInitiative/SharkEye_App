import sys
import os
from datetime import datetime
import shutil
import zipfile
import requests
import pandas as pd 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QIcon
from video_player import CustomSlider, VideoPlayer

class VerificationScreen(QMainWindow):
    go_to_video_selection = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_experiment = None
        self.current_detection_index = 0
        self.detections = []
        self.experiment_map = {}  # Mapping between formatted names and original directory names
        self.api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload"
        self.progress_dialog = None
        self.upload_thread = None
        self.is_uploading = False
        self.video_paths = []
        self.result_dict = {}
        self.current_video = None
        self.video_paths = []
        self.result_dict = {}
        self.current_video = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SharkEye - Verification")
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Experiment selection
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self.load_experiment)
        main_layout.addWidget(self.experiment_combo)

        # Image display
        self.prev_video_button = QPushButton("<")
        self.prev_video_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: grey;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: transparent;
            }
            QPushButton:disabled {
                background-color: transparent;
                color: transparent;
            }
        """)
        self.prev_video_button.clicked.connect(self.prev_video)
        self.prev_video_button.setMinimumWidth(55)
        self.prev_video_button.setMinimumHeight(480)
        
        self.image_label = VideoPlayer()
        
        self.next_video_button = QPushButton(">") 
        self.next_video_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: grey;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: transparent;
            }
            QPushButton:disabled {
                background-color: transparent;
                color: transparent;
            }
        """)
        self.next_video_button.setMinimumWidth(55) 
        self.next_video_button.setMinimumHeight(480)
        self.next_video_button.clicked.connect(self.next_video)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.prev_video_button)
        video_layout.addWidget(self.image_label)
        video_layout.addWidget(self.next_video_button)
        
        main_layout.addLayout(video_layout)

        # Place holder image for when no experiments
        self.empty_frame = QLabel("No experiments found with detections to verify")
        self.empty_frame.resize(self.image_label.size())
        self.empty_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_frame.hide()
        main_layout.addWidget(self.empty_frame)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_detection)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_detection)
        # Apply custom styling to buttons
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)
        
        self.detection_count_label = QLabel("No detections")
        self.detection_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.detection_count_label)
        nav_layout.addWidget(self.next_button)

        # Classification dropdown
        self.classification_combo = QComboBox()
        self.classification_combo.addItems(["Shark", "Kelp", "Dolphin", "Kayaker", "Surfer", "Paddleboarder", "Bat Ray", "Boat", "Other"])
        self.classification_combo.currentIndexChanged.connect(self.update_classification)
        main_layout.addWidget(self.classification_combo)

        # Save Results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_and_upload_results)
        main_layout.addWidget(self.save_button)

        # Return to Video Selection button
        self.return_button = self.create_button("Return to Video Selection", self.go_to_video_selection.emit, "icons/return.png", custom_color=True)
        main_layout.addWidget(self.return_button)

        # Load experiments
        self.refresh_experiments()

    def create_button(self, text, slot, icon_path, custom_color=False):
        button = QPushButton(text)
        button.clicked.connect(slot)
        if icon_path:
            button.setIcon(QIcon(icon_path))
        if custom_color:
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

    def load_experiments(self):
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        if os.path.exists(results_dir):
            experiments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            experiments.sort(reverse=True)  # Most recent first
            
            valid_experiments = []
            for exp in experiments:
                bounding_box_dir = os.path.join(results_dir, exp, 'bounding_boxes')
                if os.path.exists(bounding_box_dir) and any(os.scandir(bounding_box_dir)):
                    formatted_name = self.format_experiment_name(exp)
                    self.experiment_map[formatted_name] = exp
                    valid_experiments.append(formatted_name)

            self.experiment_combo.clear()
            self.experiment_combo.addItems(valid_experiments)

        if self.experiment_combo.count() > 0:
            self.experiment_combo.setCurrentIndex(0)
            self.load_experiment(0)
        else:
            self.clear_display()
            QMessageBox.information(self, "No Valid Experiments", "There are no experiments with detections to verify.")
    
    def refresh_experiments(self):
        """Clear and reload the experiments list."""
        self.experiment_combo.clear()
        self.experiment_map.clear()
        self.load_experiments()
        
    def clear_display(self):
        """Clear the display when no experiments are available."""
        self.image_label.hide()
        self.empty_frame.show()
        self.classification_combo.setEnabled(False)
        self.save_button.setEnabled(False)  # Disable the save button when there are no detections

    def format_experiment_name(self, experiment_name):
        try:
            # Parse the original datetime string
            dt = datetime.strptime(experiment_name, "%Y-%m-%d_%H-%M-%S")
            # Format it in a more readable way
            return dt.strftime("%b %d, %Y at %I:%M:%S %p")
        except ValueError:
            # If parsing fails, return the original string
            return experiment_name
        
    def load_experiment(self, index):
        if index >= 0:
            formatted_name = self.experiment_combo.itemText(index)
            self.current_experiment = self.experiment_map.get(formatted_name)
            if self.current_experiment:
                self.load_detections()
                self.separate_video_detections()
                self.separate_video_detections()
                self.current_detection_index = 0
                self.show_current_detection()
            else:
                print(f"Error: No mapping found for {formatted_name}")

    def separate_video_detections(self):
        if self.current_experiment:
            experiment_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", self.current_experiment)
            bounding_boxes_dir = os.path.join(experiment_dir, "bounding_boxes")
            if os.path.exists(bounding_boxes_dir):
                result_data_path = os.path.join(experiment_dir, "results.csv")
                result_data = pd.read_csv(result_data_path)
                self.result_dict = result_data.groupby(result_data.columns[3])[result_data.columns[2]].apply(list).to_dict() 
                self.current_video = list(self.result_dict.keys())[0]
                self.prev_video_button.setEnabled(False)
                if len(list(self.result_dict.keys())) > 1:
                    self.next_video_button.setEnabled(True)
                else:
                    self.next_video_button.setEnabled(False)
                self.empty_frame.hide()
                self.image_label.show()
                self.save_button.setEnabled(True)

    def change_video(self):
        frame_numbers = {int((1000/30) * int(frame.split("frame")[1].strip(".jpg"))):frame for frame in self.result_dict[self.current_video]}

        self.image_label.set_video_dir(self.current_video)
        self.image_label.set_interesting_points(frame_numbers)
    
    def prev_video(self):
        if self.current_video:
            self.next_video_button.setEnabled(True)
            if list(self.result_dict.keys()).index(self.current_video) > 0:
                self.current_video = list(self.result_dict.keys())[list(self.result_dict.keys()).index(self.current_video) - 1]
                self.change_video()
                if list(self.result_dict.keys()).index(self.current_video) <= 0:
                    self.prev_video_button.setEnabled(False)

    def next_video(self):
        if self.current_video:
            self.prev_video_button.setEnabled(True)
            if list(self.result_dict.keys()).index(self.current_video) + 1 < len(list(self.result_dict.keys())):
                self.current_video = list(self.result_dict.keys())[list(self.result_dict.keys()).index(self.current_video) + 1]
                self.change_video()
                if list(self.result_dict.keys()).index(self.current_video) + 1 >= len(list(self.result_dict.keys())):
                    self.next_video_button.setEnabled(False)

    def load_detections(self):
        if self.current_experiment:
            experiment_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", self.current_experiment)
            bounding_boxes_dir = os.path.join(experiment_dir, "bounding_boxes")
            self.detections = []
            if os.path.exists(bounding_boxes_dir):
                for filename in os.listdir(bounding_boxes_dir): 
                    if filename.endswith(".jpg"):
                        file_path = os.path.join(bounding_boxes_dir, filename)
                        self.detections.append({"path": file_path, "classification": "Shark"})
                        self.separate_video_detections()
                        self.change_video()
                        self.separate_video_detections()
                        self.change_video()
            print(f"Loaded {len(self.detections)} detections from {bounding_boxes_dir}")

    def load_results(self, results_dir):
        if not os.path.exists(results_dir):
            QMessageBox.warning(self, "Error", f"Results directory not found: {results_dir}")
            return

        # Update the experiment list
        self.experiment_combo.clear()
        self.experiment_map = {}
        
        formatted_name = self.format_experiment_name(os.path.basename(results_dir))
        self.experiment_map[formatted_name] = os.path.basename(results_dir)
        self.experiment_combo.addItem(formatted_name)
        
        # Load other experiments from the parent directory
        parent_dir = os.path.dirname(results_dir)
        for item in os.listdir(parent_dir):
            full_path = os.path.join(parent_dir, item)
            if os.path.isdir(full_path) and full_path != results_dir:
                formatted_name = self.format_experiment_name(item)
                self.experiment_map[formatted_name] = item
                self.experiment_combo.addItem(formatted_name)

        # Set the current experiment to the one we just loaded
        self.experiment_combo.setCurrentIndex(0)
        
        # Load the detections for this experiment
        self.load_experiment(0)

    def show_current_detection(self):
        if not self.detections:
            print("Clearing Display")
            self.clear_display()
            return

        detection = self.detections[self.current_detection_index]
        pixmap = QPixmap(detection["path"])
        self.classification_combo.setCurrentText(detection["classification"])
        self.classification_combo.setEnabled(True)
        
        # Update detection count label
        total_detections = len(self.detections)
        self.detection_count_label.setText(f"Detection {self.current_detection_index + 1} of {total_detections}")

    def show_previous_detection(self):
        if self.current_detection_index > 0:
            self.current_detection_index -= 1
            self.show_current_detection()

    def show_next_detection(self):
        if self.current_detection_index < len(self.detections) - 1:
            self.current_detection_index += 1
            self.show_current_detection()

    def update_classification(self, index):
        if self.detections:
            self.detections[self.current_detection_index]["classification"] = self.classification_combo.currentText()

    def save_and_upload_results(self):
        if not self.detections:
            QMessageBox.warning(self, "No Data", "There are no detections to save.")
            return

        experiment_dir = os.path.dirname(os.path.dirname(self.detections[0]['path']))
        false_positives_dir = os.path.join(experiment_dir, "false_positives")
        os.makedirs(false_positives_dir, exist_ok=True)

        moved_count = 0
        new_detections = []

        for detection in self.detections:
            if detection['classification'] != "Shark":
                filename = os.path.basename(detection['path'])
                frame_number = int(filename.split("frame")[1].strip(".jpg"))
                new_path = os.path.join(false_positives_dir, filename)
                shutil.move(detection['path'], new_path)
                moved_count += 1
                self.image_label.remove_indicator(frame_number)
            else:
                new_detections.append(detection)

        self.detections = new_detections
        self.current_detection_index = 0

        if self.detections:
            self.show_current_detection()
        else:
            self.image_label.hide()
            self.empty_frame.show()
            self.save_button.setEnabled(True)

        message = ''
        if moved_count > 0:
            message = f"Moved {moved_count} non-shark{'s' if moved_count != 1 else ''} detections to false_positives folder.\n"
        message += f"\n{len(self.detections)} Shark{'s' if len(self.detections) != 1 else ''} from this experiment"
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save Results")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        upload_button = msg_box.addButton("Upload", QMessageBox.ButtonRole.ActionRole)
        
        result = msg_box.exec()

        if msg_box.clickedButton() == upload_button and not self.is_uploading:
            self.is_uploading = True
            self.upload_to_gcs(experiment_dir)

    def upload_to_gcs(self, experiment_dir):
        if self.progress_dialog:
            self.progress_dialog.close()
        
        self.progress_dialog = QProgressDialog("Preparing and uploading files...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.show()

        full_experiment_dir = os.path.abspath(experiment_dir)
        self.upload_thread = UploadThread(self.api_url, full_experiment_dir)
        self.upload_thread.upload_finished.connect(self.upload_finished)
        self.upload_thread.start()

    def upload_finished(self, success, message):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        if success:
            QMessageBox.information(self, "Upload Complete", message)
        else:
            QMessageBox.warning(self, "Upload Failed", message)

        if self.upload_thread:
            self.upload_thread.wait()
            self.upload_thread = None

        self.is_uploading = False

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
            self.go_to_video_selection.emit()
        else:
            event.ignore()
    
class UploadThread(QThread):
    progress_updated = pyqtSignal(int)
    upload_finished = pyqtSignal(bool, str)

    def __init__(self, api_url, experiment_dir):
        super().__init__()
        self.api_url = api_url
        self.experiment_dir = experiment_dir

    def run(self):
        zip_path = None
        try:
            # Create a temporary zip file in the experiment directory
            zip_filename = 'upload.zip'
            zip_path = os.path.join(self.experiment_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for folder in ['bounding_boxes', 'false_positives', 'frames']:
                    folder_path = os.path.join(self.experiment_dir, folder)
                    if os.path.exists(folder_path):
                        for root, dirs, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.experiment_dir)
                                zipf.write(file_path, arcname)

            # Check if the zip file was created successfully
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Failed to create zip file at {zip_path}")

            # Upload the zip file
            with open(zip_path, 'rb') as zip_file:
                files = {'file': (zip_filename, zip_file, 'application/zip')}
                response = requests.post(self.api_url, files=files)
                response.raise_for_status()

            self.upload_finished.emit(True, "Folder uploaded successfully")
        except FileNotFoundError as e:
            self.upload_finished.emit(False, f"File not found: {str(e)}")
        except requests.RequestException as e:
            self.upload_finished.emit(False, f"Upload failed: {str(e)}")
        except Exception as e:
            self.upload_finished.emit(False, f"An unexpected error occurred: {str(e)}")
        finally:
            # Ensure the zip file is deleted even if an exception occurs
            if zip_path and os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception as e:
                    print(f"Failed to delete temporary zip file: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VerificationScreen()
    window.show()
    sys.exit(app.exec())
