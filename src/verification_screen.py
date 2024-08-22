import sys
import os
from datetime import datetime
import shutil
import zipfile
import requests
import re
import hashlib
import pandas as pd 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from video_player import VideoPlayer
from utility import get_results_dir

class VerificationScreen(QMainWindow):
    go_to_video_selection = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_experiment = None
        self.current_detection_index = 0
        self.detections = []
        self.experiment_map = {}
        self.api_url = "https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload"
        self.progress_dialog = None
        self.upload_thread = None
        self.is_uploading = False
        self.video_paths = []
        self.result_dict = {}
        self.current_video = None
        self.classification_changes = {}  # To store classification changes
        self.last_upload_hash = None
        self.is_data_changed = False
        self.results_df = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("SharkEye - Verification")
        self.setMinimumSize(640, 480)  # Set a minimum size
        self.resize(1024, 768)  # Set an initial size

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Experiment selection
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self.load_experiment)
        main_layout.addWidget(self.experiment_combo)
        
        self.image_label = VideoPlayer()
        self.image_label.request_next_video.connect(self.next_video)
        self.image_label.request_previous_video.connect(self.prev_video)
        self.image_label.indicator_changed.connect(self.update_shark_info)
        main_layout.addWidget(self.image_label)

        # Place holder image for when no experiments
        self.empty_frame = QLabel("No experiments found with detections to verify")
        self.empty_frame.resize(self.image_label.size())
        self.empty_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_frame.hide()
        main_layout.addWidget(self.empty_frame)
        
        self.detection_count_label = QLabel("No detections")
        self.detection_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Classification dropdown
        self.classification_combo = QComboBox()
        self.classification_combo.addItems(["Shark", "Kelp", "Dolphin", "Kayaker", "Surfer", "Paddleboarder", "Bat Ray", "Boat", "Other"])
        self.classification_combo.currentIndexChanged.connect(self.update_classification)
        main_layout.addWidget(self.classification_combo)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        
        # Save Results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_and_upload_results)
        button_layout.addWidget(self.save_button)
        
        # Upload Data button
        self.upload_button = QPushButton("Upload Data")
        self.upload_button.clicked.connect(self.upload_data)
        button_layout.addWidget(self.upload_button)
        
        # Create a widget to hold the button layout
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        
        # Add the button widget to the main layout
        main_layout.addWidget(button_widget)
        
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
        results_dir = get_results_dir()
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

            self.experiment_combo.addItems(valid_experiments)

        if self.experiment_combo.count() > 0:
            self.experiment_combo.setCurrentIndex(0)
            return True
        else:
            self.clear_display()
            QMessageBox.information(self, "No Valid Experiments", "There are no experiments with detections to verify.")
            return False

    def refresh_experiments(self):
        """Clear and reload the experiments list."""
        self.experiment_combo.blockSignals(True)  # Block signals temporarily
        self.experiment_combo.clear()
        self.experiment_map.clear()
        experiments_found = self.load_experiments()
        self.experiment_combo.blockSignals(False)  # Unblock signals
        
        if experiments_found and self.experiment_combo.count() > 0:
            self.experiment_combo.setCurrentIndex(0)
            self.load_experiment(0)  # Load the first experiment
        
        return experiments_found

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
            new_experiment = self.experiment_map.get(formatted_name)
            if new_experiment:
                self.current_experiment = new_experiment
                self.load_results_csv()
                self.load_detections()
                self.show_current_detection()
                return True
            else:
                print(f"Experiment {formatted_name} not found")
                self.clear_display()
        else:
            print(f"Warning: Invalid experiment index {index}")
            self.clear_display()
        return False

    def load_results_csv(self):
        experiment_dir = os.path.join(get_results_dir(), self.current_experiment)
        results_csv_path = os.path.join(experiment_dir, "results.csv")
        if os.path.exists(results_csv_path):
            self.results_df = pd.read_csv(results_csv_path)
            self.results_df.columns = ['id', 'filename', 'frame_path', 'video_path']
        else:
            print(f"Warning: results.csv not found at {results_csv_path}")
            self.results_df = pd.DataFrame(columns=['id', 'filename', 'frame_path', 'video_path'])

    def separate_video_detections(self):
        if self.current_experiment is None:
            print("Error: No experiment is currently selected.")
            return

        if self.results_df is None or self.results_df.empty:
            print("Error: No results data available.")
            return
        
        self.result_dict = self.results_df.groupby('video_path')['frame_path'].apply(list).to_dict()
        self.result_dict = {k: v for k, v in self.result_dict.items()}
        
        if self.result_dict:
            self.current_video = list(self.result_dict.keys())[0]
            self.update_video_navigation_buttons()
            self.change_video()
        else:
            print("No valid video paths found in results.")
            self.clear_display()

    def update_video_navigation_buttons(self):
        total_videos = len(self.result_dict.keys())
        current_index = list(self.result_dict.keys()).index(self.current_video) + 1
        self.image_label.set_video_index(current_index, total_videos)

    def change_video(self):
        if self.current_video:
            try:
                frame_numbers = {}
                for frame_path in self.result_dict[self.current_video]:
                    frame = os.path.basename(frame_path)
                    frame_number = int(frame.split("frame")[1].split(".")[0])
                    frame_numbers[frame_number] = frame_path
                
                if not os.path.exists(self.current_video):
                    print(f"Video file not found: {self.current_video}")
                    return
                
                self.image_label.set_video_dir(self.current_video)
                self.image_label.set_interesting_points(frame_numbers)
                                
                self.empty_frame.hide()
                self.image_label.show()
            except Exception as e:
                print(f"Error changing video: {str(e)}")
        else:
            self.clear_display()
    
    def update_shark_info(self, frame):
        print(f"Updating shark info for frame: {frame}")
        try:
            if frame is None:
                print("Received None frame in update_shark_info")
                return

            if isinstance(frame, int):  # If it's a position in milliseconds
                frame_number = self.image_label.ms_to_frame(frame)
                print(f"Converted ms {frame} to frame number {frame_number}")
                frame_path = self.image_label.frame_paths.get(frame_number)
            else:
                frame_path = frame
            
            print(f"Frame path: {frame_path}")
            
            if frame_path and os.path.exists(frame_path):
                shark_info = self.extract_shark_info(frame_path)
                if shark_info[0] is not None:  # Check if shark_id is not None
                    shark_id, shark_length, shark_confidence, frame_number = shark_info
                    current_position = self.image_label.media_player.position()
                    if current_position >= self.image_label.frame_to_ms(frame_number):
                        self.image_label.set_video_info(self.current_video, shark_id, shark_length, shark_confidence)
                        self.classification_combo.setEnabled(True)
                    else:
                        self.image_label.set_video_info(self.current_video)
                        self.classification_combo.setEnabled(False)
                    self.current_frame_path = frame_path  # Store the current frame path
                    
                    # Set the combo box to the current classification (either original or changed)
                    current_classification = self.classification_changes.get(frame_path, "Shark")
                    self.classification_combo.setCurrentText(current_classification)
                else:
                    print(f"No shark info extracted from frame: {frame_path}")
                    self.image_label.set_video_info(self.current_video)
                    self.classification_combo.setEnabled(False)
            else:
                if frame_path is not None:
                    print(f"Frame path not found or does not exist: {frame_path}")
                else:
                    print("Frame path is None")
                self.image_label.set_video_info(self.current_video)
                self.classification_combo.setEnabled(False)
        except Exception as e:
            print(f"Error updating shark info: {str(e)}")
            self.classification_combo.setEnabled(False)
    
    def extract_shark_info(self, filename):
        print(f"Extracting shark info from filename: {filename}")
        # Extract information from the filename
        # Format: TRIMMED_YYYY-MM-DD_Transect_DJI_XXXX_shark_<id>_conf_<confidence>_len_<length>_frame<number>.jpg
        basename = os.path.basename(filename)
        
        # Use regular expressions to extract the information
        match = re.search(r'shark_(\d+)_conf_([\d.]+)_len_([\d.]+)_frame(\d+)', basename)
        
        if match:
            shark_id = match.group(1)
            shark_confidence = float(match.group(2))
            shark_length = float(match.group(3))
            frame_number = int(match.group(4))
            print(f"Extracted shark info: ID={shark_id}, Length={shark_length}, Confidence={shark_confidence}, Frame={frame_number}")
            return shark_id, shark_length, shark_confidence, frame_number
        else:
            print(f"Couldn't extract shark info from filename: {basename}")
            return None, None, None, None

    def next_video(self):
        video_list = list(self.result_dict.keys())
        current_index = video_list.index(self.current_video)
        if current_index < len(video_list) - 1:
            self.image_label.reset_play_button()  # Reset play button before changing video
            self.current_video = video_list[current_index + 1]
            self.change_video()
            self.update_video_navigation_buttons()
        else:
            self.image_label.next_video_button.setEnabled(False)

    def prev_video(self):
        video_list = list(self.result_dict.keys())
        current_index = video_list.index(self.current_video)
        if current_index > 0:
            self.image_label.reset_play_button()  # Reset play button before changing video
            self.current_video = video_list[current_index - 1]
            self.change_video()
            self.update_video_navigation_buttons()
        else:
            self.image_label.prev_video_button.setEnabled(False)
    
    def load_detections(self):
        if self.current_experiment:
            self.separate_video_detections()
        else:
            print("No current experiment selected.")
            self.clear_display()

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
        if self.current_video:
            self.change_video()
        else:
            self.clear_display()

    def update_classification(self, index):
        if hasattr(self, 'current_frame_path') and self.current_frame_path:
            new_classification = self.classification_combo.currentText()
            if new_classification != "Shark":
                self.classification_changes[self.current_frame_path] = new_classification
            else:
                # If classification is set back to "Shark", remove it from changes
                self.classification_changes.pop(self.current_frame_path, None)
            print(f"Updated classification for {self.current_frame_path} to {new_classification}")
            print(f"Current classification changes: {self.classification_changes}")
            
            # Enable the Save Results button when changes are made
            self.save_button.setEnabled(bool(self.classification_changes))

    def update_filename_classification(self, filename, new_classification):
        parts = filename.split('_')
        # Find the index of 'shark' in the filename
        try:
            shark_index = parts.index('shark')
            # Replace 'shark' with the new classification
            parts[shark_index] = new_classification.lower()
        except ValueError:
            # If 'shark' is not in the filename, add the new classification before 'frame'
            frame_index = next(i for i, part in enumerate(parts) if 'frame' in part)
            parts.insert(frame_index, new_classification.lower())
        return '_'.join(parts)

    def save_and_upload_results(self):
        print("Save Results button clicked")
        print(f"Current classification changes: {self.classification_changes}")
        
        if not self.classification_changes:
            QMessageBox.information(self, "No Changes", "No classification changes to save.")
            return

        experiment_dir = os.path.join(get_results_dir(), self.current_experiment)
        bounding_boxes_dir = os.path.join(experiment_dir, "bounding_boxes")
        false_positives_dir = os.path.join(experiment_dir, "false_positives")
        os.makedirs(false_positives_dir, exist_ok=True)

        moved_count = 0
        for frame_path, new_classification in self.classification_changes.items():
            if not os.path.exists(frame_path):
                print(f"Warning: File not found: {frame_path}")
                continue

            filename = os.path.basename(frame_path)
            new_filename = self.update_filename_classification(filename, new_classification)
            
            if new_classification != "Shark":
                new_path = os.path.join(false_positives_dir, new_filename)
                shutil.move(frame_path, new_path)
                moved_count += 1
                
                # Remove from results_df
                self.results_df = self.results_df[self.results_df['frame_path'] != frame_path]
                
                # Update the video player's frame paths
                frame_number = int(filename.split("frame")[1].split(".")[0])
                self.image_label.remove_indicator(frame_number)
            else:
                new_path = os.path.join(bounding_boxes_dir, new_filename)
                shutil.move(frame_path, new_path)
                
                # Update results_df
                self.results_df.loc[self.results_df['frame_path'] == frame_path, 'frame_path'] = new_path
                self.results_df.loc[self.results_df['frame_path'] == frame_path, 'filename'] = new_filename

            print(f"Moved {filename} to {new_path}")

        # Save updated results.csv
        results_csv_path = os.path.join(experiment_dir, "results.csv")
        self.results_df.to_csv(results_csv_path, index=False)
        print(f"Updated results.csv saved to {results_csv_path}")

        shark_count = len(self.results_df)
        non_shark_count = len([f for f in os.listdir(false_positives_dir) if f.endswith('.jpg')])

        message = f"Moved {moved_count} detection{'s' if moved_count != 1 else ''} to false positives.\n"
        message += f"{shark_count} Shark{'s' if shark_count != 1 else ''} and "
        message += f"{non_shark_count} non-shark detection{'s' if non_shark_count != 1 else ''} in this experiment"

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save Results")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        upload_button = msg_box.addButton("Upload", QMessageBox.ButtonRole.ActionRole)
        
        result = msg_box.exec()

        if msg_box.clickedButton() == upload_button and not self.is_uploading:
            self.is_uploading = True
            self.upload_to_gcs(experiment_dir)

        # Clear the classification changes after saving
        self.classification_changes.clear()
        print("Classification changes cleared after saving")
        
        # Refresh the current video display
        self.refresh_current_video()

    def refresh_current_video(self):
        if self.current_video:
            self.load_detections()
            self.change_video()

    def upload_data(self):
        print("Upload Data button clicked")
        if self.is_uploading:
            QMessageBox.warning(self, "Upload in Progress", "An upload is already in progress.")
            return

        experiment_dir = os.path.join(get_results_dir(), self.current_experiment)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Upload Data")
        msg_box.setText("Do you want to upload the current data?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        if msg_box.exec() == QMessageBox.StandardButton.Yes:
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
        
    def get_data_hash(self):
        experiment_dir = self.get_results_dir(self.current_experiment)
        hash_md5 = hashlib.md5()
        
        for root, _, files in os.walk(experiment_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()

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
