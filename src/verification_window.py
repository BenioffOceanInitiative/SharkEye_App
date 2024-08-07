from PyQt6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QSpacerItem, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QKeyEvent
import os
import shutil
from datetime import datetime

class VerificationWindow(QWidget):
    def __init__(self, results_dir):
        super().__init__()
        self.results_dir = results_dir
        self.current_experiment_index = 0
        self.current_frame = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Detection Verification")
    
        # Set minimum size based on frame_display
        self.minimum_width = 720  # 640 (frame width) + 40 (left button) + 40 (right button)
        self.minimum_height = 600  # Approximate minimum height to fit all elements
    
        self.setMinimumSize(self.minimum_width, self.minimum_height)
        self.resize(self.minimum_width, self.minimum_height)
        
        self.experiments = [d for d in os.listdir(self.results_dir) 
                            if os.path.isdir(os.path.join(self.results_dir, d)) and 
                            os.path.exists(os.path.join(self.results_dir, d, "bounding_boxes"))]
        self.experiments.sort(reverse=True)
        
        self.formatted_experiments = [self.format_experiment_name(exp) for exp in self.experiments]

        if self.experiments:
            self.current_experiment_index = 0
            self.load_experiment_frames(self.current_experiment_index)
        else:
            self.current_experiment_index = -1
            self.frames = []
            self.classifications = []

        # Main Layout 
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Experiment Selection
        self.experiment_label = QComboBox()
        self.experiment_label.addItems(self.formatted_experiments)
        self.experiment_label.currentIndexChanged.connect(self.select_experiment)
        self.experiment_label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        main_layout.addWidget(self.experiment_label)
        self.experiment_label.view().pressed.connect(self.refocus_frame_display)
    
        # File Path
        self.file_path = QLabel()
        self.file_path.setStyleSheet("color: black; background-color: white; border-radius: 4px; padding: 5px;")
        self.file_path.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        main_layout.addWidget(self.file_path)

        # Display Frame
        frame_container = QWidget()
        frame_layout = QHBoxLayout(frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self.previous_frame = QPushButton("<")
        self.previous_frame.setFixedWidth(40)
        self.previous_frame.clicked.connect(self.display_previous_frame)

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setMinimumSize(640, 480)
        self.frame_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_display.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  
        self.frame_display.setFocus()

        self.next_frame = QPushButton(">")
        self.next_frame.setFixedWidth(40)
        self.next_frame.clicked.connect(self.display_next_frame)
                                                
        frame_layout.addWidget(self.previous_frame)
        frame_layout.addWidget(self.frame_display)
        frame_layout.addWidget(self.next_frame)
    
        main_layout.addWidget(frame_container)
    
        # Frame Counter
        self.frame_counter_label = QLabel()
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.frame_counter_label)
    
        # Control Group
        control_group = QWidget()
        control_layout = QVBoxLayout(control_group)
    
        # Classification Dropdown
        self.classification_dropdown = QComboBox()
        self.classification_dropdown.addItems(["Shark", "Kelp", "Surfer", "Kayaker", "Boat", "Bat Ray", "Dolphin", "Other"])
        self.classification_dropdown.currentIndexChanged.connect(self.update_classification)
        control_layout.addWidget(self.classification_dropdown)

        # Buttons
        button_layout = QHBoxLayout()
    
        self.finish_verification_button = QPushButton("Finish Verification")
        self.finish_verification_button.setStyleSheet("background-color: green;")
        self.finish_verification_button.clicked.connect(self.finish_verifications)
        button_layout.addWidget(self.finish_verification_button)

        self.return_to_main_button = QPushButton("Return to Main")
        self.return_to_main_button.setStyleSheet("background-color: orange; color: white; border-radius: 4px; width: 100px; height: 30px;")
        self.return_to_main_button.clicked.connect(self.return_to_main_window)
        button_layout.addWidget(self.return_to_main_button)
    
        control_layout.addLayout(button_layout)
    
        main_layout.addWidget(control_group)

        # Set size policies to expand horizontally
        for widget in [self.experiment_label, self.file_path, self.frame_counter_label, control_group]:
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Add spacer
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    
        self.load_experiment_frames(0)
        self.frame_display.setFocus()
    
        if self.frames:
            self.select_experiment(self.current_experiment_index)
        else:
            self.hide_ui_elements()
    
    def format_experiment_name(self, experiment_name):
        try:
            # Parse the original datetime string
            dt = datetime.strptime(experiment_name, "%Y-%m-%d %H-%M-%S")
            # Format it in a more readable way
            return dt.strftime("%b %d, %Y at %I:%M:%S %p")
        except ValueError:
            # If parsing fails, return the original string
            return experiment_name

    def load_experiment_frames(self, index):
        if self.experiments:
            self.last_run = self.experiments[index]
            bounding_boxes_dir = os.path.join(self.results_dir, self.last_run, "bounding_boxes")
            
            if os.path.exists(bounding_boxes_dir):
                self.frames = [os.path.join(bounding_boxes_dir, f) for f in os.listdir(bounding_boxes_dir) if f.endswith('.jpg')]
                self.frames.sort()
            else:
                self.frames = []
            
            self.classifications = ["Shark" for _ in self.frames]
        else:
            self.frames = []
            self.classifications = []
            
    def refocus_frame_display(self):
        # Use a short timer to allow the selection to complete before shifting focus
        QTimer.singleShot(100, self.frame_display.setFocus)

    
    def update_frame_counter(self):
        current = self.current_frame + 1
        total = len(self.frames)
        self.frame_counter_label.setText(f"Detection {current} of {total}")
    
    def finish_verifications(self):
        """Moves images from 'frames' and 'bounding_boxes' folders marked as not being sharks"""
        false_positive_path = os.path.join(self.results_dir, self.last_run, "false_positives")
        if not os.path.isdir(false_positive_path):
            os.mkdir(false_positive_path)
            
        for index, classification in enumerate(self.classifications):
            if classification != "Shark":
                bounding_box_path = self.frames[index]
                frame_path = bounding_box_path.replace("bounding_boxes", "frames")
                
                # Create new filename with classification label
                base_filename = os.path.basename(bounding_box_path)
                name, ext = os.path.splitext(base_filename)
                new_filename = f"{name}_{classification}{ext}"
                
                # Move bounding box image if it exists
                if os.path.exists(bounding_box_path):
                    new_path = os.path.join(false_positive_path, new_filename)
                    shutil.move(bounding_box_path, new_path)
                    print(f"Moved {bounding_box_path} to {new_path}")
                
                # Remove frame image if it exists
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    print(f"Removed {frame_path}")
                    
        final_count = self.classifications.count("Shark")
        QMessageBox.information(self, "Verification Complete", f"Final Count of Sharks: {final_count}")
        
        # Check if the experiment folder is empty and delete it if so
        experiment_path = os.path.join(self.results_dir, self.last_run)
        bounding_boxes_dir = os.path.join(experiment_path, "bounding_boxes")
        frames_dir = os.path.join(experiment_path, "frames")
        if not os.listdir(bounding_boxes_dir) and not os.listdir(frames_dir):
            shutil.rmtree(bounding_boxes_dir)
            shutil.rmtree(frames_dir)
            QMessageBox.information(self, "No more detections", "The detections and frames folders are empty and have been deleted.")

            # Remove the deleted experiment from the list and combo box
            self.experiments.remove(self.last_run)
            self.experiment_label.removeItem(self.current_experiment_index)
            
        # If there are no more experiments, hide UI elements
        if not self.experiments:
            self.hide_ui_elements()
            self.frame_counter_label.setText("No experiments left")
            self.frame_counter_label.show()
            return
        
        # Select the next available experiment
        new_index = min(self.current_experiment_index, len(self.experiments) - 1)
        self.experiment_label.setCurrentIndex(new_index)
        self.current_experiment_index = new_index
        
        self.select_experiment(self.current_experiment_index)

    def update_classification(self, index):
        if self.classifications:
            self.classifications[self.current_frame] = self.classification_dropdown.currentText()

    def value_change(self):
        if not self.frames:
            return
    
        index = self.current_frame
        if 0 <= index < len(self.frames):
            frame = QPixmap(self.frames[index])
            scaled_frame = frame.scaled(self.frame_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.frame_display.setPixmap(scaled_frame)
            self.file_path.setText(self.frames[index])
            self.classification_dropdown.setCurrentText(self.classifications[index])
            self.update_frame_counter()
            
            if index == 0:
                self.previous_frame.setStyleSheet("border-width: 0px; border-style: solid; color: transparent")
                self.previous_frame.setDisabled(True)
            else:
                self.previous_frame.setStyleSheet("border-width: 0px; border-style: solid")
                self.previous_frame.setEnabled(True)
                
            if index == len(self.frames) - 1:
                self.next_frame.setStyleSheet("border-width: 0px; border-style: solid; color: transparent")
                self.next_frame.setDisabled(True)
            else:
                self.next_frame.setStyleSheet("border-width: 0px; border-style: solid")
                self.next_frame.setEnabled(True)
        else:
            print(f"Invalid index: {index}, total frames: {len(self.frames)}")  # For debugging
            self.current_frame = 0  # Reset to a valid index
            self.value_change()  # Recursively call with the corrected index

    def hide_ui_elements(self):
        self.frame_counter_label.hide()
        self.frame_display.setText("Select an experiment to start verifying detections")
        self.file_path.hide()
        self.classification_dropdown.setDisabled(True)
        self.finish_verification_button.setDisabled(True)
        self.classification_dropdown.setStyleSheet("background-color: white; color: grey;")
        self.finish_verification_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.next_frame.hide()
        self.previous_frame.hide()

    def show_ui_elements(self):
        self.frame_display.show()
        self.frame_counter_label.show()
        self.file_path.show()
        self.classification_dropdown.setEnabled(True)
        self.finish_verification_button.setEnabled(True)
        self.classification_dropdown.setStyleSheet("")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.next_frame.show()
        self.previous_frame.show()

    def select_experiment(self, index):
        if not self.experiments:
            self.hide_ui_elements()
            self.frame_counter_label.setText("No valid experiments found")
            self.frame_counter_label.show()
            return
        
        self.current_experiment_index = index  # Update the current index
        self.load_experiment_frames(self.current_experiment_index)

        if len(self.frames) > 0:             
            self.classifications = ["Shark" for _ in range(len(self.frames))]
            self.current_frame = 0
            frame = QPixmap(self.frames[self.current_frame])
            scaled_frame = frame.scaled(self.frame_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.frame_display.setPixmap(scaled_frame)
            self.file_path.setText(self.frames[0])
            self.show_ui_elements()
            self.update_frame_counter()
            self.value_change()
            
            # Set focus back to frame_display after a short delay
            QTimer.singleShot(100, self.frame_display.setFocus)
        else:
            self.hide_ui_elements()
            self.frame_counter_label.setText("No detections found")
            self.frame_counter_label.show()
        
        if self.experiment_label.currentIndex() != index:
            self.experiment_label.setCurrentIndex(index)
            
    def return_to_main_window(self):
        self.parent().setCurrentWidget(self.parent().parent().central_widget)
        self.parent().removeWidget(self.parent().parent().verification_window)

    def keyPressEvent(self, event: QKeyEvent):
        if len(self.frames) > 1:
            if event.key() == Qt.Key.Key_J:
                self.display_previous_frame()
            elif event.key() == Qt.Key.Key_L:
                self.display_next_frame()
            elif event.key() == Qt.Key.Key_K:
                self.toggle_classification()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def display_next_frame(self):
        self.current_frame = min(len(self.frames) - 1, self.current_frame + 1)
        self.value_change()

    def display_previous_frame(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.value_change()
    
    def toggle_classification(self):
        current_index = self.classification_dropdown.currentIndex()
        next_index = (current_index + 1) % self.classification_dropdown.count()
        self.classification_dropdown.setCurrentIndex(next_index)
        self.update_classification(next_index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'frame_display') and self.frame_display.pixmap():
            self.value_change()  # This will rescale the image