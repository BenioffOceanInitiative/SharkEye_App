from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QSpacerItem, QSizePolicy, QMessageBox
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QKeyEvent, QIcon
import os
import shutil

class VerificationWindow(QWidget):
    def __init__(self, results_dir):
        super().__init__()
        self.results_dir = results_dir
        self.current_experiment_index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Detection Verification")
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.resize(self.initial_width, self.initial_height)
            
        self.experiments = [d for d in os.listdir(self.results_dir) 
                                if os.path.isdir(os.path.join(self.results_dir, d)) and 
                                os.path.exists(os.path.join(self.results_dir, d, "bounding_boxes"))]
        self.experiments.sort(reverse=True)

        if self.experiments:
            self.current_experiment_index = 0
            self.load_experiment_frames(self.current_experiment_index)
        else:
            self.current_experiment_index = -1
            self.frames = []
            self.verified_sharks = []
    
        # Shark Buttons 
        marking_layout = QHBoxLayout()

        self.mark_shark_button = QPushButton("Shark")
        self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.mark_shark_button.clicked.connect(self.mark_as_shark)

        self.unmark_shark_button = QPushButton("No Shark")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.unmark_shark_button.clicked.connect(self.unmark_shark) 

        marking_layout.addWidget(self.mark_shark_button)
        marking_layout.addWidget(self.unmark_shark_button)      

        self.finish_verification_button = QPushButton("Finish Verification")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.finish_verification_button.clicked.connect(self.finish_verifications)
        
        self.return_to_main_button = QPushButton("Return to Main Window")
        self.return_to_main_button.setStyleSheet("background-color: orange; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.return_to_main_button.clicked.connect(self.return_to_main_window)
        self.load_experiment_frames(self.current_experiment_index)

        # Slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setFixedHeight(30)  # Set a fixed height
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
    """)

        # Experiment Selection
        self.experiment_label = QComboBox()
        self.experiment_label.addItems(self.experiments)
        self.experiment_label.currentIndexChanged.connect(self.select_experiment)

        # Display Frame
        frame_layout = QHBoxLayout()
        self.next_frame = QPushButton(">")
        self.next_frame.setMaximumWidth(20)
        self.next_frame.setMaximumHeight(self.display_height)
        self.next_frame.setStyleSheet("border-width: 0px; border-style: solid")                                                                                                                                                                                                                                 
        self.next_frame.clicked.connect(self.display_next_frame)

        self.previous_frame = QPushButton("<")
        self.previous_frame.setMaximumWidth(20)
        self.previous_frame.setMaximumHeight(self.display_height)
        self.previous_frame.setStyleSheet("border-width: 0px; border-style: solid; color: transparent")
        self.previous_frame.clicked.connect(self.display_previous_frame)

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setMinimumSize(640, 480)
        self.frame_display.resize(self.disply_width, self.display_height)
                                                    
        frame_layout.addWidget(self.previous_frame)
        frame_layout.addWidget(self.frame_display)
        frame_layout.addWidget(self.next_frame)
        
        self.file_path = QLabel()
        self.file_path.setStyleSheet("color: black; background-color: white; border-radius: 4px; padding: 5px;")
        self.file_path.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.frame_counter_label = QLabel()
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        # Main Layout 
        tracker_layout = QVBoxLayout(self)
        tracker_layout.addWidget(self.experiment_label)
        tracker_layout.addLayout(frame_layout)        
        tracker_layout.addWidget(self.file_path)
        tracker_layout.addWidget(self.frame_slider)
        tracker_layout.addWidget(self.frame_counter_label)
        tracker_layout.addLayout(marking_layout)
        tracker_layout.addWidget(self.finish_verification_button)
        tracker_layout.addWidget(self.return_to_main_button)

        # Add spacer
        tracker_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.frame_slider.valueChanged.connect(self.value_change)
        
        if self.frames:
            self.select_experiment(self.current_experiment_index)
        else:
            self.hide_ui_elements()
        
    def load_experiment_frames(self, index):
        if self.experiments:
            self.last_run = self.experiments[index]
            bounding_boxes_dir = os.path.join(self.results_dir, self.last_run, "bounding_boxes")
            
            if os.path.exists(bounding_boxes_dir):
                self.frames = [os.path.join(bounding_boxes_dir, f) for f in os.listdir(bounding_boxes_dir) if f.endswith('.jpg')]
                self.frames.sort()
            else:
                self.frames = []
            
            self.verified_sharks = [True for _ in self.frames]
        else:
            self.frames = []
            self.verified_sharks = []
    
    def update_frame_counter(self):
        current = self.frame_slider.value() + 1
        total = len(self.frames)
        self.frame_counter_label.setText(f"Detection {current} of {total}")
    
    def finish_verifications(self):
        """Moves images from 'frames' and 'bounding_boxes' folders marked as having no sharks"""
        false_positive_path = os.path.join(self.results_dir, self.last_run, "false_positives")
        if not os.path.isdir(false_positive_path):
            os.mkdir(false_positive_path)
            
        for index, is_shark in enumerate(self.verified_sharks):
            if not is_shark:
                bounding_box_path = self.frames[index]
                frame_path = bounding_box_path.replace("bounding_boxes", "frames")
                
                # Move bounding box image if it exists
                if os.path.exists(bounding_box_path):
                    shutil.move(bounding_box_path, os.path.join(false_positive_path, os.path.basename(bounding_box_path)))
                
                # Remove frame image if it exists
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
        final_count = sum(self.verified_sharks)
        QMessageBox.information(self, "Verification Complete", f"Final Count of Sharks: {final_count}")
        
        # Check if the experiment folder is empty and delete it if so
        experiment_path = os.path.join(self.results_dir, self.last_run)
        bounding_boxes_dir = os.path.join(experiment_path, "bounding_boxes")
        frames_dir = os.path.join(experiment_path, "frames")
        if not os.listdir(bounding_boxes_dir) and not os.listdir(frames_dir):
            shutil.rmtree(bounding_boxes_dir)
            QMessageBox.information(self, "No more detections.", f"The detections folder is empty and is now deleted.")

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

    def update_slider_visibility(self):
        if len(self.frames) <= 1:
            self.frame_slider.hide()
        else:
            self.frame_slider.show()

    def value_change(self):
        """Handles display of frames and colors of verification buttons"""
        if not self.frames:  # Add this check
            return
    
        index = self.frame_slider.value()
        if 0 <= index < len(self.frames):  # Add this check
            frame = QPixmap(self.frames[index]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_frame = index

            self.frame_display.setPixmap(frame)
            self.file_path.setText(self.frames[index])
            self.update_button_styles()
            self.update_frame_counter()

            if self.frame_slider.value() == 0:
                self.previous_frame.setStyleSheet("border-width: 0px; border-style: solid; color: transparent")
                self.previous_frame.setDisabled(True)
            elif self.frame_slider.value() == self.frame_slider.maximum():
                self.next_frame.setStyleSheet("border-width: 0px; border-style: solid; color: transparent")
                self.next_frame.setDisabled(True)
            else:
                self.next_frame.setStyleSheet("border-width: 0px; border-style: solid")
                self.previous_frame.setStyleSheet("border-width: 0px; border-style: solid")
                self.next_frame.setEnabled(True)
                self.previous_frame.setEnabled(True)
        else:
            print(f"Invalid index: {index}, total frames: {len(self.frames)}")  # For debugging

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
        self.frame_counter_label.hide()

        self.frame_display.setText("Select an experiment to start verifying detections")
        self.file_path.hide()
        
        self.mark_shark_button.setDisabled(True)
        self.unmark_shark_button.setDisabled(True)
        self.finish_verification_button.setDisabled(True)

        self.mark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.next_frame.hide()
        self.previous_frame.hide()

    def show_ui_elements(self):
        """Makes display frame, verification buttons and slider appear"""
        self.frame_display.show()
        self.frame_counter_label.show()
        self.file_path.show()

        self.mark_shark_button.setEnabled(True)
        self.unmark_shark_button.setEnabled(True)
        self.finish_verification_button.setEnabled(True)
        
        self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.next_frame.show()
        self.previous_frame.show()

        self.update_slider_visibility()

    def select_experiment(self, index):
        """Selects experiment to run verification on""" 
        if not self.experiments:
            self.hide_ui_elements()
            self.frame_counter_label.setText("No valid experiments found")
            self.frame_counter_label.show()
            return
        
        self.current_experiment_index = index  # Update the current index
        self.load_experiment_frames(self.current_experiment_index)

        if len(self.frames) > 0:             
            self.verified_sharks = [True for _ in range(len(self.frames))]
            self.current_frame = 0
            self.frame_slider.setMaximum(len(self.frames)- 1)
            self.frame_slider.setValue(0)
            frame = QPixmap(self.frames[self.current_frame]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            self.frame_display.setPixmap(frame)
            self.file_path.setText(self.frames[0])

            self.show_ui_elements()
            self.update_frame_counter()
            self.update_slider_visibility()
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
            if event.key() == Qt.Key.Key_Left:
                self.display_previous_frame()
            elif event.key() == Qt.Key.Key_Right:
                self.display_next_frame()
        
        if event.key() == Qt.Key.Key_S:
            self.mark_as_shark()
        elif event.key() == Qt.Key.Key_N:
            self.unmark_shark()
        else:
            super().keyPressEvent(event)
    
    def display_next_frame(self):
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def display_previous_frame(self):
        self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))
