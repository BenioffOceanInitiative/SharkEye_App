from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QKeyEvent
import os

class VerificationWindow(QWidget):
    def __init__(self, results_dir):
        super().__init__()
        self.results_dir = results_dir
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Detection Verification")
        self.disply_width = 1024
        self.display_height = 768
        
        self.initial_width = 1024
        self.initial_height = 768
        
        self.resize(self.initial_width, self.initial_height)
    
        self.experiments = os.listdir(self.results_dir)
        self.experiments.sort(reverse=True)
    
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
        
        self.load_experiment_frames(0)
        
        # Slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)
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
        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setMinimumSize(640, 480)
        self.frame_display.resize(self.disply_width, self.display_height)
        
        self.file_path = QLabel()
        self.file_path.setStyleSheet("color: black; background-color: white; border-radius: 4px; padding: 5px;")
            
        self.verified_sharks = [True for x in range(len(self.frames))]

        self.frame_counter_label = QLabel()
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        # Main Layout 
        tracker_layout = QVBoxLayout(self)
        tracker_layout.addWidget(self.experiment_label)
        tracker_layout.addWidget(self.frame_display)        
        tracker_layout.addWidget(self.file_path)
        tracker_layout.addWidget(self.frame_slider)
        tracker_layout.addWidget(self.frame_counter_label)
        tracker_layout.addLayout(marking_layout)
        tracker_layout.addWidget(self.finish_verification_button)
        
        # Add spacer
        tracker_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.frame_slider.valueChanged.connect(self.value_change)
        self.select_experiment(0)
        
    def load_experiment_frames(self, index):
        if self.experiments:
            self.last_run = self.experiments[index]
            bounding_boxes_dir = os.path.join(self.results_dir, self.last_run, "bounding_boxes")
            self.frames = [os.path.join(bounding_boxes_dir, f) for f in os.listdir(bounding_boxes_dir) if f.endswith('.jpg')]
            self.frames.sort()
    
    def update_frame_counter(self):
        current = self.frame_slider.value() + 1
        total = len(self.frames)
        self.frame_counter_label.setText(f"Detection {current} of {total}")
    
    def finish_verifications(self):
        """Moves images from 'frames' and 'bounding_boxes' folders marked as having no sharks"""
        false_positive_path = os.path.join(self.results_dir, self.last_run, "false_positives")
        if not os.path.isdir(false_positive_path):
            os.mkdir(false_positive_path)
        for index, x in enumerate(self.verified_sharks):
            if x == False:
                print(os.path.splitext(self.frames[index])[0] + "TEST.jpg")
                os.rename(self.frames[index], os.path.join(false_positive_path, os.path.basename(self.frames[index])))
                
                frames_path = self.frames[index].replace("bounding_boxes", "frames")
                os.remove(frames_path)
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

    def update_slider_visibility(self):
        if len(self.frames) <= 1:
            self.frame_slider.hide()
        else:
            self.frame_slider.show()

    def value_change(self):
        """Handles display of frames and colors of verification buttons"""
        index = self.frame_slider.value()
        frame = QPixmap(self.frames[index]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_frame = index

        self.frame_display.setPixmap(frame)
        self.file_path.setText(self.frames[index])
        self.update_button_styles()
        self.update_frame_counter()

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
        
        self.mark_shark_button.setDisabled(True)
        self.unmark_shark_button.setDisabled(True)
        self.finish_verification_button.setDisabled(True)

        self.mark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: white; color: grey; border-radius: 4px; width: 100px; height: 30px;")

    def show_ui_elements(self):
        """Makes display frame, verification buttons and slider appear"""
        self.frame_display.show()
        self.frame_counter_label.show()
        
        self.mark_shark_button.setEnabled(True)
        self.unmark_shark_button.setEnabled(True)
        self.finish_verification_button.setEnabled(True)
        
        self.mark_shark_button.setStyleSheet("background-color: blue; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.unmark_shark_button.setStyleSheet("background-color: white; color: black; border-radius: 4px; min-width: 100px; min-height: 30px;")
        self.finish_verification_button.setStyleSheet("background-color: green; color: white; border-radius: 4px; min-width: 100px; min-height: 30px;")

        self.update_slider_visibility()

    def select_experiment(self, index):
        """Selects experiment to run verification on""" 
        self.load_experiment_frames(index)

        if len(self.frames) > 0: 
            self.current_frame = 0
            self.frame_slider.setValue(0)
            frame = QPixmap(self.frames[self.current_frame]).scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            self.frame_display.setPixmap(frame)

            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, len(self.frames) - 1))

            self.file_path.setText(self.frames[index])

            self.verified_sharks = [True for x in range(len(self.frames))]

            self.show_ui_elements()
            self.update_frame_counter()
            self.update_slider_visibility()  # Add this line
        else:
            self.hide_ui_elements()
            self.frame_counter_label.setText("No detections found")
            self.frame_counter_label.show()
            
    def keyPressEvent(self, event: QKeyEvent):
        if len(self.frames) > 1:
            if event.key() == Qt.Key.Key_Left:
                self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))
            elif event.key() == Qt.Key.Key_Right:
                self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))
        
        if event.key() == Qt.Key.Key_S:
            self.mark_as_shark()
        elif event.key() == Qt.Key.Key_N:
            self.unmark_shark()
        else:
            super().keyPressEvent(event)