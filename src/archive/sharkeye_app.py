import os
import sys
from PyQt6.QtWidgets import QApplication, QStackedWidget, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtGui import QIcon

from video_selection_screen import VideoSelectionScreen
from detections_screen import DetectionsScreen
from verification_screen import VerificationScreen
from utility import resource_path

class SharkEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SharkEye")
        self.setGeometry(100, 100, 1000, 700)
        if sys.platform.startswith('win'):
            app_icon = resource_path('assets/logo/SharkEye.ico')
        elif sys.platform.startswith('darwin'):
            app_icon = resource_path('assets/logo/SharkEye.icns')
        else:
            app_icon = resource_path('assets/logo/SharkEye.iconset/icon_32x32.png')
            
        app.setWindowIcon(QIcon(app_icon))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create and set up the stacked widget
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Create screens
        self.video_selection_screen = VideoSelectionScreen()
        self.stacked_widget.addWidget(self.video_selection_screen)
        
        self.detection_screen = None  # Will be created when needed
        self.verification_screen = None  # Will be created when needed

        # Add video selection screen to stacked widget
        self.stacked_widget.addWidget(self.video_selection_screen)

        # Connect signals
        self.video_selection_screen.start_detection.connect(self.show_detection_screen)
        self.video_selection_screen.go_to_verification.connect(lambda: self.show_verification_screen())

    def show_detection_screen(self, video_paths):
        if self.detection_screen:
            self.stacked_widget.removeWidget(self.detection_screen)
            self.detection_screen.deleteLater()

        self.detection_screen = DetectionsScreen(video_paths)
        self.stacked_widget.addWidget(self.detection_screen)
        self.stacked_widget.setCurrentWidget(self.detection_screen)

        # Connect signals
        self.detection_screen.go_to_verification.connect(self.show_verification_screen)
        self.detection_screen.go_to_video_selection.connect(self.show_video_selection_screen)

    def show_video_selection_screen(self):
        self.stacked_widget.setCurrentWidget(self.video_selection_screen)
        
        # Clean up detection screen if it exists
        if self.detection_screen:
            self.stacked_widget.removeWidget(self.detection_screen)
            self.detection_screen.deleteLater()
            self.detection_screen = None

    def show_verification_screen(self, results_dir=None):
        if not self.verification_screen:
            self.verification_screen = VerificationScreen()
            self.stacked_widget.addWidget(self.verification_screen)
            
            # Connect signals
            self.verification_screen.go_to_video_selection.connect(self.show_video_selection_screen)

        # Refresh experiments list
        if self.verification_screen.refresh_experiments():
            if results_dir:
                # Load the specific experiment if results_dir is provided
                self.verification_screen.load_results(results_dir)
            
            if self.verification_screen.current_experiment:
                self.verification_screen.separate_video_detections()
                self.stacked_widget.setCurrentWidget(self.verification_screen)
            else:
                print("No experiment selected after refreshing.")
                QMessageBox.information(self, "No Experiment", "No experiment is currently selected.")
                self.show_video_selection_screen()
        else:
            # No experiments found, show a message and return to video selection
            QMessageBox.information(self, "No Experiments", "No experiments found with detections to verify.")
            self.show_video_selection_screen()
    
    def closeEvent(self, event):
        # Perform any cleanup or save any settings before closing
        # For example, you might want to save the last used output directory
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
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
        QLabel {
            font-size: 16px;
        }
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            width: 10px;
        }
    """)

    window = SharkEyeApp()
    window.show()
    sys.exit(app.exec())
