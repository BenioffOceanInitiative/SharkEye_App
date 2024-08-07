import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent

class VideoSelectionScreen(QMainWindow):
    start_detection = pyqtSignal(list)
    go_to_verification = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.video_paths = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SharkEye - Video Selection")
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Select Videos button
        self.select_button = self.create_button("Select Videos", self.select_videos, "icons/select_video.png", custom_color=True)
        main_layout.addWidget(self.select_button)

        self.remove_button = self.create_button("Remove Selected", self.remove_selected_video, "icons/remove.png")
        self.remove_button.hide()
        self.remove_button.setEnabled(False)
        main_layout.addWidget(self.remove_button)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.file_list.itemSelectionChanged.connect(self.update_ui_state)
        main_layout.addWidget(self.file_list)

        # Start Detection button (initially hidden)
        self.start_button = self.create_button("Start Detection", self.start_detection_process, "icons/start.png")
        self.start_button.hide()
        main_layout.addWidget(self.start_button)

        # Spacing
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Go to Verification button
        self.verify_button = self.create_button("Go to Verification", self.go_to_verification.emit, "icons/verify.png", custom_color=True)
        main_layout.addWidget(self.verify_button)

    def create_button(self, text, slot, icon_path=None, custom_color=False):
        button = QPushButton(text)
        button.clicked.connect(slot)
        if icon_path:
            button.setIcon(QIcon(icon_path))
        
        button.setStyleSheet("""
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
        """)
        
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
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        
        return button

    def select_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if files:
            new_files = [f for f in files if not self.is_duplicate(f)]
            if len(new_files) < len(files):
                QMessageBox.information(self, "Duplicate Files", 
                                        f"{len(files) - len(new_files)} duplicate file(s) were not added.")
            self.video_paths.extend(new_files)
            self.update_file_list()

    def remove_selected_video(self):
        selected_items = self.file_list.selectedItems()
        if selected_items:
            for item in selected_items:
                self.video_paths.remove(item.data(Qt.ItemDataRole.UserRole))
                self.file_list.takeItem(self.file_list.row(item))
        self.update_ui_state()

    def is_duplicate(self, file_path):
        return file_path in self.video_paths

    def update_file_list(self):
        self.file_list.clear()
        for path in self.video_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
        self.update_ui_state()

    def dropEvent(self, event: QDropEvent):
        files = [
            url.toLocalFile() for url in event.mimeData().urls() 
            if url.isLocalFile() and url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        if files:
            new_files = [f for f in files if not self.is_duplicate(f)]
            if len(new_files) < len(files):
                QMessageBox.information(self, "Duplicate Files", 
                                        f"{len(files) - len(new_files)} duplicate file(s) were not added.")
            self.video_paths.extend(new_files)
            self.update_file_list()

    def update_ui_state(self):
        has_videos = bool(self.video_paths)
        self.remove_button.setVisible(has_videos)
        self.start_button.setVisible(has_videos)
        
        selected_items = self.file_list.selectedItems()
        is_enabled = bool(selected_items)
        self.remove_button.setEnabled(is_enabled)
        
        # Force style update
        self.remove_button.style().unpolish(self.remove_button)
        self.remove_button.style().polish(self.remove_button)
        self.remove_button.update()

    def start_detection_process(self):
        if self.video_paths:
            self.start_detection.emit(self.video_paths)
        else:
            QMessageBox.warning(self, "No Videos", "Please select videos before starting detection.")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSelectionScreen()
    window.show()
    sys.exit(app.exec())