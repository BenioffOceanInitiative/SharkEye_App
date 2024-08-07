from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget, QFileDialog, QListWidgetItem
from PyQt6.QtCore import pyqtSignal, Qt
import os

class VideoSelectionArea(QWidget):
    videos_selected = pyqtSignal(list)
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_paths = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.select_button = self.create_button("Select Videos", self.select_videos)
        layout.addWidget(self.select_button)

        button_layout = QHBoxLayout()
        self.remove_button = self.create_button("Remove Selected", self.remove_selected_video, enabled=False)
        self.clear_button = self.create_button("Clear All", self.clear_selection, enabled=False)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self.toggle_remove_button)
        self.file_list.setMaximumHeight(100)
        layout.addWidget(self.file_list)

    def create_button(self, text, slot, enabled=True):
        button = QPushButton(text)
        button.clicked.connect(slot)
        button.setEnabled(enabled)
        return button

    def select_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)")
        if files:
            self.video_paths.extend(files)
            self.update_file_list()
            self.videos_selected.emit(self.video_paths)

    def remove_selected_video(self):
        selected_items = self.file_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            path = item.data(Qt.ItemDataRole.UserRole)
            self.video_paths.remove(path)
            self.file_list.takeItem(self.file_list.row(item))
        self.remove_button.setEnabled(False)
        self.update_button_states()
        self.videos_selected.emit(self.video_paths)

    def clear_selection(self):
        self.video_paths.clear()
        self.update_file_list()
        self.selection_cleared.emit()

    def toggle_remove_button(self):
        self.remove_button.setEnabled(bool(self.file_list.selectedItems()))

    def update_file_list(self):
        self.file_list.clear()
        for path in self.video_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)
        self.update_button_states()

    def update_button_states(self):
        has_videos = bool(self.video_paths)
        self.clear_button.setEnabled(has_videos)
        self.remove_button.setEnabled(False)