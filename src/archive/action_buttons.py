from PyQt6.QtWidgets import QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal

class ActionButtons(QWidget):
    start_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        self.start_button = self.create_button("Start Detection", self.start_clicked.emit, enabled=False)
        self.cancel_button = self.create_button("Cancel", self.cancel_clicked.emit, enabled=False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.cancel_button)

    def create_button(self, text, slot, enabled=True):
        button = QPushButton(text)
        button.clicked.connect(slot)
        button.setEnabled(enabled)
        return button

    def set_start_enabled(self, enabled):
        self.start_button.setEnabled(enabled)

    def set_cancel_enabled(self, enabled):
        self.cancel_button.setEnabled(enabled)