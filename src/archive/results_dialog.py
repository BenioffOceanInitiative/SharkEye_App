from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from PyQt6.QtCore import pyqtSignal

class ResultsDialog(QDialog):
    run_additional_inference = pyqtSignal()
    verify_detections = pyqtSignal()

    def __init__(self, total_detections, total_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results")
        self.setup_ui(total_detections, total_time)

    def setup_ui(self, total_detections, total_time):
        layout = QVBoxLayout(self)

        formatted_time = self.format_time(total_time)
        results_label = QLabel(f"Total Detections: {total_detections}\nTotal Time: {formatted_time}")
        layout.addWidget(results_label)

        self.button_box = QDialogButtonBox()
        run_additional = self.button_box.addButton("Run Additional Inference", QDialogButtonBox.ButtonRole.ActionRole)
        verify_detections = self.button_box.addButton("Verify Detections", QDialogButtonBox.ButtonRole.ActionRole)

        run_additional.clicked.connect(self.on_run_additional)
        verify_detections.clicked.connect(self.on_verify_detections)

        layout.addWidget(self.button_box)

    def on_run_additional(self):
        self.run_additional_inference.emit()
        self.accept()

    def on_verify_detections(self):
        self.verify_detections.emit()
        self.accept()

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 120:
            return f"1 minute {seconds % 60:.0f} seconds"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes} minutes {remaining_seconds} seconds"