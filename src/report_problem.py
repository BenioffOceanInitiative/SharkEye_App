"""Report-a-problem dialog: collect feedback and POST it as JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from log_config import get_logger
from utility import read_local_doc_version, resource_path

logger = get_logger("sharkeye.feedback")

FEEDBACK_ENDPOINT = (
    "https://us-central1-sharkeye-329715.cloudfunctions.net/sign-up"
    "?request=report_feedback"
)

FEEDBACK_MAX_CHARS = 750
FEEDBACK_NAME_KEY = "feedback_name"
FEEDBACK_EMAIL_KEY = "feedback_email"


def get_commit_hash() -> str:
    """Best-effort commit SHA: bundled version.json, then env, then git HEAD."""
    try:
        with open(resource_path("version.json"), encoding="utf-8") as f:
            data = json.load(f)
        commit = data.get("commit") if isinstance(data, dict) else None
        if commit:
            return str(commit)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    env_sha = os.environ.get("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha

    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            capture_output=True,
            text=True,
            check=True,
            **kwargs,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_feedback_payload(name: str, email: str, feedback: str) -> dict:
    return {
        "name": name,
        "email": email,
        "feedback": feedback,
        "commit_hash": get_commit_hash(),
        "help_docs_version": read_local_doc_version(),
    }


def submit_feedback_payload(payload: dict) -> None:
    """POST the feedback JSON to the Cloud Function."""
    if not FEEDBACK_ENDPOINT:
        print(json.dumps(payload, indent=2))
        return

    import requests

    response = requests.post(FEEDBACK_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()


class ReportProblemDialog(QDialog):
    """Modal form for sending name/email/feedback to the team."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Report a Problem")
        self.setMinimumWidth(480)
        self.resize(520, 520)
        self._settings = QSettings("BOSL", "SharkEye_App")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setText(str(self._settings.value(FEEDBACK_NAME_KEY, "") or ""))
        layout.addWidget(self.name_input)

        layout.addWidget(QLabel("Email:"))
        self.email_input = QLineEdit()
        self.email_input.setText(str(self._settings.value(FEEDBACK_EMAIL_KEY, "") or ""))
        layout.addWidget(self.email_input)

        self.feedback_label = QLabel(f"Feedback (750 characters remaining):")
        layout.addWidget(self.feedback_label)
        self.feedback_edit = QPlainTextEdit()
        self.feedback_edit.setPlaceholderText("Describe the problem…")
        self.feedback_edit.textChanged.connect(self._on_feedback_changed)
        layout.addWidget(self.feedback_edit, 1)

        self.share_button = QPushButton("Submit")
        self.share_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.share_button.setEnabled(False)
        self.share_button.clicked.connect(self._share_feedback)
        layout.addWidget(self.share_button)

    def _on_feedback_changed(self) -> None:
        text = self.feedback_edit.toPlainText()
        if len(text) > FEEDBACK_MAX_CHARS:
            cursor = self.feedback_edit.textCursor()
            position = min(cursor.position(), FEEDBACK_MAX_CHARS)
            self.feedback_edit.blockSignals(True)
            self.feedback_edit.setPlainText(text[:FEEDBACK_MAX_CHARS])
            cursor.setPosition(position)
            self.feedback_edit.setTextCursor(cursor)
            self.feedback_edit.blockSignals(False)
            text = text[:FEEDBACK_MAX_CHARS]
        self.feedback_label.setText(f"Feedback ({FEEDBACK_MAX_CHARS - len(text)} characters remaining):")
        self.share_button.setEnabled(bool(text.strip()))

    def _share_feedback(self) -> None:
        name = self.name_input.text().strip()
        email = self.email_input.text().strip()
        feedback = self.feedback_edit.toPlainText().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter your name.")
            return
        if not email:
            QMessageBox.warning(self, "Email Required", "Please enter your email.")
            return
        if not feedback:
            QMessageBox.warning(self, "Feedback Required", "Please enter your feedback.")
            return
        if email and "@" not in email:
            QMessageBox.warning(
                self,
                "Invalid Email",
                "Please enter a valid email address, or leave the field blank.",
            )
            return

        payload = build_feedback_payload(name, email, feedback)

        try:
            submit_feedback_payload(payload)
        except Exception as exc:
            logger.error(f"Feedback submission failed: {exc}")
            QMessageBox.critical(
                self,
                "Send Failed",
                f"Could not share feedback with the team:\n{exc}",
            )
            return

        logger.info(
            "Feedback submitted (commit=%s, docs=%s)",
            payload["commit_hash"],
            payload["help_docs_version"],
        )
        self._settings.setValue(FEEDBACK_NAME_KEY, name)
        self._settings.setValue(FEEDBACK_EMAIL_KEY, email)
        QMessageBox.information(
            self,
            "Thank You",
            "Your feedback has been shared with the team.",
        )
        self.accept()
