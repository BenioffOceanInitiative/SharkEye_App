"""Interactive frame viewer for drawing a line segment on a still image."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from segmentation.segmentation_model import calculate_shark_length_from_pixel


def _default_drone_settings() -> dict:
    return {
        "Mavic 2 Pro": {
            "Resolution": {
                "(2688, 1512)": math.radians(73),
            },
        },
        "Air 2S": {
            "Resolution": {
                "(2688, 1512)": math.radians(63.5),
                "(5472, 3078)": math.radians(82.9),
            },
        },
    }


def load_drone_settings(settings_obj: QSettings | None = None) -> dict:
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    value = settings_obj.value("drone_settings")
    if not value:
        defaults = _default_drone_settings()
        settings_obj.setValue("drone_settings", json.dumps(defaults))
        return defaults
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return _default_drone_settings()


def format_length_feet(length_ft: float) -> str:
    feet, inches = divmod(length_ft, 1)
    return f"{int(feet)}ft{int(inches * 12)}in"


class ZoomableFrameView(QWidget):
    """Displays a frame with mouse-wheel zoom, drag-to-pan, and optional line drawing."""

    line_changed = pyqtSignal(object)
    line_length_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._line_start: QPoint | None = None
        self._line_end: QPoint | None = None
        self._preview_end: QPoint | None = None
        self._drawing_mode = False
        self._panning = False
        self._drawing_line = False
        self._last_mouse_pos = QPointF()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def load_image(self, image_path: str | Path) -> bool:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self._line_start = None
        self._line_end = None
        self._preview_end = None
        self._fit_to_view()
        self.update()
        return True

    def set_drawing_mode(self, enabled: bool) -> None:
        self._drawing_mode = enabled
        self._drawing_line = False
        self._preview_end = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )
        self.update()

    def clear_line(self) -> None:
        self._line_start = None
        self._line_end = None
        self._preview_end = None
        self._drawing_line = False
        self.line_changed.emit(None)
        self._emit_line_length_changed()
        self.update()

    def get_line(self) -> tuple[QPoint, QPoint] | None:
        if self._line_start is None or self._line_end is None:
            return None
        return (QPoint(self._line_start), QPoint(self._line_end))

    def get_active_line_length_pixels(self) -> float | None:
        """Return line length in original-image pixels for the active segment."""
        line_start = self._line_start
        line_end = self._preview_end if self._drawing_line else self._line_end
        if line_start is None or line_end is None:
            return None
        dx = line_end.x() - line_start.x()
        dy = line_end.y() - line_start.y()
        return math.hypot(dx, dy)

    def image_size(self) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return self._pixmap.width(), self._pixmap.height()

    def _emit_line_length_changed(self) -> None:
        self.line_length_changed.emit(self.get_active_line_length_pixels())

    def _display_scale(self) -> float:
        return self._fit_scale * self._zoom

    def _update_fit_scale(self) -> None:
        """Minimum scale that fills the viewport with no exposed background."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        widget_w = max(self.width(), 1)
        widget_h = max(self.height(), 1)
        self._fit_scale = max(
            widget_w / self._pixmap.width(),
            widget_h / self._pixmap.height(),
        )

    def _fit_to_view(self) -> None:
        self._update_fit_scale()
        self._zoom = 1.0
        self._center_image()

    def _center_image(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scale = self._display_scale()
        image_w = self._pixmap.width() * scale
        image_h = self._pixmap.height() * scale
        self._pan_offset = QPointF(
            (self.width() - image_w) / 2.0,
            (self.height() - image_h) / 2.0,
        )
        self._clamp_pan_offset()

    def _clamp_pan_offset(self) -> None:
        """Keep the image covering the viewport so no background is exposed."""
        if self._pixmap is None or self._pixmap.isNull():
            return

        scale = self._display_scale()
        image_w = self._pixmap.width() * scale
        image_h = self._pixmap.height() * scale
        view_w = self.width()
        view_h = self.height()

        if image_w <= view_w:
            pan_x = (view_w - image_w) / 2.0
        else:
            pan_x = min(0.0, max(view_w - image_w, self._pan_offset.x()))

        if image_h <= view_h:
            pan_y = (view_h - image_h) / 2.0
        else:
            pan_y = min(0.0, max(view_h - image_h, self._pan_offset.y()))

        self._pan_offset = QPointF(pan_x, pan_y)

    def _preserve_view_center_on_resize(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        old_scale = self._display_scale()
        widget_center = QPointF(self.width() / 2.0, self.height() / 2.0)
        image_center = QPointF(
            (widget_center.x() - self._pan_offset.x()) / old_scale,
            (widget_center.y() - self._pan_offset.y()) / old_scale,
        )
        self._update_fit_scale()
        new_scale = self._display_scale()
        self._pan_offset = QPointF(
            widget_center.x() - image_center.x() * new_scale,
            widget_center.y() - image_center.y() * new_scale,
        )
        self._clamp_pan_offset()

    def _widget_to_image(self, widget_pos: QPointF) -> QPoint | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale = self._display_scale()
        if scale <= 0:
            return None
        image_x = (widget_pos.x() - self._pan_offset.x()) / scale
        image_y = (widget_pos.y() - self._pan_offset.y()) / scale
        if (
            image_x < 0
            or image_y < 0
            or image_x > self._pixmap.width()
            or image_y > self._pixmap.height()
        ):
            return None
        return QPoint(int(round(image_x)), int(round(image_y)))

    def _image_to_widget(self, image_pos: QPoint) -> QPointF:
        scale = self._display_scale()
        return QPointF(
            image_pos.x() * scale + self._pan_offset.x(),
            image_pos.y() * scale + self._pan_offset.y(),
        )

    def _zoom_at(self, factor: float, anchor: QPointF) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        old_scale = self._display_scale()
        image_anchor = QPointF(
            (anchor.x() - self._pan_offset.x()) / old_scale,
            (anchor.y() - self._pan_offset.y()) / old_scale,
        )
        self._zoom = max(1.0, min(self._zoom * factor, 20.0))
        new_scale = self._display_scale()
        self._pan_offset = QPointF(
            anchor.x() - image_anchor.x() * new_scale,
            anchor.y() - image_anchor.y() * new_scale,
        )
        self._clamp_pan_offset()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom_at(factor, event.position())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap is None:
            super().mousePressEvent(event)
            return

        pos = QPointF(event.position())
        if self._drawing_mode:
            image_pos = self._widget_to_image(pos)
            if image_pos is None:
                return
            self._line_start = image_pos
            self._line_end = None
            self._preview_end = image_pos
            self._drawing_line = True
            self._emit_line_length_changed()
            self.update()
            return

        self._panning = True
        self._last_mouse_pos = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = QPointF(event.position())

        if self._drawing_line:
            image_pos = self._widget_to_image(pos)
            if image_pos is not None:
                self._preview_end = image_pos
                self._emit_line_length_changed()
                self.update()
            return

        if self._panning:
            delta = pos - self._last_mouse_pos
            self._pan_offset += delta
            self._clamp_pan_offset()
            self._last_mouse_pos = pos
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._drawing_line:
            image_pos = self._widget_to_image(QPointF(event.position()))
            if image_pos is not None and self._line_start is not None:
                self._line_end = image_pos
                self._preview_end = None
                self._drawing_line = False
                self.line_changed.emit(self.get_line())
                self._emit_line_length_changed()
                self.update()
            else:
                self._line_start = None
                self._preview_end = None
                self._drawing_line = False
                self._emit_line_length_changed()
                self.update()
            return

        if self._panning:
            self._panning = False
            self.setCursor(
                Qt.CursorShape.CrossCursor
                if self._drawing_mode
                else Qt.CursorShape.OpenHandCursor
            )
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap is not None and not self._pixmap.isNull():
            self._preserve_view_center_on_resize()
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return

        scale = self._display_scale()
        target = QRect(
            int(self._pan_offset.x()),
            int(self._pan_offset.y()),
            int(self._pixmap.width() * scale),
            int(self._pixmap.height() * scale),
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap)

        line_start = self._line_start
        line_end = self._preview_end if self._drawing_line else self._line_end
        if line_start is not None and line_end is not None:
            pen = QPen(Qt.GlobalColor.red)
            pen.setWidth(2)
            painter.setPen(pen)
            start = self._image_to_widget(line_start)
            end = self._image_to_widget(line_end)
            painter.drawLine(start, end)


class FrameLineEditorWindow(QMainWindow):
    """Window for inspecting a frame and drawing a measurement line."""

    changes_confirmed = pyqtSignal(object)

    def __init__(
        self,
        image_path: str | Path,
        parent=None,
        settings_obj: QSettings | None = None,
        drone_altitude: float = 40.0,
        initial_drone: str | None = None,
    ):
        super().__init__(parent)
        self._image_path = Path(image_path)
        self._settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
        self._drone_settings = load_drone_settings(self._settings_obj)
        self._default_altitude = drone_altitude
        self._initial_drone = initial_drone
        self._drawing_mode = False
        self._build_ui()
        if not self._view.load_image(self._image_path):
            self.setWindowTitle("Frame Line Editor - Failed to load image")
        else:
            self.setWindowTitle(f"Frame Line Editor - {self._image_path.name}")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._view = ZoomableFrameView()
        layout.addWidget(self._view, stretch=1)

        drone_row = QHBoxLayout()
        drone_row.addWidget(QLabel("Drone:"))
        self._drone_select = QComboBox()
        self._drone_select.addItems(list(self._drone_settings.keys()))
        drone_row.addWidget(self._drone_select, stretch=1)

        drone_row.addWidget(QLabel("Altitude (m):"))
        self._altitude_input = QLineEdit(str(self._default_altitude))
        self._altitude_input.setValidator(QDoubleValidator(0.0, 999.0, 2))
        self._altitude_input.setFixedWidth(80)
        drone_row.addWidget(self._altitude_input)
        layout.addLayout(drone_row)

        self._line_length_label = QLabel("Line length: — px")
        self._line_length_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._line_length_label)

        self._shark_length_label = QLabel("Shark length: —")
        self._shark_length_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._shark_length_label)

        self._drone_select.currentTextChanged.connect(self._update_line_length_label)
        self._altitude_input.textChanged.connect(self._update_line_length_label)
        self._view.line_length_changed.connect(self._update_line_length_label)
        self._populate_initial_drone()

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self._draw_line_button = QPushButton("Draw Line")
        self._draw_line_button.setCheckable(True)
        self._draw_line_button.toggled.connect(self._toggle_draw_mode)

        self._clear_line_button = QPushButton("Clear Line")
        self._clear_line_button.clicked.connect(self._view.clear_line)

        self._confirm_button = QPushButton("Confirm Changes")
        self._confirm_button.clicked.connect(self._confirm_changes)

        button_row.addWidget(self._draw_line_button)
        button_row.addWidget(self._clear_line_button)
        button_row.addStretch(1)
        button_row.addWidget(self._confirm_button)
        layout.addLayout(button_row)

        self.resize(960, 720)

    def _populate_initial_drone(self) -> None:
        preferred_drone = self._initial_drone or self._settings_obj.value("last_drone_type")
        if preferred_drone:
            index = self._drone_select.findText(str(preferred_drone))
            if index >= 0:
                self._drone_select.setCurrentIndex(index)

    def _get_image_size(self) -> tuple[int, int] | None:
        return self._view.image_size()

    def _get_selected_fov_radians(self) -> float | None:
        drone_name = self._drone_select.currentText()
        if not drone_name or drone_name not in self._drone_settings:
            return None

        image_size = self._get_image_size()
        if image_size is None:
            return None

        resolution_key = f"({image_size[0]}, {image_size[1]})"
        resolutions = self._drone_settings[drone_name].get("Resolution", {})
        return resolutions.get(resolution_key)

    def _get_altitude_m(self) -> float | None:
        text = self._altitude_input.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _toggle_draw_mode(self, enabled: bool) -> None:
        self._drawing_mode = enabled
        self._view.set_drawing_mode(enabled)

    def _update_line_length_label(self, _length=None) -> None:
        length = self._view.get_active_line_length_pixels()

        if length is None:
            self._line_length_label.setText("Line length: — px")
            self._shark_length_label.setText("Shark length: —")
            return

        self._line_length_label.setText(f"Line length: {length:.1f} px")

        drone_name = self._drone_select.currentText()
        if not drone_name:
            self._shark_length_label.setText("Shark length: —")
            return

        fov_radians = self._get_selected_fov_radians()
        image_size = self._get_image_size()
        altitude_m = self._get_altitude_m()

        if fov_radians is None:
            if image_size is not None:
                self._shark_length_label.setText(
                    f"Shark length: no FOV for {image_size[0]}x{image_size[1]}"
                )
            else:
                self._shark_length_label.setText("Shark length: —")
            return

        if altitude_m is None:
            self._shark_length_label.setText("Shark length: enter altitude")
            return

        length_ft = calculate_shark_length_from_pixel(
            length,
            original_width=image_size[0],
            original_height=image_size[1],
            drone_altitude=altitude_m,
            fov_radians=fov_radians,
        )
        self._shark_length_label.setText(f"Shark length: {format_length_feet(length_ft)}")

    def _confirm_changes(self) -> None:
        line = self._view.get_line()
        length_px = self._view.get_active_line_length_pixels()
        length_ft = None
        if length_px is not None:
            fov_radians = self._get_selected_fov_radians()
            image_size = self._get_image_size()
            altitude_m = self._get_altitude_m()
            if fov_radians is not None and image_size is not None and altitude_m is not None:
                length_ft = calculate_shark_length_from_pixel(
                    length_px,
                    original_width=image_size[0],
                    original_height=image_size[1],
                    drone_altitude=altitude_m,
                    fov_radians=fov_radians,
                )

        self.changes_confirmed.emit(
            {
                "image_path": str(self._image_path),
                "line": (
                    ((line[0].x(), line[0].y()), (line[1].x(), line[1].y()))
                    if line is not None
                    else None
                ),
                "length_pixels": length_px,
                "length_feet": length_ft,
                "drone": self._drone_select.currentText() or None,
                "altitude_m": self._get_altitude_m(),
            }
        )
        self.close()

    def view(self) -> ZoomableFrameView:
        return self._view


def test_frame_line_editor() -> None:
    """Launch the frame line editor with a sample image for manual testing."""
    test_image = Path(
        r"C:\Users\legop\Desktop\GitHub\SharkEye_App\results\02022026_155352\frames"
        r"\TRIMMED_2023-10-23_Transect_DJI_0750.mp4_1.jpg"
    )
    app = QApplication(sys.argv)
    window = FrameLineEditorWindow(test_image)
    window.changes_confirmed.connect(
        lambda result: print(f"Confirmed changes: {result}")
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    test_frame_line_editor()
