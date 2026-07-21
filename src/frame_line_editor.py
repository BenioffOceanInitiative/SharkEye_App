"""Interactive frame viewer for drawing a measurement line on a still image."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QSettings, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDoubleValidator,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from segmentation.segmentation_model import calculate_shark_length_from_pixel
from theme import colored_svg_icon, is_dark_mode
from utility import resource_path


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


def load_annotation_color(settings_obj: QSettings | None = None) -> QColor:
    """Load the bounding-box annotation color from settings (RGB, default neon orange)."""
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    default_color = (255, 96, 31)
    color_str = settings_obj.value(
        "annotation_color",
        f"{default_color[0]},{default_color[1]},{default_color[2]}",
    )
    color_parts = str(color_str).split(",")
    if len(color_parts) == 3:
        try:
            r, g, b = (int(part.strip()) for part in color_parts)
            return QColor(r, g, b)
        except ValueError:
            pass
    return QColor(*default_color)


def _overlay_button_style() -> str:
    if is_dark_mode():
        return """
        QPushButton {
            background-color: rgba(30, 30, 30, 200);
            color: white;
            border: 1px solid rgba(255, 255, 255, 90);
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: rgba(60, 60, 60, 220);
        }
        QPushButton:checked {
            background-color: rgba(40, 90, 140, 220);
        }
        """
    return """
    QPushButton {
        background-color: rgba(230, 230, 230, 200);
        color: #4a4a4a;
        border: 1px solid rgba(0, 0, 0, 70);
        border-radius: 4px;
        padding: 4px 10px;
    }
    QPushButton:hover {
        background-color: rgba(245, 245, 245, 220);
    }
    QPushButton:checked {
        background-color: rgba(200, 220, 240, 220);
    }
    """


def _overlay_panel_style() -> str:
    if is_dark_mode():
        return """
        QWidget#editorOverlayPanel {
            background-color: rgba(20, 20, 20, 170);
            border-radius: 6px;
        }
        QLabel {
            color: white;
            background: transparent;
        }
        QComboBox, QLineEdit {
            background-color: rgba(40, 40, 40, 220);
            color: white;
            border: 1px solid rgba(255, 255, 255, 80);
            border-radius: 3px;
            padding: 2px 4px;
        }
        """
    return """
    QWidget#editorOverlayPanel {
        background-color: rgba(235, 235, 235, 180);
        border-radius: 6px;
    }
    QLabel {
        color: #4a4a4a;
        background: transparent;
    }
    QComboBox, QLineEdit {
        background-color: rgba(245, 245, 245, 220);
        color: #4a4a4a;
        border: 1px solid rgba(0, 0, 0, 70);
        border-radius: 3px;
        padding: 2px 4px;
    }
    """


def _overlay_hud_style() -> str:
    """Length readout styled like the overlay buttons (boxed, translucent)."""
    if is_dark_mode():
        return """
        QLabel#editorLengthHud {
            background-color: rgba(30, 30, 30, 200);
            color: white;
            border: 1px solid rgba(255, 255, 255, 90);
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 600;
        }
        """
    return """
    QLabel#editorLengthHud {
        background-color: rgba(230, 230, 230, 200);
        color: #4a4a4a;
        border: 1px solid rgba(0, 0, 0, 70);
        border-radius: 4px;
        padding: 4px 10px;
        font-weight: 600;
    }
    """


# Matches the light overlay panel fill so the off-screen arrow reads as part of the HUD chrome.
_ARROW_COLOR = QColor(235, 235, 235, 180)


# arrow-down-short points south; positive Qt rotation is clockwise.
_ARROW_ROTATION_BY_DIR: dict[tuple[int, int], float] = {
    (0, 1): 0.0,      # S
    (-1, 1): 45.0,    # SW
    (-1, 0): 90.0,    # W
    (-1, -1): 135.0,  # NW
    (0, -1): 180.0,   # N
    (1, -1): 225.0,   # NE
    (1, 0): 270.0,    # E
    (1, 1): 315.0,    # SE
}


def _offscreen_direction(anchor: QPointF, rect: QRect) -> tuple[int, int]:
    """Return (dx, dy) in {-1, 0, 1} for where ``anchor`` sits relative to ``rect``."""
    dx = 0
    if anchor.x() < rect.left():
        dx = -1
    elif anchor.x() > rect.right():
        dx = 1
    dy = 0
    if anchor.y() < rect.top():
        dy = -1
    elif anchor.y() > rect.bottom():
        dy = 1
    return dx, dy


def _rotated_arrow_pixmap(degrees: float, size: int = 16) -> QPixmap:
    """Render arrow-down-short in the HUD panel color, rotated clockwise."""
    icon = colored_svg_icon(
        resource_path("assets/images/arrow-down-short.svg"),
        _ARROW_COLOR,
        size,
    )
    pixmap = icon.pixmap(size, size)
    return pixmap.transformed(
        QTransform().rotate(degrees),
        Qt.TransformationMode.SmoothTransformation,
    )


def _min_shift_to_clear(group: QRect, obstacle: QRect) -> QPoint:
    """Smallest axis-aligned shift that separates ``group`` from ``obstacle``."""
    inter = group.intersected(obstacle)
    if inter.isEmpty():
        return QPoint(0, 0)
    candidates = [
        QPoint(obstacle.right() + 1 - group.left(), 0),   # push right
        QPoint(obstacle.left() - group.right() - 1, 0),   # push left
        QPoint(0, obstacle.bottom() + 1 - group.top()),   # push down
        QPoint(0, obstacle.top() - group.bottom() - 1),   # push up
    ]
    return min(candidates, key=lambda p: abs(p.x()) + abs(p.y()))


class ZoomableFrameView(QWidget):
    """Displays a frame with mouse-wheel zoom, drag-to-pan, and optional line drawing.

    Image layout matches ``FramePlayer``: KeepAspectRatio, centered, with no black
    letterbox fill. Zoom/pan build on top of that fit.
    """

    line_changed = pyqtSignal(object)
    line_length_changed = pyqtSignal(object)
    # (cursor_widget_pos: QPointF | None, length_px: float | None) while dragging a segment.
    drawing_hud_changed = pyqtSignal(object, object)
    view_changed = pyqtSignal()
    resized = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._points: list[QPoint] = []
        self._preview_end: QPoint | None = None
        self._drawing_mode = False
        self._panning = False
        self._drawing_line = False
        self._last_mouse_pos = QPointF()
        self._line_color = QColor(255, 96, 31)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def load_image(self, image_path: str | Path) -> bool:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self._points = []
        self._preview_end = None
        self._fit_to_view()
        self.update()
        self.resized.emit()
        return True

    def set_line_color(self, color: QColor) -> None:
        self._line_color = QColor(color)
        self.update()

    def set_drawing_mode(self, enabled: bool) -> None:
        self._drawing_mode = enabled
        self._drawing_line = False
        self._preview_end = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )
        self._emit_drawing_hud(None)
        self.update()

    def clear_line(self) -> None:
        self._points = []
        self._preview_end = None
        self._drawing_line = False
        self.line_changed.emit(None)
        self._emit_line_length_changed()
        self._emit_drawing_hud(None)
        self.update()

    def get_line(self) -> tuple[QPoint, QPoint] | None:
        """Return the first and last polyline vertices (endpoints of the path)."""
        if len(self._points) < 2:
            return None
        return (QPoint(self._points[0]), QPoint(self._points[-1]))

    def get_polyline(self) -> list[QPoint] | None:
        """Return every vertex of the drawn multi-segment line, in order."""
        if len(self._points) < 2:
            return None
        return [QPoint(point) for point in self._points]

    def is_drawing_line(self) -> bool:
        return self._drawing_line

    def last_endpoint_widget_pos(self) -> QPointF | None:
        """Widget coordinates of the last confirmed polyline vertex."""
        if len(self._points) < 2:
            return None
        return self._image_to_widget(self._points[-1])

    def _active_points(self) -> list[QPoint]:
        """Confirmed vertices plus the live preview endpoint while drawing."""
        points = list(self._points)
        if self._drawing_line and self._preview_end is not None:
            points.append(self._preview_end)
        return points

    def get_active_line_length_pixels(self) -> float | None:
        """Return total polyline length in original-image pixels across all segments."""
        points = self._active_points()
        if len(points) < 2:
            return None
        total = 0.0
        for start, end in zip(points, points[1:]):
            total += math.hypot(end.x() - start.x(), end.y() - start.y())
        return total

    def image_size(self) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return self._pixmap.width(), self._pixmap.height()

    def content_rect(self) -> QRect:
        """Return the fixed KeepAspectRatio-centered image rect inside the widget.

        Matches ``FramePlayer.content_rect`` / ``FramePlayer.paintEvent``. This
        rect does not change with zoom — zoom/pan are clipped inside it so the
        aspect-ratio borders stay put. Overlay controls should use this rect.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return QRect()

        frame_size = self._pixmap.size()
        widget_size = self.size()
        scaled = frame_size.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
        x = (widget_size.width() - scaled.width()) // 2
        y = (widget_size.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def _emit_line_length_changed(self) -> None:
        self.line_length_changed.emit(self.get_active_line_length_pixels())

    def _emit_drawing_hud(self, cursor_pos: QPointF | None) -> None:
        if cursor_pos is None or not self._drawing_line:
            self.drawing_hud_changed.emit(None, None)
            return
        self.drawing_hud_changed.emit(cursor_pos, self.get_active_line_length_pixels())

    def _display_scale(self) -> float:
        return self._fit_scale * self._zoom

    def _update_fit_scale(self) -> None:
        """KeepAspectRatio fit scale (same as FramePlayer: full image visible)."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        widget_w = max(self.width(), 1)
        widget_h = max(self.height(), 1)
        self._fit_scale = min(
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
        rect = self.content_rect()
        if self._zoom == 1.0:
            self._pan_offset = QPointF(float(rect.x()), float(rect.y()))
            return

        scale = self._display_scale()
        image_w = self._pixmap.width() * scale
        image_h = self._pixmap.height() * scale
        self._pan_offset = QPointF(
            rect.x() + (rect.width() - image_w) / 2.0,
            rect.y() + (rect.height() - image_h) / 2.0,
        )
        self._clamp_pan_offset()

    def _clamp_pan_offset(self) -> None:
        """Keep the zoomed image inside the fixed KeepAspectRatio content_rect."""
        if self._pixmap is None or self._pixmap.isNull():
            return

        rect = self.content_rect()
        scale = self._display_scale()
        image_w = self._pixmap.width() * scale
        image_h = self._pixmap.height() * scale

        if image_w <= rect.width():
            pan_x = rect.x() + (rect.width() - image_w) / 2.0
        else:
            pan_x = min(float(rect.x()), max(rect.x() + rect.width() - image_w, self._pan_offset.x()))

        if image_h <= rect.height():
            pan_y = rect.y() + (rect.height() - image_h) / 2.0
        else:
            pan_y = min(float(rect.y()), max(rect.y() + rect.height() - image_h, self._pan_offset.y()))

        self._pan_offset = QPointF(pan_x, pan_y)

    def _preserve_view_center_on_resize(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        if self._zoom == 1.0:
            self._fit_to_view()
            return

        old_scale = self._display_scale()
        old_rect = self.content_rect()
        # Refresh fit scale for the new widget size, then recompute content_rect.
        self._update_fit_scale()
        new_rect = self.content_rect()
        if old_rect.isNull() or new_rect.isNull() or old_scale <= 0:
            self._center_image()
            return

        # Preserve the image point under the center of the previous content_rect.
        old_center = QPointF(
            old_rect.x() + old_rect.width() / 2.0,
            old_rect.y() + old_rect.height() / 2.0,
        )
        image_center = QPointF(
            (old_center.x() - self._pan_offset.x()) / old_scale,
            (old_center.y() - self._pan_offset.y()) / old_scale,
        )
        new_scale = self._display_scale()
        new_center = QPointF(
            new_rect.x() + new_rect.width() / 2.0,
            new_rect.y() + new_rect.height() / 2.0,
        )
        self._pan_offset = QPointF(
            new_center.x() - image_center.x() * new_scale,
            new_center.y() - image_center.y() * new_scale,
        )
        self._clamp_pan_offset()

    def _widget_to_image(self, widget_pos: QPointF) -> QPoint | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        # Ignore clicks outside the fixed aspect-ratio frame.
        if not self.content_rect().contains(widget_pos.toPoint()):
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
        rect = self.content_rect()
        if rect.isNull():
            return
        # Zoom relative to the content_rect; ignore anchors outside the frame.
        if not rect.contains(anchor.toPoint()):
            anchor = QPointF(
                rect.x() + rect.width() / 2.0,
                rect.y() + rect.height() / 2.0,
            )

        old_scale = self._display_scale()
        image_anchor = QPointF(
            (anchor.x() - self._pan_offset.x()) / old_scale,
            (anchor.y() - self._pan_offset.y()) / old_scale,
        )
        self._zoom = max(1.0, min(self._zoom * factor, 20.0))
        self._update_fit_scale()
        new_scale = self._display_scale()
        self._pan_offset = QPointF(
            anchor.x() - image_anchor.x() * new_scale,
            anchor.y() - image_anchor.y() * new_scale,
        )
        self._clamp_pan_offset()
        self.update()
        self.view_changed.emit()

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
            # Holding Shift after a completed line continues the path from its
            # last point instead of starting a brand new one.
            extend = (
                bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                and len(self._points) >= 2
            )
            if not extend:
                self._points = [image_pos]
            self._preview_end = image_pos
            self._drawing_line = True
            self._emit_line_length_changed()
            self._emit_drawing_hud(pos)
            self.update()
            return

        # Only pan when the cursor is inside the aspect-ratio frame.
        if not self.content_rect().contains(pos.toPoint()):
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
                self._emit_drawing_hud(pos)
                self.update()
            return

        if self._panning:
            delta = pos - self._last_mouse_pos
            self._pan_offset += delta
            self._clamp_pan_offset()
            self._last_mouse_pos = pos
            self.update()
            self.view_changed.emit()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._drawing_line:
            image_pos = self._widget_to_image(QPointF(event.position()))
            if image_pos is not None and self._points:
                self._points.append(image_pos)
                self._preview_end = None
                self._drawing_line = False
                self.line_changed.emit(self.get_line())
                self._emit_line_length_changed()
                self._emit_drawing_hud(None)
                self.update()
            else:
                # Discard an incomplete first segment; keep any prior polyline.
                if len(self._points) < 2:
                    self._points = []
                self._preview_end = None
                self._drawing_line = False
                self._emit_line_length_changed()
                self._emit_drawing_hud(None)
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
        # Mirror FramePlayer: preserve layout, emit resized, then repaint.
        super().resizeEvent(event)
        if self._pixmap is not None and not self._pixmap.isNull():
            self._preserve_view_center_on_resize()
        self.resized.emit()
        self.view_changed.emit()
        self.update()

    def paintEvent(self, event) -> None:
        # Mirror FramePlayer.paintEvent: KeepAspectRatio frame stays fixed.
        # Zoom/pan happen inside that content_rect (clipped), so borders never change.
        painter = QPainter(self)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(self.palette().color(self.foregroundRole()))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return

        rect = self.content_rect()
        if rect.isNull():
            return

        self._update_fit_scale()
        if self._zoom == 1.0:
            self._pan_offset = QPointF(float(rect.x()), float(rect.y()))

        scale = self._display_scale()
        target = QRect(
            int(round(self._pan_offset.x())),
            int(round(self._pan_offset.y())),
            int(round(self._pixmap.width() * scale)),
            int(round(self._pixmap.height() * scale)),
        )

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setClipRect(rect)
        painter.drawPixmap(target, self._pixmap)

        points = self._active_points()
        if len(points) >= 2:
            pen = QPen(self._line_color)
            pen.setWidth(2)
            painter.setPen(pen)
            widget_points = [self._image_to_widget(point) for point in points]
            for start, end in zip(widget_points, widget_points[1:]):
                painter.drawLine(start, end)


class FrameLineEditorWidget(QWidget):
    """Drop-in editor that fills its parent and overlays controls inside the frame."""

    changes_confirmed = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        settings_obj: QSettings | None = None,
        drone_altitude: float = 40.0,
        initial_drone: str | None = None,
    ):
        super().__init__(parent)
        self._image_path: Path | None = None
        self._settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
        self._drone_settings = load_drone_settings(self._settings_obj)
        self._default_altitude = drone_altitude
        self._initial_drone = initial_drone
        self._drawing_mode = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build_ui()
        self._populate_initial_drone()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._view = ZoomableFrameView()
        self._view.set_line_color(load_annotation_color(self._settings_obj))
        root.addWidget(self._view, stretch=1)

        # Overlays are children of the view so they sit inside the frame area.
        self._top_panel = QWidget(self._view)
        self._top_panel.setObjectName("editorOverlayPanel")
        top_layout = QHBoxLayout(self._top_panel)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.setSpacing(8)

        top_layout.addWidget(QLabel("Drone:"))
        self._drone_select = QComboBox()
        self._drone_select.addItems(list(self._drone_settings.keys()))
        self._drone_select.setMinimumWidth(120)
        top_layout.addWidget(self._drone_select)

        top_layout.addWidget(QLabel("Altitude (m):"))
        self._altitude_input = QLineEdit(str(self._default_altitude))
        self._altitude_input.setValidator(QDoubleValidator(0.0, 999.0, 2))
        self._altitude_input.setFixedWidth(70)
        top_layout.addWidget(self._altitude_input)

        # Floating shark-length HUD that follows the cursor while drawing.
        self._length_hud = QLabel(self._view)
        self._length_hud.setObjectName("editorLengthHud")
        self._length_hud.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._length_hud.setVisible(False)
        self._length_hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Off-screen direction indicator (arrow-down-short, rotated).
        self._length_hud_arrow = QLabel(self._view)
        self._length_hud_arrow.setFixedSize(16, 16)
        self._length_hud_arrow.setStyleSheet("background: transparent; border: none;")
        self._length_hud_arrow.setVisible(False)
        self._length_hud_arrow.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._length_hud_arrow.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, True
        )

        self._bottom_left_panel = QWidget(self._view)
        self._bottom_left_panel.setObjectName("editorOverlayPanel")
        left_layout = QHBoxLayout(self._bottom_left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        self._cancel_button = QPushButton("Cancel Changes")
        self._cancel_button.clicked.connect(self._cancel_changes)
        left_layout.addWidget(self._cancel_button)

        self._confirm_button = QPushButton("Confirm Changes")
        self._confirm_button.clicked.connect(self._confirm_changes)
        left_layout.addWidget(self._confirm_button)

        self._bottom_right_panel = QWidget(self._view)
        self._bottom_right_panel.setObjectName("editorOverlayPanel")
        right_layout = QHBoxLayout(self._bottom_right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        self._draw_line_button = QPushButton()
        self._draw_line_button.setCheckable(True)
        self._draw_line_button.toggled.connect(self._toggle_draw_mode)
        self._update_draw_button_text(False)
        right_layout.addWidget(self._draw_line_button)

        self._clear_line_button = QPushButton("Clear Line")
        self._clear_line_button.clicked.connect(self._view.clear_line)
        right_layout.addWidget(self._clear_line_button)

        self._toggle_draw_shortcut = QShortcut(QKeySequence(Qt.Key.Key_R), self)
        self._toggle_draw_shortcut.activated.connect(self._draw_line_button.toggle)

        self._drone_select.currentTextChanged.connect(self._refresh_length_hud)
        self._altitude_input.textChanged.connect(self._refresh_length_hud)
        self._view.drawing_hud_changed.connect(self._on_drawing_hud_changed)
        self._view.line_changed.connect(lambda _line: self._refresh_length_hud())
        self._view.resized.connect(self._position_overlays)
        self._view.view_changed.connect(self._refresh_length_hud)

        self._apply_overlay_styles()
        self._position_overlays()

    def _apply_overlay_styles(self) -> None:
        """Apply light/dark overlay styles that match the rest of the app."""
        panel_style = _overlay_panel_style()
        button_style = _overlay_button_style()
        self._top_panel.setStyleSheet(panel_style)
        self._bottom_left_panel.setStyleSheet(panel_style)
        self._bottom_right_panel.setStyleSheet(panel_style)
        self._draw_line_button.setStyleSheet(button_style)
        self._clear_line_button.setStyleSheet(button_style)
        self._cancel_button.setStyleSheet(button_style)
        self._confirm_button.setStyleSheet(button_style)
        self._length_hud.setStyleSheet(_overlay_hud_style())
        # Refresh arrow tint if theme changed since last show.
        if self._length_hud_arrow.isVisible():
            self._refresh_length_hud()

    def _position_overlays(self) -> None:
        """Place overlay controls inside the FramePlayer-style content_rect."""
        margin = 8
        rect = self._view.content_rect()
        if rect.isNull():
            rect = self._view.rect()

        self._top_panel.adjustSize()
        top_w = min(self._top_panel.sizeHint().width(), max(rect.width() - 2 * margin, 1))
        top_h = self._top_panel.sizeHint().height()
        self._top_panel.setGeometry(rect.x() + margin, rect.y() + margin, top_w, top_h)
        self._top_panel.raise_()

        self._bottom_left_panel.adjustSize()
        left_hint = self._bottom_left_panel.sizeHint()
        left_y = rect.y() + rect.height() - left_hint.height() - margin
        self._bottom_left_panel.setGeometry(
            rect.x() + margin,
            max(left_y, rect.y() + margin),
            left_hint.width(),
            left_hint.height(),
        )
        self._bottom_left_panel.raise_()

        self._bottom_right_panel.adjustSize()
        right_hint = self._bottom_right_panel.sizeHint()
        right_x = rect.x() + rect.width() - right_hint.width() - margin
        right_y = rect.y() + rect.height() - right_hint.height() - margin
        self._bottom_right_panel.setGeometry(
            max(right_x, rect.x() + margin),
            max(right_y, rect.y() + margin),
            right_hint.width(),
            right_hint.height(),
        )
        self._bottom_right_panel.raise_()
        if self._length_hud.isVisible():
            self._length_hud.raise_()
        if self._length_hud_arrow.isVisible():
            self._length_hud_arrow.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_overlays()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._view.set_line_color(load_annotation_color(self._settings_obj))
        self._apply_overlay_styles()
        self._position_overlays()

    def load_image(
        self,
        image_path: str | Path,
        drone_altitude: float | None = None,
        initial_drone: str | None = None,
    ) -> bool:
        """Load a frame into the editor and reset drawing state."""
        self._image_path = Path(image_path)
        self._view.set_line_color(load_annotation_color(self._settings_obj))
        if drone_altitude is not None:
            self._altitude_input.setText(str(drone_altitude))
        if initial_drone is not None:
            self._initial_drone = initial_drone
            self._populate_initial_drone()

        loaded = self._view.load_image(self._image_path)
        if self._draw_line_button.isChecked():
            self._draw_line_button.setChecked(False)
        else:
            self._toggle_draw_mode(False)
        self._length_hud.setVisible(False)
        self._length_hud_arrow.setVisible(False)
        self._position_overlays()
        return loaded

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
        self._update_draw_button_text(enabled)
        self._refresh_length_hud()

    def _update_draw_button_text(self, drawing: bool) -> None:
        self._draw_line_button.setText(
            "Move Frame (R)" if drawing else "Draw Line (R)"
        )

    def _format_shark_length_text(self, length_px: float | None) -> str | None:
        """Return a compact shark-length string (e.g. ``8ft4in``), or None if unknown."""
        if length_px is None:
            return None
        fov_radians = self._get_selected_fov_radians()
        image_size = self._get_image_size()
        altitude_m = self._get_altitude_m()
        if fov_radians is None or image_size is None or altitude_m is None:
            return None
        length_ft = calculate_shark_length_from_pixel(
            length_px,
            original_width=image_size[0],
            original_height=image_size[1],
            drone_altitude=altitude_m,
            fov_radians=fov_radians,
        )
        return format_length_feet(length_ft)

    def _corner_hud_inset(self) -> int:
        """Inset from frame edges when the length HUD is in a corner.

        Matches cancel/confirm panel height + 2× the overlay margin used to place it.
        """
        margin = 8
        panel = self._bottom_left_panel
        panel_h = panel.height() if panel.height() > 0 else panel.sizeHint().height()
        return panel_h + 2 * margin

    def _overlay_obstacle_rects(self, padding: int = 0) -> list[QRect]:
        """Geometries of other in-frame overlays the length HUD must not cover."""
        rects: list[QRect] = []
        for widget in (
            self._top_panel,
            self._bottom_left_panel,
            self._bottom_right_panel,
        ):
            if widget.isVisible():
                geo = widget.geometry()
                if padding:
                    geo = geo.adjusted(-padding, -padding, padding, padding)
                rects.append(geo)
        return rects

    def _preferred_hud_rect(self, anchor: QPointF, hud_w: int, hud_h: int) -> QRect:
        """Ideal length-box placement above the endpoint (before clamping)."""
        return QRect(
            int(anchor.x() - hud_w / 2),
            int(anchor.y() - hud_h - 12),
            hud_w,
            hud_h,
        )

    def _compute_hud_direction(
        self, anchor: QPointF, hud_w: int, hud_h: int, frame: QRect
    ) -> tuple[int, int]:
        """8-way direction treating frame edges *and* UI overlays as off-screen.

        If the preferred HUD box would hit an overlay (or the endpoint sits in one),
        an arrow direction is returned before a hard collision so the HUD switches
        to edge/arrow mode early.
        """
        frame_dx, frame_dy = _offscreen_direction(anchor, frame)

        # Slight padding so the arrow appears just before a visible overlap.
        obstacles = self._overlay_obstacle_rects(padding=6)
        preferred = self._preferred_hud_rect(anchor, hud_w, hud_h)
        blocked = False
        for obs in obstacles:
            if obs.contains(anchor.toPoint()) or preferred.intersects(obs):
                blocked = True
                break

        obs_dx = obs_dy = 0
        if blocked:
            # Direction from the free frame center toward the endpoint.
            mid = QPointF(frame.center())
            deadzone = 8.0
            if anchor.x() < mid.x() - deadzone:
                obs_dx = -1
            elif anchor.x() > mid.x() + deadzone:
                obs_dx = 1
            if anchor.y() < mid.y() - deadzone:
                obs_dy = -1
            elif anchor.y() > mid.y() + deadzone:
                obs_dy = 1
            # If still undecided (near center but blocked), aim at the blocking panel.
            if obs_dx == 0 and obs_dy == 0:
                for obs in obstacles:
                    if obs.contains(anchor.toPoint()) or preferred.intersects(obs):
                        oc = obs.center()
                        if abs(oc.x() - mid.x()) >= abs(oc.y() - mid.y()):
                            obs_dx = -1 if oc.x() < mid.x() else 1
                        else:
                            obs_dy = -1 if oc.y() < mid.y() else 1
                        break

        dx = frame_dx if frame_dx != 0 else obs_dx
        dy = frame_dy if frame_dy != 0 else obs_dy
        return dx, dy

    def _arrow_offset_for_box(
        self,
        box_x: int,
        box_y: int,
        hud_w: int,
        hud_h: int,
        arrow_w: int,
        arrow_h: int,
        dx: int,
        dy: int,
        gap: int,
    ) -> tuple[int, int]:
        """Arrow position locked to the box's outward face."""
        if dx != 0 and dy != 0:
            arrow_x = box_x + hud_w + gap if dx == 1 else box_x - gap - arrow_w
            arrow_y = box_y + hud_h + gap if dy == 1 else box_y - gap - arrow_h
        elif dx != 0:
            arrow_x = box_x + hud_w + gap if dx == 1 else box_x - gap - arrow_w
            arrow_y = box_y + (hud_h - arrow_h) // 2
        else:
            arrow_x = box_x + (hud_w - arrow_w) // 2
            arrow_y = box_y + hud_h + gap if dy == 1 else box_y - gap - arrow_h
        return arrow_x, arrow_y

    def _resolve_hud_overlay_collisions(
        self,
        box_x: int,
        box_y: int,
        hud_w: int,
        hud_h: int,
        arrow_w: int,
        arrow_h: int,
        dx: int,
        dy: int,
        gap: int,
        show_arrow: bool,
        frame: QRect,
        margin: int,
    ) -> tuple[int, int]:
        """Nudge the length box until the HUD (+ arrow) clears other overlays."""
        obstacles = self._overlay_obstacle_rects()
        if not obstacles:
            return box_x, box_y

        for _ in range(24):
            if show_arrow:
                arrow_x, arrow_y = self._arrow_offset_for_box(
                    box_x, box_y, hud_w, hud_h, arrow_w, arrow_h, dx, dy, gap
                )
                group = QRect(box_x, box_y, hud_w, hud_h).united(
                    QRect(arrow_x, arrow_y, arrow_w, arrow_h)
                )
            else:
                group = QRect(box_x, box_y, hud_w, hud_h)

            shift = QPoint(0, 0)
            for obstacle in obstacles:
                part = _min_shift_to_clear(group, obstacle)
                if part.x() != 0 or part.y() != 0:
                    # Prefer the smallest single-obstacle separation this pass.
                    if shift == QPoint(0, 0) or (
                        abs(part.x()) + abs(part.y()) < abs(shift.x()) + abs(shift.y())
                    ):
                        shift = part

            if shift == QPoint(0, 0):
                break

            box_x += shift.x()
            box_y += shift.y()
            box_x = max(
                frame.x() + margin,
                min(box_x, frame.x() + frame.width() - margin - hud_w),
            )
            box_y = max(
                frame.y() + margin,
                min(box_y, frame.y() + frame.height() - margin - hud_h),
            )

        return box_x, box_y

    def _place_length_hud_at(self, anchor: QPointF, text: str) -> None:
        """Position the boxed length readout; attach a direction arrow when off-frame."""
        self._length_hud.setText(text)
        self._length_hud.adjustSize()
        hud_w = self._length_hud.width()
        hud_h = self._length_hud.height()
        gap = 4
        margin = 8

        rect = self._view.content_rect()
        if rect.isNull():
            rect = self._view.rect()

        dx, dy = self._compute_hud_direction(anchor, hud_w, hud_h, rect)

        if dx == 0 and dy == 0:
            self._length_hud_arrow.setVisible(False)
            box_x = int(anchor.x() - hud_w / 2)
            box_y = int(anchor.y() - hud_h - 12)
            box_x = max(rect.x(), min(box_x, rect.x() + rect.width() - hud_w))
            box_y = max(rect.y(), min(box_y, rect.y() + rect.height() - hud_h))
            box_x, box_y = self._resolve_hud_overlay_collisions(
                box_x, box_y, hud_w, hud_h, 0, 0, 0, 0, gap, False, rect, margin
            )
            self._length_hud.move(box_x, box_y)
            self._length_hud.setVisible(True)
            self._length_hud.raise_()
            return

        rotation = _ARROW_ROTATION_BY_DIR[(dx, dy)]
        arrow_pixmap = _rotated_arrow_pixmap(rotation, 16)
        arrow_w = arrow_pixmap.width()
        arrow_h = arrow_pixmap.height()
        self._length_hud_arrow.setFixedSize(arrow_w, arrow_h)
        self._length_hud_arrow.setPixmap(arrow_pixmap)

        if dx != 0 and dy != 0:
            # Corner: equal clearance from both edges (clears bottom control strips).
            inset = self._corner_hud_inset()
            box_x = (
                rect.x() + rect.width() - inset - hud_w
                if dx == 1
                else rect.x() + inset
            )
            box_y = (
                rect.y() + rect.height() - inset - hud_h
                if dy == 1
                else rect.y() + inset
            )
        else:
            if dx == 1:
                box_x = rect.x() + rect.width() - margin - arrow_w - gap - hud_w
            elif dx == -1:
                box_x = rect.x() + margin + arrow_w + gap
            else:
                box_x = int(anchor.x() - hud_w / 2)

            if dy == 1:
                box_y = rect.y() + rect.height() - margin - arrow_h - gap - hud_h
            elif dy == -1:
                box_y = rect.y() + margin + arrow_h + gap
            else:
                box_y = int(anchor.y() - hud_h - 12)

        box_x = max(rect.x() + margin, min(box_x, rect.x() + rect.width() - margin - hud_w))
        box_y = max(rect.y() + margin, min(box_y, rect.y() + rect.height() - margin - hud_h))
        box_x, box_y = self._resolve_hud_overlay_collisions(
            box_x, box_y, hud_w, hud_h, arrow_w, arrow_h, dx, dy, gap, True, rect, margin
        )

        arrow_x, arrow_y = self._arrow_offset_for_box(
            box_x, box_y, hud_w, hud_h, arrow_w, arrow_h, dx, dy, gap
        )

        self._length_hud.move(box_x, box_y)
        self._length_hud_arrow.move(arrow_x, arrow_y)
        self._length_hud.setVisible(True)
        self._length_hud_arrow.setVisible(True)
        self._length_hud.raise_()
        self._length_hud_arrow.raise_()

    def _refresh_length_hud(self, _value=None) -> None:
        """Show length at the cursor while drawing, else above the last endpoint."""
        if self._view.is_drawing_line():
            return

        endpoint = self._view.last_endpoint_widget_pos()
        length_px = self._view.get_active_line_length_pixels()
        text = self._format_shark_length_text(length_px)
        if endpoint is None or text is None:
            self._length_hud.setVisible(False)
            self._length_hud_arrow.setVisible(False)
            return
        self._place_length_hud_at(endpoint, text)

    def _on_drawing_hud_changed(self, cursor_pos, length_px) -> None:
        """Follow the cursor while drawing; snap to the endpoint when finished."""
        if cursor_pos is None:
            self._refresh_length_hud()
            return

        text = self._format_shark_length_text(length_px)
        if text is None:
            self._length_hud.setVisible(False)
            self._length_hud_arrow.setVisible(False)
            return
        self._place_length_hud_at(QPointF(cursor_pos), text)

    def _build_result_dict(self) -> dict:
        line = self._view.get_line()
        polyline = self._view.get_polyline()
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

        return {
            "image_path": str(self._image_path) if self._image_path is not None else None,
            "line": (
                ((line[0].x(), line[0].y()), (line[1].x(), line[1].y()))
                if line is not None
                else None
            ),
            "polyline": (
                [(point.x(), point.y()) for point in polyline]
                if polyline is not None
                else None
            ),
            "length_pixels": length_px,
            "length_feet": length_ft,
            "drone": self._drone_select.currentText() or None,
            "altitude_m": self._get_altitude_m(),
        }

    def _confirm_changes(self) -> None:
        self.changes_confirmed.emit(self._build_result_dict())

    def _cancel_changes(self) -> None:
        self.changes_confirmed.emit({})

    def view(self) -> ZoomableFrameView:
        return self._view


class FrameLineEditorWindow(QMainWindow):
    """Standalone window wrapping FrameLineEditorWidget for manual testing."""

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
        self._editor = FrameLineEditorWidget(
            parent=self,
            settings_obj=settings_obj,
            drone_altitude=drone_altitude,
            initial_drone=initial_drone,
        )
        self.setCentralWidget(self._editor)
        self._editor.changes_confirmed.connect(self._on_editor_result)
        if not self._editor.load_image(image_path):
            self.setWindowTitle("Frame Line Editor - Failed to load image")
        else:
            self.setWindowTitle(f"Frame Line Editor - {Path(image_path).name}")
        self.resize(960, 720)

    def _on_editor_result(self, result) -> None:
        self.changes_confirmed.emit(result)
        self.close()

    def view(self) -> ZoomableFrameView:
        return self._editor.view()


def test_frame_line_editor() -> None:
    """Launch the frame line editor with a sample image for manual testing."""
    test_image = Path(
        r"C:\Users\legop\Desktop\GitHub\SharkEye_App\results\07142026_121851\frames"
        r"\TRIMMED_2023-04-23_Transect_DJI_0502.mp4_1.jpg"
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
