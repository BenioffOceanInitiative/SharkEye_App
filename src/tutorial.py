"""First-run tutorial: welcome popup + interactive guided walkthrough."""

from __future__ import annotations

import os
from typing import Callable

from PyQt6.QtCore import QObject, QPoint, QRect, Qt, QSettings, QTimer, QEvent
from PyQt6.QtGui import QImageReader, QKeyEvent, QCloseEvent, QPixmap, QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from theme import banner_surface_style
from utility import resource_path


TUTORIAL_SETTING_KEY = "tutorial_completed"
# Testing: show welcome + guided tour on every app launch.
FORCE_TUTORIAL_FOR_TESTING = False

TUTORIAL_WIDTH = 560
TUTORIAL_HEIGHT = 560
TUTORIAL_IMAGE_MAX_HEIGHT = TUTORIAL_HEIGHT // 2
TUTORIAL_PAGE_HORIZONTAL_MARGIN = 56
TUTORIAL_IMAGE_WIDTH = TUTORIAL_WIDTH - TUTORIAL_PAGE_HORIZONTAL_MARGIN
TUTORIAL_IMAGE_BOTTOM_MARGIN = 16
BANNER_HEIGHT = 60

EXAMPLE_FOOTAGE_FILENAME = "example_footage.mp4"
SAMPLE_RESULTS_DIRNAME = "sample_results"
GUIDED_DRONE = "Air 2S"
GUIDED_ALTITUDE = "40"
GUIDED_LOCATION = "Goleta"
DETECTION_TABLE_TOOLTIP_MARGIN = 20
# Visible historical_items columns (Experiment=0 and ID=2 are hidden in review).
DETECTION_COLUMN_TIPS: list[tuple[int, str]] = [
    (1, "Video: the source file for the detection."),
    (3, "Time: the timestamp in the video the detection was recorded."),
    (4, "Confidence: the model's confidence in the detection."),
    (5, "Length: estimated shark length."),
    (6, "Label: the type of object the model detected. You can correct the object type before confirming if needed."),
]

HIGHLIGHT_STYLE = (
    "QFrame#tutorialHighlight {"
    "  border: 2px solid #ff6020;"
    "  border-radius: 4px;"
    "  background: transparent;"
    "}"
)

# Secondary-axis alignment of the bubble along the host edge chosen by ``position``.
# top/bottom → left|center|right; left/right → top|center|bottom.
VALID_TOOLTIP_ALIGNS = frozenset({"center", "left", "right", "top", "bottom"})


def _normalize_align(align: str | None, position: str) -> str:
    """Return a valid align for ``position``, defaulting to center."""
    value = (align or "center").lower().strip()
    if value not in VALID_TOOLTIP_ALIGNS:
        return "center"
    if position == "center":
        return "center"
    if position in ("top", "bottom") and value in ("top", "bottom"):
        return "center"
    if position in ("left", "right") and value in ("left", "right"):
        return "center"
    return value


def _aligned_origin_offset(
    *,
    position: str,
    align: str,
    host_w: int,
    host_h: int,
    tip_w: int,
    tip_h: int,
) -> tuple[int, int]:
    """Return (dx, dy) from the host-edge origin to the bubble top-left."""
    align = _normalize_align(align, position)
    if position == "center":
        return host_w // 2 - tip_w // 2, host_h // 2 - tip_h // 2
    if position == "top":
        y = -tip_h
        if align == "left":
            return 0, y
        if align == "right":
            return host_w - tip_w, y
        return host_w // 2 - tip_w // 2, y
    if position == "bottom":
        y = 0
        if align == "left":
            return 0, y
        if align == "right":
            return host_w - tip_w, y
        return host_w // 2 - tip_w // 2, y
    if position == "left":
        x = -tip_w
        if align == "top":
            return x, 0
        if align == "bottom":
            return x, host_h - tip_h
        return x, host_h // 2 - tip_h // 2
    # right
    x = 0
    if align == "top":
        return x, 0
    if align == "bottom":
        return x, host_h - tip_h
    return x, host_h // 2 - tip_h // 2


def _edge_anchor_point(position: str, base: QPoint, host_w: int, host_h: int) -> QPoint:
    """Host-local point on the edge used by ``position`` (before align offset)."""
    if position == "center":
        return base + QPoint(0, 0)
    if position == "top":
        return base + QPoint(0, 0)
    if position == "bottom":
        return base + QPoint(0, host_h)
    if position == "left":
        return base + QPoint(0, 0)
    return base + QPoint(host_w, 0)


def _qt_is_valid(widget: QWidget | None) -> bool:
    if widget is None:
        return False
    try:
        from PyQt6 import sip

        return not sip.isdeleted(widget)
    except (ImportError, TypeError):
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False


def _make_logo_banner() -> QWidget:
    """Navy brand banner with the white SharkEye logo only (no buttons)."""
    banner = QWidget()
    banner.setStyleSheet(banner_surface_style())
    banner.setFixedHeight(BANNER_HEIGHT)

    layout = QHBoxLayout(banner)
    layout.setContentsMargins(20, 8, 20, 8)

    logo_label = QLabel()
    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    logo_label.setScaledContents(False)
    logo_label.setFixedHeight(40)

    pixmap = QPixmap(resource_path("assets/images/logo-white.png"))
    if not pixmap.isNull():
        dpr = logo_label.devicePixelRatioF()
        pixmap = pixmap.scaledToHeight(
            int(40 * dpr),
            Qt.TransformationMode.SmoothTransformation,
        )
        pixmap.setDevicePixelRatio(dpr)
        logo_label.setPixmap(pixmap)

    layout.addStretch(1)
    layout.addWidget(logo_label)
    layout.addStretch(1)
    return banner


def tutorial_completed(settings_obj: QSettings | None = None) -> bool:
    if FORCE_TUTORIAL_FOR_TESTING:
        return False
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    return str(settings_obj.value(TUTORIAL_SETTING_KEY, "false")).lower() == "true"


def mark_tutorial_completed(settings_obj: QSettings | None = None) -> None:
    if FORCE_TUTORIAL_FOR_TESTING:
        return
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    settings_obj.setValue(TUTORIAL_SETTING_KEY, "true")


def should_show_tutorial(settings_obj: QSettings | None = None) -> bool:
    """True when the welcome popup and guided tour should run."""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "minimal":
        return False
    if FORCE_TUTORIAL_FOR_TESTING:
        return True
    return not tutorial_completed(settings_obj)


def example_footage_path() -> str | None:
    """Sample video path under sample_data/ (dev) or bundled data/ (frozen)."""
    candidates = (
        os.path.join("sample_data", EXAMPLE_FOOTAGE_FILENAME),
        os.path.join("data", EXAMPLE_FOOTAGE_FILENAME),
    )
    for relative in candidates:
        path = resource_path(relative)
        if os.path.isfile(path):
            return path
    # Dev fallback: repo-root sample_data next to src/
    repo_sample = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "sample_data",
            EXAMPLE_FOOTAGE_FILENAME,
        )
    )
    return repo_sample if os.path.isfile(repo_sample) else None


def sample_results_path() -> str | None:
    """Bundled tutorial fallback experiment folder (frames/masks/CSV/clips)."""
    relative = os.path.join("sample_data", SAMPLE_RESULTS_DIRNAME)
    path = resource_path(relative)
    if os.path.isdir(path) and os.path.isdir(os.path.join(path, "detection_results")):
        return path
    repo_sample = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "sample_data",
            SAMPLE_RESULTS_DIRNAME,
        )
    )
    if os.path.isdir(repo_sample) and os.path.isdir(
        os.path.join(repo_sample, "detection_results")
    ):
        return repo_sample
    return None


def count_tracks_in_experiment_dir(experiment_dir: str) -> int:
    """Count detection rows across CSVs in ``detection_results/``."""
    det_dir = os.path.join(experiment_dir, "detection_results")
    if not os.path.isdir(det_dir):
        return 0
    total = 0
    try:
        import pandas as pd
    except ImportError:
        return 0
    for name in os.listdir(det_dir):
        if not name.lower().endswith(".csv"):
            continue
        csv_path = os.path.join(det_dir, name)
        try:
            df = pd.read_csv(csv_path)
            total += len(df)
        except Exception:
            continue
    return total


def _has_example_footage(window) -> bool:
    for row in range(window.video_list.rowCount()):
        item = window.video_list.item(row, 0)
        if not item:
            continue
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.basename(path) == EXAMPLE_FOOTAGE_FILENAME:
            return True
    return False


def _try_add_example_footage(window) -> bool:
    """Add sample footage to the video list. Returns True if present after the call."""
    if _has_example_footage(window):
        return True
    bundled = example_footage_path()
    if not bundled:
        return False
    window.add_video_paths([bundled])
    return _has_example_footage(window)


def _load_pixmap(image_path: str) -> QPixmap:
    reader = QImageReader(resource_path(image_path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return QPixmap()
    return QPixmap.fromImage(image)


def _pixmap_fit(source: QPixmap, width: int, height: int, dpr: float = 1.0) -> QPixmap:
    if source.isNull() or width <= 0 or height <= 0:
        return QPixmap()

    dpr = dpr if dpr > 0 else 1.0
    phys_w = max(1, round(width * dpr))
    phys_h = max(1, round(height * dpr))

    image = source.toImage()
    fitted = image.scaled(
        phys_w,
        phys_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    result = QPixmap.fromImage(fitted)
    result.setDevicePixelRatio(dpr)
    return result


class TutorialPage(QWidget):
    """Base class for one tutorial slide."""

    title: str = ""
    body: str = ""
    image_path: str | None = None

    def __init__(self, image_max_height: int = TUTORIAL_IMAGE_MAX_HEIGHT, parent=None):
        super().__init__(parent)
        self._image_max_height = image_max_height
        self._source_pixmap: QPixmap | None = None
        self._fitted_pixmap_key: tuple[int, int, float] | None = None

        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 8)
        layout.setSpacing(12)

        self.title_label = QLabel(self.title)
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(left)
        title_font = self.title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        content = QVBoxLayout()
        content.setSpacing(12)

        self.body_label = QLabel(self.body)
        self.body_label.setWordWrap(True)
        self.body_label.setAlignment(left)
        body_font = self.body_label.font()
        body_font.setPointSize(10)
        self.body_label.setFont(body_font)

        text_scroll = QScrollArea()
        text_scroll.setWidgetResizable(True)
        text_scroll.setFrameShape(QFrame.Shape.NoFrame)
        text_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_scroll.setWidget(self.body_label)
        content.addWidget(text_scroll, stretch=1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(TUTORIAL_IMAGE_WIDTH, image_max_height)
        content.addWidget(self.image_label, stretch=0)

        if self.image_path:
            pixmap = _load_pixmap(self.image_path)
            if not pixmap.isNull():
                self._source_pixmap = pixmap
                self.image_label.show()
                content.setContentsMargins(0, 0, 0, TUTORIAL_IMAGE_BOTTOM_MARGIN)
        else:
            self.image_label.hide()
            self.image_label.setMaximumHeight(0)

        layout.addLayout(content, stretch=1)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._update_image()

    def _update_image(self) -> None:
        if self._source_pixmap is None or self.image_label.isHidden():
            return
        width = TUTORIAL_IMAGE_WIDTH
        height = self._image_max_height
        dpr = self.image_label.devicePixelRatioF() or 1.0
        cache_key = (width, height, dpr)
        if cache_key == self._fitted_pixmap_key:
            return
        fitted = _pixmap_fit(self._source_pixmap, width, height, dpr)
        self.image_label.setPixmap(fitted)
        self._fitted_pixmap_key = cache_key


class WelcomePage(TutorialPage):
    title = "Welcome to SharkEye"
    body = (
        "SharkEye lets you analyse your own drone footage to detect and estimate the length "
        "of sharks and other objects. We'll walk you through your first experiment next."
    )
    image_path = ""


class WelcomeDialog(QDialog):
    """Single-screen welcome popup shown before the guided tour."""

    def __init__(self, settings_obj: QSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
        self._finished = False

        self.setWindowTitle("Welcome")
        self.setModal(True)
        self.setFixedSize(TUTORIAL_WIDTH, TUTORIAL_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setSizeGripEnabled(False)
        flags = (
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        if os.name == "nt":
            flags |= Qt.WindowType.MSWindowsFixedSizeDialogHint
        self.setWindowFlags(flags)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_make_logo_banner())
        root.addWidget(WelcomePage(), stretch=1)

        footer = QHBoxLayout()
        footer.setContentsMargins(28, 12, 28, 20)
        footer.addStretch(1)
        get_started = QPushButton("Get Started")
        get_started.setDefault(True)
        get_started.clicked.connect(self._finish)
        footer.addWidget(get_started)
        root.addLayout(footer)
        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            app = QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _finish(self) -> None:
        self._finished = True
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._finished:
            event.accept()
            return
        event.ignore()


class TutorialCompletePage(TutorialPage):
    title = "Tutorial Complete"
    body = (
        "You're all set! You can revisit this walkthrough any time from the Help menu."
    )
    image_path = ""


class TutorialCompleteDialog(QDialog):
    """Shown when the user finishes the guided walkthrough."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tutorial Complete")
        self.setModal(True)
        self.setFixedSize(TUTORIAL_WIDTH, TUTORIAL_HEIGHT)
        flags = (
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        if os.name == "nt":
            flags |= Qt.WindowType.MSWindowsFixedSizeDialogHint
        self.setWindowFlags(flags)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_make_logo_banner())
        root.addWidget(TutorialCompletePage(), stretch=1)

        footer = QHBoxLayout()
        footer.setContentsMargins(28, 12, 28, 20)
        footer.addStretch(1)
        done = QPushButton("Done")
        done.setDefault(True)
        done.clicked.connect(self.accept)
        footer.addWidget(done)
        root.addLayout(footer)


def show_tutorial_complete_dialog(parent=None) -> None:
    TutorialCompleteDialog(parent=parent).exec()


# ---------------------------------------------------------------------------
# Tutorial tooltip overlay + per-widget guides
# ---------------------------------------------------------------------------


def _widget_on_screen(widget: QWidget, window: QWidget) -> bool:
    if not _qt_is_valid(widget) or not _qt_is_valid(window):
        return False
    if not widget.isVisible():
        return False
    if widget.isVisibleTo(window):
        return True
    top_left = widget.mapToGlobal(QPoint(0, 0))
    size = widget.size()
    if size.width() <= 0 or size.height() <= 0:
        size = widget.sizeHint()
    host_rect = QRect(top_left, size)
    win_top_left = window.mapToGlobal(QPoint(0, 0))
    win_rect = QRect(win_top_left, window.size())
    return win_rect.intersects(host_rect)


def _surface_for_window(window) -> QWidget | None:
    """Client area for tooltips: centralWidget when present, else the window itself."""
    if window is None or not _qt_is_valid(window):
        return None
    if hasattr(window, "centralWidget"):
        surface = window.centralWidget()
        if surface is not None and _qt_is_valid(surface):
            return surface
    return window


def tutorial_surface_for(host: QWidget) -> QWidget | None:
    return _surface_for_window(host.window())


class _TutorialOverlaySync(QObject):
    def __init__(self, surface: QWidget, overlay: QWidget):
        super().__init__(surface)
        self._surface = surface
        self._overlay = overlay
        surface.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self._overlay.setGeometry(self._surface.rect())
            self._overlay.raise_()
        return False


def _tutorial_overlay_for(window, surface: QWidget) -> QWidget:
    overlay = getattr(window, "_tutorial_overlay", None)
    if overlay is None or not _qt_is_valid(overlay):
        overlay = QWidget(surface)
        overlay.setObjectName("tutorialOverlay")
        overlay.setGeometry(surface.rect())
        overlay.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        overlay.setEnabled(True)
        overlay.hide()
        window._tutorial_overlay = overlay
        sync = getattr(window, "_tutorial_overlay_sync", None)
        if sync is None or not _qt_is_valid(sync):
            window._tutorial_overlay_sync = _TutorialOverlaySync(surface, overlay)
    return overlay


def _sync_tutorial_overlay(window) -> None:
    surface = _surface_for_window(window)
    overlay = getattr(window, "_tutorial_overlay", None)
    if surface is not None and overlay is not None and _qt_is_valid(overlay):
        overlay.setGeometry(surface.rect())
        overlay.setEnabled(True)
        overlay.raise_()


def _show_tutorial_overlay(window) -> None:
    """Always show/raise the overlay while a bubble should be visible."""
    overlay = getattr(window, "_tutorial_overlay", None)
    if overlay is None or not _qt_is_valid(overlay):
        return
    _sync_tutorial_overlay(window)
    overlay.setEnabled(True)
    overlay.show()
    overlay.raise_()


def _set_tutorial_overlay_blocking(window, *, blocking: bool) -> None:
    """Show overlay and set whether it blocks clicks (Next steps) or is click-through."""
    overlay = getattr(window, "_tutorial_overlay", None)
    if overlay is None or not _qt_is_valid(overlay):
        return
    overlay.setEnabled(True)
    overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not blocking)
    _show_tutorial_overlay(window)


def _ensure_tooltip_parent_layers_enabled(window) -> None:
    surface = _surface_for_window(window)
    if surface is not None and _qt_is_valid(surface):
        surface.setEnabled(True)
    overlay = getattr(window, "_tutorial_overlay", None)
    if overlay is not None and _qt_is_valid(overlay):
        overlay.setEnabled(True)


def _maybe_hide_tutorial_overlay(window) -> None:
    """Hide overlay only when no bubble is painting on this window."""
    # Hosts may be registered on MainWindow while the bubble lives on a dialog.
    registries = [window]
    parent = window.parentWidget() if hasattr(window, "parentWidget") else None
    while parent is not None:
        registries.append(parent)
        parent = parent.parentWidget()
    seen: set[int] = set()
    for registry in registries:
        if not _qt_is_valid(registry):
            continue
        for host in getattr(registry, "_tutorial_tooltip_hosts", []):
            if not _qt_is_valid(host) or id(host) in seen:
                continue
            seen.add(id(host))
            if host.window() is not window:
                continue
            tooltip = getattr(host, "tutorial_tooltip", None)
            if tooltip is not None and tooltip.is_visible():
                return
    overlay = getattr(window, "_tutorial_overlay", None)
    if overlay is not None and _qt_is_valid(overlay):
        overlay.hide()


class TutorialTooltipBubble(QFrame):
    """Floating message bubble parented to the tutorial overlay."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("tutorialTooltipBubble")
        self.setStyleSheet(
            "#tutorialTooltipBubble {"
            "  background-color: palette(window);"
            "  border: 1px solid palette(mid);"
            "  border-radius: 6px;"
            "}"
        )
        self._on_next: Callable[[], None] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        layout.addWidget(self._message_label)

        self._next_btn = QPushButton("Next")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._next_btn)
        layout.addLayout(row)
        self._next_btn.clicked.connect(self._emit_next)
        self.setMaximumWidth(360)

    def _emit_next(self) -> None:
        if self._on_next is not None:
            self._on_next()

    def _set_click_through(self, enabled: bool) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        self._message_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled
        )

    def prepare_bubble(
        self,
        message: str,
        *,
        press_next: bool,
        on_next: Callable[[], None] | None,
    ) -> None:
        self._message_label.setText(message)
        self._next_btn.setVisible(press_next)
        self._on_next = on_next if press_next else None
        if press_next:
            self.ensure_interactive()
            self._set_click_through(False)
        else:
            self._next_btn.setEnabled(False)
            self._set_click_through(True)
        self.adjustSize()

    def ensure_interactive(self) -> None:
        self.setEnabled(True)
        self._message_label.setEnabled(True)
        if self._next_btn.isVisible():
            self._next_btn.setEnabled(True)
            self._next_btn.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
            )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        parent = self.parentWidget()
        if parent is not None:
            parent.setEnabled(True)
        self.raise_()


class TutorialTooltip(QObject):
    """Per-widget tutorial guide: highlight border + tracking tooltip."""

    def __init__(
        self,
        host: QWidget,
        message: str,
        *,
        press_next: bool = False,
        enable_parent: bool = True,
        position: str = "bottom",
        align: str = "center",
        highlight: bool = True,
        margin: int = 8,
    ):
        super().__init__(host)
        self.host = host
        self._default_message = message
        self.press_next = press_next
        self.enable_parent = enable_parent
        self.position = position
        self.align = _normalize_align(align, position)
        self.highlight = highlight
        self._margin = margin
        self._highlighted = False
        self._highlight_frame: QFrame | None = None
        self._surface: QWidget | None = None
        self._overlay: QWidget | None = None
        self._bubble: TutorialTooltipBubble | None = None
        self._anchor_rect: QRect | None = None
        self._anchor_provider: Callable[[], QRect | None] | None = None
        self._table_tracking_installed = False
        self._pending_show: (
            tuple[
                str | None,
                bool,
                Callable[[], None] | None,
                QRect | None,
                Callable[[], QRect | None] | None,
                str | None,
                bool | None,
            ]
            | None
        ) = None
        host.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if not self.is_visible():
            return False
        if event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        ):
            tracked = {self.host, self._surface, self._overlay}
            if isinstance(self.host, QTableWidget):
                tracked.add(self.host.viewport())
                tracked.add(self.host.horizontalHeader())
            if obj in tracked:
                QTimer.singleShot(0, self._reposition)
        return False

    def is_visible(self) -> bool:
        return (
            self._bubble is not None
            and _qt_is_valid(self._bubble)
            and self._bubble.isVisible()
        )

    def show(
        self,
        *,
        message: str | None = None,
        press_next: bool | None = None,
        on_next: Callable[[], None] | None = None,
        anchor_rect: QRect | None = None,
        anchor_provider: Callable[[], QRect | None] | None = None,
        margin: int | None = None,
        align: str | None = None,
        highlight: bool | None = None,
    ) -> None:
        self._pending_show = None
        if margin is not None:
            self._margin = margin
        if align is not None:
            self.align = _normalize_align(align, self.position)
        if highlight is not None:
            self.highlight = highlight
        self._anchor_provider = anchor_provider
        self._anchor_rect = anchor_rect if anchor_provider is None else None
        text = message if message is not None else self._default_message
        show_next = self.press_next if press_next is None else press_next
        if anchor_provider is not None:
            self._install_table_tracking()
        if not self._ensure_bubble():
            self._pending_show = (
                text,
                show_next,
                on_next,
                anchor_rect,
                anchor_provider,
                align,
                highlight,
            )
            QTimer.singleShot(100, self._retry_pending_show)
            return
        window = self.host.window()
        if window is not None:
            _ensure_tooltip_parent_layers_enabled(window)
        self.set_highlight(self.highlight)
        self._bubble.prepare_bubble(text, press_next=show_next, on_next=on_next)
        if not self._reposition():
            if window is not None:
                # Keep overlay click-through while retrying; do not leave a blocker.
                _set_tutorial_overlay_blocking(window, blocking=False)
                _maybe_hide_tutorial_overlay(window)
            self._pending_show = (
                text,
                show_next,
                on_next,
                anchor_rect,
                anchor_provider,
                align,
                highlight,
            )
            QTimer.singleShot(100, self._retry_pending_show)
            return
        registry_window = getattr(self.host, "_tutorial_registry_window", None)
        if registry_window is not None and _qt_is_valid(registry_window):
            hide_other_tutorial_tooltips(registry_window, except_host=self.host)
        # Always show the overlay (bubble is its child). Blocking only for Next steps.
        if window is not None:
            _set_tutorial_overlay_blocking(window, blocking=show_next)
        self._bubble.show()
        self._bubble.raise_()
        if show_next:
            self._bubble.ensure_interactive()
        elif window is not None:
            # Click-through steps: bubble must not steal clicks from the host.
            self._bubble._set_click_through(True)
        if window is not None:
            _show_tutorial_overlay(window)
            _ensure_tooltip_parent_layers_enabled(window)

    def ensure_bubble_enabled(self) -> None:
        if self._bubble is not None and _qt_is_valid(self._bubble):
            self._bubble.ensure_interactive()

    def hide(self, *, skip_overlay_hide: bool = False) -> None:
        self._pending_show = None
        self._anchor_rect = None
        self._anchor_provider = None
        self._clear_table_tracking()
        self.set_highlight(False)
        if self._bubble is not None:
            if _qt_is_valid(self._bubble):
                self._bubble.hide()
            self._bubble = None
        self._overlay = None
        if skip_overlay_hide:
            return
        window = self.host.window() if _qt_is_valid(self.host) else None
        if window is not None:
            _maybe_hide_tutorial_overlay(window)

    def set_highlight(self, enabled: bool) -> None:
        """Draw an orange ring on the overlay (host or anchor), never greying icons."""
        if not enabled:
            self._clear_highlight_frame()
            self._highlighted = False
            return
        if not _qt_is_valid(self.host):
            self._clear_highlight_frame()
            self._highlighted = False
            return
        if self._overlay is None or not _qt_is_valid(self._overlay):
            self._highlighted = False
            return
        self._ensure_highlight_frame()
        self._sync_highlight_geometry()
        self._highlighted = True

    def _ensure_highlight_frame(self) -> None:
        if self._overlay is None or not _qt_is_valid(self._overlay):
            return
        if self._highlight_frame is None or not _qt_is_valid(self._highlight_frame):
            frame = QFrame(self._overlay)
            frame.setObjectName("tutorialHighlight")
            frame.setStyleSheet(HIGHLIGHT_STYLE)
            frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            frame.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
            self._highlight_frame = frame
        elif self._highlight_frame.parent() is not self._overlay:
            self._highlight_frame.setParent(self._overlay)

    def _clear_highlight_frame(self) -> None:
        if self._highlight_frame is not None and _qt_is_valid(self._highlight_frame):
            self._highlight_frame.hide()
            self._highlight_frame.deleteLater()
        self._highlight_frame = None

    def _highlight_local_rect(self) -> QRect | None:
        if self._anchor_provider is not None:
            return self._anchor_provider()
        if self._anchor_rect is not None:
            return QRect(self._anchor_rect)
        if not _qt_is_valid(self.host):
            return None
        return QRect(0, 0, max(self.host.width(), 1), max(self.host.height(), 1))

    def _sync_highlight_geometry(self) -> None:
        if self._highlight_frame is None or not _qt_is_valid(self._highlight_frame):
            return
        local = self._highlight_local_rect()
        if local is None or local.isNull():
            self._highlight_frame.hide()
            return
        top_left = self._host_point_on_surface(local.topLeft())
        geo = QRect(top_left, local.size())
        # Slightly pad so the ring sits outside the control edge.
        self._highlight_frame.setGeometry(geo.adjusted(-3, -3, 3, 3))
        self._highlight_frame.show()
        self._highlight_frame.raise_()
        if self._bubble is not None and _qt_is_valid(self._bubble):
            self._bubble.raise_()

    def _retry_pending_show(self) -> None:
        pending = self._pending_show
        if pending is None:
            return
        text, show_next, on_next, anchor_rect, anchor_provider, align, highlight = pending
        self.show(
            message=text,
            press_next=show_next,
            on_next=on_next,
            anchor_rect=anchor_rect,
            anchor_provider=anchor_provider,
            align=align,
            highlight=highlight,
        )

    def _install_table_tracking(self) -> None:
        if self._table_tracking_installed or not isinstance(self.host, QTableWidget):
            return
        self._table_tracking_installed = True
        viewport = self.host.viewport()
        header = self.host.horizontalHeader()
        if viewport is not None:
            viewport.installEventFilter(self)
        if header is not None:
            header.installEventFilter(self)
        for bar in (self.host.horizontalScrollBar(), self.host.verticalScrollBar()):
            if bar is not None:
                bar.valueChanged.connect(self._schedule_reposition)

    def _clear_table_tracking(self) -> None:
        if not self._table_tracking_installed or not isinstance(self.host, QTableWidget):
            self._table_tracking_installed = False
            return
        viewport = self.host.viewport()
        header = self.host.horizontalHeader()
        if viewport is not None:
            viewport.removeEventFilter(self)
        if header is not None:
            header.removeEventFilter(self)
        for bar in (self.host.horizontalScrollBar(), self.host.verticalScrollBar()):
            if bar is not None:
                try:
                    bar.valueChanged.disconnect(self._schedule_reposition)
                except TypeError:
                    pass
        self._table_tracking_installed = False

    def _schedule_reposition(self, *_args) -> None:
        if self.is_visible():
            QTimer.singleShot(0, self._reposition)

    def _ensure_bubble(self) -> bool:
        surface = tutorial_surface_for(self.host)
        window = self.host.window()
        if surface is None or window is None:
            return False
        overlay = _tutorial_overlay_for(window, surface)
        _sync_tutorial_overlay(window)
        self._surface = surface
        self._overlay = overlay
        if self._bubble is None or not _qt_is_valid(self._bubble):
            self._bubble = TutorialTooltipBubble(overlay)
            self._bubble.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        elif self._bubble.parent() is not overlay:
            self._bubble.setParent(overlay)
        return True

    def _tooltip_parent(self) -> QWidget | None:
        if self._bubble is not None and _qt_is_valid(self._bubble):
            return self._bubble.parentWidget()
        return self._overlay

    def _host_point_on_surface(self, local_point: QPoint) -> QPoint:
        parent = self._tooltip_parent() or self._surface
        if parent is None:
            return QPoint(0, 0)
        return parent.mapFromGlobal(self.host.mapToGlobal(local_point))

    def _reposition(self) -> bool:
        if self._bubble is None or self._surface is None:
            return False
        if not _qt_is_valid(self._bubble) or not _qt_is_valid(self.host):
            self._bubble = None
            return False
        window = self.host.window()
        if window is None or not _widget_on_screen(self.host, window):
            return False

        self._bubble.adjustSize()
        tip_w = self._bubble.width()
        tip_h = self._bubble.height()
        anchor = None
        if self._anchor_provider is not None:
            anchor = self._anchor_provider()
            if anchor is None:
                return False
        elif self._anchor_rect is not None:
            anchor = self._anchor_rect
        if anchor is not None:
            host_w = anchor.width()
            host_h = anchor.height()
            base = anchor.topLeft()
        else:
            host_w = self.host.width()
            host_h = self.host.height()
            base = QPoint(0, 0)
        surface = self._tooltip_parent() or self._surface
        if surface is None:
            return False

        pos = self.position
        align = self.align

        if pos == "center":
            # Place the bubble dead-center over the host/anchor (both axes).
            center_local = base + QPoint(host_w // 2, host_h // 2)
            center = self._host_point_on_surface(center_local)
            x = center.x() - tip_w // 2
            y = center.y() - tip_h // 2
            x = max(8, min(x, surface.width() - tip_w - 8))
            y = max(8, min(y, surface.height() - tip_h - 8))
            self._bubble.move(x, y)
            self._bubble.raise_()
            if self._next_btn_active():
                self._bubble.ensure_interactive()
            if self._highlighted:
                self._sync_highlight_geometry()
            return True

        edge = _edge_anchor_point(pos, base, host_w, host_h)
        dx, dy = _aligned_origin_offset(
            position=pos,
            align=align,
            host_w=host_w,
            host_h=host_h,
            tip_w=tip_w,
            tip_h=tip_h,
        )
        # Margin pushes the bubble away from the host along the primary axis.
        if pos == "top":
            dy -= self._margin
        elif pos == "bottom":
            dy += self._margin
        elif pos == "left":
            dx -= self._margin
        elif pos == "right":
            dx += self._margin

        origin = self._host_point_on_surface(edge)
        x = origin.x() + dx
        y = origin.y() + dy

        x = max(8, min(x, surface.width() - tip_w - 8))
        y = max(8, min(y, surface.height() - tip_h - 8))
        bubble_rect = QRect(x, y, tip_w, tip_h)
        host_rect = QRect(
            self._host_point_on_surface(base),
            self._host_point_on_surface(base + QPoint(host_w, host_h)),
        ).normalized()
        if bubble_rect.intersects(host_rect) and pos == "right":
            # Fall back below the host when a right-side tip overlaps it.
            edge = _edge_anchor_point("bottom", base, host_w, host_h)
            dx, dy = _aligned_origin_offset(
                position="bottom",
                align="center",
                host_w=host_w,
                host_h=host_h,
                tip_w=tip_w,
                tip_h=tip_h,
            )
            dy += self._margin
            origin = self._host_point_on_surface(edge)
            x = origin.x() + dx
            y = origin.y() + dy
            x = max(8, min(x, surface.width() - tip_w - 8))
            y = max(8, min(y, surface.height() - tip_h - 8))
        self._bubble.move(x, y)
        self._bubble.raise_()
        if self._next_btn_active():
            self._bubble.ensure_interactive()
        if self._highlighted:
            self._sync_highlight_geometry()
        return True

    def _next_btn_active(self) -> bool:
        return (
            self._bubble is not None
            and _qt_is_valid(self._bubble)
            and self._bubble._next_btn.isVisible()
        )


def attach_tutorial_tooltip(
    host: QWidget,
    message: str,
    *,
    press_next: bool = False,
    enable_parent: bool = True,
    position: str = "bottom",
    align: str = "center",
    highlight: bool = True,
    margin: int = 8,
) -> TutorialTooltip:
    tooltip = TutorialTooltip(
        host,
        message,
        press_next=press_next,
        enable_parent=enable_parent,
        position=position,
        align=align,
        highlight=highlight,
        margin=margin,
    )
    host.tutorial_tooltip = tooltip
    return tooltip


def register_tutorial_tooltip_host(window, host: QWidget) -> None:
    host._tutorial_registry_window = window
    hosts = getattr(window, "_tutorial_tooltip_hosts", None)
    if hosts is None:
        window._tutorial_tooltip_hosts = [host]
    elif host not in hosts:
        hosts.append(host)


def hide_other_tutorial_tooltips(window, *, except_host: QWidget | None = None) -> None:
    except_id = id(except_host) if except_host is not None else None
    for host in getattr(window, "_tutorial_tooltip_hosts", []):
        if not _qt_is_valid(host):
            continue
        if except_id is not None and id(host) == except_id:
            continue
        tooltip = getattr(host, "tutorial_tooltip", None)
        if tooltip is not None:
            # Skip overlay hide during transition; caller re-shows after bubble.show().
            tooltip.hide(skip_overlay_hide=True)


def hide_all_tutorial_tooltips(window) -> None:
    for host in getattr(window, "_tutorial_tooltip_hosts", []):
        if not _qt_is_valid(host):
            continue
        tooltip = getattr(host, "tutorial_tooltip", None)
        if tooltip is not None:
            tooltip.hide()


def setup_tutorial_tooltips(window) -> None:
    """Attach tutorial tooltips to widgets during UI setup (hidden until triggered)."""
    hosts: list[QWidget] = []

    def attach(
        widget: QWidget,
        message: str,
        *,
        press_next: bool = False,
        enable_parent: bool = True,
        position: str = "bottom",
        align: str = "center",
        highlight: bool = True,
        margin: int = 8,
    ) -> QWidget:
        attach_tutorial_tooltip(
            widget,
            message,
            press_next=press_next,
            enable_parent=enable_parent,
            position=position,
            align=align,
            highlight=highlight,
            margin=margin,
        )
        widget._tutorial_registry_window = window
        hosts.append(widget)
        return widget

    attach(
        window.banner_left_button,
        "Revisit saved experiments and review past detection results using this button.",
        press_next=True,
        enable_parent=False,
        position="bottom",
        align="left",
    )
    attach(
        window.banner_right_button,
        "Open Settings to configure drones, detection settings, cloud features, and other options.",
        press_next=True,
        enable_parent=False,
        position="bottom",
        align="right",
    )
    attach(
        window.banner_help_button,
        "The help menu provides an in-depth walkthrough of SharkEye features.",
        press_next=True,
        enable_parent=False,
        position="bottom",
        align="right",
    )
    attach(
        window.banner_report_button,
        "Report a problem or send feedback to the SharkEye team.",
        press_next=True,
        enable_parent=False,
        position="bottom",
        align="right",
    )
    attach(
        window.select_videos_button,
        "Click Select Video(s) to select videos for procesing. We'll load an example video for this tutorial.",
        press_next=False,
        enable_parent=True,
        position="top",
        align="center",
        margin=32,
    )
    attach(
        window.drone_select,
        f"Confirm the drone model. We've selected {GUIDED_DRONE} for this tutorial.",
        press_next=True,
        enable_parent=False,
        position="top",
        align="center",
    )
    attach(
        window.altitude_input,
        f"Confirm altitude ({GUIDED_ALTITUDE} m) and flight location ({GUIDED_LOCATION}).",
        press_next=True,
        enable_parent=False,
        position="top",
    )
    attach(
        window.process_button,
        "Press Process Videos to run detection on the sample video.",
        press_next=False,
        enable_parent=True,
        position="bottom",
    )
    attach(
        window.frame_player,
        "This is the review screen. Here you can inspect and refine the detection results before saving your experiment.",
        press_next=True,
        enable_parent=False,
        highlight=False,
        position="center",
    )
    attach(
        window.toggle_display_switch,
        "Click the mask toggle to preview the segmentation mask on this frame.",
        press_next=False,
        enable_parent=True,
        position="top",
        highlight=True,
    )
    attach(
        window.edit_frame_button,
        "This button opens an editor allowing you to manually draw the line segment to be used for length estimation.",
        press_next=True,
        enable_parent=False,
        position="top",
        align="left",
    )
    attach(
        window.overlay_settings_button,
        "This menu allows you to configure the overlay settings, such as whether to show or hide bounding boxes or confidence values, and the box color and thickness.",
        press_next=True,
        enable_parent=False,
        position="top",
    )
    attach(
        window.historical_items,
        "Each row is one detection from your experiment.",
        press_next=True,
        enable_parent=False,
        position="top",
    )
    attach(
        window.confirm_detections_button,
        "When you're satisfied with the labels, press Confirm Detections to finish.",
        press_next=False,
        enable_parent=True,
        position="top",
    )
    window._tutorial_tooltip_hosts = hosts


def _table_column_top_rect(table, col: int) -> QRect | None:
    if col < 0 or col >= table.columnCount() or table.isColumnHidden(col):
        return None
    header = table.horizontalHeader()
    if header is None:
        return None
    x_in_viewport = header.sectionViewportPosition(col)
    w = max(header.sectionSize(col), table.columnWidth(col), 1)
    if w <= 0:
        return None
    # If the column is scrolled off-screen, still produce a usable header anchor.
    if x_in_viewport + w <= 0:
        x_in_viewport = 0
    # Map the header section into table coordinates so only that header is highlighted.
    top_left = header.mapTo(table, QPoint(max(x_in_viewport, 0), 0))
    anchor_h = max(header.height() if header.isVisible() else 0, 1)
    return QRect(top_left.x(), top_left.y(), w, anchor_h)


def _processing_status_anchor(window) -> QRect | None:
    """Union of status label, timer, and progress bar (in status-label coords)."""
    status = getattr(window, "progress_status_label", None)
    timer = getattr(window, "timer_label", None)
    bar = getattr(window, "progress_bar", None)
    parts = [w for w in (status, timer, bar) if w is not None and _qt_is_valid(w) and w.isVisible()]
    if not parts or status is None or not _qt_is_valid(status):
        return None
    united: QRect | None = None
    for widget in parts:
        top_left = status.mapFromGlobal(widget.mapToGlobal(QPoint(0, 0)))
        part = QRect(top_left, widget.size())
        united = part if united is None else united.united(part)
    return united


def _flight_fields_anchor_rect(window) -> QRect | None:
    altitude = window.altitude_input
    location = window.flight_location_input
    if not _qt_is_valid(altitude) or not _qt_is_valid(location):
        return None
    br = altitude.mapFromGlobal(location.mapToGlobal(QPoint(location.width(), location.height())))
    return QRect(QPoint(0, 0), br).normalized()


def _frame_center_anchor(frame_player) -> QRect | None:
    """Return the displayed video rect (or full widget) for centering a tooltip."""
    rect = frame_player.content_rect()
    if rect.isNull():
        # Fallback when no clip is loaded yet so the frame intro still shows.
        rect = frame_player.rect()
    if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
        return None
    return rect


# ---------------------------------------------------------------------------
# Guided tour controller
# ---------------------------------------------------------------------------

# Input events swallowed for widgets locked by the tour (widgets stay enabled so
# they keep the native look — no greying / stylesheet fallback).
_LOCKED_INPUT_EVENTS = frozenset(
    {
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.Wheel,
        QEvent.Type.KeyPress,
        QEvent.Type.KeyRelease,
        QEvent.Type.ShortcutOverride,
        QEvent.Type.ContextMenu,
        QEvent.Type.TouchBegin,
        QEvent.Type.TouchUpdate,
        QEvent.Type.TouchEnd,
    }
)


class _TutorialInputGuard(QObject):
    """App-level filter: block interaction with tour-locked widgets."""

    def __init__(self, tour: "GuidedTour"):
        super().__init__(tour)
        self._tour = tour

    def eventFilter(self, obj, event) -> bool:
        return self._tour._filter_locked_input(obj, event)


class GuidedTour(QObject):
    """Full walkthrough: home → processing → review."""

    _STEP_REVIEW_PAST = 0
    _STEP_SETTINGS = 1
    _STEP_HELP = 2
    _STEP_REPORT = 3
    _STEP_SELECT_VIDEOS = 4
    _STEP_DRONE = 5
    _STEP_FLIGHT = 6
    _STEP_PROCESS = 7
    _STEP_PROCESSING = 8
    _STEP_FRAME = 9
    _STEP_MASK_ON = 10
    _STEP_EDIT_FRAME = 11
    _STEP_MASK_OFF = 12
    _STEP_GEAR = 13
    _STEP_DETECTION_COL_BASE = 14
    _STEP_CONFIRM = _STEP_DETECTION_COL_BASE + len(DETECTION_COLUMN_TIPS)
    _STEP_HOME = _STEP_CONFIRM + 1

    def __init__(self, main_window):
        super().__init__(main_window)
        self.window = main_window
        self._step_index = 0
        self._allowed_ids: set[int] = set()
        self._input_lock_active = False
        self._input_guard: _TutorialInputGuard | None = None
        self._select_hooked = False
        self._process_hooked = False
        self._mask_hooked = False
        self._mask_saw_on = False
        self._confirm_hooked = False
        self._home_hooked = False
        self._processing_dialog_acknowledged = False
        self._review_ui_ready = False

    def start(self) -> None:
        self.window.stack_widget.setCurrentWidget(self.window.home_widget)
        self._install_input_guard()
        try:
            self.window.stack_widget.currentChanged.disconnect(self._on_stack_changed)
        except TypeError:
            pass
        self.window.stack_widget.currentChanged.connect(self._on_stack_changed)
        QTimer.singleShot(150, self._show_current_step)

    def _install_input_guard(self) -> None:
        if self._input_guard is not None:
            return
        app = QApplication.instance()
        if app is None:
            return
        self._input_guard = _TutorialInputGuard(self)
        self._input_lock_active = True
        self._allowed_ids = set()
        app.installEventFilter(self._input_guard)

    def _remove_input_guard(self) -> None:
        self._input_lock_active = False
        self._allowed_ids = set()
        app = QApplication.instance()
        if app is not None and self._input_guard is not None:
            app.removeEventFilter(self._input_guard)
        self._input_guard = None

    def _is_tutorial_chrome(self, widget: QWidget) -> bool:
        """Tooltip bubbles / overlay must always receive input."""
        current: QWidget | None = widget
        while current is not None:
            if isinstance(current, TutorialTooltipBubble):
                return True
            if current.objectName() in ("tutorialOverlay", "tutorialHighlight"):
                return True
            current = current.parentWidget()
        return False

    def _is_cancel_processing_button(self, widget: QWidget) -> bool:
        cancel = getattr(self.window, "cancel_processsing_button", None)
        if cancel is None or not _qt_is_valid(cancel):
            return False
        current: QWidget | None = widget
        while current is not None:
            if current is cancel:
                return True
            if current.isWindow():
                break
            current = current.parentWidget()
        return False

    def _is_interaction_allowed(self, widget: QWidget) -> bool:
        if not self._input_lock_active:
            return True
        # Never allow Cancel Processing during the tour.
        if self._is_cancel_processing_button(widget):
            return False
        current: QWidget | None = widget
        while current is not None:
            if id(current) in self._allowed_ids:
                return True
            # Processing Complete (and similar) must stay clickable during the tour.
            if isinstance(current, QMessageBox):
                return True
            if current is self.window:
                break
            if current.isWindow() and current is not self.window:
                break
            current = current.parentWidget()
        return False

    def _filter_locked_input(self, obj, event) -> bool:
        if event.type() not in _LOCKED_INPUT_EVENTS:
            return False
        if not isinstance(obj, QWidget):
            return False
        if self._is_tutorial_chrome(obj):
            return False
        if self._is_interaction_allowed(obj):
            return False
        return True

    def _on_stack_changed(self, _index: int = 0) -> None:
        QTimer.singleShot(0, self._raise_active_overlays)

    def _raise_active_overlays(self) -> None:
        for host in getattr(self.window, "_tutorial_tooltip_hosts", []):
            if not _qt_is_valid(host):
                continue
            tip = getattr(host, "tutorial_tooltip", None)
            if tip is None or not tip.is_visible():
                continue
            win = host.window()
            if win is not None:
                _show_tutorial_overlay(win)
            if tip._bubble is not None and _qt_is_valid(tip._bubble):
                tip._bubble.raise_()
                if tip._next_btn_active():
                    tip._bubble.ensure_interactive()

    def before_processing_complete_dialog(self) -> None:
        self._hide_all_tooltips()
        hide_all_tutorial_tooltips(self.window)
        # Progress tooltip paints on the dialog overlay, not MainWindow.
        dlg = getattr(self.window, "progress_display_dialog", None)
        if dlg is not None and _qt_is_valid(dlg):
            hide_all_tutorial_tooltips(dlg)
            overlay = getattr(dlg, "_tutorial_overlay", None)
            if overlay is not None and _qt_is_valid(overlay):
                overlay.hide()

    def acknowledge_processing_complete(self) -> None:
        self._processing_dialog_acknowledged = True
        self._try_start_review_tour()

    def on_review_ui_ready(self) -> None:
        self._review_ui_ready = True
        self._try_start_review_tour()

    def _try_start_review_tour(self) -> None:
        if not self._processing_dialog_acknowledged or not self._review_ui_ready:
            return
        if self._step_index != self._STEP_PROCESSING:
            return
        self.window.raise_()
        self.window.activateWindow()
        self._advance()

    def _set_allowed_widgets(self, *allowed: QWidget) -> None:
        """Allow input only on the given widgets; all others stay visually enabled."""
        self._allowed_ids = {id(w) for w in allowed if _qt_is_valid(w)}
        _ensure_tooltip_parent_layers_enabled(self.window)

    def _allow_tooltip_target(self, target: QWidget) -> None:
        # Only the step target (and its descendants via ancestry check) may be clicked.
        self._set_allowed_widgets(target)

    def _apply_step_widget_lock(self, target: QWidget) -> None:
        tooltip = getattr(target, "tutorial_tooltip", None)
        enable_parent = tooltip.enable_parent if tooltip is not None else False
        if enable_parent:
            self._allow_tooltip_target(target)
        else:
            self._set_allowed_widgets()

    def _hide_all_tooltips(self) -> None:
        hide_all_tutorial_tooltips(self.window)

    def _show_tooltip(
        self,
        target: QWidget,
        *,
        message: str | None = None,
        press_next: bool | None = None,
        anchor_rect: QRect | None = None,
        anchor_provider: Callable[[], QRect | None] | None = None,
        margin: int | None = None,
        align: str | None = None,
        highlight: bool | None = None,
    ) -> None:
        if not _qt_is_valid(target):
            return
        tooltip = getattr(target, "tutorial_tooltip", None)
        if tooltip is None:
            attach_tutorial_tooltip(
                target,
                message or "Follow the highlighted control to continue.",
                position="bottom",
                align=align or "center",
                highlight=True if highlight is None else highlight,
                enable_parent=press_next is False,
            )
            register_tutorial_tooltip_host(self.window, target)
            tooltip = target.tutorial_tooltip
        self._apply_step_widget_lock(target)
        show_next = tooltip.press_next if press_next is None else press_next
        tooltip.show(
            message=message,
            press_next=show_next,
            on_next=self._advance if show_next else None,
            anchor_rect=anchor_rect,
            anchor_provider=anchor_provider,
            margin=margin,
            align=align,
            highlight=highlight,
        )
        _ensure_tooltip_parent_layers_enabled(self.window)

    def _review_target_ready(self, target: QWidget) -> bool:
        if not _qt_is_valid(target):
            return False
        if not target.isVisible():
            return False
        return _widget_on_screen(target, self.window)

    def _show_review_tooltip(
        self,
        target: QWidget,
        *,
        message: str | None = None,
        press_next: bool | None = None,
        anchor_rect: QRect | None = None,
        anchor_provider: Callable[[], QRect | None] | None = None,
        margin: int | None = None,
        align: str | None = None,
        highlight: bool | None = None,
        expected_step: int | None = None,
    ) -> None:
        if expected_step is not None and self._step_index != expected_step:
            return
        if not self._review_target_ready(target):
            QTimer.singleShot(
                100,
                lambda: self._show_review_tooltip(
                    target,
                    message=message,
                    press_next=press_next,
                    anchor_rect=anchor_rect,
                    anchor_provider=anchor_provider,
                    margin=margin,
                    align=align,
                    highlight=highlight,
                    expected_step=expected_step or self._step_index,
                ),
            )
            return
        if self.window.stack_widget.currentWidget() is not self.window.review_widget:
            QTimer.singleShot(
                100,
                lambda: self._show_review_tooltip(
                    target,
                    message=message,
                    press_next=press_next,
                    anchor_rect=anchor_rect,
                    anchor_provider=anchor_provider,
                    margin=margin,
                    align=align,
                    highlight=highlight,
                    expected_step=expected_step or self._step_index,
                ),
            )
            return
        self._show_tooltip(
            target,
            message=message,
            press_next=press_next,
            anchor_rect=anchor_rect,
            anchor_provider=anchor_provider,
            margin=margin,
            align=align,
            highlight=highlight,
        )

    def _show_current_step(self) -> None:
        steps: list[Callable[[], None]] = [
            self._step_review_past,
            self._step_settings,
            self._step_help,
            self._step_report,
            self._step_select_videos,
            self._step_drone,
            self._step_flight_details,
            self._step_process_videos,
            self._step_processing,
            self._step_frame_intro,
            self._step_mask_on,
            self._step_edit_frame,
            self._step_mask_off,
            self._step_gear,
        ]
        for tip_index, (col, message) in enumerate(DETECTION_COLUMN_TIPS):
            steps.append(
                lambda c=col, m=message, ti=tip_index: self._step_detection_column(
                    c, m, ti
                )
            )
        steps.append(self._step_confirm)
        steps.append(self._step_return_home)
        if self._step_index >= len(steps):
            self._finish()
            return
        steps[self._step_index]()

    def _step_review_past(self) -> None:
        self.window.stack_widget.setCurrentWidget(self.window.home_widget)
        self.window.toggle_banner_buttons(review=False)
        self._show_tooltip(self.window.banner_left_button)

    def _step_settings(self) -> None:
        self._show_tooltip(self.window.banner_right_button)

    def _step_help(self) -> None:
        self._show_tooltip(self.window.banner_help_button)

    def _step_report(self) -> None:
        self._show_tooltip(self.window.banner_report_button)

    def _step_select_videos(self) -> None:
        target = self.window.select_videos_button
        self._show_tooltip(target)
        self._hook_select_videos()

    def _step_drone(self) -> None:
        self.window.update_available_drones(select_drone=GUIDED_DRONE)
        self._show_tooltip(self.window.drone_select)

    def _step_flight_details(self) -> None:
        self.window.altitude_input.setText(GUIDED_ALTITUDE)
        self.window.flight_location_input.setText(GUIDED_LOCATION)
        self._show_tooltip(
            self.window.altitude_input,
            anchor_provider=lambda: _flight_fields_anchor_rect(self.window),
        )

    def _step_process_videos(self) -> None:
        self.window.update_remove_buttons()
        self._show_tooltip(self.window.process_button)
        self._hook_process_videos()

    def _step_processing(self) -> None:
        self._set_allowed_widgets()
        self._try_show_processing_tooltip()

    def _try_show_processing_tooltip(self) -> None:
        if self._step_index != self._STEP_PROCESSING:
            return
        dlg = getattr(self.window, "progress_display_dialog", None)
        status = getattr(self.window, "progress_status_label", None)
        if dlg is None or not dlg.isVisible() or status is None:
            QTimer.singleShot(200, self._try_show_processing_tooltip)
            return
        cancel_btn = getattr(self.window, "cancel_processsing_button", None)
        if cancel_btn is not None and _qt_is_valid(cancel_btn):
            cancel_btn.setEnabled(False)
        if not hasattr(status, "tutorial_tooltip"):
            attach_tutorial_tooltip(
                status,
                "Processing logs show live status, progress, and elapsed time while inference runs.",
                press_next=False,
                enable_parent=False,
                position="top",
                highlight=True,
                margin=4,
            )
            register_tutorial_tooltip_host(self.window, status)
        self._show_tooltip(
            status,
            message=(
                "Processing logs show live status, progress, and elapsed time "
                "while inference runs."
            ),
            highlight=True,
            anchor_provider=lambda: _processing_status_anchor(self.window),
        )

    def _step_frame_intro(self) -> None:
        self.window.stack_widget.setCurrentWidget(self.window.review_widget)
        self.window.toggle_review_buttons(enable=True)
        QTimer.singleShot(0, self._raise_active_overlays)
        self._show_review_tooltip(
            self.window.frame_player,
            anchor_provider=lambda: _frame_center_anchor(self.window.frame_player),
            expected_step=self._STEP_FRAME,
        )

    def _step_mask_on(self) -> None:
        self._prepare_mask_toggle_ui()
        self._mask_saw_on = False
        self._hook_mask_toggle()
        self._show_review_tooltip(
            self.window.toggle_display_switch,
            highlight=True,
            expected_step=self._STEP_MASK_ON,
        )

    def _step_edit_frame(self) -> None:
        """Show the draw-line control while the mask overlay is still on."""
        mw = self.window
        mw.stack_widget.setCurrentWidget(mw.review_widget)
        # Ensure the button is visible: it appears when the mask is showing.
        if not getattr(mw, "mask_active", False):
            if mw.frame_player.has_mask():
                mw.frame_player.set_mask_visible(True)
            mw.mask_active = True
            mw.update_frame_elements()
        mw._update_edit_frame_button()
        btn = mw.edit_frame_button
        btn.setVisible(True)
        btn.setEnabled(True)
        btn.raise_()
        self._show_review_tooltip(
            btn,
            expected_step=self._STEP_EDIT_FRAME,
        )

    def _step_mask_off(self) -> None:
        self._hook_mask_toggle()
        self._show_review_tooltip(
            self.window.toggle_display_switch,
            message="Click the mask toggle again to turn the overlay off and continue.",
            highlight=True,
            expected_step=self._STEP_MASK_OFF,
        )

    def _prepare_mask_toggle_ui(self) -> None:
        mw = self.window
        mw.mask_active = False
        if mw.frame_player.has_mask():
            mw.frame_player.set_mask_visible(False)
        switch = mw.toggle_display_switch
        switch.reset_position(checked=False, animate=False)
        switch.setEnabled(True)
        switch.setVisible(True)
        mw.update_frame_elements()
        # Raise only the toggle — leave mask/box icons unhighlighted underneath.
        switch.raise_()

    def _step_gear(self) -> None:
        controls = getattr(self.window, "playback_controls", None)
        if controls is not None and _qt_is_valid(controls):
            controls.show()
        btn = self.window.overlay_settings_button
        btn.setVisible(True)
        btn.setEnabled(True)
        self._show_review_tooltip(
            btn,
            expected_step=self._STEP_GEAR,
        )

    def _step_detection_column(self, col: int, message: str, tip_index: int) -> None:
        self.window.stack_widget.setCurrentWidget(self.window.review_widget)
        table = self.window.historical_items
        table.show()
        QTimer.singleShot(
            100,
            lambda: self._show_detection_column_tooltip(col, message, tip_index),
        )

    def _show_detection_column_tooltip(
        self,
        col: int,
        message: str,
        tip_index: int,
        expected_step: int | None = None,
    ) -> None:
        expected_step = (
            self._STEP_DETECTION_COL_BASE + tip_index
            if expected_step is None
            else expected_step
        )
        if self._step_index != expected_step:
            return
        table = self.window.historical_items
        if table.rowCount() == 0:
            QTimer.singleShot(
                100,
                lambda: self._show_detection_column_tooltip(
                    col, message, tip_index, expected_step
                ),
            )
            return
        table.show()
        if table.isColumnHidden(col):
            # Skip hidden columns if review layout changes.
            self._advance()
            return
        table.scrollTo(table.model().index(0, col))
        if _table_column_top_rect(table, col) is None:
            QTimer.singleShot(
                100,
                lambda: self._show_detection_column_tooltip(
                    col, message, tip_index, expected_step
                ),
            )
            return
        self._show_review_tooltip(
            table,
            message=message,
            press_next=True,
            highlight=True,
            anchor_provider=lambda c=col, t=table: _table_column_top_rect(t, c),
            margin=DETECTION_TABLE_TOOLTIP_MARGIN,
            expected_step=expected_step,
        )

    def _step_confirm(self) -> None:
        btn = self.window.confirm_detections_button
        btn.setVisible(True)
        self._show_review_tooltip(btn, expected_step=self._STEP_CONFIRM)
        self._hook_confirm_detections()

    def _step_return_home(self) -> None:
        btn = self.window.banner_left_button
        # Confirm clears confirming_detections; ensure Home is visible for this tip.
        self.window.confirming_detections = False
        if hasattr(self.window, "_apply_review_ui_state"):
            self.window._apply_review_ui_state()
        btn.setVisible(True)
        tip = getattr(btn, "tutorial_tooltip", None)
        if tip is not None:
            # Opening banner tip is Next-only; Home must be clickable.
            tip.enable_parent = True
        self._show_review_tooltip(
            btn,
            message="Press Home to return to the main screen and start your own experiments.",
            press_next=False,
            highlight=True,
            expected_step=self._STEP_HOME,
        )
        self._hook_return_home()

    def _hook_select_videos(self) -> None:
        if self._select_hooked:
            return
        self._select_hooked = True
        btn = self.window.select_videos_button
        try:
            btn.clicked.disconnect(self.window.select_videos)
        except TypeError:
            pass
        btn.clicked.connect(self._on_guided_select_videos)

    def _on_guided_select_videos(self) -> None:
        # Tutorial mode: skip the file picker and load the bundled sample clip.
        if not _try_add_example_footage(self.window):
            QMessageBox.warning(
                self.window,
                "Sample Footage Missing",
                f"Could not find {EXAMPLE_FOOTAGE_FILENAME} under sample_data/.",
            )
            return
        self.window.update_remove_buttons()
        self._allow_tooltip_target(self.window.select_videos_button)
        self._advance()

    def _hook_process_videos(self) -> None:
        if self._process_hooked:
            return
        self._process_hooked = True
        btn = self.window.process_button
        try:
            btn.clicked.disconnect(self.window.toggle_processing)
        except TypeError:
            pass
        btn.clicked.connect(self._on_guided_process)

    def _on_guided_process(self) -> None:
        self._unhook_process_videos()
        self.window.toggle_processing()
        self._advance()

    def _hook_mask_toggle(self) -> None:
        if self._mask_hooked:
            return
        self._mask_hooked = True
        self.window.toggle_display_switch.clicked.connect(self._on_guided_mask_toggle)

    def _on_guided_mask_toggle(self, _checked: bool = False) -> None:
        if self._step_index == self._STEP_MASK_ON:
            if not self._mask_saw_on:
                self._mask_saw_on = True
                self.window.toggle_display_switch.raise_()
                self._advance()
            return
        if self._step_index == self._STEP_MASK_OFF:
            self._hide_all_tooltips()
            self._unhook_mask_toggle()
            self._advance()

    def _hook_confirm_detections(self) -> None:
        if self._confirm_hooked:
            return
        self._confirm_hooked = True
        btn = self.window.confirm_detections_button
        try:
            btn.clicked.disconnect(self.window.confirm_detections)
        except TypeError:
            pass
        btn.clicked.connect(self._on_guided_confirm)

    def _on_guided_confirm(self) -> None:
        self._unhook_confirm_detections()
        # confirm_detections returns False on cancel; None/True means success.
        result = self.window.confirm_detections()
        if result is False:
            self._hook_confirm_detections()
            return
        self._advance()

    def _hook_return_home(self) -> None:
        if self._home_hooked:
            return
        self._home_hooked = True
        btn = self.window.banner_left_button
        try:
            btn.clicked.disconnect(self.window.go_to_home)
        except TypeError:
            pass
        btn.clicked.connect(self._on_guided_return_home)

    def _on_guided_return_home(self) -> None:
        self._unhook_return_home()
        was_on_review = (
            self.window.stack_widget.currentWidget() is self.window.review_widget
        )
        self.window.go_to_home()
        # User cancelled an unsaved-changes prompt — stay on this tip.
        if (
            was_on_review
            and self.window.stack_widget.currentWidget() is self.window.review_widget
        ):
            self._hook_return_home()
            return
        self._finish()
        show_tutorial_complete_dialog(parent=self.window)

    def _unhook_return_home(self) -> None:
        if not self._home_hooked:
            return
        btn = self.window.banner_left_button
        try:
            btn.clicked.disconnect(self._on_guided_return_home)
        except TypeError:
            pass
        # Re-wire the normal Home action if still on the review banner.
        try:
            btn.clicked.disconnect(self.window.go_to_home)
        except TypeError:
            pass
        if self.window.stack_widget.currentWidget() is self.window.review_widget:
            btn.clicked.connect(self.window.go_to_home)
        self._home_hooked = False

    def _unhook_confirm_detections(self) -> None:
        if not self._confirm_hooked:
            return
        btn = self.window.confirm_detections_button
        try:
            btn.clicked.disconnect(self._on_guided_confirm)
        except TypeError:
            pass
        btn.clicked.connect(self.window.confirm_detections)
        self._confirm_hooked = False

    def _unhook_select_videos(self) -> None:
        if not self._select_hooked:
            return
        btn = self.window.select_videos_button
        try:
            btn.clicked.disconnect(self._on_guided_select_videos)
        except TypeError:
            pass
        btn.clicked.connect(self.window.select_videos)
        self._select_hooked = False

    def _unhook_process_videos(self) -> None:
        if not self._process_hooked:
            return
        btn = self.window.process_button
        try:
            btn.clicked.disconnect(self._on_guided_process)
        except TypeError:
            pass
        btn.clicked.connect(self.window.toggle_processing)
        self._process_hooked = False

    def _unhook_mask_toggle(self) -> None:
        if not self._mask_hooked:
            return
        switch = self.window.toggle_display_switch
        try:
            switch.clicked.disconnect(self._on_guided_mask_toggle)
        except TypeError:
            pass
        self._mask_hooked = False

    def _advance(self) -> None:
        self._unhook_select_videos()
        self._unhook_confirm_detections()
        self._unhook_return_home()
        self._hide_all_tooltips()
        self._step_index += 1
        self._show_current_step()

    def _finish(self) -> None:
        try:
            self.window.stack_widget.currentChanged.disconnect(self._on_stack_changed)
        except TypeError:
            pass
        self._unhook_select_videos()
        self._unhook_process_videos()
        self._unhook_mask_toggle()
        self._unhook_confirm_detections()
        self._unhook_return_home()
        self._hide_all_tooltips()
        self._remove_input_guard()
        mark_tutorial_completed(self.window.settings_obj)
        self.window._guided_tour = None
        self._step_index = self._STEP_HOME + 1


def start_guided_tour(main_window) -> GuidedTour:
    tour = GuidedTour(main_window)
    main_window._guided_tour = tour
    tour.start()
    return tour


def maybe_show_tutorial(parent=None, settings_obj: QSettings | None = None) -> bool:
    """Show welcome popup, then the guided tour. Returns True if shown."""
    settings_obj = settings_obj or QSettings("BOSL", "SharkEye_App")
    if not should_show_tutorial(settings_obj):
        return False
    WelcomeDialog(settings_obj=settings_obj, parent=parent).exec()
    if parent is not None:
        start_guided_tour(parent)
    return True


def main() -> None:
    """Standalone preview: ``python src/tutorial.py``."""
    import sys

    from theme import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    WelcomeDialog().show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
