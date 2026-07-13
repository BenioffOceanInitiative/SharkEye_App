"""Central theming for SharkEye.

Consolidates color decisions and reusable Qt style snippets that were previously scattered
across inline ``setStyleSheet`` calls, and keeps them adaptive to the OS light/dark palette.
Import colors and styles from here instead of hardcoding them in widgets so the app stays
consistent and legible in both themes.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QPalette
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication


# --- Brand constants -------------------------------------------------------
# The home banner is a fixed brand surface (dark navy) in both themes; text and icons
# sitting on it are therefore always light regardless of the OS theme.
BRAND_NAVY = "#1d2633"
ON_BRAND = "white"


# --- Theme detection -------------------------------------------------------
def is_dark_mode() -> bool:
    """Best-effort detection of the OS/app dark color scheme."""
    app = QApplication.instance()
    if app is None:
        return False
    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except (AttributeError, TypeError):
        pass
    # Fallback for older Qt: infer from the window background lightness.
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def theme_icon_color() -> QColor:
    """Icon tint that keeps contrast against the themed window background."""
    return QColor("white") if is_dark_mode() else QColor("#4a4a4a")


def warning_text_color() -> str:
    """Amber warning text that stays legible in both light and dark mode.

    Previously hardcoded to ``#FFFFFF``, which was invisible on light backgrounds.
    """
    return "#ffd166" if is_dark_mode() else "#9a6a00"


# --- Icon helpers ----------------------------------------------------------
def colored_svg_icon(svg_path: str, color: QColor, size: int = 16) -> QIcon:
    """Render an SVG tinted to a solid ``color`` (used to theme monochrome glyphs)."""
    renderer = QSvgRenderer(svg_path)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()
    return QIcon(pixmap)


def banner_icon(svg_path: str, size: int = 20) -> QIcon:
    """Banner glyph tinted for the fixed dark brand surface (always light)."""
    return colored_svg_icon(svg_path, QColor(ON_BRAND), size)


# --- Reusable style snippets ----------------------------------------------
# Borderless, transparent icon-only button. Was copy-pasted verbatim at several sites.
FLAT_ICON_BUTTON = "background: transparent; border: none;"

# Banner button: flat + light text/glyph on the brand surface.
BANNER_BUTTON = f"color: {ON_BRAND}; {FLAT_ICON_BUTTON} font-size: 18px;"


def banner_surface_style() -> str:
    """Background style for the home banner surface."""
    return f"background-color: {BRAND_NAVY};"


def app_stylesheet() -> str:
    """App-wide QSS applied once at startup as the single home for shared styling.

    Kept deliberately conservative (palette-relative rules only) so it improves consistency
    without overriding the native look of individual widgets.
    """
    return """
    QToolTip {
        border: 1px solid palette(mid);
        padding: 2px;
    }
    """


def apply_theme(app) -> None:
    """Install the app-wide stylesheet. Call once after creating the QApplication."""
    app.setStyleSheet(app_stylesheet())
