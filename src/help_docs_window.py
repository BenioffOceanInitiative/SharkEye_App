"""Help documentation viewer: markdown headings become navigable pages."""

from __future__ import annotations

import argparse
import base64
import html as html_module
import mimetypes
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QRegularExpression, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QImageReader,
    QPalette,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
_IMAGE_REF_PATTERN = re.compile(r"!\[[^\]]*\]\[([^\]]+)\]")
_REFERENCE_DEF_PATTERN = re.compile(r"^\[([^\]]+)\]:\s*(\S+)\s*$")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")

_CONTENT_HEADING_STYLE = """
body {
    background-color: transparent;
    line-height: 200%;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
p, li, h1, h2, h3, h4, h5, h6 {
    line-height: 200%;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
p, ul, ol {
    margin-top: 0;
}
pre, code {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
h1, h2, h3, h4, h5, h6 {
    font-weight: normal;
    margin-bottom: 0;
}
img {
    max-width: 100%;
    height: auto;
}
"""

_IMAGE_FRAME_STYLE = "background-color: #dcdcdc; padding: 16px;"
_CONTENT_SPACING_LINES = 2
_IMAGE_SCALE_MODE = Qt.TransformationMode.SmoothTransformation


@dataclass
class HelpDocPage:
    """A single navigable section parsed from markdown."""

    title: str
    level: int
    content: str
    children: list[HelpDocPage] = field(default_factory=list)


def _strip_image_refs(text: str) -> str:
    return _IMAGE_REF_PATTERN.sub("", text).strip()


def _extract_reference_definitions(markdown_text: str) -> tuple[str, dict[str, str]]:
    definitions: dict[str, str] = {}
    kept_lines: list[str] = []
    for line in markdown_text.splitlines():
        match = _REFERENCE_DEF_PATTERN.match(line.strip())
        if match:
            definitions[match.group(1)] = match.group(2)
        else:
            kept_lines.append(line)
    return "\n".join(kept_lines), definitions


def _resolve_image_path(ref_id: str, doc_path: Path, ref_defs: dict[str, str]) -> Path | None:
    if ref_id in ref_defs:
        rel_path = ref_defs[ref_id]
        candidate = (doc_path.parent / rel_path).resolve()
        if candidate.exists():
            return candidate

    images_dir = doc_path.parent / "images"
    for ext in _IMAGE_EXTENSIONS:
        candidate = images_dir / f"{ref_id}{ext}"
        if candidate.exists():
            return candidate.resolve()

    return None


def _file_to_data_uri(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_ref_to_html(ref_id: str, doc_path: Path, ref_defs: dict[str, str]) -> str:
    image_path = _resolve_image_path(ref_id, doc_path, ref_defs)
    if image_path is None:
        return ""
    data_uri = _file_to_data_uri(image_path)
    return (
        f'<p><img src="{data_uri}" style="max-width:100%;height:auto;" '
        f'alt="{html_module.escape(ref_id)}"></p>'
    )


def _extract_html_body(fragment: str) -> str:
    match = re.search(r"<body[^>]*>(.*)</body>", fragment, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return fragment.strip()


def _markdown_fragment_to_html(markdown: str) -> str:
    if not markdown.strip():
        return ""

    document = QTextDocument()
    document.setDefaultStyleSheet(_CONTENT_HEADING_STYLE)
    document.setMarkdown(markdown.strip())
    return _extract_html_body(document.toHtml())


def _split_content_segments(content: str) -> list[tuple[str, str]]:
    """Split page body into alternating text and image segments."""
    segments: list[tuple[str, str]] = []
    last_index = 0
    for match in _IMAGE_REF_PATTERN.finditer(content):
        text_segment = content[last_index:match.start()]
        if text_segment.strip():
            segments.append(("text", text_segment))
        segments.append(("image", match.group(1)))
        last_index = match.end()

    trailing_text = content[last_index:]
    if trailing_text.strip():
        segments.append(("text", trailing_text))
    return segments


class _AutoHeightTextBrowser(QTextBrowser):
    """Read-only HTML block that grows vertically to fit its contents."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("QTextBrowser, QTextBrowser QWidget { background: transparent; }")
        self.viewport().setAutoFillBackground(False)
        self._match_window_background()
        self.document().setDocumentMargin(0)
        self.document().setDefaultStyleSheet(_CONTENT_HEADING_STYLE)
        self.document().contentsChanged.connect(self._update_height)

    def _match_window_background(self) -> None:
        window_color = self.palette().color(QPalette.ColorRole.Window)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, window_color)
        palette.setColor(QPalette.ColorRole.Window, window_color)
        self.setPalette(palette)
        self.viewport().setPalette(palette)

    def set_html_content(self, html: str) -> None:
        self.setHtml(html)
        QTimer.singleShot(0, self.reflow)

    def reflow(self, width: int | None = None) -> None:
        if width is not None:
            self.setMaximumWidth(width)
        self._update_height(width)

    def _update_height(self, width: int | None = None) -> None:
        layout_width = width if width is not None else max(self.width(), 1)
        layout_width = max(layout_width, 1)
        self.document().setTextWidth(layout_width)
        document_height = int(self.document().size().height())
        self.setFixedHeight(max(document_height + 2, 0))


class _FitWidthImageLabel(QLabel):
    """Shows a native-resolution pixmap, scaling only when the panel is narrower."""

    def __init__(self, pixmap: QPixmap, frame_padding: int = 16, parent: QWidget | None = None):
        super().__init__(parent)
        self._source_pixmap = pixmap
        self._frame_padding = frame_padding
        self._cached_width = -1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(_IMAGE_FRAME_STYLE)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.reflow_to_width(self._source_pixmap.width())

    def reflow_to_width(self, available_width: int) -> None:
        self.setMaximumWidth(max(available_width, 1))
        inner_width = max(available_width - 2 * self._frame_padding, 1)
        self._apply_scaled_pixmap(inner_width)

    def _device_ratio(self) -> float:
        window = self.window()
        if window is not None:
            return window.devicePixelRatioF()
        return self.devicePixelRatioF()

    def _display_pixmap(self, target_width: int) -> QPixmap:
        source = self._source_pixmap
        if source.isNull():
            return source

        target_width = max(min(target_width, source.width()), 1)
        if target_width >= source.width():
            display = QPixmap(source)
            display.setDevicePixelRatio(1.0)
            return display

        device_ratio = self._device_ratio()
        physical_width = max(int(target_width * device_ratio), 1)
        display = source.scaled(
            physical_width,
            source.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            _IMAGE_SCALE_MODE,
        )
        display.setDevicePixelRatio(device_ratio)
        return display

    def _apply_scaled_pixmap(self, target_width: int) -> None:
        if target_width == self._cached_width:
            return

        display = self._display_pixmap(target_width)
        if display.isNull():
            return

        self._cached_width = target_width
        self.setPixmap(display)
        logical_height = int(display.height() / display.devicePixelRatio())
        self.setFixedHeight(logical_height + 2 * self._frame_padding)


def _load_image_pixmap(image_path: Path) -> QPixmap | None:
    reader = QImageReader(str(image_path))
    reader.setAutoTransform(True)
    image: QImage = reader.read()
    if image.isNull():
        return None
    return QPixmap.fromImage(image)


def _make_image_label(
    ref_id: str,
    doc_path: Path,
    ref_defs: dict[str, str],
) -> QLabel | None:
    image_path = _resolve_image_path(ref_id, doc_path, ref_defs)
    if image_path is None:
        return None

    pixmap = _load_image_pixmap(image_path)
    if pixmap is None or pixmap.isNull():
        return None

    return _FitWidthImageLabel(pixmap)


def _render_page_html(page: HelpDocPage, doc_path: Path, ref_defs: dict[str, str]) -> str:
    heading_tag = f"h{page.level}"
    parts = [
        f"<{heading_tag}>{html_module.escape(page.title)}</{heading_tag}>",
    ]

    content = page.content
    last_index = 0
    for match in _IMAGE_REF_PATTERN.finditer(content):
        text_segment = content[last_index:match.start()]
        if text_segment.strip():
            parts.append(_markdown_fragment_to_html(text_segment))
        parts.append(_image_ref_to_html(match.group(1), doc_path, ref_defs))
        last_index = match.end()

    trailing_text = content[last_index:]
    if trailing_text.strip():
        parts.append(_markdown_fragment_to_html(trailing_text))

    body = "".join(part for part in parts if part)
    return f"<!DOCTYPE HTML><html><head></head><body>{body}</body></html>"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.casefold())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:80]


def _wrap_export_html(page_html: str, title: str) -> str:
    style_block = f"<style>{_CONTENT_HEADING_STYLE}</style>"
    head = (
        "<head>"
        '<meta charset="utf-8">'
        f"<title>{html_module.escape(title)}</title>"
        f"{style_block}"
        "</head>"
    )
    if "<head></head>" in page_html:
        return page_html.replace("<head></head>", head)
    return page_html.replace("<html>", f"<html>{head}", 1)


def export_help_docs_html(
    markdown_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Render every help page to HTML files for local browser testing."""
    doc_path = Path(markdown_path).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    pages, ref_defs = read_markdown_help_doc(doc_path)
    entries = _flatten_pages(pages)
    index_links: list[tuple[str, str]] = []

    for index, entry in enumerate(entries):
        page_html = _render_page_html(entry.page, doc_path, ref_defs)
        export_html = _wrap_export_html(page_html, entry.page.title)
        slug = _slugify(entry.page.title) or f"page-{index}"
        filename = f"{index:02d}-{slug}.html"
        (output_path / filename).write_text(export_html, encoding="utf-8")
        index_links.append((filename, entry.display_title.strip()))

    index_items = "\n".join(
        f'<li><a href="{html_module.escape(name)}">{html_module.escape(label)}</a></li>'
        for name, label in index_links
    )
    index_html = f"""<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <title>{html_module.escape(doc_path.stem)} — Help Docs Preview</title>
  <style>{_CONTENT_HEADING_STYLE}</style>
</head>
<body>
  <h1>{html_module.escape(doc_path.name)}</h1>
  <p>Generated from <code>{html_module.escape(str(doc_path))}</code></p>
  <ul>
    {index_items}
  </ul>
</body>
</html>
"""
    (output_path / "index.html").write_text(index_html, encoding="utf-8")
    return output_path


def parse_markdown_into_pages(markdown_text: str) -> list[HelpDocPage]:
    """Parse markdown headings into a page tree.

    Top-level ``#`` headings become root pages. ``##`` headings become children of
    the most recent ``#`` heading. Deeper headings (``###`` and below) stay in the
    parent page body as markdown text.
    """
    lines = markdown_text.splitlines()
    roots: list[HelpDocPage] = []
    current_root: HelpDocPage | None = None
    current_page: HelpDocPage | None = None
    body_lines: list[str] = []

    def flush_body() -> None:
        nonlocal body_lines
        if current_page is None:
            body_lines = []
            return
        text = "\n".join(body_lines).strip()
        if text:
            if current_page.content:
                current_page.content = f"{current_page.content}\n\n{text}"
            else:
                current_page.content = text
        body_lines = []

    for line in lines:
        match = _HEADING_PATTERN.match(line)
        if not match:
            body_lines.append(line)
            continue

        hashes, _title = match.group(1), match.group(2).strip()
        title_line = line[len(hashes) :].lstrip()
        title = _strip_image_refs(title_line) or "Untitled"
        heading_images = _IMAGE_REF_PATTERN.findall(title_line)
        level = len(hashes)

        if level == 1:
            flush_body()
            current_root = HelpDocPage(title=title, level=1, content="")
            roots.append(current_root)
            current_page = current_root
            for ref_id in heading_images:
                body_lines.append(f"![][{ref_id}]")
            continue

        if level == 2:
            flush_body()
            if current_root is None:
                current_page = HelpDocPage(title=title, level=1, content="")
                roots.append(current_page)
                for ref_id in heading_images:
                    body_lines.append(f"![][{ref_id}]")
                continue
            child = HelpDocPage(title=title, level=2, content="")
            current_root.children.append(child)
            current_page = child
            for ref_id in heading_images:
                body_lines.append(f"![][{ref_id}]")
            continue

        body_lines.append(line)

    flush_body()
    return roots


def read_markdown_help_doc(path: str | Path) -> tuple[list[HelpDocPage], dict[str, str]]:
    """Read a markdown file and return its navigable page tree and image refs."""
    doc_path = Path(path)
    text = doc_path.read_text(encoding="utf-8")
    text, ref_defs = _extract_reference_definitions(text)
    return parse_markdown_into_pages(text), ref_defs


@dataclass
class _NavEntry:
    page: HelpDocPage
    display_title: str


def _page_search_text(page: HelpDocPage) -> str:
    return f"{page.title}\n{page.content}".casefold()


def _search_terms(query: str) -> list[str]:
    return [term for term in query.casefold().split() if term]


def _page_matches_query(page: HelpDocPage, terms: list[str]) -> bool:
    if not terms:
        return True
    haystack = _page_search_text(page)
    return all(term in haystack for term in terms)


def _flatten_pages(pages: list[HelpDocPage]) -> list[_NavEntry]:
    entries: list[_NavEntry] = []
    for root in pages:
        entries.append(
            _NavEntry(
                page=root,
                display_title=root.title,
            )
        )
        for child in root.children:
            entries.append(
                _NavEntry(
                    page=child,
                    display_title=f"    {child.title}",
                )
            )
    return entries


_TRANSPARENT_BORDER_STYLE = """
QLineEdit, QListWidget, QTextBrowser, QScrollArea, QSplitter::handle {
    border-color: transparent;
}
QListWidget::item, QListWidget::item:selected {
    font-weight: normal;
}
"""

_HIGHLIGHT_COLOR = QColor("#fff59d")


class HelpDocContentPanel(QScrollArea):
    """Scrollable page content built from HTML text blocks and native image labels."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        self.viewport().setAutoFillBackground(False)

        self._container = QWidget()
        self._container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._container.setStyleSheet("background: transparent;")
        self._container.setMinimumWidth(0)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(self._block_spacing())
        self.setWidget(self._container)
        self.viewport().installEventFilter(self)

        self._text_blocks: list[_AutoHeightTextBrowser] = []
        self._image_labels: list[_FitWidthImageLabel] = []

    def content_width(self) -> int:
        left, _, right, _ = self._layout.getContentsMargins()
        return max(self.viewport().width() - left - right, 1)

    def _block_spacing(self) -> int:
        return self.fontMetrics().lineSpacing() * _CONTENT_SPACING_LINES

    def _reflow_blocks(self) -> None:
        viewport_width = max(self.viewport().width(), 1)
        self._container.setMaximumWidth(viewport_width)
        content_width = self.content_width()

        for text_block in self._text_blocks:
            text_block.reflow(content_width)
        for image_label in self._image_labels:
            image_label.reflow_to_width(content_width)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._reflow_blocks)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reflow_blocks)

    def clear_content(self) -> None:
        self._text_blocks.clear()
        self._image_labels.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.verticalScrollBar().setValue(0)

    def show_page(
        self,
        page: HelpDocPage,
        doc_path: Path,
        ref_defs: dict[str, str],
    ) -> None:
        self.clear_content()
        self._layout.setSpacing(self._block_spacing())

        segments = _split_content_segments(page.content)
        heading_tag = f"h{page.level}"
        title_html = f"<{heading_tag}>{html_module.escape(page.title)}</{heading_tag}>"
        start_index = 0
        if segments and segments[0][0] == "text":
            title_html += _markdown_fragment_to_html(segments[0][1])
            start_index = 1

        title_block = _AutoHeightTextBrowser(self._container)
        title_block.set_html_content(title_html)
        self._layout.addWidget(title_block)
        self._text_blocks.append(title_block)

        for kind, segment in segments[start_index:]:
            if kind == "text":
                text_block = _AutoHeightTextBrowser(self._container)
                text_block.set_html_content(_markdown_fragment_to_html(segment))
                self._layout.addWidget(text_block)
                self._text_blocks.append(text_block)
                continue

            image_label = _make_image_label(segment, doc_path, ref_defs)
            if image_label is not None:
                self._layout.addWidget(image_label)
                self._image_labels.append(image_label)

        self._layout.addStretch(1)
        QTimer.singleShot(0, self._reflow_blocks)

    def highlight_terms(self, terms: list[str]) -> None:
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(_HIGHLIGHT_COLOR)

        for text_block in self._text_blocks:
            document = text_block.document()
            for term in terms:
                pattern = QRegularExpression(
                    re.escape(term),
                    QRegularExpression.PatternOption.CaseInsensitiveOption,
                )
                cursor = QTextCursor(document)
                while True:
                    cursor = document.find(pattern, cursor)
                    if cursor.isNull():
                        break
                    cursor.mergeCharFormat(highlight_format)
                    cursor.setPosition(cursor.selectionEnd())


class HelpDocsWindow(QMainWindow):
    """Window for browsing help documentation parsed from markdown."""

    def __init__(self, markdown_path: str | Path, parent: QWidget | None = None):
        super().__init__(parent)
        self._doc_path = Path(markdown_path).resolve()
        self._pages, self._image_ref_defs = read_markdown_help_doc(self._doc_path)
        self._all_entries = _flatten_pages(self._pages)
        self._entry_items: dict[int, QListWidgetItem] = {}
        self._active_search_terms: list[str] = []

        self.setWindowTitle("Help")
        self.resize(960, 640)
        self._build_ui()
        self._populate_navigation(list(enumerate(self._all_entries)))
        self._activate_initial_selection()

    def pages(self) -> list[HelpDocPage]:
        return self._pages

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        search_row = QHBoxLayout()
        search_label = QLabel("Search:")
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search documentation…")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.setFrame(False)
        self._search_input.textChanged.connect(self._on_search_changed)
        search_row.addWidget(search_label)
        search_row.addWidget(self._search_input)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._nav_list = QListWidget()
        self._nav_list.setMinimumWidth(220)
        self._nav_list.setFrameShape(QFrame.Shape.NoFrame)
        nav_font = QFont(self._nav_list.font())
        nav_font.setWeight(QFont.Weight.Normal)
        self._nav_list.setFont(nav_font)
        self._nav_list.currentRowChanged.connect(self._on_nav_selection_changed)
        splitter.addWidget(self._nav_list)

        self._content_view = HelpDocContentPanel()
        splitter.addWidget(self._content_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 680])
        splitter.splitterMoved.connect(lambda *_: self._content_view._reflow_blocks())

        layout.addWidget(splitter)
        central.setStyleSheet(_TRANSPARENT_BORDER_STYLE)

    def _populate_navigation(self, entries: list[tuple[int, _NavEntry]]) -> None:
        self._nav_list.blockSignals(True)
        self._nav_list.clear()
        self._entry_items.clear()

        for source_index, entry in entries:
            item = QListWidgetItem(entry.display_title)
            item.setData(Qt.ItemDataRole.UserRole, source_index)
            if entry.page.level == 1:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            elif entry.page.level == 2:
                item.setToolTip(entry.page.title)
            self._nav_list.addItem(item)
            self._entry_items[source_index] = item

        self._nav_list.blockSignals(False)

    def _is_selectable_nav_row(self, row: int) -> bool:
        item = self._nav_list.item(row)
        return item is not None and bool(item.flags() & Qt.ItemFlag.ItemIsSelectable)

    def _first_selectable_row(self) -> int:
        for row in range(self._nav_list.count()):
            if self._is_selectable_nav_row(row):
                return row
        return -1

    def _activate_initial_selection(self) -> None:
        row = self._first_selectable_row()
        if row < 0:
            if self._nav_list.count() > 0:
                item = self._nav_list.item(0)
                entry_index = item.data(Qt.ItemDataRole.UserRole) if item else None
                if entry_index is not None and 0 <= entry_index < len(self._all_entries):
                    self._show_page(entry_index)
                else:
                    self._content_view.clear_content()
            else:
                self._content_view.clear_content()
            return

        self._nav_list.blockSignals(True)
        self._nav_list.setCurrentRow(row)
        self._nav_list.blockSignals(False)
        self._show_page_for_nav_row(row)

    def _show_page_for_nav_row(self, row: int) -> None:
        if row < 0:
            self._content_view.clear_content()
            return

        item = self._nav_list.item(row)
        if item is None:
            return

        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return

        entry_index = item.data(Qt.ItemDataRole.UserRole)
        if entry_index is None or not (0 <= entry_index < len(self._all_entries)):
            return

        self._show_page(entry_index)

    def _on_search_changed(self, text: str) -> None:
        self._active_search_terms = _search_terms(text.strip())
        if not self._active_search_terms:
            self._populate_navigation(list(enumerate(self._all_entries)))
            self._activate_initial_selection()
            return

        matching = [
            (index, entry)
            for index, entry in enumerate(self._all_entries)
            if _page_matches_query(entry.page, self._active_search_terms)
        ]
        self._populate_navigation(matching)
        if self._nav_list.count() > 0:
            self._activate_initial_selection()
        else:
            self._content_view.clear_content()

    def _show_page(self, entry_index: int) -> None:
        page = self._all_entries[entry_index].page
        self._content_view.show_page(page, self._doc_path, self._image_ref_defs)
        if self._active_search_terms:
            self._content_view.highlight_terms(self._active_search_terms)

    def _on_nav_selection_changed(self, row: int) -> None:
        self._show_page_for_nav_row(row)


def test_help_docs_window(guide_path: Path | None = None) -> None:
    """Launch the help viewer with the project user guide for manual testing."""
    guide_path = guide_path or (
        Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE_VISUAL.md"
    )
    app = QApplication(sys.argv)
    window = HelpDocsWindow(guide_path)
    window.show()
    sys.exit(app.exec())


def main() -> None:
    default_guide = Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE_VISUAL.md"
    parser = argparse.ArgumentParser(description="SharkEye help documentation viewer")
    parser.add_argument(
        "--export-html",
        nargs="?",
        const="help_docs_html",
        metavar="OUTPUT_DIR",
        help="Export rendered HTML pages to OUTPUT_DIR (default: ./help_docs_html)",
    )
    parser.add_argument(
        "--guide",
        type=Path,
        default=default_guide,
        help="Path to the markdown help guide",
    )
    args = parser.parse_args()

    if args.export_html:
        output_dir = export_help_docs_html(args.guide, args.export_html)
        print(f"Exported help docs HTML to: {output_dir}")
        print(f"Open in browser: {output_dir / 'index.html'}")
        return

    test_help_docs_window(args.guide)


if __name__ == "__main__":
    main()
