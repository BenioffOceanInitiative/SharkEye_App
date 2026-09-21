"""Live SharkEye inference: YOLO + tracking + interval SAM on video or camera.

PyQt preview stays at source speed. Detection, tracking, and segmentation run on
separate workers with bounded queues so lag drops frames instead of slowing playback.

Run:
    python src/live_inference_app.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from frame_sampling import parse_detections, downscale_for_preview
from log_config import get_logger
from segmentation.segmentation_model import (
    run_prediction,
    find_pixel_length,
    calculate_shark_length_from_pixel,
    draw_mask,
    release_sam_model,
)
from tracking import CustomTracker, resolve_fov_radians, load_drone_settings
from utility import resource_path, get_results_dir, select_torch_device

logger = get_logger("sharkeye.live")

MODEL_PATH = resource_path("model_weights/runs-detect-train-weights-best.pt")

# SharkEye-style sampling defaults (see frame_sampling.iter_sampled_frames).
DEFAULT_MIN_SKIP = 5
DEFAULT_MAX_SKIP = 8
DEFAULT_MAX_SKIP_SECONDS = .25
INGEST_WIDTH = 1920
INGEST_HEIGHT = 1080


def _torch_device_label(device) -> str:
    return str(device)


def _ultralytics_device_arg(device) -> Any:
    """Map a torch.device to the device arg Ultralytics predict() expects."""
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    if device.type == "mps":
        return "mps"
    return "cpu"


def prepare_ingest_source(source, work_dir: Path, status_cb=None):
    """Inspect source; if a video file is not 1920x1080, rewrite it once before inference.

    Camera sources cannot be pre-transcoded; we request 1080p from the device instead.
    Returns ``(capture_source, meta)`` where ``capture_source`` is what CaptureWorker opens.
    """
    def _status(msg: str):
        print(msg, flush=True)
        if status_cb is not None:
            status_cb(msg)

    is_camera = isinstance(source, int) or (isinstance(source, str) and str(source).isdigit())
    if is_camera:
        cam_idx = int(source)
        probe = cv2.VideoCapture(cam_idx)
        if not probe.isOpened():
            raise ValueError(f"Could not open camera {cam_idx}")
        probe.set(cv2.CAP_PROP_FRAME_WIDTH, INGEST_WIDTH)
        probe.set(cv2.CAP_PROP_FRAME_HEIGHT, INGEST_HEIGHT)
        got_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        got_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        probe.release()
        _status(
            f"[ingest] camera {cam_idx}: requested {INGEST_WIDTH}x{INGEST_HEIGHT}, "
            f"device reports {got_w}x{got_h}"
        )
        return cam_idx, {
            "kind": "camera",
            "original_width": got_w,
            "original_height": got_h,
            "ingest_path": None,
            "resized": False,
        }

    path = Path(source)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if src_w == INGEST_WIDTH and src_h == INGEST_HEIGHT:
        cap.release()
        _status(f"[ingest] {path.name} already {INGEST_WIDTH}x{INGEST_HEIGHT}; no rewrite")
        return str(path), {
            "kind": "video",
            "original_width": src_w,
            "original_height": src_h,
            "ingest_path": str(path),
            "resized": False,
            "fps": fps,
            "frame_count": frame_count,
        }

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{path.stem}_{INGEST_WIDTH}x{INGEST_HEIGHT}.mp4"
    _status(
        f"[ingest] {path.name} is {src_w}x{src_h}; rewriting to "
        f"{INGEST_WIDTH}x{INGEST_HEIGHT} -> {out_path.name} …"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (INGEST_WIDTH, INGEST_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != INGEST_WIDTH or frame.shape[0] != INGEST_HEIGHT:
                frame = cv2.resize(
                    frame, (INGEST_WIDTH, INGEST_HEIGHT), interpolation=cv2.INTER_LINEAR
                )
            writer.write(frame)
            written += 1
            if written % 100 == 0:
                _status(f"[ingest] rewritten {written}/{frame_count or '?'} frames…")
    finally:
        writer.release()
        cap.release()

    _status(f"[ingest] rewrite complete ({written} frames) -> {out_path}")
    return str(out_path), {
        "kind": "video",
        "original_width": src_w,
        "original_height": src_h,
        "ingest_path": str(out_path),
        "resized": True,
        "fps": fps,
        "frame_count": written,
    }


def parse_start_timestamp(value: str) -> float:
    """Parse a start timestamp as seconds, MM:SS, or HH:MM:SS."""
    text = (value or "").strip()
    if not text:
        return 0.0
    parts = text.replace(",", ".").split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, seconds_part = parts
            seconds = float(minutes) * 60.0 + float(seconds_part)
        elif len(parts) == 3:
            hours, minutes, seconds_part = parts
            seconds = (
                float(hours) * 3600.0
                + float(minutes) * 60.0
                + float(seconds_part)
            )
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Start timestamp must be seconds, MM:SS, or HH:MM:SS") from exc
    if seconds < 0:
        raise ValueError("Start timestamp cannot be negative")
    return seconds


# ---------------------------------------------------------------------------
# Shared buffers / packets
# ---------------------------------------------------------------------------

class LatestFrameBuffer:
    """Single-slot latest-frame store; writers always overwrite, readers never block."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.frame_idx: int = -1
        self.timestamp_ms: float = 0.0
        self.source_time_s: float = 0.0
        self.capture_wall: float = 0.0
        self.eof: bool = False
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def write(self, frame, frame_idx, timestamp_ms, source_time_s, fps=None, eof=False):
        with self._lock:
            self.frame = frame
            self.frame_idx = frame_idx
            self.timestamp_ms = timestamp_ms
            self.source_time_s = source_time_s
            self.capture_wall = time.perf_counter()
            self.eof = eof
            if fps is not None:
                self.fps = fps
            if frame is not None:
                self.height, self.width = frame.shape[:2]

    def snapshot(self):
        with self._lock:
            if self.frame is None:
                return None
            return {
                "frame": self.frame,  # capture owns fresh buffers; consumers copy if mutate
                "frame_idx": self.frame_idx,
                "timestamp_ms": self.timestamp_ms,
                "source_time_s": self.source_time_s,
                "capture_wall": self.capture_wall,
                "eof": self.eof,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
            }


@dataclass
class DetectionPacket:
    frame: np.ndarray
    frame_idx: int
    timestamp_ms: float
    source_time_s: float
    detections: list
    yolo_ms: float
    detection_done_wall: float
    dropped_before: int = 0


@dataclass
class SegJob:
    track_id: int
    frame: np.ndarray
    box_xywh: tuple  # (cx, cy, w, h)
    confidence: float
    timestamp_ms: float
    detection_done_wall: float
    tracking_done_wall: float
    tracking_lag_ms: float
    enqueued_wall: float = field(default_factory=time.perf_counter)


@dataclass
class TrackOverlay:
    track_id: int
    box_xywh: tuple
    confidence: float
    length_ft: Optional[float]
    length_source: str  # "bbox" | "sam"
    n_dets: int
    # Kinematics for display-time extrapolation (box is from timestamp_ms; preview
    # frames are usually newer because YOLO only runs every N frames).
    timestamp_ms: float = 0.0
    velocity_xy: tuple = (0.0, 0.0)  # pixels per millisecond (matches CustomTracker)


class LiveTrackState:
    """Thread-safe overlay + export state published by tracking/seg workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self.overlays: list[TrackOverlay] = []
        self.active_tracks: int = 0
        self.seg_done: int = 0
        # track_id -> export record
        self.records: dict[int, dict] = {}

    def set_overlays(self, overlays: list[TrackOverlay], active: int):
        with self._lock:
            self.overlays = list(overlays)
            self.active_tracks = active

    def get_overlays(self):
        with self._lock:
            return list(self.overlays), self.active_tracks

    def upsert_record(self, track_id: int, **kwargs):
        with self._lock:
            rec = self.records.setdefault(track_id, {"track_id": track_id})
            rec.update(kwargs)

    def get_field(self, track_id: int, key: str, default=None):
        with self._lock:
            rec = self.records.get(track_id)
            if rec is None:
                return default
            return rec.get(key, default)

    def snapshot_records(self) -> dict[int, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self.records.items()}


class MetricsLogger:
    """Console + jsonl metrics for lag, timing, and memory."""

    def __init__(self, metrics_path: Optional[Path] = None):
        self.metrics_path = metrics_path
        self._lock = threading.Lock()
        self.dropped_frames = 0
        self.yolo_ms = deque(maxlen=200)
        self.track_ms = deque(maxlen=200)
        self.tracking_lag_ms = deque(maxlen=200)
        self.sam_ms = deque(maxlen=50)
        self.detection_to_seg_ms = deque(maxlen=50)
        self.tracking_to_seg_ms = deque(maxlen=50)
        self.yolo_frames = 0
        self._yolo_window_t0 = time.perf_counter()
        self._yolo_window_n = 0
        self.last_lag_s = 0.0
        self._last_perf_print = 0.0
        self._fh = None
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(metrics_path, "a", encoding="utf-8")

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def _write(self, event: dict):
        event["wall"] = time.time()
        line = json.dumps(event, default=float)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    @staticmethod
    def _rss_gb() -> float:
        try:
            import psutil  # optional
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        except Exception:
            pass
        # Windows fallback without adding a hard dependency.
        if sys.platform.startswith("win"):
            try:
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return counters.WorkingSetSize / (1024 ** 3)
            except Exception:
                pass
        try:
            import resource
            # Linux reports KB; macOS bytes. Prefer ru_maxrss best-effort.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 ** 3)
            return rss / (1024 ** 2) / 1024.0
        except Exception:
            return -1.0

    @staticmethod
    def _vram_gb() -> Optional[float]:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return None

    def record_drop(self, n: int = 1):
        with self._lock:
            self.dropped_frames += n
        self._write({"event": "drop", "n": n, "dropped_total": self.dropped_frames})

    def record_yolo(self, ms: float, lag_s: float = 0.0):
        with self._lock:
            self.yolo_ms.append(ms)
            self.yolo_frames += 1
            self._yolo_window_n += 1
            self.last_lag_s = lag_s
        self._write({"event": "yolo", "ms": ms, "lag_s": lag_s})
        self.maybe_perf_summary()

    def record_track(self, ms: float, track_id: Optional[int] = None, n_dets: int = 0,
                     detection_to_track_ms: Optional[float] = None):
        with self._lock:
            self.track_ms.append(ms)
            if detection_to_track_ms is not None:
                self.tracking_lag_ms.append(detection_to_track_ms)
        self._write({
            "event": "track",
            "ms": ms,
            "track_id": track_id,
            "n_dets": n_dets,
            "detection_to_track_ms": detection_to_track_ms,
        })

    def record_sam(self, ms: float, track_id: int, length_ft: float,
                   detection_to_seg_ms: Optional[float] = None,
                   tracking_to_seg_ms: Optional[float] = None):
        with self._lock:
            self.sam_ms.append(ms)
            if detection_to_seg_ms is not None:
                self.detection_to_seg_ms.append(detection_to_seg_ms)
            if tracking_to_seg_ms is not None:
                self.tracking_to_seg_ms.append(tracking_to_seg_ms)
        self._write({
            "event": "sam",
            "ms": ms,
            "track_id": track_id,
            "length_ft": length_ft,
            "detection_to_seg_ms": detection_to_seg_ms,
            "tracking_to_seg_ms": tracking_to_seg_ms,
        })

    def record_lag(self, lag_s: float, detail: str = ""):
        with self._lock:
            self.last_lag_s = lag_s
        if lag_s > 0.25:
            self._write({"event": "lag", "lag_s": lag_s, "detail": detail})

    @staticmethod
    def _p50(values) -> float:
        if not values:
            return 0.0
        arr = sorted(values)
        return float(arr[len(arr) // 2])

    def maybe_perf_summary(self, force: bool = False):
        now = time.perf_counter()
        if not force and (now - self._last_perf_print) < 5.0:
            return
        with self._lock:
            elapsed = max(1e-6, now - self._yolo_window_t0)
            yolo_fps = self._yolo_window_n / elapsed
            self._yolo_window_t0 = now
            self._yolo_window_n = 0
            lag = self.last_lag_s
            drop = self.dropped_frames
            track_p50 = self._p50(self.track_ms)
            tracking_lag_p50 = self._p50(self.tracking_lag_ms)
            sam_p50 = self._p50(self.sam_ms)
            detection_to_seg_p50 = self._p50(self.detection_to_seg_ms)
            tracking_to_seg_p50 = self._p50(self.tracking_to_seg_ms)
            yolo_p50 = self._p50(self.yolo_ms)
        rss = self._rss_gb()
        vram = self._vram_gb()
        vram_s = f" vram={vram:.2f}GB" if vram is not None else ""
        msg = (
            f"[perf] lag={lag:.2f}s drop={drop} yolo={yolo_fps:.1f}fps "
            f"yolo_p50={yolo_p50:.0f}ms track_p50={track_p50:.1f}ms "
            f"track_lag_p50={tracking_lag_p50:.1f}ms sam_p50={sam_p50:.0f}ms "
            f"det_to_seg_p50={detection_to_seg_p50:.0f}ms "
            f"track_to_seg_p50={tracking_to_seg_p50:.0f}ms rss={rss:.2f}GB{vram_s}"
        )
        print(msg, flush=True)
        logger.info(msg)
        self._write({
            "event": "perf",
            "lag_s": lag,
            "drop": drop,
            "yolo_fps": yolo_fps,
            "yolo_p50_ms": yolo_p50,
            "track_p50_ms": track_p50,
            "tracking_lag_p50_ms": tracking_lag_p50,
            "sam_p50_ms": sam_p50,
            "detection_to_seg_p50_ms": detection_to_seg_p50,
            "tracking_to_seg_p50_ms": tracking_to_seg_p50,
            "rss_gb": rss,
            "vram_gb": vram,
        })
        self._last_perf_print = now


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class CaptureWorker(QThread):
    """Read video/camera at source pace; never wait on inference.

    When file playback falls behind wall-clock, skip frames with ``grab()`` so the
    displayed stream stays near real time instead of crawling.
    """

    status = pyqtSignal(str)
    finished_capture = pyqtSignal()

    def __init__(self, source, frame_buf: LatestFrameBuffer,
                 metrics: Optional[MetricsLogger] = None,
                 start_time_s: float = 0.0, parent=None):
        super().__init__(parent)
        self.source = source
        self.frame_buf = frame_buf
        self.metrics = metrics
        self.start_time_s = max(0.0, float(start_time_s or 0.0))
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        # Camera index as int, else path string.
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            src = int(self.source)
            is_camera = True
        else:
            src = str(self.source)
            is_camera = False

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status.emit(f"Could not open source: {self.source}")
            self.finished_capture.emit()
            return

        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, INGEST_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INGEST_HEIGHT)

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(fps) or fps <= 1e-3:
            fps = 30.0
        frame_period = 1.0 / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_buf.fps = fps
        self.status.emit(
            f"Capture started ({'camera' if is_camera else 'video'} @ {fps:.2f} fps, "
            f"{width}x{height})"
        )

        start_frame = 0
        if not is_camera and self.start_time_s > 0:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            start_frame = int(round(self.start_time_s * fps))
            if frame_count > 0:
                start_frame = min(start_frame, max(0, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.status.emit(
                f"[capture] seeking to {start_frame / fps:.2f}s "
                f"(frame {start_frame})"
            )
        elif is_camera and self.start_time_s > 0:
            self.status.emit("[capture] start timestamp ignored for camera sources")

        frame_idx = start_frame
        skipped_total = 0
        t0 = time.perf_counter() - (frame_idx * frame_period)
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                self.frame_buf.write(None, frame_idx, frame_idx / fps * 1000.0,
                                     frame_idx / fps, fps=fps, eof=True)
                break

            if not is_camera:
                # Skip-ahead: if wall clock is ahead of this frame's schedule, grab through
                # unread frames until we land on the frame that should be showing now.
                now = time.perf_counter()
                desired_idx = int((now - t0) * fps)
                while frame_idx < desired_idx:
                    if not cap.grab():
                        ret = False
                        break
                    frame_idx += 1
                    skipped_total += 1
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    now = time.perf_counter()
                    desired_idx = int((now - t0) * fps)
                if not ret:
                    self.frame_buf.write(None, frame_idx, frame_idx / fps * 1000.0,
                                         frame_idx / fps, fps=fps, eof=True)
                    break
                if skipped_total and skipped_total % 30 == 0:
                    self.status.emit(
                        f"[capture] skipped ahead; total_skipped={skipped_total} at frame={frame_idx}"
                    )

            timestamp_ms = frame_idx / fps * 1000.0
            source_time_s = frame_idx / fps
            self.frame_buf.write(frame, frame_idx, timestamp_ms, source_time_s, fps=fps)

            frame_idx += 1
            if not is_camera:
                target = t0 + frame_idx * frame_period
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            else:
                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.001:
                    time.sleep(0.001)

        if skipped_total:
            msg = f"[capture] finished with {skipped_total} skipped frame(s) to stay real-time"
            print(msg, flush=True)
            if self.metrics is not None:
                self.metrics.record_drop(skipped_total)
            self.status.emit(msg)

        cap.release()
        self.status.emit("Capture finished")
        self.finished_capture.emit()


class DetectionWorker(QThread):
    """YOLO at SharkEye-like adaptive skip; drops frames when lagging."""

    status = pyqtSignal(str)

    def __init__(
        self,
        model: YOLO,
        frame_buf: LatestFrameBuffer,
        det_queue: queue.Queue,
        metrics: MetricsLogger,
        confidence: float = 0.40,
        min_skip: int = DEFAULT_MIN_SKIP,
        max_skip: int = DEFAULT_MAX_SKIP,
        max_skip_seconds: float = DEFAULT_MAX_SKIP_SECONDS,
        device: Any = "cpu",
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.frame_buf = frame_buf
        self.det_queue = det_queue
        self.metrics = metrics
        self.confidence = confidence
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.max_skip_seconds = max_skip_seconds
        self.device = device
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        self.status.emit("Detection worker started")
        frame_skip = self.min_skip
        consecutive_empty = 0
        last_inferred_idx = -10**9
        dropped = 0

        while not self._stop.is_set():
            snap = self.frame_buf.snapshot()
            if snap is None:
                time.sleep(0.005)
                continue
            if snap["eof"] and snap["frame"] is None:
                break

            frame = snap["frame"]
            if frame is None:
                time.sleep(0.005)
                continue

            fps = snap["fps"] or 30.0
            empty_backoff = max(1, int(fps))
            eff_max_skip = max(
                self.min_skip,
                min(self.max_skip, int(round(fps * self.max_skip_seconds))),
            )

            # Lag vs source: wall time since this frame was captured.
            lag_s = max(0.0, time.perf_counter() - snap["capture_wall"])
            self.metrics.record_lag(lag_s, detail="pre_yolo")

            # If lagging, force a larger skip (lag-free policy).
            if lag_s > 0.15:
                frame_skip = min(eff_max_skip, max(frame_skip, self.min_skip * 2))
            if lag_s > 0.40:
                frame_skip = eff_max_skip

            idx = snap["frame_idx"]
            if idx - last_inferred_idx < frame_skip:
                # Not yet due — wait for a newer frame rather than re-inferring.
                time.sleep(0.001)
                continue

            # Count skipped frames since last inference for telemetry.
            gap = idx - last_inferred_idx
            if last_inferred_idx >= 0 and gap > frame_skip:
                extra = gap - frame_skip
                dropped += extra
                self.metrics.record_drop(extra)

            t0 = time.perf_counter()
            results = self.model(
                frame, classes=[0], verbose=False, device=self.device,
            )
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            detections = parse_detections(results, self.confidence)
            detection_done_wall = time.perf_counter()
            last_inferred_idx = idx
            self.metrics.record_yolo(yolo_ms, lag_s=lag_s)

            if detections:
                consecutive_empty = 0
                frame_skip = self.min_skip
            else:
                consecutive_empty += frame_skip
                if consecutive_empty >= empty_backoff:
                    frame_skip = min(eff_max_skip, frame_skip * 2)

            packet = DetectionPacket(
                frame=frame,
                frame_idx=idx,
                timestamp_ms=snap["timestamp_ms"],
                source_time_s=snap["source_time_s"],
                detections=detections,
                yolo_ms=yolo_ms,
                detection_done_wall=detection_done_wall,
                dropped_before=dropped,
            )
            # Bounded queue: replace stale packet instead of blocking capture path.
            try:
                self.det_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self.det_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.det_queue.put_nowait(packet)
                except queue.Full:
                    self.metrics.record_drop(1)

        self.status.emit("Detection worker stopped")


class TrackingWorker(QThread):
    """Group detections with CustomTracker; enqueue one SAM job per stable track."""

    status = pyqtSignal(str)
    console_line = pyqtSignal(str)

    def __init__(
        self,
        det_queue: queue.Queue,
        seg_pending: dict,
        seg_pending_lock: threading.Lock,
        seg_wakeup: threading.Event,
        track_state: LiveTrackState,
        metrics: MetricsLogger,
        drone_type: str,
        altitude: float,
        fov_radians: float,
        confidence: float = 0.40,
        min_frames: int = 5,
        parent=None,
    ):
        super().__init__(parent)
        self.det_queue = det_queue
        self.seg_pending = seg_pending
        self.seg_pending_lock = seg_pending_lock
        self.seg_wakeup = seg_wakeup
        self.track_state = track_state
        self.metrics = metrics
        self.drone_type = drone_type
        self.altitude = altitude
        self.fov_radians = fov_radians
        self.confidence = confidence
        self.min_frames = min_frames
        self._stop = threading.Event()
        self._enqueued_sam: set[int] = set()
        self.tracker = CustomTracker(
            min_frames=min_frames,
            confidence_threshold=confidence,
            fov_radians=fov_radians,
            drone_altitude=altitude,
        )

    def stop(self):
        self._stop.set()

    def run(self):
        self.status.emit("Tracking worker started")
        while not self._stop.is_set():
            try:
                packet: DetectionPacket = self.det_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            active_track_ids = set()
            if packet.detections:
                # Align FOV to actual frame geometry on first detection batch.
                if not getattr(self, "_fov_resolved", False):
                    h, w = packet.frame.shape[:2]
                    settings = load_drone_settings()
                    fov = resolve_fov_radians(self.drone_type, w, h, settings)
                    if fov is not None:
                        self.tracker.fov_radians = fov
                        self.fov_radians = fov
                    self._fov_resolved = True
                active_track_ids = self.tracker.update(
                    packet.detections, packet.frame, packet.timestamp_ms
                )
            tracking_done_wall = time.perf_counter()
            track_ms = (tracking_done_wall - t0) * 1000.0
            tracking_lag_ms = max(
                0.0, (tracking_done_wall - packet.detection_done_wall) * 1000.0
            )
            self.metrics.record_track(
                track_ms,
                n_dets=len(packet.detections),
                detection_to_track_ms=(tracking_lag_ms if packet.detections else None),
            )

            overlays = []
            for tid, track in self.tracker.tracks.items():
                if not track.get("positions"):
                    continue
                pos = track["positions"][-1]
                conf = float(track["confidences"][-1]) if track.get("confidences") else 0.0
                n_dets = len(track["positions"])
                last_ts = float(track["timestamps"][-1]) if track.get("timestamps") else float(packet.timestamp_ms)
                vel = track.get("velocity")
                if vel is None:
                    velocity_xy = (0.0, 0.0)
                else:
                    velocity_xy = (float(vel[0]), float(vel[1]))
                # SAM lengths are written by the segmentation worker into track_state.
                sam_len = self.track_state.get_field(tid, "sam_length_ft")
                if sam_len is None:
                    sam_len = track.get("sam_length")
                length = float(sam_len) if sam_len is not None else float(track.get("best_length") or 0.0)
                src = "sam" if sam_len is not None else "bbox"
                overlays.append(TrackOverlay(
                    track_id=tid,
                    box_xywh=tuple(pos),
                    confidence=conf,
                    length_ft=length,
                    length_source=src,
                    n_dets=n_dets,
                    timestamp_ms=last_ts,
                    velocity_xy=velocity_xy,
                ))

                age_s = 0.0
                if track.get("timestamps"):
                    age_s = max(0.0, (packet.timestamp_ms - track["timestamps"][0]) / 1000.0)

                best_conf = float(track.get("best_conf") or conf)
                significant = self.tracker.is_significant_track(track)
                selected = significant and tid not in self._enqueued_sam

                line = (
                    f"[track] id={tid} dets={n_dets} conf={best_conf:.2f} "
                    f"age={age_s:.1f}s track_lag={tracking_lag_ms:.1f}ms "
                    f"selected_for_sam={'yes' if selected else 'no'}"
                )
                # Throttle console: print on new det batches for significant / newly selected.
                if packet.detections and (significant or n_dets == 1 or selected):
                    print(line, flush=True)
                    self.console_line.emit(line)

                self.track_state.upsert_record(
                    tid,
                    n_dets=n_dets,
                    best_conf=best_conf,
                    avg_conf=float(np.mean(track["confidences"])) if track.get("confidences") else 0.0,
                    best_timestamp_ms=float(track.get("best_timestamp") or 0.0),
                    bbox_length_ft=float(track.get("best_length") or 0.0),
                    sam_length_ft=track.get("sam_length"),
                    label=track.get("label", "Shark"),
                    meets_thresholds=significant,
                    best_pos=track.get("best_pos") or pos,
                    last_pos=tuple(pos),
                    tracking_lag_ms=(
                        tracking_lag_ms if tid in active_track_ids
                        else self.track_state.get_field(tid, "tracking_lag_ms")
                    ),
                    tracking_done_wall=(
                        tracking_done_wall if tid in active_track_ids
                        else self.track_state.get_field(tid, "tracking_done_wall")
                    ),
                )

                if selected:
                    # Prefer highest-confidence frame (Priority 2).
                    seg_frame = track.get("best_frame")
                    seg_box = track.get("best_pos") or pos
                    if seg_frame is not None and seg_box is not None:
                        job = SegJob(
                            track_id=tid,
                            frame=seg_frame.copy(),
                            box_xywh=tuple(seg_box),
                            confidence=best_conf,
                            timestamp_ms=float(track.get("best_timestamp") or packet.timestamp_ms),
                            detection_done_wall=packet.detection_done_wall,
                            tracking_done_wall=tracking_done_wall,
                            tracking_lag_ms=tracking_lag_ms,
                        )
                        with self.seg_pending_lock:
                            # Replace pending job for this track (no backlog growth).
                            prev = self.seg_pending.get(tid)
                            if prev is None or job.confidence >= prev.confidence:
                                self.seg_pending[tid] = job
                        self.seg_wakeup.set()
                        self._enqueued_sam.add(tid)
                        # Keep a copy of the selected frame for export even before SAM.
                        self.track_state.upsert_record(
                            tid,
                            export_frame=job.frame,
                            export_box=job.box_xywh,
                        )

                # If already queued but a better conf appears before SAM starts, replace.
                elif significant and tid in self._enqueued_sam:
                    best_conf_now = float(track.get("best_conf") or 0.0)
                    with self.seg_pending_lock:
                        pending = self.seg_pending.get(tid)
                        if pending is not None and best_conf_now > pending.confidence + 1e-6:
                            seg_frame = track.get("best_frame")
                            seg_box = track.get("best_pos")
                            if seg_frame is not None and seg_box is not None:
                                self.seg_pending[tid] = SegJob(
                                    track_id=tid,
                                    frame=seg_frame.copy(),
                                    box_xywh=tuple(seg_box),
                                    confidence=best_conf_now,
                                    timestamp_ms=float(track.get("best_timestamp") or packet.timestamp_ms),
                                    detection_done_wall=packet.detection_done_wall,
                                    tracking_done_wall=tracking_done_wall,
                                    tracking_lag_ms=tracking_lag_ms,
                                )
                                self.seg_wakeup.set()
                                self.track_state.upsert_record(
                                    tid,
                                    export_frame=seg_frame.copy(),
                                    export_box=tuple(seg_box),
                                )

            self.track_state.set_overlays(overlays, active=len(overlays))

        self.status.emit("Tracking worker stopped")


class SegmentationWorker(QThread):
    """SAM on pending track jobs; one-at-a-time with replace-before-start semantics."""

    status = pyqtSignal(str)
    console_line = pyqtSignal(str)
    length_updated = pyqtSignal(int, float)

    def __init__(
        self,
        seg_pending: dict,
        seg_pending_lock: threading.Lock,
        seg_wakeup: threading.Event,
        track_state: LiveTrackState,
        metrics: MetricsLogger,
        altitude: float,
        fov_radians: float,
        parent=None,
    ):
        super().__init__(parent)
        self.seg_pending = seg_pending
        self.seg_pending_lock = seg_pending_lock
        self.seg_wakeup = seg_wakeup
        self.track_state = track_state
        self.metrics = metrics
        self.altitude = altitude
        self.fov_radians = fov_radians
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        self.seg_wakeup.set()

    def _pop_job(self) -> Optional[SegJob]:
        with self.seg_pending_lock:
            if not self.seg_pending:
                return None
            # Prefer oldest enqueued to avoid starvation.
            tid = min(self.seg_pending.keys(), key=lambda k: self.seg_pending[k].enqueued_wall)
            return self.seg_pending.pop(tid)

    def run(self):
        self.status.emit("Segmentation worker started")
        while not self._stop.is_set():
            job = self._pop_job()
            if job is None:
                self.seg_wakeup.wait(timeout=0.1)
                self.seg_wakeup.clear()
                continue

            cx, cy, bw, bh = job.box_xywh
            box = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
            rgb = cv2.cvtColor(job.frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            try:
                mask = run_prediction(rgb, box)
                pixel_length = find_pixel_length(mask, draw_line=False)
                length_ft = float(calculate_shark_length_from_pixel(
                    pixel_length,
                    original_width=job.frame.shape[1],
                    original_height=job.frame.shape[0],
                    drone_altitude=self.altitude,
                    fov_radians=self.fov_radians,
                ))
                mask_overlay = draw_mask(mask, rgb)
                mask_area = int(np.count_nonzero(mask))
            except Exception as e:
                sam_ms = (time.perf_counter() - t0) * 1000.0
                msg = f"[seg] track={job.track_id} FAILED after {sam_ms:.0f}ms: {e}"
                print(msg, flush=True)
                logger.exception(msg)
                continue

            segmentation_done_wall = time.perf_counter()
            sam_ms = (segmentation_done_wall - t0) * 1000.0
            detection_to_seg_ms = max(
                0.0, (segmentation_done_wall - job.detection_done_wall) * 1000.0
            )
            tracking_to_seg_ms = max(
                0.0, (segmentation_done_wall - job.tracking_done_wall) * 1000.0
            )
            self.metrics.record_sam(
                sam_ms,
                job.track_id,
                length_ft,
                detection_to_seg_ms=detection_to_seg_ms,
                tracking_to_seg_ms=tracking_to_seg_ms,
            )

            # Publish length into shared track_state (UI + export). Avoid mutating
            # CustomTracker from this thread — the tracking worker owns it.
            self.track_state.upsert_record(
                job.track_id,
                sam_length_ft=length_ft,
                sam_ms=sam_ms,
                mask_overlay=mask_overlay,
                export_frame=job.frame,
                export_box=job.box_xywh,
                mask_area=mask_area,
                tracking_lag_ms=job.tracking_lag_ms,
                detection_to_seg_ms=detection_to_seg_ms,
                tracking_to_seg_ms=tracking_to_seg_ms,
                segmentation_done_wall=segmentation_done_wall,
            )
            self.length_updated.emit(job.track_id, length_ft)

            msg = (
                f"[seg] track={job.track_id} length={length_ft:.1f}ft "
                f"sam={sam_ms / 1000.0:.2f}s "
                f"det_to_seg={detection_to_seg_ms:.0f}ms "
                f"track_to_seg={tracking_to_seg_ms:.0f}ms "
                f"mask_area={mask_area}"
            )
            print(msg, flush=True)
            self.console_line.emit(msg)

        self.status.emit("Segmentation worker stopped")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_live_results(
    output_dir: Path,
    source_name: str,
    track_state: LiveTrackState,
    metrics: MetricsLogger,
    drone_type: str,
    altitude: float,
    flight_location: str = "",
    ingest_meta: Optional[dict] = None,
    start_time_s: float = 0.0,
):
    """Write SharkEye-like frames/masks/CSV under output_dir."""
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    det_dir = output_dir / "detection_results"
    for d in (frames_dir, masks_dir, det_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = track_state.snapshot_records()
    video_stem = Path(source_name).name
    csv_path = det_dir / f"{video_stem}.csv"
    frames_written = 0
    masks_written = 0
    sam_tracks = 0
    ingest_meta = ingest_meta or {}

    def _metric_values(key: str) -> list[float]:
        values = []
        for rec in records.values():
            value = rec.get(key)
            if value in ("", None):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        return values

    def _mean(values: list[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    def _p50(values: list[float]) -> Optional[float]:
        return float(np.median(values)) if values else None

    fieldnames = [
        "video_name", "Flight Location", "Drone", "Altitude", "Track Id", "Length (ft)",
        "Highest Conf Timestamp", "Longest Length Timestamp", "Highest Confidence",
        "Average Confidence", "Lowest Confidence", "Longest Length",
        "Highest Confidence Length", "Number of Detections", "Meets Thresholds",
        "Confidence of Longest Length", "Label", "manual_length_px", "manual_length_ft",
        "SAM Time (ms)", "Detection to Tracking Lag (ms)",
        "Detection to Segmentation Lag (ms)", "Tracking to Segmentation Lag (ms)",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for tid, rec in sorted(records.items()):
            best_ts = float(rec.get("best_timestamp_ms") or 0.0)
            ts_str = CustomTracker._format_timestamp(best_ts)
            bbox_len = float(rec.get("bbox_length_ft") or 0.0)
            sam_len = rec.get("sam_length_ft")
            length = float(sam_len) if sam_len is not None else bbox_len
            n_dets = int(rec.get("n_dets") or 0)
            best_conf = float(rec.get("best_conf") or 0.0)
            avg_conf = float(rec.get("avg_conf") or best_conf)

            writer.writerow({
                "video_name": source_name,
                "Flight Location": flight_location,
                "Drone": drone_type,
                "Altitude": altitude,
                "Track Id": tid,
                "Length (ft)": length,
                "Highest Conf Timestamp": ts_str,
                "Longest Length Timestamp": ts_str,
                "Highest Confidence": best_conf,
                "Average Confidence": avg_conf,
                "Lowest Confidence": best_conf,  # live path does not retain full conf history
                "Longest Length": bbox_len,
                "Highest Confidence Length": length if sam_len is not None else bbox_len,
                "Number of Detections": n_dets,
                "Meets Thresholds": bool(rec.get("meets_thresholds")),
                "Confidence of Longest Length": best_conf,
                "Label": rec.get("label", "Shark"),
                "manual_length_px": "",
                "manual_length_ft": "",
                "SAM Time (ms)": rec.get("sam_ms", ""),
                "Detection to Tracking Lag (ms)": rec.get("tracking_lag_ms", ""),
                "Detection to Segmentation Lag (ms)": rec.get("detection_to_seg_ms", ""),
                "Tracking to Segmentation Lag (ms)": rec.get("tracking_to_seg_ms", ""),
            })

            filename = f"{video_stem}_{tid}.jpg"
            frame = rec.get("export_frame")
            if frame is not None:
                # Burn a simple box for the frames/ artifact.
                out = frame.copy()
                box = rec.get("export_box") or rec.get("best_pos") or rec.get("last_pos")
                if box is not None:
                    cx, cy, bw, bh = box
                    cv2.rectangle(
                        out,
                        (int(cx - bw / 2), int(cy - bh / 2)),
                        (int(cx + bw / 2), int(cy + bh / 2)),
                        (0, 255, 0),
                        2,
                    )
                    label = f"ID {tid}: {best_conf:.2f}"
                    if sam_len is not None:
                        label += f" {float(sam_len):.1f}ft"
                    cv2.putText(
                        out, label,
                        (int(cx - bw / 2), max(20, int(cy - bh / 2) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    )
                cv2.imwrite(str(frames_dir / filename), out)
                frames_written += 1

            mask_overlay = rec.get("mask_overlay")
            if mask_overlay is not None:
                cv2.imwrite(str(masks_dir / filename), mask_overlay)
                masks_written += 1
                sam_tracks += 1

    metrics.maybe_perf_summary(force=True)
    rss_gb = MetricsLogger._rss_gb()
    tracking_lags = _metric_values("tracking_lag_ms")
    detection_to_seg_lags = _metric_values("detection_to_seg_ms")
    tracking_to_seg_lags = _metric_values("tracking_to_seg_ms")
    summary_path = output_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": source_name,
        "tracks": len(records),
        "sam_tracks": sam_tracks,
        "frames_written": frames_written,
        "masks_written": masks_written,
        "dropped_frames": metrics.dropped_frames,
        "yolo_frames": metrics.yolo_frames,
        "rss_gb": rss_gb,
        "ingest_width": INGEST_WIDTH,
        "ingest_height": INGEST_HEIGHT,
        "original_width": ingest_meta.get("original_width"),
        "original_height": ingest_meta.get("original_height"),
        "ingest_resized": bool(ingest_meta.get("resized")),
        "ingest_path": ingest_meta.get("ingest_path"),
        "start_time_s": float(start_time_s or 0.0),
        "tracking_lag_mean_ms": _mean(tracking_lags),
        "tracking_lag_p50_ms": _p50(tracking_lags),
        "detection_to_seg_mean_ms": _mean(detection_to_seg_lags),
        "detection_to_seg_p50_ms": _p50(detection_to_seg_lags),
        "tracking_to_seg_mean_ms": _mean(tracking_to_seg_lags),
        "tracking_to_seg_p50_ms": _p50(tracking_to_seg_lags),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[export] wrote {len(records)} track(s) -> {output_dir}", flush=True)
    logger.info("[export] wrote %s track(s) -> %s", len(records), output_dir)
    return summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _predict_overlay_box(ov: TrackOverlay, now_ms: Optional[float],
                         max_extrapolate_ms: float = 1000.0) -> tuple:
    """Extrapolate a track box from its last detection time to ``now_ms``.

    Preview shows the latest capture frame, but overlays are only refreshed when YOLO
    runs (every ``min_skip``+ frames). Without this, boxes sit where the shark *was*
    and visually lag behind the moving animal.
    """
    cx, cy, bw, bh = ov.box_xywh
    if now_ms is None or ov.timestamp_ms <= 0:
        return (cx, cy, bw, bh)
    dt = float(now_ms) - float(ov.timestamp_ms)
    if dt <= 0:
        return (cx, cy, bw, bh)
    dt = min(dt, max_extrapolate_ms)
    vx, vy = ov.velocity_xy
    return (cx + vx * dt, cy + vy * dt, bw, bh)


def _draw_overlays(frame: np.ndarray, overlays: list[TrackOverlay],
                   now_ms: Optional[float] = None) -> np.ndarray:
    out = frame.copy()
    for ov in overlays:
        cx, cy, bw, bh = _predict_overlay_box(ov, now_ms)
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        color = (0, 255, 0) if ov.length_source == "bbox" else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {ov.track_id}: {ov.confidence:.2f}"
        if ov.length_ft is not None:
            label += f" {ov.length_ft:.1f}ft"
        cv2.putText(out, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


class LiveInferenceWindow(QMainWindow):
    def __init__(self, initial_source: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("SharkEye Live Inference")
        self.resize(1100, 720)

        self.frame_buf = LatestFrameBuffer()
        self.track_state = LiveTrackState()
        self.det_queue: queue.Queue = queue.Queue(maxsize=2)
        self.seg_pending: dict[int, SegJob] = {}
        self.seg_pending_lock = threading.Lock()
        self.seg_wakeup = threading.Event()

        self.model: Optional[YOLO] = None
        self.device = None
        self.ultralytics_device: Any = "cpu"
        self.ingest_meta: dict = {}
        self.start_time_s: float = 0.0
        self.metrics: Optional[MetricsLogger] = None
        self.output_dir: Optional[Path] = None

        self.capture_worker: Optional[CaptureWorker] = None
        self.detect_worker: Optional[DetectionWorker] = None
        self.track_worker: Optional[TrackingWorker] = None
        self.seg_worker: Optional[SegmentationWorker] = None

        self._build_ui()
        if initial_source:
            self.source_edit.setText(initial_source)

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)  # ~30 UI fps
        self.preview_timer.timeout.connect(self._refresh_preview)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Video path or camera index (0)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.source_kind = QComboBox()
        self.source_kind.addItems(["Video file", "Camera index"])
        row.addWidget(QLabel("Source:"))
        row.addWidget(self.source_edit, stretch=1)
        row.addWidget(self.source_kind)
        row.addWidget(browse)
        layout.addLayout(row)

        opts = QHBoxLayout()
        self.drone_edit = QLineEdit("Mavic 2 Pro")
        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(1.0, 500.0)
        self.alt_spin.setValue(40.0)
        self.alt_spin.setSuffix(" m")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.40)
        self.min_frames_spin = QSpinBox()
        self.min_frames_spin.setRange(1, 100)
        self.min_frames_spin.setValue(5)
        self.start_time_edit = QLineEdit("0")
        self.start_time_edit.setPlaceholderText("seconds, MM:SS, or HH:MM:SS")
        for label, widget in (
            ("Drone", self.drone_edit),
            ("Altitude", self.alt_spin),
            ("Conf", self.conf_spin),
            ("Min frames", self.min_frames_spin),
            ("Start time", self.start_time_edit),
        ):
            opts.addWidget(QLabel(label))
            opts.addWidget(widget)
        opts.addStretch(1)
        layout.addLayout(opts)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.preview = QLabel("Select a source and press Start")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(480)
        self.preview.setStyleSheet("background:#111; color:#aaa;")
        layout.addWidget(self.preview, stretch=1)

        self.setStatusBar(QStatusBar())
        self.status_label = QLabel("Idle")
        self.statusBar().addWidget(self.status_label, stretch=1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*.*)",
        )
        if path:
            self.source_kind.setCurrentIndex(0)
            self.source_edit.setText(path)

    def _resolve_source(self):
        text = self.source_edit.text().strip()
        if not text:
            raise ValueError("Provide a video path or camera index")
        if self.source_kind.currentIndex() == 1 or text.isdigit():
            return int(text)
        if not os.path.isfile(text):
            raise ValueError(f"File not found: {text}")
        return text

    def _ensure_model(self):
        if self.model is not None:
            return
        self.status_label.setText("Loading YOLO…")
        QApplication.processEvents()
        import torch
        device = select_torch_device()
        self.device = device
        self.ultralytics_device = _ultralytics_device_arg(device)
        print(
            f"[live] torch device select: {device} "
            f"(cuda_available={torch.cuda.is_available()}, "
            f"mps_available={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()})",
            flush=True,
        )
        self.model = YOLO(MODEL_PATH)
        self.model.to(device)
        # Confirm weights actually landed on the chosen accelerator.
        try:
            param_device = next(self.model.model.parameters()).device
        except Exception:
            param_device = "unknown"
        print(
            f"[live] YOLO loaded; requested={device} param_device={param_device} "
            f"ultralytics_device={self.ultralytics_device}",
            flush=True,
        )
        if device.type in ("cuda", "mps") and "cpu" in str(param_device).lower():
            print(
                "[live] WARNING: accelerator requested but YOLO params are on CPU",
                flush=True,
            )
        # Warm-up on the same device used for live inference.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, classes=[0], verbose=False, device=self.ultralytics_device)
        self.status_label.setText("Warming SAM…")
        QApplication.processEvents()
        try:
            from segmentation.segmentation_model import get_sam_predictor
            get_sam_predictor()
            print(f"[live] SAM ready on {select_torch_device()}", flush=True)
        except Exception as e:
            print(f"[live] SAM warmup skipped/failed: {e}", flush=True)

    def start_pipeline(self):
        try:
            source = self._resolve_source()
            start_time_s = parse_start_timestamp(self.start_time_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Source", str(e))
            return

        self._ensure_model()

        stamp = datetime.now().strftime("%m%d%Y_%H%M%S") + "_live"
        self.output_dir = Path(get_results_dir()) / stamp
        for sub in ("frames", "masks", "detection_results", "metrics", "ingest"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsLogger(self.output_dir / "metrics" / "metrics.jsonl")

        self.status_label.setText("Preparing ingest (may rewrite to 1080p)…")
        QApplication.processEvents()
        try:
            capture_source, ingest_meta = prepare_ingest_source(
                source,
                self.output_dir / "ingest",
                status_cb=lambda msg: (self.status_label.setText(msg), QApplication.processEvents()),
            )
        except Exception as e:
            logger.exception("Ingest prepare failed")
            QMessageBox.warning(self, "Ingest", str(e))
            return
        self.ingest_meta = ingest_meta
        self.start_time_s = start_time_s

        drone = self.drone_edit.text().strip() or "Mavic 2 Pro"
        altitude = float(self.alt_spin.value())
        # FOV from the working ingest resolution (1920x1080 after rewrite).
        settings = load_drone_settings()
        fov = resolve_fov_radians(drone, INGEST_WIDTH, INGEST_HEIGHT, settings)
        if fov is None:
            # Fall back to common native drone map, then default.
            fov = resolve_fov_radians(drone, 2688, 1512, settings)
        if fov is None:
            fov = 1.274090354
            print(f"[live] no FOV for {drone!r}; using default {fov}", flush=True)

        # Reset shared state
        self.frame_buf = LatestFrameBuffer()
        self.track_state = LiveTrackState()
        self.det_queue = queue.Queue(maxsize=2)
        self.seg_pending = {}
        self.seg_wakeup.clear()

        conf = float(self.conf_spin.value())
        min_frames = int(self.min_frames_spin.value())

        self.capture_worker = CaptureWorker(
            capture_source, self.frame_buf, metrics=self.metrics,
            start_time_s=start_time_s,
        )
        self.detect_worker = DetectionWorker(
            self.model, self.frame_buf, self.det_queue, self.metrics,
            confidence=conf,
            device=self.ultralytics_device,
        )
        self.track_worker = TrackingWorker(
            self.det_queue, self.seg_pending, self.seg_pending_lock, self.seg_wakeup,
            self.track_state, self.metrics, drone, altitude, fov,
            confidence=conf, min_frames=min_frames,
        )
        self.seg_worker = SegmentationWorker(
            self.seg_pending, self.seg_pending_lock, self.seg_wakeup,
            self.track_state, self.metrics,
            altitude=altitude, fov_radians=fov,
        )

        self.capture_worker.status.connect(self._on_status)
        self.detect_worker.status.connect(self._on_status)
        self.track_worker.status.connect(self._on_status)
        self.seg_worker.status.connect(self._on_status)
        self.capture_worker.finished_capture.connect(self._on_capture_finished)

        self.capture_worker.start()
        self.detect_worker.start()
        self.track_worker.start()
        self.seg_worker.start()
        self.preview_timer.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"Running → {self.output_dir}")
        print(f"[live] results dir: {self.output_dir}", flush=True)

    def _on_status(self, msg: str):
        self.status_label.setText(msg)
        print(msg, flush=True)

    def _on_capture_finished(self):
        # Let in-flight detection/track/seg drain briefly, then stop + export.
        QTimer.singleShot(500, self.stop_pipeline)

    def _refresh_preview(self):
        snap = self.frame_buf.snapshot()
        if snap is None or snap["frame"] is None:
            overlays, active = self.track_state.get_overlays()
            lag = self.metrics.last_lag_s if self.metrics else 0.0
            self.status_label.setText(
                f"fps={self.frame_buf.fps:.1f} lag={lag:.2f}s tracks={active} "
                f"drop={self.metrics.dropped_frames if self.metrics else 0}"
            )
            return

        overlays, active = self.track_state.get_overlays()
        framed = _draw_overlays(snap["frame"], overlays, now_ms=snap["timestamp_ms"])
        preview = downscale_for_preview(framed, max_dim=960)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimg))

        lag = max(0.0, time.perf_counter() - snap["capture_wall"])
        if self.metrics:
            self.metrics.record_lag(lag, detail="preview")
        self.status_label.setText(
            f"frame={snap['frame_idx']} src={snap['source_time_s']:.1f}s "
            f"fps={snap['fps']:.1f} lag={lag:.2f}s tracks={active} "
            f"drop={self.metrics.dropped_frames if self.metrics else 0}"
        )

    def stop_pipeline(self):
        self.preview_timer.stop()
        for worker in (self.capture_worker, self.detect_worker, self.track_worker, self.seg_worker):
            if worker is not None:
                worker.stop()
        for worker in (self.capture_worker, self.detect_worker, self.track_worker, self.seg_worker):
            if worker is not None:
                worker.wait(5000)

        if self.output_dir is not None and self.metrics is not None:
            source_name = self.source_edit.text().strip() or "live"
            try:
                summary = export_live_results(
                    self.output_dir,
                    source_name,
                    self.track_state,
                    self.metrics,
                    drone_type=self.drone_edit.text().strip() or "Mavic 2 Pro",
                    altitude=float(self.alt_spin.value()),
                    ingest_meta=getattr(self, "ingest_meta", None),
                    start_time_s=getattr(self, "start_time_s", 0.0),
                )
                print(
                    "[summary] "
                    f"source={summary['source']} "
                    f"start={summary['start_time_s']:.2f}s "
                    f"orig={summary.get('original_width')}x{summary.get('original_height')} "
                    f"ingest={summary['ingest_width']}x{summary['ingest_height']} "
                    f"resized={summary.get('ingest_resized')} "
                    f"tracks={summary['tracks']} "
                    f"sam_tracks={summary['sam_tracks']} "
                    f"frames={summary['frames_written']} "
                    f"masks={summary['masks_written']} "
                    f"yolo_frames={summary['yolo_frames']} "
                    f"dropped={summary['dropped_frames']} "
                    f"track_lag_p50={summary.get('tracking_lag_p50_ms') or 0:.1f}ms "
                    f"det_to_seg_p50={summary.get('detection_to_seg_p50_ms') or 0:.1f}ms "
                    f"track_to_seg_p50={summary.get('tracking_to_seg_p50_ms') or 0:.1f}ms "
                    f"rss={summary['rss_gb']:.2f}GB "
                    f"csv={summary['csv_path']}",
                    flush=True,
                )
            except Exception:
                logger.exception("Export failed")
            self.metrics.close()

        self.capture_worker = None
        self.detect_worker = None
        self.track_worker = None
        self.seg_worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        print("[live] stopped", flush=True)

    def closeEvent(self, event):
        if self.stop_btn.isEnabled():
            self.stop_pipeline()
        try:
            release_sam_model()
        except Exception:
            pass
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SharkEye live inference")
    parser.add_argument("--source", default=None, help="Video path or camera index")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (overrides --source)")
    args, qt_args = parser.parse_known_args(argv)

    app = QApplication([sys.argv[0], *qt_args])
    initial = None
    if args.camera is not None:
        initial = str(args.camera)
    elif args.source:
        initial = args.source

    win = LiveInferenceWindow(initial_source=initial)
    if args.camera is not None:
        win.source_kind.setCurrentIndex(1)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
