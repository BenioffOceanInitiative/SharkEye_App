# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SharkEye is a PyQt6 desktop app (Benioff Ocean Science Laboratory) that analyzes drone footage to detect, track, and measure sharks. It runs a YOLO detector per frame, links detections into per-object tracks, segments the best frame with SAM (Segment Anything) to estimate length, and presents a Review screen for humans to correct labels and export/upload results.

## Commands

Setup and run (README):
```bash
python -m venv ocean && source ocean/bin/activate   # the venv lives in ./ocean, not ./venv
pip install -r requirements.txt
python model_weights/download_models_from_gcs.py     # needs GCS auth or falls back to a signed-URL API
python src/sharkeye_app.py                            # launch the GUI
```

Headless batch prediction (no GUI, CLI over a directory of .mp4s):
```bash
python src/headless_prediction.py --sam_model_path model_weights/sam_vit_b_01ec64.pth \
    --input_dir <dir_of_mp4s> --output_dir ./headless_predictions
```
`sharkeye_app.py` also accepts `--headless` / `--testing` flags (`--testing` forces the `minimal` Qt platform for CI/no-display runs).

Build the distributable app (PyInstaller):
```bash
pyinstaller SharkEye.spec --noconfirm      # bundles model weights + Qt/torch/cv2 data into dist/
```
`update_build_in_bucket.py` uploads a built `SharkEye_{macOS_Intel,macOS_Silicon,Windows}_*.zip` to the `sharkeye-app-build` GCS bucket. CI equivalents live in `.github/workflows/` (per-platform compile jobs).

There is **no test suite** — the `*_test.yml` workflows are build/upload smoke checks, not unit tests.

## Architecture

### The processing pipeline
Video → YOLO detection (per frame) → `CustomTracker` → post-processing (segmentation, CSV, GIF/MP4 clips) → Review UI.

- **Detection**: `MainWindow.setup_model()` loads the YOLO weights (`MODEL_PATH`) onto CUDA/MPS/CPU and runs a dummy warm-up inference so the first real video doesn't stall.
- **`CustomTracker`** (defined in *both* `sharkeye_app.py` and `headless_prediction.py`) links detections across frames using motion prediction + the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). One track = one physical object. Tracks are never deleted mid-video; `min_frames`/`confidence_threshold` only gate the "significant"/"Shark Detected" count. See `docs/CUSTOM_TRACKER_BREAKDOWN.md` for the per-track data structures and cost function.
- **Length estimation** uses Ground Sample Distance from drone altitude, sensor width, focal length, and the segmentation mask's pixel length (`calculate_shark_length` / `calculate_shark_length_from_pixel`). Constants like `DRONE_ALTITUDE_M`, `SENSOR_WIDTH_MM` are duplicated across the app and headless modules.

### Threading model (important)
The app is single-window but heavily threaded — three distinct Qt concurrency mechanisms:
- **`VideoProcessingWorker(QObject)`** runs YOLO inference + tracking on a `QThread` (moved via `moveToThread`), communicating back to `MainWindow` via `pyqtSignal`s wired in `connect_worker_signals`. Videos are processed one at a time from `video_queue` (`process_next_video`).
- **`PostProcessJob(QRunnable)`** runs on `self.postproc_pool` (a `QThreadPool`) so per-video export (GIF/MP4 clip encoding, training-frame export) happens in the background while the *next* video's inference already begins. Dispatched via `dispatch_postproc_job`.
- **`UploadThread(QThread)`** handles GCS uploads asynchronously; results come back via the `upload_finished` signal.

When editing the processing flow, respect the signal boundaries — never touch Qt widgets from a worker thread; emit a signal and update UI in the `MainWindow` slot.

### Segmentation
`src/segmentation/segmentation_model.py` wraps SAM. `SamPredictorCache` lazily loads and caches the SAM predictor (device via `_get_device()`); `release_sam_model()` frees it. SAM is the heavyweight dependency (`sam_vit_b` weight is ~375 MB).

### UI structure
`sharkeye_app.py` is a single ~5300-line file. `MainWindow` is a `QMainWindow` using a `QStackedWidget` to switch between Home (video selection), a live processing/preview view, and the Review screen. Custom widgets of note: `FramePlayer` (QLabel subclass that plays frame sequences / video / GIF for detection review), `DraggableListWidget`, and `SwitchControl`/`QComboBox` shadowing subclasses. Settings live in nested `QDialog` pages (`SettingsDialog` → `DroneSettingsPage`, `ConfidencePage`, `DetectionLabelsPage`, `AccessibilityPage`, `CloudUploadPage`, `HistoricalExperimentsPage`, etc.).

### Persistence & results
- **Settings**: `QSettings("BOSL", "SharkEye_App")` — drone/resolution→FOV map (stored as JSON), confidence threshold, min_frames, detection labels, auto-upload flag. Defaults are seeded in `MainWindow.initialize_settings()`.
- **Results**: written under `results/<MMDDYYYY_HHMMSS>/` (one folder per experiment/run), each containing `frames/`, `bounding_boxes/`, `masks/`, `detection_results/*.csv`, `tracking_gifs/`, `false_positives/`, and `experiment_note.txt`. The "Previous Experiments" history UI reads these folders back. `utility.get_results_dir()` / `resource_path()` resolve paths differently in dev vs. a frozen PyInstaller bundle (`sys._MEIPASS`, Mac `.app` Resources) — use these helpers rather than hardcoding paths.

### Cloud
Uploads go to the Google Cloud Function at `https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload`. Model weights are pulled from the `sharkeye-app-models` GCS bucket; builds are pushed to `sharkeye-app-build`.

## Conventions & gotchas

- **`src/archive/` is dead code** — an older Kivy/earlier-architecture version. Ignore it unless explicitly asked.
- **Model weights and the `ocean/` venv are gitignored and huge** — never commit them. `model_weights/*.pt` and `*.pth` are downloaded, not versioned.
- `CustomTracker` and the GSD/length constants are **intentionally duplicated** between `sharkeye_app.py` and `headless_prediction.py`; changing tracker behavior usually means editing both.
- Length/measurement math assumes a specific source resolution (`ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512`) and downscaled model input (640×640).
