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
`sharkeye_app.py` also has an embedded headless batch path: passing **both** `--input_dir` and `--output_dir` runs `mass_prediction(...)` and never creates a `QApplication` (writes a single `output.csv` + `frames/bounding_boxes/masks/false_positives/`, a subset of the GUI outputs). With no `--input_dir` it launches the GUI; the `--testing` flag only affects that GUI path, forcing the `minimal` Qt platform for CI/no-display boot smoke-tests. There is **no `--headless` flag** — batch mode is triggered by the `--input_dir`/`--output_dir` pair. A third headless path, `src/headless_review_harness.py`, drives the *real* GUI offscreen (`QT_QPA_PLATFORM=offscreen`) end-to-end and is the only one with full pipeline parity.

Build the distributable app (PyInstaller):
```bash
pyinstaller SharkEye.spec --noconfirm      # bundles model weights + Qt/torch/cv2 data into dist/
```
`update_build_in_bucket.py` uploads a built `SharkEye_{macOS_Intel,macOS_Silicon,Windows}_*.zip` to the `sharkeye-app-build` GCS bucket. CI equivalents live in `.github/workflows/` (per-platform compile jobs).

There is **no test suite** — the `*_test.yml` workflows are build/upload smoke checks, not unit tests.

## Architecture

### The processing pipeline
Video → YOLO detection (per frame) → `CustomTracker` → post-processing (segmentation, CSV, per-track `.mp4` clips, `shark_frames/` training export) → Review UI.

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
`sharkeye_app.py` is a single ~6500-line file. `MainWindow` is a `QMainWindow` using a `QStackedWidget` to switch between Home (video selection), a live processing/preview view, and the Review screen. Custom widgets of note: `FramePlayer` (QLabel subclass that plays frame sequences / video / GIF for detection review), `DraggableListWidget`, and `SwitchControl`/`QComboBox` shadowing subclasses. Settings live in nested `QDialog` pages (`SettingsDialog` → `DroneSettingsPage`, `ConfidencePage`, `DetectionLabelsPage`, `AccessibilityPage`, `CloudUploadPage`, `HistoricalExperimentsPage`, etc.).

### Persistence & results
- **Settings**: `QSettings("BOSL", "SharkEye_App")` — drone/resolution→FOV map (stored as JSON), confidence threshold, min_frames, detection labels, auto-upload flag. Defaults are seeded in `MainWindow.initialize_settings()`.
- **Results**: written under `results/<MMDDYYYY_HHMMSS>/` (one folder per experiment/run). Per-video/per-track outputs, with where each is written:
  - `frames/<video>.mp4_<trackID>.jpg` — raw best-confidence frame per track.
  - `bounding_boxes/<video>.mp4_<trackID>.jpg` — same frame with a baked-in box + ID/conf/length label.
  - `masks/<video>.mp4_<trackID>.jpg` — SAM mask overlay, written **only for significant tracks** (sub-threshold tracks get a frame + box but no mask).
  - `detection_results/<video>.mp4.csv` — one CSV per video, one row per track (the file the Review screen edits).
  - `tracking_gifs/<video>_<trackID>.mp4` — the per-track clip. Despite the folder name these are **`.mp4` (mp4v)** files, not GIFs; GIFs only survive as a legacy *read* fallback in the Review player.
  - `shark_frames/<video>_<trackID>/` — every sampled frame at full res + a YOLO `.txt` label per frame (empty = negative) + `meta.json` + a top-level `classes.txt`. This is the **largest** output and feeds retraining/upload; it is written asynchronously by `PostProcessJob`.
  - `experiment_note.txt` — seeded with the video names, editable in Review.
  - `false_positives/` — created but **never populated** in the current pipeline (dead output, still listed in `UPLOAD_FOLDERS`).
  Frames/boxes/masks/CSV are written synchronously on the worker thread; `tracking_gifs/` and `shark_frames/` are written later on `postproc_pool`, so they can lag the CSV. The "Previous Experiments" history UI reads these folders back. `utility.get_results_dir()` / `resource_path()` resolve paths differently in dev vs. a frozen PyInstaller bundle (`sys._MEIPASS`, Mac `.app` Resources) — use these helpers rather than hardcoding paths.

### Cloud
Uploads go to the Google Cloud Function at `https://us-central1-sharkeye-329715.cloudfunctions.net/sharkeye-app-upload`. Model weights are pulled from the `sharkeye-app-models` GCS bucket; builds are pushed to `sharkeye-app-build`.

## Conventions & gotchas

- **`src/archive/` is dead code** — an older Kivy/earlier-architecture version. Ignore it unless explicitly asked.
- **Model weights and the `ocean/` venv are gitignored and huge** — never commit them. `model_weights/*.pt` and `*.pth` are downloaded, not versioned.
- `CustomTracker` and the GSD/length constants are **intentionally duplicated** between `sharkeye_app.py` and `headless_prediction.py`; changing tracker behavior usually means editing both.
- Length/measurement math assumes a specific source resolution (`ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512`) and downscaled model input (640×640).
- **Frame sampling** is shared across all three inference loops via generators with a `send(had_detection)` feedback contract. `frame_sampling.iter_sampled_frames` is the grab-through (decode-every-frame) sampler; `keyframe_sampling` decodes only keyframes over empty water and goes dense around detections (`SHARKEYE_KEYFRAME_SAMPLING`, default **on**) — on 5.3K HEVC drone footage it is ~5–10x faster than grab-through with equal recall. In keyframe mode, detection timestamps come from `frame_num/fps`, not `cap.get(POS_MSEC)` — the shared `cap` is not advanced by reads.
