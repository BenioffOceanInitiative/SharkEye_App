# Live inference

Dedicated PyQt live-inference app (separate from the main SharkEye batch Review flow).

## Run

```bash
python src/live_inference_app.py
python src/live_inference_app.py --source path/to/video.mp4
python src/live_inference_app.py --camera 0
```

## Behavior

- Capture advances at source FPS (video paced / camera wall-clock); preview never waits on YOLO or SAM.
- YOLO uses SharkEye-style adaptive skip (`min_skip=5`, backoff); if lag grows, skip increases so playback stays real-time.
- Tracking (`CustomTracker`) and SAM segmentation run on separate workers.
- Once a track is significant, the highest-confidence frame is queued for SAM (pending job replaced if a better frame appears before SAM starts).
- Console prints `[track]`, `[seg]`, and periodic `[perf]` lines (lag, drops, timings, RSS).
- Track/segmentation latency fields are exported in the CSV and summary:
  detection-to-tracking, detection-to-segmentation, and tracking-to-segmentation.
- On Stop / end-of-video, exports under `results/<MMDDYYYY_HHMMSS_live>/`:
  - `detection_results/`
  - `frames/`
  - `masks/`
  - `metrics/metrics.jsonl` + `metrics/summary.json`
