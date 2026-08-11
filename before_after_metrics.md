# Before/After Metrics — Priority 1, 2, 4 changes

Tracks the measurable state of the pipeline **before** and **after** the Priority 1 (length
calibration), Priority 2 (measurement-frame selection), and Priority 4 (observability + minor
fixes) changes. Same two-video session throughout:

```
cd src && ../ocean/bin/python headless_review_harness.py \
  --video ~/Downloads/DJI_0031.MP4 --video ~/Downloads/DJI_0034.MP4 \
  --drone "Air 2S" --altitude 40 --shots ../scratchpad/shots --timeout 3000
```

Both sources: 5472×3078 @ 29.97 fps, ~3330 frames, ~111 s. Device: Apple Silicon (`mps`).
"Before" numbers are from the verified run on **2026-08-10** (new logging present, old math).

---

## Priority 1 + 2 — Shark length (the headline correctness fix)

Length is reported two ways: the **SAM** mask major-axis (trusted) and the **bbox** GSD estimate
(the CSV `Longest Length` column). Before the fix they disagreed ~2.6–4.5×; the target is
bbox≈SAM (the `[length]` WARN should go silent). Priority 2 additionally moves the segmented
frame from the max-bbox/entry frame to the **highest-confidence** frame.

### `[length]` provenance — per significant track

| Track | SAM ft | bbox ft (BEFORE) | ratio BEFORE | bbox ft (AFTER) | ratio AFTER | seg_frame BEFORE (idx/conf/t) | seg_frame AFTER (idx/conf/t) |
|-------|-------:|-----------------:|-------------:|----------------:|------------:|-------------------------------|------------------------------|
| DJI_0031 #1 | 9.8  | 29.0 | 3.0× | _TBD_ | _TBD_ | idx12 / 0.86 / 00:35 | _TBD_ |
| DJI_0031 #2 | 5.8  | 26.0 | 4.5× | _TBD_ | _TBD_ | idx17 / 0.85 / 01:37 | _TBD_ |
| DJI_0031 #3 | 12.0 | 31.7 | 2.6× | _TBD_ | _TBD_ | idx29 / 0.91 / 01:44 | _TBD_ |
| DJI_0031 #4 | 5.5  | 22.5 | 4.1× | _TBD_ | _TBD_ | idx0  / 0.78 / 01:43 | _TBD_ |
| DJI_0034 #1 | 5.1  | 17.3 | 3.4× | _TBD_ | _TBD_ | idx0  / 0.65 / 01:23 (peak 0.79) | _TBD_ |

> **BEFORE:** all five tracks trip the WARN (ratio ≥ 2×). Tracks #4 and 0034#1 segment the
> **entry frame** (`idx0`), and 0034#1 measures a conf-0.65 frame though the track peaks at 0.79.
> **AFTER target:** ratios collapse toward ~1.0–1.3, no WARN; `seg_frame` becomes the
> highest-confidence frame (higher conf, not `idx0`). SAM ft may shift slightly because SAM now
> runs on a different (better) frame.

### CSV `Longest Length` column (raw `max` of per-frame bbox lengths)

| Track | BEFORE (ft) | AFTER (ft) |
|-------|------------:|-----------:|
| DJI_0031 #1 | 30.7 | _TBD_ |
| DJI_0031 #2 | 26.0 | _TBD_ |
| DJI_0031 #3 | 31.7 | _TBD_ |
| DJI_0031 #4 | 23.0 | _TBD_ |
| DJI_0034 #1 | 18.7 | _TBD_ |

> Biologically implausible before (23–32 ft). After the calibration fix these become the
> recalibrated per-frame bbox max — expect single-digit-to-low-teens ft.

---

## Priority 4 — Observability + minor fixes

### 4a/4b — new decode/timing diagnostics (BEFORE = not logged)

| Metric | BEFORE | AFTER |
|--------|--------|-------|
| keyframe `seeks` per video | not logged | _TBD_ |
| keyframe `mode_switches` per video | not logged | _TBD_ |
| decode scan p50/p95 ms/f | not logged (mean only) | _TBD_ |
| decode dense p50/p95 ms/f | not logged (mean only) | _TBD_ |
| yolo p50/p95 ms/f | not logged (mean only) | _TBD_ |

### 4c — completion popup label

| | BEFORE | AFTER |
|--|--------|-------|
| Popup text | `Total detections: 5` (5 is the track count, not the 118 detections) | _TBD_ |

### 4d — libav dual-bundle warning

| | BEFORE | AFTER |
|--|--------|-------|
| Startup env warning | none (risk buried in objc noise) | _TBD_ |

---

## Timing (context — Priority 1/2/4 are not expected to change these materially)

| Metric | BEFORE | AFTER |
|--------|-------:|------:|
| Batch wall clock | 59 s | _TBD_ |
| Batch processing(loop) | 44.8 s | _TBD_ |
| Batch segmentation | 14.1 s | _TBD_ |
| DJI_0031 loop / decode% / yolo ms/f | 26.8 s / 54% / 50 | _TBD_ |
| DJI_0034 loop / decode% / yolo ms/f | 18.0 s / 53% / 52 | _TBD_ |
| Tracks / detections | 5 / 118 | _TBD_ |

---

## Association census (context — unchanged by Priority 1/2/4)

| Video | BEFORE (`[assoc]`) | AFTER |
|-------|--------------------|-------|
| DJI_0031 | frames_with_dets=96 dets=107 matched=103 new: first=1 unassigned=0 gate=3; created=4 sig=4 filtered=0 | _TBD_ |
| DJI_0034 | frames_with_dets=11 dets=11 matched=10 new: first=1 unassigned=0 gate=0; created=1 sig=1 filtered=0 | _TBD_ |

> Tracker association is deliberately untouched — this row is a regression guard: it should
> read identically after the changes.
