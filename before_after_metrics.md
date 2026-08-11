# Before/After Metrics — Priority 1, 2, 4 changes

State of the pipeline **before** and **after** the Priority 1 (length calibration), Priority 2
(measurement-frame selection), and Priority 4 (observability + minor fixes) changes. Same
two-video session throughout:

```
cd src && ../ocean/bin/python headless_review_harness.py \
  --video ~/Downloads/DJI_0031.MP4 --video ~/Downloads/DJI_0034.MP4 \
  --drone "Air 2S" --altitude 40 --shots ../scratchpad/shots --timeout 3000
```

Both sources: 5472×3078 @ 29.97 fps, ~3330 frames, ~111 s. Device: Apple Silicon (`mps`).
**Before** = run of 2026-08-10 (new logging present, old math). **After** = run of 2026-08-11
(all Priority 1/2/4 changes). Changes applied to `sharkeye_app.py`, `keyframe_sampling.py`,
`frame_sampling.py`. The mass_prediction (`HeadlessVideoProcessor`) copy is **not** patched — see
Priority 3.

---

## TL;DR

- **Priority 1 (calibration):** bbox length dropped from **23–32 ft → 11–16 ft** (biologically
  plausible now); bbox↔SAM divergence collapsed from **2.6–4.5× → 1.3–2.0×**. Four of five
  `[length]` WARNs cleared; the one survivor (0031 #2 at exactly 2.0×) is now a *real* loose-box
  signal, not a calibration artifact.
- **Priority 2 (best frame):** the two tracks that were segmenting their **entry frame** now
  segment their highest-confidence frame. 0031 #4 moved idx0→idx10 and its SAM mask area grew
  **6 214 → 22 354 px** (it had been measuring a partial entering-edge shark); 0034 #1 moved
  from a conf-0.65 frame to its conf-0.79 peak.
- **Priority 4:** new `seeks`/`mode_switches` and `p50/p95` diagnostics confirm decode is **not**
  seek-thrashing (only 5 seeks/video) — it's genuinely expensive keyframe decode. Popup relabeled;
  the libav-collision `[env]` WARNING now fires.
- **Regression guards:** `[assoc]` census and tracker output are **identical** before/after —
  tracking was not perturbed.

---

## Priority 1 + 2 — Shark length

### `[length]` provenance — per significant track

| Track | SAM ft (before→after) | bbox ft (before→after) | ratio (before→after) | seg_frame idx/conf (before→after) |
|-------|-----------------------|------------------------|----------------------|-----------------------------------|
| DJI_0031 #1 | 9.8 → 9.8   | 29.0 → **13.0** | 3.0× → **1.3×** | idx12/0.86 → idx12/0.86 (already best) |
| DJI_0031 #2 | 5.8 → 5.8   | 26.0 → **11.6** | 4.5× → **2.0×** | idx17/0.85 → idx17/0.85 (already best) |
| DJI_0031 #3 | 12.0 → 11.6 | 31.7 → **15.2** | 2.6× → **1.3×** | idx29/0.91 → **idx32/0.91** |
| DJI_0031 #4 | 5.5 → **8.2** | 22.5 → **11.4** | 4.1× → **1.4×** | **idx0/0.78 → idx10/0.83** (mask area 6 214→22 354 px) |
| DJI_0034 #1 | 5.1 → 5.0   | 17.3 → **7.7**  | 3.4× → **1.6×** | **idx0/0.65 → idx6/0.79** (was entry frame; track peaks 0.79) |

> `SAM ft` changes only where Priority 2 moved the segmented frame (#3, #4, 0034#1). #4 is the
> headline: measuring the highest-confidence frame instead of the entry frame nearly quadrupled the
> mask area and changed the reported length 5.5→8.2 ft — a methodological correctness fix, not a
> tuning tweak. Remaining WARN: **0031 #2 (2.0×)** — a genuinely loose/diagonal box now, worth a
> human glance rather than a systematic bug.

### CSV `Longest Length` column (raw `max` of per-frame bbox lengths)

| Track | before (ft) | after (ft) |
|-------|------------:|-----------:|
| DJI_0031 #1 | 30.7 | **13.4** |
| DJI_0031 #2 | 26.0 | **11.6** |
| DJI_0031 #3 | 31.7 | **15.7** |
| DJI_0031 #4 | 23.0 | **12.7** |
| DJI_0034 #1 | 18.7 | **8.7**  |

> Was biologically implausible (23–32 ft); now single-digit-to-teens ft. This column is still a raw
> `max` (outlier-sensitive) — deciding whether to switch it to a percentile or drop it is the
> remaining open item under Priority-1 step 4 in `inference_analysis.md`.

---

## Priority 4 — Observability + minor fixes

### 4a — keyframe decode seeks / mode-switches (new: `[stats]`)

| Video | seeks (before→after) | mode_switches (before→after) |
|-------|----------------------|------------------------------|
| DJI_0031 | not logged → **5** | not logged → **4** |
| DJI_0034 | not logged → **5** | not logged → **4** |

> Only 5 seeks/video ⇒ decode cost is **not** seek-thrashing; the ~55–63 ms/f scan decode is just
> expensive 5.3K keyframe decode. Rules out a whole class of "fix the seeking" optimization.

### 4b — decode/yolo p50/p95 ms/frame (new: `[timing-dist]`)

| Video | scan-decode p50/p95 | dense-decode p50/p95 | yolo p50/p95 |
|-------|---------------------|----------------------|--------------|
| DJI_0031 | 57 / 92 | 82 / 136 | 49 / 72 |
| DJI_0034 | 60 / 82 | 68 / 99  | 47 / 71 |

> Moderate tails (dense p95 136 vs p50 82); YOLO is tight. Before: means only.

### 4c — completion popup label

| | before | after |
|--|--------|-------|
| Popup text | `Total detections: 5` | **`Sharks detected: 5`** |

### 4d — libav dual-bundle warning

| | before | after |
|--|--------|-------|
| Startup env warning | none (buried in objc noise) | **`[env] OpenCV (libavcodec 61) and PyAV (libavcodec 62) bundle different libav builds…`** |

---

## Timing (context — not a target; expect ~noise, changes add negligible compute)

| Metric | before | after |
|--------|-------:|------:|
| Batch wall clock | 59 s | 64 s |
| Batch processing(loop) | 44.8 s | 48.3 s |
| Batch segmentation | 14.1 s | 14.8 s |
| DJI_0031 loop / decode% / yolo ms/f | 26.8 s / 54% / 50 | 29.9 s / 55% / 54 |
| DJI_0034 loop / decode% / yolo ms/f | 18.0 s / 53% / 52 | 18.4 s / 52% / 52 |
| Tracks / detections | 5 / 118 | 5 / 118 |

> The ~5 s wall increase is run-to-run variance (thermals/scheduler) plus marginally more SAM work
> on the larger, better masks; the new logging itself is a handful of list appends + two
> `np.percentile` calls per video.

---

## Regression guards (should be identical — they are)

### `[assoc]` census

| Video | before | after |
|-------|--------|-------|
| DJI_0031 | dets=107 matched=103 new(first=1,unassigned=0,gate=3) created=4 sig=4 filtered=0 | **identical** |
| DJI_0034 | dets=11 matched=10 new(first=1,unassigned=0,gate=0) created=1 sig=1 filtered=0 | **identical** |

### `[track]` spatial signatures
Unchanged before/after (centers, x/y spans, time spans match to the pixel) — confirming Priority
1/2/4 did not perturb detection or tracking, only length measurement and observability. The 1:34–1:51
window remains 4 spatially-distinct sharks (§2.2 of `inference_analysis.md`).

---

## Priority 3 — De-duplication into a shared module (`src/tracking.py`)

`CustomTracker` and the length calibration were duplicated (and had drifted) across
`sharkeye_app.py` and `headless_prediction.py`. They now live once in **`src/tracking.py`**;
all three prediction paths import it. The Priority 1/2 fixes therefore apply everywhere, and
two **pre-existing** bugs in the non-GUI paths are fixed as a side effect.

### Shark length parity across all three paths (same two videos)

`Highest Confidence Length` (SAM), feet:

| Track | A: GUI/harness (per-video FOV 1.4469) | B: mass_prediction (default FOV 1.274) | C: headless CLI (default FOV 1.274) |
|-------|--------------------------------------:|---------------------------------------:|------------------------------------:|
| 0031 #1 | 9.8 | 8.2 | 8.2 |
| 0031 #2 | 5.8 | 4.9 | 4.9 |
| 0031 #3 | 11.6 | 9.7 | 9.7 |
| 0031 #4 | 8.2 | 6.9 | 6.9 |
| 0034 #1 | 5.0 | 4.2 | 4.2 |

B and C agree to the decimal; they read ~16% lower than A **only** because they have no drone
selection and fall back to the default FOV (1.274 rad) where A uses the per-video 1.4469 rad.
The bbox/SAM ratios, segmented frame index, `pixel_len`, and `mask_area` are identical across
all three — i.e. the calc fix + best-frame selection are shared, only the FOV input differs.

### What the de-dup fixed in each path

| Path | Before | After |
|------|--------|-------|
| **A. GUI/harness** | already fixed (P1/P2) | **identical** — full regression pass (every `[length]`/`[assoc]`/`[track]`/`[batch]` number unchanged) |
| **B. mass_prediction** | `Highest Confidence Length` written in **raw pixels**; `longest_frame`; ~3× bbox | **feet** (8.2/4.9/9.7/6.9/4.2), best-frame, `Segmentation Duration` intact; bbox/SAM 1.3–2.0× |
| **C. headless CLI** | own ~3×-inflated `calculate_shark_length`; `longest_frame`; SAM w/ default FOV | shared calc (bbox/SAM 1.3–2.0×), best-frame; `output.csv` in plausible feet |

### Known limitation (pre-existing, not introduced here)
- **B and C have no drone/altitude selection**, so both use the default FOV (1.274 rad) rather
  than the per-video value. Correcting this means threading drone/altitude/resolution through the
  `mass_prediction` and `headless_prediction` CLIs — a separate enhancement.
- **Path B launcher bug (found while testing):** the embedded entry uses
  `input_dir.rglob("*.mp4")` — **case-sensitive**, so real `.MP4` drone files match nothing, and
  the `if not video_paths:` guard is dead (rglob returns a generator, never falsy). Independent of
  the de-dup; worth a one-line fix (`{".mp4",".mov"}` case-insensitive, like `headless_prediction`).
- Output filenames for B/C changed to the shared `{video}.mp4_{id}.jpg` convention (C previously
  encoded length/conf in the filename). The data is unchanged; only artifact names differ.

### Files
- **New:** `src/tracking.py` (shared `CustomTracker` + length calibration + `segment_and_measure`-style save).
- `src/sharkeye_app.py`, `src/headless_prediction.py`: local copies deleted, now import from `tracking`.
