# SharkEye Inference Analysis — DJI_0031 + DJI_0034 (single session)

**Run date:** 2026-08-10 · **Device:** Apple Silicon (`mps`) · **Sampler:** keyframe-scan (default, PyAV + VideoToolbox HW decode) · **Detector:** YOLO · **Segmentation:** SAM `sam_vit_b`
**Settings:** `confidence_threshold=0.40`, `min_frames=5` (defaults, not overridden) · **Drone:** Air 2S · **Altitude:** 40 m

Both clips were queued into **one** processing session via `headless_review_harness.py` (offscreen, real `MainWindow`). To run two videos in a single session I extended the harness's `--video` flag to be repeatable (`action="append"`) — otherwise unchanged. Reproduce with:

```bash
cd src && ../ocean/bin/python headless_review_harness.py \
  --video ~/Downloads/DJI_0031.MP4 --video ~/Downloads/DJI_0034.MP4 \
  --drone "Air 2S" --altitude 40 --shots ../scratchpad/shots --timeout 3000
```

Both sources: **5472×3078 @ 29.97 fps, ~3330 frames, ~111 s**.

---

## 1. What the current logs already tell us (this run)

The logging added in the recent hardening work is genuinely good — per-video `[timing]`, per-track discovery lines, `[stats]`, a `[timeline]`, and a `[batch]` roll-up. Headline numbers from this session:

| Video | Loop wall | Realtime× | Decode | YOLO | Tracks | Dets | Seg |
|-------|-----------|-----------|--------|------|--------|------|-----|
| DJI_0031 | 27.0 s | 4.1× | 14.7 s (55%) | 11.1 s (226f, 49 ms/f) | 4 | 107 | 9.2 s |
| DJI_0034 | 16.3 s | 6.8× | 8.3 s (51%) | 7.5 s (152f, 49 ms/f) | 1 | 11 | 2.6 s |
| **Batch** | **43.2 s loop** | — | — | — | **5** | **118** | **11.8 s** |

Batch wall clock 56 s; model + SAM load/warmup added ~10.7 s before processing (not in the 56 s). End-to-end harness time 69 s.

**Confirmed, well-characterized behaviors:**

- **Decode — not YOLO — is the bottleneck**, even with keyframe sampling (51–55% of the loop). A 5.3K keyframe decode costs ~53–56 ms/f in scan mode; dense-window decode ~66–78 ms/f. YOLO is a flat ~49 ms/f on MPS regardless of clip.
- **Adaptive sampler is doing its job**: it skipped ~88 s (0031) / ~101 s (0034) of empty-water *inference* by acceleration; the `[timeline]` shows tight dense windows exactly over the detection clusters (0031: 0:33–0:39 and 1:34–1:51; 0034: 1:23–1:32).
- **Throughput scales inversely with shark density**: 0034 (1 shark, mostly empty) hit 6.8× realtime; 0031 (4 sharks, a long dense window) only 4.1× because dense sampling decodes every 5th frame.
- **Async post-processing hides latency well**: 0031's background export+clip (7.2 s + 4.9 s = 12 s) ran *during* 0034's inference, so it cost ~0 wall time.

---

## 2. Data-quality findings surfaced by these two videos

These are the things the run *uncovered* — some are latent bugs, some are measurement-reliability concerns. Several are only visible because I cross-referenced the CSVs against the logs; the logs alone would not have surfaced them (§3 addresses that gap).

### 2.1 Bbox-derived length is *systematically* ~3× the segmentation length ⚠️ (verified) — ✅ FIXED
> **Resolved (2026-08-11):** unified the bbox estimator onto SAM's per-video ground pixel-size and
> switched it to the box diagonal. bbox length 23–32 ft → 11–16 ft; bbox/SAM divergence 2.6–4.5× →
> 1.3–2.0×. See `before_after_metrics.md`. **Resolved (follow-up):** added a canonical **`Length (ft)`**
> CSV column (precedence manual > SAM, updated when a human draws a measurement line), and demoted
> `Longest Length` from the outlier-prone `max(lengths)` to the bbox length at the best-confidence
> frame (`best_length`) — a diagnostic, not the headline number. Every consumer should read `Length (ft)`.

**Update after checking `shark_frames/*/meta.json`:** this is not a one-frame outlier — the raw bbox length is ~26–31 ft on **every** sampled frame of track 1 (SAM: 9.8 ft), ~28–31 ft across track 3 (SAM: 12 ft), and so on. The two length columns disagree by a **consistent ~3×** for every track. Root cause: `calculate_shark_length` (`sharkeye_app.py:285`) uses **only the bounding-box height** (`height * MODEL_HEIGHT/MODEL_WIDTH * GSD`), whereas SAM measures the mask's actual major axis. A loose or vertically-elongated axis-aligned box (these sharks transit top→bottom, so the box is tall) massively overestimates. The bbox values (26–31 ft) are biologically implausible for these animals; the SAM values (~5–12 ft) are plausible — so the bbox column is the wrong one, and it's the column labeled "Longest Length." **Action item #1 below.**

The per-video CSV writes two length columns from different sources:

| Track | `Longest Length` (max raw bbox GSD) | `Highest Confidence Length` (SAM) | Ratio |
|-------|------------------------------------:|----------------------------------:|------:|
| 0031 #1 | **30.7 ft** | 9.8 ft | 3.1× |
| 0031 #2 | 26.0 ft | 5.8 ft | 4.5× |
| 0031 #3 | 31.7 ft | 12.0 ft | 2.6× |
| 0031 #4 | 23.0 ft | 5.5 ft | 4.2× |
| 0034 #1 | 18.7 ft | 5.1 ft | 3.7× |

`Longest Length = max(track['lengths'])` is the max over *every per-frame bounding-box* GSD estimate (`save_detections_csv`, `sharkeye_app.py:2531`). A single bad frame — a smeared box, a shark crossing the frame edge, a merged double-detection — produces a 20–30 ft "shark." Every track here has at least one such spike. The `[track ...]` log line only prints the SAM number (`longest_length`), so **the log looks clean while a 30 ft outlier sits in the CSV a reviewer may trust.** This column is effectively noise unless outlier-filtered, and nothing flags it.

### 2.2 DJI_0031's 1:34–1:51 window is **4 distinct sharks, not fragmentation** ✅ (verified)
Initially this looked like possible fragmentation (3 IDs born in one dense span). I resolved it by extracting each track's box-center trajectory from the `shark_frames/*/frame_*.txt` YOLO labels:

| Track | Time span | x-band (px) | y sweep (px) |
|-------|-----------|-------------|--------------|
| #2 | 94.1–99.6 s | 691–776 | 145→1052 |
| #3 | 99.4–106.1 s | 502–569 | 34→1065 |
| #4 | 103.1–108.1 s | 856–876 | 111→887 |

Tracks **#3 and #4 are active simultaneously for ~3 s** (103.1–106.1 s) in **non-overlapping x-bands** (~502–569 vs ~856–876, ~300 px apart) — two objects present at once ⇒ two real sharks. #2 hands off to #3 at ~99.5 s but from opposite corners (#2 exiting bottom at x≈691, #3 entering top at x≈569). Every track holds a tight x-band while sweeping the full frame height in 3–7 s (consistent drone pan / transit). **Conclusion: the tracker behaved correctly on this clip.** The point still stands that reaching this conclusion required hand-diffing on-disk label files — §3.1's per-track spatial-signature log line would have made it a one-glance read.

### 2.3 For tracks that never exceed conf 0.80, SAM measures the *entry* frame, not the best frame — ✅ FIXED
> **Resolved (2026-08-11):** `save_best_frames` now segments the highest-confidence frame (`best_*`),
> not `longest_frame`. 0031 #4 moved idx0→idx10 (mask area 6 214→22 354 px, length 5.5→8.2 ft); 0034 #1
> moved from a conf-0.65 entry frame to its conf-0.79 peak. See `before_after_metrics.md`.

`longest_frame`/`longest_conf` are only updated when `confidence > 0.8` (`sharkeye_app.py:1708`). **DJI_0034 #1 peaks at 0.79**, so that branch never fires and `longest_frame` stays at its init value — the *first* detection (shark just entering, 01:23). `save_best_frames` then runs SAM on that first frame, so the reported 5.1 ft length and the saved mask come from the weakest, entering-edge detection, while the *highest-confidence* frame (01:25, used for the `frames/` JPG and Review) is a different, better frame. Any track whose whole life sits in 0.40–0.80 confidence has its length measured on a near-worst frame. Nothing logs which frame/conf the length was actually derived from, so this is invisible today.

### 2.4 Completion popup mislabels tracks as detections
The "Processing Complete" dialog reported **"Total detections: 5"** — but there were **118** detections; 5 is the *track* count (`finish_processing` sums `len(tracks)`, `sharkeye_app.py:3780`). The `[batch]` log line gets it right ("5 tracks, 118 detections"). Minor, user-facing only, but it's a real conflation.

### 2.5 Environment: duplicate libav bundled by `cv2` and `av`
Boot logs show `objc` warnings that `AVFFrameReceiver`/`AVFAudioReceiver` are implemented in *both* `cv2`'s `libavdevice.61` and PyAV's `libavdevice.62` ("may cause spurious casting failures and mysterious crashes"). Benign in this run, but it's a real latent-instability flag now that the keyframe sampler pulls in PyAV alongside OpenCV — worth tracking, especially for the frozen build.

---

## 3. Additional logging worth adding (ranked)

Ordered by how much each would improve our ability to catch bugs and reason about speed/quality. Each is cheap (a few INFO lines per video, or per-N-frames), and none touches Qt/thread boundaries.

### 3.1 🔴 Tracker association decisions + per-track spatial signature — *highest value* — ✅ IMPLEMENTED
`CustomTracker.update()` used to make the fragmentation-critical decisions (re-link vs. new ID, gate rejection, unassigned detection) **completely silently**. Now instrumented:

- **`[assoc]` census line per video** (`sharkeye_app.py`, after the `[timing]` line): `frames_with_dets`, `detections`, `matched`, and — the key split — `new_ids: first_frame / unassigned / gate_rejected`, plus an always-on `tracks_created / significant / filtered` census (§3.5). A run where `gate_rejected` spikes inside one shark window is fragmentation, caught immediately.
- **Per-track spatial signature** appended to each `[track T]` line: `t=<first>–<last> (dur Xs) center=(x0,y0)->(x1,y1) x_span=..px y_span=..px`. §2.2 is now a one-glance read: disjoint x-bands + simultaneous spans ⇒ distinct sharks; shared location + abutting spans ⇒ one animal split. (Computed from `positions`/`timestamps`, which are the retained deque tail, maxlen=100.)
- Counters live in `CustomTracker.assoc_stats`, incremented at each decision site in `update()`, reset in `reset()`.
- *Not yet done:* per-new-ID `cost=… gate=… elapsed=…ms` at the decision site (would further instrument `reassociation_grace_ms` / `distance_threshold` / `_GATE_GROWTH_PX_PER_S` tuning). The `[assoc]` counts are usually enough to know *whether* to dig in.

### 3.2 🔴 Length-estimation provenance + sanity — ✅ IMPLEMENTED
Each length number is now self-explaining and outliers auto-flag:

- **`[length]` line per segmented track** in `save_best_frames`: `sam=..ft bbox=..ft (bbox/sam=..x) | pixel_len=..px mask_area=..px | seg_frame idx=.. conf=.. t=..`. This directly exposes both §2.1 (the ~3× bbox/SAM divergence) and §2.3 (which frame/conf SAM measured — a track that never crosses conf 0.8 will show a low `idx`/`conf`).
- **Auto-WARN** on a degenerate mask (`area<50px` or `len<=0`) or a bbox/SAM ratio `≥2×` or `≤0.5×` (loose box or GSD/altitude mismatch).
- **`[gsd]` line once per video**: `drone=.. altitude=..m fov=..rad resolution=.. | module_GSD=..m/px (bbox estimator)` — the calibration inputs every length derives from, now echoed.
- *Not yet done:* the per-track bbox-length **distribution** (`min/median/max`) and switching the CSV's "Longest Length" from raw `max()` to a percentile — that's a behavior change (§ Priority 1 decision, below), left for you.

### 3.3 🟠 Finer performance instrumentation
- **Decode timing percentiles, not just the mean:** `scan p50/p95 ms/f`, `dense p50/p95 ms/f`. A seek after each scan↔dense switch is far pricier than a steady keyframe decode; a p95≫p50 would prove it.
- **Keyframe-sampler internals:** number of `scan↔dense` mode switches and `container.seek()` calls per video. Seeks flush the decoder (`_keyframe_gen`, `keyframe_sampling.py:154`) and are a prime suspect for the 53 ms/f scan cost; today we can't see how many happened.
- **Segmentation is bundled** into one `segmentation=Xs` number covering N tracks. Log per-track SAM wall time + device (0031 was 9.2 s / 4 tracks ≈ 2.3 s each on MPS) so SAM cost is separable from I/O.
- **Model + SAM load/warmup duration:** currently just "Model warmup complete" with no number, yet it's ~10.7 s of the run. Time it (`load=… warmup=…`).
- **Peak memory / MPS allocation** per video (one line via `torch.mps.current_allocated_memory()` / `resource.getrusage`) — the app buffers up to 100 full 5.3K frames per track in `deque(maxlen=100)`; on a dense multi-shark clip that's the real memory ceiling and it's currently unmonitored.
- **Batch throughput roll-up:** add source-seconds processed per wall-second across the batch, and total-detections and detections/track, to the `[batch]` line.

### 3.4 🟠 Confidence / detection distribution
Per dense window (or per video) log the detection-confidence histogram or `p50/p90/max`, and how many raw detections fell just under `confidence_threshold`. Only per-track peak/avg exist now, so you can't see how close a missed/faint shark came to the 0.40 gate — directly relevant to the cold-start-recall concern noted in project memory.

### 3.5 🟡 Track-lifecycle census (always, not only when >0) — ✅ IMPLEMENTED
Folded into the new `[assoc]` line: `tracks_created / significant / filtered` now prints on every video, so "no sub-threshold tracks existed" (filtered=0) is distinguishable from "the line didn't fire." The legacy conditional `Filtered out N` line is kept.

### 3.6 🟡 Fix the two labeling issues found (§2.4, §2.5)
Correct the popup's "Total detections" → "Sharks detected (tracks)", and add a startup WARNING (or dedupe the bundled dylib) for the cv2/av `libavdevice` collision so the known-risky condition is recorded rather than buried in objc noise.

---

## 4. Patch set — status

| # | Patch | Log tag added | Status |
|---|-------|---------------|--------|
| 1 | `CustomTracker.update()` association counters + per-track spatial signature | `[assoc]`, enriched `[track]` | ✅ done (§3.1) |
| 2 | `save_best_frames()` length provenance + mask sanity; GSD provenance | `[length]`, `[gsd]` | ✅ done (§3.2) |
| 3 | Always-on created/significant/filtered census | folded into `[assoc]` | ✅ done (§3.5) |
| 4 | `keyframe_sampling._keyframe_gen` seek/mode-switch counts into `stats` | `[stats]` seeks/mode_switches | ✅ done (§3.3) |
| 5 | decode/YOLO timer p50/p95 | `[timing-dist]` | ✅ done (§3.3) |
| 6 | popup label fix, dylib-collision warning | `[env]` + popup relabel | ✅ done (§3.6) |
| 7 | **Length calibration fix (Priority 1)** | quiets `[length]` WARN | ✅ done — see `before_after_metrics.md` |
| 8 | **Segment best-confidence frame (Priority 2)** | `[length] seg_frame` | ✅ done — see `before_after_metrics.md` |
| 9 | **De-duplicate into shared `tracking.py`; bring all paths to parity** | shared `CustomTracker` + `[length]` | ✅ done (Priority 3) — see `before_after_metrics.md` |

The ✅ items change **no pipeline behavior** — they only make the run legible, and #1–#2 turn §2's "I had to diff the CSV against the log to notice this" into single readable log lines. All edits are in `src/sharkeye_app.py`.

> **⚠️ Duplication caveat:** `CustomTracker` and the length constants are intentionally duplicated in `src/headless_prediction.py` (per CLAUDE.md). These patches were applied to **`sharkeye_app.py` only** (the GUI + harness path this analysis exercises). If you use the `headless_prediction.py` batch CLI, the same `[assoc]`/`[length]` instrumentation would need porting there.

## 5. Verification run (with new logging)

Re-running the same two-video session after the patches, the new lines reproduce the §2 conclusions automatically. Representative output:

```
[gsd] DJI_0031.MP4: drone='Air 2S' altitude=40.0m fov=1.4469rad resolution=5472x3078 | module_GSD=0.02946m/px (bbox estimator)
...
[length] DJI_0031.MP4_1: sam=9.8ft bbox=29.0ft (bbox/sam=3.0x) | pixel_len=264px mask_area=11766px | seg_frame idx=12 conf=0.86 t=00:35
WARNING  [length] DJI_0031.MP4_1: bbox 29.0ft vs SAM 9.8ft diverge 3.0x — check box tightness / GSD
[length] DJI_0031.MP4_4: sam=5.5ft bbox=22.5ft (bbox/sam=4.1x) | pixel_len=148px mask_area=6214px | seg_frame idx=0  conf=0.78 t=01:43
[assoc] DJI_0031.MP4: frames_with_dets=96 detections=107 matched=103 | new_ids: first_frame=1 unassigned=0 gate_rejected=3 | tracks_created=4 significant=4 filtered=0
[track 2] ... length=5.8ft  | t=01:34-01:39 (5.5s) center=(2212,412)->(1970,2999) x_span=241px y_span=2587px
[track 3] ... length=12.0ft | t=01:39-01:46 (6.7s) center=(1622,98)->(1431,3036)  x_span=191px y_span=2938px
[track 4] ... length=5.5ft  | t=01:43-01:48 (5.0s) center=(2495,316)->(2477,2528) x_span=56px  y_span=2212px
[length] DJI_0034.MP4_1: sam=5.1ft bbox=17.3ft (bbox/sam=3.4x) | ... seg_frame idx=0 conf=0.65 t=01:23   # peak was 0.79 (<0.8) -> measured entry frame
[assoc] DJI_0034.MP4: frames_with_dets=11 detections=11 matched=10 | new_ids: first_frame=1 unassigned=0 gate_rejected=0 | tracks_created=1 significant=1 filtered=0
[batch] 2 videos | 5 tracks, 118 detections, 378 frames sampled | processing(loop)=44.8s segmentation=14.1s ... | wall clock=59.00 seconds
```

**What the verification run proves, at a glance (no CSV-diffing required):**
- **§2.1** — every `[length]` line WARNs: bbox/SAM diverges **3.0–4.5×** on all five tracks. The `[gsd]` line pins a likely cause: the bbox estimator's **fixed `module_GSD=0.02946 m/px`** (computed from default constants at import) is a *different calibration* from the per-video `fov=1.4469rad` SAM uses — a calibration mismatch, not just loose boxes. **→ Priority-1 lead below.**
- **§2.2** — tracks **#3 and #4 are simultaneously active** (01:43–01:46) in x-bands ~850 px apart (≈1431–1622 vs ≈2477–2495 in 5472-wide pixels) ⇒ **distinct sharks, not fragmentation**. `[assoc]` corroborates: 4 IDs = 1 first-frame + 3 deliberate `gate_rejected` opens, `unassigned=0`.
- **§2.3** — `DJI_0034.MP4_1` shows `seg_frame idx=0 conf=0.65` while its peak was 0.79: because it never crosses 0.8, SAM measured the **entry frame** (and not even the peak-conf one). `DJI_0031.MP4_4` likewise sits at `idx=0` because its *bbox* length peaked on the entry frame.
