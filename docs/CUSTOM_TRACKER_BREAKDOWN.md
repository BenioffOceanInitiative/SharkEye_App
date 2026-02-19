# Custom Tracker — How It Works

The **CustomTracker** in SharkEye links object detections across video frames into **tracks**: one track per physical object (e.g. one shark). It uses **motion prediction** and the **Hungarian algorithm** to decide which detection in the current frame belongs to which existing track, and when to start a new track.

---

## 1. Role in the Pipeline

- The **YOLO model** runs on each frame and outputs bounding boxes `(x, y, w, h)` and confidence scores.
- **CustomTracker** takes those detections (per frame) and:
  - Assigns each detection to an existing track, or creates a new track.
  - Keeps a short history per track (positions, confidences, frames, lengths, velocity).
  - Does **not** delete tracks mid-video; it only adds and updates.

After the video is processed, **save_best_frames**, **save_detections_csv**, and GIF export use `custom_tracker.tracks` (all tracks). The **Settings → Confidence / Minimum Frames** values are used for the live “Shark Detected” count and for the notion of a “significant” track; they do not remove tracks from the tracker during the run.

---

## 2. Data Structures

### State

- **`self.tracks`** — Dict `track_id → track dict`. Every track that was ever created stays here.
- **`self.next_id`** — Next integer ID for a new track (incremented when creating a track).

### Per-track dict (created in `_create_new_track`, updated in `_update_track`)

| Key | Type | Purpose |
|-----|------|--------|
| `id`, `unique_id` | int | Track ID. |
| `positions` | `deque` (maxlen 100) | Recent boxes `(x, y, w, h)` in image coordinates (center x, y and width, height). |
| `confidences` | `deque` (maxlen 100) | Confidence per position. |
| `frames` | `deque` (maxlen 100) | Copy of the frame image at each detection. |
| `timestamps` | `deque` (maxlen 100) | Time in milliseconds for each detection. |
| `lengths` | `deque` (maxlen 100) | Estimated length (feet) per detection from `calculate_shark_length(bbox)`. |
| `velocity` | `np.array([vx, vy])` | Simple 2D motion from last position to current (pixels per step). |
| `best_conf`, `best_frame`, `best_timestamp`, `best_length` | scalar / array | Best detection by **confidence**. |
| `longest_conf`, `longest_frame`, `longest_timestamp`, `longest_length` | scalar / array | Best detection by **length** among high-confidence detections (conf > 0.8). |
| `frames_since_last_detection` | int | Frames since this track was last matched to a detection (used in cost). |
| `label` | str | Default `'Shark'`. |
| `track_frames` | list | Reserved; not filled in current code. |

So each track is a short sliding window of the last 100 detections plus “best” and “longest” summaries.

---

## 3. Parameters (from Settings or constructor)

- **`distance_threshold`** (default 250) — Maximum cost for assigning a detection to an existing track. If the best assignment cost is ≥ this, the detection starts a **new** track instead.
- **`min_frames`** — From Settings. Used in `_count_significant_tracks()`: a track is “significant” if it has at least this many positions and mean confidence above the threshold. Used for the “Shark Detected: Shark Count: N” message only.
- **`confidence_threshold`** — From Settings. Used in the same “significant” check and when filtering raw YOLO boxes (worker uses it as `detection_threshold` before passing detections to the tracker).
- **`fov_radians`**, **`drone_altitude`** — Set by the worker from drone settings and altitude; used later in **save_best_frames** for length-from-segmentation and for drawing/saving.

---

## 4. Main Update Loop: `update(detections, frame, timestamp)`

Called once per frame with:

- **detections** — List of `(x, y, w, h, confidence)` (center x, y; width, height; confidence).
- **frame** — Current image (BGR).
- **timestamp** — Current time in milliseconds.

Logic:

1. **First frame (no tracks)**  
   For each detection, call `_create_new_track(detection, frame, timestamp)`. Mark all as active and return.

2. **Later frames (existing tracks)**  
   - **Predict** where each track “should” be in this frame:
     - `predicted_position = last_position + velocity`
   - **Cost matrix**  
     Rows = existing tracks, columns = current detections.  
     `cost[i, j] = _calculate_cost(track_i, detection_j, predicted_position_i)`.
   - **Assignment**  
     Use `scipy.optimize.linear_sum_assignment(cost_matrix)` to get a one-to-one assignment that minimizes total cost.
   - **Apply assignment**
     - If `cost[track_idx, det_idx] < distance_threshold`: **update** that track with `detections[det_idx]` (and the current frame/timestamp).
     - Else: treat as no match for that pair — **create a new track** for that detection.
   - **Unassigned detections**  
     Any detection that was not assigned to any track creates a **new** track.

3. **Bookkeeping**
   - For every track, set `frames_since_last_detection = 0` if it was updated this frame, else increment it.
   - Recompute “significant” track count and, if it increased, print “Shark Detected: Shark Count: …”.

So: **prediction → cost matrix → Hungarian assignment → update or create track**. Tracks are never removed; they only accumulate.

---

## 5. Cost: `_calculate_cost(track, detection, predicted_position)`

```text
cost = position_cost + (frames_since_last_detection * 10)
```

- **position_cost** — Euclidean distance (in pixels) between:
  - **predicted_position** = last position + velocity,
  - and **detection center** `detection[:2]` (x, y).
- **frames_since_last_detection * 10** — Penalty for tracks that haven’t been seen recently. The longer a track is missing, the higher its cost, so new detections are less likely to be assigned to it and more likely to start a new track.

So the tracker prefers:
- Detections that are **close** to where the track was predicted to be, and
- Tracks that were **recently** matched.

---

## 6. Motion Model: `_predict_new_position(track)`

Very simple constant-velocity model:

- If the track has at least one position:
  - **predicted_position = last_position + velocity**
  - `velocity` is updated in `_update_track`: `velocity = current_center - previous_center` (in pixel space).
- If there are no positions (shouldn’t happen in normal use), returns `[0, 0]`.

So it’s a one-step predictor: “where would this track be in the next frame if it kept moving the same way?”

---

## 7. Creating a Track: `_create_new_track(detection, frame, timestamp)`

- Unpack `(x, y, w, h, confidence)` from `detection`.
- Compute `length = calculate_shark_length((x, y, w, h))` (from bbox and GSD/drone params).
- Create a new track dict with:
  - Single-element deques for positions, confidences, frames, timestamps, lengths.
  - `best_*` and `longest_*` set from this first detection.
  - `velocity = [0, 0]`, `frames_since_last_detection = 0`, `label = 'Shark'`.
- Insert into `self.tracks[self.next_id]` and increment `self.next_id`.

---

## 8. Updating a Track: `_update_track(track_id, detection, frame, timestamp)`

- Append to the track’s deques: new position, confidence, frame copy, timestamp, length.
- **Best by confidence:** if `confidence > track['best_conf']`, update `best_conf`, `best_frame`, `best_timestamp`, `best_length`.
- **Longest by length (high confidence only):** if `confidence > 0.8` and `length > track['longest_length']`, update `longest_conf`, `longest_frame`, `longest_timestamp`, `longest_length`.
- **Velocity:** if there are at least 2 positions, set  
  `velocity = current_center - previous_center` (both in pixel coordinates).

So each track keeps a rolling history and two “summary” detections: the highest-confidence one and the longest one (among conf > 0.8).

---

## 9. Significant Tracks: `_count_significant_tracks()`

Used only for the live “Shark Detected” message:

- A track counts if:
  - `len(track['positions']) >= self.min_frames`, and
  - `np.mean(track['confidences']) > self.confidence_threshold`.

So **min_frames** and **confidence_threshold** from Settings only affect this count and any UI that depends on it; they do **not** remove tracks from `self.tracks` during the video.

---

## 10. After the Video: `save_best_frames`, CSV, GIFs

- **save_best_frames** loops over **all** `self.tracks`. For each track it uses `longest_frame` / `longest_timestamp` / `longest_length` (and related fields), runs the segmentation model to refine length, draws the mask, and writes frames, bounding-box images, and masks to disk. So every track gets output, not only “significant” ones.
- **save_detections_csv** writes one row per track; in the current code, `meets_thresholds` is always `True` (the check is commented out), so all tracks are written to the CSV.
- GIFs are generated for all tracks that have frames.

So the **filtering** by min_frames and confidence is only used for the in-run “significant” count; the actual exported results include every track the tracker created.

---

## 11. Flow Summary

```text
For each frame:
  1. YOLO → list of (x, y, w, h, conf) above detection_threshold.
  2. If no tracks yet:
       → create one new track per detection.
  3. Else:
       a. For each track: predicted_pos = last_pos + velocity.
       b. Build cost matrix: cost[i,j] = distance(predicted_i, det_j) + 10 * frames_since_last_i.
       c. linear_sum_assignment(cost_matrix) → (track_indices, det_indices).
       d. For each (track_idx, det_idx):
             if cost < distance_threshold → _update_track(track, det, frame, timestamp).
             else → _create_new_track(det, frame, timestamp).
       e. For each unassigned detection → _create_new_track(...).
  4. Update frames_since_last_detection for all tracks.
  5. Optionally print “Shark Detected: Shark Count: N” if significant count went up.
```

That’s the full behavior of the custom tracker: **constant-velocity prediction + Hungarian assignment with a distance + “time since last seen” cost**, with no track deletion until the end of the run.
