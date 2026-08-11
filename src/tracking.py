"""Shared shark tracking + length calibration.

Single source of truth for the ``CustomTracker`` and the ground-sample-distance /
length math that were previously duplicated (and had drifted) across
``sharkeye_app.py`` and ``headless_prediction.py``. Every prediction path — the GUI,
the offscreen harness, the embedded ``mass_prediction`` batch, and the standalone
``headless_prediction`` CLI — imports the tracker and length helpers from here so a
fix lands once instead of in N copies.

This module is intentionally PyQt-free at import time: QSettings is imported lazily
inside ``CustomTracker.__init__`` only when detection thresholds are not passed in, so
the headless CLI can use it without a hard Qt dependency.
"""

import os
import time
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from log_config import get_logger
from segmentation.segmentation_model import (
    run_prediction, find_pixel_length, calculate_shark_length_from_pixel, draw_mask,
)

logger = get_logger("sharkeye.tracking")

# Default drone field of view (radians); overwritten per-video by callers that know the
# drone/resolution (the GUI worker). ~73 degrees; matches segmentation's FOV_RADIANS.
DEFAULT_FOV_RADIANS = 1.274090354

# --- Length-calibration constants (moved verbatim from sharkeye_app) ---
DRONE_ALTITUDE_M = 40
SENSOR_WIDTH_MM = 13.2
FOCAL_LENGTH_MM = 28
MODEL_WIDTH = MODEL_HEIGHT = 640
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 2688, 1512
ASPECT_RATIO = ORIGINAL_WIDTH / ORIGINAL_HEIGHT

# --- Length / GSD helpers (moved verbatim from sharkeye_app) ---
def calculate_gsd(altitude, sensor_width, focal_length, image_width):
    """Calculate Ground Sample Distance (GSD)"""
    return (altitude * sensor_width) / (focal_length * image_width)

GSD = calculate_gsd(DRONE_ALTITUDE_M, SENSOR_WIDTH_MM, FOCAL_LENGTH_MM, MODEL_WIDTH)

def calculate_shark_length(bbox):
    """Calculate shark length in feet based on bounding box"""
    _, _, _, height = bbox
    adjusted_height = height * (MODEL_HEIGHT / MODEL_WIDTH)
    length_m = adjusted_height * GSD
    # depth_correction_factor = (1 + DRONE_ALTITUDE_M) / DRONE_ALTITUDE_M
    return length_m * 3.28084 # * depth_correction_factor  # Convert meters to feet

def calculate_bbox_area(bbox):
    """Calculate area of bbox detection"""
    _, _, width, height = bbox
    return width * height

def calculate_adjusted_shark_length(length_raw):
    """Calculate adjusted shark length in feet using correction factors"""
    asl_correction_factor = 1
    depth_correction_factor = (1 + DRONE_ALTITUDE_M)/DRONE_ALTITUDE_M
    length_adj = length_raw * asl_correction_factor * depth_correction_factor
    return length_adj


class CustomTracker:
    # Cost returned for a track whose last sighting is older than the grace period:
    # large enough to always exceed distance_threshold (so the Hungarian pairing is
    # rejected and the detection starts a new id), but finite — linear_sum_assignment
    # rejects inf/NaN.
    _UNMATCHABLE_COST = 1e9

    # The match gate (distance_threshold) grows by this many pixels per second of dropout:
    # position uncertainty and extrapolation error both increase with the gap, so a longer
    # dropout should tolerate a larger jump before the detection is ruled a different
    # object. See _match_threshold.
    _GATE_GROWTH_PX_PER_S = 60

    def __init__(self, distance_threshold=250, min_frames=None, confidence_threshold=None,
                 fov_radians=None, drone_altitude=None, sam_model_path=None):
        # Detection thresholds default to the GUI's QSettings when not passed (so the app
        # keeps auto-configuring from user settings), but callers without Qt — the headless
        # CLI — pass them explicitly and never touch QSettings. Lazy import keeps this
        # module PyQt-free at import time.
        if min_frames is None or confidence_threshold is None:
            try:
                from PyQt6.QtCore import QSettings
                _s = QSettings("BOSL", "SharkEye_App")
                if min_frames is None:
                    min_frames = _s.value("min_frames", "5")
                if confidence_threshold is None:
                    confidence_threshold = _s.value("confidence_threshold", "0.40")
            except Exception:
                pass

        self.tracks = {}
        self.next_id = 1
        self.distance_threshold = distance_threshold
        # Re-association grace period, in video time (ms). A shark's detection often
        # blinks out for a moment and resurfaces after moving a short distance; without a
        # grace window the tracker spawns a fresh id each time and fragments one animal
        # into several. If a detection reappears within this window of a track's last
        # sighting we re-link it to that track — matching against a velocity-extrapolated
        # prediction (see _predict_new_position); past the window the track is treated as
        # gone and a new id is started (see _calculate_cost). Measured in video time, not
        # frame count, because a dropout advances no frame counter — update() only runs on
        # frames that had a detection — so only the timestamp reflects how long the object
        # was actually missing. Sized to cover real shark dropouts (glint, a side-on turn,
        # a brief submersion), which routinely run past 2s — the keyframe sampler already
        # keeps looking densely for ~2.5s (see keyframe_sampling._DENSE_HOLD_SECONDS), so a
        # shorter grace here re-splits a shark the sampler was still tracking.
        self.reassociation_grace_ms = 4000
        self.min_frames = int(min_frames) if min_frames is not None else 5
        self.confidence_threshold = float(confidence_threshold) if confidence_threshold is not None else 0.4
        self.unique_sharks = 0
        self.last_reported_sharks = 0
        self.fov_radians = fov_radians if fov_radians is not None else DEFAULT_FOV_RADIANS
        self.drone_altitude = drone_altitude if drone_altitude is not None else DRONE_ALTITUDE_M
        # SAM checkpoint for length segmentation. None -> run_prediction uses its default
        # cached predictor (GUI path); the CLI passes an explicit --sam_model_path.
        self.sam_model_path = sam_model_path
        # Per-video association telemetry. update() makes the fragmentation-critical
        # decisions (re-link vs. new id, gate rejection, unassigned detection) silently;
        # these counters let the app emit an [assoc] summary so id-churn is legible from
        # the log instead of only reconstructable by diffing on-disk labels. Reset() clears
        # them; the worker constructs a fresh tracker per video so they're per-video anyway.
        self.assoc_stats = {
            'frames_with_dets': 0,   # update() calls (only fires on frames with a detection)
            'detections': 0,         # raw detections handed to the tracker
            'matched': 0,            # detections re-linked to an existing track (cost < gate)
            'new_first_frame': 0,    # ids opened by the very first detected frame
            'new_from_gate': 0,      # ids opened because the best match exceeded the gate
            'new_unassigned': 0,     # ids opened for detections the Hungarian step left unpaired
        }

    def _pixel_size_m(self, frame_width, frame_height):
        """Meters-per-pixel on the ground for THIS video's camera geometry.

        Same derivation SAM's ``calculate_shark_length_from_pixel`` uses: project the
        drone field of view onto the ground and divide by the frame's pixel width. Uses
        the per-video ``fov_radians`` / ``drone_altitude`` / actual frame size — NOT the
        stale module-level ``GSD`` constant, which was fixed at import for a 640px-wide
        image and a default FOV and over-reported length ~2.6x. Unifying the bbox
        estimator onto this calibration makes the bbox length agree with SAM.
        """
        aspect_ratio = frame_width / frame_height
        long_side = ((2 * aspect_ratio * self.drone_altitude * math.tan(self.fov_radians / 2))
                     / math.sqrt(1 + aspect_ratio ** 2))
        return long_side / frame_width

    def _bbox_length_ft(self, bbox, frame_width, frame_height):
        """Length (feet) of a detection box using the per-video ground pixel size.

        Measures the box DIAGONAL, not just its height: for a tight box around a straight
        shark at angle theta the diagonal equals the true body length at any orientation
        (w=L*sin, h=L*cos -> sqrt(w^2+h^2)=L), whereas height-only under-read diagonal
        sharks and over-read vertical ones. SAM's mask major-axis stays the gold standard
        for significant tracks; this is the coarse per-frame / pre-segmentation estimate
        and the sub-threshold fallback.
        """
        _, _, w, h = bbox
        diag_px = math.hypot(w, h)
        return diag_px * self._pixel_size_m(frame_width, frame_height) * 3.28084

    def update(self, detections, frame, timestamp):
        active_tracks = set()
        new_unique_shark = False

        self.assoc_stats['frames_with_dets'] += 1
        self.assoc_stats['detections'] += len(detections)

        if not self.tracks:
            for detection in detections:
                self._create_new_track(detection, frame, timestamp)
                active_tracks.add(self.next_id - 1)
            self.assoc_stats['new_first_frame'] += len(detections)
            new_unique_shark = True
            self.unique_sharks = 1
        else:
            predicted_positions = {track_id: self._predict_new_position(track, timestamp)
                                   for track_id, track in self.tracks.items()}

            cost_matrix = np.array([[self._calculate_cost(track, det, predicted_positions[track_id], timestamp)
                                     for det in detections]
                                    for track_id, track in self.tracks.items()])
            
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            # Snapshot the key order once; cost-matrix row indices map to it. (Previously
            # rebuilt list(self.tracks.keys()) for every matched pair — O(n^2) per frame.)
            track_ids = list(self.tracks.keys())
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                track_id = track_ids[track_idx]
                elapsed_ms = timestamp - self.tracks[track_id]['timestamps'][-1]
                if cost_matrix[track_idx, detection_idx] < self._match_threshold(elapsed_ms):
                    self._update_track(track_id, detections[detection_idx], frame, timestamp)
                    active_tracks.add(track_id)
                    self.assoc_stats['matched'] += 1
                else:
                    self._create_new_track(detections[detection_idx], frame, timestamp)
                    active_tracks.add(self.next_id - 1)
                    self.assoc_stats['new_from_gate'] += 1

            unassigned_detections = set(range(len(detections))) - set(detection_indices)
            for i in unassigned_detections:
                self._create_new_track(detections[i], frame, timestamp)
                active_tracks.add(self.next_id - 1)
            self.assoc_stats['new_unassigned'] += len(unassigned_detections)

        current_unique_sharks = self._count_significant_tracks()
        if current_unique_sharks > self.unique_sharks:
            new_unique_shark = True
            self.unique_sharks = current_unique_sharks

        for track_id in self.tracks:
            self.tracks[track_id]['frames_since_last_detection'] = 0 if track_id in active_tracks else self.tracks[track_id]['frames_since_last_detection'] + 1

        if self.unique_sharks != self.last_reported_sharks:
            logger.info("Shark detected — unique shark count: %d", self.unique_sharks)
            self.last_reported_sharks = self.unique_sharks

        return active_tracks

    def _create_new_track(self, detection, frame, timestamp):
        x, y, w, h, confidence = detection
        length = self._bbox_length_ft((x, y, w, h), frame.shape[1], frame.shape[0])
        self.tracks[self.next_id] = {
            'id': self.next_id,
            'unique_id': self.next_id,
            'positions': deque([(x, y, w, h)], maxlen=100),
            'confidences': deque([confidence], maxlen=100),
            # Each cap.retrieve() returns a fresh buffer that is never mutated in place
            # (all consumers copy before drawing), so store references rather than paying
            # for a full-frame copy per detection.
            'frames': deque([frame], maxlen=100),
            'timestamps': deque([timestamp], maxlen=100),
            'lengths': deque([length], maxlen=100),
            'best_frame': frame,
            'best_conf': confidence,
            'best_timestamp': timestamp,
            'best_length': length,
            'best_pos': (x, y, w, h),   # box at the highest-confidence frame; what SAM segments
            'segmentation_duration': 0.0,  # set when SAM runs in save_best_frames (mass_prediction CSV reads it)
            'longest_frame': frame,
            'longest_conf': confidence, 
            'longest_timestamp': timestamp,
            'longest_length': length,            
            'frames_since_last_detection': 0,
            'velocity': np.array([0, 0]),
            'label': 'Shark',
            'track_frames': []
        }
        self.next_id += 1

    def _update_track(self, track_id, detection, frame, timestamp):
        x, y, w, h, confidence = detection
        length = self._bbox_length_ft((x, y, w, h), frame.shape[1], frame.shape[0])
        track = self.tracks[track_id]
        
        # Store frame with bounding box
        # frame_with_box = frame.copy()
        # cv2.rectangle(frame_with_box, 
        #              (int(x - w/2), int(y - h/2)), 
        #              (int(x + w/2), int(y + h/2)), 
        #              (0, 255, 0), 2)
        # track['track_frames'].append(frame_with_box)
        
        track['positions'].append((x, y, w, h))
        track['confidences'].append(confidence)
        track['frames'].append(frame)
        track['timestamps'].append(timestamp)
        track['lengths'].append(length)

        if confidence > track['best_conf']:
            track['best_conf'] = confidence
            track['best_frame'] = frame  # fresh un-mutated buffer; no copy needed
            track['best_timestamp'] = timestamp
            track['best_length'] = length
            track['best_pos'] = (x, y, w, h)  # box SAM will segment (Priority 2)

        if confidence > .8 and length > track['longest_length']:
            track['longest_conf'] = confidence
            track['longest_frame'] = frame  # fresh un-mutated buffer; no copy needed
            track['longest_timestamp'] = timestamp
            track['longest_length'] = length

        if len(track['positions']) > 1:
            prev_pos = np.array(track['positions'][-2][:2])
            curr_pos = np.array([x, y])
            # Velocity in pixels-per-ms so _predict_new_position can extrapolate over the
            # real elapsed time of a dropout (frame counts don't advance while a shark is
            # undetected). Guard duplicate / zero-dt samples: keep the prior velocity
            # rather than dividing by zero.
            dt_ms = track['timestamps'][-1] - track['timestamps'][-2]
            if dt_ms > 0:
                track['velocity'] = (curr_pos - prev_pos) / dt_ms

    @staticmethod
    def _format_timestamp(milliseconds):
        """Format timestamp in MM:SS format for CSV"""
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%M:%S")

    @staticmethod
    def _format_timestamp_filename(milliseconds):
        """Format timestamp in MMSS format for filename"""
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%M%S")

    def save_best_frames(self, output_dir, video_path):
        """Save best frames for each significant track"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        images_saved = 0
        
        for track_id, track in self.tracks.items():
            if not self.is_significant_track(track):
                continue

            num_frames = len(track['positions'])
            avg_confidence = np.mean(track['confidences'])

            # Priority 2: segment the HIGHEST-CONFIDENCE frame, not longest_frame.
            # longest_frame is chosen by max bbox length gated at conf>0.8, which defaults
            # to the ENTRY frame when a track never crosses 0.8 (e.g. a faint shark that
            # peaks at 0.79) and otherwise picks the frame with the largest/noisiest box.
            # best_* is the clearest view of the animal -> best mask + most reliable length.
            seg_frame = track.get('best_frame')
            seg_conf = track.get('best_conf', 0.0)
            seg_ts = track.get('best_timestamp', 0)
            seg_bbox = track.get('best_pos') or track['positions'][0]

            if seg_frame is None:
                continue

            x, y, w, h = seg_bbox

            # SAM is the expensive post-processing step and it runs once per track. Only
            # run it on significant tracks (same criteria as _count_significant_tracks);
            # sub-threshold tracks are almost always false positives, keep their
            # bbox-estimated length, and get no mask.
            is_significant = (num_frames >= self.min_frames
                              and avg_confidence > self.confidence_threshold)
            if is_significant:
                # SAM (and draw_mask) expect RGB; seg_frame is BGR from the decoder.
                # Feeding BGR gave the model channel-swapped pixels and measurably worse
                # masks (it under-captured the shark's extent — up to ~18% shorter length),
                # so convert once and use RGB for both segmentation and the overlay.
                rgb_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
                box = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
                _seg_t0 = time.perf_counter()
                # sam_model_path None -> run_prediction uses its default cached predictor
                # (GUI); the CLI passes an explicit checkpoint.
                if self.sam_model_path:
                    mask = run_prediction(rgb_frame, box, checkpoint_path=self.sam_model_path)
                else:
                    mask = run_prediction(rgb_frame, box)
                track['segmentation_duration'] = time.perf_counter() - _seg_t0
                pixel_length = find_pixel_length(mask, draw_line=False, viz_name = f'{video_name}-viz')
                segmentation_length = calculate_shark_length_from_pixel(pixel_length,
                                                                         original_width=seg_frame.shape[1], original_height=seg_frame.shape[0],
                                                                         drone_altitude=self.drone_altitude,
                                                                         fov_radians=self.fov_radians)
                track['longest_length'] = segmentation_length

                # Length provenance + sanity. Both estimators now share the per-video
                # calibration (_bbox_length_ft uses the same ground pixel-size as SAM), so a
                # surviving bbox/SAM divergence means a genuinely loose/tight box, not the old
                # ~3x calibration gap. Log both, the pixel/mask evidence, and WHICH frame SAM
                # measured (now the highest-confidence frame, per Priority 2). WARN on a
                # degenerate mask (near-empty -> unreliable length) or a >=2x divergence.
                try:
                    bbox_length = self._bbox_length_ft((x, y, w, h), seg_frame.shape[1], seg_frame.shape[0])
                    mask_area = int(np.count_nonzero(mask))
                    try:
                        frame_idx = track['confidences'].index(seg_conf)
                    except ValueError:
                        frame_idx = -1
                    ratio = (bbox_length / segmentation_length) if segmentation_length > 0 else float('inf')
                    logger.info(
                        f"[length] {Path(video_path).name}_{track_id}: sam={segmentation_length:.1f}ft "
                        f"bbox={bbox_length:.1f}ft (bbox/sam={ratio:.1f}x) | pixel_len={pixel_length:.0f}px "
                        f"mask_area={mask_area}px | seg_frame idx={frame_idx} conf={seg_conf:.2f} "
                        f"t={CustomTracker._format_timestamp(seg_ts)}")
                    if segmentation_length <= 0 or mask_area < 50:
                        logger.warning(f"[length] {Path(video_path).name}_{track_id}: degenerate SAM mask "
                                       f"(area={mask_area}px, len={segmentation_length:.1f}ft); length unreliable")
                    elif ratio >= 2.0 or ratio <= 0.5:
                        logger.warning(f"[length] {Path(video_path).name}_{track_id}: bbox {bbox_length:.1f}ft vs "
                                       f"SAM {segmentation_length:.1f}ft diverge {ratio:.1f}x — check box tightness / GSD")
                except Exception as e:  # logging must never break the pipeline
                    logger.warning(f"[length] provenance log failed for track {track_id}: {e}")

                mask_overlay = draw_mask(mask, rgb_frame)
                track['mask_overlay'] = mask_overlay

            filename = f"{Path(video_path).name}_{track_id}.jpg"

            # Save the segmented (highest-confidence) frame — the same one SAM measured —
            # so the Review frame view, the mask overlay, and the reported length all agree.
            # The annotated "bounding_boxes/" copy is no longer written: it duplicated this
            # frame with a box reconstructable from the YOLO label + CSV, was never read back
            # by the app, and was dropped from upload.
            cv2.imwrite(os.path.join(output_dir, 'frames', filename), seg_frame)

            # Mask image only exists for segmented (significant) tracks.
            if is_significant:
                mask_path = os.path.join(output_dir, 'masks', filename)
                cv2.imwrite(mask_path, mask_overlay)

            images_saved += 1

        logger.info(f"[segmentation] saved {images_saved} track image(s)")

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1
        self.unique_sharks = 0
        for k in self.assoc_stats:
            self.assoc_stats[k] = 0

    def _predict_new_position(self, track, timestamp):
        """Predict where the track's object should be at `timestamp`.

        Velocity is stored in pixels-per-ms (see _update_track), so we extrapolate the
        last known position by the *elapsed video time* since the last detection. That is
        what lets a shark which dropped out for a moment and kept swimming re-link to its
        existing track: over a ~1s gap the prediction moves with the animal instead of
        sitting at its stale last position. Time-based (not frame count) because the
        tracker only sees frames that had a detection — a dropout advances no frame
        counter, so only the timestamp reflects how long the object was missing.
        """
        if not track['positions']:
            return np.array([0, 0])  # Default prediction if no positions available
        last_pos = np.array(track['positions'][-1][:2])
        elapsed_ms = timestamp - track['timestamps'][-1]
        return last_pos + track['velocity'] * elapsed_ms

    def _match_threshold(self, elapsed_ms):
        """Distance gate for re-linking a detection to a track, widening with the dropout.

        Near-continuous detections use the base distance_threshold; as the gap grows, both
        the animal's possible travel and the extrapolation error grow, so the gate expands
        linearly with elapsed video time. Kept well below _UNMATCHABLE_COST so a
        past-grace track (see _calculate_cost) is still rejected."""
        return self.distance_threshold + self._GATE_GROWTH_PX_PER_S * (elapsed_ms / 1000.0)

    def _calculate_cost(self, track, detection, predicted_position, timestamp):
        """Cost for the Hungarian assignment.

        Within the re-association grace period the cost is just the distance from the
        (time-extrapolated) predicted position to the detection, so a shark that briefly
        dropped out re-links to its existing id. Past the grace period the track is
        considered gone and made effectively unmatchable, so the detection starts a fresh
        id rather than resurrecting a stale track."""
        elapsed_ms = timestamp - track['timestamps'][-1]
        if elapsed_ms > self.reassociation_grace_ms:
            return self._UNMATCHABLE_COST
        return np.linalg.norm(predicted_position - np.array(detection[:2]))

    def is_significant_track(self, track):
        """Return True if a track meets the confidence and minimum-frame settings."""
        return (len(track['positions']) >= self.min_frames
                and np.mean(track['confidences']) > self.confidence_threshold)

    def get_significant_tracks(self):
        """Return only tracks that meet the confidence and minimum-frame settings."""
        return {track_id: track for track_id, track in self.tracks.items()
                if self.is_significant_track(track)}

    def _count_significant_tracks(self):
        """Count tracks that meet the criteria for being a significant detection"""
        return sum(1 for track in self.tracks.values() if self.is_significant_track(track))
