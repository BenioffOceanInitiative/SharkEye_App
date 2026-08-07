"""Sequential frame sampling shared by the GUI and headless video processors.

The processors previously seeked with ``cap.set(CAP_PROP_POS_FRAMES, n)`` before every
``cap.read()``. A random seek forces the decoder back to the nearest keyframe and
re-decodes forward on each iteration, which dominates runtime on long 4K clips. Because
sampling only ever moves *forward*, we instead ``grab()`` through the skipped frames
(decode is skipped) and ``retrieve()`` (decode) only the frame we actually keep.

This module holds the one shared implementation so a change to the sampling or detection
parsing lands in one place instead of the three copies that used to exist.
"""

import cv2


def iter_sampled_frames(cap, min_skip=5, max_skip=60, empty_backoff_frames=None,
                        max_skip_seconds=2.0):
    """Yield ``(frame_index, frame)`` sampled forward through an open ``cv2.VideoCapture``.

    The stride adapts to detections: the consumer sends back a truthy value when the
    yielded frame produced a detection. On a hit the stride resets to ``min_skip``; after a
    run of empty frames spanning ``empty_backoff_frames`` it doubles toward ``max_skip``
    (matching the original inline behavior).

    ``max_skip`` is a frame count, but the meaningful ceiling on how long an object can go
    unsampled is wall-clock, not frames — the same 60-frame stride is 2s at 30fps but 2.5s
    at 24fps. ``max_skip_seconds`` therefore imposes a wall-clock cap: the effective
    ``max_skip`` is clamped to ``round(fps * max_skip_seconds)``. This only ever *tightens*
    the stride (for low-frame-rate clips where 60 frames would exceed the time budget) and
    never loosens it, so higher-fps footage keeps its existing, denser cadence. Pass
    ``max_skip_seconds=None`` to disable the clamp and use the raw frame count.

    On normal completion the generator *returns* a stats dict (readable via
    ``StopIteration.value``) describing how the adaptive stride spent the clip — how many
    frames were sampled vs. skipped, and how much of the skipping came from the
    acceleration backoff rather than the baseline ``min_skip`` stride. See
    ``format_sampling_stats`` for turning it into a human-readable analytics line.

    Usage (the first ``send`` starts the generator)::

        sampler = iter_sampled_frames(cap)
        had_detection = None
        try:
            while True:
                idx, frame = sampler.send(had_detection)
                had_detection = process(frame)
        except StopIteration as stop:
            stats = stop.value
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if empty_backoff_frames is None:
        empty_backoff_frames = int(fps)  # ~1 second of consecutive empty frames

    # Cap the accelerated stride at a wall-clock budget (never below min_skip, never
    # looser than the caller's frame-count max_skip).
    if max_skip_seconds:
        max_skip = max(min_skip, min(max_skip, int(round(fps * max_skip_seconds))))

    frame_skip = min_skip
    consecutive_empty = 0
    frame_num = 0
    grabbed_index = -1  # index of the last frame pulled off the decoder via grab()

    stats = {
        'fps': fps,
        'total_frames': total_frames,
        'min_skip': min_skip,
        'max_skip': max_skip,
        'sampled_frames': 0,             # frames decoded + handed to inference
        'baseline_skipped_frames': 0,    # frames skipped at the resting min_skip stride
        'accelerated_skipped_frames': 0, # extra frames skipped by the backoff (stride > min_skip)
        'segments': [],                  # contiguous runs of full-rate vs. accelerated sampling
    }

    while frame_num < total_frames:
        # Walk the decoder forward to frame_num, decoding nothing along the way.
        while grabbed_index < frame_num:
            if not cap.grab():
                return stats
            grabbed_index += 1

        ret, frame = cap.retrieve()
        if not ret:
            return stats

        stats['sampled_frames'] += 1
        had_detection = yield frame_num, frame

        if had_detection:
            consecutive_empty = 0
            frame_skip = min_skip
        else:
            consecutive_empty += frame_skip
            if consecutive_empty >= empty_backoff_frames:
                frame_skip = min(max_skip, frame_skip * 2)

        # Attribute the frames this stride jumps over (bounded by the clip end) to the
        # baseline stride vs. the acceleration backoff, so callers can report how much
        # video time the acceleration actually saved.
        stride = min(frame_skip, total_frames - frame_num)
        skipped = max(0, stride - 1)  # the landing frame becomes the next sample, not a skip
        baseline = min(skipped, min_skip - 1)
        stats['baseline_skipped_frames'] += baseline
        stats['accelerated_skipped_frames'] += skipped - baseline

        # Record contiguous runs by sampling mode so callers can print a timeline (with
        # source-video timestamps) of when acceleration was engaged vs. when inference ran
        # at full rate. The stride only exceeds min_skip once the backoff has kicked in, so
        # `frame_skip > min_skip` is the truthful "acceleration is active" signal — a brief
        # empty gap that never triggered the backoff stays classified as full-rate.
        kind = 'accelerated' if frame_skip > min_skip else 'full'
        seg_end = min(frame_num + frame_skip, total_frames)
        segments = stats['segments']
        if segments and segments[-1]['kind'] == kind:
            segments[-1]['end_frame'] = seg_end
        else:
            segments.append({'kind': kind, 'start_frame': frame_num,
                             'end_frame': seg_end, 'detections': 0})
        if had_detection:
            segments[-1]['detections'] += 1

        frame_num += frame_skip

    return stats


def format_sampling_stats(video_name, infer_wall_seconds, stats):
    """Return a one-line ``[stats]`` summary of adaptive frame sampling for a video.

    ``stats`` is the dict returned by ``iter_sampled_frames`` (via ``StopIteration.value``);
    ``infer_wall_seconds`` is the real wall-clock time spent running inference on the
    sampled frames. Reports the source-video duration, the wall time and throughput of
    inference, and — the acceleration payoff — how much source-video time was skipped in
    total and specifically by the adaptive backoff.
    """
    stats = stats or {}
    fps = stats.get('fps') or 30
    total_frames = stats.get('total_frames', 0)
    sampled = stats.get('sampled_frames', 0)
    baseline_skipped = stats.get('baseline_skipped_frames', 0)
    accel_skipped = stats.get('accelerated_skipped_frames', 0)
    skipped = baseline_skipped + accel_skipped

    duration_s = total_frames / fps
    skipped_s = skipped / fps
    accel_skipped_s = accel_skipped / fps
    realtime_x = duration_s / infer_wall_seconds if infer_wall_seconds > 0 else 0.0

    # NOTE: the sampler skips *inference* on `skipped` frames, but still decodes every
    # frame to advance the capture — so `skipped` frames are NOT decode savings. The
    # per-phase decode vs. yolo split lives in the app's [timing] line; here we report
    # only the sampling coverage and how much source-video time inference actually
    # visited (the acceleration payoff).
    return (f"[stats] {video_name}: source={duration_s:.1f}s ({total_frames}f @{fps:.0f}fps) | "
            f"processed wall={infer_wall_seconds:.1f}s ({realtime_x:.1f}x realtime) | "
            f"inferred on {sampled}f, inference-skipped {skipped}f | "
            f"video time inference-skipped total={skipped_s:.1f}s, by acceleration={accel_skipped_s:.1f}s")


def _format_clock(seconds):
    """Format a number of seconds as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_sampling_timeline(video_name, stats):
    """Return a multi-line timeline of when adaptive sampling was accelerating over empty
    water vs. running full-rate inference (typically on/around detections), in source-video
    timestamps.

    ``stats`` is the dict returned by ``iter_sampled_frames``. Returns ``None`` when no
    segments were recorded (e.g. the capture failed before any frame was read).
    """
    stats = stats or {}
    fps = stats.get('fps') or 30
    segments = stats.get('segments') or []
    if not segments:
        return None

    # Collapse brief full-rate scans that found nothing (e.g. the ~1s startup before the
    # backoff can engage) into their neighbour so the timeline reads as clean alternating
    # skip/inference spans instead of littering slivers between them.
    MIN_FULL_SCAN_S = 2.0
    cleaned, pending_start = [], None
    for seg in segments:
        duration = (seg['end_frame'] - seg['start_frame']) / fps
        trivial = (seg['kind'] == 'full' and seg.get('detections', 0) == 0
                   and duration < MIN_FULL_SCAN_S)
        if trivial:
            if cleaned:
                cleaned[-1]['end_frame'] = seg['end_frame']      # extend the previous span over it
            elif pending_start is None:
                pending_start = seg['start_frame']               # leading sliver: fold into the next span
            continue
        seg = dict(seg)
        if pending_start is not None:
            seg['start_frame'] = pending_start
            pending_start = None
        cleaned.append(seg)
    if cleaned:
        segments = cleaned  # else the whole clip was one trivial scan; keep it rather than emit nothing

    lines = [f"[timeline] {video_name} (adaptive sampling):"]
    for seg in segments:
        start_s = seg['start_frame'] / fps
        end_s = seg['end_frame'] / fps
        if seg['kind'] == 'accelerated':
            label = 'accelerated skip'
        elif seg.get('detections', 0) > 0:
            label = f"shark inference ({seg['detections']} det)"
        else:
            label = 'full-rate scan'
        span = f"{_format_clock(start_s)}–{_format_clock(end_s)}"
        lines.append(f"  {span:<13} {label:<22} ({end_s - start_s:.1f}s)")
    return "\n".join(lines)


def parse_detections(results, threshold):
    """Extract ``(x, y, w, h, confidence)`` tuples above ``threshold`` from a YOLO result."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xywh = boxes.xywh.cpu()
    confidences = boxes.conf.cpu().tolist()
    return [(float(x), float(y), float(w), float(h), confidence)
            for (x, y, w, h), confidence in zip(xywh, confidences)
            if confidence > threshold]


def downscale_for_preview(frame, max_dim=960):
    """Shrink ``frame`` so its longest side is <= ``max_dim`` before RGB conversion.

    The live preview widget is far smaller than a 4K source frame, so color-converting and
    shipping the full-resolution array across the thread boundary wastes CPU and memory
    bandwidth. Returns the original array unchanged when it is already small enough.
    """
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    # INTER_LINEAR, not INTER_AREA: this is a throwaway courtesy preview a human glances
    # at, and on a 5.3K frame INTER_AREA costs ~17ms vs ~0.4ms for INTER_LINEAR (~40x).
    # Over a video that's seconds of worker time for antialiasing no one will notice.
    return cv2.resize(frame, (int(width * scale), int(height * scale)),
                      interpolation=cv2.INTER_LINEAR)
