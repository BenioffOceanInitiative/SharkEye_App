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


def iter_sampled_frames(cap, min_skip=10, max_skip=60, empty_backoff_frames=None):
    """Yield ``(frame_index, frame)`` sampled forward through an open ``cv2.VideoCapture``.

    The stride adapts to detections: the consumer sends back a truthy value when the
    yielded frame produced a detection. On a hit the stride resets to ``min_skip``; after a
    run of empty frames spanning ``empty_backoff_frames`` it doubles toward ``max_skip``
    (matching the original inline behavior).

    Usage (the first ``send`` starts the generator)::

        sampler = iter_sampled_frames(cap)
        had_detection = None
        try:
            while True:
                idx, frame = sampler.send(had_detection)
                had_detection = process(frame)
        except StopIteration:
            pass
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if empty_backoff_frames is None:
        empty_backoff_frames = int(fps)  # ~1 second of consecutive empty frames

    frame_skip = min_skip
    consecutive_empty = 0
    frame_num = 0
    grabbed_index = -1  # index of the last frame pulled off the decoder via grab()

    while frame_num < total_frames:
        # Walk the decoder forward to frame_num, decoding nothing along the way.
        while grabbed_index < frame_num:
            if not cap.grab():
                return
            grabbed_index += 1

        ret, frame = cap.retrieve()
        if not ret:
            return

        had_detection = yield frame_num, frame

        if had_detection:
            consecutive_empty = 0
            frame_skip = min_skip
        else:
            consecutive_empty += frame_skip
            if consecutive_empty >= empty_backoff_frames:
                frame_skip = min(max_skip, frame_skip * 2)

        frame_num += frame_skip


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
    return cv2.resize(frame, (int(width * scale), int(height * scale)),
                      interpolation=cv2.INTER_AREA)
