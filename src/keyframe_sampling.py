"""Keyframe-scan frame sampling — a decode-cheap alternative to grab-through sampling.

``frame_sampling.iter_sampled_frames`` advances an OpenCV ``VideoCapture`` by
``grab()``-ing through *every* frame, so on long-GOP 10-bit HEVC drone footage it
software-decodes thousands of frames it immediately discards. Decode — not YOLO —
then dominates the run (measured 89% of the loop on a 5.3K clip).

This module skips the decode of frames we never look at. It uses PyAV (libav
in-process, no external ``ffmpeg`` binary) with ``skip_frame='NONKEY'`` to decode
**only keyframes** while scanning empty water, and switches to dense sequential
decode in a short window once a detection appears — so coverage stays tight around
sharks while empty stretches cost one decoded frame per GOP (~1s here) instead of 30.
On macOS it also stacks VideoToolbox hardware decode on top (which, in PyAV, honors
``skip_frame`` — the ffmpeg CLI could not), taking scan-region decode near-free.

It matches ``iter_sampled_frames``'s generator protocol exactly — yields
``(frame_index, bgr_frame)``, receives the consumer's ``had_detection`` via
``send()``, and *returns* a stats dict (via ``StopIteration.value``) with the same
keys, so ``format_sampling_stats`` / ``format_sampling_timeline`` and the app's
``[timing]`` split work unchanged.

``try_keyframe_sampler`` is the entry point every call site should use: it honors the
``SHARKEYE_KEYFRAME_SAMPLING`` env flag, validates the file can actually be decoded
this way, and returns ``None`` on any problem so the caller transparently falls back
to grab-through — the keyframe path can never regress a run that used to work.
"""

import os

import av

# Default for the SHARKEYE_KEYFRAME_SAMPLING flag when it is unset. Now on ("1")
# by default; set SHARKEYE_KEYFRAME_SAMPLING=0 to force the legacy grab-through
# path. ``try_keyframe_sampler`` still validates each file decodes cleanly and
# falls back to grab-through on any problem, so this can never regress a run.
_DEFAULT_ENABLED = "1"

# Keep dense (every-`dense_stride`) sampling alive through a detection dropout for at
# least this long before giving up and returning to keyframe scan. This MUST stay >= the
# tracker's re-association grace (CustomTracker.reassociation_grace_ms = 2000 ms): the
# tracker re-links a shark that blinks out for up to that long, so if the sampler
# accelerated any sooner its slow keyframe-only re-acquisition would stretch the real
# detection-to-detection gap past the grace window and split one shark into multiple
# track ids. The extra 0.5 s of margin covers re-acquisition latency after the hold
# expires. Expressed in seconds (not a frame count) so it holds across fps and any
# change to dense_stride — a fixed count silently shrinks the hold when either changes.
_DENSE_HOLD_SECONDS = 2.5


def keyframe_sampling_requested():
    """True when keyframe sampling is enabled (env flag, defaulting to _DEFAULT_ENABLED)."""
    return os.environ.get("SHARKEYE_KEYFRAME_SAMPLING", _DEFAULT_ENABLED) == "1"


def _pick_hwaccel():
    """Return a HWAccel for the platform's decoder, or None to decode in software.

    VideoToolbox (macOS) decodes this 10-bit HEVC ~6x faster than software AND
    composes with ``skip_frame='NONKEY'`` (keyframe-only), so it stacks with the
    scan strategy. ``allow_software_fallback=True`` means an unsupported stream just
    falls back to software instead of raising — so this is always safe to request.
    Returns None when no hardware decoder is compiled in (e.g. a plain Linux build).
    """
    try:
        from av.codec.hwaccel import HWAccel, hwdevices_available
        available = set(hwdevices_available())
        for device in ('videotoolbox', 'cuda', 'd3d11va', 'dxva2', 'vaapi'):
            if device in available:
                return HWAccel(device_type=device, allow_software_fallback=True)
    except Exception:
        pass  # any PyAV/hwaccel API mismatch -> software decode
    return None


def try_keyframe_sampler(path, logger=None):
    """Return a ready keyframe sampler for ``path``, or ``None`` to use grab-through.

    Returns ``None`` (never raises) when keyframe sampling is not requested, or when
    the file cannot be opened / keyframe-decoded — so every caller can do::

        sampler = try_keyframe_sampler(path, logger)
        use_keyframe = sampler is not None
        if not use_keyframe:
            sampler = iter_sampled_frames(cap)

    and be guaranteed the old grab-through behavior whenever anything is off.
    """
    if not keyframe_sampling_requested():
        return None
    try:
        # Constructing the sampler opens the container and decodes one keyframe as a
        # probe, so unsupported codecs / missing video streams / HW-decode failures all
        # surface here rather than mid-run.
        return iter_keyframe_sampled_frames(path)
    except Exception as e:
        if logger is not None:
            logger.warning(f"[decode] keyframe sampling unavailable ({e!r}); using grab-through")
        return None


def iter_keyframe_sampled_frames(path, dense_stride=5, dense_hold_seconds=_DENSE_HOLD_SECONDS):
    """Open ``path`` and return the keyframe-scan sampling generator (see module docstring).

    ``dense_hold_seconds`` is how long dense sampling persists through a detection dropout
    before giving up and returning to keyframe scan; it is converted to a consecutive-empty
    sample count from the clip's fps and ``dense_stride`` inside the generator. See
    ``_DENSE_HOLD_SECONDS`` for why it must stay >= the tracker's re-association grace.

    The container is opened and a single keyframe decoded *eagerly* (before the
    generator is returned) so decode problems raise here, where ``try_keyframe_sampler``
    can catch them and fall back.
    """
    hwaccel = _pick_hwaccel()
    container = av.open(path, hwaccel=hwaccel) if hwaccel else av.open(path)
    try:
        stream = container.streams.video[0]        # raises if there is no video stream
        stream.codec_context.skip_frame = 'NONKEY'
        next(container.decode(stream))             # probe: raises if keyframe decode fails
    except Exception:
        container.close()
        raise
    return _keyframe_gen(container, stream, dense_stride, dense_hold_seconds)


def _keyframe_gen(container, stream, dense_stride, dense_hold_seconds):
    """The actual sampling generator. See ``iter_keyframe_sampled_frames``.

    Two modes, driven by the ``had_detection`` value the consumer ``send()``s back:

    * **scan** (empty water): decode keyframes only. One decoded frame per GOP.
    * **dense** (a shark is visible): from the triggering keyframe, decode forward and
      sample every ``dense_stride`` frames until dense sampling has seen no detection for
      ``dense_hold_seconds`` (a consecutive-empty count derived from fps below), then
      return to scanning past the window.

    Switching modes re-``seek()``s (changing ``skip_frame`` needs a clean decoder
    state); decode resumes from a keyframe at/*before* the target and frames before
    the intended index are dropped.
    """
    time_base = stream.time_base
    fps = float(stream.average_rate)
    total_frames = stream.frames or 0

    # Convert the wall-clock dense-hold into a consecutive-empty sample budget. Each dense
    # sample is dense_stride frames apart, so covering dense_hold_seconds of dropout takes
    # dense_hold_seconds * fps / dense_stride empty samples. Floored at 3 (the historical
    # value) so a pathologically low fps can never make dense mode bail after 1-2 misses.
    dense_empty_limit = max(3, int(round(dense_hold_seconds * fps / dense_stride)))

    def index_of(frame):
        # CFR footage: pts * time_base = seconds; * fps = exact frame index.
        return int(round(float(frame.pts * time_base) * fps))

    def seek_to(frame_index):
        # backward=True lands on the keyframe at/before the target; we then skip
        # forward to the frame we actually want.
        stats['seeks'] += 1  # late-bound: stats is defined below, seek_to is only called after
        target_ticks = int(frame_index / fps / time_base)
        container.seek(max(0, target_ticks), stream=stream, backward=True, any_frame=False)

    stats = {
        'fps': fps,
        'total_frames': total_frames,
        'min_skip': dense_stride,
        'max_skip': int(round(fps)),          # a GOP; the scan cadence
        'sampled_frames': 0,
        'baseline_skipped_frames': 0,
        'accelerated_skipped_frames': 0,
        'segments': [],
        'seeks': 0,           # container.seek() calls — each flushes+re-primes the decoder
        'mode_switches': 0,   # scan<->dense transitions (each triggers a seek)
    }

    def record(kind, start_idx, end_idx, had_det):
        """Grow the timeline segment list and attribute skipped frames."""
        skipped = max(0, end_idx - start_idx - 1)
        if kind == 'accelerated':
            stats['accelerated_skipped_frames'] += skipped
        else:
            baseline = min(skipped, dense_stride - 1)
            stats['baseline_skipped_frames'] += baseline
            stats['accelerated_skipped_frames'] += skipped - baseline
        segs = stats['segments']
        if segs and segs[-1]['kind'] == kind:
            segs[-1]['end_frame'] = end_idx
        else:
            segs.append({'kind': kind, 'start_frame': start_idx,
                         'end_frame': end_idx, 'detections': 0})
        if had_det:
            segs[-1]['detections'] += 1

    try:
        mode = 'scan'
        next_scan_index = 0        # skip keyframes already covered by a dense window
        while True:
            if mode == 'scan':
                stream.codec_context.skip_frame = 'NONKEY'
                seek_to(next_scan_index)
                switched = False
                for frame in container.decode(stream):
                    idx = index_of(frame)
                    if idx < next_scan_index:
                        continue
                    arr = frame.to_ndarray(format='bgr24')
                    stats['sampled_frames'] += 1
                    had_detection = yield idx, arr
                    # Attribute the ~GOP of undecoded frames to acceleration.
                    record('accelerated', idx, idx + int(round(fps)), had_detection)
                    if had_detection:
                        dense_start = idx
                        mode = 'dense'
                        stats['mode_switches'] += 1
                        switched = True
                        break
                if not switched:
                    break  # scan exhausted the stream
            else:  # dense
                stream.codec_context.skip_frame = 'DEFAULT'
                seek_to(dense_start)
                empties = 0
                last_idx = dense_start
                want = dense_start
                for frame in container.decode(stream):
                    idx = index_of(frame)
                    if idx < want:
                        continue
                    arr = frame.to_ndarray(format='bgr24')
                    stats['sampled_frames'] += 1
                    had_detection = yield idx, arr
                    record('full', idx, idx + dense_stride, had_detection)
                    last_idx = idx
                    empties = 0 if had_detection else empties + 1
                    if empties >= dense_empty_limit:
                        break
                    want = idx + dense_stride
                mode = 'scan'
                stats['mode_switches'] += 1
                next_scan_index = last_idx + 1
    finally:
        container.close()

    return stats
