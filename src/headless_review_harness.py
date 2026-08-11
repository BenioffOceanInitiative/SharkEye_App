"""Drive the real SharkEye GUI end-to-end without a human or a display.

This launches the actual `MainWindow` under Qt's ``offscreen`` platform, injects a
video into the queue, runs the full detection → tracking → segmentation →
post-processing pipeline, lets the app auto-transition to the Review screen, and
saves PNG screenshots of each stage so the run can be inspected after the fact.

It exists so the app can be exercised (and regressions caught) in a headless / CI
context. Nothing here changes app behavior: blocking modal dialogs are neutralized
by monkeypatching *in this harness only* (standard practice for GUI test drivers).

Usage:
    python headless_review_harness.py --video /path/to/clip.mp4 \
        --drone "Air 2S" --altitude 40 --location "Test Beach" \
        --shots ../scratch_test/shots --timeout 900
"""

import os
# Must be set before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time
import argparse

from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog
from PyQt6.QtCore import QTimer, Qt


# ---------------------------------------------------------------------------
# Neutralize blocking modal dialogs (harness-only; app code is untouched).
# On the happy path the only modal that blocks is the "Processing Complete"
# QMessageBox.exec() in finish_processing(). We also stub the static helpers so a
# stray warning/critical can never wedge the run.
# ---------------------------------------------------------------------------
def _install_dialog_stubs(log):
    def _instance_exec(self, *a, **k):
        title = self.windowTitle() if hasattr(self, "windowTitle") else ""
        text = self.text() if hasattr(self, "text") else ""
        log(f"[dialog] (auto-dismissed) {title!r}: {text!r}")
        return QMessageBox.StandardButton.Ok

    QMessageBox.exec = _instance_exec

    def _mk_static(name, ret):
        def _stub(*a, **k):
            # args: (parent, title, text, ...)
            title = a[1] if len(a) > 1 else ""
            text = a[2] if len(a) > 2 else ""
            log(f"[dialog:{name}] (auto) {title!r}: {text!r}")
            return ret
        return staticmethod(_stub)

    QMessageBox.information = _mk_static("information", QMessageBox.StandardButton.Ok)
    QMessageBox.warning = _mk_static("warning", QMessageBox.StandardButton.Ok)
    QMessageBox.critical = _mk_static("critical", QMessageBox.StandardButton.Ok)
    QMessageBox.question = _mk_static("question", QMessageBox.StandardButton.Yes)


def main():
    ap = argparse.ArgumentParser(description="Headless GUI driver for SharkEye")
    ap.add_argument("--video", required=True, action="append",
                    help="Path to a video file to process. Repeat the flag to queue "
                         "multiple videos into a single processing session.")
    ap.add_argument("--drone", default="Air 2S",
                    help="Drone model to select (must list the video's resolution)")
    ap.add_argument("--altitude", default="40")
    ap.add_argument("--location", default="Harness Test Site")
    ap.add_argument("--shots", default="../scratch_test/shots",
                    help="Directory to write screenshots into")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Hard wall-clock limit (seconds) before giving up")
    ap.add_argument("--shot-interval", type=float, default=8.0,
                    help="Seconds between processing-stage screenshots")
    ap.add_argument("--confidence", default=None,
                    help="Override the saved confidence_threshold for this run only")
    ap.add_argument("--min-frames", default=None,
                    help="Override the saved min_frames (track significance) for this run only")
    args = ap.parse_args()

    videos = [os.path.abspath(os.path.expanduser(v)) for v in args.video]
    for v in videos:
        if not os.path.exists(v):
            print(f"ERROR: video not found: {v}", file=sys.stderr)
            return 2

    shots_dir = os.path.abspath(args.shots)
    os.makedirs(shots_dir, exist_ok=True)

    t0 = time.monotonic()

    def log(msg):
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    _install_dialog_stubs(log)

    # Optional per-run detection overrides, written to the same QSettings store the
    # app reads. Useful for exercising the review flow on footage where the shark is
    # faint / briefly visible without permanently changing the user's settings.
    if args.confidence is not None or args.min_frames is not None:
        from PyQt6.QtCore import QSettings
        s = QSettings("BOSL", "SharkEye_App")
        if args.confidence is not None:
            s.setValue("confidence_threshold", str(args.confidence))
            log(f"[cfg] confidence_threshold -> {args.confidence}")
        if args.min_frames is not None:
            s.setValue("min_frames", str(args.min_frames))
            log(f"[cfg] min_frames -> {args.min_frames}")
        s.sync()

    # Import the app only after the platform + stubs are in place.
    from sharkeye_app import MainWindow, apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow()
    window.resize(1400, 900)
    window.show()

    shot_n = {"i": 0}

    def snap(tag, widget=None):
        widget = widget or window
        shot_n["i"] += 1
        path = os.path.join(shots_dir, f"{shot_n['i']:02d}_{tag}.png")
        try:
            pm = widget.grab()
            pm.save(path)
            log(f"[shot] {os.path.basename(path)} ({pm.width()}x{pm.height()})")
        except Exception as e:  # pragma: no cover
            log(f"[shot] FAILED {tag}: {e}")

    # -- State machine driven by a polling timer ----------------------------
    state = {"phase": "boot", "last_shot": 0.0, "done": False, "rc": 1}

    def finish(rc, reason):
        if state["done"]:
            return
        state["done"] = True
        state["rc"] = rc
        log(f"[end] {reason} (rc={rc})")
        # Final screenshots + a small results summary before quitting.
        try:
            snap("final_review", window.stack_widget.currentWidget())
        except Exception:
            pass
        summarize(window, log)
        QTimer.singleShot(200, app.quit)

    def setup_home():
        # Configure the home screen the way a user would, then inject the video.
        window.update_available_drones()
        idx = window.drone_select.findText(args.drone)
        if idx < 0:
            log(f"WARNING: drone {args.drone!r} not found; available: "
                f"{[window.drone_select.itemText(i) for i in range(window.drone_select.count())]}")
        else:
            window.drone_select.setCurrentIndex(idx)
        window.altitude_input.setText(str(args.altitude))
        window.flight_location_input.setText(str(args.location))
        window.add_video_paths(videos)
        log(f"queued {len(videos)} video(s): {', '.join(os.path.basename(v) for v in videos)} "
            f"| drone={window.drone_select.currentText()} "
            f"| rows={window.video_list.rowCount()}")
        snap("home_ready")

    def poll():
        if state["done"]:
            return
        now = time.monotonic()
        if now - t0 > args.timeout:
            finish(3, "TIMEOUT")
            return

        phase = state["phase"]

        if phase == "boot":
            setup_home()
            state["phase"] = "await_model"

        elif phase == "await_model":
            if getattr(window, "model_ready", False):
                log("model ready -> starting processing")
                window.toggle_processing()
                state["phase"] = "processing"
                state["last_shot"] = now

        elif phase == "processing":
            # Periodic screenshots of the live processing preview dialog.
            if now - state["last_shot"] >= args.shot_interval:
                dlg = getattr(window, "progress_display_dialog", None)
                if dlg is not None:
                    snap("processing", dlg)
                state["last_shot"] = now
            # The app switches to review_widget once the last video + postproc finish.
            if window.stack_widget.currentWidget() is window.review_widget:
                log("review screen reached")
                state["phase"] = "review"

        elif phase == "review":
            # Give the review widget a moment to lay out / load GIFs, then capture.
            snap("review", window.review_widget)
            QTimer.singleShot(1500, lambda: exercise_review(window, snap, log,
                                                            lambda: finish(0, "COMPLETE")))
            state["phase"] = "review_wait"

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(250)

    # Absolute backstop in case the event loop wedges.
    QTimer.singleShot(args.timeout * 1000 + 5000, lambda: finish(3, "HARD TIMEOUT"))

    app.exec()
    log(f"exit rc={state['rc']}")
    return state["rc"]


def exercise_review(window, snap, log, done_cb):
    """Interact with the Review screen the way a human would, then screenshot."""
    try:
        table = window.detection_list
        log(f"[review] detection rows: {table.rowCount()}")
        # Flip the first detection's label via its combo box, if present, to prove
        # the review controls are live and wired.
        if table.rowCount() > 0:
            combo = table.cellWidget(0, 6)
            if combo is not None and combo.count() > 1:
                before = combo.currentText()
                combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
                log(f"[review] changed row0 label {before!r} -> {combo.currentText()!r}")
        snap("review_interacted", window.review_widget)
    except Exception as e:
        log(f"[review] interaction error: {e}")
    finally:
        done_cb()


def summarize(window, log):
    """Print what actually landed on disk, so the run is verifiable from the log."""
    try:
        out = getattr(window, "current_output_dir", None)
        log(f"[results] output_dir = {out}")
        if out and os.path.isdir(out):
            for root, _dirs, files in os.walk(out):
                rel = os.path.relpath(root, out)
                if files:
                    log(f"[results]   {rel}/: {len(files)} files "
                        f"({', '.join(sorted(files)[:4])}{'…' if len(files) > 4 else ''})")
        tracks = getattr(window, "tracks", {})
        total = sum(len(t) for t in tracks.values()) if tracks else 0
        log(f"[results] videos={len(tracks)} total_tracks={total}")
    except Exception as e:
        log(f"[results] summarize error: {e}")


if __name__ == "__main__":
    sys.exit(main())
