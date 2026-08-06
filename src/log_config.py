"""Central logging configuration for SharkEye.

The app historically wrote diagnostics with bare ``print()`` — no timestamps, no
severity, no way to tell a routine status line from a real failure, and no way to
filter. ``setup_logging()`` installs a single consistent scheme for every entry
point (GUI, headless CLI, and the test harness):

* a timestamp + level + logger-name prefix on every line, so runs are diagnosable
  after the fact (``10:15:23 INFO  sharkeye.app: [timing] ...``);
* INFO/DEBUG go to stdout, WARNING/ERROR/CRITICAL go to stderr, so failures are
  separable from normal output;
* module loggers are children of the ``sharkeye`` parent (``sharkeye.app``,
  ``sharkeye.headless``, ``sharkeye.segment``), so one call configures them all.

Existing inline category tags (``[timing]``, ``[track]``, ``[upload]`` …) are kept
in the messages — the formatter adds time/level/name around them.

Set ``SHARKEYE_LOG_LEVEL=DEBUG`` (or WARNING, etc.) to change verbosity.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# The parent logger every module logger hangs off of. Configuring this one
# configures the whole tree via propagation.
ROOT_NAME = "sharkeye"

_configured = False

# Absolute path of the per-process log file, once a file handler is attached.
# Exposed so the UI / crash reporter can point a user at "the log to send us".
log_file_path = None


def _attach_file_handler(root, fmt, level):
    """Best-effort: also write the log to a rotating file under ``results/logs/``.

    Console output vanishes the moment the terminal closes and doesn't exist at all
    in a windowed (``console=False``) frozen build, so a durable file is the only way
    a user can send us a log after a crash. Failure here must never break logging, so
    everything is guarded — if the results dir can't be resolved we simply stay
    console-only.
    """
    global log_file_path
    try:
        from datetime import datetime

        from utility import get_results_dir

        logs_dir = os.path.join(get_results_dir(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        # One file per process (timestamp + pid), rotated if a single run gets chatty.
        name = f"sharkeye_{datetime.now().strftime('%m%d%Y_%H%M%S')}_{os.getpid()}.log"
        path = os.path.join(logs_dir, name)
        handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024,
                                      backupCount=3, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)
        log_file_path = path
        root.info("Logging to %s", path)
    except Exception as e:  # noqa: BLE001 - logging setup must never be fatal
        root.warning("Could not attach file log handler: %s", e)


class _MaxLevelFilter(logging.Filter):
    """Allow only records strictly below ``level`` (keeps WARNING+ off stdout)."""

    def __init__(self, level):
        super().__init__()
        self._level = level

    def filter(self, record):
        return record.levelno < self._level


def setup_logging(level=None):
    """Install SharkEye's logging handlers once (idempotent).

    Safe to call from every entry point and from ``MainWindow`` construction; the
    first call wins and later calls are no-ops, so we never double-log.
    """
    global _configured
    if _configured:
        return logging.getLogger(ROOT_NAME)

    if level is None:
        level = os.environ.get("SHARKEYE_LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)

    root = logging.getLogger(ROOT_NAME)
    root.setLevel(level)
    root.propagate = False  # don't also hit the global root handler / basicConfig

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # include the date: runs span days and crash logs get compared later
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    # Replace any handlers a prior partial setup or library left on our logger.
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)

    _configured = True
    # Attach the durable file handler after _configured is set so a failure in here
    # (which logs via `root`) can't recurse back into setup_logging.
    _attach_file_handler(root, fmt, level)
    return root


def get_logger(name):
    """Return a ``sharkeye.*`` child logger (ensures logging is configured)."""
    setup_logging()
    return logging.getLogger(name)


_crash_handlers_installed = False


def install_crash_handlers():
    """Route uncaught Python exceptions and hard native faults into the log.

    Without this, an uncaught exception on a non-Qt thread or a native crash
    (the ``0xc0000409`` aborts the frozen Windows build is prone to) leaves nothing
    in the log file — the user has nothing to send us. ``sys.excepthook`` captures
    Python-level uncaught exceptions with a full traceback; ``faulthandler`` dumps
    the C stack on a segfault/abort. Idempotent and best-effort. Call once per
    process after ``setup_logging()``.
    """
    global _crash_handlers_installed
    if _crash_handlers_installed:
        return
    log = logging.getLogger(ROOT_NAME)

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook

    # faulthandler dumps native tracebacks on fatal signals. Point it at the file
    # handler's stream when we have one so the C stack lands in the same log.
    try:
        import faulthandler

        stream = None
        for h in log.handlers:
            if isinstance(h, RotatingFileHandler):
                stream = h.stream
                break
        faulthandler.enable(file=stream or sys.stderr, all_threads=True)
    except Exception as e:  # noqa: BLE001 - never fatal
        log.warning("Could not enable faulthandler: %s", e)

    _crash_handlers_installed = True
