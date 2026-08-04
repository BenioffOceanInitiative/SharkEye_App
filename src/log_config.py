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

# The parent logger every module logger hangs off of. Configuring this one
# configures the whole tree via propagation.
ROOT_NAME = "sharkeye"

_configured = False


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
        datefmt="%H:%M:%S",
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
    return root


def get_logger(name):
    """Return a ``sharkeye.*`` child logger (ensures logging is configured)."""
    setup_logging()
    return logging.getLogger(name)
