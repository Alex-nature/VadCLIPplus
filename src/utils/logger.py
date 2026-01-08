import os
import sys
import time
import logging
from typing import Optional


class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        try:
            self.stream.write(message)
        except Exception:
            pass
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass


def start_logging(log_dir: str = "logs", filename: Optional[str] = None) -> str:
    """
    Start logging: redirect stdout and stderr to a timestamped log file while still printing to console.
    Returns the path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"train_{timestr}.log"

    log_path = os.path.join(log_dir, filename)
    log_f = open(log_path, "a", encoding="utf-8")

    # Replace stdout and stderr with TeeStream that writes to both console and file
    sys.stdout = TeeStream(sys.__stdout__, log_f)
    sys.stderr = TeeStream(sys.__stderr__, log_f)

    # Also set up python logging to write to the same file (optional)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.__stdout__),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    logging.info(f"Logging started. Log file: {log_path}")
    return log_path


def stop_logging():
    """Restore stdout/stderr to original and shutdown logging handlers."""
    try:
        if hasattr(sys.stdout, "stream"):
            sys.stdout.stream.flush()
        if hasattr(sys.stderr, "stream"):
            sys.stderr.stream.flush()
    except Exception:
        pass
    logging.info("Logging stopped.")
    logging.shutdown()


