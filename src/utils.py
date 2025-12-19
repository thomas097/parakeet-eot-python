import time
import logging
from datetime import datetime

logging.basicConfig(
    filename=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Timer:
    def __init__(self, label=None, logger=None):
        self.label = label
        self.logger = logger or logging.getLogger(__name__)

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        self.elapsed = end - self.start

        name = f"[{self.label}] " if self.label else ""
        self.logger.info("%sElapsed time: %.6f seconds", name, self.elapsed)