import time
import logging
from datetime import datetime

logging.basicConfig(
    filename=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Timer:
    def __init__(self, label=None, logger=None, verbose: bool = False):
        self.label = label
        self.logger = logger or logging.getLogger(__name__)
        self._verbose = verbose

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        self.elapsed = end - self.start

        name = f"[{self.label}] " if self.label else ""
        msg = f"{name}Elapsed time: {self.elapsed:.6f} seconds"
        self.logger.info(msg)

        if self._verbose:
            print(msg)