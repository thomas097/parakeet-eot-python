import time

class Timer:
    def __init__(self, label=None):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        name = f"[{self.label}] " if self.label else ""
        print(f"{name}Elapsed time: {self.elapsed:.6f} seconds")
