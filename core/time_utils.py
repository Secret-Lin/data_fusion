# core/time_utils.py
import time

_t0 = time.perf_counter()

def now():
    """Return backend-relative monotonic time in seconds."""
    return time.perf_counter() - _t0
