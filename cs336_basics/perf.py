"""
performance.py holde Perf class which is used to test out wallclock
memory (peak RSS) and throughput

"""

import functools
import time as t
import tracemalloc


def perf(walltime=False, memory=False, enabled=False):
    """
    The performance measuring function which measures time, memory only when it
    is enabled

    Args:
        time: Whether to measure timtime
        memory: Whether to profile memory
    """
    metrics = {}

    def decorator(func):
        if not enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if walltime:
                start = t.perf_counter_ns()

            if memory:
                tracemalloc.start()
                tracemalloc.reset_peak()

            res = func(*args, **kwargs)

            if walltime:
                duration = t.perf_counter_ns() - start
                metrics["walltime_s"] = duration / 1e9

            if memory:
                current, peak = tracemalloc.get_traced_memory()
                metrics["current_mb"] = current / 1e6
                metrics["peak_mb"] = peak / 1e6
                tracemalloc.stop()

            return (metrics, res)

        return wrapper

    return decorator
