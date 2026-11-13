"""Latency measurement for any predict(image) callable."""
import statistics
import time
from typing import Callable, Dict, List


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def measure_latency(
    predict: Callable, images: List, warmup: int = 2, repeats: int = 1
) -> Dict[str, float]:
    """Time predict() on each image. Warmup calls are not counted.

    Returns median, p95 and mean latency in milliseconds, plus throughput in
    images per second computed from the timed calls.
    """
    if not images:
        raise ValueError("need at least one image")
    for i in range(warmup):
        predict(images[i % len(images)])
    times_ms = []
    for _ in range(repeats):
        for image in images:
            start = time.perf_counter()
            predict(image)
            times_ms.append((time.perf_counter() - start) * 1000)
    total_s = sum(times_ms) / 1000
    return {
        "n_calls": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "mean_ms": statistics.mean(times_ms),
        "images_per_s": len(times_ms) / total_s if total_s > 0 else float("inf"),
    }
