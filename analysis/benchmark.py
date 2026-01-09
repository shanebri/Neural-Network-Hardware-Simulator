from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev
import time
from typing import Sequence

import torch


@dataclass(frozen=True)
class MeasurementResult:
    device: str
    batch_size: int
    input_shape: Sequence[int]
    runs: int
    warmup: int
    mean_s: float
    median_s: float
    stdev_s: float
    min_s: float
    max_s: float
    samples_per_s: float


def _maybe_sync(device: str) -> None:
    # synchronize GPU to make sure timing excludes queued kernels
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_forward_time(
    model,
    input_shape: Sequence[int],
    batch_size: int = 1,
    device: str = "cpu",
    runs: int = 50,
    warmup: int = 10,
    use_inference_mode: bool = True,
) -> MeasurementResult:
    """
    Measure forward-pass runtime (inference) for a model.

    input_shape is the per-sample shape (excluding batch dimension).
    """
    # validate inputs before allocating any tensors
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    # move model and inputs to the target device
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, *input_shape, device=device)

    timer_samples: list[float] = []
    context = torch.inference_mode() if use_inference_mode else torch.no_grad()

    with context:
        # warmup runs to stabilize caches and kernel selection
        for _ in range(warmup):
            _ = model(x)
        _maybe_sync(device)

        # timed runs for actual measurements
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x)
            _maybe_sync(device)
            end = time.perf_counter()
            timer_samples.append(end - start)

    # summarize timings
    mean_s = mean(timer_samples)
    median_s = median(timer_samples)
    stdev_s = stdev(timer_samples) if len(timer_samples) > 1 else 0.0
    min_s = min(timer_samples)
    max_s = max(timer_samples)
    samples_per_s = batch_size / mean_s if mean_s > 0 else 0.0

    return MeasurementResult(
        device=device,
        batch_size=batch_size,
        input_shape=input_shape,
        runs=runs,
        warmup=warmup,
        mean_s=mean_s,
        median_s=median_s,
        stdev_s=stdev_s,
        min_s=min_s,
        max_s=max_s,
        samples_per_s=samples_per_s,
    )
