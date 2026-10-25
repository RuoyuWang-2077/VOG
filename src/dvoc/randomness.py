"""Independent random streams for sampling, reward dropout, and diagnostics."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_process(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)


@contextmanager
def fixed_rng(seed: int, device: torch.device):
    """Temporarily seed Python, NumPy, CPU, and this process's CUDA device."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[device.index].manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
