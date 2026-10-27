"""Synthetic scan-line generator.

Genuine notes: a regular micro-print pattern (period ~8 samples) with low sensor noise.
Counterfeits: printer-dot jitter + extra noise + (usually) 1-3 localised ink-dropout bursts.
15% of counterfeits are "high quality" (no bursts, small noise) to keep the task non-trivial.
"""
import numpy as np

L = 256


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    x = np.empty((n, L))
    t = np.arange(L)
    for i, label in enumerate(y):
        amp = rng.uniform(0.8, 1.2) * 0.30
        phase = rng.uniform(0, 2 * np.pi)
        if label == 0:
            sig = 0.5 + amp * np.sin(2 * np.pi * t / 8.0 + phase)
            sig += rng.normal(0, rng.uniform(0.02, 0.05), L)
        else:
            hq = rng.random() < 0.15
            jitter = np.cumsum(rng.normal(0, 0.012 if hq else 0.03, L))
            sig = 0.5 + amp * rng.uniform(0.85, 1.0) * np.sin(2 * np.pi * t / 8.0 + phase + jitter)
            sig += rng.normal(0, rng.uniform(0.03, 0.06) if hq else rng.uniform(0.05, 0.10), L)
            if not hq:
                for _ in range(rng.integers(1, 4)):
                    start = rng.integers(0, L - 40); width = rng.integers(16, 40)
                    sig[start:start + width] = 0.5 + 0.35 * (sig[start:start + width] - 0.5) + rng.normal(0, 0.06, width)
        x[i] = sig
    return x, y.astype(np.float64)
