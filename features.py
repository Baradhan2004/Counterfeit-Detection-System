"""Window-level feature extraction for a 1-D banknote scan-line intensity profile.

A 256-sample profile is cut into 32 windows of 8 samples. Each window yields 4 features:
  mean, population std, mean |first difference|, zero-crossings of the mean-centred signal.
The C++ engine re-implements this exactly; tests assert numerical parity.
"""
import numpy as np

WINDOW = 8


def window_features(profile):
    """profile: (N, L) or (L,) -> (N, L//WINDOW, 4) float64."""
    p = np.atleast_2d(np.asarray(profile, dtype=np.float64))
    n, length = p.shape
    w = p[:, : (length // WINDOW) * WINDOW].reshape(n, -1, WINDOW)
    mean = w.mean(axis=2)
    std = w.std(axis=2)
    mad = np.abs(np.diff(w, axis=2)).mean(axis=2)
    c = w - mean[..., None]
    zc = (c[:, :, :-1] * c[:, :, 1:] < 0).sum(axis=2).astype(np.float64)
    return np.stack([mean, std, mad, zc], axis=2)
