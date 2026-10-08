"""otlab: optimal transport from first principles in NumPy."""
from .barycenter import barycenter
from .exact import ExactResult, emd, emd_cost
from .one_d import quantile_barycenter_1d, sliced_wasserstein, wasserstein_1d
from .sinkhorn import (SinkhornResult, sinkhorn, sinkhorn_divergence,
                       sinkhorn_unbalanced)

__version__ = "0.1.0"


def cost_matrix(x, y, p: float = 2.0):
    """Pairwise |x_i - y_j|^p for point clouds of shape (m, d) and (n, d)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) ** p


__all__ = ["emd", "emd_cost", "ExactResult", "sinkhorn", "sinkhorn_unbalanced",
           "sinkhorn_divergence", "SinkhornResult", "barycenter",
           "wasserstein_1d", "sliced_wasserstein", "quantile_barycenter_1d",
           "cost_matrix"]
