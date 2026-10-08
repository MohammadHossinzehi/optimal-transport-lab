"""Fixed support Wasserstein barycenters by iterative Bregman projections.

Given histograms b_1..b_K on a shared grid with cost C, find the histogram p
minimising sum_k w_k OT_eps(p, b_k). Benamou et al. (2015) show this is a
sequence of KL projections; in scaling form

    u_k = b_k / (K v_k),  p = prod_k (K^T u_k)^{w_k},  v_k = p / (K^T u_k)

with K = exp(-C / eps). We run the identical recursion on logarithms so a
small eps does not underflow the kernel.
"""
from __future__ import annotations

import numpy as np

from .sinkhorn import logsumexp

__all__ = ["barycenter"]


def barycenter(B, C, eps, weights=None, *, max_iter: int = 5000, tol: float = 1e-10):
    """Entropic Wasserstein barycenter.

    Parameters
    ----------
    B : array, shape (n, K)
        One histogram per column, each summing to one, on a shared support.
    C : array, shape (n, n)
        Ground cost between support points.
    eps : float
        Entropic regularisation. Smaller is sharper and slower.
    weights : array, shape (K,)
        Barycentric weights (default uniform).

    Returns
    -------
    p : array, shape (n,)
    """
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    n, k = B.shape
    C = np.asarray(C, dtype=float)
    if C.shape != (n, n):
        raise ValueError("C must be square with one row per support point")
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (k,) or (w < 0).any():
        raise ValueError("weights must be non negative, one per histogram")
    w = w / w.sum()
    with np.errstate(divide="ignore"):
        logB = np.log(B / B.sum(axis=0, keepdims=True))
    logK = -C / eps                     # log kernel, never exponentiated whole
    logv = np.zeros((n, k))
    logp = np.full(n, -np.log(n))
    for _ in range(max_iter):
        # log(K v_k)_i = lse_j(logK_ij + logv_jk)
        logKv = logsumexp(logK[:, :, None] + logv[None, :, :], axis=1)
        logu = logB - logKv
        # log(K^T u_k)_j = lse_i(logK_ij + logu_ik)
        logKtu = logsumexp(logK[:, :, None] + logu[:, None, :], axis=0)
        new_logp = logKtu @ w
        logv = new_logp[:, None] - logKtu
        if np.max(np.abs(np.exp(new_logp) - np.exp(logp))) < tol:
            logp = new_logp
            break
        logp = new_logp
    p = np.exp(logp)
    return p / p.sum()
