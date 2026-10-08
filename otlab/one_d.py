"""Closed form optimal transport on the real line, and sliced Wasserstein.

In one dimension the optimal coupling for any convex cost |x - y|^p is the
monotone (quantile) coupling, so

    W_p(mu, nu)^p = integral_0^1 |F^-1(t) - G^-1(t)|^p dt

which for discrete measures is an exact finite sum over the merged
breakpoints of the two cumulative distribution functions. No solver needed,
O((m + n) log(m + n)).
"""
from __future__ import annotations

import numpy as np

__all__ = ["wasserstein_1d", "sliced_wasserstein", "quantile_barycenter_1d"]


def _weights(x, w):
    x = np.asarray(x, dtype=float).ravel()
    w = np.full(x.size, 1.0 / x.size) if w is None else np.asarray(w, dtype=float).ravel()
    if w.shape != x.shape:
        raise ValueError("weights and support must have the same length")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non negative with positive mass")
    order = np.argsort(x, kind="stable")
    return x[order], np.cumsum(w[order]) / w.sum()


def _quantile(xs, cdf, t):
    idx = np.searchsorted(cdf, t, side="left")
    return xs[np.minimum(idx, xs.size - 1)]


def wasserstein_1d(x, y, a=None, b=None, p: float = 2.0) -> float:
    """Exact p Wasserstein distance between two weighted samples on R."""
    if p < 1:
        raise ValueError("p must be >= 1")
    xs, ca = _weights(x, a)
    ys, cb = _weights(y, b)
    t = np.unique(np.concatenate(([0.0], ca, cb, [1.0])))
    t = t[(t >= 0) & (t <= 1)]
    mids = 0.5 * (t[:-1] + t[1:])
    dt = np.diff(t)
    diff = np.abs(_quantile(xs, ca, mids) - _quantile(ys, cb, mids))
    return float(np.sum(dt * diff ** p) ** (1.0 / p))


def sliced_wasserstein(X, Y, a=None, b=None, *, n_projections: int = 200,
                       p: float = 2.0, seed=None) -> float:
    """Monte Carlo sliced Wasserstein distance in R^d.

    Projects both clouds onto random unit directions and averages the exact
    1D cost W_p^p. Cheap (no coupling is ever formed) and a true metric.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2D arrays with the same number of columns")
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_projections, X.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    px, py = X @ theta.T, Y @ theta.T
    total = sum(wasserstein_1d(px[:, k], py[:, k], a, b, p) ** p for k in range(n_projections))
    return float((total / n_projections) ** (1.0 / p))


def quantile_barycenter_1d(samples, weights=None, lambdas=None, n_quantiles: int = 512):
    """Exact W2 barycenter on R: average the quantile functions.

    Returns the barycenter's quantile function evaluated at ``n_quantiles``
    midpoints, which doubles as an equally weighted sample of it.
    """
    k = len(samples)
    weights = [None] * k if weights is None else weights
    lambdas = np.full(k, 1.0 / k) if lambdas is None else np.asarray(lambdas, dtype=float)
    lambdas = lambdas / lambdas.sum()
    t = (np.arange(n_quantiles) + 0.5) / n_quantiles
    out = np.zeros(n_quantiles)
    for lam, s, w in zip(lambdas, samples, weights):
        xs, c = _weights(s, w)
        out += lam * _quantile(xs, c, t)
    return out
