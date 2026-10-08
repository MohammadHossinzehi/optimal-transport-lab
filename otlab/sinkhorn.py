"""Entropic optimal transport solved with log domain Sinkhorn iterations.

Entropic OT adds eps * KL(P | a b^T) to the transport cost. The optimal plan
has the form

    P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)

and Sinkhorn alternately picks f and g so that the row and column marginals
match. Everything here runs on the dual potentials f, g with logsumexp, so it
stays stable for eps far smaller than the cost scale, where the textbook
K = exp(-C/eps) version underflows to zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SinkhornResult", "sinkhorn", "sinkhorn_unbalanced",
           "sinkhorn_divergence", "logsumexp"]


def logsumexp(M, axis):
    """Numerically stable log(sum(exp(M))) that tolerates -inf entries."""
    mx = np.max(M, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(M - mx), axis=axis, keepdims=True)) + mx
    return np.squeeze(out, axis=axis)


@dataclass
class SinkhornResult:
    plan: np.ndarray
    f: np.ndarray
    g: np.ndarray
    cost: float          # <C, P>, the transport part only
    objective: float     # entropic dual value <a, f> + <b, g>
    iterations: int
    marginal_error: float
    converged: bool


def _prep(a, b, C):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    C = np.asarray(C, dtype=float)
    if C.shape != (a.size, b.size):
        raise ValueError(f"cost matrix has shape {C.shape}, expected {(a.size, b.size)}")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("marginals must be non negative")
    with np.errstate(divide="ignore"):
        return a, b, C, np.log(a), np.log(b)


def _plan(f, g, C, eps, loga, logb):
    return np.exp((f[:, None] + g[None, :] - C) / eps + loga[:, None] + logb[None, :])


def _iterate(a, b, C, eps, loga, logb, f, g, lam, max_iter, tol, check_every):
    """Shared Sinkhorn loop. lam = 1 is balanced, lam < 1 is the KL relaxed
    (unbalanced) update of Sejourne et al."""
    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        f = -lam * eps * logsumexp((g[None, :] - C) / eps + logb[None, :], axis=1)
        g = -lam * eps * logsumexp((f[:, None] - C) / eps + loga[:, None], axis=0)
        if it % check_every == 0 or it == max_iter:
            if lam == 1.0:
                P = _plan(f, g, C, eps, loga, logb)
                err = float(np.abs(P.sum(axis=1) - a).sum())
                if err < tol:
                    break
            else:
                # Unbalanced problems have no fixed marginals to hit; test the
                # fixed point instead.
                f_next = -lam * eps * logsumexp((g[None, :] - C) / eps + logb[None, :], axis=1)
                err = float(np.abs(f_next - f).max())
                if err < tol:
                    break
    f[~np.isfinite(loga)] = 0.0
    g[~np.isfinite(logb)] = 0.0
    return f, g, it, err


def sinkhorn(a, b, C, eps, *, max_iter: int = 10_000, tol: float = 1e-9,
             eps_scaling: float | None = 0.5, eps_start: float | None = None,
             check_every: int = 10) -> SinkhornResult:
    """Balanced entropic OT.

    With ``eps_scaling`` set, the solver starts at a large regularisation
    (default: max |C|) and shrinks it geometrically towards ``eps``, warm
    starting the potentials each time. This annealing typically cuts the
    iteration count by an order of magnitude for small eps.
    """
    a, b, C, loga, logb = _prep(a, b, C)
    if abs(a.sum() - b.sum()) > 1e-9 * max(1.0, a.sum()):
        raise ValueError("balanced Sinkhorn needs equal masses; use sinkhorn_unbalanced")
    f = np.zeros(a.size)
    g = np.zeros(b.size)
    total = 0
    if eps_scaling:
        e = eps_start if eps_start is not None else max(float(np.abs(C).max()), eps)
        while e > eps:
            f, g, k, _ = _iterate(a, b, C, e, loga, logb, f, g, 1.0,
                                  max_iter, max(tol, 1e-4), check_every)
            total += k
            e *= eps_scaling
    f, g, k, err = _iterate(a, b, C, eps, loga, logb, f, g, 1.0, max_iter, tol, check_every)
    total += k
    P = _plan(f, g, C, eps, loga, logb)
    return SinkhornResult(plan=P, f=f, g=g, cost=float((P * C).sum()),
                          objective=float(a @ f + b @ g), iterations=total,
                          marginal_error=err, converged=err < tol)


def sinkhorn_unbalanced(a, b, C, eps, rho, *, max_iter: int = 10_000,
                        tol: float = 1e-9, check_every: int = 10) -> SinkhornResult:
    """Unbalanced entropic OT with KL marginal penalties of strength rho.

    Mass may be created or destroyed at a price rho * KL, so the inputs do not
    need equal total mass. rho -> infinity recovers balanced OT; rho -> 0
    transports nothing.
    """
    a, b, C, loga, logb = _prep(a, b, C)
    lam = rho / (rho + eps)
    f, g, it, err = _iterate(a, b, C, eps, loga, logb, np.zeros(a.size),
                             np.zeros(b.size), lam, max_iter, tol, check_every)
    P = _plan(f, g, C, eps, loga, logb)
    return SinkhornResult(plan=P, f=f, g=g, cost=float((P * C).sum()),
                          objective=float(a @ f + b @ g), iterations=it,
                          marginal_error=err, converged=err < tol)


def sinkhorn_divergence(x, y, a=None, b=None, eps=0.1, p=2, **kw) -> float:
    """Debiased Sinkhorn divergence between two weighted point clouds.

        S(a, b) = OT_eps(a, b) - (OT_eps(a, a) + OT_eps(b, b)) / 2

    Unlike raw entropic OT, S(a, a) = 0, and S is non negative and
    metrises weak convergence (Feydy et al., 2019).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if x.shape[0] == 1 and x.shape[1] > 1 and y.shape[1] != x.shape[1]:
        x, y = x.T, y.T
    a = np.full(len(x), 1 / len(x)) if a is None else np.asarray(a, dtype=float)
    b = np.full(len(y), 1 / len(y)) if b is None else np.asarray(b, dtype=float)

    def cost(u, w):
        d = np.linalg.norm(u[:, None, :] - w[None, :, :], axis=-1)
        return d ** p

    ab = sinkhorn(a, b, cost(x, y), eps, **kw).objective
    aa = sinkhorn(a, a, cost(x, x), eps, **kw).objective
    bb = sinkhorn(b, b, cost(y, y), eps, **kw).objective
    return ab - 0.5 * (aa + bb)
