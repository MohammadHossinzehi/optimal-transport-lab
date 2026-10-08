"""Exact discrete optimal transport with the transportation simplex.

The transportation problem

    minimise  sum_ij C_ij P_ij
    subject   P 1 = a,  P^T 1 = b,  P >= 0

is a linear program whose basic feasible solutions are spanning trees of the
complete bipartite graph between the m sources and n sinks (m + n - 1 basic
cells). This module keeps that tree explicitly, which is the trick that makes
the method robust: degenerate basic cells carrying zero flow stay in the basis,
so the tree never falls apart and the dual potentials are always well defined.

One pivot does the following:

1. Solve u_i + v_j = C_ij on the basic cells by walking the tree (u_0 = 0).
2. Price every cell: r_ij = C_ij - u_i - v_j. If min r >= -tol the current
   plan is optimal (complementary slackness holds and u, v are dual feasible).
3. Add the entering cell, which closes exactly one cycle in the tree. Push
   theta units around that cycle with alternating signs, where theta is the
   smallest flow on a "minus" cell; that cell leaves the basis.

The entering rule is Dantzig's (most negative reduced cost). After a long run
of degenerate pivots it switches to Bland's lowest index rule, which is the
classic guard against cycling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = ["ExactResult", "emd", "emd_cost"]


@dataclass
class ExactResult:
    plan: np.ndarray        # (m, n) optimal transport plan
    cost: float             # <C, plan>
    u: np.ndarray           # row potentials
    v: np.ndarray           # column potentials
    iterations: int         # number of simplex pivots
    basis: list             # list of (i, j) basic cells, length m + n - 1


def _check_inputs(a, b, C, tol):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    C = np.asarray(C, dtype=float)
    if C.shape != (a.size, b.size):
        raise ValueError(f"cost matrix has shape {C.shape}, expected {(a.size, b.size)}")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("marginals must be non negative")
    if not np.isfinite(C).all():
        raise ValueError("cost matrix must be finite")
    sa, sb = a.sum(), b.sum()
    if abs(sa - sb) > tol * max(1.0, sa, sb):
        raise ValueError(f"marginals must have equal mass (got {sa} and {sb}); "
                         "use otlab.sinkhorn_unbalanced for unequal masses")
    # Remove floating point dust so the north west corner closes exactly.
    b = b * (sa / sb) if sb > 0 else b
    return a, b, C


def _north_west_corner(a, b):
    """Initial basic feasible solution with exactly m + n - 1 basic cells."""
    m, n = a.size, b.size
    ra, rb = a.copy(), b.copy()
    flow = {}
    i = j = 0
    while True:
        q = min(ra[i], rb[j])
        flow[(i, j)] = q
        ra[i] -= q
        rb[j] -= q
        if i == m - 1 and j == n - 1:
            break
        # Advance exactly one index per step so the basis stays a tree,
        # even when a row and a column are exhausted simultaneously.
        if j == n - 1 or (i < m - 1 and ra[i] <= rb[j]):
            i += 1
        else:
            j += 1
    return flow


def _potentials(flow, C, m, n):
    """Solve u_i + v_j = C_ij on the basis tree by breadth first search."""
    adj = [[] for _ in range(m + n)]
    for (i, j) in flow:
        adj[i].append(m + j)
        adj[m + j].append(i)
    pot = np.full(m + n, np.nan)
    parent = np.full(m + n, -1, dtype=int)
    pot[0] = 0.0
    queue = deque([0])
    while queue:
        x = queue.popleft()
        for y in adj[x]:
            if np.isnan(pot[y]):
                pot[y] = C[x, y - m] - pot[x] if x < m else C[y, x - m] - pot[x]
                parent[y] = x
                queue.append(y)
    if np.isnan(pot).any():  # pragma: no cover - would mean a broken invariant
        raise RuntimeError("basis is not a spanning tree")
    return pot[:m], pot[m:], adj


def _tree_path(adj, start, goal):
    """Node path from start to goal in the basis tree."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        x = queue.popleft()
        if x == goal:
            break
        for y in adj[x]:
            if y not in parent:
                parent[y] = x
                queue.append(y)
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]


def emd(a, b, C, *, tol: float = 1e-12, max_iter: int = 100_000,
        degenerate_switch: int = 50) -> ExactResult:
    """Solve the discrete optimal transport problem exactly.

    Parameters
    ----------
    a, b : array_like
        Source and target weights (equal total mass).
    C : array_like, shape (len(a), len(b))
        Ground cost matrix.
    tol : float
        Optimality tolerance on reduced costs (relative to max |C|).
    max_iter : int
        Pivot limit.
    degenerate_switch : int
        After this many consecutive zero length pivots, entering cells are
        chosen with Bland's rule until a non degenerate pivot happens.
    """
    a, b, C = _check_inputs(a, b, C, 1e-9)
    m, n = a.size, b.size
    flow = _north_west_corner(a, b)
    scale = max(1.0, float(np.abs(C).max()))
    degenerate_run = 0

    for it in range(max_iter):
        u, v, adj = _potentials(flow, C, m, n)
        reduced = C - u[:, None] - v[None, :]
        if degenerate_run >= degenerate_switch:
            neg = np.flatnonzero(reduced.ravel() < -tol * scale)
            if neg.size == 0:
                break
            i, j = divmod(int(neg[0]), n)
        else:
            k = int(np.argmin(reduced))
            i, j = divmod(k, n)
            if reduced[i, j] >= -tol * scale:
                break

        # The entering cell (i, j) closes a cycle with the tree path
        # row i -> ... -> column j. Edges along that path alternate - / +.
        nodes = _tree_path(adj, i, m + j)
        cycle_minus, cycle_plus = [], []
        for k in range(len(nodes) - 1):
            x, y = nodes[k], nodes[k + 1]
            cell = (x, y - m) if x < m else (y, x - m)
            (cycle_minus if k % 2 == 0 else cycle_plus).append(cell)

        theta = min(flow[c] for c in cycle_minus)
        # Leaving cell: a minus cell attaining theta (lowest index on ties).
        leaving = min(c for c in cycle_minus if flow[c] <= theta)
        for c in cycle_minus:
            flow[c] -= theta
        for c in cycle_plus:
            flow[c] += theta
        del flow[leaving]
        flow[(i, j)] = theta
        degenerate_run = degenerate_run + 1 if theta <= 0 else 0
    else:
        raise RuntimeError(f"transportation simplex did not converge in {max_iter} pivots")

    plan = np.zeros((m, n))
    for (i, j), q in flow.items():
        plan[i, j] = max(q, 0.0)
    return ExactResult(plan=plan, cost=float((plan * C).sum()), u=u, v=v,
                       iterations=it, basis=sorted(flow))


def emd_cost(a, b, C, **kw) -> float:
    """Convenience wrapper returning only the optimal cost."""
    return emd(a, b, C, **kw).cost
