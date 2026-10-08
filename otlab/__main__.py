"""Command line demo: python -m otlab [compare|barycenter] [--n N] [--seed S]"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import (barycenter, cost_matrix, emd, quantile_barycenter_1d, sinkhorn,
               sliced_wasserstein, wasserstein_1d)


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def compare(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Y = rng.normal(size=(n, 2)) * [1.5, 0.5] + [2.0, 1.0]
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    C = cost_matrix(X, Y, 2)
    scale = C.max()

    exact, ms = _timed(lambda: emd(a, b, C))
    print(f"Two random weighted clouds in R^2, n = {n}, squared Euclidean cost\n")
    print(f"{'method':<34}{'W2^2':>12}{'rel. gap':>12}{'time ms':>10}")
    print(f"{'exact transportation simplex':<34}{exact.cost:>12.6f}{'':>12}{ms:>10.1f}")
    print(f"{'  pivots / nonzeros':<34}{exact.iterations:>12d}{int((exact.plan > 0).sum()):>12d}")
    for rel in (1e-1, 1e-2, 1e-3):
        eps = rel * scale
        res, ms = _timed(lambda: sinkhorn(a, b, C, eps))
        gap = (res.cost - exact.cost) / exact.cost
        print(f"{f'sinkhorn eps = {rel:g} * max C':<34}{res.cost:>12.6f}{gap:>12.2e}{ms:>10.1f}")
    sw, ms = _timed(lambda: sliced_wasserstein(X, Y, a, b, n_projections=500, seed=seed))
    print(f"{'sliced W2^2 (500 projections)':<34}{sw ** 2:>12.6f}{'(lower bnd)':>12}{ms:>10.1f}")

    x1, y1 = X[:, 0], Y[:, 0]
    closed = wasserstein_1d(x1, y1, a, b, p=2) ** 2
    simplex = emd(a, b, cost_matrix(x1, y1, 2)).cost
    print(f"\n1D sanity check on the first coordinate: closed form {closed:.9f}, "
          f"simplex {simplex:.9f}")


def barycenter_demo(seed: int) -> None:
    grid = np.linspace(0, 1, 120)
    C = cost_matrix(grid, grid, 2)

    def bump(mu, s):
        h = np.exp(-0.5 * ((grid - mu) / s) ** 2)
        return h / h.sum()

    B = np.stack([bump(0.2, 0.04), bump(0.8, 0.08)], axis=1)
    print("W2 barycenter of N(0.2, 0.04^2) and N(0.8, 0.08^2), weights t, 1 - t\n")
    print(f"{'t':>5}{'mean':>10}{'std':>10}{'exact mean':>12}{'exact std':>11}")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = barycenter(B, C, eps=2e-4, weights=[1 - t, t])
        mean = p @ grid
        std = np.sqrt(p @ (grid - mean) ** 2)
        q = quantile_barycenter_1d([grid, grid], [B[:, 0], B[:, 1]], [1 - t, t])
        print(f"{t:>5.2f}{mean:>10.4f}{std:>10.4f}{q.mean():>12.4f}{q.std():>11.4f}")
        bar = "".join(" .:-=+*#%@"[min(9, int(9 * v / p.max()))] for v in p[::2])
        print("      |" + bar + "|")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="otlab", description=__doc__)
    ap.add_argument("command", nargs="?", default="compare", choices=["compare", "barycenter"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command == "compare":
        compare(args.n, args.seed)
    else:
        barycenter_demo(args.seed)


if __name__ == "__main__":
    main()
