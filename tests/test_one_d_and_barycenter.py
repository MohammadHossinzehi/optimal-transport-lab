import numpy as np
import pytest

from otlab import (barycenter, cost_matrix, quantile_barycenter_1d,
                   sliced_wasserstein, wasserstein_1d)


def test_dirac_masses():
    assert wasserstein_1d([0.0], [3.0], p=1) == pytest.approx(3.0)
    assert wasserstein_1d([0.0], [3.0], p=2) == pytest.approx(3.0)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_translation_moves_every_quantile(p):
    x = np.random.default_rng(0).normal(size=200)
    assert wasserstein_1d(x, x + 1.7, p=p) == pytest.approx(1.7)


def test_metric_axioms_on_random_samples():
    rng = np.random.default_rng(2)
    x, y, z = rng.normal(size=50), rng.exponential(size=40), rng.uniform(-2, 2, 30)
    d = lambda u, v: wasserstein_1d(u, v, p=2)
    assert d(x, x) == pytest.approx(0.0)
    assert d(x, y) == pytest.approx(d(y, x))
    assert d(x, z) <= d(x, y) + d(y, z) + 1e-12


def test_sliced_translation_scales_with_dimension():
    # For a pure translation t, every projection moves by <theta, t>, and
    # E[<theta, t>^2] = |t|^2 / d for theta uniform on the sphere.
    rng = np.random.default_rng(5)
    d = 4
    X = rng.normal(size=(100, d))
    t = np.array([2.0, -1.0, 0.5, 1.0])
    sw = sliced_wasserstein(X, X + t, n_projections=4000, seed=1)
    assert sw ** 2 == pytest.approx(t @ t / d, rel=0.05)


def test_sliced_is_below_exact_w2():
    from otlab import emd
    rng = np.random.default_rng(9)
    X, Y = rng.normal(size=(30, 3)), rng.normal(0.5, 2, size=(30, 3))
    w = np.full(30, 1 / 30)
    exact = emd(w, w, cost_matrix(X, Y, 2)).cost
    assert sliced_wasserstein(X, Y, n_projections=300, seed=0) ** 2 <= exact


def _bump(grid, mu, s):
    h = np.exp(-0.5 * ((grid - mu) / s) ** 2)
    return h / h.sum()


def test_barycenter_interpolates_gaussians():
    # The W2 barycenter of two 1D Gaussians is the Gaussian with the averaged
    # mean and averaged standard deviation (displacement interpolation).
    grid = np.linspace(0, 1, 150)
    B = np.stack([_bump(grid, 0.25, 0.03), _bump(grid, 0.7, 0.06)], axis=1)
    p = barycenter(B, cost_matrix(grid, grid, 2), eps=1e-4)
    mean = p @ grid
    std = np.sqrt(p @ (grid - mean) ** 2)
    assert p.sum() == pytest.approx(1.0)
    assert mean == pytest.approx(0.475, abs=2e-3)
    assert std == pytest.approx(0.045, abs=3e-3)
    # unimodal, unlike the Euclidean average which has two humps
    peaks = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]))
    assert len(peaks) == 1


def test_barycenter_agrees_with_quantile_average():
    grid = np.linspace(0, 1, 150)
    B = np.stack([_bump(grid, 0.3, 0.05), _bump(grid, 0.6, 0.05)], axis=1)
    p = barycenter(B, cost_matrix(grid, grid, 2), eps=1e-4, weights=[0.3, 0.7])
    q = quantile_barycenter_1d([grid, grid], [B[:, 0], B[:, 1]], [0.3, 0.7], 4000)
    assert p @ grid == pytest.approx(q.mean(), abs=2e-3)
    # entropic blur makes the regularised answer slightly wider, never narrower
    assert np.sqrt(p @ (grid - p @ grid) ** 2) >= q.std() - 1e-3


def test_barycenter_with_one_weight_returns_input():
    grid = np.linspace(0, 1, 80)
    B = np.stack([_bump(grid, 0.4, 0.05), _bump(grid, 0.9, 0.02)], axis=1)
    p = barycenter(B, cost_matrix(grid, grid, 2), eps=1e-4, weights=[1.0, 0.0])
    assert np.abs(p - B[:, 0]).sum() < 0.05
