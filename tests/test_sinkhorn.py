import numpy as np
import pytest

from otlab import cost_matrix, emd, sinkhorn, sinkhorn_divergence, sinkhorn_unbalanced


@pytest.fixture
def problem():
    rng = np.random.default_rng(7)
    X, Y = rng.normal(size=(40, 2)), rng.normal(1, 1, size=(30, 2))
    a, b = rng.dirichlet(np.ones(40)), rng.dirichlet(np.ones(30))
    return a, b, cost_matrix(X, Y, 2)


def test_marginals_are_matched(problem):
    a, b, C = problem
    res = sinkhorn(a, b, C, eps=0.05)
    assert res.converged
    np.testing.assert_allclose(res.plan.sum(1), a, atol=1e-8)
    np.testing.assert_allclose(res.plan.sum(0), b, atol=1e-8)


def test_converges_to_exact_cost_as_eps_vanishes(problem):
    a, b, C = problem
    exact = emd(a, b, C).cost
    gaps = []
    for eps in (1.0, 0.1, 0.01, 0.001):
        res = sinkhorn(a, b, C, eps=eps, max_iter=50_000)
        # Any feasible plan costs at least the optimum, and the entropic dual
        # value upper bounds its own transport cost.
        assert res.cost >= exact - 1e-7
        assert res.objective >= res.cost - 1e-7
        gaps.append(res.cost - exact)
    assert all(g1 > g2 for g1, g2 in zip(gaps, gaps[1:]))
    assert gaps[-1] / exact < 5e-3


def test_tiny_eps_does_not_underflow(problem):
    # exp(-C/eps) is exactly zero in float64 here; the log domain solver
    # must still return a finite, feasible plan.
    a, b, C = problem
    assert np.exp(-C.max() / 1e-4) == 0.0
    res = sinkhorn(a, b, C, eps=1e-4, max_iter=50_000, tol=1e-7)
    assert np.isfinite(res.plan).all()
    np.testing.assert_allclose(res.plan.sum(0), b, atol=1e-7)


def test_eps_scaling_matches_cold_start(problem):
    a, b, C = problem
    warm = sinkhorn(a, b, C, eps=0.05)
    cold = sinkhorn(a, b, C, eps=0.05, eps_scaling=None)
    np.testing.assert_allclose(warm.plan, cold.plan, atol=1e-8)


def test_unbalanced_large_rho_recovers_balanced(problem):
    a, b, C = problem
    bal = sinkhorn(a, b, C, eps=0.1, eps_scaling=None)
    unb = sinkhorn_unbalanced(a, b, C, eps=0.1, rho=1e5, max_iter=50_000)
    np.testing.assert_allclose(unb.plan, bal.plan, atol=1e-4)


def test_unbalanced_handles_unequal_mass(problem):
    a, b, C = problem
    a2 = 3 * a
    res = sinkhorn_unbalanced(a2, b, C, eps=0.1, rho=50.0)
    assert res.converged
    assert b.sum() < res.plan.sum() < a2.sum()
    # weaker penalties make it cheaper to destroy mass than to move it
    weak = sinkhorn_unbalanced(a2, b, C, eps=0.1, rho=0.5)
    assert weak.plan.sum() < res.plan.sum()


def test_unbalanced_free_transport_mass_matches_closed_form():
    # With zero ground cost the plan is a scaled product measure of mass m.
    # Minimising rho KL + rho KL + eps KL over m by hand gives
    #     m = (|a| |b|) ** ((rho + eps) / (2 rho + eps)),
    # which tends to the geometric mean sqrt(|a| |b|) as eps -> 0.
    a = np.full(5, 3.0 / 5)
    b = np.full(4, 1.0 / 4)
    for rho, eps in [(1.0, 0.1), (2.0, 0.5), (0.3, 1e-3)]:
        res = sinkhorn_unbalanced(a, b, np.zeros((5, 4)), eps=eps, rho=rho)
        expected = 3.0 ** ((rho + eps) / (2 * rho + eps))
        assert res.plan.sum() == pytest.approx(expected, rel=1e-6)


def test_balanced_rejects_unequal_mass(problem):
    a, b, C = problem
    with pytest.raises(ValueError):
        sinkhorn(2 * a, b, C, eps=0.1)


def test_sinkhorn_divergence_properties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    Y = rng.normal(size=(25, 2)) + [3.0, 0.0]
    assert sinkhorn_divergence(X, X, eps=0.1) == pytest.approx(0.0, abs=1e-8)
    s_xy = sinkhorn_divergence(X, Y, eps=0.1)
    s_yx = sinkhorn_divergence(Y, X, eps=0.1)
    assert s_xy > 0
    assert s_xy == pytest.approx(s_yx, rel=1e-6)
    # Shifting further apart increases the divergence.
    assert sinkhorn_divergence(X, Y + [2.0, 0.0], eps=0.1) > s_xy
