import itertools

import numpy as np
import pytest

from otlab import cost_matrix, emd, wasserstein_1d


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_assignment(seed):
    # With n uniform points on each side, OT reduces to the best permutation
    # (Birkhoff von Neumann), which we can enumerate for n = 6.
    rng = np.random.default_rng(seed)
    n = 6
    C = rng.random((n, n))
    best = min(sum(C[i, s[i]] for i in range(n)) for s in itertools.permutations(range(n)))
    res = emd(np.full(n, 1 / n), np.full(n, 1 / n), C)
    assert res.cost == pytest.approx(best / n, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_kkt_conditions_on_unequal_shapes(seed):
    rng = np.random.default_rng(100 + seed)
    m, n = rng.integers(2, 15, size=2)
    a = rng.dirichlet(np.ones(m))
    b = rng.dirichlet(np.ones(n))
    C = rng.random((m, n)) * 10
    res = emd(a, b, C)
    P = res.plan
    # primal feasibility
    assert (P >= 0).all()
    np.testing.assert_allclose(P.sum(1), a, atol=1e-12)
    np.testing.assert_allclose(P.sum(0), b, atol=1e-12)
    # dual feasibility
    assert (res.u[:, None] + res.v[None, :] <= C + 1e-9).all()
    # strong duality
    assert a @ res.u + b @ res.v == pytest.approx(res.cost, abs=1e-9)
    # complementary slackness and a vertex solution
    assert np.all(np.abs((C - res.u[:, None] - res.v[None, :])[P > 1e-14]) < 1e-9)
    assert len(res.basis) == m + n - 1
    assert (P > 1e-14).sum() <= m + n - 1


@pytest.mark.parametrize("p", [1, 2, 3])
def test_agrees_with_closed_form_1d(p):
    rng = np.random.default_rng(p)
    x, y = rng.normal(size=25), rng.normal(1, 2, size=17)
    a, b = rng.dirichlet(np.ones(25)), rng.dirichlet(np.ones(17))
    exact = emd(a, b, cost_matrix(x, y, p)).cost
    assert exact == pytest.approx(wasserstein_1d(x, y, a, b, p) ** p, rel=1e-10)


def test_highly_degenerate_problem_terminates():
    # Integer costs with many ties and equal uniform marginals: every north
    # west corner step exhausts a row and a column together.
    n = 30
    C = (np.add.outer(np.arange(n), np.arange(n)) % 3).astype(float)
    res = emd(np.ones(n), np.ones(n), C)
    assert res.cost == pytest.approx(0.0)


def test_zero_mass_entries_are_allowed():
    a = np.array([0.5, 0.0, 0.5])
    b = np.array([0.0, 1.0])
    C = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    res = emd(a, b, C)
    assert res.cost == pytest.approx(3.0)
    np.testing.assert_allclose(res.plan[:, 0], 0)


def test_identity_coupling_when_supports_coincide():
    x = np.linspace(0, 1, 12)
    w = np.random.default_rng(3).dirichlet(np.ones(12))
    res = emd(w, w, cost_matrix(x, x, 2))
    assert res.cost == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(res.plan, np.diag(w), atol=1e-14)


def test_input_validation():
    with pytest.raises(ValueError, match="equal mass"):
        emd([1.0], [2.0], [[0.0]])
    with pytest.raises(ValueError, match="shape"):
        emd([1.0, 0.0], [1.0], [[0.0, 1.0]])
    with pytest.raises(ValueError, match="non negative"):
        emd([1.5, -0.5], [1.0], [[0.0], [1.0]])
