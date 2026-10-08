# otlab

Optimal transport from first principles, in about 600 lines of NumPy.

Optimal transport asks for the cheapest way to move one distribution of mass onto another. The minimum cost, the Wasserstein distance, has become a standard tool wherever you compare distributions that live on a geometry: generative model losses, domain adaptation, colour and histogram transfer, single cell trajectory inference, shape matching. Libraries like POT hide the solvers behind compiled code. This repo builds each of the main ones from scratch so you can read exactly how they work, and then checks them against each other and against closed form answers.

| Module | What it solves | How |
| --- | --- | --- |
| `otlab.exact` | Exact discrete OT (the earth mover's distance) | Transportation simplex on an explicit spanning tree basis, with dual potentials |
| `otlab.sinkhorn` | Entropic OT, unbalanced OT, Sinkhorn divergence | Log domain Sinkhorn with epsilon annealing; KL relaxed updates for unequal mass |
| `otlab.barycenter` | Wasserstein barycenters on a fixed grid | Iterative Bregman projections, run on logarithms |
| `otlab.one_d` | Exact OT on the line, sliced Wasserstein in R^d | Quantile coupling; random 1D projections |

The only dependency is NumPy.

## Quick start

```bash
git clone https://github.com/MohammadHossinzehi/optimal-transport-lab
cd optimal-transport-lab
pip install -e ".[test]"     # or just: pip install numpy pytest

python -m otlab                  # exact vs entropic vs sliced on two random clouds
python -m otlab barycenter       # displacement interpolation between two Gaussians
python -m pytest                 # 44 tests, about 12 s
```

Sample output of `python -m otlab --n 60`:

```
method                                    W2^2    rel. gap   time ms
exact transportation simplex          4.664892                  82.5
  pivots / nonzeros                        321         119
sinkhorn eps = 0.1 * max C            6.658617    4.27e-01       4.7
sinkhorn eps = 0.01 * max C           4.951764    6.15e-02      15.3
sinkhorn eps = 0.001 * max C          4.674568    2.07e-03     107.9
sliced W2^2 (500 projections)         2.095381 (lower bnd)      29.0

1D sanity check on the first coordinate: closed form 3.111705130, simplex 3.111705130
```

That table is the whole trade off of computational OT in one picture. The simplex is exact and returns a sparse vertex plan (119 nonzeros = m + n - 1). Sinkhorn is a few dense matrix operations per iteration, so it is fast and GPU friendly, but it pays a bias that shrinks with epsilon while the iteration count grows. Sliced Wasserstein never builds a coupling at all and gives a cheap lower bound.

And `python -m otlab barycenter`, interpolating N(0.2, 0.04²) into N(0.8, 0.08²):

```
    t      mean       std  exact mean  exact std
 0.50    0.4994    0.0600      0.4994     0.0591
      |                       .:-+*%%%%#*=-:.                      |
```

The mass slides across as a single bump whose width also interpolates, rather than fading one hump out and another in, which is what averaging the histograms would do.

## Library usage

```python
import numpy as np
from otlab import emd, sinkhorn, sinkhorn_unbalanced, sinkhorn_divergence, barycenter
from otlab import wasserstein_1d, sliced_wasserstein, cost_matrix

X, Y = np.random.randn(50, 2), np.random.randn(40, 2) + 2
a, b = np.full(50, 1 / 50), np.full(40, 1 / 40)
C = cost_matrix(X, Y, p=2)

exact = emd(a, b, C)              # .plan .cost .u .v .iterations .basis
ent = sinkhorn(a, b, C, eps=0.05) # .plan .f .g .cost .objective .converged
unb = sinkhorn_unbalanced(3 * a, b, C, eps=0.05, rho=1.0)
S = sinkhorn_divergence(X, Y, eps=0.05)

wasserstein_1d(X[:, 0], Y[:, 0], p=1)
sliced_wasserstein(X, Y, n_projections=200, seed=0)
```

## Design notes

**The simplex keeps its basis as a tree, zeros included.** Every basic feasible solution of the transportation LP is a spanning tree with m + n - 1 cells. The usual pitfall is degeneracy: when a row and a column are exhausted at the same time, a naive implementation loses a basic cell, the tree splits, and the dual potentials become undefined. Here the north west corner start advances exactly one index per step, and zero flow cells stay in the basis, so the potentials u, v are always obtainable by one BFS. Each pivot prices all cells, adds the most negative one, walks the unique tree cycle it closes, and removes the bottleneck. After a run of 50 degenerate (zero length) pivots it falls back to Bland's rule to break any cycle.

**Sinkhorn works on potentials, not on the kernel.** The textbook iteration multiplies by K = exp(-C/eps), which underflows to exactly zero once eps is a few percent of the cost scale. Running the same fixed point on f, g with logsumexp is stable at any eps; one of the tests picks an eps where `exp(-C.max()/eps) == 0.0` and still requires a feasible plan. Epsilon scaling (start large, halve, warm start) keeps the iteration count manageable as eps shrinks.

**Unbalanced OT is one extra scalar.** Replacing the hard marginal constraints with rho * KL penalties changes the Sinkhorn update to `f = -lam * eps * lse(...)` with `lam = rho / (rho + eps)`. Everything else is shared with the balanced loop.

**Barycenters are Sinkhorn with a coupling step.** Iterative Bregman projections alternate a Sinkhorn scaling against each input with a weighted geometric mean of the K^T u_k terms. It is implemented in log space for the same underflow reason.

## How it is tested

The tests lean on independent ground truths rather than snapshots:

* **Brute force.** For 6 uniform points per side, OT is an assignment problem (Birkhoff von Neumann), so the simplex is checked against all 720 permutations.
* **LP certificates.** On random rectangular problems with random weights the test checks primal feasibility, dual feasibility `u_i + v_j <= C_ij`, strong duality `a·u + b·v = cost`, complementary slackness, and that the plan is a vertex with at most m + n - 1 nonzeros.
* **Closed forms.** The simplex agrees with the 1D quantile formula to 1e-10 relative for p = 1, 2, 3. A translation by c gives W_p = |c|. For sliced Wasserstein a translation t gives SW_2² = |t|²/d. Zero cost unbalanced OT has an analytic transported mass of (|a||b|)^((rho+eps)/(2rho+eps)), which the solver matches to 1e-6.
* **Convergence and bounds.** Sinkhorn's transport cost is always at least the exact optimum, the gap shrinks monotonically as eps goes 1 → 0.001, and its dual value bounds its own cost. Unbalanced with huge rho reproduces balanced. The Sinkhorn divergence is zero on identical inputs, symmetric and positive otherwise.
* **Barycenters.** The barycenter of two Gaussians matches the analytic displacement interpolation (averaged mean and standard deviation), is unimodal, and agrees with the exact 1D quantile barycenter.
* **Degenerate and edge cases.** A 30 × 30 problem engineered so every north west step is degenerate, zero mass entries, coincident supports, and input validation.

## Limitations

The simplex is O(mn) per pivot in pure Python and comfortably handles a few hundred points per side; a production network simplex would maintain the tree with thread indices and price in blocks. Barycenters use a fixed support. Nothing here uses a GPU.

## References

* Peyré and Cuturi, *Computational Optimal Transport*, 2019.
* Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013.
* Benamou, Carlier, Cuturi, Nenna, Peyré, *Iterative Bregman Projections for Regularized Transportation Problems*, 2015.
* Chizat, Peyré, Schmitzer, Vialard, *Scaling Algorithms for Unbalanced Transport Problems*, 2018.
* Feydy et al., *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*, AISTATS 2019.

## License

MIT
