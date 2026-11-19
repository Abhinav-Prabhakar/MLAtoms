"""Histogram density — the oldest estimator, and the one that shows the curse.

Chapter:   09-density-and-anomaly
Companion: histogram_density.md
Tier 1 · 4 variants: fd_rule, sturges, scott, bayesian_blocks

    p_hat(x) = count(bin(x)) / (n * volume(bin(x)))                        (1)
    Freedman-Diaconis:  h = 2 IQR n^(-1/3)                                 (2)
    Scott:              h = 3.49 sd n^(-1/3)                               (3)
    Sturges:            k = ceil(log2 n) + 1                               (4)
    blocks:  max over partitions  sum_b N_b log(N_b / W_b) - prior * B     (5)

A histogram is a KDE whose kernel is a box and whose bandwidth is allowed to
change from bin to bin. That is the whole family: (1) is the estimate, and the
four variants are four ways to choose the bins. The rules in (2)-(4) all have the
same $n^{-1/3}$ width, because that is the bias-variance optimum for a
piecewise-constant estimate: the bias is $O(h)$ inside a bin and the variance is
$O(1/(nh))$, so the sum is minimised at $h \\propto n^{-1/3}$ — a *slower* rate
than the KDE's $n^{-1/5}$, which is the price of the discontinuity. Hence an
error rate of $n^{-2/3}$ against the KDE's $n^{-4/5}$.

That exponent is the useful thing to know about histograms. The estimator is
fast, exact to compute and completely transparent, and it dies in high dimensions
faster than anything else in this chapter: the demo's held-out log density falls
by 39 nats between one and ten dimensions at a fixed $n = 1000$.

The last variant is different in kind. `bayesian_blocks` does not choose a width;
it chooses a partition, maximising (5) over every way of cutting the data into
intervals, with a penalty per block. Wide where the density is smooth, narrow
where it changes — the answer to the bandwidth problem, solved exactly by a
dynamic program rather than by a rule of thumb.

Everything here is one-dimensional per axis; a $d$-dimensional estimate is the
product of $d$ one-dimensional ones, exactly as in `product_kernel`, so a joint
histogram never arises and the $k^d$ emptiness below never has to be materialised.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("fd_rule", "sturges", "scott", "bayesian_blocks")
FLOOR = 1e-300


def _sturges_bins(n: int) -> int:
    """Equation (4): the rule that assumes the data are roughly normal."""
    return max(1, int(math.ceil(math.log2(n))) + 1)


def _rule_edges(x: np.ndarray, variant: str) -> np.ndarray:
    """Bin edges for one column, from one of the closed-form rules."""
    n = len(x)
    low, high = float(x.min()), float(x.max())
    span = high - low
    if span <= 0.0:
        return np.array([low - 0.5, high + 0.5])
    bins = _sturges_bins(n)
    if variant == "fd_rule":
        q75, q25 = np.percentile(x, [75.0, 25.0])
        width = 2.0 * (q75 - q25) * n ** (-1.0 / 3.0)
        if width > 0.0:
            bins = max(1, int(math.ceil(span / width)))
    elif variant == "scott":
        width = 3.49 * float(x.std(ddof=1)) * n ** (-1.0 / 3.0)
        if width > 0.0:
            bins = max(1, int(math.ceil(span / width)))
    return np.linspace(low, high, bins + 1)


def _bayesian_blocks(x: np.ndarray, prior: float, n_grid: int) -> np.ndarray:
    """Scargle's dynamic program on a fine grid, equation (5).

    The grid fixes the smallest resolvable block. Without it a block containing
    one data point has zero width and an infinite density, and the optimum is a
    partition into singletons. With it, the problem is a shortest path over
    `n_grid` cells, solved exactly in `O(n_grid^2)`.
    """
    low, high = float(x.min()), float(x.max())
    if high <= low:
        return np.array([low - 0.5, high + 0.5])
    grid = np.linspace(low, high, n_grid + 1)
    counts, _ = np.histogram(x, grid)
    cumulative = np.concatenate([[0], np.cumsum(counts)])
    width = float(grid[1] - grid[0])

    best = np.full(n_grid + 1, -np.inf)
    best[0] = 0.0
    cut = np.zeros(n_grid + 1, dtype=int)
    for j in range(1, n_grid + 1):
        i = np.arange(j)
        mass = cumulative[j] - cumulative[i]
        length = (j - i) * width
        # A block with no points has zero likelihood contribution and is never
        # useful; -inf keeps it out of the argmax rather than letting 0 log 0
        # stand for a density of zero.
        fitness = np.where(mass > 0.0, mass * np.log(np.maximum(mass, 1.0) / length),
                           -np.inf)
        candidate = best[:j] + fitness - prior
        pick = int(np.argmax(candidate))
        best[j] = candidate[pick]
        cut[j] = pick

    edges = [n_grid]
    while edges[-1] > 0:
        edges.append(int(cut[edges[-1]]))
    return grid[sorted(edges)]


class HistogramDensity:
    """Piecewise-constant density estimate on a per-axis grid.

    Parameters
    ----------
    variant : {"fd_rule", "sturges", "scott", "bayesian_blocks"}
        Bin selection rule. The first three choose a constant width per axis;
        `bayesian_blocks` chooses a partition and is one-dimensional only.
    bins : int or None
        Fixed number of equal-width bins per axis, overriding the variant. The
        escape hatch that makes the variants comparable on the same grid.
    prior : float
        Penalty per block for `bayesian_blocks`, in log-likelihood units.
        Larger values give fewer, wider blocks.
    n_grid : int
        Resolution of the dynamic program for `bayesian_blocks`. A block cannot
        be narrower than `range / n_grid`.

    Attributes
    ----------
    edges_ : list of ndarray
        Bin edges per feature, length `bins + 1`.
    densities_ : list of ndarray
        Density of each bin per feature.
    counts_ : list of ndarray
        Raw counts per bin.
    n_features_ : int
    """

    def __init__(self, *, variant: str = "fd_rule", bins: int | None = None,
                 prior: float = 1.0, n_grid: int = 256) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if bins is not None and bins < 1:
            raise ValueError("bins must be positive")
        if prior < 0.0:
            raise ValueError("prior must be non-negative")
        if n_grid < 2:
            raise ValueError("n_grid must be at least two")
        self.variant = variant
        self.bins = bins
        self.prior = float(prior)
        self.n_grid = int(n_grid)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> HistogramDensity:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        n, d = X.shape
        if n < 2:
            raise ValueError("need at least two points")
        if self.variant == "bayesian_blocks" and d != 1:
            raise ValueError("bayesian_blocks is one-dimensional; fit one column")
        self.n_features_ = d
        self.edges_: list[np.ndarray] = []
        self.densities_: list[np.ndarray] = []
        self.counts_: list[np.ndarray] = []
        for j in range(d):
            column = X[:, j]
            if self.bins is not None:
                low, high = float(column.min()), float(column.max())
                if high <= low:
                    edges = np.array([low - 0.5, high + 0.5])
                else:
                    edges = np.linspace(low, high, self.bins + 1)
            elif self.variant == "bayesian_blocks":
                edges = _bayesian_blocks(column, self.prior, self.n_grid)
            else:
                edges = _rule_edges(column, self.variant)
            counts, _ = np.histogram(column, edges)
            widths = np.diff(edges)
            self.edges_.append(edges)
            self.counts_.append(counts)
            self.densities_.append(counts / (n * widths))
        return self

    # -- scoring ---------------------------------------------------------- #
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log density of each row; outside the fitted range is the floor.

        With a product estimate over axes, a point outside the range on any axis
        has zero density, so the minimum over axes is the answer.
        """
        if not hasattr(self, "n_features_"):
            raise RuntimeError("fit before scoring")
        X = self._check(X)
        total = np.zeros(len(X))
        for j in range(self.n_features_):
            column = X[:, j]
            edges, density = self.edges_[j], self.densities_[j]
            index = np.searchsorted(edges, column, side="right") - 1
            index = np.clip(index, 0, len(density) - 1)
            outside = (column < edges[0]) | (column > edges[-1])
            value = np.where(outside, 0.0, density[index])
            total += np.log(np.maximum(value, FLOOR))
        return total

    def score(self, X: np.ndarray) -> float:
        """Mean log density, the criterion cross-validation would use."""
        return float(np.mean(self.score_samples(X)))

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _grid_ise(edges: np.ndarray, density: np.ndarray, truth) -> float:
    """Integrated squared error of a step function against a callable."""
    width = np.diff(edges)
    middle = 0.5 * (edges[1:] + edges[:-1])
    return float(np.sum((density - truth(middle)) ** 2 * width))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _demo_rules() -> None:
    """The three closed forms, their identities, and their error rate."""
    rng = np.random.default_rng(0)
    sample = rng.lognormal(mean=0.0, sigma=0.6, size=500)
    rows = {}
    for variant in ("fd_rule", "sturges", "scott"):
        model = HistogramDensity(variant=variant).fit(sample)
        edges = model.edges_[0]
        width = float(np.diff(edges)[0])
        mass = float(np.sum(model.densities_[0] * np.diff(edges)))
        assert abs(mass - 1.0) < 1e-12
        rows[variant] = (len(edges) - 1, width)

    # The identities in (2)-(4), recomputed from the sample.
    n = len(sample)
    q75, q25 = np.percentile(sample, [75.0, 25.0])
    fd_width = 2.0 * (q75 - q25) * n ** (-1.0 / 3.0)
    scott_width = 3.49 * sample.std(ddof=1) * n ** (-1.0 / 3.0)
    span = float(sample.max() - sample.min())
    assert rows["fd_rule"][0] == max(1, int(math.ceil(span / fd_width)))
    assert rows["scott"][0] == max(1, int(math.ceil(span / scott_width)))
    assert rows["sturges"][0] == _sturges_bins(n)

    # Scaling the data scales the edges and divides the density by the factor.
    scaled = 7.0 * sample
    a = HistogramDensity(variant="fd_rule").fit(sample)
    b = HistogramDensity(variant="fd_rule").fit(scaled)
    assert np.allclose(a.edges_[0] * 7.0, b.edges_[0])
    assert np.allclose(a.densities_[0] / 7.0, b.densities_[0])

    # The bin grid is tied to the sample, so a translation changes the answer
    # at fixed query points -- there is no origin-invariant histogram.
    fixed = HistogramDensity(bins=20).fit(sample)
    width = float(np.diff(fixed.edges_[0])[0])
    shifted = HistogramDensity(bins=20).fit(sample + 0.5 * width)
    probes = np.linspace(sample.min() + width, sample.max() - width, 400)
    here = np.exp(fixed.score_samples(probes))
    there = np.exp(shifted.score_samples(probes))
    wobble = float(0.5 * np.sum(np.abs(here - there)) * (probes[1] - probes[0]))
    assert wobble > 0.01

    # The error rate: MISE falls like n^(-2/3), so quadrupling n should cut
    # the ISE of the same estimator by about 2^(-4/3) = 0.40.
    normal = rng.normal(size=2000)
    more = rng.normal(size=8000)
    small = HistogramDensity(variant="fd_rule").fit(normal)
    large = HistogramDensity(variant="fd_rule").fit(more)
    ise_small = _grid_ise(small.edges_[0], small.densities_[0], _normal_pdf)
    ise_large = _grid_ise(large.edges_[0], large.densities_[0], _normal_pdf)
    ratio = ise_large / ise_small
    assert 0.25 < ratio < 0.75

    print(f"rules: 500 lognormal points, skewed so the mean is not a useful "
          f"scale. Freedman-Diaconis targets h = 2 IQR n^(-1/3) = "
          f"{fd_width:.4f}, which needs {rows['fd_rule'][0]} equal bins of "
          f"width {rows['fd_rule'][1]:.4f}; Scott targets 3.49 sd n^(-1/3) = "
          f"{scott_width:.4f} and gets {rows['scott'][0]} bins of width "
          f"{rows['scott'][1]:.4f}; Sturges ignores the data and returns "
          f"ceil(log2 500) + 1 = {rows['sturges'][0]} equal bins. Each density "
          f"sums to 1 against its bin widths to 1e-12. Multiplying the sample "
          f"by 7 multiplies the edges by 7 and divides the density by 7 "
          f"exactly, but moving the sample half a bin to the right moves "
          f"{wobble:.1%} of the probability mass to a different bin: "
          f"equal-width bins are not translation invariant, because the edges "
          f"are recomputed from the shifted sample. Against a standard normal, "
          f"n = 2000 has ISE {ise_small:.2e} and n = 8000 has {ise_large:.2e}, "
          f"a ratio of {ratio:.2f} where n^(-2/3) predicts 0.40 -- the "
          f"histogram learns slower than a KDE, whose ratio would be 0.33.")


def _demo_blocks() -> None:
    """Bayesian blocks find the step, and the prior sets the resolution."""
    rng = np.random.default_rng(1)
    first = rng.uniform(0.0, 1.0, size=800)
    second = rng.uniform(1.0, 3.0, size=400)
    sample = np.concatenate([first, second])
    order = np.argsort(sample)
    sample = sample[order]

    rows = {}
    for prior in (1.0, 2.0, 4.0):
        model = HistogramDensity(variant="bayesian_blocks", prior=prior,
                                 n_grid=512).fit(sample)
        edges = model.edges_[0]
        inner = edges[1:-1]
        near = float(inner[np.argmin(np.abs(inner - 1.0))])
        rows[prior] = (len(edges) - 1, near)
    counts = [rows[p][0] for p in (1.0, 2.0, 4.0)]
    assert counts[0] >= counts[1] >= counts[2]
    assert abs(rows[4.0][1] - 1.0) < 0.01

    fixed = HistogramDensity(variant="fd_rule").fit(sample)
    assert len(fixed.edges_[0]) - 1 >= 2
    print(f"blocks: 800 points uniform on [0, 1] and 400 uniform on [1, 3] -- a "
          f"density that drops by a factor of four at x = 1. Bayesian blocks "
          f"with prior 1 finds {rows[1.0][0]} blocks, with prior 2 gives "
          f"{rows[2.0][0]}, and with prior 4 gives {rows[4.0][0]}: the penalty "
          f"buys resolution in the only currency that matters, the number of "
          f"pieces. The two-block answer is the generating density, and its "
          f"boundary sits at {rows[4.0][1]:.3f} -- one grid step from 1, since "
          f"the dynamic program can only cut on the 512-cell grid. "
          f"Freedman-Diaconis spends {len(fixed.edges_[0]) - 1} equal bins on "
          f"the same data and can put an edge near 1 only by luck of the "
          f"arithmetic. The price of the exact partition is that blocks are not "
          f"nested: changing the prior re-solves all of it.")


def _demo_curse() -> None:
    """What happens to a histogram when the dimension goes up."""
    rng = np.random.default_rng(2)
    report = {}
    for d in (1, 2, 5, 10):
        train = rng.normal(size=(1000, d))
        test = rng.normal(size=(500, d))
        model = HistogramDensity(variant="fd_rule").fit(train)
        scores = model.score_samples(test)
        codes = np.column_stack([np.digitize(train[:, j], model.edges_[j])
                                 for j in range(d)])
        occupied = int(len(np.unique(codes, axis=0)))
        empty = float(np.mean(scores <= math.log(FLOOR) + 1.0))
        report[d] = (float(scores.mean()), empty, occupied)
    order = [report[d][0] for d in (1, 2, 5, 10)]
    assert all(a > b for a, b in zip(order, order[1:], strict=False))
    print("curse: 1000 standard normal points per dimension, 500 fresh points "
          "scored, Freedman-Diaconis per axis:")
    for d, (mean, empty, occupied) in report.items():
        print(f"  d = {d:2d}  mean log density {mean:9.2f}  floor hits "
              f"{empty:5.1%}  occupied cells {occupied}")
    print(f"  The estimate loses {order[0] - order[-1]:.1f} nats between one and "
          f"ten dimensions at the same sample size. The last column is the "
          f"mechanism: the number of non-empty product cells grows towards the "
          f"sample size, so each occupied cell holds about one point and its "
          f"density is a count of one divided by a volume. That is not an "
          f"estimate of the density; it is a restatement of where the points "
          f"happen to be. The floor hits are the second mechanism, and they get "
          f"worse in the tail.")


def _demo_broken() -> None:
    """Bad inputs, empty bins, and the floor."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    constant = np.ones(40)
    cases = [
        ("an unknown variant", lambda: HistogramDensity(variant="kde")),
        ("zero bins", lambda: HistogramDensity(bins=0)),
        ("a negative prior", lambda: HistogramDensity(prior=-1.0)),
        ("a one-cell grid", lambda: HistogramDensity(n_grid=1)),
        ("blocks in 2-D", lambda: HistogramDensity(variant="bayesian_blocks").fit(X)),
        ("one point", lambda: HistogramDensity().fit(X[:1])),
        ("scoring before fitting", lambda: HistogramDensity().score_samples(X)),
        ("the wrong width", lambda: HistogramDensity().fit(X).score_samples(X[:, :1])),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name} did not raise")

    flat = HistogramDensity(variant="fd_rule").fit(constant)
    raw = float(flat.score_samples(np.array([[1.0]]))[0])
    far = float(flat.score_samples(np.array([[1e6]]))[0])
    assert far == math.log(FLOOR)
    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. A constant column has IQR "
          f"zero, so (2)'s width is zero; the code falls back to Sturges and a "
          f"single bin of width one, and scores the constant {raw:.3f} rather "
          f"than dividing by zero. A query at 1e6 is outside every edge and "
          f"returns the floor, {far:.1f}, the same number for every point "
          f"however far outside it is -- a histogram has no tail, only a last "
          f"bin and then silence.")


def main() -> None:
    _demo_rules()
    _demo_blocks()
    _demo_curse()
    _demo_broken()
    print("histogram_density demo passed")


if __name__ == "__main__":
    main()
