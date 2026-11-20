r"""HBOS — one histogram per feature, and the sum of their surprises.

Chapter:   09-density-and-anomaly
Companion: hbos.md
Tier 2 · 3 variants: static_bins, dynamic_bins, adaptive

    p_hat_j(v) = max(N_b, 1) / (n * w_b)                                   (1)
    HBOS(x)    = sum_j -log p_hat_j(x_j)                                   (2)
    dynamic:   k_j = 2 IQR_j n^(-1/3) / span_j, floored and capped         (3)
    adaptive:  k_j = arg max_k  sum_i log p_hat_j^(-i)(x_i)                (4)
    chi-square: 2 d HBOS ~ chi^2_{2d} under the product model              (5)

HBOS is the KDE's crude cousin with a different job. It builds one histogram per
feature, converts each bin to a density, and scores a point by summing minus the
log density over features — equation (2). Two assumptions do all the work: that
features are independent (the joint density is the product, so the score is the
sum), and that a histogram is an adequate summary of each marginal. Both are
false in general and one of them is false in an interesting way, which is why the
method is worth knowing: it is the fastest multivariate detector that is not
simply a distance, it scales linearly in both $n$ and $d$, and it fails exactly
where the product model fails.

The independence assumption is not only a convenience. Under it the score is a sum
of $d$ terms and the paper proposes

$$2\,\mathrm{HBOS}(x) \sim \chi^2_{2d}$$

under the null — a threshold that needs no labelled sample. That law is not a
theorem and the demo shows why. Each term has mean equal to its marginal's
differential entropy, 1.419 for a standard normal, not the 1 that chi-square
assumes, so the score sits $0.44\,d$ above where the law puts it and the flag rate
at a nominal 5% climbs 5.9% → 10.5% → 20.7% over $d$ = 2, 4, 8. Worse, the score
is dimensionful: multiply one feature by 1000 and every score shifts by exactly
$\log 1000 = 6.91$ nats while the threshold stays put, so the same detector flags
100% of an untouched sample. The demo's fix is to resample the score from the
fitted model, which is calibrated by construction and shifts with the units.

The assumption this method cannot survive is independence. Plant anomalies on the
anti-diagonal of two features correlated at 0.97, at a radius where each coordinate
is inside the data's range and near its mode: the joint density there is
essentially zero, but the product model cannot see the pair, and HBOS ranks the
planted points *below* the ordinary ones — AUC 0.365 to 0.474 across the three
variants, none of the forty flagged at a 10% budget. The score of $(1.2, 1.2)$ and
$(1.2, -1.2)$ differs by 1.13 nats, the sampling noise in two bin counts, while
their true bivariate log-density differs by 46.9.

The three variants are three answers to "how many bins":

- `static_bins` uses a fixed count for every feature — ten by default, which is the
  original paper's choice. It is indifferent to $n$, so with a million rows it
  throws away almost all the resolution and with fifty rows it is too fine.
- `dynamic_bins` sets a count per feature from the Freedman-Diaconis width (3),
  which is the same rule as in the histogram atom and scales like $n^{1/3}$. A
  skewed feature gets more bins than a flat one at the same sample size, because
  the rule is in units of the feature's own robust spread, and a peaky one gets
  so many that `max_bins` is what actually decides.
- `adaptive` chooses the count per feature by maximising a leave-one-out
  log-likelihood (4), and places the bins at quantiles rather than at equal
  widths. Both changes matter: the count stops being a rule of thumb, and equal-
  frequency bins spend their resolution where the data are instead of where the
  range is.

Neither of the data-driven rules is uniformly better, and the demo says so. On a
sample that is 85% a narrow Gaussian and 15% a wide one, ten equal-width bins put
the whole core in a single bin and cost 3.02 nats per held-out point against 0.94
for quantile bins; on a plain Gaussian the same comparison is 2.89 against 2.96,
because twenty bins already match the scale of the data and the extra ones are
noise. A fixed count is a fixed *resolution*, and resolution is adequate only when
the data's shape agrees with it. Everything true of a single histogram is true
here: the arbitrary bin origin, the missing tail, the resolution lost to a
skewed feature.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("static_bins", "dynamic_bins", "adaptive")
FLOOR = 1e-300
_entropy = 0.5 * math.log(2.0 * math.pi * math.e)


def _fd_count(x: np.ndarray) -> int:
    """Equation (3): Freedman-Diaconis bin count, Sturges when the IQR is zero."""
    n = len(x)
    span = float(x.max() - x.min())
    if span <= 0.0:
        return 1
    q75, q25 = np.percentile(x, [75.0, 25.0])
    width = 2.0 * float(q75 - q25) * n ** (-1.0 / 3.0)
    if width <= 0.0:
        return max(1, int(math.ceil(math.log2(n))) + 1)
    return max(1, int(math.ceil(span / width)))


def _width_edges(x: np.ndarray, k: int) -> np.ndarray:
    low, high = float(x.min()), float(x.max())
    if high <= low:
        return np.array([low - 0.5, high + 0.5])
    return np.linspace(low, high, k + 1)


def _frequency_edges(x: np.ndarray, k: int) -> np.ndarray:
    """Quantile edges, deduplicated so ties cannot produce empty-width bins."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, k + 1))
    return np.unique(edges)


def _loo_score(x: np.ndarray, edges: np.ndarray) -> float:
    """Equation (4): mean held-out log density of a histogram, exactly.

    Leaving point i out of its own bin changes that bin from N to N - 1, so the
    held-out density of every point has a closed form -- no refitting is needed.
    """
    n = len(x)
    k = len(edges) - 1
    if k < 1 or n < 2:
        return -np.inf
    counts, _ = np.histogram(x, edges)
    widths = np.diff(edges)
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    held = counts[index] - 1
    if np.any(held <= 0):
        return -np.inf
    density = held / ((n - 1.0) * widths[index])
    return float(np.mean(np.log(density)))


class HBOS:
    """Histogram-based outlier score, one marginal histogram per feature.

    Parameters
    ----------
    variant : {"static_bins", "dynamic_bins", "adaptive"}
        How the bin count is chosen. `static_bins` uses `n_bins`; the other two
        choose per feature, from the Freedman-Diaconis rule and from a
        leave-one-out criterion respectively.
    n_bins : int
        Bin count for `static_bins`; ignored by the other variants.
    max_bins : int
        Upper limit for a per-feature count, so that `adaptive` cannot fit the
        noise and `dynamic_bins` cannot ask for more bins than points.
    contamination : float or None
        Flag budget for `predict`. `None` uses the chi-square threshold of (5).
    alpha : float
        Nominal tail probability for (5), used only when `contamination` is None.

    Attributes
    ----------
    edges_ : list of ndarray
        Bin edges per feature.
    counts_ : list of ndarray
        Counts per bin, per feature.
    n_bins_ : ndarray
        Realised bin count per feature.
    scores_ : ndarray
        Training score per row, higher meaning more anomalous.
    threshold_ : float
    n_features_, n_ : int
    """

    def __init__(self, *, variant: str = "static_bins", n_bins: int = 10,
                 max_bins: int = 64, contamination: float | None = 0.1,
                 alpha: float = 0.05) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_bins < 1 or max_bins < 2:
            raise ValueError("n_bins must be positive and max_bins at least two")
        if contamination is not None and not 0.0 < contamination < 1.0:
            raise ValueError("contamination must lie in (0, 1) or be None")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.variant = variant
        self.n_bins = int(n_bins)
        self.max_bins = int(max_bins)
        self.contamination = contamination
        self.alpha = float(alpha)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> HBOS:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        n, d = X.shape
        if n < 3 or d < 1:
            raise ValueError("need at least three points")
        self.n_, self.n_features_ = n, d
        self.edges_: list[np.ndarray] = []
        self.counts_: list[np.ndarray] = []
        counts = []
        for j in range(d):
            column = X[:, j]
            if self.variant == "static_bins":
                edges = _width_edges(column, self.n_bins)
            elif self.variant == "dynamic_bins":
                k = min(self.max_bins, max(1, _fd_count(column)))
                edges = _width_edges(column, k)
            else:
                edges = self._adaptive_edges(column)
            bins, _ = np.histogram(column, edges)
            self.edges_.append(edges)
            self.counts_.append(bins)
            counts.append(len(edges) - 1)
        self.n_bins_ = np.array(counts)
        self.scores_ = self.score_samples(X)
        self.threshold_ = self._threshold()
        return self

    def _adaptive_edges(self, column: np.ndarray) -> np.ndarray:
        """Pick the quantile bin count that maximises (4)."""
        best_score, best = -np.inf, _frequency_edges(column, 2)
        for k in range(2, min(self.max_bins, max(3, len(column) // 2)) + 1):
            edges = _frequency_edges(column, k)
            score = _loo_score(column, edges)
            if score > best_score:
                best_score, best = score, edges
        return best

    def _threshold(self) -> float:
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        dof = 2 * self.n_features_
        return float(_chi2_ppf(1.0 - self.alpha, dof) / 2.0)

    # -- scoring ---------------------------------------------------------- #
    def feature_scores(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contributions to (2), shape (m, d)."""
        X = self._check(X)
        out = np.empty(X.shape)
        for j in range(self.n_features_):
            edges, counts = self.edges_[j], self.counts_[j]
            widths = np.diff(edges)
            k = len(counts)
            index = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                            0, k - 1)
            # Equation (1): never a zero density, so one point in a bin is
            # enough to make it plausible and an empty bin is not a wall.
            density = np.maximum(counts[index], 1) / (self.n_ * widths[index])
            outside = (X[:, j] < edges[0]) | (X[:, j] > edges[-1])
            # A value beyond the fitted range is worth half a point in the edge
            # bin: less plausible than anything observed, still finite.
            edge = np.where(X[:, j] < edges[0], 0, k - 1)
            density = np.where(outside, 0.5 / (self.n_ * widths[edge]), density)
            out[:, j] = -np.log(np.maximum(density, FLOOR))
        return out

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.feature_scores(X).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def _check(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "n_features_"):
            raise RuntimeError("fit before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X


# --------------------------------------------------------------------------- #
# Chi-square quantile, without scipy
# --------------------------------------------------------------------------- #
def _gammainc(a: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = total = 1.0 / a
        for i in range(1, 1000):
            term *= x / (a + i)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_ppf(q: float, df: int) -> float:
    lo, hi = 0.0, max(1.0, float(df))
    while _gammainc(df / 2.0, hi / 2.0) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc(df / 2.0, mid / 2.0) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based area under the ROC curve, ties averaged."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    positive = labels.astype(bool)
    n1 = int(positive.sum())
    n0 = len(labels) - n1
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _demo_product() -> None:
    """Additivity, the manual density, and the joint impossibility HBOS cannot see."""
    rng = np.random.default_rng(4)
    n = 400
    a = rng.normal(size=n)
    b = 0.97 * a + np.sqrt(1.0 - 0.97 ** 2) * rng.normal(size=n)
    good = np.column_stack([a, b])
    radius = rng.uniform(0.8, 1.5, size=40)
    bad = np.column_stack([radius, -radius])
    X = np.vstack([good, bad])
    labels = np.zeros(len(X), dtype=bool)
    labels[n:] = True

    # Equation (2) is a sum, and feature_scores is exactly its terms.
    model = HBOS(variant="static_bins", contamination=0.1).fit(X)
    parts = model.feature_scores(X)
    assert np.allclose(parts.sum(axis=1), model.score_samples(X), atol=1e-12)

    # And each term is minus the log of the marginal density in (1).
    j = 0
    edges, counts = model.edges_[j], model.counts_[j]
    index = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                    0, len(counts) - 1)
    manual = -np.log(counts[index] / (model.n_ * np.diff(edges)[index]))
    assert np.allclose(manual, parts[:, j], atol=1e-12)

    rows = {}
    for variant in VARIANTS:
        fitted = HBOS(variant=variant, contamination=0.1).fit(X)
        scores = fitted.score_samples(X)
        diag = float(fitted.score_samples(np.array([[1.2, 1.2]]))[0])
        anti = float(fitted.score_samples(np.array([[1.2, -1.2]]))[0])
        rows[variant] = (int(fitted.n_bins_.sum()), _auc(scores, labels),
                         float(scores[labels].mean()),
                         float(scores[~labels].mean()),
                         float(np.mean(fitted.predict(X)[labels] == -1)),
                         diag, anti)
    assert rows["static_bins"][1] < 0.45
    assert rows["adaptive"][1] < 0.5
    assert all(row[4] == 0.0 for row in rows.values())

    mean = good.mean(axis=0)
    cov = np.cov(good.T)
    precision = np.linalg.inv(cov)

    def nll(point: np.ndarray) -> float:
        diff = point - mean
        return float(0.5 * diff @ precision @ diff)

    joint = abs(nll(np.array([1.2, 1.2])) - nll(np.array([1.2, -1.2])))

    print(f"product: the two features have correlation "
          f"{np.corrcoef(good.T)[0, 1]:.3f}, and 40 planted points sit on the "
          f"anti-diagonal at radius 0.8 to 1.5 -- every coordinate inside the "
          f"data's range and near its mode, but jointly impossible. Scoring the "
          f"correlated pair with each variant:")
    for variant, row in rows.items():
        print(f"  {variant:13s} bins {row[0]:3d}  AUC {row[1]:.3f}  "
              f"planted {row[2]:5.2f}  normals {row[3]:5.2f}  flagged "
              f"{row[4]:.0%}")
    print(f"  Equation (2) gives (1.2, 1.2) and (1.2, -1.2) scores of "
          f"{rows['static_bins'][5]:.2f} and {rows['static_bins'][6]:.2f} -- a "
          f"gap of {abs(rows['static_bins'][5] - rows['static_bins'][6]):.2f} "
          f"nats, all of it the sampling noise in two marginal bin counts. The "
          f"true bivariate Gaussian log-density differs by {joint:.1f} nats. "
          f"Every variant puts the planted points *below* the normals (AUC "
          f"{min(r[1] for r in rows.values()):.3f} to "
          f"{max(r[1] for r in rows.values()):.3f}) and flags none of them: a "
          f"point that is ordinary in each feature and impossible in the pair "
          f"is the one thing a product model cannot represent.")


def _demo_bins() -> None:
    """A fixed count is a fixed resolution."""
    rng = np.random.default_rng(12)
    d = 2
    features = 600

    def peaky(size: int) -> np.ndarray:
        core = rng.random(size) < 0.85
        narrow = rng.normal(scale=0.15, size=(size, d))
        wide = rng.normal(scale=4.0, size=(size, d))
        return np.where(core[:, None], narrow, wide)

    sharp = peaky(features)
    sharp_test = peaky(3000)
    smooth = rng.normal(size=(features, d))
    smooth_test = rng.normal(size=(3000, d))

    table = {}
    for name, train, test in (("peaky", sharp, sharp_test),
                              ("gaussian", smooth, smooth_test)):
        table[name] = {v: (int(HBOS(variant=v, contamination=0.1).fit(train)
                               .n_bins_.sum()),
                           float(HBOS(variant=v, contamination=0.1).fit(train)
                                 .score_samples(test).mean()))
                       for v in VARIANTS}
    assert table["peaky"]["static_bins"][1] > 2.5
    assert table["peaky"]["adaptive"][1] < 1.2
    assert table["gaussian"]["static_bins"][1] <= table["gaussian"]["adaptive"][1]

    skew = np.exp(rng.normal(size=features))
    fitted = {v: HBOS(variant=v, contamination=0.05).fit(skew[:, None])
              for v in VARIANTS}
    counts = {v: len(m.counts_[0]) for v, m in fitted.items()}
    share = {v: float(m.counts_[0].max() / features) for v, m in fitted.items()}
    curve = [(k, _loo_score(skew, _frequency_edges(skew, k)))
             for k in (16, 24, 32, 48)]
    assert share["static_bins"] > 0.5 and share["adaptive"] < 0.1
    assert curve[-1][1] < curve[1][1]

    print("bins: held-out mean negative log density (nats, lower is a better "
          "density model) for 2 features, trained on 600 points and scored on "
          "3000 fresh ones:")
    for name, row in table.items():
        cells = "  ".join(f"{v[:8]:8s} {row[v][1]:5.2f} ({row[v][0]:3d} bins)"
                          for v in VARIANTS)
        print(f"  {name:9s} {cells}")
    print(f"  On the peaky sample -- 85% inside a Gaussian of scale 0.15 and "
          f"15% inside one of scale 4 -- ten equal-width bins over the whole "
          f"range put the entire core in one bin, and the estimate costs "
          f"{table['peaky']['static_bins'][1]:.2f} nats per point against "
          f"{table['peaky']['adaptive'][1]:.2f} for quantile bins. On a plain "
          f"Gaussian the ordering reverses: {table['gaussian']['static_bins'][1]:.2f} "
          f"against {table['gaussian']['adaptive'][1]:.2f}, because 20 bins "
          f"already match the scale of the data and the extra ones are noise. "
          f"On a lognormal feature the three rules choose "
          f"{counts['static_bins']}, {counts['dynamic_bins']} and "
          f"{counts['adaptive']} bins, and the fixed count puts "
          f"{share['static_bins']:.0%} of the sample in its largest bin against "
          f"{share['adaptive']:.0%} for quantile bins. The leave-one-out "
          f"criterion (4) is {curve[0][1]:.4f} at k = {curve[0][0]}, "
          f"{curve[1][1]:.4f} at k = {curve[1][0]} and {curve[-1][1]:.4f} at "
          f"k = {curve[-1][0]}: a flat optimum, which is why the variant can "
          f"land anywhere inside it and still be right.")


def _simulated_threshold(model: HBOS, reps: int = 20000, seed: int = 0) -> float:
    """The 95th percentile of the score under the fitted product model.

    A point drawn from the fitted model lands in bin b of feature j with
    probability N_b / n, and its contribution there is known, so the null
    distribution of (2) can be sampled exactly without touching the data.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(reps)
    for j in range(model.n_features_):
        widths = np.diff(model.edges_[j])
        value = -np.log(np.maximum(model.counts_[j], 1) / (model.n_ * widths))
        mass = model.counts_[j] / model.n_
        total += value[rng.choice(len(mass), size=reps, p=mass)]
    return float(np.quantile(total, 0.95))


def _demo_calibration() -> None:
    """Equation (5) is optimistic; the fitted model is not."""
    rng = np.random.default_rng(1)
    rows = {}
    for d in (2, 4, 8):
        train = rng.normal(size=(500, d))
        test = rng.normal(size=(4000, d))
        rates = []
        for variant in VARIANTS:
            model = HBOS(variant=variant, contamination=None, alpha=0.05).fit(train)
            rates.append(float(np.mean(model.predict(test) == -1)))
        fitted = HBOS(variant="static_bins", contamination=None).fit(train)
        simulated = float(np.mean(fitted.score_samples(test)
                                  > _simulated_threshold(fitted)))
        empirical = HBOS(variant="static_bins", contamination=0.05).fit(train)
        rows[d] = (rates, simulated,
                   float(np.mean(empirical.predict(train) == -1)),
                   float(_chi2_ppf(0.95, 2 * d) / 2.0),
                   float(fitted.score_samples(test).mean() / d))
    assert rows[2][0][0] < 0.10
    assert rows[8][0][0] > 3.0 * 0.05
    assert abs(rows[2][1] - 0.05) < 0.02
    assert all(row[2] <= 0.05 for row in rows.values())

    growth = {}
    for n in (500, 2000, 8000):
        train = rng.normal(size=(n, 8))
        test = rng.normal(size=(4000, 8))
        model = HBOS(variant="static_bins", contamination=None).fit(train)
        scores = model.score_samples(test)
        growth[n] = (float(np.mean(scores > _simulated_threshold(model))),
                     float(np.mean(scores > rows[8][3])))
    assert growth[8000][0] < growth[500][0]
    assert growth[8000][1] > 2.5 * growth[8000][0]

    print("calibration: flag rate on 4000 held-out standard normal points at a "
          "nominal 5%, using the chi-square threshold of (5):")
    for d, (rates, simulated, _, limit, _) in rows.items():
        cells = "  ".join(f"{v[:8]:8s} {r:5.1%}" for v, r in zip(VARIANTS, rates,
                                                                 strict=True))
        print(f"  d = {d}  threshold {limit:5.2f}  {cells}  simulated "
              f"{simulated:5.1%}")
    print(f"  Every variant over-flags, and by more as the dimension grows: "
          f"{rows[2][0][0]:.1%} at d = 2, {rows[4][0][0]:.1%} at d = 4, "
          f"{rows[8][0][0]:.1%} at d = 8 against a nominal 5%. The law is not "
          f"merely approximate, it is mis-scaled: $-\\log \\hat p_j$ has mean "
          f"equal to the marginal's differential entropy, {_entropy:.3f} for a "
          f"standard normal, not the 1 that (5) assumes, and ten bins measure "
          f"{rows[2][4]:.3f} because a step function is not the density. The "
          f"score therefore sits {rows[2][4] - 1.0:.3f}·d above where chi-square "
          f"puts it while the threshold allows only $1.645\\sqrt{{d}}$ of tail "
          f"room, so the calibration decays like $\\sqrt{{d}}$. Resampling the "
          f"score from the fitted model -- 20 000 seeded draws, no data needed "
          f"-- gives {rows[2][1]:.1%}, {rows[4][1]:.1%} and {rows[8][1]:.1%}, "
          f"and at d = 8 that improves from {growth[500][0]:.1%} at n = 500 to "
          f"{growth[2000][0]:.1%} at n = 2000 and then stops at "
          f"{growth[8000][0]:.1%} at n = 8000, because the residue is no longer "
          f"estimation but the gap between a ten-bin histogram and the Gaussian "
          f"it is modelling. The chi-square rate stays at "
          f"{growth[8000][1]:.1%} throughout: it is measuring nothing. An "
          f"empirical threshold flags at most the requested share -- "
          f"{rows[2][2]:.1%} of the training sample here, where ties at the "
          f"quantile resolve outward -- at every dimension, and makes no "
          f"distributional claim that could be wrong.")


def _demo_broken() -> None:
    """Bad inputs, and the degenerate columns that are not errors."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    cases = [
        ("an unknown variant", lambda: HBOS(variant="quantile_bins")),
        ("zero bins", lambda: HBOS(n_bins=0)),
        ("one max bin", lambda: HBOS(max_bins=1)),
        ("contamination 0", lambda: HBOS(contamination=0.0)),
        ("contamination 1", lambda: HBOS(contamination=1.0)),
        ("alpha 0", lambda: HBOS(alpha=0.0)),
        ("two points", lambda: HBOS().fit(X[:2])),
        ("scoring before fitting", lambda: HBOS().score_samples(X)),
        ("predicting before fitting", lambda: HBOS().predict(X)),
        ("the wrong width", lambda: HBOS().fit(X).score_samples(X[:, :2])),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name} did not raise")

    constant = np.column_stack([np.full(50, 3.0), rng.normal(size=50)])
    flat = HBOS(contamination=0.1).fit(constant)
    assert np.allclose(flat.feature_scores(constant)[:, 0], 0.0)

    repeated = np.repeat(rng.normal(size=(4, 2)), 10, axis=0)
    coarse = HBOS(variant="adaptive", contamination=0.1).fit(repeated)
    assert np.isfinite(coarse.score_samples(repeated)).all()
    occupied = int(np.count_nonzero(coarse.counts_[0]))
    assert occupied <= 4

    wide = HBOS(variant="static_bins", n_bins=1, contamination=0.1).fit(X)
    assert np.allclose(wide.scores_, wide.scores_[0])
    outside = float(wide.score_samples(np.array([[50.0, 50.0, 50.0]]))[0])

    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. Three degenerate cases are "
          f"answers rather than errors. A constant column gets one bin of width "
          f"one holding all {50} points, so its density is exactly 1 and its "
          f"contribution is exactly 0 -- such a column cannot distinguish "
          f"anything and saying so is correct. Four distinct values repeated "
          f"ten times each occupy {occupied} quantile bins, because duplicate "
          f"edges are removed and the rest interpolate across the gaps; "
          f"`adaptive` narrowing itself is the right response, not a division by "
          f"a zero-width bin. And `n_bins=1` puts everything in one bin: every "
          f"training point is tied at {wide.scores_[0]:.2f} and a point far "
          f"outside scores {outside:.2f}, exactly "
          f"{3.0 * math.log(2 * len(X)):.2f} more -- the price of being outside "
          f"at all, and the same number whether the point is one unit or a "
          f"thousand beyond the edge. A single bin carries the total mass and "
          f"no shape, which is the smallest illustration of why the bin count "
          f"is the method.")


def _demo_units() -> None:
    """A score that changes with the units cannot have a fixed threshold."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 2))
    test = rng.normal(size=(4000, 2))
    rows = {}
    for scale in (1.0, 1e3, 1e6):
        moved = X.copy()
        moved[:, 0] *= scale
        probes = test.copy()
        probes[:, 0] *= scale
        model = HBOS(variant="static_bins", contamination=None).fit(moved)
        scores = model.score_samples(probes)
        rows[scale] = (float(scores.mean()),
                       float(np.mean(scores > _chi2_ppf(0.95, 4) / 2.0)),
                       float(np.mean(scores > _simulated_threshold(model))))
    shifts = [rows[s][0] - rows[1.0][0] for s in (1e3, 1e6)]
    assert abs(shifts[0] - math.log(1e3)) < 1e-6
    assert abs(shifts[1] - math.log(1e6)) < 1e-6
    assert rows[1e6][1] > 0.9
    assert all(abs(row[2] - 0.05) < 0.02 for row in rows.values())

    print("units: the same standard normal data with the first feature "
          "multiplied by a constant, and the same score threshold in (5):")
    for scale, (mean, chi, simulation) in rows.items():
        print(f"  x1 x {scale:<7.0e}  mean score {mean:6.2f}  "
              f"chi-square flags {chi:5.1%}  resampled flags {simulation:5.1%}")
    print(f"  Rescaling one feature by 1000 shifts every score by exactly "
          f"$\\log 1000 = {shifts[0]:.2f}$ nats -- the density has units, so the "
          f"negative log density does too -- and the fixed chi-square threshold "
          f"then flags {rows[1e3][1]:.0%} of the data instead of 5%. At "
          f"$10^6$ the shift is {shifts[1]:.2f} and everything is an outlier. "
          f"Any calibrated threshold must move with the score, and (5) cannot: "
          f"it is a constant. A threshold resampled from the fitted model shifts "
          f"with the units by construction and stays at "
          f"{rows[1e6][2]:.0%}, which is the honest version of the same idea.")


def main() -> None:
    _demo_product()
    _demo_bins()
    _demo_calibration()
    _demo_units()
    _demo_broken()
    print("hbos demo passed")


if __name__ == "__main__":
    main()
