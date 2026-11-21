r"""Change-point detection — where the process changed, not just that it is odd.

Chapter:   09-density-and-anomaly
Companion: change_point.md
Tier 2 · 5 variants: bocpd, pelt, cusum, binary_segmentation, window_based

    cost:      C(a, b) = (b-a)/2 [ log 2 pi + 1 + log( SS(a,b)/(b-a) ) ]      (1)
    objective: min over 0 = t0 < ... < tm = n of
               sum_k C(t_{k-1}, t_k) + beta m                                 (2)
    pelt:      F(t) = min_{s in R_t} F(s) + C(s, t) + beta                    (3)
    cusum:     g_t = max(0, g_{t-1} + x_t - mu_0 - k)                         (4)
    bocpd:     p(r_t | x_1:t) prop. p(r_t | r_{t-1}) p(x_t | r_{t-1})
               times p(r_{t-1} | x_1:t-1)                                     (5)
    window:    G(t) = w [ 2 log v - log v_L - log v_R ]                       (6)

An anomaly detector asks whether a point is unusual. A change-point detector asks
a different question: whether the *process* that produced the points is the same
one as before. The answers are not related. A single spike is anomalous and is
not a change point; a shift from 0 to 0.01 standard deviations is not anomalous
at any single point and is a change point if it lasts. This atom is the second
question, and every variant is a different answer to "how do you compare the
recent past with the distant past without being fooled by noise".

One is an exact optimiser, one is greedy, and three are detectors:

- `pelt` minimises the penalised segmentation objective (2) exactly, in time
  that is linear in the number of change points rather than quadratic in $n$.
- `binary_segmentation` optimises the same objective greedily, splitting at the
  best point and recursing. Faster, not exact.
- `cusum` is Page's sequential test: accumulate the evidence for a shift and
  alarm when it crosses a threshold. The classical online detector.
- `bocpd` is the Bayesian version: keep a posterior over the *run length* since
  the last change and let the data move it. Online, and it returns a
  distribution rather than a point.
- `window_based` runs a likelihood-ratio test between two adjacent windows, so
  it sees a change in variance as well as one in mean.

The distinction that matters in practice is online versus offline. `cusum` and
`bocpd` decide at time $t$ using only $x_{1:t}$ and can be deployed on a stream;
`pelt`, `binary_segmentation` and `window_based` look at the whole series and are
run after the fact. An online detector necessarily pays a detection delay; the
demo measures it.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("bocpd", "pelt", "cusum", "binary_segmentation", "window_based")
_LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Segment costs from prefix sums
# --------------------------------------------------------------------------- #
def _prefix(x: np.ndarray) -> tuple:
    """Cumulative sums of x and x**2, with a leading zero, for O(1) segments."""
    return np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]


def _costs(s1: np.ndarray, s2: np.ndarray, starts: np.ndarray, end: int,
           model: str, sigma2: float) -> np.ndarray:
    """C(a, end) for every a in `starts`, vectorised.

    `gaussian` is the profile negative log-likelihood of a segment with its own
    mean and variance, equation (1). `mean` fixes the variance at `sigma2`, so a
    segment that changes only its spread costs nothing extra.
    """
    m = end - starts
    total = s1[end] - s1[starts]
    total2 = s2[end] - s2[starts]
    ss = np.maximum(total2 - total * total / m, 1e-300)
    if model == "mean":
        return ss / (2.0 * sigma2)
    return 0.5 * m * (_LOG2PI + 1.0 + np.log(ss / m))


def _robust_variance(x: np.ndarray) -> float:
    """0.5 E[(x_t - x_{t-1})^2]: unbiased for the noise, blind to the changes."""
    return float(0.5 * np.mean(np.diff(x) ** 2))


# --------------------------------------------------------------------------- #
# Exact: PELT
# --------------------------------------------------------------------------- #
def _pelt(x: np.ndarray, model: str, penalty: float, min_size: int,
          sigma2: float) -> tuple:
    """Pruned exact linear time; returns (changepoints, evaluations, cost)."""
    n = len(x)
    s1, s2 = _prefix(x)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)
    candidates = [0]
    evaluations = 0
    for t in range(min_size, n + 1):
        starts = np.array([s for s in candidates if t - s >= min_size])
        if starts.size == 0:
            continue
        reach = best[starts] + _costs(s1, s2, starts, t, model, sigma2)
        evaluations += starts.size
        k = int(np.argmin(reach))
        best[t] = reach[k] + penalty
        prev[t] = int(starts[k])
        # keep s only while it can still beat the best segmentation ending here
        keep = {int(s) for s, ok in zip(starts, reach <= best[t], strict=True) if ok}
        candidates = [s for s in candidates if s in keep or t - s < min_size]
        if t + min_size <= n:
            candidates.append(t)
    if not np.isfinite(best[n]):
        return np.array([], dtype=int), evaluations, np.inf
    path, t = [], n
    while t > 0:
        t = int(prev[t])
        if t:
            path.append(t)
    return np.array(sorted(path), dtype=int), evaluations, float(best[n])


# --------------------------------------------------------------------------- #
# Greedy: binary segmentation
# --------------------------------------------------------------------------- #
def _binary_segmentation(x: np.ndarray, model: str, penalty: float,
                         min_size: int, sigma2: float) -> tuple:
    """Split at the best point while the gain exceeds the penalty."""
    n = len(x)
    s1, s2 = _prefix(x)
    found = []
    stack = [(0, n)]
    evaluations = 0
    while stack:
        a, b = stack.pop()
        starts = np.arange(a + min_size, b - min_size + 1)
        if starts.size == 0:
            continue
        base = _costs(s1, s2, np.array([a]), b, model, sigma2)[0]
        left = _costs(s1, s2, np.full(starts.size, a), starts, model, sigma2)
        right = _costs(s1, s2, starts, np.full(starts.size, b), model, sigma2)
        evaluations += starts.size
        gain = base - left - right - penalty
        k = int(np.argmax(gain))
        if gain[k] <= 0.0:
            continue
        t = int(starts[k])
        found.append(t)
        stack.append((a, t))
        stack.append((t, b))
    return np.array(sorted(found), dtype=int), evaluations


# --------------------------------------------------------------------------- #
# Online: CUSUM
# --------------------------------------------------------------------------- #
def _cusum(x: np.ndarray, threshold: float, drift: float, min_size: int,
           sigma2: float) -> tuple:
    """Two-sided Page CUSUM with the reference re-estimated after each alarm.

    The noise scale is global -- `0.5 E[(x_t - x_{t-1})^2]` -- because the
    segment-local estimate is too noisy for a threshold that is exponential in
    it. The change-point estimate is the last index at which the winning
    statistic was zero: the start of the excursion, not the alarm.
    """
    n = len(x)
    sd = math.sqrt(sigma2)
    k, h = drift * sd, threshold * sd
    found, alarms = [], []
    start = 0
    while start + min_size < n:
        warm = x[start:min(start + max(20, min_size), n)]
        mu = float(warm.mean())
        up = down = 0.0
        last_up = last_down = start
        hit = None
        for t in range(start + len(warm), n):
            up = max(0.0, up + (x[t] - mu) - k)
            down = max(0.0, down - (x[t] - mu) - k)
            if up == 0.0:
                last_up = t + 1
            if down == 0.0:
                last_down = t + 1
            if up > h or down > h:
                hit = (t, last_up if up > h else last_down)
                break
        if hit is None:
            break
        alarms.append(hit[0])
        found.append(hit[1])
        start = max(hit[0], start + min_size)
    return np.array(found, dtype=int), np.array(alarms, dtype=int)


# --------------------------------------------------------------------------- #
# Online: Bayesian run-length posterior
# --------------------------------------------------------------------------- #
def _student_t_logpdf(x: float, df: np.ndarray, loc: np.ndarray,
                      scale: np.ndarray, log_norm: np.ndarray) -> np.ndarray:
    """log density of a Student-t, vectorised over the run lengths.

    `log_norm` is the run-length-only part of the normaliser, precomputed because
    the degrees of freedom are fixed by the run length alone.
    """
    z = (x - loc) / scale
    return (log_norm - np.log(scale)
            - 0.5 * (df + 1.0) * np.log1p(z * z / df))


def _bocpd(x: np.ndarray, hazard: float, kappa0: float, alpha0: float,
           beta0: float, min_size: int) -> tuple:
    """Adams-MacKay run-length recursion with a Normal-Inverse-Gamma prior."""
    n = len(x)
    # centre the prior on the series mean rather than the first point: a prior
    # centred on x[0] makes every value near it look like a plausible restart,
    # which manufactures change points wherever the series revisits its origin
    mu0 = float(x.mean())
    alpha_grid = alpha0 + 0.5 * np.arange(n + 1)
    log_norm = (np.array([math.lgamma(a + 0.5) for a in alpha_grid])
                - np.array([math.lgamma(a) for a in alpha_grid])
                - 0.5 * np.log(2.0 * alpha_grid * math.pi))
    # predictive density of a fresh observation under the prior, for the
    # r_t = 0 branch: without it the change branch would be scored as if every
    # value were equally likely a priori
    prior_df = 2.0 * alpha0
    prior_scale = math.sqrt(beta0 * (kappa0 + 1.0) / (alpha0 * kappa0))
    prior_norm = (math.lgamma(alpha0 + 0.5) - math.lgamma(alpha0)
                  - 0.5 * math.log(2.0 * alpha0 * math.pi))
    mass = np.zeros(n + 1)
    mass[0] = 1.0
    mu = np.full(n + 1, mu0)
    kappa = np.full(n + 1, kappa0)
    alpha = np.full(n + 1, alpha0)
    beta = np.full(n + 1, beta0)
    map_run = np.zeros(n, dtype=int)
    short_mass = np.zeros(n)
    for t in range(n):
        m = t + 1
        df = 2.0 * alpha_grid[:m]
        scale = np.sqrt(beta[:m] * (kappa[:m] + 1.0)
                        / (alpha_grid[:m] * kappa[:m]))
        grow = mass[:m] * np.exp(_student_t_logpdf(x[t], df, mu[:m], scale,
                                                   log_norm[:m]))
        z = (x[t] - mu0) / prior_scale
        prior_density = math.exp(prior_norm - math.log(prior_scale)
                                 - 0.5 * (prior_df + 1.0)
                                 * math.log1p(z * z / prior_df))
        updated = np.empty(m + 1)
        updated[0] = hazard * prior_density
        updated[1:] = (1.0 - hazard) * grow
        total = float(updated.sum())
        if total <= 0.0 or not np.isfinite(total):
            mass[:m + 1] = 0.0
            mass[0] = 1.0
            continue
        mass[:m + 1] = updated / total
        kappa_new = kappa[:m] + 1.0
        mu[1:m + 1] = (kappa[:m] * mu[:m] + x[t]) / kappa_new
        kappa[1:m + 1] = kappa_new
        alpha[1:m + 1] = alpha[:m] + 0.5
        beta[1:m + 1] = beta[:m] + 0.5 * (kappa[:m] * (x[t] - mu[:m]) ** 2
                                          / kappa_new)
        mu[0], kappa[0], alpha[0], beta[0] = mu0, kappa0, alpha0, beta0
        map_run[t] = int(np.argmax(mass[:m + 1]))
        short_mass[t] = float(mass[:min_size].sum())
    # The signal is the MAP run length collapsing, not the mass at zero: a
    # single surprising point lifts p(r_t = 0) without moving the mode. Estimate
    # the change as the index the collapsed run length points back to.
    found = []
    for t in range(1, n):
        previous = map_run[t - 1]
        if map_run[t] < previous and map_run[t] <= max(min_size, 0.2 * previous):
            found.append(max(0, t - int(map_run[t])))
    merged = []
    for t in found:
        if not merged or t - merged[-1] >= min_size:
            merged.append(t)
    return np.array(merged, dtype=int), map_run, short_mass


# --------------------------------------------------------------------------- #
# Window: likelihood ratio between adjacent halves
# --------------------------------------------------------------------------- #
def _window(x: np.ndarray, window: int, threshold: float | None,
            min_size: int, alpha: float) -> tuple:
    """Generalised likelihood ratio between the left and right windows.

    For two windows of length w the statistic is

        G = w [ 2 log v - log v_L - log v_R ],

    with v the variance of the pooled window and v_L, v_R the within-window
    variances. Under no change G ~ chi-square with two degrees of freedom, so
    the threshold is the chi-square quantile at alpha divided over the number of
    positions -- a Bonferroni correction, not a tuning knob.
    """
    n = len(x)
    statistic = np.zeros(n)
    for t in range(window, n - window + 1):
        left, right = x[t - window:t], x[t:t + window]
        v1 = float(left.var(ddof=1))
        v2 = float(right.var(ddof=1))
        if v1 <= 0.0 or v2 <= 0.0:
            continue
        m1, m2 = float(left.mean()), float(right.mean())
        pooled = 0.5 * (v1 + v2) + 0.25 * (m1 - m2) ** 2
        statistic[t] = window * (2.0 * math.log(pooled)
                                 - math.log(v1) - math.log(v2))
    if threshold is None:
        positions = max(1, n - 2 * window + 1)
        threshold = -2.0 * math.log(alpha / positions)
    peaks = []
    for t in range(window, n - window + 1):
        if statistic[t] <= threshold:
            continue
        if statistic[t] < statistic[t - 1] or statistic[t] < statistic[t + 1]:
            continue
        peaks.append(t)
    # A change smears the statistic over roughly one window either side, so
    # merge within a window and keep the strongest point of each cluster.
    merged = []
    for t in peaks:
        if not merged or t - merged[-1] >= window:
            merged.append(t)
        elif statistic[t] > statistic[merged[-1]]:
            merged[-1] = t
    return np.array(merged, dtype=int), statistic, float(threshold)


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class ChangePoint:
    """Find where a one-dimensional sequence changes behaviour.

    Parameters
    ----------
    variant : {"bocpd", "pelt", "cusum", "binary_segmentation", "window_based"}
        Which detector or optimiser to run.
    model : {"gaussian", "mean"}
        Cost model for `pelt` and `binary_segmentation`. `gaussian` lets each
        segment have its own variance, so a change in spread is a change point;
        `mean` fixes the variance, so it is not.
    penalty : float or None
        Cost of adding a change point, in nats. `None` uses the BIC penalty:
        2 log n for `gaussian`, log n for `mean`.
    min_size : int
        Minimum segment length. Also the minimum distance between two reported
        change points.
    threshold : float or None
        For `cusum`, the alarm threshold in standard deviations of the noise.
        8 gives a one-sided average run length of about 17000 samples under the
        null when the reference mean is known; estimating that mean from a short
        warm-up shortens it by roughly a factor of eight (the demo measures
        this). For `window_based`, an explicit statistic threshold; `None` uses
        the chi-square quantile of `alpha`.
    drift : float
        CUSUM allowance in standard deviations: the half-shift the test is tuned
        to detect quickly.
    window : int
        Half-width of the windows compared by `window_based`.
    alpha : float
        Family-wise false-alarm rate for `window_based`'s chi-square threshold,
        spread over the number of window positions.
    hazard : float
        BOCPD prior probability that a change happens at any given step. 1/250
        means a change every 250 steps a priori.
    kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma prior. `beta0=None` sets it to half the robust
        variance of the series.

    Attributes
    ----------
    changepoints_ : ndarray of int
        Indices at which a new segment starts, ascending.
    statistic_ : ndarray
        The per-step statistic for `window_based` (the likelihood ratio) and
        `bocpd` (posterior mass on run lengths shorter than `min_size`). Zero for
        the offline optimisers and for `cusum`, which reports `alarms_` instead.
    run_length_ : ndarray of int
        `bocpd` only: the MAP run length at each step.
    alarms_ : ndarray of int
        `cusum` only: the indices at which the statistic crossed the threshold,
        which trail `changepoints_` by the detection delay.
    cost_ : float
        `pelt` and `binary_segmentation`: the objective (2) at the solution.
    evaluations_ : int
        Candidate splits costed. PELT's pruning is what keeps this small.
    """

    def __init__(self, *, variant: str = "pelt", model: str = "gaussian",
                 penalty: float | None = None, min_size: int = 2,
                 threshold: float | None = None, drift: float = 0.5,
                 window: int = 40, alpha: float = 0.05,
                 hazard: float = 1.0 / 250.0,
                 kappa0: float = 1.0, alpha0: float = 1.0,
                 beta0: float | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if model not in ("gaussian", "mean"):
            raise ValueError("model must be 'gaussian' or 'mean'")
        if min_size < 1:
            raise ValueError("min_size must be positive")
        if penalty is not None and penalty < 0.0:
            raise ValueError("penalty must be non-negative or None")
        if drift < 0.0:
            raise ValueError("drift must be non-negative")
        if window < 2:
            raise ValueError("window must be at least two")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must lie in (0, 1)")
        if kappa0 <= 0.0 or alpha0 <= 0.0 or (beta0 is not None and beta0 <= 0.0):
            raise ValueError("prior parameters must be positive")
        self.variant = variant
        self.model = model
        self.penalty = penalty
        self.min_size = int(min_size)
        self.threshold = threshold
        self.drift = float(drift)
        self.window = int(window)
        self.alpha = float(alpha)
        self.hazard = float(hazard)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = beta0

    def fit(self, x: np.ndarray) -> ChangePoint:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 2 * self.min_size:
            raise ValueError(f"need at least {2 * self.min_size} points")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        self.n_ = x.size
        sigma2 = _robust_variance(x)
        self.sigma2_ = sigma2
        self.penalty_ = self._penalty()

        if self.variant in ("pelt", "binary_segmentation"):
            if self.variant == "pelt":
                cps, evaluations, cost = _pelt(x, self.model, self.penalty_,
                                               self.min_size, sigma2)
            else:
                cps, evaluations = _binary_segmentation(
                    x, self.model, self.penalty_, self.min_size, sigma2)
                cost = self._objective(x, cps, sigma2)
            self.changepoints_ = cps
            self.evaluations_ = evaluations
            self.cost_ = cost
            self.statistic_ = np.zeros(x.size)
        elif self.variant == "cusum":
            threshold = 8.0 if self.threshold is None else float(self.threshold)
            cps, alarms = _cusum(x, threshold, self.drift, self.min_size,
                                 sigma2)
            self.changepoints_ = cps
            self.alarms_ = alarms
            self.statistic_ = np.zeros(x.size)
            self.evaluations_ = int(x.size)
            self.cost_ = float("nan")
        elif self.variant == "window_based":
            cps, statistic, threshold = _window(x, self.window, self.threshold,
                                                self.min_size, self.alpha)
            self.changepoints_ = cps
            self.statistic_ = statistic
            self.threshold_ = threshold
            self.evaluations_ = int(x.size)
            self.cost_ = float("nan")
        else:
            beta0 = 0.5 * sigma2 if self.beta0 is None else float(self.beta0)
            cps, map_run, short_mass = _bocpd(x, self.hazard, self.kappa0,
                                              self.alpha0, beta0, self.min_size)
            self.changepoints_ = cps
            self.run_length_ = map_run
            self.statistic_ = short_mass
            self.evaluations_ = int(x.size)
            self.cost_ = float("nan")
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).changepoints_

    def _penalty(self) -> float:
        if self.penalty is not None:
            return float(self.penalty)
        parameters = 2.0 if self.model == "gaussian" else 1.0
        return parameters * math.log(max(self.n_, 2))

    def _objective(self, x: np.ndarray, cps: np.ndarray, sigma2: float) -> float:
        s1, s2 = _prefix(x)
        edges = [0, *[int(c) for c in cps], x.size]
        total = self.penalty_ * len(cps)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            total += float(_costs(s1, s2, np.array([a]), b, self.model, sigma2)[0])
        return total


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _series(rng: np.random.Generator, segments: list) -> np.ndarray:
    """Concatenate (length, mean, sd) blocks into one series."""
    parts = [mean + sd * rng.normal(size=length) for length, mean, sd in segments]
    return np.concatenate(parts)


def _naive_dp(x: np.ndarray, model: str, penalty: float, min_size: int,
              sigma2: float) -> tuple:
    """The O(n^2) recursion PELT prunes; used only to check PELT."""
    n = len(x)
    s1, s2 = _prefix(x)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)
    evaluations = 0
    for t in range(min_size, n + 1):
        starts = np.arange(0, t - min_size + 1)
        costs = best[starts] + _costs(s1, s2, starts, t, model, sigma2) + penalty
        evaluations += starts.size
        k = int(np.argmin(costs))
        best[t] = costs[k]
        prev[t] = int(starts[k])
    path, t = [], n
    while t > 0:
        t = int(prev[t])
        if t:
            path.append(t)
    return np.array(sorted(path), dtype=int), evaluations, float(best[n])


def _demo_segment() -> None:
    """The offline variants on a series with three changes."""
    rng = np.random.default_rng(0)
    x = _series(rng, [(90, 0.0, 1.0), (90, 2.0, 1.0),
                      (90, -1.0, 1.0), (90, 0.5, 1.0)])
    truth = np.array([90, 180, 270])
    rows = {}
    for label, kwargs in (
        ("pelt/gaussian", {"variant": "pelt"}),
        ("pelt/mean", {"variant": "pelt", "model": "mean"}),
        ("binary", {"variant": "binary_segmentation"}),
        ("window", {"variant": "window_based"}),
    ):
        model = ChangePoint(**kwargs).fit(x)
        rows[label] = (model.changepoints_.tolist(), model.evaluations_,
                       model.cost_)
        assert np.all(np.abs(model.changepoints_ - truth) <= 2), label

    print("four blocks of ninety points with means 0, 2, -1, 0.5 and unit "
          "noise; true change points 90, 180, 270:")
    print("  method          change points    evaluations   objective")
    for label, (cps, evaluations, cost) in rows.items():
        shown = "-" if not np.isfinite(cost) else f"{cost:.2f}"
        print(f"  {label:<14s}  {str(cps):<16s} {evaluations:>11d}   "
              f"{shown:>9s}")
    narrow = ChangePoint(variant="window_based", window=20).fit(x)
    assert len(narrow.changepoints_) < 3
    print("  All four recover the truth to within two samples. The exact "
          "optimiser and the greedy one agree here, which is the common case. "
          "The window method is the cheapest by an order of magnitude and the "
          "least precise: its resolution is its window, and at window 20 the "
          f"same test reports {narrow.changepoints_.tolist()} -- it loses the "
          f"third change because the statistic is smeared over two windows.")


def _demo_exact() -> None:
    """PELT's pruning must not change the answer, only the work."""
    rng = np.random.default_rng(1)
    x = _series(rng, [(60, 0.0, 1.0), (70, 1.5, 1.0), (80, 1.5, 2.0),
                      (90, -0.5, 1.0)])
    pruned = ChangePoint(variant="pelt").fit(x)
    exact, naive_evaluations, exact_cost = _naive_dp(
        x, "gaussian", pruned.penalty_, pruned.min_size, pruned.sigma2_)
    assert np.array_equal(pruned.changepoints_, exact)
    assert abs(pruned.cost_ - exact_cost) < 1e-9

    sweep = {}
    for penalty in (0.0, 4.0, 12.0, 24.0, 48.0):
        model = ChangePoint(variant="pelt", penalty=penalty).fit(x)
        fit_cost = model.cost_ - penalty * len(model.changepoints_)
        sweep[penalty] = (len(model.changepoints_), model.cost_, fit_cost)
    counts = [count for count, _, _ in sweep.values()]
    fits = [fit for _, _, fit in sweep.values()]
    assert counts == sorted(counts, reverse=True)
    assert fits == sorted(fits)

    print(f"three hundred points, three changes, BIC penalty "
          f"{pruned.penalty_:.2f}:")
    print(f"  PELT {pruned.changepoints_.tolist()} costs "
          f"{pruned.evaluations_} candidate segments; the unpruned recursion "
          f"finds the same segmentation for {naive_evaluations}, a factor of "
          f"{naive_evaluations / pruned.evaluations_:.1f} more work.")
    print("  The two agree exactly, which is the only thing that makes the "
          "pruning safe: the bound in (3) is a theorem about this cost, not a "
          "heuristic.")
    print("  penalty sweep:")
    print("    beta   changes   penalised   fit only")
    for penalty, (count, cost, fit_cost) in sweep.items():
        print(f"    {penalty:5.1f}  {count:7d}   {cost:9.2f}  {fit_cost:9.2f}")
    print("  As the price of a change rises the count falls and the fit "
          "degrades -- the fit-only column is the sum of (1) over the segments, "
          "which can only get worse. At beta 0 the optimum is a segment every "
          "min_size points.")

    # a series where the greedy recursion cannot afford the second split
    rng = np.random.default_rng(5)
    y = np.zeros(240)
    edges = [0, 60, 90, 150, 200, 240]
    levels = [0.0, 1.2, -0.6, 2.0, 0.3]
    for k, level in enumerate(levels):
        a, b = edges[k], edges[k + 1]
        y[a:b] = level + rng.normal(size=b - a)
    pelt = ChangePoint(variant="pelt").fit(y)
    greedy = ChangePoint(variant="binary_segmentation").fit(y)
    assert greedy.cost_ > pelt.cost_ + 5.0
    print(f"  On a five-level series the greedy recursion stops at "
          f"{greedy.changepoints_.tolist()} and pays {greedy.cost_:.2f}, while "
          f"the exact optimum {pelt.changepoints_.tolist()} pays "
          f"{pelt.cost_:.2f}: {greedy.cost_ - pelt.cost_:.2f} nats for "
          f"{len(greedy.changepoints_)} of the {len(pelt.changepoints_)} "
          f"changes. A split that is not the best one for the whole series can "
          f"still be the best one for its own part, and the recursion has "
          f"already committed.")


def _demo_online() -> None:
    """CUSUM pays a delay, and the delay is the price of a low false-alarm rate."""
    delays, outcomes = {}, {}
    for delta in (0.5, 1.0, 2.0, 3.0):
        found, early = [], 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x = np.r_[rng.normal(size=200), delta + rng.normal(size=200)]
            alarms = ChangePoint(variant="cusum").fit(x).alarms_
            if len(alarms) and alarms[0] < 200:
                early += 1
            after = [int(a) - 200 for a in alarms if 200 <= a < 300]
            if after:
                found.append(after[0])
        delays[delta] = float(np.median(found)) if found else float("nan")
        outcomes[delta] = (len(found), early)

    alarms = 0
    for seed in range(20):
        x = np.random.default_rng(seed).normal(size=400)
        alarms += len(ChangePoint(variant="cusum").fit(x).changepoints_)
    assert delays[3.0] < delays[1.0] < delays[0.5]
    assert outcomes[1.0][0] == 20
    assert alarms <= 12

    rng = np.random.default_rng(0)
    x = _series(rng, [(90, 0.0, 1.0), (90, 2.0, 1.0),
                      (90, -1.0, 1.0), (90, 0.5, 1.0)])
    bocpd = ChangePoint(variant="bocpd").fit(x)
    truth = np.array([90, 180, 270])
    distance = np.abs(bocpd.changepoints_[:, None] - truth[None, :]).min(axis=1)
    assert int((distance <= 3).sum()) == 3
    extra = [int(c) for c, d in zip(bocpd.changepoints_, distance, strict=True)
             if d > 3]

    print("a single step of size delta at t = 200, twenty replications, "
          "threshold 8 sigma and allowance 0.5 sigma:")
    print("  shift   median alarm delay   detected   false alarm first")
    for delta, delay in delays.items():
        found, early = outcomes[delta]
        print(f"  {delta:5.1f}   {delay:19.1f}   {found:6d}/20   {early:16d}")
    print(f"  A one-sigma shift is caught in {delays[1.0]:.0f} samples every "
          f"time; a half-sigma shift is caught {outcomes[0.5][0]} times in "
          f"twenty and takes {delays[0.5]:.0f} samples when it is. The delay is "
          f"not a property of the data, it is the threshold divided by the "
          f"drift, and the threshold is a decision about false alarms: on eight "
          f"thousand stationary points the same test fires {alarms} times, one "
          f"alarm per {8000 // max(alarms, 1)} samples.")
    print(f"  BOCPD on the same three-change series finds "
          f"{bocpd.changepoints_.tolist()} without a threshold at all: the "
          f"hazard is the only prior and the MAP run length is what collapses. "
          f"The extra point at {extra[0]} is the one 3.8-sigma value in the "
          f"series: the posterior cannot tell a restart from an outlier, and "
          f"the run length only collapses six steps later, so the change it "
          f"reports is the outlier itself.")


def _demo_variance() -> None:
    """A variance change is invisible to a mean model and to a mean test."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    x[150:] *= 4.0
    gaussian = ChangePoint(variant="pelt").fit(x)
    mean = ChangePoint(variant="pelt", model="mean").fit(x)
    window = ChangePoint(variant="window_based").fit(x)
    assert np.all(np.abs(gaussian.changepoints_ - 150) <= 5)
    assert not np.any(np.abs(mean.changepoints_ - 150) <= 30)
    assert np.all(np.abs(window.changepoints_ - 150) <= 40)

    left, right = x[110:150], x[150:190]
    se = math.sqrt(left.var(ddof=1) / 40 + right.var(ddof=1) / 40)
    t_stat = (right.mean() - left.mean()) / se

    print("three hundred points with mean zero, unit variance before 150 and "
          "four times the variance after:")
    print(f"  pelt/gaussian {gaussian.changepoints_.tolist()}   "
          f"pelt/mean {mean.changepoints_.tolist()}   "
          f"window {window.changepoints_.tolist()} at threshold "
          f"{window.threshold_:.1f}")
    print(f"  The mean difference across the change is "
          f"{right.mean() - left.mean():+.2f}, or {t_stat:+.2f} standard "
          f"errors: a mean test has nothing to see. The Gaussian cost sees the "
          f"variance ratio of 16 and finds the boundary; the mean model, with "
          f"one variance for the whole series, spends its budget on noise "
          f"inside the noisy half and misses the boundary entirely.")
    print(f"  The window statistic reaches {window.statistic_[150]:.1f} at the "
          f"change against a chi-square threshold of {window.threshold_:.1f}, "
          f"because the pooled variance in (6) includes both the between-window "
          f"mean difference and the within-window spreads, so it reacts to "
          f"either.")


def _demo_calibration() -> None:
    """Two thresholds, two nominal rates, two measured rates."""
    window_alarms = 0
    positions = 0
    cusum_alarms = 0
    for seed in range(20):
        x = np.random.default_rng(seed).normal(size=400)
        window_alarms += len(ChangePoint(variant="window_based").fit(x).changepoints_)
        cusum_alarms += len(ChangePoint(variant="cusum").fit(x).changepoints_)
        positions += 400 - 2 * 40 + 1
    assert window_alarms <= 4

    w_plural = "" if window_alarms == 1 else "s"
    c_plural = "" if cusum_alarms == 1 else "s"
    print("twenty stationary series of four hundred points:")
    print(f"  window_based: {window_alarms} alarm{w_plural} over {positions} "
          f"window positions; the chi-square threshold at alpha 0.05 predicts "
          f"{0.05 * 20:.0f} over twenty series.")
    print(f"  cusum: {cusum_alarms} alarm{c_plural}, against the "
          f"{8000 / 8350:.0f} the known-mean average run length predicts. "
          f"The gap is the estimated "
          f"reference mean: its standard error on a twenty-point warm-up is "
          f"0.22 standard deviations, a fifth of the allowance, and a "
          f"reference that is off by that much is a drift the test never "
          f"budgeted for.")


def _demo_broken() -> None:
    """Malformed calls are refused, and short series are refused."""
    x = np.arange(40.0)
    calls = [
        lambda: ChangePoint(variant="nope"),
        lambda: ChangePoint(model="poisson"),
        lambda: ChangePoint(min_size=0),
        lambda: ChangePoint(penalty=-1.0),
        lambda: ChangePoint(drift=-0.1),
        lambda: ChangePoint(window=1),
        lambda: ChangePoint(alpha=0.0),
        lambda: ChangePoint(hazard=0.0),
        lambda: ChangePoint(hazard=1.0),
        lambda: ChangePoint(kappa0=0.0),
        lambda: ChangePoint(alpha0=-1.0),
        lambda: ChangePoint(beta0=0.0),
        lambda: ChangePoint(min_size=20).fit(x[:30]),
        lambda: ChangePoint().fit(np.r_[x, np.nan]),
        lambda: ChangePoint(variant="cusum", min_size=20).fit(x[:30]),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant or model, "
          f"a negative penalty, a one-point window, out-of-range probabilities, "
          f"non-positive priors, a series too short for two segments, and a "
          f"non-finite value.")


def main() -> None:
    _demo_segment()
    _demo_exact()
    _demo_online()
    _demo_variance()
    _demo_calibration()
    _demo_broken()
    print("change point demo passed")


if __name__ == "__main__":
    main()
