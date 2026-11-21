"""Calibration — turning scores into probabilities that mean what they say.

Chapter:   20-evaluation
Companion: calibration.md
Tier 1 · 6 variants: platt, isotonic, temperature, beta, dirichlet,
reliability_diagram

A classifier that says "0.9" should be right about ninety percent of the time
among all the cases where it said 0.9. Most modern models are not: boosted trees
are underconfident, deep nets are overconfident, and both are worse after
training longer. Calibration learns a monotone map from the scores the model
emits to probabilities that hold up.

    platt                sigmoid(a logit(p) + b)         2 params, smooth
    isotonic             monotone step function           non-parametric
    temperature          softmax(z / T)                  one parameter, multiclass
    beta                 sigmoid(a log p + b log(1-p) + c)
    dirichlet            softmax(W log p + b)            matrix scaling
    reliability_diagram  the diagnostic, not a calibrator

All of them are fitted by minimising log-loss on held-out predictions. None of
them changes the ranking of the scores (`platt`, `isotonic` and the one-parameter
`temperature` are strictly increasing), so accuracy and AUC are unchanged.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["platt", "isotonic", "temperature", "beta", "dirichlet",
     "reliability_diagram"]
)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _fit_logistic(F: np.ndarray, y: np.ndarray, l2: float = 1e-6,
                  iters: int = 100) -> np.ndarray:
    """Newton/IRLS for binary logistic regression; `F` includes the intercept."""
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        p = _sigmoid(F @ w)
        grad = F.T @ (p - y) + l2 * w
        weight = np.maximum(p * (1.0 - p), EPS)
        hess = (F * weight[:, None]).T @ F + l2 * np.eye(F.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def _minimise(fun, grad, x0: np.ndarray, iters: int = 400) -> np.ndarray:
    """Gradient descent with Armijo backtracking — enough for a softmax fit."""
    x = x0.astype(float).copy()
    fx = float(fun(x))
    step = 1.0
    for _ in range(iters):
        g = grad(x)
        if np.max(np.abs(g)) < 1e-9:
            break
        slope = float(g @ g)
        while step > 1e-12:
            trial = x - step * g
            ft = float(fun(trial))
            if ft <= fx - 1e-4 * step * slope:
                x, fx = trial, ft
                step = min(step * 1.5, 1e3)
                break
            step *= 0.5
        else:
            break
    return x


def _pava(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool adjacent violators: the least-squares monotone fit to `labels`."""
    uniq, inv = np.unique(scores, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    sums = np.bincount(inv, weights=labels.astype(float))
    means = sums / counts

    blocks: list[list[float]] = []       # [fitted value, weight, first unique index]
    for j in range(uniq.size):
        blocks.append([float(means[j]), float(counts[j]), j])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, _ = blocks.pop()
            v1, w1, start = blocks.pop()
            total = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / total, total, start])

    fitted = np.empty(uniq.size)
    for pos, (value, _, start) in enumerate(blocks):
        end = uniq.size - 1 if pos == len(blocks) - 1 else int(blocks[pos + 1][2]) - 1
        fitted[start:end + 1] = value
    return uniq, np.clip(fitted, 0.0, 1.0)


def reliability(p: np.ndarray, y: np.ndarray,
                n_bins: int = 15) -> dict[str, np.ndarray | float]:
    """Equal-width reliability bins, ECE and MCE.

    `p` is `(n,)` of P(y=1) or `(n, k)` of class probabilities. Bins are on the
    confidence `max(p)`, which is the standard multiclass definition and reduces
    to `p` for the binary case.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if p.ndim == 1:
        pred = (p >= 0.5).astype(int)
        conf = np.where(p >= 0.5, p, 1.0 - p)
    else:
        pred = p.argmax(axis=1)
        conf = p.max(axis=1)
    correct = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    keep = counts > 0
    mean_conf = conf_sum[keep] / counts[keep]
    mean_acc = acc_sum[keep] / counts[keep]
    n = float(conf.size)
    gaps = np.abs(mean_acc - mean_conf)
    ece = float(np.sum(counts[keep] / n * gaps))
    mce = float(gaps.max()) if gaps.size else 0.0
    return {
        "edges": edges, "counts": counts[keep], "confidence": mean_conf,
        "accuracy": mean_acc, "ece": ece, "mce": mce,
    }


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #
class Calibrator:
    """Map predicted probabilities to calibrated ones.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    n_bins : int
        Bins used by `reliability_diagram`.
    l2 : float
        Ridge strength for the parametric fits; `dirichlet` shrinks toward the
        identity map, the others toward zero.
    random_state : int
        Unused today, kept so every atom has the same constructor shape.

    Notes
    -----
    Fit on a held-out set, never on the training data of the base model: the
    calibrator is learning the base model's *errors*, and those are only visible
    out of sample.
    """

    def __init__(self, variant: str = "platt", *, n_bins: int = 15,
                 l2: float | None = None, random_state: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.n_bins = int(n_bins)
        self.l2 = (1e-6 if variant != "dirichlet" else 1e-3) if l2 is None else float(l2)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y)
        self._one_dimensional = p.ndim == 1
        if self.variant == "dirichlet" and self._one_dimensional:
            raise ValueError("dirichlet needs an (n, k) probability matrix")

        if self._one_dimensional:
            p = np.clip(p, EPS, 1.0 - EPS)
            yb = y.astype(float)
        else:
            p = np.clip(p, EPS, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            yb = y.astype(int)

        if self.variant == "platt":
            self._fit_platt(p, yb)
        elif self.variant == "beta":
            self._fit_beta(p, yb)
        elif self.variant == "isotonic":
            self.x_, self.y_ = _pava(p, yb)
        elif self.variant == "temperature":
            self._fit_temperature(p, yb)
        elif self.variant == "dirichlet":
            self._fit_dirichlet(p, yb)
        else:
            self.diagram_ = reliability(p, y, self.n_bins)
        return self

    def _fit_platt(self, p: np.ndarray, y: np.ndarray) -> None:
        n_pos = float(y.sum())
        n_neg = float(y.size - n_pos)
        target = np.where(y > 0.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        F = np.column_stack([_logit(p), np.ones(p.size)])
        w = _fit_logistic(F, target, self.l2)
        self.coef_, self.intercept_ = float(w[0]), float(w[1])

    def _fit_beta(self, p: np.ndarray, y: np.ndarray) -> None:
        F = np.column_stack([np.log(p), np.log1p(-p), np.ones(p.size)])
        w = _fit_logistic(F, y, self.l2)
        self.coef_ = w[:2].copy()
        self.intercept_ = float(w[2])

    def _fit_temperature(self, p: np.ndarray, y: np.ndarray) -> None:
        z = np.log(p) if p.ndim == 2 else _logit(p)[:, None]
        if p.ndim == 1:
            z = np.column_stack([-z[:, 0], z[:, 0]]) / 2.0
        target = y.astype(int)

        def loss(log_t: np.ndarray) -> float:
            t = float(np.exp(log_t[0]))
            logp = _log_softmax(z / t)
            return -float(np.mean(logp[np.arange(target.size), target]))

        grid = np.linspace(-3.0, 3.0, 601)
        vals = np.array([loss(np.array([g])) for g in grid])
        best = float(grid[int(np.argmin(vals))])
        self.temperature_ = float(np.exp(best))

    def _fit_dirichlet(self, p: np.ndarray, y: np.ndarray) -> None:
        z = np.log(p)
        k = p.shape[1]
        target = np.eye(k)[y]

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return theta[:k * k].reshape(k, k), theta[k * k:]

        def loss(theta: np.ndarray) -> float:
            W, b = unpack(theta)
            logp = _log_softmax(z @ W.T + b)
            nll = -float(np.mean(logp[np.arange(y.size), y]))
            return nll + self.l2 * (float(np.sum((W - np.eye(k)) ** 2))
                                    + float(np.sum(b ** 2)))

        def grad(theta: np.ndarray) -> np.ndarray:
            W, b = unpack(theta)
            s = _softmax(z @ W.T + b)
            g = (s - target) / y.size
            gW = g.T @ z + 2.0 * self.l2 * (W - np.eye(k))
            gb = g.sum(axis=0) + 2.0 * self.l2 * b
            return np.concatenate([gW.ravel(), gb])

        theta0 = np.concatenate([np.eye(k).ravel(), np.zeros(k)])
        theta = _minimise(loss, grad, theta0)
        self.matrix_, self.bias_ = unpack(theta)

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self._one_dimensional:
            p = np.clip(p, EPS, 1.0 - EPS)

        if self.variant == "platt":
            out = _sigmoid(self.coef_ * _logit(p) + self.intercept_)
        elif self.variant == "beta":
            z = self.coef_[0] * np.log(p) + self.coef_[1] * np.log1p(-p)
            out = _sigmoid(z + self.intercept_)
        elif self.variant == "isotonic":
            idx = np.searchsorted(self.x_, p, side="right") - 1
            out = self.y_[np.clip(idx, 0, self.y_.size - 1)]
        elif self.variant == "temperature":
            if p.ndim == 1:
                out = _sigmoid(_logit(p) / self.temperature_)
            else:
                q = np.clip(p, EPS, 1.0)
                logp = np.log(q) - np.log(q.sum(axis=1, keepdims=True))
                out = _softmax(logp / self.temperature_)
        elif self.variant == "dirichlet":
            q = np.clip(p, EPS, 1.0)
            q = q / q.sum(axis=1, keepdims=True)
            out = _softmax(np.log(q) @ self.matrix_.T + self.bias_)
        else:
            out = p
        if self._one_dimensional and self.variant == "dirichlet":
            return out
        return out

    def fit_transform(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(p, y).transform(p)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Binary or multiclass log-loss; lower is better."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if p.ndim == 1:
        q = np.clip(p, EPS, 1.0 - EPS)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
    q = np.clip(p, EPS, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    return -float(np.mean(np.log(q[np.arange(y.size), y])))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- a model that is exactly twice as sharp as it should be ---
    n = 2000
    x = rng.normal(size=n)
    p_true = _sigmoid(2.0 * x)
    y = (rng.random(n) < p_true).astype(int)
    p_raw = _sigmoid(4.0 * x)               # overconfident by a factor of two
    cut = n // 2
    tr, te = slice(0, cut), slice(cut, n)

    ece_raw = reliability(p_raw[te], y[te])["ece"]
    assert ece_raw > 0.05, ece_raw
    loss_raw = log_loss(p_raw[te], y[te])

    # --- Platt: two parameters, slope should recover the 2x miscalibration ---
    platt = Calibrator("platt").fit(p_raw[tr], y[tr])
    p_platt = platt.transform(p_raw[te])
    assert abs(platt.coef_ - 0.5) < 0.08, platt.coef_
    assert log_loss(p_platt, y[te]) < loss_raw
    assert reliability(p_platt, y[te])["ece"] < ece_raw
    err_platt = float(np.mean(np.abs(p_platt - p_true[te])))
    assert err_platt < 0.02, err_platt
    assert np.array_equal(np.argsort(p_platt), np.argsort(p_raw[te]))
    print(f"calibration: platt slope {platt.coef_:.3f} (truth 0.5), log-loss "
          f"{loss_raw:.3f} -> {log_loss(p_platt, y[te]):.3f}, ece "
          f"{ece_raw:.3f} -> {reliability(p_platt, y[te])['ece']:.3f}, "
          f"bias to truth {err_platt:.3f}")

    # --- isotonic: monotone by construction, no parametric form ---
    iso = Calibrator("isotonic").fit(p_raw[tr], y[tr])
    grid = np.linspace(0.001, 0.999, 500)
    iso_grid = iso.transform(grid)
    assert np.all(np.diff(iso_grid) >= -1e-12), np.min(np.diff(iso_grid))
    p_iso = iso.transform(p_raw[te])
    assert reliability(p_iso, y[te])["ece"] < ece_raw
    assert p_iso.min() >= 0.0 and p_iso.max() <= 1.0
    err_iso = float(np.mean(np.abs(p_iso - p_true[te])))
    assert err_iso < 0.04, err_iso
    print(f"calibration: isotonic {iso.x_.size} breakpoints, ece "
          f"{reliability(p_iso, y[te])['ece']:.3f} (platt "
          f"{reliability(p_platt, y[te])['ece']:.3f}), bias to truth "
          f"{err_iso:.3f}")

    # --- beta: log p and log(1-p); recovers the same 2x correction ---
    beta = Calibrator("beta").fit(p_raw[tr], y[tr])
    p_beta = beta.transform(p_raw[te])
    err_beta = float(np.mean(np.abs(p_beta - p_true[te])))
    assert err_beta < 0.02, err_beta
    assert log_loss(p_beta, y[te]) < loss_raw
    print(f"calibration: beta coefs [{beta.coef_[0]:.3f}, {beta.coef_[1]:.3f}], "
          f"intercept {beta.intercept_:+.3f}, bias to truth {err_beta:.3f}")

    # --- multiclass: temperature and dirichlet on 3 classes ---
    xc = rng.normal(size=(1500, 2))
    A = np.array([[2.0, 0.0, -2.0], [0.0, 2.0, -2.0]])
    z_true = xc @ A
    p_true_c = _softmax(z_true)
    u = rng.random((1500, 1))
    y_c = (u > np.cumsum(p_true_c, axis=1)).sum(axis=1)
    y_c = np.clip(y_c, 0, 2)
    p_raw_c = _softmax(2.0 * z_true)        # again twice as sharp
    ctr, cte = slice(0, 750), slice(750, 1500)

    temp = Calibrator("temperature").fit(p_raw_c[ctr], y_c[ctr])
    p_temp = temp.transform(p_raw_c[cte])
    assert abs(temp.temperature_ - 2.0) < 0.25, temp.temperature_
    ece_c = reliability(p_raw_c[cte], y_c[cte])["ece"]
    assert reliability(p_temp, y_c[cte])["ece"] < ece_c
    print(f"calibration: temperature T {temp.temperature_:.3f} (truth 2.0), "
          f"ece {ece_c:.3f} -> {reliability(p_temp, y_c[cte])['ece']:.3f}")

    diric = Calibrator("dirichlet").fit(p_raw_c[ctr], y_c[ctr])
    p_dir = diric.transform(p_raw_c[cte])
    assert diric.matrix_.shape == (3, 3)
    assert reliability(p_dir, y_c[cte])["ece"] < ece_c
    before = np.mean(p_raw_c[cte].argmax(axis=1) == y_c[cte])
    after = np.mean(p_dir.argmax(axis=1) == y_c[cte])
    assert abs(after - before) < 0.02, (after, before)
    print(f"calibration: dirichlet 3x3 scaling, ece "
          f"{ece_c:.3f} -> {reliability(p_dir, y_c[cte])['ece']:.3f}, accuracy "
          f"{before:.3f} -> {after:.3f}")

    # --- reliability diagram: bins cover the test set and ECE is reproducible ---
    diag = Calibrator("reliability_diagram", n_bins=10).fit(p_raw[te], y[te])
    bins = diag.diagram_
    assert float(bins["counts"].sum()) == float(te.stop - te.start)
    manual = float(np.sum(bins["counts"] / (te.stop - te.start)
                          * np.abs(bins["accuracy"] - bins["confidence"])))
    assert abs(manual - float(bins["ece"])) < 1e-12
    assert diag.transform(p_raw[te]).shape == p_raw[te].shape
    print(f"calibration: reliability diagram {bins['counts'].size} occupied "
          f"bins, ece {bins['ece']:.3f}, mce {bins['mce']:.3f}")

    # --- a genuinely calibrated model has almost no calibration error ---
    p_ok = _sigmoid(2.0 * rng.normal(size=4000))
    y_ok = (rng.random(p_ok.size) < p_ok).astype(int)
    ece_ok = reliability(p_ok, y_ok, 15)["ece"]
    assert ece_ok < 0.03, ece_ok
    print(f"calibration: honest probabilities have ece {ece_ok:.3f} "
          f"(sampling noise only)")

    print("calibration demo passed")


if __name__ == "__main__":
    main()
