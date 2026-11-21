"""Logistic regression — 8 variants of the linear classifier.

Chapter:   02-linear-models
Companion: logistic_regression.md
Tier 1 · 8 variants: binomial, multinomial, ordinal, irls, lbfgs, elastic_net,
class_weight, separable_fix

One model, `P(y = c | x) = softmax_c(Wᵀx + b)`, and eight things people mean when
they say "logistic regression": the binary case (a scalar `W`), the multinomial
case, the *ordinal* case where the classes are ordered, three different solvers
(gradient, Newton/IRLS, L-BFGS), a regularised version, a reweighted version for
imbalance, and the penalty you need when the data is separable.

Everything is fitted by minimising the mean negative log-likelihood, so the models
differ only in the likelihood, and the solvers differ only in how they move.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["binomial", "multinomial", "ordinal", "irls", "lbfgs", "elastic_net",
     "class_weight", "separable_fix"]
)


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) without overflow: the two branches never evaluate exp of a large
    positive number."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    """log(1 + e^z), accurate for both signs."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _design(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if fit_intercept:
        return np.column_stack([X, np.ones(len(X))])
    return X


# --------------------------------------------------------------------------- #
# Generic solvers. Each takes a closure fun(theta) -> (loss, grad).
# --------------------------------------------------------------------------- #
def gd(fun, x0: np.ndarray, iters: int = 2000, lr: float = 1.0):
    """Gradient descent with backtracking. Monotone by construction, which is what
    makes it the reference the other solvers are checked against."""
    x = x0.copy()
    f, g = fun(x)
    hist = [f]
    for _ in range(iters):
        t = lr
        while True:
            xn = x + t * (-g)
            fn, gn = fun(xn)
            if fn <= f + 1e-4 * t * float(g @ (-g)) or t < 1e-14:
                break
            t *= 0.5
        if fn > f:
            break
        x, f, g = xn, fn, gn
        lr = min(2.0 * t, 1.0)
        hist.append(f)
    return x, hist


def lbfgs(fun, x0: np.ndarray, iters: int = 500, memory: int = 10, tol: float = 1e-10):
    """Limited-memory BFGS: keep the last `memory` (s, y) pairs, build the inverse
    Hessian action by the two-loop recursion. The default solver for smooth convex
    problems when Newton is too expensive."""
    x = x0.copy()
    f, g = fun(x)
    S: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    hist = [f]
    for _ in range(iters):
        q = g.copy()
        alphas = []
        for s, yv in zip(reversed(S), reversed(Y), strict=True):
            rho = 1.0 / float(yv @ s)
            a = rho * float(s @ q)
            alphas.append((a, rho))
            q -= a * yv
        if S:
            gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
        else:
            gamma = 1.0 / (np.linalg.norm(g) + 1e-12)
        r = gamma * q
        for (a, rho), s, yv in zip(reversed(alphas), S, Y, strict=True):
            b = rho * float(yv @ r)
            r += s * (a - b)
        d = -r
        if float(g @ d) >= 0.0:            # not a descent direction: reset
            d = -g
            S.clear()
            Y.clear()
        t = 1.0
        while True:
            xn = x + t * d
            fn, gn = fun(xn)
            if fn <= f + 1e-4 * t * float(g @ d) or t < 1e-14:
                break
            t *= 0.5
        s = xn - x
        yv = gn - g
        if float(s @ yv) > 1e-12:
            S.append(s)
            Y.append(yv)
            if len(S) > memory:
                S.pop(0)
                Y.pop(0)
        converged = abs(f - fn) < tol * (1.0 + abs(f))
        x, f, g = xn, fn, gn
        hist.append(f)
        if converged:
            break
    return x, hist


# --------------------------------------------------------------------------- #
# Binomial likelihood (the target of gd, irls, lbfgs, elastic_net, class_weight)
# --------------------------------------------------------------------------- #
def _binomial_fun(Xa, y, sw, n, l2, l1, theta):
    z = Xa @ theta
    loss = float(np.sum(sw * (softplus(z) - y * z))) / n
    grad = Xa.T @ (sw * (sigmoid(z) - y)) / n
    loss += 0.5 * l2 * float(theta[:-1] @ theta[:-1])
    grad[:-1] += l2 * theta[:-1]
    if l1:
        loss += l1 * float(np.abs(theta[:-1]).sum())
        grad[:-1] += l1 * np.sign(theta[:-1])
    return loss, grad


def _irls_binomial(Xa, y, sw, n, l2, iters=50, tol=1e-10):
    """Fisher scoring: Newton's method on the log-likelihood, with the observed
    information `Xᵀ diag(p(1−p)) X` as the Hessian. Quadratic convergence once the
    iterate is close, and never more than a handful of steps in practice."""
    d1 = Xa.shape[1]
    theta = np.zeros(d1)
    hist = []
    penalty = np.full(d1, l2)
    penalty[-1] = 0.0
    for _ in range(iters):
        z = Xa @ theta
        p = sigmoid(z)
        loss, grad = _binomial_fun(Xa, y, sw, n, l2, 0.0, theta)
        hist.append(loss)
        s = sw * p * (1.0 - p)
        H = (Xa * s[:, None]).T @ Xa / n + np.diag(penalty)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(d1), grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(20):               # damped Newton: never increase the loss
            cand = theta - t * step
            fc = _binomial_fun(Xa, y, sw, n, l2, 0.0, cand)[0]
            if fc <= loss:
                break
            t *= 0.5
        theta = theta - t * step
        if np.linalg.norm(t * step) < tol:
            break
    return theta, hist


def _ista_binomial(Xa, y, sw, n, l2, l1, iters=3000, tol=1e-12):
    """Proximal gradient for the elastic net. The prox of `l1|β|` is a soft
    threshold, so each step is `β ← S_{t·l1}(β − t∇)` with `t = 1/L`."""
    d1 = Xa.shape[1]
    theta = np.zeros(d1)
    L = 0.25 * float(sw.max()) * float(np.linalg.norm(Xa, 2) ** 2) / n + l2
    t = 1.0 / max(L, 1e-12)
    hist = []
    for _ in range(iters):
        _, grad = _binomial_fun(Xa, y, sw, n, l2, 0.0, theta)
        z = theta - t * grad
        new = np.sign(z) * np.maximum(np.abs(z) - t * l1, 0.0)
        new[-1] = z[-1]                   # the intercept is never penalised
        if np.linalg.norm(new - theta) < tol:
            theta = new
            break
        theta = new
        if _ % 25 == 0:
            hist.append(_binomial_fun(Xa, y, sw, n, l2, l1, theta)[0])
    return theta, hist


# --------------------------------------------------------------------------- #
# Multinomial
# --------------------------------------------------------------------------- #
def _multinomial_fun(Xa, Y, sw, n, l2):
    d1, K = Xa.shape[1], Y.shape[1]

    def fun(flat: np.ndarray):
        W = flat.reshape(d1, K)
        z = Xa @ W
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -float(np.sum(sw * (Y * logp).sum(axis=1))) / n
        loss += 0.5 * l2 * float((W[:-1] ** 2).sum())
        G = Xa.T @ (sw[:, None] * (np.exp(logp) - Y)) / n
        G[:-1] += l2 * W[:-1]
        return loss, G.ravel()

    return fun


# --------------------------------------------------------------------------- #
# Ordinal (cumulative logit / proportional odds)
# --------------------------------------------------------------------------- #
def _ordinal_fun(Xa, y, n_classes, sw, n, l2):
    """θ₁ free, θ_k = θ_{k−1} + softplus(δ_k) so the thresholds are ordered by
    construction. P(y ≤ k) = σ(θ_k − xᵀβ), P(y = k) = p_k − p_{k−1}."""
    d1 = Xa.shape[1]
    K = n_classes

    def unpack(flat):
        beta = flat[:d1]
        c = flat[d1]
        delta = flat[d1 + 1:]
        sp = softplus(delta)
        theta = np.concatenate([[c], c + np.cumsum(sp)])
        return beta, c, delta, sp, theta

    def fun(flat: np.ndarray):
        beta, c, delta, sp, theta = unpack(flat)
        z = Xa @ beta
        p = sigmoid(theta[None, :] - z[:, None])       # (n, K-1)
        p_ext = np.concatenate([np.zeros((n, 1)), p, np.ones((n, 1))], axis=1)
        idx = np.arange(n)
        prob = p_ext[idx, y + 1] - p_ext[idx, y]
        prob = np.maximum(prob, 1e-12)
        loss = -float(np.sum(sw * np.log(prob))) / n + 0.5 * l2 * float(beta @ beta)

        # dNLL/dz and dNLL/dθ_k, per observation
        pc = p_ext[idx, y + 1]
        pm = p_ext[idx, y]
        # dprob/dθ_k: +p_k(1−p_k) when k = c, −p_k(1−p_k) when k = c−1, where
        # pc = q[c+1] = σ(θ_c − z) and pm = q[c] = σ(θ_{c−1} − z).
        density = np.zeros((n, K - 1))
        for k in range(K - 1):
            density[:, k] = (y == k) * pc * (1 - pc)
            density[:, k] -= (y == k + 1) * pm * (1 - pm)
        g_z = density.sum(axis=1) / prob
        g_theta = -density / prob[:, None]      # NLL = -log prob

        grad_beta = Xa.T @ (sw * g_z) / n + l2 * beta
        g_t = (sw[:, None] * g_theta).sum(axis=0) / n
        grad_c = float(g_t.sum())
        # θ_k depends on δ_j through softplus for j ≤ k
        tail = np.cumsum(g_t[::-1])[::-1]              # tail[j] = Σ_{k ≥ j} gθ_k
        grad_delta = tail[1:] * sigmoid(delta)         # d softplus/dδ = σ(δ)
        grad = np.concatenate([grad_beta, [grad_c], grad_delta])
        return loss, grad

    return fun, unpack


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    variant: str = "lbfgs",
    *,
    l2: float = 0.0,
    l1: float = 0.0,
    iters: int = 1000,
    sample_weight: np.ndarray | None = None,
    fit_intercept: bool = True,
) -> dict:
    """Fit one of the eight variants. `y` holds labels; for the binomial variants
    there must be exactly two, ordered so that `classes_[1]` is the positive class.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    Xa = _design(X, fit_intercept)
    n, d1 = Xa.shape

    def weights() -> np.ndarray:
        if sample_weight is not None:
            return np.asarray(sample_weight, dtype=float)
        return np.ones(n)

    out: dict = {"variant": variant, "classes": classes, "intercept": None,
                 "thresholds": None, "coef": None, "hist": [],
                 "fit_intercept": fit_intercept}

    if variant in ("binomial", "irls", "lbfgs", "elastic_net", "separable_fix",
                   "class_weight"):
        if classes.size != 2:
            raise ValueError("binomial variants need exactly two classes")
        yb = (y == classes[1]).astype(float)
        sw = weights()
        if variant == "class_weight":
            counts = np.array([np.sum(sw[yb == 0]), np.sum(sw[yb == 1])])
            per_class = n / (2.0 * counts)          # balanced: equal total weight
            sw = sw * np.where(yb == 1, per_class[1], per_class[0])
        sw = sw / sw.mean()

        fun = lambda th: _binomial_fun(Xa, yb, sw, n, l2, l1, th)  # noqa: E731
        if variant == "irls":
            theta, hist = _irls_binomial(Xa, yb, sw, n, l2, iters=min(iters, 100))
        elif variant == "separable_fix":
            # Same solver, but a strictly positive penalty: the MLE does not exist
            # on separable data, and a tiny Gaussian prior is what makes it finite.
            theta, hist = _irls_binomial(Xa, yb, sw, n, max(l2, 1e-3),
                                         iters=min(iters, 100))
        elif variant == "elastic_net":
            theta, hist = _ista_binomial(Xa, yb, sw, n, l2, l1, iters=iters)
        elif variant == "lbfgs":
            theta, hist = lbfgs(fun, np.zeros(d1), iters=iters)
        else:                                    # binomial, class_weight
            theta, hist = gd(fun, np.zeros(d1), iters=iters)

        out.update(coef=theta[:-1], hist=hist,
                   intercept=float(theta[-1]) if fit_intercept else 0.0)

    elif variant == "multinomial":
        Y = (y[:, None] == classes[None, :]).astype(float)
        sw = weights()
        sw = sw / sw.mean()
        fun = _multinomial_fun(Xa, Y, sw, n, l2)
        flat, hist = lbfgs(fun, np.zeros(d1 * classes.size), iters=iters)
        W = flat.reshape(d1, classes.size)
        out.update(coef=W[:-1], hist=hist,
                   intercept=W[-1] if fit_intercept else np.zeros(classes.size))

    elif variant == "ordinal":
        sw = weights()
        sw = sw / sw.mean()
        fun, unpack = _ordinal_fun(Xa, y.astype(int), classes.size, sw, n, l2)
        flat, hist = lbfgs(fun, np.zeros(d1 + 1 + classes.size - 2), iters=iters)
        beta, _, _, _, theta = unpack(flat)
        out.update(coef=beta, hist=hist, thresholds=theta,
                   intercept=float(theta[0]) if fit_intercept else 0.0)

    return out


def predict_proba(fit: dict, X: np.ndarray) -> np.ndarray:
    """Class probabilities for whichever variant was fitted."""
    X = np.asarray(X, dtype=float)
    theta = fit["coef"]
    if fit["variant"] == "ordinal":
        z = _design(X, fit["fit_intercept"]) @ theta
        thr = fit["thresholds"]
        p = sigmoid(thr[None, :] - z[:, None])
        p = np.concatenate([np.zeros((len(X), 1)), p, np.ones((len(X), 1))], axis=1)
        return np.diff(p, axis=1)
    b = np.asarray(fit["intercept"], dtype=float)
    if theta.ndim == 1:
        p1 = sigmoid(X @ theta + float(b))
        return np.column_stack([1.0 - p1, p1])
    z = X @ theta + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 600, 4
    X = rng.normal(size=(n, d))
    beta_true = np.array([1.5, -2.0, 0.8, 0.0])
    b_true = 0.4
    p = sigmoid(X @ beta_true + b_true)
    y = (rng.random(n) < p).astype(int)

    # 1. Three solvers, one optimum. GD is the reference (monotone), IRLS gets
    #    there in a handful of Newton steps, L-BFGS in a few dozen.
    fits = {v: logistic_regression(X, y, v) for v in ("binomial", "irls", "lbfgs")}
    ref = fits["lbfgs"]["coef"]
    for v, f in fits.items():
        assert np.allclose(f["coef"], ref, atol=1e-3), (v, np.abs(f["coef"] - ref).max())
    assert np.linalg.norm(ref - beta_true) < 0.5, ref
    assert len(fits["irls"]["hist"]) <= 10, len(fits["irls"]["hist"])
    assert len(fits["lbfgs"]["hist"]) <= 60, len(fits["lbfgs"]["hist"])
    gd_hist = fits["binomial"]["hist"]
    assert all(gd_hist[i] >= gd_hist[i + 1] - 1e-12 for i in range(len(gd_hist) - 1))
    print(f"binomial/irls/lbfgs agree to {np.abs(fits['binomial']['coef'] - ref).max():.1e}; "
          f"irls {len(fits['irls']['hist'])} iters, lbfgs {len(fits['lbfgs']['hist'])}, "
          f"gd {len(gd_hist)}")

    # 2. Gradient check against central differences, for both likelihoods.
    Xa = _design(X[:20], True)
    theta = rng.normal(size=Xa.shape[1])
    yb = y[:20].astype(float)
    sw = np.ones(20)
    f0, g0 = _binomial_fun(Xa, yb, sw, 20, 0.1, 0.0, theta)
    num = np.array([
        (_binomial_fun(Xa, yb, sw, 20, 0.1, 0.0, theta + 1e-6 * e)[0]
         - _binomial_fun(Xa, yb, sw, 20, 0.1, 0.0, theta - 1e-6 * e)[0]) / 2e-6
        for e in np.eye(theta.size)
    ])
    assert np.allclose(g0, num, atol=1e-6), np.abs(g0 - num).max()

    Y3 = np.eye(3)[np.arange(20) % 3]
    mfun = _multinomial_fun(Xa, Y3, sw, 20, 0.05)
    th_m = rng.normal(size=Xa.shape[1] * 3)
    _, gm = mfun(th_m)
    num_m = np.array([
        (mfun(th_m + 1e-6 * e)[0] - mfun(th_m - 1e-6 * e)[0]) / 2e-6
        for e in np.eye(th_m.size)
    ])
    assert np.allclose(gm, num_m, atol=1e-6), np.abs(gm - num_m).max()

    ofun, _ = _ordinal_fun(Xa, (np.arange(20) % 3), 3, sw, 20, 0.05)
    th_o = rng.normal(size=Xa.shape[1] + 2)     # β, θ₀, δ₁
    _, go = ofun(th_o)
    num_o = np.array([
        (ofun(th_o + 1e-6 * e)[0] - ofun(th_o - 1e-6 * e)[0]) / 2e-6
        for e in np.eye(th_o.size)
    ])
    assert np.allclose(go, num_o, atol=1e-6), np.abs(go - num_o).max()
    print(f"gradient checks: binomial {np.abs(g0 - num).max():.1e}, "
          f"multinomial {np.abs(gm - num_m).max():.1e}, "
          f"ordinal {np.abs(go - num_o).max():.1e}")

    # 3. Multinomial recovers a known softmax model and its probabilities sum to 1.
    W_true = np.array([[1.2, -1.0, 0.3], [-0.8, 0.6, 0.1], [0.5, 0.2, -0.9],
                       [0.0, 0.0, 0.0]])
    z = X @ W_true
    z -= z.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y3 = np.array([rng.choice(3, p=pi) for pi in P])
    mfit = logistic_regression(X, y3, "multinomial", l2=1e-3)
    proba = predict_proba(mfit, X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    acc = float(np.mean(proba.argmax(axis=1) == y3))
    assert acc > 0.6, acc
    print(f"multinomial: train accuracy {acc:.3f}, rows of P sum to 1")

    # 4. Ordinal: three ordered classes, thresholds recovered in order.
    z_lat = X @ np.array([1.5, -1.0, 0.5, 0.0]) + rng.logistic(size=n) * 0.7
    y_ord = np.digitize(z_lat, [-0.6, 0.9])
    ofit = logistic_regression(X, y_ord, "ordinal")
    thr = ofit["thresholds"]
    assert np.all(np.diff(thr) > 0), thr
    oacc = float(np.mean(predict_proba(ofit, X).argmax(axis=1) == y_ord))
    assert oacc > 0.5, oacc
    print(f"ordinal: thresholds {np.round(thr, 2)} (ordered), accuracy {oacc:.3f}")

    # 5. Elastic net produces exact zeros, and l1 = 0 reproduces the smooth fit.
    Xs = rng.normal(size=(300, 8))
    b_sparse = np.array([1.5, -1.2, 0, 0, 0, 0, 0, 0])
    ys = (rng.random(300) < sigmoid(Xs @ b_sparse)).astype(int)
    en = logistic_regression(Xs, ys, "elastic_net", l1=0.05, l2=0.01, iters=4000)
    zeros = int((np.abs(en["coef"]) < 1e-12).sum())
    smooth = logistic_regression(Xs, ys, "lbfgs", l2=0.01)
    en0 = logistic_regression(Xs, ys, "elastic_net", l1=0.0, l2=0.01, iters=4000)
    assert zeros >= 5, en["coef"]
    assert np.allclose(en0["coef"], smooth["coef"], atol=2e-2), \
        np.abs(en0["coef"] - smooth["coef"]).max()
    print(f"elastic_net: {zeros}/8 coefficients exactly 0; "
          f"l1=0 matches the penalised MLE to {np.abs(en0['coef'] - smooth['coef']).max():.1e}")

    # 6. Class weights: on 95/5 data, the unweighted fit predicts the majority
    #    class everywhere and the weighted fit recovers the minority class.
    Xw = rng.normal(size=(2000, 3))
    yw = (rng.random(2000) < sigmoid(Xw @ np.array([2.0, -1.5, 0.0]) - 2.5)).astype(int)
    plain = logistic_regression(Xw, yw, "binomial", iters=3000)
    bal = logistic_regression(Xw, yw, "class_weight", iters=3000)

    def balanced_acc(fit):
        pred = predict_proba(fit, Xw).argmax(axis=1)
        return float(np.mean([np.mean(pred[yw == c] == c) for c in (0, 1)]))

    b_plain, b_bal = balanced_acc(plain), balanced_acc(bal)
    assert b_bal > b_plain and b_bal > 0.65, (b_plain, b_bal)
    print(f"class_weight ({yw.mean():.2%} positive): balanced accuracy "
          f"{b_plain:.3f} -> {b_bal:.3f}")

    # 7. Separable data: the MLE does not exist. Without a penalty the norm keeps
    #    growing with the iteration budget; with one it settles.
    Xsep = rng.normal(size=(80, 2))
    ysep = (Xsep[:, 0] > 0).astype(int)
    raw = logistic_regression(Xsep, ysep, "irls", l2=0.0, iters=100)
    fixed = logistic_regression(Xsep, ysep, "separable_fix", iters=100)
    norm_raw = float(np.linalg.norm(raw["coef"]))
    norm_fixed = float(np.linalg.norm(fixed["coef"]))
    # The unpenalised fit drives the loss to zero by driving the coefficients to
    # infinity; the penalty buys a finite answer for a slightly larger loss.
    assert raw["hist"][-1] < 1e-4, raw["hist"][-1]
    assert norm_raw > 100 * norm_fixed, (norm_raw, norm_fixed)
    print(f"separable data: ||β|| = {norm_raw:.0f} (loss {raw['hist'][-1]:.1e}) vs "
          f"{norm_fixed:.1f} (loss {fixed['hist'][-1]:.3f}) with a 1e-3 penalty")
    print("logistic_regression demo passed")


if __name__ == "__main__":
    main()
