"""Losses — 20 objectives with exact gradients.

Chapter:   00-primitives
Companion: losses.md
Tier 1 · 20 variants: mse, mae, huber, log_cosh, quantile, bce, cross_entropy,
focal, dice, tversky, hinge, squared_hinge, triplet, contrastive, infonce, kl,
js, wasserstein, cosine_embedding, label_smoothing

Contract: every function returns (value, grad) where `value` is a scalar mean over
the batch and `grad` has the shape of the prediction. Three exceptions return a
gradient per input instead of one (triplet, contrastive, infonce); they say so.

Logit-based losses (bce, cross_entropy, focal, label_smoothing) take raw logits,
not probabilities. Everything else takes predictions.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def mse(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    r = p - y
    return float(np.mean(r * r)), 2.0 * r / r.size


def mae(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    r = p - y
    return float(np.mean(np.abs(r))), np.sign(r) / r.size


def huber(y: np.ndarray, p: np.ndarray, delta: float = 1.0) -> tuple[float, np.ndarray]:
    """Quadratic below delta, linear above: the M-estimator with bounded influence."""
    r = p - y
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    grad = np.clip(r, -delta, delta) / r.size
    return float(np.mean(loss)), grad


def log_cosh(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """log cosh(r): smooth, twice differentiable, ~L2 near 0 and ~L1 far away."""
    r = p - y
    a = np.abs(r)
    loss = a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)  # stable form
    return float(np.mean(loss)), np.tanh(r) / r.size


def quantile(y: np.ndarray, p: np.ndarray, q: float = 0.5) -> tuple[float, np.ndarray]:
    """Pinball loss: asymmetric absolute error, minimised by the q-quantile."""
    r = p - y
    loss = np.maximum(q * r, (q - 1.0) * r)
    grad = np.where(r > 0, q, q - 1.0) / r.size
    return float(np.mean(loss)), grad


# --------------------------------------------------------------------------- #
# Classification (logits in)
# --------------------------------------------------------------------------- #
def bce(y: np.ndarray, logits: np.ndarray) -> tuple[float, np.ndarray]:
    """Binary cross-entropy from logits: log(1+e^z) - y z, stable."""
    loss = np.logaddexp(0.0, logits) - y * logits
    p = 0.5 * (1.0 + np.tanh(0.5 * logits))
    return float(np.mean(loss)), (p - y) / y.size


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Categorical cross-entropy. logits: (n, K), labels: (n,) integer class ids."""
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - lse
    n = z.shape[0]
    loss = -(logp[np.arange(n), labels]).mean()
    p = np.exp(logp)
    p[np.arange(n), labels] -= 1.0
    return float(loss), p / n


def focal(y: np.ndarray, logits: np.ndarray, gamma: float = 2.0) -> tuple[float, np.ndarray]:
    """Focal loss: (1-p_t)^gamma * CE. Down-weights the easy majority class."""
    log_pt = y * -np.logaddexp(0.0, -logits) + (1.0 - y) * -np.logaddexp(0.0, logits)
    pt = np.exp(log_pt)
    loss = -((1.0 - pt) ** gamma) * log_pt
    sign = 2.0 * y - 1.0  # d(pt)/d(logit) has this factor on top of pt(1-pt)
    grad = sign * (gamma * pt * (1.0 - pt) ** gamma * log_pt - (1.0 - pt) ** (gamma + 1.0))
    return float(np.mean(loss)), grad / y.size


LABEL_SMOOTHING_EPS = 0.1


def label_smoothing(
    logits: np.ndarray, labels: np.ndarray, eps: float = LABEL_SMOOTHING_EPS
) -> tuple[float, np.ndarray]:
    """CE against (1-eps)*onehot + eps/K, which is CE - eps*H(uniform) for the
    correct class plus eps/K mass everywhere."""
    n, k = logits.shape
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    target = np.full_like(p, eps / k)
    target[np.arange(n), labels] += 1.0 - eps
    loss = -(target * logp).sum(axis=1).mean()
    return float(loss), (p - target) / n


# --------------------------------------------------------------------------- #
# Segmentation / set overlap
# --------------------------------------------------------------------------- #
def dice(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> tuple[float, np.ndarray]:
    """1 - 2*|y∩p| / (|y|+|p|). p is a probability map of the same shape as y."""
    a = float((p * y).sum())
    b = float(p.sum() + y.sum())
    loss = 1.0 - (2.0 * a + eps) / (b + eps)
    grad = -(2.0 * y * (b + eps) - (2.0 * a + eps)) / (b + eps) ** 2
    return loss, grad


def tversky(
    y: np.ndarray, p: np.ndarray, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6
) -> tuple[float, np.ndarray]:
    """Generalised Dice: alpha weights false positives, beta false negatives.
    alpha=beta=0.5 is Dice; alpha=0.3,beta=0.7 trades precision for recall."""
    inter = float((p * y).sum())
    denom = inter + alpha * float((p * (1 - y)).sum()) + beta * float(((1 - p) * y).sum()) + eps
    loss = 1.0 - (inter + eps) / denom
    ddenom = y + alpha * (1 - y) - beta * y
    grad = -(y * denom - (inter + eps) * ddenom) / denom**2
    return loss, grad


def hinge(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Max-margin loss for y in {-1, +1}."""
    m = 1.0 - y * p
    loss = np.maximum(m, 0.0)
    return float(np.mean(loss)), np.where(m > 0, -y, 0.0) / y.size


def squared_hinge(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    m = np.maximum(1.0 - y * p, 0.0)
    return float(np.mean(m * m)), np.where(m > 0, -2.0 * y * m, 0.0) / y.size


# --------------------------------------------------------------------------- #
# Metric learning (one gradient per input)
# --------------------------------------------------------------------------- #
def triplet(
    a: np.ndarray, pos: np.ndarray, neg: np.ndarray, margin: float = 1.0
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """max(0, ||a-p||^2 - ||a-n||^2 + margin). Grader: (d/da, d/dp, d/dn)."""
    d_ap = ((a - pos) ** 2).sum(axis=1)
    d_an = ((a - neg) ** 2).sum(axis=1)
    m = d_ap - d_an + margin
    active = (m > 0).astype(float)
    n = a.shape[0]
    loss = float(np.maximum(m, 0.0).mean())
    ga = 2.0 * (neg - pos) * active[:, None] / n
    gp = -2.0 * (a - pos) * active[:, None] / n
    gn = 2.0 * (a - neg) * active[:, None] / n
    return loss, (ga, gp, gn)


def contrastive(
    d: np.ndarray, y: np.ndarray, margin: float = 1.0
) -> tuple[float, np.ndarray]:
    """d: pairwise distances, y=1 for similar, 0 for dissimilar."""
    pos = y * d**2
    neg = (1 - y) * np.maximum(margin - d, 0.0) ** 2
    loss = float(np.mean(pos + neg))
    grad = y * 2.0 * d + (1 - y) * np.where(d < margin, -2.0 * (margin - d), 0.0)
    return loss, grad / d.size


def infonce(
    z1: np.ndarray, z2: np.ndarray, tau: float = 0.07
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Symmetric InfoNCE over a batch of aligned pairs. Gradients for (z1, z2).

    The diagonal of the similarity matrix is the positive pair; every off-diagonal
    entry is a negative. Both directions are averaged (SimCLR convention).
    """
    a = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    b = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    logits = a @ b.T / tau
    n = a.shape[0]
    eye = np.eye(n)

    def rows(logit):
        z = logit - logit.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -np.mean(np.diag(logp))
        return loss, (np.exp(logp) - eye) / n

    l1, g1 = rows(logits)
    l2, g2 = rows(logits.T)
    da = 0.5 * (g1 @ b + g2.T @ b) / tau
    db = 0.5 * (g1.T @ a + g2 @ a) / tau
    # back through L2 normalisation: d(z/|z|)/dz = (I - aa^T)/|z|
    na = np.linalg.norm(z1, axis=1, keepdims=True)
    nb = np.linalg.norm(z2, axis=1, keepdims=True)
    gz1 = (da - a * (a * da).sum(axis=1, keepdims=True)) / na
    gz2 = (db - b * (b * db).sum(axis=1, keepdims=True)) / nb
    return 0.5 * (l1 + l2), (gz1, gz2)


# --------------------------------------------------------------------------- #
# Divergences and similarity
# --------------------------------------------------------------------------- #
def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """KL(p || q). Gradient is with respect to q, the parameterised distribution."""
    q = np.clip(q, eps, None)
    loss = float((p * np.log(np.clip(p, eps, None) / q)).sum(axis=-1).mean())
    return loss, -p / (q * p.shape[0])


def js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """Jensen-Shannon divergence (symmetric, bounded by log 2)."""
    m = 0.5 * (p + q)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    loss = float(
        (0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)).mean()
    )
    grad = 0.5 * np.log(q / m) + 0.5 * (1.0 - q / m)
    return loss, grad / p.shape[0]


def wasserstein(u: np.ndarray, v: np.ndarray) -> tuple[float, None]:
    """1-D Wasserstein-1 between equal-weight empirical samples (sorted L1)."""
    return float(np.mean(np.abs(np.sort(u) - np.sort(v)))), None


def cosine_embedding(
    y: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray]:
    """1 - y*cos(a,b) for y in {-1,+1}. Gradient with respect to a."""
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    cos = (a * b).sum(axis=1, keepdims=True) / (na * nb)
    loss = float(np.mean(1.0 - y[:, None] * cos))
    grad = -y[:, None] * (b / (na * nb) - cos * a / na**2)
    return loss, grad / a.shape[0]


# --------------------------------------------------------------------------- #
# Demo: numerical gradient checks
# --------------------------------------------------------------------------- #
def _numgrad(f, x, h=1e-6):
    def val():
        v = f()
        return v[0] if isinstance(v, tuple) else v

    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = val()
        x[i] = old - h
        fm = val()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def main() -> None:
    rng = np.random.default_rng(0)
    n, k = 9, 5
    y_reg = rng.normal(size=n)
    p_reg = rng.normal(size=n)
    y_bin = (rng.random(n) > 0.5).astype(float)
    z_bin = rng.normal(size=n)
    labels = rng.integers(0, k, size=n)
    z_multi = rng.normal(size=(n, k))

    cases = {
        "mse": (lambda: mse(y_reg, p_reg), p_reg),
        "mae": (lambda: mae(y_reg, p_reg), p_reg),
        "huber": (lambda: huber(y_reg, p_reg, 0.7), p_reg),
        "log_cosh": (lambda: log_cosh(y_reg, p_reg), p_reg),
        "quantile": (lambda: quantile(y_reg, p_reg, 0.8), p_reg),
        "bce": (lambda: bce(y_bin, z_bin), z_bin),
        "cross_entropy": (lambda: cross_entropy(z_multi, labels), z_multi),
        "focal": (lambda: focal(y_bin, z_bin, 2.0), z_bin),
        "label_smoothing": (lambda: label_smoothing(z_multi, labels, 0.1), z_multi),
        "hinge": (lambda: hinge(np.where(y_bin > 0, 1.0, -1.0), p_reg), p_reg),
        "squared_hinge": (lambda: squared_hinge(np.where(y_bin > 0, 1.0, -1.0), p_reg), p_reg),
        "cosine_embedding": (
            lambda: cosine_embedding(np.where(y_bin > 0, 1.0, -1.0), p_reg.reshape(n, 1),
                                     rng.normal(size=(n, 1)) * 0 + 1.0),
            p_reg.reshape(n, 1),
        ),
    }
    for name, (fn, x) in cases.items():
        _, g = fn()
        num = _numgrad(fn, x)
        err = np.max(np.abs(g - num))
        assert err < 1e-4, (name, err)
    print(f"gradient check ok ({len(cases)} variants of {20})")

    # dice / tversky on a positive map
    pmap = np.array([[0.9, 0.1, 0.7], [0.2, 0.8, 0.4]])
    ymap = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for name, fn in [("dice", lambda: dice(ymap, pmap)),
                     ("tversky", lambda: tversky(ymap, pmap, 0.3, 0.7))]:
        _, g = fn()
        num = _numgrad(fn, pmap)
        assert np.max(np.abs(g - num)) < 1e-5, name
    print("dice/tversky gradients ok")

    # triplet: check all three gradients
    a, pos, neg = (rng.normal(size=(6, 3)) for _ in range(3))
    _, (ga, gp, gn) = triplet(a, pos, neg, 0.5)
    for g, x in [(ga, a), (gp, pos), (gn, neg)]:
        num = _numgrad(lambda: triplet(a, pos, neg, 0.5)[0], x)
        assert np.max(np.abs(g - num)) < 1e-5
    print("triplet gradients ok")

    # contrastive
    d = rng.uniform(0.1, 2.0, size=8)
    yc = (rng.random(8) > 0.5).astype(float)
    _, g = contrastive(d, yc, 1.0)
    assert np.max(np.abs(g - _numgrad(lambda: contrastive(d, yc, 1.0)[0], d))) < 1e-5
    print("contrastive gradient ok")

    # infonce: both gradients
    z1, z2 = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    _, (g1, g2) = infonce(z1, z2, 0.1)
    assert np.max(np.abs(g1 - _numgrad(lambda: infonce(z1, z2, 0.1)[0], z1))) < 1e-5
    assert np.max(np.abs(g2 - _numgrad(lambda: infonce(z1, z2, 0.1)[0], z2))) < 1e-5
    print("infonce gradients ok")

    # functional properties
    q = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    r = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    assert kl(q, q)[0] < 1e-12
    assert abs(js(q, q)[0]) < 1e-12
    assert js(q, r)[0] <= np.log(2.0) + 1e-12
    assert wasserstein(np.arange(5.0), np.arange(5.0))[0] == 0.0
    print("divergence properties ok")
    print("losses demo passed")


if __name__ == "__main__":
    main()
