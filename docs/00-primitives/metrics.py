"""Metrics — 18 scores, each with the property that makes it the right choice.

Chapter:   00-primitives
Companion: metrics.md
Tier 1 · 18 variants: accuracy, balanced_accuracy, precision_recall_f1,
micro_macro_weighted, confusion, roc_auc, pr_auc, average_precision, log_loss,
brier, mcc, cohen_kappa, ndcg, map, mrr, hit_rate, silhouette, adjusted_rand

All functions are pure and take arrays. Classification metrics assume integer
labels in [0, K); ranking metrics take a relevance/binary array plus a score array.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int | None = None) -> np.ndarray:
    """C[i, j] = #(true i, predicted j)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = int(max(y_true.max(), y_pred.max())) + 1 if k is None else k
    c = np.zeros((k, k), dtype=int)
    np.add.at(c, (y_true, y_pred), 1)
    return c


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall: immune to class imbalance, blind to prevalence."""
    c = confusion_matrix(y_true, y_pred)
    recall = np.diag(c) / np.maximum(c.sum(axis=1), 1)
    present = c.sum(axis=1) > 0
    return float(recall[present].mean())


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", beta: float = 1.0
) -> tuple[float, float, float]:
    """Per-class P/R/F_beta reduced by 'macro' (unweighted), 'micro' (pooled),
    'weighted' (support-weighted) or 'binary' (positive class = 1)."""
    c = confusion_matrix(y_true, y_pred)
    tp = np.diag(c).astype(float)
    fp = c.sum(axis=0) - tp
    fn = c.sum(axis=1) - tp
    support = c.sum(axis=1).astype(float)

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        p = tp / max(tp + fp, 1e-12)
        r = tp / max(tp + fn, 1e-12)
        f = (1 + beta**2) * p * r / max(beta**2 * p + r, 1e-12)
        return float(p), float(r), float(f)
    if average == "binary":
        p = tp[1] / max(tp[1] + fp[1], 1e-12)
        r = tp[1] / max(tp[1] + fn[1], 1e-12)
        f = (1 + beta**2) * p * r / max(beta**2 * p + r, 1e-12)
        return float(p), float(r), float(f)

    p = tp / np.maximum(tp + fp, 1e-12)
    r = tp / np.maximum(tp + fn, 1e-12)
    f = (1 + beta**2) * p * r / np.maximum(beta**2 * p + r, 1e-12)
    present = support > 0
    if average == "weighted":
        w = support[present] / support[present].sum()
        return float((p[present] * w).sum()), float((r[present] * w).sum()), float((f[present] * w).sum())
    if average == "macro":
        return float(p[present].mean()), float(r[present].mean()), float(f[present].mean())
    raise ValueError(f"unknown average {average!r}")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (the Mann-Whitney convention)."""
    a = np.asarray(a, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties = 1/2.

    Computed from ranks, so it is O(n log n) and exact with ties.
    """
    y = np.asarray(y_true).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes")
    r = _rankdata(scores)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, step-wise (the honest AP)."""
    y = np.asarray(y_true).astype(bool)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal area under the PR curve over unique thresholds."""
    y = np.asarray(y_true).astype(float)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(s)[::-1]
    prec, rec = [1.0], [0.0]
    for t in thresholds:
        pred = s >= t
        tp = float((pred & (y == 1)).sum())
        prec.append(tp / max(pred.sum(), 1))
        rec.append(tp / max((y == 1).sum(), 1))
    return float(np.trapezoid(np.array(prec), np.array(rec)))


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probabilities: calibration and sharpness together."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation: balanced even at 1:1000, undefined if a row/col is empty."""
    c = confusion_matrix(y_true, y_pred, k=2).astype(float)
    tp, tn, fp, fn = c[1, 1], c[0, 0], c[0, 1], c[1, 0]
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Agreement corrected for chance: (p_o - p_e)/(1 - p_e)."""
    c = confusion_matrix(y_true, y_pred).astype(float)
    n = c.sum()
    po = np.trace(c) / n
    pe = float((c.sum(axis=0) * c.sum(axis=1)).sum()) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def ndcg(rel: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Normalised discounted cumulative gain with exponential gain 2^rel - 1."""
    rel = np.asarray(rel, dtype=float)
    k = rel.size if k is None else k
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    gains = 2.0 ** rel[order] - 1.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(((2.0**ideal - 1.0) * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(rel: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Mean average precision. rel/scores: (n_queries, n_items)."""
    rel = np.atleast_2d(np.asarray(rel))
    scores = np.atleast_2d(np.asarray(scores))
    aps = []
    for r, s in zip(rel, scores, strict=True):
        ap = average_precision(r.astype(bool), s)
        if k is not None:
            order = np.argsort(-s, kind="mergesort")[:k]
            ap = average_precision(r[order].astype(bool), s[order])
        aps.append(ap)
    return float(np.mean(aps))


def mrr(rel: np.ndarray, scores: np.ndarray) -> float:
    """Mean reciprocal rank of the first relevant item, per query."""
    rel = np.atleast_2d(np.asarray(rel)).astype(bool)
    scores = np.atleast_2d(np.asarray(scores))
    out = []
    for r, s in zip(rel, scores, strict=True):
        order = np.argsort(-s, kind="mergesort")
        hits = np.flatnonzero(r[order])
        out.append(1.0 / (hits[0] + 1) if hits.size else 0.0)
    return float(np.mean(out))


def hit_rate(rel: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Fraction of queries with at least one relevant item in the top k."""
    rel = np.atleast_2d(np.asarray(rel)).astype(bool)
    scores = np.atleast_2d(np.asarray(scores))
    hits = []
    for r, s in zip(rel, scores, strict=True):
        order = np.argsort(-s, kind="mergesort")[:k]
        hits.append(bool(r[order].any()))
    return float(np.mean(hits))


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette: (b - a)/max(a, b) with a = mean intra-cluster distance and
    b = mean distance to the nearest other cluster. O(n^2) memory."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    uniq = np.unique(labels)
    if uniq.size < 2:
        raise ValueError("silhouette needs at least 2 clusters")
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        if own.sum() == 0:  # singleton cluster: silhouette is 0 by convention
            continue
        a = d[i, own].mean()
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def adjusted_rand(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rand index corrected for chance: 1 for identical partitions, ~0 for random."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _, ti = np.unique(y_true, return_inverse=True)
    _, pi = np.unique(y_pred, return_inverse=True)
    c = np.zeros((ti.max() + 1, pi.max() + 1), dtype=float)
    np.add.at(c, (ti, pi), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_comb = comb(c).sum()
    a = comb(c.sum(axis=1)).sum()
    b = comb(c.sum(axis=0)).sum()
    expected = a * b / comb(len(y_true))
    max_index = 0.5 * (a + b)
    return float((sum_comb - expected) / (max_index - expected)) if max_index != expected else 1.0


def main() -> None:
    rng = np.random.default_rng(0)

    # Binary case worked by hand: scores [0.1, 0.4, 0.35, 0.8], labels [0,0,1,1].
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(roc_auc(y, s) - 0.75) < 1e-12
    assert abs(average_precision(y, s) - (1.0 + 2 / 3) / 2) < 1e-12
    assert roc_auc(y, y.astype(float)) == 1.0
    assert roc_auc(y, -y.astype(float)) == 0.0
    print(f"auc={roc_auc(y, s):.3f} ap={average_precision(y, s):.3f}")

    # Perfect and inverted classifiers.
    yt = rng.integers(0, 3, size=200)
    yb = rng.integers(0, 2, size=200)
    assert accuracy(yt, yt) == 1.0
    assert mcc(yb, yb) == 1.0
    assert cohen_kappa(yb, yb) == 1.0
    assert adjusted_rand(yt, yt) == 1.0
    p, r, f1 = precision_recall_f1(yt, yt, "macro")
    assert (p, r, f1) == (1.0, 1.0, 1.0)
    assert mcc(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == -1.0
    print("classification metrics: perfect = 1, inverted mcc = -1")

    # Imbalanced case: accuracy lies, mcc and balanced accuracy do not.
    yt = np.array([0] * 990 + [1] * 10)
    yp = np.zeros(1000, dtype=int)
    assert accuracy(yt, yp) == 0.99
    assert balanced_accuracy(yt, yp) == 0.5
    assert mcc(yt, yp) == 0.0
    print("imbalance: accuracy 0.99, balanced 0.5, mcc 0.0")

    # Ranking metrics.
    rel = np.array([3, 2, 1, 0, 0])
    assert ndcg(rel, rel) == 1.0                       # ideal order
    assert ndcg(rel, -rel) < ndcg(rel, rel)
    assert map_at_k(np.array([[1, 0, 1, 0]]), np.array([[0.9, 0.1, 0.8, 0.2]])) == 1.0
    assert mrr(np.array([[0, 1, 0]]), np.array([[0.9, 0.8, 0.1]])) == 0.5
    assert hit_rate(np.array([[0, 1, 0]]), np.array([[0.9, 0.8, 0.1]]), k=1) == 0.0
    print("ranking metrics ok")

    # Clustering: two well-separated blobs score high, random labels score ~0.
    X = np.r_[rng.normal(0, 0.3, (60, 2)), rng.normal(6, 0.3, (60, 2))]
    lab = np.r_[np.zeros(60), np.ones(60)]
    assert silhouette(X, lab) > 0.9
    assert adjusted_rand(lab, lab) == 1.0
    assert abs(adjusted_rand(lab, rng.integers(0, 2, 120))) < 0.15
    print(f"silhouette={silhouette(X, lab):.3f}, ari(random)~0")

    # Calibration-shaped metrics.
    assert brier(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert log_loss(np.array([1, 0]), np.array([1.0 - 1e-9, 1e-9])) < 1e-6
    print("metrics demo passed")


if __name__ == "__main__":
    main()
