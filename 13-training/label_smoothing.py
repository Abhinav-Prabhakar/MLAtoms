"""Label smoothing — four schedules for turning a one-hot target soft.

Chapter:   13-training
Companion: label_smoothing.md
Tier 3 · 4 variants: uniform, per_class, adaptive, mixup_smoothing
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("uniform", "per_class", "adaptive", "mixup_smoothing")

N_CLASSES = 3
COUNTS = (150, 80, 30)
WIDTH = 64
SIGMA = 0.62
RADIUS = 1.25
NOISE_RATE = 0.2
TEST_PER_CLASS = 300
STEPS = 800
LR = 3e-3
WEIGHT_DECAY = 1e-4
ALPHA = 0.1
BINS = 10


# --------------------------------------------------------------------------- #
# Data: three overlapping blobs, deliberately imbalanced and optionally noisy
# --------------------------------------------------------------------------- #
def _ring(seed: int, counts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(N_CLASSES) / N_CLASSES
    angles = angles + rng.uniform(-0.25, 0.25, N_CLASSES)
    centres = RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    x = np.vstack([centres[k] + rng.normal(0.0, SIGMA, (counts[k], 2))
                   for k in range(N_CLASSES)])
    y = np.concatenate([np.full(counts[k], k) for k in range(N_CLASSES)])
    order = rng.permutation(len(y))
    return x[order], y[order]


def _flip(y: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Move a fraction of labels to a different class."""
    rng = np.random.default_rng(seed)
    y = y.copy()
    hit = rng.random(len(y)) < rate
    for i in np.flatnonzero(hit):
        y[i] = rng.choice([k for k in range(N_CLASSES) if k != y[i]])
    return y


def make_data(seed: int = 0, *, noise: float = 0.0) -> tuple:
    """Return (x_train, y_train, x_test, y_test); the test set is clean."""
    x_train, y_train = _ring(seed, COUNTS)
    if noise:
        y_train = _flip(y_train, noise, seed + 5)
    x_test, y_test = _ring(seed + 1, (TEST_PER_CLASS,) * N_CLASSES)
    return x_train, y_train, x_test, y_test


# --------------------------------------------------------------------------- #
# A two-layer MLP and its gradients for an arbitrary soft target
# --------------------------------------------------------------------------- #
def _init(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 2.0), (2, WIDTH)),
        "b1": np.zeros(WIDTH),
        "W2": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, WIDTH)),
        "b2": np.zeros(WIDTH),
        "W3": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, N_CLASSES)),
        "b3": np.zeros(N_CLASSES),
    }


def _forward(params: dict, x: np.ndarray):
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(z2, 0.0)
    return a2 @ params["W3"] + params["b3"], (x, z1, a1, z2, a2)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    return prob / prob.sum(axis=1, keepdims=True)


def _backward(params: dict, x: np.ndarray, target: np.ndarray) -> dict:
    logits, cache = _forward(params, x)
    x_, z1, a1, z2, a2 = cache
    delta = (_softmax(logits) - target) / len(x)
    grads = {"W3": a2.T @ delta, "b3": delta.sum(0)}
    dz2 = (delta @ params["W3"].T) * (z2 > 0.0)
    grads["W2"] = a1.T @ dz2
    grads["b2"] = dz2.sum(0)
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0.0)
    grads["W1"] = x_.T @ dz1
    grads["b1"] = dz1.sum(0)
    return grads


def _adam(params: dict, grads: dict, state: tuple, step: int, lr: float) -> None:
    first, second = state
    for key, grad in grads.items():
        first[key] = 0.9 * first[key] + 0.1 * grad
        second[key] = 0.999 * second[key] + 0.001 * grad ** 2
        scaled = (first[key] / (1.0 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)
        params[key] -= lr * scaled + lr * WEIGHT_DECAY * params[key]


# --------------------------------------------------------------------------- #
# The four target schedules
# --------------------------------------------------------------------------- #
def targets(y: np.ndarray, variant: str, alpha: float, counts: np.ndarray,
            prob: np.ndarray | None = None) -> np.ndarray:
    """Soft target matrix (n, K) for a batch, one row per example."""
    gold = np.zeros((len(y), N_CLASSES))
    gold[np.arange(len(y)), y] = 1.0
    if variant == "adaptive":
        if prob is None:
            raise ValueError("adaptive targets need the current probabilities")
        weight = alpha * prob[np.arange(len(y)), y]
    elif variant == "per_class":
        weight = alpha * counts[y] / counts.max()
    elif variant == "uniform":
        weight = np.full(len(y), alpha)
    else:
        raise ValueError(f"variant {variant!r} has no per-row schedule")
    return (1.0 - weight)[:, None] * gold + (weight / N_CLASSES)[:, None]


def expected_calibration_error(prob: np.ndarray, y: np.ndarray,
                               bins: int = BINS) -> float:
    """Gap between confidence and accuracy, averaged over equal-width bins."""
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    gap = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            gap += inside.mean() * abs(correct[inside].mean()
                                       - confidence[inside].mean())
    return float(gap)


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
class LabelSmoother:
    """Train one model with one target schedule.

    `fit()` returns `self`; `metrics()` reports accuracy, negative log-likelihood
    and expected calibration error. `alphas` holds the per-class weights after
    fitting, and `weight()` gives the per-example weight for a batch.
    """

    def __init__(self, variant: str = "uniform", *, alpha: float = ALPHA,
                 seed: int = 1, steps: int = STEPS, lr: float = LR):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must lie in [0, 1)")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.alpha = float(alpha)
        self.seed = seed
        self.steps = steps
        self.lr = lr
        self.params: dict | None = None
        self.alphas: np.ndarray | None = None

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray) -> LabelSmoother:
        x, y = _check(x, y)
        counts = np.bincount(y, minlength=N_CLASSES).astype(float)
        self.alphas = self.alpha * counts / counts.max()
        params = _init(self.seed)
        state = ({key: np.zeros_like(value) for key, value in params.items()},
                 {key: np.zeros_like(value) for key, value in params.items()})
        rng = np.random.default_rng(self.seed)
        for step in range(1, self.steps + 1):
            if self.variant == "mixup_smoothing":
                lam = rng.beta(0.4, 0.4, len(x))[:, None]
                order = rng.permutation(len(x))
                inputs = lam * x + (1.0 - lam) * x[order]
                left = targets(y, "uniform", self.alpha, counts)
                right = targets(y[order], "uniform", self.alpha, counts)
                batch_target = lam * left + (1.0 - lam) * right
            else:
                inputs = x
                prob = _softmax(_forward(params, x)[0])
                batch_target = targets(y, self.variant, self.alpha, counts,
                                       prob)
            _adam(params, _backward(params, inputs, batch_target), state, step,
                  self.lr)
        self.params = params
        return self

    # -- inference ------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> np.ndarray:
        return _probabilities(self, x).argmax(axis=1)

    def metrics(self, x: np.ndarray, y: np.ndarray) -> dict:
        x, y = _check(x, y)
        if self.params is None:
            raise RuntimeError("call fit first")
        prob = _probabilities(self, x)
        picked = prob[np.arange(len(y)), y]
        return {
            "accuracy": float(np.mean(prob.argmax(axis=1) == y)),
            "nll": float(-np.mean(np.log(picked + 1e-12))),
            "ece": expected_calibration_error(prob, y),
        }

    def weight(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-example smoothing weight under the fitted model."""
        x, y = _check(x, y)
        if self.alphas is None:
            raise RuntimeError("call fit first")
        if self.variant == "per_class":
            return self.alphas[y]
        if self.variant == "adaptive":
            prob = _probabilities(self, x)
            return self.alpha * prob[np.arange(len(y)), y]
        return np.full(len(y), self.alpha)


def _probabilities(model: LabelSmoother, x: np.ndarray) -> np.ndarray:
    if model.params is None:
        raise RuntimeError("call fit first")
    return _softmax(_forward(model.params, x)[0])


def _check(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be (n, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("y must be (n,) and match x")
    if np.any((y < 0) | (y >= N_CLASSES)):
        raise ValueError(f"labels must lie in [0, {N_CLASSES})")
    return x, y


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _line(name: str, alpha: float, scores: dict) -> str:
    return (f"  {name:<16s} {alpha:5.2f} {scores['accuracy']:7.3f} "
            f"{scores['nll']:7.3f} {scores['ece']:7.3f}")


def demo() -> None:
    clean = make_data(noise=0.0)
    noisy = make_data(noise=NOISE_RATE)

    print("label smoothing. classes of 150/80/30 points, 900 balanced test:")
    print(f"  {'schedule':<16s} {'a':>5s} {'acc':>7s} {'nll':>7s} {'ece':>7s}")
    hard = LabelSmoother("uniform", alpha=0.0).fit(clean[0], clean[1])
    hard_scores = hard.metrics(clean[2], clean[3])
    print(_line("hard targets", 0.0, hard_scores))
    clean_scores = {}
    clean_models = {}
    for variant in VARIANTS:
        model = LabelSmoother(variant).fit(clean[0], clean[1])
        clean_models[variant] = model
        clean_scores[variant] = model.metrics(clean[2], clean[3])
        print(_line(variant, ALPHA, clean_scores[variant]))

    print(f"label smoothing. the same data with {int(NOISE_RATE * 100)}% of "
          "training labels flipped:")
    print(f"  {'schedule':<16s} {'a':>5s} {'acc':>7s} {'nll':>7s} {'ece':>7s}")
    hard = LabelSmoother("uniform", alpha=0.0).fit(noisy[0], noisy[1])
    noisy_hard = hard.metrics(noisy[2], noisy[3])
    print(_line("hard targets", 0.0, noisy_hard))
    noisy_scores = {}
    noisy_models = {}
    for variant in VARIANTS:
        model = LabelSmoother(variant).fit(noisy[0], noisy[1])
        noisy_models[variant] = model
        noisy_scores[variant] = model.metrics(noisy[2], noisy[3])
        print(_line(variant, ALPHA, noisy_scores[variant]))

    per_class = clean_models["per_class"].alphas
    adaptive = noisy_models["adaptive"]
    weight = adaptive.weight(noisy[0], noisy[1])
    flipped = noisy[1] != clean[1]
    print("label smoothing. what the schedules do, measured on that batch:")
    print(f"  per-class alpha by class     {np.round(per_class, 3)}")
    print(f"  adaptive alpha, flipped      {weight[flipped].mean():.3f}")
    print(f"  adaptive alpha, clean        {weight[~flipped].mean():.3f}")

    print("label smoothing. malformed calls are refused:")
    bad = [
        lambda: LabelSmoother("uniform_v2"),
        lambda: LabelSmoother("uniform", alpha=1.0),
        lambda: LabelSmoother("uniform", steps=0),
        lambda: LabelSmoother("uniform", lr=0.0),
        lambda: LabelSmoother("uniform").fit(np.zeros((4, 3)), np.zeros(4)),
        lambda: LabelSmoother("uniform").fit(np.zeros((4, 2)), np.zeros(3)),
        lambda: LabelSmoother("uniform").fit(np.zeros((4, 2)), np.full(4, 7)),
        lambda: LabelSmoother("uniform").metrics(np.zeros((4, 2)), np.zeros(4)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)

    uniform_clean = clean_scores["uniform"]
    uniform_noisy = noisy_scores["uniform"]
    print(f"  {refused} of {len(bad)} malformed calls refused")
    assert uniform_clean["ece"] < 0.5 * hard_scores["ece"]
    assert uniform_clean["nll"] < 0.5 * hard_scores["nll"]
    assert abs(uniform_clean["accuracy"] - hard_scores["accuracy"]) < 0.03
    assert per_class[0] > per_class[1] > per_class[2]
    assert noisy_scores["uniform"]["accuracy"] > noisy_hard["accuracy"]
    assert noisy_scores["adaptive"]["accuracy"] > noisy_hard["accuracy"]
    assert noisy_scores["mixup_smoothing"]["accuracy"] > noisy_hard["accuracy"]
    assert noisy_scores["mixup_smoothing"]["ece"] > uniform_noisy["ece"]
    sample = targets(np.array([0, 1]), "uniform", 0.1,
                     np.array([2.0, 2.0, 2.0]))
    assert np.allclose(sample.sum(axis=1), 1.0)
    assert sample.min() >= 0.0 and sample.max() < 1.0
    print("label smoothing demo passed")


if __name__ == "__main__":
    demo()
