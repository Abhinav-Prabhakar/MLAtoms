"""Early stopping — four ways to decide when a training run is finished.

Chapter:   13-training
Companion: early_stopping.md
Tier 3 · 4 variants: patience, min_delta, restore_best, plateau_decay
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("patience", "min_delta", "restore_best", "plateau_decay")

INNER = 0.55
OUTER = 1.45
SPREAD = 0.16
TRAIN = 240
VALID = 600
TEST = 1000
NOISE = 0.1
WIDTH = 48
EPOCHS = 400
LR = 5e-3
PATIENCE = 15
MIN_DELTA = 0.02
FACTOR = 0.5


# --------------------------------------------------------------------------- #
# Data: two concentric rings, with a fraction of the training labels flipped
# --------------------------------------------------------------------------- #
def make_data(seed: int, n: int, noise: float = 0.0) -> tuple:
    """Return (x, y) with n points split evenly between the two rings."""
    rng = np.random.default_rng(seed)
    half = n // 2
    radius = np.concatenate([rng.normal(INNER, SPREAD, half),
                             rng.normal(OUTER, SPREAD, n - half)])
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    x = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    y = np.concatenate([np.zeros(half, dtype=int),
                        np.ones(n - half, dtype=int)])
    if noise:
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
    order = rng.permutation(n)
    return x[order], y[order]


# --------------------------------------------------------------------------- #
# A two-layer MLP, one full-batch step per epoch
# --------------------------------------------------------------------------- #
def _init(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 2.0), (2, WIDTH)),
        "b1": np.zeros(WIDTH),
        "W2": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, WIDTH)),
        "b2": np.zeros(WIDTH),
        "W3": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, 2)),
        "b3": np.zeros(2),
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


def _loss(params: dict, x: np.ndarray, y: np.ndarray) -> float:
    prob = _softmax(_forward(params, x)[0])
    picked = prob[np.arange(len(y)), y]
    return float(-np.mean(np.log(picked + 1e-12)))


def _backward(params: dict, x: np.ndarray, y: np.ndarray) -> dict:
    logits, cache = _forward(params, x)
    x_, z1, a1, z2, a2 = cache
    delta = _softmax(logits)
    delta[np.arange(len(y)), y] -= 1.0
    delta /= len(y)
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
        params[key] -= lr * (first[key] / (1.0 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)


def _new_state(params: dict) -> tuple:
    zeros = {key: np.zeros_like(value) for key, value in params.items()}
    return {key: value.copy() for key, value in zeros.items()}, zeros


def _copy(params: dict) -> dict:
    return {key: value.copy() for key, value in params.items()}


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
class EarlyStopper:
    """Train on (x, y), watch (x_val, y_val), and decide when to stop.

    `history` holds the validation loss after every epoch, `stopped` the number
    of epochs actually run, and `best_epoch` the epoch whose weights are kept
    when the variant is `restore_best`.
    """

    def __init__(self, variant: str = "patience", *, patience: int = PATIENCE,
                 min_delta: float = MIN_DELTA, factor: float = FACTOR,
                 epochs: int = EPOCHS, lr: float = LR, seed: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if patience < 1:
            raise ValueError("patience must be at least one epoch")
        if not np.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("min_delta must be non-negative and finite")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must lie in (0, 1)")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.patience = patience
        self.min_delta = min_delta
        self.factor = factor
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.params: dict | None = None
        self.history: list[float] = []
        self.stopped = 0
        self.best_epoch = 0
        self.best_val = float("inf")
        self.decays = 0

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray) -> EarlyStopper:
        x, y = _check(x, y)
        x_val, y_val = _check(x_val, y_val)
        params = _init(self.seed)
        state = _new_state(params)
        best = float("inf")
        best_params = _copy(params)
        best_epoch = 0
        bad = 0
        lr = self.lr
        threshold = self.min_delta if self.variant == "min_delta" else 0.0
        self.history = []
        self.decays = 0
        self.stopped = self.epochs
        for epoch in range(1, self.epochs + 1):
            _adam(params, _backward(params, x, y), state, epoch, lr)
            current = _loss(params, x_val, y_val)
            self.history.append(current)
            if current < best - threshold:
                best = current
                best_epoch = epoch
                bad = 0
                best_params = _copy(params)
            else:
                bad += 1
            if self.variant == "plateau_decay":
                if bad >= self.patience:
                    lr *= self.factor
                    self.decays += 1
                    bad = 0
            elif bad >= self.patience:
                self.stopped = epoch
                break
        if self.variant == "restore_best":
            params = best_params
        self.params = params
        self.lr = lr
        self.best_epoch = best_epoch
        self.best_val = best
        return self

    # -- evaluation ----------------------------------------------------------- #
    def metrics(self, x: np.ndarray, y: np.ndarray) -> dict:
        x, y = _check(x, y)
        if self.params is None:
            raise RuntimeError("call fit first")
        prob = _softmax(_forward(self.params, x)[0])
        return {
            "loss": _loss(self.params, x, y),
            "accuracy": float(np.mean(prob.argmax(axis=1) == y)),
        }


def fixed_budget(x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, *, epochs: int = EPOCHS,
                 seed: int = 1) -> EarlyStopper:
    """The baseline: a patience longer than the budget can never trigger."""
    model = EarlyStopper("patience", patience=epochs + 1, epochs=epochs,
                         seed=seed)
    return model.fit(x, y, x_val, y_val)


def _check(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be (n, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("y must be (n,) and match x")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("y must be binary")
    return x, y


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    x, y = make_data(0, TRAIN, NOISE)
    x_val, y_val = make_data(1, VALID)
    x_test, y_test = make_data(2, TEST)

    print(f"early stopping. {TRAIN} training points, "
          f"{int(NOISE * 100)}% of the labels flipped:")
    print(f"  {VALID} validation points, {TEST} test points, {EPOCHS} epochs "
          "of budget")
    print(f"  {'schedule':<16s} {'epochs':>7s} {'best':>6s} {'train':>7s} "
          f"{'val':>7s} {'test':>7s} {'acc':>6s}")

    runs = {}
    baseline = fixed_budget(x, y, x_val, y_val)
    for name, model in (
            ("fixed budget", baseline),
            ("patience 15", EarlyStopper("patience")),
            ("min_delta 0.02", EarlyStopper("min_delta")),
            ("restore_best", EarlyStopper("restore_best")),
            ("plateau_decay", EarlyStopper("plateau_decay"))):
        model.fit(x, y, x_val, y_val)
        runs[name] = model
        train_scores = model.metrics(x, y)
        val_scores = model.metrics(x_val, y_val)
        test_scores = model.metrics(x_test, y_test)
        print(f"  {name:<16s} {model.stopped:7d} {model.best_epoch:6d} "
              f"{train_scores['loss']:7.3f} {val_scores['loss']:7.3f} "
              f"{test_scores['loss']:7.3f} {test_scores['accuracy']:6.3f}")

    decay = runs["plateau_decay"]
    print("early stopping. the learning-rate schedule:")
    print(f"  plateau_decay halved the rate {decay.decays} times, "
          f"from {decay.lr / decay.factor ** decay.decays:.0e} "
          f"to {decay.lr:.1e}")

    print("early stopping. malformed calls are refused:")
    bad = [
        lambda: EarlyStopper("patience_v2"),
        lambda: EarlyStopper("patience", patience=0),
        lambda: EarlyStopper("patience", min_delta=-1.0),
        lambda: EarlyStopper("patience", factor=1.0),
        lambda: EarlyStopper("patience", epochs=0),
        lambda: EarlyStopper("patience").fit(
            np.zeros((4, 3)), np.zeros(4), np.zeros((4, 2)), np.zeros(4)),
        lambda: EarlyStopper("patience").fit(
            np.zeros((4, 2)), np.full(4, 7), np.zeros((4, 2)), np.zeros(4)),
        lambda: EarlyStopper("patience").metrics(np.zeros((4, 2)), np.zeros(4)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)

    patience = runs["patience 15"]
    restore = runs["restore_best"]
    early = runs["min_delta 0.02"]
    base_test = runs["fixed budget"].metrics(x_test, y_test)["loss"]
    patience_test = patience.metrics(x_test, y_test)["loss"]
    restore_test = restore.metrics(x_test, y_test)["loss"]
    decay_test = decay.metrics(x_test, y_test)["loss"]
    print(f"  {refused} of {len(bad)} malformed calls refused")
    assert baseline.stopped == EPOCHS
    assert patience_test < base_test
    assert restore_test <= patience_test
    assert early.stopped < patience.stopped
    assert decay_test < base_test
    assert decay.decays > 0 and decay.lr < decay.factor
    assert restore.best_epoch == int(np.argmin(restore.history)) + 1
    base_val = runs["fixed budget"].metrics(x_val, y_val)["loss"]
    assert base_val > patience.metrics(x_val, y_val)["loss"]
    assert (runs["fixed budget"].metrics(x, y)["loss"]
            < patience.metrics(x, y)["loss"])
    print(f"  patience saves {EPOCHS - patience.stopped} of {EPOCHS} epochs "
          "and restore_best keeps the lowest validation loss")
    print("early stopping demo passed")


if __name__ == "__main__":
    demo()
