"""Federated learning — six ways to combine models that never see the data.

Chapter:   13-training
Companion: federated.md
Tier 3 · 6 variants: fedavg, fedprox, scaffold, secure_agg, dp_fedavg,
personalization
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("fedavg", "fedprox", "scaffold", "secure_agg", "dp_fedavg",
            "personalization")

NCLS = 10
CLIENTS = 5
RADIUS = 1.6
SPREAD = 0.4
WIDTH = 24
PER_CLASS = 40
LOCAL_EPOCHS = (40, 40, 40, 40, 200)
ROUNDS = 30
LR = 0.02
MU = 0.1
CLIP = 1.0
NOISE = 0.1
FINETUNE = 200
MASK = 1.0


# --------------------------------------------------------------------------- #
# Data: ten classes on a ring, split so that no client ever sees them all
# --------------------------------------------------------------------------- #
def ring(seed: int, per_class: int = 150) -> tuple:
    """Return (x, y) with `per_class` labelled points on each of ten arcs."""
    rng = np.random.default_rng(seed)
    angle = 2.0 * np.pi * np.arange(NCLS) / NCLS
    centres = RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    x = np.vstack([c + rng.normal(0.0, SPREAD, (per_class, 2)) for c in centres])
    y = np.repeat(np.arange(NCLS), per_class)
    order = rng.permutation(len(y))
    return x[order], y[order]


def partition(seed: int = 10, per_class: int = PER_CLASS) -> list:
    """Return CLIENTS (x, y) pairs; client j holds classes 2j and 2j+1."""
    rng = np.random.default_rng(seed)
    angle = 2.0 * np.pi * np.arange(NCLS) / NCLS
    centres = RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    out = []
    for j in range(CLIENTS):
        rows, labels = [], []
        for c in (2 * j, 2 * j + 1):
            rows.append(centres[c] + rng.normal(0.0, SPREAD, (per_class, 2)))
            labels.append(np.full(per_class, c))
        x = np.vstack(rows)
        y = np.concatenate(labels)
        order = rng.permutation(len(y))
        out.append((x[order], y[order]))
    return out


# --------------------------------------------------------------------------- #
# A 2 -> 24 -> 24 -> 10 network, one full-batch step per local epoch
# --------------------------------------------------------------------------- #
def _init(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 2.0), (2, WIDTH)),
        "b1": np.zeros(WIDTH),
        "W2": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, WIDTH)),
        "b2": np.zeros(WIDTH),
        "W3": rng.normal(0.0, np.sqrt(2.0 / WIDTH), (WIDTH, NCLS)),
        "b3": np.zeros(NCLS),
    }


def _forward(model: dict, x: np.ndarray):
    z1 = x @ model["W1"] + model["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ model["W2"] + model["b2"]
    a2 = np.maximum(z2, 0.0)
    return a2 @ model["W3"] + model["b3"], (x, z1, a1, z2, a2)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    return prob / prob.sum(axis=1, keepdims=True)


def _prob(model: dict, x: np.ndarray) -> np.ndarray:
    return _softmax(_forward(model, x)[0])


def _loss(model: dict, x: np.ndarray, y: np.ndarray) -> float:
    picked = _prob(model, x)[np.arange(len(y)), y]
    return float(-np.mean(np.log(picked + 1e-12)))


def _accuracy(model: dict, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(_forward(model, x)[0].argmax(axis=1) == y))


def _grad(model: dict, x: np.ndarray, y: np.ndarray) -> dict:
    logits, (x_, z1, a1, z2, a2) = _forward(model, x)
    delta = _softmax(logits)
    delta[np.arange(len(y)), y] -= 1.0
    delta /= len(y)
    grads = {"W3": a2.T @ delta, "b3": delta.sum(0)}
    dz2 = (delta @ model["W3"].T) * (z2 > 0.0)
    grads["W2"] = a1.T @ dz2
    grads["b2"] = dz2.sum(0)
    dz1 = (dz2 @ model["W2"].T) * (z1 > 0.0)
    grads["W1"] = x_.T @ dz1
    grads["b1"] = dz1.sum(0)
    return grads


def _copy(model: dict) -> dict:
    return {key: value.copy() for key, value in model.items()}


def _zeros(model: dict) -> dict:
    return {key: np.zeros_like(value) for key, value in model.items()}


def _norm(model: dict) -> float:
    return float(np.sqrt(sum(float((v ** 2).sum()) for v in model.values())))


def local_sgd(model: dict, x: np.ndarray, y: np.ndarray, epochs: int,
              lr: float, *, mu: float = 0.0, anchor: dict | None = None,
              c: dict | None = None, ci: dict | None = None) -> dict:
    """Run `epochs` full-batch steps from `model` and return the new model.

    `mu` adds the FedProx term against `anchor`; `c`/`ci` add the SCAFFOLD
    control-variate correction.
    """
    current = _copy(model)
    for _ in range(epochs):
        grads = _grad(current, x, y)
        for key, grad in grads.items():
            if c is not None:
                grad = grad + c[key] - ci[key]
            if mu:
                grad = grad + mu * (current[key] - anchor[key])
            current[key] = current[key] - lr * grad
    return current


def mask_updates(models: list, *, scale: float = MASK, seed: int = 0) -> tuple:
    """Mask each model with pairwise shares; returns (masked, total).

    Every share added by client i for client j is subtracted by j, so the
    masked models sum to the same thing the plain models do.
    """
    rng = np.random.default_rng(seed)
    n = len(models)
    share = {(i, j): {k: rng.normal(0.0, scale, v.shape) for k, v in models[0].items()}
             for i in range(n) for j in range(i + 1, n)}
    masked = []
    for i in range(n):
        out = _copy(models[i])
        for j in range(n):
            if j > i:
                for key in out:
                    out[key] = out[key] + share[i, j][key]
            elif j < i:
                for key in out:
                    out[key] = out[key] - share[j, i][key]
        masked.append(out)
    total = {key: sum(m[key] for m in masked) for key in models[0]}
    return masked, total


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #
class Federated:
    """Run `rounds` of federated training over a list of (x, y) clients.

    `history` holds the monitored accuracy after each round, `personal` the
    per-client models for `personalization`, and `clipped` the number of client
    updates that reached the clipping norm in `dp_fedavg`.
    """

    def __init__(self, variant: str = "fedavg", *, rounds: int = ROUNDS,
                 lr: float = LR, mu: float = MU, epochs=LOCAL_EPOCHS,
                 clip: float = CLIP, noise: float = NOISE,
                 finetune: int = FINETUNE, seed: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if rounds < 1:
            raise ValueError("rounds must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError("mu must be non-negative and finite")
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip must be positive and finite")
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError("noise must be non-negative and finite")
        if finetune < 0:
            raise ValueError("finetune must be non-negative")
        self.variant = variant
        self.rounds = rounds
        self.lr = lr
        self.mu = mu
        self.epochs = epochs
        self.clip = clip
        self.noise = noise
        self.finetune = finetune
        self.seed = seed
        self.model: dict | None = None
        self.personal: list | None = None
        self.history: list[float] = []
        self.clipped = 0
        self.last_updates: list = []
        self.comm = 2.0 if variant == "scaffold" else 1.0

    # -- training ------------------------------------------------------------- #
    def fit(self, clients: list, monitor: tuple | None = None) -> Federated:
        n = len(clients)
        if n < 2:
            raise ValueError("federated learning needs at least two clients")
        clients = [_check(x, y) for x, y in clients]
        epochs = self._schedule(n)
        self.model = _init(self.seed)
        self.history = []
        self.clipped = 0
        c = _zeros(self.model)
        ci = [_zeros(self.model) for _ in range(n)]
        for turn in range(1, self.rounds + 1):
            if self.variant == "scaffold":
                updates = []
                for j, (x, y) in enumerate(clients):
                    updates.append(local_sgd(self.model, x, y, epochs[j],
                                             self.lr, c=c, ci=ci[j]))
                drift = [{k: (self.model[k] - q[k]) / (epochs[j] * self.lr)
                          for k in q} for j, q in enumerate(updates)]
                self.model = self._average(updates)
                c = {k: np.mean([d[k] for d in drift], axis=0) for k in c}
                ci = drift
            else:
                updates = []
                for j, (x, y) in enumerate(clients):
                    updates.append(local_sgd(
                        self.model, x, y, epochs[j], self.lr,
                        mu=self.mu if self.variant == "fedprox" else 0.0,
                        anchor=self.model))
                if self.variant == "dp_fedavg":
                    self.model = self._noisy_average(updates, turn)
                elif self.variant == "secure_agg":
                    self.model = self._masked_average(updates, turn)
                else:
                    self.model = self._average(updates)
            self.last_updates = updates
            if monitor is not None:
                self.history.append(_accuracy(self.model, *monitor))
        if self.variant == "personalization":
            self.personal = [local_sgd(self.model, x, y, self.finetune, self.lr)
                             for x, y in clients]
        return self

    # -- aggregation ---------------------------------------------------------- #
    def _average(self, updates: list) -> dict:
        return {key: np.mean([u[key] for u in updates], axis=0)
                for key in updates[0]}

    def _noisy_average(self, updates: list, turn: int) -> dict:
        rng = np.random.default_rng(self.seed + turn)
        n = len(updates)
        deltas, scales = [], []
        for update in updates:
            delta = {key: self.model[key] - update[key] for key in update}
            scales.append(min(1.0, self.clip / (_norm(delta) + 1e-12)))
            deltas.append(delta)
        self.clipped += sum(1 for s in scales if s < 1.0)
        pairs = list(zip(deltas, scales, strict=True))
        out = {}
        for key in self.model:
            mean = np.mean([d[key] * s for d, s in pairs], axis=0)
            noise = rng.normal(0.0, self.noise * self.clip / n,
                               self.model[key].shape)
            out[key] = self.model[key] - mean + noise
        return out

    def _masked_average(self, updates: list, turn: int) -> dict:
        masked, total = mask_updates(updates, seed=self.seed + turn)
        self.masked = masked
        return {key: total[key] / len(updates) for key in total}

    # -- evaluation ----------------------------------------------------------- #
    def metrics(self, x: np.ndarray, y: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("call fit first")
        x, y = _check(x, y)
        return {"loss": _loss(self.model, x, y),
                "accuracy": _accuracy(self.model, x, y)}

    def local_metrics(self, tests: list) -> list:
        """Per-client accuracy, using the personal model when there is one."""
        if self.model is None:
            raise RuntimeError("call fit first")
        models = self.personal if self.personal is not None else (
            [self.model] * len(tests))
        return [_accuracy(m, *_check(x, y))
                for m, (x, y) in zip(models, tests, strict=True)]

    def _schedule(self, n: int) -> tuple:
        epochs = self.epochs
        if isinstance(epochs, (int, np.integer)):
            epochs = (int(epochs),) * n
        epochs = tuple(int(e) for e in epochs)
        if len(epochs) != n:
            raise ValueError("epochs must have one entry per client")
        if any(e < 1 for e in epochs):
            raise ValueError("every client needs at least one local epoch")
        return epochs


def _check(x: np.ndarray, y: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be (n, 2)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("y must be (n,) and match x")
    if np.any((y < 0) | (y >= NCLS)):
        raise ValueError(f"y must lie in [0, {NCLS})")
    return x, y


def centralised(clients: list, steps: int, *, lr: float = LR,
                seed: int = 1) -> dict:
    """The reference: one model trained on the pooled data for `steps`."""
    model = _init(seed)
    x = np.vstack([c[0] for c in clients])
    y = np.concatenate([c[1] for c in clients])
    for _ in range(steps):
        grads = _grad(model, x, y)
        for key, grad in grads.items():
            model[key] = model[key] - lr * grad
    return model


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    clients = partition()
    xt, yt = ring(99)
    tests = partition(100, per_class=100)
    pooled = sum(len(y) for _, y in clients)
    budget = sum(len(y) * e
                 for (_, y), e in zip(clients, LOCAL_EPOCHS, strict=True))
    steps = budget * ROUNDS // pooled

    print(f"federated learning. {CLIENTS} clients, {NCLS} classes, "
          "2 classes each:")
    print(f"  {len(clients[0][1])} points per client, local budgets "
          f"{list(LOCAL_EPOCHS)}, {ROUNDS} rounds")
    print(f"  {len(yt)} test points, {budget * ROUNDS // 1000}k examples, "
          f"{steps} pooled steps for the reference")

    reference = centralised(clients, steps)
    runs = {}
    for name in ("fedavg", "fedprox", "scaffold", "secure_agg", "dp_fedavg",
                 "personalization"):
        runs[name] = Federated(name).fit(clients, monitor=(xt, yt))

    print(f"  {'variant':<15s} {'global':>7s} {'local':>7s} {'worst':>6s} "
          f"{'comm':>6s}")
    central_local = [_accuracy(reference, *t) for t in tests]
    print(f"  {'centralised':<15s} {_accuracy(reference, xt, yt):7.3f} "
          f"{np.mean(central_local):7.3f} {np.min(central_local):6.3f} "
          f"{'--':>6s}")
    for name, run in runs.items():
        local = run.local_metrics(tests)
        print(f"  {name:<15s} {run.metrics(xt, yt)['accuracy']:7.3f} "
              f"{np.mean(local):7.3f} {np.min(local):6.3f} "
              f"{run.comm:5.0f}x")

    print("federated learning. test accuracy by round:")
    print(f"  {'round':>5s} {'fedavg':>7s} {'fedprox':>8s} {'scaffold':>9s}")
    for turn in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30):
        print(f"  {turn:5d} {runs['fedavg'].history[turn - 1]:7.3f} "
              f"{runs['fedprox'].history[turn - 1]:8.3f} "
              f"{runs['scaffold'].history[turn - 1]:9.3f}")

    last = runs["fedavg"].last_updates
    spread = max(_norm({k: a[k] - b[k] for k in a})
                 for i, a in enumerate(last) for b in last[i + 1:])
    masked, total = mask_updates(last[:2])
    hidden = _norm({k: masked[0][k] - last[0][k] for k in masked[0]})
    gap = max(float(np.abs(total[k] - last[0][k] - last[1][k]).max())
              for k in total)
    print("federated learning. what the server is allowed to see:")
    print(f"  the masks move one update {hidden:.1f} from its true value; "
          f"two clients are {spread:.1f} apart")
    print(f"  the masked sum is off by {gap:.1e} and averages to the plain "
          "mean")

    print("federated learning. noise against the clipping norm:")
    sweeps = {}
    for noise in (0.0, 0.1, 0.2, 0.4):
        sweeps[noise] = Federated("dp_fedavg", noise=noise).fit(clients)
        local = sweeps[noise].local_metrics(tests)
        scores = sweeps[noise].metrics(xt, yt)
        print(f"  noise {noise:4.2f}  global {scores['accuracy']:.3f}  "
              f"worst {np.min(local):.3f}  "
              f"clipped {sweeps[noise].clipped} updates")

    personal = runs["personalization"]
    base = runs["fedavg"].local_metrics(tests)
    own = personal.local_metrics(tests)
    print("federated learning. one global model, then one per client:")
    for j in range(CLIENTS):
        print(f"  client {j}  global {base[j]:.3f}  personal {own[j]:.3f}")

    print("federated learning. malformed calls are refused:")
    bad = [
        lambda: Federated("fedavg_v2"),
        lambda: Federated("fedavg", rounds=0),
        lambda: Federated("fedavg", lr=0.0),
        lambda: Federated("fedprox", mu=-0.1),
        lambda: Federated("dp_fedavg", clip=0.0),
        lambda: Federated("dp_fedavg", noise=-1.0),
        lambda: Federated("fedavg").fit(clients[:1]),
        lambda: Federated("fedavg", epochs=(20, 20)).fit(clients),
        lambda: Federated("fedavg").metrics(xt, yt),
        lambda: Federated("fedavg").fit([(np.zeros((4, 3)), np.zeros(4))]
                                        + clients[1:]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    avg = runs["fedavg"]
    prox = runs["fedprox"]
    scaf = runs["scaffold"]
    scores = {name: runs[name].metrics(xt, yt)["accuracy"] for name in runs}
    assert scores["fedavg"] < _accuracy(reference, xt, yt)
    assert scores["fedprox"] > scores["fedavg"]
    assert scores["scaffold"] > scores["fedavg"]
    assert np.min(prox.local_metrics(tests)) > np.min(avg.local_metrics(tests))
    assert scaf.history[9] > avg.history[9] + 0.05
    assert all(np.allclose(runs["secure_agg"].model[k], avg.model[k],
                           atol=1e-9) for k in avg.model)
    assert gap < 1e-9
    assert hidden > 10.0 * spread
    assert scaf.comm == 2.0 * avg.comm
    assert np.mean(own) > np.mean(base) + 0.1
    assert sweeps[0.4].metrics(xt, yt)["accuracy"] < scores["fedavg"]
    assert sweeps[0.0].clipped > 0
    print(f"  fedavg {scores['fedavg']:.3f} vs centralised "
          f"{_accuracy(reference, xt, yt):.3f}; fedprox recovers "
          f"{scores['fedprox'] - scores['fedavg']:+.3f} and scaffold "
          f"{scores['scaffold'] - scores['fedavg']:+.3f}")
    print("federated demo passed")


if __name__ == "__main__":
    demo()
