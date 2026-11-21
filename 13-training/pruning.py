"""Pruning — eight ways to delete weights and keep the accuracy.

Chapter:   13-training
Companion: pruning.md
Tier 2 · 8 variants: magnitude, unstructured, structured, movement,
gradual, one_shot, lottery_ticket, svd_prune
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("magnitude", "unstructured", "structured", "movement",
            "gradual", "one_shot", "lottery_ticket", "svd_prune")
LEARNED = ("W1", "W2")


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _forward(parameters: dict, masks: dict, x: np.ndarray):
    pre = x @ (parameters["W1"] * masks["W1"]) + parameters["b1"]
    hidden = np.tanh(pre)
    logits = hidden @ (parameters["W2"] * masks["W2"]) + parameters["b2"]
    return logits, {"x": x, "pre": pre, "hidden": hidden}


def _loss_and_grads(parameters: dict, masks: dict, x: np.ndarray,
                    y: np.ndarray):
    logits, cache = _forward(parameters, masks, x)
    probs = _softmax(logits)
    batch = x.shape[0]
    loss = -float(np.mean(np.log(probs[np.arange(batch), y] + 1e-12)))
    dlogits = probs.copy()
    dlogits[np.arange(batch), y] -= 1.0
    dlogits /= batch
    dhidden = (dlogits @ (parameters["W2"] * masks["W2"]).T) * (
        1.0 - cache["hidden"] ** 2)
    grads = {
        "W1": x.T @ dhidden * masks["W1"],
        "b1": dhidden.sum(axis=0),
        "W2": cache["hidden"].T @ dlogits * masks["W2"],
        "b2": dlogits.sum(axis=0),
    }
    return loss, grads, cache


def _keep_top(scores: np.ndarray, sparsity: float, shape) -> np.ndarray:
    flat = np.asarray(scores, dtype=float).reshape(-1)
    keep = max(1, int(round((1.0 - sparsity) * flat.size)))
    keep = min(keep, flat.size)
    order = np.argsort(-flat, kind="stable")[:keep]
    mask = np.zeros(flat.size)
    mask[order] = 1.0
    return mask.reshape(shape)


# --------------------------------------------------------------------------- #
# The pruner
# --------------------------------------------------------------------------- #
class Pruner:
    """A small MLP, its binary masks, and eight pruning procedures.

    `fit(x, y)` trains with the variant's procedure and leaves the model
    pruned at `sparsity`. `accuracy`, `n_stored` and `layer_sparsity` then
    describe the result. Only `W1` and `W2` are prunable; the biases stay.
    """

    def __init__(self, variant: str = "magnitude", *, dim: int = 16,
                 hidden: int = 32, classes: int = 3, sparsity: float = 0.8,
                 rounds: int = 4, tolerance: float = 0.05, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(dim, hidden, classes) < 1:
            raise ValueError("all sizes must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must lie in [0, 1)")
        if rounds < 1:
            raise ValueError("rounds must be positive")
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must lie in (0, 1)")
        self.variant = variant
        self.dim = dim
        self.hidden = hidden
        self.classes = classes
        self.sparsity = float(sparsity)
        self.rounds = rounds
        self.tolerance = float(tolerance)
        rng = np.random.default_rng(seed)
        self.init = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, classes)),
            "b2": np.zeros(classes),
        }
        self.parameters = {key: value.copy()
                           for key, value in self.init.items()}
        self.masks = {name: np.ones_like(self.parameters[name])
                      for name in LEARNED}
        self.history: list[float] = []
        self.scores: dict[str, np.ndarray] = {
            name: np.zeros_like(self.parameters[name]) for name in LEARNED}

    # -- properties ---------------------------------------------------------- #
    @property
    def n_dense(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @property
    def n_stored(self) -> int:
        if self.variant == "svd_prune":
            return int(sum(self.rank[name] * (self.parameters[name].shape[0]
                                              + self.parameters[name].shape[1])
                           for name in LEARNED)
                       + self.parameters["b1"].size
                       + self.parameters["b2"].size)
        return int(sum(self.masks[name].sum() for name in LEARNED)
                   + self.parameters["b1"].size
                   + self.parameters["b2"].size)

    def layer_sparsity(self, name: str) -> float:
        return float(1.0 - self.masks[name].mean())

    # -- validation ---------------------------------------------------------- #
    def _check(self, x, y):
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("x must be (batch, dim)")
        if values.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if values.shape[1] != self.dim:
            raise ValueError("x has the wrong feature count")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must be finite")
        labels = np.asarray(y)
        if labels.shape != (values.shape[0],):
            raise ValueError("y must hold one label per example")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return values, labels

    # -- training ------------------------------------------------------------ #
    def _step(self, x, y, lr: float) -> float:
        loss, grads, _ = _loss_and_grads(self.parameters, self.masks, x, y)
        for key in self.parameters:
            self.parameters[key] -= lr * grads[key]
        return loss

    def _train(self, x, y, steps: int, lr: float) -> None:
        for _ in range(steps):
            self.history.append(self._step(x, y, lr))

    def fit(self, x, y, *, steps: int = 400, lr: float = 0.3,
            interval: int = 40) -> list[float]:
        values, labels = self._check(x, y)
        if steps < 1:
            raise ValueError("steps must be positive")
        if interval < 1:
            raise ValueError("interval must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.history = []
        self.scores = {name: np.zeros_like(self.parameters[name])
                       for name in LEARNED}
        if self.variant == "gradual":
            self._fit_gradual(values, labels, steps, lr, interval)
        elif self.variant in ("magnitude", "lottery_ticket"):
            self._fit_iterative(values, labels, steps, lr,
                                rewind=self.variant == "lottery_ticket")
        elif self.variant == "movement":
            self._train(values, labels, max(1, steps - 100), lr)
            self._accumulate(values, labels, 100, lr)
            self._prune_masks()
        elif self.variant == "svd_prune":
            self._train(values, labels, steps, lr)
            self._prune_svd()
        elif self.variant == "structured":
            self._train(values, labels, steps, lr)
            self._prune_structured()
        else:
            self._train(values, labels, steps, lr)
            self._prune_masks()
        return self.history

    def _accumulate(self, x, y, steps: int, lr: float) -> None:
        for _ in range(steps):
            _, grads, _ = _loss_and_grads(self.parameters, self.masks, x, y)
            for name in LEARNED:
                self.scores[name] += np.abs(
                    self.parameters[name] * grads[name])
            self.history.append(self._step(x, y, lr))

    def _fit_gradual(self, x, y, steps: int, lr: float,
                     interval: int) -> None:
        for step in range(steps):
            fraction = (step + 1) / steps
            target = self.sparsity * (1.0 - (1.0 - fraction) ** 3)
            if step % interval == 0 or step == steps - 1:
                self._prune_masks(target)
            self.history.append(self._step(x, y, lr))

    def _fit_iterative(self, x, y, steps: int, lr: float,
                       rewind: bool) -> None:
        round_steps = max(1, steps // self.rounds)
        for round_index in range(self.rounds):
            if rewind:
                for name in LEARNED:
                    self.parameters[name] = self.init[name] * self.masks[name]
                for key in ("b1", "b2"):
                    self.parameters[key] = self.init[key].copy()
            self._train(x, y, round_steps, lr)
            fraction = (round_index + 1) / self.rounds
            target = 1.0 - (1.0 - self.sparsity) ** fraction
            self._prune_masks(target)

    # -- criteria ------------------------------------------------------------ #
    def _prune_masks(self, sparsity: float | None = None) -> None:
        target = self.sparsity if sparsity is None else sparsity
        if self.variant == "movement":
            scores = self.scores
        else:
            scores = {name: np.abs(self.parameters[name]) for name in LEARNED}
        if self.variant == "unstructured":
            flat = np.concatenate([scores[name].reshape(-1)
                                   for name in LEARNED])
            keep = _keep_top(flat, target, flat.shape)
            offset = 0
            for name in LEARNED:
                size = scores[name].size
                mask = keep[offset:offset + size].reshape(scores[name].shape)
                self.masks[name] = self.masks[name] * mask
                offset += size
        else:
            for name in LEARNED:
                mask = _keep_top(scores[name], target, scores[name].shape)
                self.masks[name] = self.masks[name] * mask

    def _prune_structured(self) -> None:
        norm_in = np.linalg.norm(self.parameters["W1"], axis=0)
        norm_out = np.linalg.norm(self.parameters["W2"], axis=1)
        score = norm_in * norm_out
        keep = max(1, int(round((1.0 - self.sparsity) * self.hidden)))
        order = np.argsort(-score, kind="stable")[:keep]
        keep_mask = np.zeros(self.hidden)
        keep_mask[order] = 1.0
        self.masks["W1"] = keep_mask[None, :] * np.ones_like(self.masks["W1"])
        self.masks["W2"] = keep_mask[:, None] * np.ones_like(self.masks["W2"])

    def _prune_svd(self) -> None:
        self.rank = {}
        self.energy = {}
        budget = int((1.0 - self.sparsity)
                     * sum(self.parameters[name].size for name in LEARNED))
        factor = {name: np.linalg.svd(self.parameters[name],
                                      full_matrices=False)
                  for name in LEARNED}
        total = sum(float(np.sum(factor[name][1] ** 2)) for name in LEARNED)
        ranks = {name: 0 for name in LEARNED}
        used = 0
        recovered = 0.0
        while recovered < (1.0 - self.tolerance) * total:
            best = None
            for name in LEARNED:
                singular = factor[name][1]
                if ranks[name] >= singular.size:
                    continue
                cost = (self.parameters[name].shape[0]
                        + self.parameters[name].shape[1])
                if used + cost > budget:
                    continue
                gain = float(singular[ranks[name]] ** 2) / cost
                if best is None or gain > best[0]:
                    best = (gain, name, cost)
            if best is None:
                break
            _, name, cost = best
            recovered += float(factor[name][1][ranks[name]] ** 2)
            ranks[name] += 1
            used += cost
        for name in LEARNED:
            rank = max(1, ranks[name])
            left, singular, right = factor[name]
            rank = min(rank, singular.size)
            self.rank[name] = rank
            self.energy[name] = float(np.sum(singular[:rank] ** 2)) / float(
                np.sum(singular ** 2))
            self.parameters[name] = (
                left[:, :rank] * singular[:rank]) @ right[:rank]

    # -- evaluation ---------------------------------------------------------- #
    def accuracy(self, x, y) -> float:
        values, labels = self._check(x, y)
        logits, _ = _forward(self.parameters, self.masks, values)
        return float(np.mean(np.argmax(logits, axis=-1) == labels))

    def loss(self, x, y) -> float:
        values, labels = self._check(x, y)
        return _loss_and_grads(self.parameters, self.masks, values, labels)[0]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _centres(seed: int, classes: int = 3, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(classes, dim)) * 1.4


def _task(seed: int, centres: np.ndarray, per_class: int = 80,
          noise: float = 0.6):
    rng = np.random.default_rng(seed)
    classes, dim = centres.shape
    labels = np.repeat(np.arange(classes), per_class)
    x = centres[labels] + rng.normal(0.0, noise, (labels.size, dim))
    order = rng.permutation(labels.size)
    return x[order], labels[order]


def demo() -> None:
    centres = _centres(1)
    train_x, train_y = _task(2, centres)
    valid_x, valid_y = _task(3, centres)
    test_x, test_y = _task(4, centres)

    dense = Pruner("one_shot", sparsity=0.0, seed=4)
    dense.fit(train_x, train_y, steps=400, lr=0.3)
    dense_score = dense.accuracy(test_x, test_y)
    print("pruning. a dense model first:")
    print(f"  {dense.n_dense} parameters, {dense_score:.3f} held-out"
          f" accuracy after 400 steps")

    print("pruning. global against layer-wise magnitude at 80%:")
    results = {}
    for variant in ("unstructured", "one_shot"):
        model = Pruner(variant, sparsity=0.8, seed=4)
        model.fit(train_x, train_y, steps=400, lr=0.3)
        results[variant] = model
        print(f"    {variant:<13s} accuracy {model.accuracy(test_x, test_y):.3f}"
              f"  W1 {model.layer_sparsity('W1'):.0%} zero"
              f"  W2 {model.layer_sparsity('W2'):.0%} zero")
    assert abs(results["one_shot"].layer_sparsity("W1") - 0.8) < 0.01
    assert results["unstructured"].layer_sparsity("W1") != 0.8
    print("  one threshold over all weights spends the budget where the")
    print("  large weights are; a per-layer rule spends it evenly")

    print("pruning. structured pruning removes whole hidden units:")
    model = Pruner("structured", sparsity=0.8, seed=4)
    model.fit(train_x, train_y, steps=400, lr=0.3)
    kept = int(round((1.0 - 0.8) * 32))
    print(f"  {kept} of 32 hidden units survive, accuracy "
          f"{model.accuracy(test_x, test_y):.3f}")
    print(f"  {model.n_stored} numbers remain against {dense.n_dense}; the")
    print("  saving is real compute, not a scatter of zeros")
    columns = np.unique(model.masks["W1"].sum(axis=0))
    assert set(columns) <= {0.0, 16.0}
    assert model.n_stored < dense.n_dense / 2

    print("pruning. movement scores the effect of removal:")
    movement = Pruner("movement", sparsity=0.8, seed=4)
    movement.fit(train_x, train_y, steps=400, lr=0.3)
    plain = Pruner("one_shot", sparsity=0.8, seed=4)
    plain.fit(train_x, train_y, steps=400, lr=0.3)
    differ = int(np.sum(movement.masks["W1"] != plain.masks["W1"]))
    print(f"  accuracy {movement.accuracy(test_x, test_y):.3f};"
          f" {differ} of 512 W1 entries are masked differently")
    print("  from a pure magnitude rule, because the score is |w times g|")
    print("  accumulated along the training path, not |w| at the end")
    assert differ > 0

    print("pruning. recovering from 98% sparsity:")
    scores = {}
    for variant in ("one_shot", "magnitude", "gradual"):
        model = Pruner(variant, sparsity=0.98, seed=7)
        model.fit(train_x, train_y, steps=800, lr=0.5, interval=40)
        scores[variant] = model.accuracy(test_x, test_y)
        print(f"    {variant:<9s} {scores[variant]:.3f} accuracy,"
              f" {model.n_stored} numbers")
    assert scores["magnitude"] > scores["one_shot"]
    assert scores["gradual"] > scores["one_shot"]

    print("pruning. rewinding to the init finds a better mask:")
    tickets = {}
    for variant in ("one_shot", "magnitude", "lottery_ticket"):
        model = Pruner(variant, sparsity=0.98, rounds=4, seed=7)
        model.fit(train_x, train_y, steps=800, lr=0.5)
        tickets[variant] = model.accuracy(test_x, test_y)
        print(f"    {variant:<14s} {tickets[variant]:.3f} accuracy,"
              f" {model.n_stored} numbers")
    assert tickets["lottery_ticket"] > tickets["magnitude"]
    assert tickets["magnitude"] > tickets["one_shot"]
    print("  each round rewinds to the same initial weights and retrains")
    print("  through the surviving mask, so the mask and the weights are")
    print("  chosen together rather than once")

    print("pruning. svd pruning keeps the leading subspace:")
    revenues = []
    for sparsity in (0.5, 0.8, 0.9):
        model = Pruner("svd_prune", sparsity=sparsity, seed=4)
        model.fit(train_x, train_y, steps=400, lr=0.3)
        ranks = model.rank
        revenues.append(model.energy["W1"])
        print(f"    {sparsity:.0%} of the entries -> W1 rank {ranks['W1']},"
              f" W2 rank {ranks['W2']}, energy {model.energy['W1']:.0%},"
              f" accuracy {model.accuracy(test_x, test_y):.3f}")
    assert revenues[0] >= revenues[-1]
    print("  rank is allocated greedily across both matrices by energy per")
    print("  stored number, so the budget buys the largest singular values")
    print("  first; at 90% the whole second layer is rank one and three")
    print("  classes need rank two, which is the price of a low-rank bias")

    print("pruning. eight criteria, one table at 80%:")
    table = {}
    for variant in VARIANTS:
        model = Pruner(variant, sparsity=0.8, rounds=3, seed=7)
        model.fit(train_x, train_y, steps=500, lr=0.4)
        table[variant] = (model.accuracy(test_x, test_y), model.n_stored)
        print(f"    {variant:<14s} {table[variant][0]:.3f} accuracy"
              f"  {table[variant][1]:4d} numbers")
    others = [score for variant, (score, _) in table.items()
              if variant != "svd_prune"]
    assert min(others) > 0.95
    assert table["svd_prune"][0] < min(others)
    assert all(stored < dense.n_dense for _, stored in table.values())

    print("pruning. malformed calls are refused:")
    guard = Pruner("magnitude", dim=8, hidden=8, classes=2, seed=8)
    good_x = np.random.default_rng(0).normal(size=(8, 8))
    good_y = train_y[:8] % 2
    guard.fit(good_x, good_y, steps=2, lr=0.1)
    bad = [
        lambda: Pruner("magnitude_v2"),
        lambda: Pruner("magnitude", dim=0),
        lambda: Pruner("magnitude", hidden=0),
        lambda: Pruner("magnitude", classes=1),
        lambda: Pruner("magnitude", sparsity=1.0),
        lambda: Pruner("magnitude", sparsity=-0.1),
        lambda: Pruner("lottery_ticket", rounds=0),
        lambda: Pruner("svd_prune", tolerance=0.0),
        lambda: guard.fit(good_x, good_y, steps=0),
        lambda: guard.fit(good_x, good_y, interval=0),
        lambda: guard.fit(good_x, good_y, lr=0.0),
        lambda: guard.fit(good_x[0], good_y, steps=2),
        lambda: guard.fit(good_x[:, :4], good_y, steps=2),
        lambda: guard.fit(np.zeros((0, 8)), good_y[:0], steps=2),
        lambda: guard.fit(np.full_like(good_x, np.nan), good_y, steps=2),
        lambda: guard.fit(good_x, good_y[:3], steps=2),
        lambda: guard.fit(good_x, good_y.astype(float), steps=2),
        lambda: guard.fit(good_x, good_y + 5, steps=2),
        lambda: guard.accuracy(good_x[:, :4], good_y),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, one")
    print("  class, a sparsity outside [0, 1), a zero round count, a zero")
    print("  tolerance, zero steps, a zero interval, a zero learning rate,")
    print("  a 1-d input, wrong features, an empty or non-finite batch,")
    print("  wrong or float or out-of-range labels, and an evaluation on")
    print("  the wrong feature count")
    print("pruning demo passed")


if __name__ == "__main__":
    demo()
