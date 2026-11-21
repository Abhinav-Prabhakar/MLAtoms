"""Neural architecture search — six ways to choose the activations.

Chapter:   13-training
Companion: nas.md
Tier 2 · 6 variants: darts, enas, amoebanet, regularized_evolution,
once_for_all, supernet
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("darts", "enas", "amoebanet", "regularized_evolution",
            "once_for_all", "supernet")
OPS = ("affine", "tanh", "relu", "sin")
GRID = tuple((a, b) for a in OPS for b in OPS)
ACTS = {"affine": lambda z: z, "tanh": np.tanh,
        "relu": lambda z: np.maximum(z, 0.0), "sin": np.sin}
DERS = {"affine": lambda z, a: np.ones_like(z),
        "tanh": lambda z, a: 1.0 - a ** 2,
        "relu": lambda z, a: (z > 0.0).astype(float),
        "sin": lambda z, a: np.cos(z)}
WIDTH = 8


# --------------------------------------------------------------------------- #
# Data and a two-cell network over four activation choices
# --------------------------------------------------------------------------- #
def data(seed: int, n: int = 240, frequency: float = 6.0,
         noise: float = 0.03):
    """A boundary that only a periodic activation represents cheaply."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, (n, 2))
    labels = (np.sin(frequency * x[:, 0]) + 0.3 * x[:, 1] > 0.0).astype(int)
    flip = rng.random(n) < noise
    labels[flip] = 1 - labels[flip]
    return x, labels


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _init(rng: np.random.Generator, width: int = WIDTH) -> dict:
    """One weight block per (layer, activation); the head is shared."""
    params = {}
    for layer in (0, 1):
        rows = 2 if layer == 0 else width
        for op in OPS:
            params[(layer, op)] = {
                "W": rng.normal(0.0, 0.7, (rows, width)),
                "b": np.zeros(width)}
    params["head"] = {"W": rng.normal(0.0, 0.7, (width, 2)),
                      "b": np.zeros(2)}
    return params


def _forward(params: dict, x: np.ndarray, genome):
    pre_one = x @ params[(0, genome[0])]["W"] + params[(0, genome[0])]["b"]
    first = ACTS[genome[0]](pre_one)
    pre_two = first @ params[(1, genome[1])]["W"] + params[(1, genome[1])]["b"]
    second = ACTS[genome[1]](pre_two)
    logits = second @ params["head"]["W"] + params["head"]["b"]
    return logits, (x, pre_one, first, pre_two, second)


def _grads(params: dict, x: np.ndarray, labels: np.ndarray, genome):
    logits, (x, pre_one, first, pre_two, second) = _forward(
        params, x, genome)
    delta = (_softmax(logits) - np.eye(2)[labels]) / len(labels)
    grads = {("head", "W"): second.T @ delta, ("head", "b"): delta.sum(0)}
    d_second = (delta @ params["head"]["W"].T) * DERS[genome[1]](
        pre_two, second)
    grads[((1, genome[1]), "W")] = first.T @ d_second
    grads[((1, genome[1]), "b")] = d_second.sum(0)
    d_first = (d_second @ params[(1, genome[1])]["W"].T) * DERS[genome[0]](
        pre_one, first)
    grads[((0, genome[0]), "W")] = x.T @ d_first
    grads[((0, genome[0]), "b")] = d_first.sum(0)
    return grads


def _optimiser_state(params: dict) -> tuple:
    keys = [(layer, name) for layer in params for name in params[layer]]
    first = {key: np.zeros_like(params[key[0]][key[1]]) for key in keys}
    second = {key: np.zeros_like(params[key[0]][key[1]]) for key in keys}
    return first, second


def _apply(params: dict, grads: dict, first: dict, second: dict, step: int,
           lr: float, decay: float) -> None:
    for key, grad in grads.items():
        layer, name = key
        value = params[layer][name]
        grad = grad + (decay * value if name == "W" else 0.0)
        first[key] = 0.9 * first[key] + 0.1 * grad
        second[key] = 0.999 * second[key] + 0.001 * grad ** 2
        value -= lr * (first[key] / (1.0 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1.0 - 0.999 ** step)) + 1e-8)


def _train(params: dict, x: np.ndarray, labels: np.ndarray, genome,
           steps: int, lr: float, decay: float) -> None:
    first, second = _optimiser_state(params)
    for step in range(1, steps + 1):
        _apply(params, _grads(params, x, labels, genome), first, second,
               step, lr, decay)


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
class Search:
    """A 16-architecture space, six search strategies, and one task.

    `run(x, y, test_x, test_y)` searches the activation pair, then retrains the
    winner from scratch so every variant is scored the same way. `genome` is
    the winner, `score` its test accuracy, `n_trained` the number of fresh
    networks fitted, and `n_steps` the number of gradient steps paid.
    """

    def __init__(self, variant: str = "darts", *, steps: int = 300,
                 lr: float = 0.02, decay: float = 1e-3,
                 samples: int = 16, population: int = 6, tournament: int = 2,
                 generations: int = 10, restarts: int = 3, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if decay < 0.0:
            raise ValueError("decay must be non-negative")
        if samples < 1 or generations < 1:
            raise ValueError("samples and generations must be positive")
        if population < 2 or tournament < 1:
            raise ValueError("population must be at least two")
        if restarts < 1:
            raise ValueError("restarts must be positive")
        self.variant = variant
        self.steps = steps
        self.lr = lr
        self.decay = decay
        self.samples = samples
        self.population = population
        self.tournament = tournament
        self.generations = generations
        self.restarts = restarts
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache: dict[tuple, float] = {}
        self.n_trained = 0
        self.n_steps = 0
        self.history: list[tuple] = []
        self.shared_estimate = float("nan")
        self.genome: tuple | None = None
        self.score = 0.0

    # -- evaluation ----------------------------------------------------------- #
    def _code(self, tag: str) -> int:
        """A stable per-tag seed, independent of Python's salted hash."""
        value = self.seed
        for index, char in enumerate(tag):
            value = (value * 131 + ord(char) * (index + 1)) % (2 ** 31)
        return int(value)

    def evaluate(self, genome, x, y, test_x, test_y) -> float:
        """Mean test accuracy over `restarts` fixed initialisations.

        The seeds are the same for every genome and every variant, so all six
        searches optimise one deterministic benchmark instead of six noisy
        ones.
        """
        if genome not in self.cache:
            scores = []
            for seed in range(self.restarts):
                params = _init(np.random.default_rng(seed))
                _train(params, x, y, genome, self.steps, self.lr, self.decay)
                self.n_trained += 1
                self.n_steps += self.steps
                probs = _softmax(_forward(params, test_x, genome)[0])
                scores.append(float(np.mean(probs.argmax(axis=1) == test_y)))
            self.cache[genome] = float(np.mean(scores))
            self.history.append((genome, self.cache[genome]))
        return self.cache[genome]

    def exhaustive(self, x, y, test_x, test_y) -> tuple:
        scores = {(a, b): self.evaluate((a, b), x, y, test_x, test_y)
                  for a in OPS for b in OPS}
        return max(scores, key=scores.get), scores

    # -- relaxation ----------------------------------------------------------- #
    def _mix_forward(self, params, x, alpha):
        weight1 = _softmax(alpha[0])
        weight2 = _softmax(alpha[1])
        pre1 = [x @ params[(0, op)]["W"] + params[(0, op)]["b"] for op in OPS]
        act1 = [ACTS[op](value) for op, value in zip(OPS, pre1, strict=True)]
        hidden = sum(weight1[i] * act1[i] for i in range(len(OPS)))
        pre2 = [hidden @ params[(1, op)]["W"] + params[(1, op)]["b"]
                for op in OPS]
        act2 = [ACTS[op](value) for op, value in zip(OPS, pre2, strict=True)]
        mixed = sum(weight2[i] * act2[i] for i in range(len(OPS)))
        logits = mixed @ params["head"]["W"] + params["head"]["b"]
        return logits, (x, weight1, pre1, act1, hidden, weight2, pre2, act2,
                        mixed)

    def _mix_grads(self, params, cache, delta):
        x, weight1, pre1, act1, hidden, weight2, pre2, act2, mixed = cache
        grads = {("head", "W"): mixed.T @ delta,
                 ("head", "b"): delta.sum(0)}
        d_mixed = delta @ params["head"]["W"].T
        d_alpha = np.zeros((2, len(OPS)))
        d_hidden = np.zeros_like(hidden)
        for index, op in enumerate(OPS):
            d_pre = weight2[index] * d_mixed * DERS[op](pre2[index],
                                                       act2[index])
            grads[((1, op), "W")] = hidden.T @ d_pre
            grads[((1, op), "b")] = d_pre.sum(0)
            d_hidden += d_pre @ params[(1, op)]["W"].T
            d_alpha[1, index] = float(np.sum(
                d_mixed * weight2[index] * (act2[index] - mixed)))
        for index, op in enumerate(OPS):
            d_pre = weight1[index] * d_hidden * DERS[op](pre1[index],
                                                        act1[index])
            grads[((0, op), "W")] = x.T @ d_pre
            grads[((0, op), "b")] = d_pre.sum(0)
            d_alpha[0, index] = float(np.sum(
                d_hidden * weight1[index] * (act1[index] - hidden)))
        return grads, d_alpha

    def _darts(self, x, y, test_x, test_y):
        rng = np.random.default_rng(self._code("darts"))
        params = _init(rng)
        alpha = np.zeros((2, len(OPS)))
        first, second = _optimiser_state(params)
        target = np.eye(2)[y]
        for step in range(1, self.steps + 1):
            logits, cache = self._mix_forward(params, x, alpha)
            delta = (_softmax(logits) - target) / len(y)
            grads, d_alpha = self._mix_grads(params, cache, delta)
            _apply(params, grads, first, second, step, self.lr, self.decay)
            alpha -= 0.05 * d_alpha
        self.n_steps += self.steps
        return (OPS[int(np.argmax(alpha[0]))], OPS[int(np.argmax(alpha[1]))])

    # -- controller ----------------------------------------------------------- #
    def _enas(self, x, y, test_x, test_y):
        theta = np.zeros((2, len(OPS)))
        baseline = 0.0
        best, best_score = None, -1.0
        for _ in range(self.samples):
            probs = _softmax(theta)
            genome = tuple(OPS[int(self.rng.choice(len(OPS), p=probs[layer]))]
                           for layer in range(2))
            score = self.evaluate(genome, x, y, test_x, test_y)
            if score > best_score:
                best, best_score = genome, score
            baseline = 0.7 * baseline + 0.3 * score
            for layer in range(2):
                grad = -probs[layer].copy()
                grad[OPS.index(genome[layer])] += 1.0
                theta[layer] += 0.5 * (score - baseline) * grad
        return best

    # -- evolution ------------------------------------------------------------ #
    def _mutate(self, genome):
        draw = self.rng.random()
        if draw < 0.4:
            return (OPS[self.rng.integers(len(OPS))], genome[1])
        if draw < 0.8:
            return (genome[0], OPS[self.rng.integers(len(OPS))])
        return (OPS[self.rng.integers(len(OPS))],
                OPS[self.rng.integers(len(OPS))])

    def _tournament(self, population, scores):
        picks = self.rng.choice(len(population), self.tournament,
                                replace=False)
        return population[max(picks, key=lambda index: scores[index])]

    def _amoebanet(self, x, y, test_x, test_y):
        initial = self.rng.choice(len(OPS), min(self.population, len(OPS)),
                                  replace=False)
        population = [OPS[int(index)] for index in initial]
        scores = [self.evaluate((cell, cell), x, y, test_x, test_y)
                  for cell in population]
        best, best_score = population[0], scores[0]
        for _ in range(self.generations):
            parent = self._tournament(population, scores)
            child = parent if self.rng.random() < 0.5 else OPS[
                self.rng.integers(len(OPS))]
            score = self.evaluate((child, child), x, y, test_x, test_y)
            population.pop(0)
            scores.pop(0)
            population.append(child)
            scores.append(score)
            if score > best_score:
                best, best_score = child, score
        return (best, best)

    def _regularized_evolution(self, x, y, test_x, test_y):
        population = [(OPS[self.rng.integers(len(OPS))],
                       OPS[self.rng.integers(len(OPS))])
                      for _ in range(self.population)]
        scores = [self.evaluate(genome, x, y, test_x, test_y)
                  for genome in population]
        best = population[int(np.argmax(scores))]
        best_score = max(scores)
        for _ in range(self.generations):
            parent = self._tournament(population, scores)
            child = self._mutate(parent)
            score = self.evaluate(child, x, y, test_x, test_y)
            worst = int(np.argmin(scores))
            population[worst] = child
            scores[worst] = score
            if score > best_score:
                best, best_score = child, score
        return best

    # -- weight sharing ------------------------------------------------------- #
    def _supernet(self, x, y, test_x, test_y):
        """One shared pool, paths sampled in shuffled rounds."""
        rng = np.random.default_rng(self._code("supernet"))
        params = _init(rng)
        first, second = _optimiser_state(params)
        rounds = 4
        order = list(range(len(GRID)))
        for step in range(1, rounds * self.steps + 1):
            slot = (step - 1) % len(GRID)
            if slot == 0:
                order = [int(index) for index in self.rng.permutation(
                    len(GRID))]
            genome = GRID[order[slot]]
            _apply(params, _grads(params, x, y, genome), first, second,
                   step, self.lr, self.decay)
        self.n_steps += rounds * self.steps
        shared = {}
        for genome in GRID:
            probs = _softmax(_forward(params, test_x, genome)[0])
            shared[genome] = float(np.mean(probs.argmax(axis=1) == test_y))
        winner = max(shared, key=shared.get)
        self.shared_estimate = shared[winner]
        return winner

    def _once_for_all(self, x, y, test_x, test_y):
        """One wide network per second op; narrower widths are slices of it."""
        best, best_score = None, -1.0
        for op in OPS:
            rng = np.random.default_rng(self._code("ofa" + op))
            params = _init(rng)
            _train(params, x, y, ("tanh", op), self.steps, self.lr,
                   self.decay)
            self.n_trained += 1
            self.n_steps += self.steps
            for width in (4, 6, 8):
                sliced = {
                    (0, "tanh"): {
                        "W": params[(0, "tanh")]["W"][:, :width],
                        "b": params[(0, "tanh")]["b"][:width]},
                    (1, op): {
                        "W": params[(1, op)]["W"][:width, :width],
                        "b": params[(1, op)]["b"][:width]},
                    "head": {"W": params["head"]["W"][:width],
                             "b": params["head"]["b"]}}
                probs = _softmax(_forward(sliced, test_x, ("tanh", op))[0])
                score = float(np.mean(probs.argmax(axis=1) == test_y))
                if score > best_score:
                    best, best_score = ("tanh", op), score
        return best

    # -- entry point ---------------------------------------------------------- #
    def run(self, x, y, test_x, test_y) -> Search:
        values = np.asarray(x, dtype=float)
        labels = np.asarray(y)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("this demo lives in two dimensions")
        if labels.shape != (len(values),):
            raise ValueError("labels must be one per row")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integers")
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError("labels must be binary")
        if np.bincount(labels, minlength=2).min() < 2:
            raise ValueError("both classes need at least two points")
        if not np.all(np.isfinite(values)):
            raise ValueError("the data must be finite")
        if self.variant == "darts":
            chosen = self._darts(values, labels, test_x, test_y)
        elif self.variant == "enas":
            chosen = self._enas(values, labels, test_x, test_y)
        elif self.variant == "amoebanet":
            chosen = self._amoebanet(values, labels, test_x, test_y)
        elif self.variant == "regularized_evolution":
            chosen = self._regularized_evolution(
                values, labels, test_x, test_y)
        elif self.variant == "supernet":
            chosen = self._supernet(values, labels, test_x, test_y)
        else:
            chosen = self._once_for_all(values, labels, test_x, test_y)
        self.genome = chosen
        self.score = self.evaluate(chosen, values, labels, test_x, test_y)
        return self


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    train_x, train_y = data(seed=1)
    test_x, test_y = data(seed=2)
    print("nas. one boundary, two cells, four activations each:")
    print("  the target is sin(6x) + 0.3y > 0, a periodic frontier")
    print("  the space has 4 x 4 = 16 architectures, so it is searchable")

    probe = Search("darts", seed=99)
    truth, table = probe.exhaustive(train_x, train_y, test_x, test_y)
    affine = table[("affine", "affine")]
    print(f"  exhaustive search finds {truth[0]}/{truth[1]} at"
          f" {table[truth]:.3f}")
    print(f"  the all-affine network scores {affine:.3f}, so the signal is"
          " real")
    print(f"  the exhaustive pass trains {probe.n_trained} networks"
          f" for {probe.n_steps} steps")

    print("nas. six searches, one budget:")
    results = {}
    for variant in VARIANTS:
        search = Search(variant, seed=3).run(
            train_x, train_y, test_x, test_y)
        results[variant] = search
        detail = ""
        if variant == "supernet":
            detail = f"  shared {search.shared_estimate:.3f}"
        plural = "" if search.n_trained == 1 else "s"
        print(f"    {variant:<21s} {search.genome[0]}/{search.genome[1]}"
              f"  {search.score:.3f}  {search.n_trained:2d} net{plural}"
              f" {search.n_steps:5d} steps{detail}")

    print("nas. what each method saw:")
    for variant in VARIANTS:
        search = results[variant]
        seen = {genome for genome, _ in search.history}
        plural = "" if len(seen) == 1 else "s"
        print(f"    {variant:<21s} {len(seen):2d} distinct architecture"
              f"{plural}")

    print("nas. the winner and the cost:")
    winner = max(results, key=lambda name: results[name].score)
    print(f"  {winner} reaches {results[winner].score:.3f} for"
          f" {results[winner].n_steps} gradient steps")
    print(f"  exhaustive search needs {probe.n_steps} for"
          f" {table[truth]:.3f}")
    print("  the shared supernet estimate is the number to distrust:")
    shared = results["supernet"]
    print(f"    its own ranking says {shared.shared_estimate:.3f}; the"
          f" benchmark says {shared.score:.3f}")
    assert all(search.score > affine + 0.1 for search in results.values())
    assert results["amoebanet"].score > affine + 0.25
    assert results["regularized_evolution"].score > affine + 0.25
    assert results["supernet"].score > affine + 0.25
    assert results["supernet"].shared_estimate < shared.score
    assert table[truth] > 0.85

    print("nas. malformed calls are refused:")
    bad = [
        lambda: Search("darts_v2"),
        lambda: Search("darts", steps=0),
        lambda: Search("darts", lr=0.0),
        lambda: Search("darts", decay=-1.0),
        lambda: Search("darts", samples=0),
        lambda: Search("darts", generations=0),
        lambda: Search("darts", population=1),
        lambda: Search("darts", tournament=0),
        lambda: Search("darts", restarts=0),
        lambda: Search("darts").run(train_x[:, :1], train_y, test_x, test_y),
        lambda: Search("darts").run(train_x, train_y[:-1], test_x, test_y),
        lambda: Search("darts").run(train_x, train_y.astype(float),
                                    test_x, test_y),
        lambda: Search("darts").run(train_x, train_y + 2, test_x, test_y),
        lambda: Search("darts").run(train_x,
                                    np.zeros(len(train_y), dtype=int),
                                    test_x, test_y),
        lambda: Search("darts").run(np.full_like(train_x, np.nan), train_y,
                                    test_x, test_y),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero step")
    print("  count, a non-positive learning rate, a negative decay, zero")
    print("  samples or generations, a population below two, a zero")
    print("  tournament, zero restarts, a one-dimensional input, a label")
    print("  mismatch, float labels, non-binary labels, a single-class")
    print("  set, and a non-finite input")
    print("nas demo passed")


if __name__ == "__main__":
    demo()
