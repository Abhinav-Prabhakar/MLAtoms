"""Hyperparameter search — eight ways to spend a tuning budget.

Chapter:   13-training
Companion: hyperparameter_search.md
Tier 2 · 8 variants: grid, random, tpe, gp_bo, bohb, hyperband, asha, pruner
"""

from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np

VARIANTS = ("grid", "random", "tpe", "gp_bo", "bohb", "hyperband", "asha",
            "pruner")
SINGLE = ("grid", "random", "tpe", "gp_bo")
MULTI = ("hyperband", "asha", "bohb", "pruner")


# --------------------------------------------------------------------------- #
# The landscape
# --------------------------------------------------------------------------- #
def _curve(delta: np.ndarray, epochs: float) -> float:
    """A synthetic training curve: transient plus a quadratic bowl."""
    curvature = np.array([40.0, 15.0, 8.0])
    final = 0.2 + float(np.sum(curvature * delta ** 2)) + 1.5 * (
        delta[0] * delta[1])
    amplitude = 0.7 + 1.6 * float(delta[0] + 0.35)
    return final + amplitude * float(np.exp(-epochs / 6.0))


def _noise(point: np.ndarray, epochs: float) -> float:
    phase = (12.9898 * point[0] + 78.233 * point[1] + 37.719 * point[2]
             + 1.7 * epochs)
    return 0.01 * float(np.sin(phase))


def _logsumexp(values: np.ndarray) -> np.ndarray:
    peak = values.max(axis=1, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=1, keepdims=True)))[
        :, 0]


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _matern(left: np.ndarray, right: np.ndarray,
            length: float = 0.22) -> np.ndarray:
    """Matern kernel with the usual 5/2 closed form, no scipy."""
    gap = left[:, None, :] - right[None, :, :]
    distance = np.sqrt(np.sum(gap ** 2, axis=-1)) / length
    root = sqrt(5.0) * distance
    return (1.0 + root + 5.0 / 3.0 * distance ** 2) * np.exp(-root)


# --------------------------------------------------------------------------- #
# The tuner
# --------------------------------------------------------------------------- #
class Tuner:
    """A black-box objective, eight search strategies, and one budget.

    `fit()` runs the chosen strategy on the unit cube and leaves `history` as
    a list of `(point, epochs, loss)` tuples. `best` is the best point seen at
    full fidelity, and `best_loss` its loss. The objective is deliberately a
    training curve, so fidelity is meaningful: a point can be cheap and bad.
    """

    def __init__(self, variant: str = "random", *, dim: int = 3,
                 trials: int = 40, eta: int = 3, max_epochs: int = 27,
                 budget: int = 1080, gamma: float = 0.25, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if dim < 1:
            raise ValueError("dim must be positive")
        if trials < 4:
            raise ValueError("trials must be at least four")
        if eta < 2:
            raise ValueError("eta must be at least two")
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if budget < 1:
            raise ValueError("budget must be positive")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        self.variant = variant
        self.dim = dim
        self.trials = trials
        self.eta = eta
        self.max_epochs = max_epochs
        self.budget = budget
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.centre = np.array([0.35, 0.65, 0.25])[:dim]
        self.history: list[tuple[np.ndarray, int, float]] = []
        self.epochs_used = 0
        self.n_start = max(5, trials // 4)
        self._cache: dict[tuple, float] = {}

    # -- objective ----------------------------------------------------------- #
    def evaluate(self, point, epochs: int) -> float:
        values = np.asarray(point, dtype=float)
        if values.shape != (self.dim,):
            raise ValueError("point has the wrong dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("point must be finite")
        if values.min() < 0.0 or values.max() > 1.0:
            raise ValueError("point must lie in the unit cube")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        key = (tuple(np.round(values, 12)), int(epochs))
        if key not in self._cache:
            delta = values - self.centre
            self._cache[key] = _curve(delta, epochs) + _noise(values, epochs)
        loss = self._cache[key]
        self.history.append((values.copy(), int(epochs), loss))
        self.epochs_used += int(epochs)
        return loss

    # -- single fidelity ----------------------------------------------------- #
    def _grid(self) -> None:
        side = max(2, int(round(self.trials ** (1.0 / self.dim))))
        axis = np.linspace(0.0, 1.0, side)
        mesh = np.meshgrid(*([axis] * self.dim), indexing="ij")
        points = np.stack([grid.reshape(-1) for grid in mesh], axis=1)
        for point in points:
            self.evaluate(point, self.max_epochs)

    def _random(self) -> None:
        for _ in range(self.trials):
            self.evaluate(self.rng.random(self.dim), self.max_epochs)

    def _kde_logpdf(self, candidates: np.ndarray,
                    sample: np.ndarray) -> np.ndarray:
        if len(sample) < 2:
            return np.zeros(len(candidates))
        bandwidth = np.maximum(sample.std(axis=0), 0.05) * len(
            sample) ** (-0.2)
        gap = (candidates[:, None, :] - sample[None, :, :]) / bandwidth
        logp = -0.5 * gap ** 2 - np.log(bandwidth) - 0.5 * np.log(2.0 * pi)
        logp = _logsumexp(logp.reshape(len(candidates), -1)) - np.log(
            len(sample))
        return np.logaddexp(np.log(1.0 - self.gamma) + logp,
                            np.log(self.gamma))

    def _sample_kde(self, sample: np.ndarray) -> np.ndarray:
        bandwidth = np.maximum(sample.std(axis=0), 0.08) * len(
            sample) ** (-0.2)
        base = sample[self.rng.integers(len(sample))]
        return np.clip(base + self.rng.normal(size=self.dim) * bandwidth,
                       0.0, 1.0)

    def _tpe_point(self) -> np.ndarray:
        if len(self.history) < self.n_start:
            return self.rng.random(self.dim)
        losses = np.array([entry[2] for entry in self.history])
        cut = float(np.quantile(losses, self.gamma))
        points = np.array([entry[0] for entry in self.history])
        good = points[losses <= cut]
        bad = points[losses > cut]
        proposal = self._sample_kde(good)
        good_log = float(self._kde_logpdf(proposal[None], good)[0])
        bad_log = float(self._kde_logpdf(proposal[None], bad)[0])
        accept = float(np.exp(good_log - np.logaddexp(good_log, bad_log)))
        return proposal if self.rng.random() < accept else self._sample_kde(bad)

    def _tpe(self) -> None:
        for _ in range(self.trials):
            self.evaluate(self._tpe_point(), self.max_epochs)

    def _gp_point(self) -> np.ndarray:
        if len(self.history) < self.n_start:
            return self.rng.random(self.dim)
        points = np.array([entry[0] for entry in self.history])
        losses = np.array([entry[2] for entry in self.history])
        mean = float(losses.mean())
        spread = float(losses.std()) + 1e-9
        target = (losses - mean) / spread
        kernel = _matern(points, points) + 1e-6 * np.eye(len(points))
        cholesky = np.linalg.cholesky(kernel)
        weights = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, target))
        candidates = self.rng.random((512, self.dim))
        cross = _matern(points, candidates)
        predicted = cross.T @ weights
        projected = np.linalg.solve(cholesky, cross)
        variance = np.maximum(1.0 - np.sum(projected ** 2, axis=0), 1e-9)
        deviation = np.sqrt(variance)
        best = float(target.min())
        z = (best - predicted) / deviation
        improvement = ((best - predicted) * _norm_cdf(z)
                       + deviation * np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi))
        return candidates[int(np.argmax(improvement))]

    def _gp_bo(self) -> None:
        for _ in range(self.trials):
            self.evaluate(self._gp_point(), self.max_epochs)

    # -- multi fidelity ------------------------------------------------------ #
    def _halving(self, propose) -> None:
        """Hyperband's brackets, with pluggable proposal."""
        levels = int(round(np.log(self.max_epochs) / np.log(self.eta)))
        scale = max(1, self.budget // ((levels + 1) ** 2 * self.max_epochs))
        for start in range(levels, -1, -1):
            count = scale * int(np.ceil(
                (levels + 1) / (start + 1) * self.eta ** start))
            epochs = self.max_epochs / self.eta ** start
            points = [propose() for _ in range(count)]
            for _ in range(start + 1):
                results = [self.evaluate(point, int(round(epochs)))
                           for point in points]
                order = np.argsort(results, kind="stable")
                keep = max(1, count // self.eta)
                points = [points[index] for index in order[:keep]]
                count = keep
                epochs *= self.eta

    def _hyperband(self) -> None:
        self._halving(lambda: self.rng.random(self.dim))

    def _bohb(self) -> None:
        self._halving(self._tpe_point)

    def _asha(self) -> None:
        count = max(4, self.budget // 4)
        points = [self.rng.random(self.dim) for _ in range(count)]
        rungs = [1, self.eta, self.eta ** 2, self.max_epochs]
        for rung in rungs:
            results = [self.evaluate(point, rung) for point in points]
            order = np.argsort(results, kind="stable")
            points = [points[index] for index in order[:max(1, count
                                                            // self.eta)]]
            count = len(points)

    def _pruner(self) -> None:
        count = max(4, self.budget // 8)
        points = [self.rng.random(self.dim) for _ in range(count)]
        for rung in (1, self.eta, self.eta ** 2, self.max_epochs):
            results = [self.evaluate(point, rung) for point in points]
            median = float(np.median(results))
            points = [point for point, loss in zip(points, results,
                                                   strict=True)
                      if loss <= median]

    # -- entry point --------------------------------------------------------- #
    def fit(self) -> Tuner:
        if self.variant == "grid":
            self._grid()
        elif self.variant == "random":
            self._random()
        elif self.variant == "tpe":
            self._tpe()
        elif self.variant == "gp_bo":
            self._gp_bo()
        elif self.variant == "hyperband":
            self._hyperband()
        elif self.variant == "bohb":
            self._bohb()
        elif self.variant == "asha":
            self._asha()
        else:
            self._pruner()
        return self

    # -- results ------------------------------------------------------------- #
    @property
    def best(self) -> np.ndarray:
        full = [entry for entry in self.history
                if entry[1] == self.max_epochs]
        pool = full if full else self.history
        return min(pool, key=lambda entry: entry[2])[0]

    @property
    def best_loss(self) -> float:
        full = [entry for entry in self.history
                if entry[1] == self.max_epochs]
        pool = full if full else self.history
        return min(entry[2] for entry in pool)

    @property
    def n_evaluations(self) -> int:
        return len(self.history)

    @property
    def best_config(self) -> dict:
        names = ("learning_rate", "weight_decay", "dropout")
        point = self.best
        return {name: float(value) for name, value in zip(
            names[:self.dim], point, strict=True)}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    print("search. one landscape, three knobs:")
    probe = Tuner("random", seed=0)
    centre = probe.centre
    print(f"  the optimum sits at {np.round(centre, 2)} on the unit cube")
    print(f"  and its loss is {probe.evaluate(centre, 1):.3f} after one")
    print(f"  epoch, {probe.evaluate(centre, probe.max_epochs):.3f} at 27")
    print("  every method below sees only the loss it asks for")

    print("search. grid against random at the same budget:")
    results = {}
    for variant in ("grid", "random"):
        tuner = Tuner(variant, seed=1, trials=27).fit()
        results[variant] = tuner
        print(f"    {variant:<7s} best {tuner.best_loss:.3f}"
              f"  {tuner.n_evaluations:3d} evaluations"
              f"  {tuner.epochs_used:5d} epochs")
    assert results["grid"].n_evaluations == 27
    assert results["random"].n_evaluations == 27
    assert results["random"].best_loss < results["grid"].best_loss
    print("  the grid spends its samples on a fixed lattice; random spends")
    print("  them on distinct points, which matters more in three dimensions")

    print("search. model-based proposals:")
    baseline = Tuner("random", seed=4, trials=60).fit()
    results["random"] = baseline
    for variant in ("tpe", "gp_bo"):
        tuner = Tuner(variant, seed=4, trials=60).fit()
        results[variant] = tuner
        print(f"    {variant:<7s} best {tuner.best_loss:.3f}"
              f"  after {tuner.n_evaluations} full-fidelity trials")
        assert tuner.best_loss < baseline.best_loss
    print("  TPE samples the good density and accepts it against the bad")
    print("  one; the GP fits a posterior and maximises expected")
    print("  improvement. Both beat random with the same number of trials")

    print("search. successive halving spends epochs, not trials:")
    board = {}
    for variant in MULTI:
        tuner = Tuner(variant, seed=4).fit()
        board[variant] = tuner
        print(f"    {variant:<10s} best {tuner.best_loss:.3f}"
              f"  {tuner.n_evaluations:3d} evaluations"
              f"  {tuner.epochs_used:5d} epochs")
        assert tuner.epochs_used <= tuner.budget * 1.1
        assert tuner.best_loss < baseline.best_loss
    assert board["bohb"].best_loss < board["hyperband"].best_loss
    print("  a bracket evaluates many points for one epoch, keeps the best")
    print("  third, and repeats at the next fidelity; BOHB replaces the")
    print("  random proposals with TPE, and the median pruner keeps only")
    print("  the configurations that beat the running median")

    print("search. eight methods, one table:")
    table = {}
    for variant in VARIANTS:
        tuner = Tuner(variant, seed=6, trials=60).fit()
        table[variant] = tuner
        print(f"    {variant:<10s} {tuner.best_loss:.3f} loss"
              f"  {tuner.n_evaluations:3d} evaluations"
              f"  {tuner.epochs_used:5d} epochs")
    winner = min(table, key=lambda name: table[name].best_loss)
    print(f"  {winner} finds the best point at {table[winner].best_loss:.3f}")
    assert winner in ("tpe", "gp_bo")
    assert table["tpe"].best_loss < 0.35
    assert table["gp_bo"].best_loss < 0.35
    assert table["grid"].best_loss < 0.35
    for variant in MULTI:
        assert table[variant].best_loss < table["random"].best_loss
    assert table["random"].best_loss > 0.8

    print("search. malformed calls are refused:")
    guard = Tuner("random", seed=5)
    bad = [
        lambda: Tuner("random_v2"),
        lambda: Tuner("random", dim=0),
        lambda: Tuner("random", trials=2),
        lambda: Tuner("random", eta=1),
        lambda: Tuner("random", max_epochs=0),
        lambda: Tuner("random", budget=0),
        lambda: Tuner("random", gamma=0.0),
        lambda: guard.evaluate([0.5, 0.5], 1),
        lambda: guard.evaluate([1.5, 0.5, 0.5], 1),
        lambda: guard.evaluate([np.nan, 0.5, 0.5], 1),
        lambda: guard.evaluate([0.5, 0.5, 0.5], 0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero dimension,")
    print("  fewer than four trials, a halving ratio below two, a zero")
    print("  fidelity or budget, a gamma outside (0, 1), a point of the")
    print("  wrong dimension, a point outside the unit cube, a non-finite")
    print("  point, and a non-positive epoch count")
    print("search demo passed")


if __name__ == "__main__":
    demo()
