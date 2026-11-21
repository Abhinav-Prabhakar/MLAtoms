"""Active learning — eight ways to choose the next label to buy.

Chapter:   13-training
Companion: active_learning.md
Tier 2 · 8 variants: uncertainty, margin, entropy, diversity, qbc, bald,
core_set, expected_model_change
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("uncertainty", "margin", "entropy", "diversity", "qbc", "bald",
            "core_set", "expected_model_change")
ENSEMBLE = ("qbc", "bald")


# --------------------------------------------------------------------------- #
# The model: multinomial logistic regression on Gaussian bumps
# --------------------------------------------------------------------------- #
_AXIS = np.linspace(-2.2, 2.2, 9)
_CENTRES = np.stack(np.meshgrid(_AXIS, _AXIS, indexing="ij"), axis=-1).reshape(
    -1, 2)
_WIDTH = 0.42


def _features(x: np.ndarray) -> np.ndarray:
    """RBF features on a fixed 9x9 grid, plus an intercept."""
    values = np.asarray(x, dtype=float)
    gap = ((values[:, None, :] - _CENTRES[None, :, :]) ** 2).sum(axis=-1)
    return np.column_stack([np.exp(-gap / (2.0 * _WIDTH ** 2)),
                            np.ones(len(values))])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _fit(features: np.ndarray, labels: np.ndarray, classes: int,
         steps: int = 500, lr: float = 0.5,
         penalty: float = 1e-3) -> np.ndarray:
    """Full-batch gradient descent from zero, so every fit is repeatable."""
    weights = np.zeros((features.shape[1], classes))
    target = np.zeros((len(labels), classes))
    target[np.arange(len(labels)), labels] = 1.0
    for _ in range(steps):
        probs = _softmax(features @ weights)
        gradient = features.T @ (probs - target) / len(labels)
        gradient += penalty * weights
        weights -= lr * gradient
    return weights


def _predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _softmax(features @ weights)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class ActiveLearner:
    """A pool, a budget, and one acquisition function.

    `run(pool_x, pool_y, test_x, test_y)` labels a stratified seed set, then
    alternates between fitting the model and buying `batch` labels, and returns
    the learner itself. `history` is the test accuracy after each fit, so the
    last entry is the accuracy at the full budget; `queried` is the order in
    which labels were bought.
    """

    def __init__(self, variant: str = "uncertainty", *, initial: int = 8,
                 batch: int = 4, rounds: int = 10, classes: int = 4,
                 ensemble: int = 7, steps: int = 500, lr: float = 0.5,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if initial < 1 or batch < 1 or rounds < 1:
            raise ValueError("initial, batch and rounds must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if ensemble < 2:
            raise ValueError("the ensemble must have at least two members")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.initial = initial
        self.batch = batch
        self.rounds = rounds
        self.classes = classes
        self.ensemble = ensemble
        self.steps = steps
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []
        self.queried: list[int] = []
        self.models: list[np.ndarray] = []
        self.coverage: list[float] = []
        self.spread: list[float] = []

    # -- validation ---------------------------------------------------------- #
    def _check(self, pool_x, pool_y, test_x, test_y):
        pool = np.asarray(pool_x, dtype=float)
        test = np.asarray(test_x, dtype=float)
        if pool.ndim != 2 or test.ndim != 2:
            raise ValueError("both splits must be matrices")
        if pool.shape[1] != 2 or test.shape[1] != 2:
            raise ValueError("this demo lives in two dimensions")
        if len(pool) < self.initial + self.batch * self.rounds:
            raise ValueError("the pool is smaller than the budget")
        if len(test) < 1:
            raise ValueError("the test set must be non-empty")
        if not np.all(np.isfinite(pool)) or not np.all(np.isfinite(test)):
            raise ValueError("the data must be finite")
        pool_labels = np.asarray(pool_y)
        test_labels = np.asarray(test_y)
        for name, labels, size in (("pool", pool_labels, len(pool)),
                                   ("test", test_labels, len(test))):
            if labels.shape != (size,):
                raise ValueError(f"{name} labels must be one per row")
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"{name} labels must be integers")
            if labels.min() < 0 or labels.max() >= self.classes:
                raise ValueError(f"{name} labels are out of range")
        return pool, pool_labels, test, test_labels

    # -- model --------------------------------------------------------------- #
    def _fit_models(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.models = []
        if self.variant in ENSEMBLE:
            for member in range(self.ensemble):
                member_rng = np.random.default_rng(
                    int(self.rng.integers(0, 2 ** 31 - 1)) + member)
                index = member_rng.integers(0, len(labels), len(labels))
                self.models.append(_fit(features[index], labels[index],
                                        self.classes, self.steps, self.lr))
        else:
            self.models = [_fit(features, labels, self.classes, self.steps,
                                self.lr)]

    def _probs(self, features: np.ndarray) -> np.ndarray:
        return np.mean([_predict(features, weights)
                        for weights in self.models], axis=0)

    # -- acquisition --------------------------------------------------------- #
    def _scores(self, pool: np.ndarray, chosen: np.ndarray) -> np.ndarray:
        features = _features(pool)
        probs = self._probs(features)
        if self.variant == "uncertainty":
            return 1.0 - probs.max(axis=1)
        if self.variant == "margin":
            ordered = np.sort(probs, axis=1)
            return -(ordered[:, -1] - ordered[:, -2])
        if self.variant == "entropy":
            return _entropy(probs)
        if self.variant == "qbc":
            votes = np.stack([_predict(features, weights).argmax(axis=1)
                              for weights in self.models])
            shares = np.stack([(votes == label).mean(axis=0)
                               for label in range(self.classes)], axis=1)
            return _entropy(shares)
        if self.variant == "bald":
            members = np.array([_predict(features, weights)
                                for weights in self.models])
            return (_entropy(probs)
                    - np.mean([_entropy(member) for member in members],
                              axis=0))
        if self.variant == "expected_model_change":
            expected = np.zeros(len(pool))
            for label in range(self.classes):
                residual = probs.copy()
                residual[:, label] -= 1.0
                expected += probs[:, label] * np.sum(
                    residual ** 2, axis=1)
            return np.sqrt(expected * np.sum(features ** 2, axis=1))
        if self.variant in ("diversity", "core_set"):
            if len(chosen):
                gap = pool[:, None, :] - pool[None, chosen, :]
                return np.min(np.sqrt(np.sum(gap ** 2, axis=-1)), axis=1)
            return np.ones(len(pool))
        raise ValueError(f"unknown variant {self.variant!r}")

    def _query(self, pool: np.ndarray, available: np.ndarray,
               chosen: np.ndarray) -> np.ndarray:
        if chosen.size:
            gap = pool[:, None, :] - pool[None, chosen, :]
            nearest = np.min(np.sqrt(np.sum(gap ** 2, axis=-1)), axis=1)
        else:
            nearest = np.max(pool.max(axis=0) - pool.min(axis=0)) * np.ones(
                len(pool))
        if self.variant == "core_set":
            picked = []
            scores = nearest.copy()
            for _ in range(min(self.batch, len(available))):
                candidates = np.where(available)[0]
                index = candidates[int(np.argmax(scores[candidates]))]
                picked.append(index)
                available = available.copy()
                available[index] = False
                gap = pool - pool[index]
                scores = np.minimum(scores,
                                    np.sqrt(np.sum(gap ** 2, axis=1)))
            return np.array(picked)
        if self.variant == "diversity":
            candidates = np.where(available)[0]
            width = min(self.batch, len(candidates))
            centres = candidates[self.rng.choice(len(candidates), width,
                                                 replace=False)]
            for _ in range(20):
                gap = pool[candidates, None, :] - pool[centres][None, :, :]
                near = np.argmin(np.sum(gap ** 2, axis=-1), axis=1)
                for cluster in range(width):
                    members = candidates[near == cluster]
                    if len(members):
                        centroid = pool[members].mean(axis=0)
                        centres[cluster] = members[int(np.argmin(
                            np.sum((pool[members] - centroid) ** 2, axis=1)))]
            picked = list(np.unique(centres))
            while len(picked) < width:
                gap = pool[candidates, None, :] - pool[picked][None, :, :]
                nearest = np.min(np.sqrt(np.sum(gap ** 2, axis=-1)), axis=1)
                nearest[np.isin(candidates, picked)] = -1.0
                picked.append(int(candidates[int(np.argmax(nearest))]))
            return np.array(picked)
        scores = self._scores(pool, chosen)
        scores[~available] = -np.inf
        return np.argsort(-scores, kind="stable")[:self.batch]

    # -- loop ---------------------------------------------------------------- #
    def run(self, pool_x, pool_y, test_x, test_y) -> ActiveLearner:
        pool, pool_labels, test, test_labels = self._check(
            pool_x, pool_y, test_x, test_y)
        self.history = []
        self.queried = []
        self.spread = []
        available = np.ones(len(pool), dtype=bool)
        labelled: list[int] = []
        for label in range(self.classes):
            members = np.where(pool_labels == label)[0]
            take = max(1, self.initial // self.classes)
            picked = self.rng.choice(members, min(take, len(members)),
                                     replace=False)
            for index in picked:
                labelled.append(int(index))
                available[index] = False
        for _ in range(self.rounds):
            index = np.array(labelled)
            self._fit_models(_features(pool[index]), pool_labels[index])
            probs = self._probs(_features(test))
            self.history.append(float(np.mean(
                np.argmax(probs, axis=1) == test_labels)))
            chosen = np.array(labelled)
            query = self._query(pool, available, chosen)
            gap = pool[query][:, None, :] - pool[query][None, :, :]
            upper = np.triu_indices(len(query), 1)
            self.spread.append(float(np.mean(
                np.sqrt(np.sum(gap ** 2, axis=-1))[upper])))
            self.queried.extend(int(index) for index in query)
            available[query] = False
            labelled.extend(int(index) for index in query)
        index = np.array(labelled)
        self._fit_models(_features(pool[index]), pool_labels[index])
        probs = self._probs(_features(test))
        self.history.append(float(np.mean(
            np.argmax(probs, axis=1) == test_labels)))
        selected = np.array(self.queried)
        gap = pool[:, None, :] - pool[None, selected, :]
        self.coverage.append(float(np.mean(
            np.min(np.sqrt(np.sum(gap ** 2, axis=-1)), axis=1))))
        return self


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _quadrants(seed: int, n: int = 600):
    """Uniform points on a square, labelled by quadrant: a four-way XOR."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, (n, 2))
    labels = (x[:, 0] > 0.0).astype(int) * 2 + (x[:, 1] > 0.0).astype(int)
    return x, labels


def _random_run(pool_x, pool_y, test_x, test_y, *, initial=8, batch=4,
                rounds=10, classes=4, seed=99):
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool_x), dtype=bool)
    labelled = []
    for label in range(classes):
        members = np.where(pool_y == label)[0]
        picked = rng.choice(members, max(1, initial // classes),
                            replace=False)
        labelled.extend(int(index) for index in picked)
        available[picked] = False
    history = []
    for _ in range(rounds + 1):
        index = np.array(labelled)
        weights = _fit(_features(pool_x[index]), pool_y[index], classes)
        probs = _predict(_features(test_x), weights)
        history.append(float(np.mean(np.argmax(probs, axis=1) == test_y)))
        if len(history) <= rounds:
            query = rng.choice(np.where(available)[0], batch, replace=False)
            available[query] = False
            labelled.extend(int(value) for value in query)
    return history


def demo() -> None:
    pool_x, pool_y = _quadrants(seed=1)
    test_x, test_y = _quadrants(seed=2)
    random_history = _random_run(pool_x, pool_y, test_x, test_y)
    budget = 8 + 4 * 10

    print(f"active. a pool of six hundred points, a budget of {budget}:")
    print(f"  random sampling ends at {random_history[-1]:.3f} test accuracy")
    print("  after ten rounds; every method below buys the same number of")
    print("  labels, so only the choice of which labels differs")

    print("active. uncertainty, margin and entropy are one idea:")
    scores = {}
    for variant in ("uncertainty", "margin", "entropy"):
        learner = ActiveLearner(variant, seed=3).run(
            pool_x, pool_y, test_x, test_y)
        scores[variant] = learner.history[-1]
        print(f"    {variant:<12s} {scores[variant]:.3f} at"
              f" {learner.initial + learner.batch * learner.rounds} labels")
        assert scores[variant] > random_history[-1]
    print("  all three buy points where the model is least sure, so they")
    print("  concentrate on the two axes where the boundaries live")

    print("active. disagreement needs more than one model:")
    for variant in ("qbc", "bald"):
        learner = ActiveLearner(variant, seed=3).run(
            pool_x, pool_y, test_x, test_y)
        scores[variant] = learner.history[-1]
        print(f"    {variant:<12s} {scores[variant]:.3f} from"
              f" {learner.ensemble} bootstrap models")
        assert scores[variant] > random_history[-1]
    assert scores["bald"] > scores["qbc"] - 0.08
    print("  QBC asks where the committee's votes disagree; BALD subtracts")
    print("  the average member entropy, keeping only the disagreement")
    print("  left over. At this budget the subtraction is too eager")

    print("active. BALD pays for itself with a longer budget:")
    longer = {}
    for variant in ("uncertainty", "qbc", "bald"):
        longer[variant] = ActiveLearner(variant, seed=3, rounds=30).run(
            pool_x, pool_y, test_x, test_y).history[-1]
        print(f"    {variant:<12s} {longer[variant]:.3f} at 128 labels")
    assert longer["bald"] > scores["bald"] + 0.04
    print("  with 128 labels the ensembles have seen enough bootstrap")
    print("  resamples for disagreement to mean something")

    print("active. coverage is a different objective:")
    coverage = {}
    for variant in ("uncertainty", "diversity", "core_set"):
        learner = ActiveLearner(variant, seed=3).run(
            pool_x, pool_y, test_x, test_y)
        coverage[variant] = (learner.history[-1], learner.spread)
        print(f"    {variant:<12s} {coverage[variant][0]:.3f} accuracy,"
              f" mean distance within a queried batch"
              f" {np.mean(coverage[variant][1]):.2f}")
    assert coverage["diversity"][1] > coverage["uncertainty"][1]
    assert coverage["core_set"][1] > coverage["uncertainty"][1]
    for variant in ("diversity", "core_set"):
        assert coverage[variant][0] > coverage["uncertainty"][0] - 0.05
    print("  uncertainty's four queries sit on top of each other against the")
    print("  same boundary; k-means centres and greedy farthest points")
    print("  spread the batch, and both reach similar accuracy without")
    print("  ever reading the model's probabilities")

    print("active. expected model change buys the largest gradient:")
    learner = ActiveLearner("expected_model_change", seed=3).run(
        pool_x, pool_y, test_x, test_y)
    scores["expected_model_change"] = learner.history[-1]
    print(f"    expected_model_change {learner.history[-1]:.3f} accuracy")
    assert learner.history[-1] > random_history[-1]
    print("  the score is the expected squared gradient norm under the")
    print("  model's own predictive distribution, so it prefers points")
    print("  that are both uncertain and far from the labelled set")

    print("active. the learning curves, one row per method:")
    print("    labels " + "".join(f"{8 + 4 * step:>8d}"
                                   for step in (0, 2, 4, 6, 8, 10)))
    for name in ("uncertainty", "entropy", "core_set"):
        learner = ActiveLearner(name, seed=4).run(
            pool_x, pool_y, test_x, test_y)
        path = [learner.history[step] for step in (0, 2, 4, 6, 8, 10)]
        print(f"    {name:<10s}" + "".join(f"{value:8.3f}"
                                           for value in path))

    print("active. eight acquisitions, one table:")
    table = {}
    for variant in VARIANTS:
        learner = ActiveLearner(variant, seed=5).run(
            pool_x, pool_y, test_x, test_y)
        table[variant] = learner.history[-1]
        print(f"    {variant:<21s} {table[variant]:.3f} accuracy")
    assert all(score > random_history[-1] - 0.03
               for score in table.values())
    assert max(table.values()) > random_history[-1] + 0.04

    print("active. malformed calls are refused:")
    guard = ActiveLearner("uncertainty", seed=6)
    good_x, good_y = pool_x[:120], pool_y[:120]
    guard.run(good_x, good_y, test_x[:60], test_y[:60])
    bad = [
        lambda: ActiveLearner("uncertainty_v2"),
        lambda: ActiveLearner("uncertainty", initial=0),
        lambda: ActiveLearner("uncertainty", batch=0),
        lambda: ActiveLearner("uncertainty", rounds=0),
        lambda: ActiveLearner("uncertainty", classes=1),
        lambda: ActiveLearner("qbc", ensemble=1),
        lambda: ActiveLearner("uncertainty", steps=0),
        lambda: ActiveLearner("uncertainty", lr=0.0),
        lambda: guard.run(good_x[:, :1], good_y, test_x[:60], test_y[:60]),
        lambda: guard.run(good_x, good_y[:-1], test_x[:60], test_y[:60]),
        lambda: guard.run(good_x, good_y.astype(float), test_x[:60],
                          test_y[:60]),
        lambda: guard.run(good_x, good_y + 5, test_x[:60], test_y[:60]),
        lambda: guard.run(np.full_like(good_x, np.nan), good_y, test_x[:60],
                          test_y[:60]),
        lambda: guard.run(good_x, good_y, test_x[:0], test_y[:0]),
        lambda: guard.run(good_x[:10], good_y[:10], test_x[:60], test_y[:60]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero budget")
    print("  component, one class, a one-member ensemble, a zero step or")
    print("  learning rate, a one-dimensional pool, mismatched or float or")
    print("  out-of-range labels, a non-finite pool, an empty test set, and")
    print("  a pool smaller than the requested budget")
    print("active demo passed")


if __name__ == "__main__":
    demo()
