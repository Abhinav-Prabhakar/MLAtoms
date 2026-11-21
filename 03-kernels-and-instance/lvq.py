"""Learning vector quantisation — prototype classifiers with attraction and repulsion.

Chapter:   03-kernels-and-instance
Companion: lvq.md
Tier 3 · 5 variants: lvq1, lvq2, lvq3, glvq, grlvq
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("lvq1", "lvq2", "lvq3", "glvq", "grlvq")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_data(X, y):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X needs at least two rows and one column")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    y = np.asarray(y)
    if y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError("y must have one label per row of X")
    if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    classes, codes = np.unique(y, return_inverse=True)
    if classes.size < 2:
        raise ValueError("at least two classes are required")
    return X, codes, classes


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class LearningVectorQuantization:
    """Nearest-prototype classifier trained by attraction and repulsion.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    prototypes_per_class : int
        How many labelled prototypes to seed per class.
    epochs : int
        Passes over the training set. The learning rate decays linearly to zero.
    lr : float
        Initial step size for the prototype updates.
    lr_relevance : float
        Initial step size for the `grlvq` relevance vector.
    window : float
        `lvq2`/`lvq3` accept an update when the distance ratio of the two
        nearest prototypes exceeds this value in both directions.
    epsilon : float
        `lvq3` shrink factor applied to two correct prototypes outside the
        window.
    seed : int
        Seed for prototype initialisation and the per-epoch shuffles.

    Notes
    -----
    `fit` stores `prototypes_` `(P, d)`, `prototype_labels_` `(P,)` and, for
    `grlvq`, `relevance_` `(d,)` on the non-negative unit sphere. Prototypes
    never change class: unlike k-means, the labels are supervision, not a
    quantity to discover.
    """

    def __init__(self, variant: str = "lvq1", *, prototypes_per_class: int = 1,
                 epochs: int = 30, lr: float = 0.1, lr_relevance: float = 0.01,
                 window: float = 0.7, epsilon: float = 0.2, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        for name, value in (("lr", lr), ("lr_relevance", lr_relevance)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < window <= 1.0:
            raise ValueError("window must lie in (0, 1]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.variant = variant
        self.prototypes_per_class = int(prototypes_per_class)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.lr_relevance = float(lr_relevance)
        self.window = float(window)
        self.epsilon = float(epsilon)
        self.seed = int(seed)

    # -- distances -------------------------------------------------------- #
    def distances(self, X: np.ndarray) -> np.ndarray:
        """Weighted squared distances to every prototype, shape `(n, P)`."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.prototypes_.shape[1]:
            raise ValueError("X must be (n, d) with the fitted width")
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        if self.relevance_ is None:
            return np.einsum("npd,npd->np", diff, diff)
        return np.einsum("d,npd,npd->np", self.relevance_ ** 2, diff, diff)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> LearningVectorQuantization:
        """Seed labelled prototypes from the data, then run the update rule."""
        X, codes, classes = _check_data(X, y)
        n, width = X.shape
        rng = np.random.default_rng(self.seed)
        picks, labels = [], []
        for code in range(classes.size):
            members = np.flatnonzero(codes == code)
            if members.size < self.prototypes_per_class:
                raise ValueError("a class has fewer rows than prototypes requested")
            chosen = rng.choice(members, self.prototypes_per_class, replace=False)
            picks.append(chosen)
            labels.append(np.full(self.prototypes_per_class, code))
        self.prototypes_ = X[np.concatenate(picks)].copy()
        self.prototype_codes_ = np.concatenate(labels)
        self.classes_ = classes
        self.n_features_in_ = width
        self.relevance_ = (np.full(width, 1.0 / np.sqrt(width))
                           if self.variant == "grlvq" else None)
        self.cost_path_ = []
        self.n_updates_ = 0
        tracked = self.variant in ("glvq", "grlvq")

        for epoch in range(self.epochs):
            rate = self.lr * (1.0 - epoch / self.epochs)
            relevance_rate = self.lr_relevance * (1.0 - epoch / self.epochs)
            loss = 0.0
            for index in rng.permutation(n):
                loss += self._step(X[index], int(codes[index]), rate,
                                   relevance_rate)
            if tracked:
                self.cost_path_.append(loss / n)
        return self

    def _step(self, x, label, rate, relevance_rate) -> float:
        if self.variant == "lvq1":
            return self._step_lvq1(x, label, rate)
        if self.variant in ("lvq2", "lvq3"):
            return self._step_window(x, label, rate)
        return self._step_gradient(x, label, rate, relevance_rate)

    def _step_lvq1(self, x, label, rate) -> float:
        d = self.distances(x[None])[0]
        k = int(np.argmin(d))
        delta = rate * (x - self.prototypes_[k])
        if self.prototype_codes_[k] == label:
            self.prototypes_[k] += delta
        else:
            self.prototypes_[k] -= delta
        self.n_updates_ += 1
        return 0.0

    def _step_window(self, x, label, rate) -> float:
        d = self.distances(x[None])[0]
        first, second = np.argsort(d)[:2]
        near, far = d[first], d[second]
        if near + far <= 0.0:
            return 0.0
        both_correct = (self.prototype_codes_[first] == label
                        and self.prototype_codes_[second] == label)
        if both_correct:
            if self.variant == "lvq3" and self.epsilon > 0.0:
                for k in (first, second):
                    self.prototypes_[k] += self.epsilon * rate * (
                        x - self.prototypes_[k])
                self.n_updates_ += 1
            return 0.0
        if self.prototype_codes_[first] == self.prototype_codes_[second]:
            return 0.0
        if near / far <= self.window:
            return 0.0
        correct = first if self.prototype_codes_[first] == label else second
        if self.prototype_codes_[correct] != label:
            return 0.0
        wrong = second if correct == first else first
        self.prototypes_[correct] += rate * (x - self.prototypes_[correct])
        self.prototypes_[wrong] -= rate * (x - self.prototypes_[wrong])
        self.n_updates_ += 1
        return 0.0

    def _step_gradient(self, x, label, rate, relevance_rate) -> float:
        d = self.distances(x[None])[0]
        same = self.prototype_codes_ == label
        if not same.any() or same.all():
            return 0.0
        plus = int(np.argmin(np.where(same, d, np.inf)))
        minus = int(np.argmin(np.where(same, np.inf, d)))
        total = d[plus] + d[minus]
        if total <= 0.0:
            return 0.0
        toward = x - self.prototypes_[plus]
        away = x - self.prototypes_[minus]
        weight = 4.0 * rate / total ** 2
        if self.relevance_ is None:
            self.prototypes_[plus] += weight * d[minus] * toward
            self.prototypes_[minus] -= weight * d[plus] * away
        else:
            lam_sq = self.relevance_ ** 2
            self.prototypes_[plus] += weight * d[minus] * lam_sq * toward
            self.prototypes_[minus] -= weight * d[plus] * lam_sq * away
            gradient = (4.0 * self.relevance_ / total ** 2) * (
                d[minus] * toward ** 2 - d[plus] * away ** 2)
            proposal = np.maximum(self.relevance_ - relevance_rate * gradient,
                                  0.0)
            norm = float(np.linalg.norm(proposal))
            if norm > 0.0:
                self.relevance_ = proposal / norm
        self.n_updates_ += 1
        return float((d[plus] - d[minus]) / total)

    # -- prediction ------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Labels of the nearest prototypes, shape `(n,)`."""
        d = self.distances(X)
        return self.classes_[self.prototype_codes_[np.argmin(d, axis=1)]]

    def _check_fitted(self):
        if not hasattr(self, "prototypes_"):
            raise RuntimeError("call fit before distances or predict")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _blobs(n: int, rng: np.random.Generator):
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    labels = rng.integers(0, 3, n)
    return centres[labels] + rng.normal(0.0, 0.7, (n, 2)), labels


def _buried_line(n: int, rng: np.random.Generator):
    """A linear boundary in two axes, six noise axes with twice the spread."""
    signal = rng.uniform(-1.0, 1.0, (n, 2))
    labels = np.where(signal[:, 0] + signal[:, 1] > 0.0, 1, -1)
    return np.hstack([signal, rng.normal(0.0, 2.0, (n, 6))]), labels


def demo() -> None:
    rng = np.random.default_rng(0)
    train, train_y = _blobs(240, rng)
    test, test_y = _blobs(600, rng)
    print("three isotropic blobs: every variant separates them, and the")
    print("variants differ in what they do, not in accuracy")
    fits = {}
    for variant in ("lvq1", "lvq2", "lvq3", "glvq", "grlvq"):
        model = LearningVectorQuantization(
            variant, prototypes_per_class=3, epochs=30,
            seed=0).fit(train, train_y)
        fits[variant] = model
        accuracy = float(np.mean(model.predict(test) == test_y))
        line = f"  {variant:6s} {accuracy:.4f}  {model.n_updates_:5d} updates"
        if model.cost_path_:
            line += (f"  mu {model.cost_path_[0]:+.4f} -> "
                     f"{model.cost_path_[-1]:+.4f}")
        print(line)
        assert accuracy > 0.98
    assert fits["lvq2"].n_updates_ < fits["lvq1"].n_updates_ / 100
    print("lvq2 moved its prototypes on a handful of samples: the window only")
    print("opens where two classes' prototypes are almost equidistant")
    assert fits["glvq"].cost_path_[-1] < fits["glvq"].cost_path_[0]
    relevance = fits["grlvq"].relevance_
    print(f"grlvq's relevance stays near uniform on isotropic data: "
          f"{np.round(relevance, 4)}")
    assert relevance.max() / relevance.min() < 1.1
    assert np.isclose(np.linalg.norm(relevance), 1.0)

    print("now bury a linear boundary under six axes with twice the spread:")
    rng = np.random.default_rng(11)
    train, train_y = _buried_line(300, rng)
    test, test_y = _buried_line(1000, rng)
    scores = {}
    for variant in ("lvq1", "glvq", "grlvq"):
        model = LearningVectorQuantization(variant, prototypes_per_class=2,
                                           epochs=30, seed=0).fit(train, train_y)
        scores[variant] = float(np.mean(model.predict(test) == test_y))
        extra = ""
        if model.relevance_ is not None:
            extra = "  rel " + np.array2string(model.relevance_, precision=2)
        print(f"  {variant:6s} {scores[variant]:.4f}{extra}")
    assert scores["lvq1"] < 0.7 and scores["glvq"] < 0.7
    assert scores["grlvq"] > 0.95
    assert np.all(model.relevance_[:2] > 0.6)
    assert np.all(model.relevance_[2:] < 1e-9)
    assert np.isclose(np.linalg.norm(model.relevance_), 1.0)
    assert np.all(model.relevance_ >= 0.0)
    print("the relevance vector put all of its mass on the two informative")
    print("axes and set the six loud ones to zero")

    print("mechanics. lvq1 is a closed-form interpolation, toward or away:")
    tiny = LearningVectorQuantization("lvq1", prototypes_per_class=1, epochs=1,
                                      seed=0)
    tiny.fit(np.array([[0.0, 0.0], [3.0, 3.0]]), np.array([0, 1]))
    x = np.array([1.0, 2.0])
    tiny.prototypes_ = np.array([[0.0, 0.0], [3.0, 3.0]])
    tiny.prototype_codes_ = np.array([0, 1])
    tiny._step_lvq1(x, 0, 0.5)
    assert np.allclose(tiny.prototypes_[0], 0.5 * x)
    far = np.array([2.5, 2.5])
    tiny._step_lvq1(far, 0, 0.5)
    assert np.allclose(tiny.prototypes_[1], np.array([3.0, 3.0])
                       + 0.5 * (np.array([3.0, 3.0]) - far))
    print(f"  a match gives (1 - a)w + ax = {np.round(tiny.prototypes_[0], 3)}")
    print(f"  a mismatch gives (1 + a)w - ax = "
          f"{np.round(tiny.prototypes_[1], 3)}")

    window = LearningVectorQuantization("lvq2", prototypes_per_class=1,
                                        epochs=1, window=0.7, seed=0)
    window.fit(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([0, 1]))
    window.prototypes_ = np.array([[0.0, 0.0], [2.0, 0.0]])
    window.prototype_codes_ = np.array([0, 1])
    outside = window.prototypes_.copy()
    window._step_window(np.array([0.4, 0.0]), 1, 0.5)
    assert np.allclose(window.prototypes_, outside)
    inside = window.prototypes_.copy()
    window._step_window(np.array([0.95, 0.0]), 1, 0.5)
    moved = np.abs(window.prototypes_ - inside).sum()
    print("  d1/d2 = 0.0625 is outside the window, so no move; 0.818 is")
    print(f"  inside, and the pair travels {moved:.2f} in total")
    assert moved > 0.0
    assert window.prototypes_[1, 0] < inside[1, 0]

    print("glvq's prototype update is the gradient of its own cost, checked")
    print("against a finite difference of mu = (d+ - d-)/(d+ + d-):")
    tiny = LearningVectorQuantization("glvq", prototypes_per_class=1, epochs=1,
                                      lr=0.1, seed=0)
    tiny.fit(np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
             np.array([0, 1, 0, 1]))
    base = np.array([[0.2, -0.1], [1.4, 1.1]])
    tiny.prototypes_ = base.copy()
    tiny.prototype_codes_ = np.array([0, 1])
    point = np.array([0.4, 0.1])

    def margin(prototypes):
        tiny.prototypes_ = prototypes
        d = tiny.distances(point[None])[0]
        return float((d[0] - d[1]) / (d[0] + d[1]))

    d = tiny.distances(point[None])[0]
    analytic = -4.0 * d[1] / (d[0] + d[1]) ** 2 * (point - base[0])
    numeric = np.empty(2)
    for j in range(2):
        up, down = base.copy(), base.copy()
        up[0, j] += 1e-6
        down[0, j] -= 1e-6
        numeric[j] = (margin(up) - margin(down)) / 2e-6
    assert np.allclose(numeric, analytic, atol=1e-7)
    tiny.prototypes_ = base.copy()
    print(f"  analytic {np.round(analytic, 6)}  numeric "
          f"{np.round(numeric, 6)}")

    plain = LearningVectorQuantization("glvq", prototypes_per_class=1,
                                       epochs=1, seed=0)
    plain.fit(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([0, 1]))
    weighted = LearningVectorQuantization("grlvq", prototypes_per_class=1,
                                          epochs=1, seed=0)
    weighted.fit(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([0, 1]))
    start = np.array([[0.3, 0.1], [1.2, 1.6]])
    plain.prototypes_ = start.copy()
    plain.prototype_codes_ = np.array([0, 1])
    weighted.prototypes_ = start.copy()
    weighted.prototype_codes_ = np.array([0, 1])
    sample = np.array([0.5, 0.4])
    plain._step_gradient(sample, 0, 0.1, 0.0)
    weighted._step_gradient(sample, 0, 0.1, 0.0)
    assert np.allclose(plain.prototypes_, weighted.prototypes_)
    print("  with uniform relevance, grlvq's prototype step equals glvq's")
    print("  exactly: the metric adds directions, not a different rule")

    bad = [
        lambda: LearningVectorQuantization("som"),
        lambda: LearningVectorQuantization(prototypes_per_class=0),
        lambda: LearningVectorQuantization(epochs=0),
        lambda: LearningVectorQuantization(lr=0.0),
        lambda: LearningVectorQuantization(lr_relevance=-1.0),
        lambda: LearningVectorQuantization(window=0.0),
        lambda: LearningVectorQuantization(window=1.5),
        lambda: LearningVectorQuantization(epsilon=-0.1),
        lambda: LearningVectorQuantization(epsilon=1.5),
        lambda: LearningVectorQuantization(seed=-1),
        lambda: LearningVectorQuantization().fit(train[:, 0], train_y),
        lambda: LearningVectorQuantization().fit(train, np.zeros(300)),
        lambda: LearningVectorQuantization().fit(train, train_y[:-1]),
        lambda: LearningVectorQuantization().fit(
            np.full((10, 8), np.nan), np.arange(10) % 2),
        lambda: LearningVectorQuantization().fit(
            train, np.where(train_y == train_y[0], np.nan, train_y)),
        lambda: LearningVectorQuantization().predict(train),
        lambda: LearningVectorQuantization().fit(train, train_y).predict(
            np.ones((5, 3))),
        lambda: LearningVectorQuantization(prototypes_per_class=5).fit(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]),
            np.array([0, 0, 0, 1])),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, zero prototypes, zero epochs, a zero or negative step,")
    print("  a window outside (0, 1], epsilon outside [0, 1], a negative seed,")
    print("  a 1-d design, one class, a short or non-finite label vector, a")
    print("  non-finite design, predict before fit, a wrong width, and a")
    print("  class with fewer rows than prototypes requested")
    print("lvq demo passed")


if __name__ == "__main__":
    demo()
