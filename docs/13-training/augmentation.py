"""Data augmentation — eight policies that make one image many.

Chapter:   13-training
Companion: augmentation.md
Tier 2 · 8 variants: mixup, cutmix, cutout, random_erasing, randaugment,
augmix, color_jitter, auto_augment
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("mixup", "cutmix", "cutout", "random_erasing", "randaugment",
            "augmix", "color_jitter", "auto_augment")


# --------------------------------------------------------------------------- #
# The elementary transforms
# --------------------------------------------------------------------------- #
def _solarize(image: np.ndarray, m: float, rng) -> np.ndarray:
    return np.where(image > 1.0 - 0.9 * m, 1.0 - image, image)


def _posterize(image: np.ndarray, m: float, rng) -> np.ndarray:
    levels = 2.0 ** (8 - int(round(6.0 * m))) - 1.0
    return np.round(image * levels) / levels


def _brightness(image: np.ndarray, m: float, rng) -> np.ndarray:
    return (1.0 - 0.5 * m) * image + 0.5 * m


def _contrast(image: np.ndarray, m: float, rng) -> np.ndarray:
    mean = image.mean(axis=(1, 2), keepdims=True)
    return (1.0 - 0.9 * m) * (image - mean) + mean


def _rotate(image: np.ndarray, m: float, rng) -> np.ndarray:
    quarter = 1 + int(round(2.0 * m))
    if rng.random() < 0.5:
        quarter = -quarter
    return np.rot90(image, quarter, axes=(1, 2))


def _translate(image: np.ndarray, m: float, rng) -> np.ndarray:
    reach = int(round(m * image.shape[1] / 2.0))
    down = int(rng.integers(-reach, reach + 1))
    right = int(rng.integers(-reach, reach + 1))
    return np.roll(np.roll(image, down, axis=1), right, axis=2)


def _invert(image: np.ndarray, m: float, rng) -> np.ndarray:
    return 1.0 - image


OPS = {"solarize": _solarize, "posterize": _posterize,
       "brightness": _brightness, "contrast": _contrast,
       "rotate": _rotate, "translate": _translate, "invert": _invert}

POLICIES = (
    (("solarize", 1.0, 4.0), ("contrast", 1.0, 4.0)),
    (("posterize", 1.0, 5.0), ("rotate", 1.0, 5.0)),
    (("brightness", 1.0, 6.0), ("translate", 1.0, 4.0)),
)


def _onehot(labels: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((labels.size, classes))
    out[np.arange(labels.size), labels] = 1.0
    return out


def _hue_matrix(angle: float) -> np.ndarray:
    """The CSS hue-rotate matrix, which preserves grey and luminance."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [0.213 + 0.787 * c - 0.213 * s,
         0.715 - 0.715 * c - 0.715 * s,
         0.072 - 0.072 * c + 0.928 * s],
        [0.213 - 0.213 * c + 0.143 * s,
         0.715 + 0.285 * c + 0.140 * s,
         0.072 - 0.072 * c - 0.283 * s],
        [0.213 - 0.213 * c - 0.787 * s,
         0.715 - 0.715 * c + 0.715 * s,
         0.072 + 0.928 * c + 0.072 * s],
    ])


# --------------------------------------------------------------------------- #
# The augmenter
# --------------------------------------------------------------------------- #
class Augmenter:
    """One class, eight policies, batches of `(B, H, W, C)` images in [0, 1].

    `__call__(x, y)` returns `(images, targets, lam)`. Targets are a soft
    `(B, classes)` matrix: one-hot for the geometric policies, the convex
    combination of two rows for `mixup` and `cutmix`. `lam` is the weight of
    the untouched image, and 1.0 for every policy that does not mix.
    """

    def __init__(self, variant: str = "mixup", *, seed: int = 0,
                 channels: int = 3, classes: int = 2, alpha: float = 0.4,
                 coverage: float = 0.5, p: float = 0.5,
                 erase_scale: tuple[float, float] = (0.02, 0.25),
                 ratio: tuple[float, float] = (0.3, 3.3),
                 fill: str = "zero", ops: tuple[str, ...] | None = None,
                 n: int = 2, m: float = 5.0,
                 k: int = 3, depth: int = 2, brightness: float = 0.3,
                 contrast: float = 0.3, saturation: float = 0.3,
                 hue: float = 0.1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if channels < 1 or classes < 2:
            raise ValueError("channels must be positive and classes at least 2")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly inside (0, 1)")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        low, high = erase_scale
        if low <= 0.0 or high < low or high > 1.0:
            raise ValueError("erase_scale must be a positive interval in (0, 1]")
        if ratio[0] <= 0.0 or ratio[1] < ratio[0]:
            raise ValueError("ratio must be a positive interval")
        if fill not in ("zero", "mean", "random"):
            raise ValueError("fill must be zero, mean or random")
        if variant == "color_jitter" and channels != 3:
            raise ValueError("color_jitter needs three channels")
        if ops is not None and (not ops or any(name not in OPS for name in ops)):
            raise ValueError("ops must name at least one known transform")
        if n < 1 or k < 1 or depth < 1:
            raise ValueError("n, k and depth must be positive")
        if m < 0.0 or brightness < 0.0 or contrast < 0.0 or saturation < 0.0:
            raise ValueError("magnitudes must be non-negative")
        if not 0.0 <= hue <= 1.0:
            raise ValueError("hue must lie in [0, 1]")
        self.variant = variant
        self.channels = channels
        self.classes = classes
        self.alpha = float(alpha)
        self.coverage = float(coverage)
        self.p = float(p)
        self.erase_scale = (float(low), float(high))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.fill = fill
        self.ops = tuple(OPS) if ops is None else tuple(ops)
        self.n = int(n)
        self.m = float(m)
        self.k = int(k)
        self.depth = int(depth)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.rng = np.random.default_rng(seed)
        self.counts = {"ops": 0, "erased": 0, "policies": 0}
        self.branches: list[np.ndarray] = []
        self.weights = np.zeros((0, 0))

    # -- validation ---------------------------------------------------------- #
    def _check(self, x) -> np.ndarray:
        images = np.asarray(x, dtype=float)
        if images.ndim != 4:
            raise ValueError("x must be (batch, height, width, channels)")
        if images.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if images.shape[1] < 4 or images.shape[2] < 4:
            raise ValueError("images must be at least 4 by 4")
        if images.shape[3] != self.channels:
            raise ValueError("x has the wrong number of channels")
        if not np.all(np.isfinite(images)):
            raise ValueError("x must be finite")
        if images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("x must lie in [0, 1]")
        return images

    def _labels(self, y, batch: int) -> np.ndarray:
        labels = np.asarray(y)
        if labels.shape != (batch,):
            raise ValueError("y must hold one label per image")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return labels

    # -- dispatch ------------------------------------------------------------ #
    def __call__(self, x, y=None):
        images = self._check(x)
        labels = None if y is None else self._labels(y, images.shape[0])
        self.counts = {"ops": 0, "erased": 0, "policies": 0}
        self.branches = []
        if self.variant in ("mixup", "cutmix"):
            if labels is None:
                raise ValueError(f"{self.variant} needs labels")
            return getattr(self, self.variant)(images, labels)
        result = getattr(self, self.variant)(images)
        target = None if labels is None else _onehot(labels, self.classes)
        return result, target, 1.0

    # -- mixing policies ----------------------------------------------------- #
    def mixup(self, x: np.ndarray, labels: np.ndarray):
        lam = float(self.rng.beta(self.alpha, self.alpha))
        other = self.rng.permutation(x.shape[0])
        images = lam * x + (1.0 - lam) * x[other]
        target = lam * _onehot(labels, self.classes) + (1.0 - lam) * _onehot(
            labels[other], self.classes)
        return images, target, lam

    def cutmix(self, x: np.ndarray, labels: np.ndarray):
        lam = float(self.rng.beta(self.alpha, self.alpha))
        height, width = x.shape[1], x.shape[2]
        box_h = max(1, int(round(height * np.sqrt(1.0 - lam))))
        box_w = max(1, int(round(width * np.sqrt(1.0 - lam))))
        top = int(self.rng.integers(0, height - box_h + 1))
        left = int(self.rng.integers(0, width - box_w + 1))
        other = self.rng.permutation(x.shape[0])
        images = x.copy()
        images[:, top:top + box_h, left:left + box_w] = x[
            other, top:top + box_h, left:left + box_w]
        lam = 1.0 - box_h * box_w / float(height * width)
        target = lam * _onehot(labels, self.classes) + (1.0 - lam) * _onehot(
            labels[other], self.classes)
        return images, target, lam

    # -- occlusion policies -------------------------------------------------- #
    def cutout(self, x: np.ndarray) -> np.ndarray:
        height, width = x.shape[1], x.shape[2]
        box_h = max(1, int(round(self.coverage * height)))
        box_w = max(1, int(round(self.coverage * width)))
        out = x.copy()
        for index in range(x.shape[0]):
            top = int(self.rng.integers(0, height - box_h + 1))
            left = int(self.rng.integers(0, width - box_w + 1))
            out[index, top:top + box_h, left:left + box_w] = 0.0
            self.counts["erased"] += box_h * box_w
        return out

    def random_erasing(self, x: np.ndarray) -> np.ndarray:
        height, width = x.shape[1], x.shape[2]
        out = x.copy()
        for index in range(x.shape[0]):
            if self.rng.random() > self.p:
                continue
            area = self.rng.uniform(*self.erase_scale) * height * width
            aspect = np.exp(self.rng.uniform(np.log(self.ratio[0]),
                                             np.log(self.ratio[1])))
            box_h = min(height, max(1, int(round(np.sqrt(area / aspect)))))
            box_w = min(width, max(1, int(round(np.sqrt(area * aspect)))))
            top = int(self.rng.integers(0, height - box_h + 1))
            left = int(self.rng.integers(0, width - box_w + 1))
            if self.fill == "random":
                patch = self.rng.normal(0.5, 0.2, (box_h, box_w, self.channels))
            elif self.fill == "mean":
                patch = out[index].mean(axis=(0, 1))
            else:
                patch = 0.0
            out[index, top:top + box_h, left:left + box_w] = patch
            self.counts["erased"] += box_h * box_w
        return np.clip(out, 0.0, 1.0)

    # -- policy policies ----------------------------------------------------- #
    def _apply(self, name: str, image: np.ndarray, magnitude: float,
               rng) -> np.ndarray:
        if magnitude <= 0.0:
            return image
        return OPS[name](image, magnitude, rng)

    def randaugment(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for _ in range(self.n):
            name = str(self.rng.choice(list(self.ops)))
            out = self._apply(name, out, self.m / 10.0, self.rng)
            self.counts["ops"] += 1
        return np.clip(out, 0.0, 1.0)

    def auto_augment(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for index in range(x.shape[0]):
            policy = POLICIES[int(self.rng.integers(len(POLICIES)))]
            self.counts["policies"] += 1
            for name, probability, magnitude in policy:
                if self.rng.random() < probability:
                    out[index:index + 1] = self._apply(
                        name, out[index:index + 1], magnitude / 10.0, self.rng)
                    self.counts["ops"] += 1
        return np.clip(out, 0.0, 1.0)

    def augmix(self, x: np.ndarray) -> np.ndarray:
        weights = self.rng.dirichlet(np.ones(self.k), size=x.shape[0])
        branches = []
        for _ in range(self.k):
            branch = x.copy()
            for _ in range(self.depth):
                name = str(self.rng.choice(list(self.ops)))
                magnitude = float(self.rng.random()) * self.m / 10.0
                branch = self._apply(name, branch, magnitude, self.rng)
            branches.append(branch)
        mixed = sum(weights[:, index, None, None, None] * branch
                    for index, branch in enumerate(branches))
        self.branches = branches
        self.weights = weights
        return np.clip(mixed, 0.0, 1.0)

    @staticmethod
    def jsd(probs: list[np.ndarray]) -> np.ndarray:
        """Jensen-Shannon divergence of (batch, classes) distributions."""
        stack = np.stack([np.asarray(p, dtype=float) for p in probs])
        mean = stack.mean(axis=0)
        terms = []
        for member in stack:
            safe = np.maximum(member, 1e-12)
            terms.append(np.sum(safe * (np.log(safe)
                                        - np.log(np.maximum(mean, 1e-12))),
                                axis=-1))
        return np.mean(terms, axis=0)

    # -- photometric policy -------------------------------------------------- #
    def color_jitter(self, x: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        out = x.copy()
        for step in self.rng.permutation(4):
            if step == 0:
                factor = self.rng.uniform(1.0 - self.brightness,
                                          1.0 + self.brightness, (batch, 1, 1, 1))
                out = out * factor
            elif step == 1:
                factor = self.rng.uniform(1.0 - self.contrast,
                                          1.0 + self.contrast, (batch, 1, 1, 1))
                mean = out.mean(axis=(1, 2), keepdims=True)
                out = (out - mean) * factor + mean
            elif step == 2:
                factor = self.rng.uniform(1.0 - self.saturation,
                                          1.0 + self.saturation,
                                          (batch, 1, 1, 1))
                grey = out.mean(axis=-1, keepdims=True)
                out = (out - grey) * factor + grey
            else:
                angle = self.rng.uniform(-self.hue, self.hue, batch)
                matrix = np.stack([_hue_matrix(value) for value in angle])
                out = np.einsum("bij,bhwj->bhwi", matrix, out)
        return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# A linear model that can consume soft targets
# --------------------------------------------------------------------------- #
class _MLP:
    """A one-hidden-layer tanh classifier that consumes soft targets."""

    def __init__(self, dim: int, hidden: int, classes: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, classes))
        self.b2 = np.zeros(classes)

    def probabilities(self, x: np.ndarray):
        pre = x @ self.W1 + self.b1
        hidden = np.tanh(pre)
        logits = hidden @ self.W2 + self.b2
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True), hidden

    def step(self, x: np.ndarray, target: np.ndarray, lr: float) -> None:
        probs, hidden = self.probabilities(x)
        difference = (probs - target) / x.shape[0]
        grad_pre = (difference @ self.W2.T) * (1.0 - hidden ** 2)
        self.W2 -= lr * hidden.T @ difference
        self.b2 -= lr * difference.sum(axis=0)
        self.W1 -= lr * x.T @ grad_pre
        self.b1 -= lr * grad_pre.sum(axis=0)

    def accuracy(self, x: np.ndarray, labels: np.ndarray) -> float:
        probs, _ = self.probabilities(x)
        return float(np.mean(np.argmax(probs, axis=-1) == labels))


# --------------------------------------------------------------------------- #
def _images(seed: int, per_class: int = 30, size: int = 8,
            noise: float = 0.12):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(2), per_class)
    horizontal = np.zeros((size, size, 3))
    horizontal[:2, :, :] = 1.0
    vertical = np.zeros((size, size, 3))
    vertical[:, :2, :] = 1.0
    x = np.zeros((labels.size, size, size, 3))
    for index, label in enumerate(labels):
        bar = horizontal if label == 0 else vertical
        shift = int(rng.integers(size))
        x[index] = np.roll(bar, shift, axis=label)
    x += rng.normal(0.0, noise, x.shape)
    order = rng.permutation(labels.size)
    return np.clip(x[order], 0.0, 1.0), labels[order]


def _rgb(seed: int, batch: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = rng.random((batch, 6, 6, 3))
    return np.clip(images, 0.0, 1.0)


def demo() -> None:
    x, y = _images(seed=1)

    print("augmentation. the mixup weight is a Beta draw:")
    aug = Augmenter("mixup", seed=2, alpha=0.4, classes=2)
    pair_x, pair_y = x[:2], y[:2]
    mixed, target, lam = aug(pair_x, pair_y)
    assert 0.0 < lam < 1.0
    swapped = not np.isclose(target[0, 0], 1.0)
    source = pair_x[[1, 0]] if swapped else pair_x
    assert np.allclose(mixed, lam * pair_x + (1.0 - lam) * source)
    draws = np.array([Augmenter("mixup", seed=s, alpha=0.4)
                      (pair_x, pair_y)[2] for s in range(400)])
    assert abs(draws.mean() - 0.5) < 0.03
    print(f"  lam {lam:.3f}; over 400 seeds the mean is {draws.mean():.3f},")
    print("  because a symmetric Beta is centred on one half")
    print(f"  the target row sums to {target[0].sum():.3f} and every mixed")
    print("  pixel lies on the segment between the two source images")

    print("augmentation. cutmix pastes a box and rescales the target:")
    aug = Augmenter("cutmix", seed=3, alpha=1.0, classes=2)
    mixed, target, lam = aug(pair_x, pair_y)
    swapped = not np.isclose(target[0, 0], 1.0)
    source = pair_x[[1, 0]] if swapped else pair_x
    changed = np.any(mixed != pair_x, axis=(1, 2, 3))
    assert changed.any()
    pasted = np.isclose(mixed, source) & ~np.isclose(pair_x, source)
    rows = np.flatnonzero(pasted[0].any(axis=(1, 2)))
    cols = np.flatnonzero(pasted[0].any(axis=(0, 2)))
    box = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    assert np.isclose(lam, 1.0 - box / 64.0)
    print(f"  a {rows.size}x{cols.size} box is replaced, so lam is")
    print(f"  1 - {box}/64 = {lam:.3f}; the target moves with it")

    print("augmentation. cutout erases an exact fraction:")
    aug = Augmenter("cutout", seed=4, coverage=0.5)
    blank, _, _ = aug(np.ones_like(x[:1]))
    erased = float(np.mean(blank == 0.0))
    assert np.isclose(erased, 0.25)
    print(f"  a 4x4 hole in an 8x8 image zeroes {erased:.0%} of the pixels")
    print("  the box is one contiguous square, not scattered noise")

    print("augmentation. random erasing samples area and aspect:")
    aug = Augmenter("random_erasing", seed=5, p=1.0, fill="mean")
    erased, _, _ = aug(x)
    zero = Augmenter("random_erasing", seed=5, p=0.0)
    untouched, _, _ = zero(x)
    assert np.array_equal(untouched, x)
    assert aug.counts["erased"] >= x.shape[0]
    print(f"  at p=1 the policy touches {aug.counts['erased']} pixels in all;")
    print("  at p=0 the batch comes back bit-identical, so the switch is real")

    print("augmentation. randaugment applies n magnitude-scaled ops:")
    off = Augmenter("randaugment", seed=6, n=3, m=0.0)
    same, _, _ = off(x)
    assert np.array_equal(same, x)
    on = Augmenter("randaugment", seed=6, n=3, m=8.0)
    changed, _, _ = on(x)
    assert on.counts["ops"] == 3 and not np.array_equal(changed, x)
    ramp = np.linspace(0.0, 1.0, 64).reshape(1, 8, 8, 1)
    flipped = [float(np.mean(_solarize(ramp, m, None) != ramp))
               for m in (0.0, 0.3, 0.7, 1.0)]
    assert flipped == sorted(flipped) and flipped[-1] > 0.85
    print("  severity 0 is the identity, by construction; severity 8 runs")
    print("  three ops and changes the batch")
    print("  solarize on a ramp flips a growing share as severity rises: "
          + " ".join(f"{value:.2f}" for value in flipped))

    print("augmentation. augmix mixes branches with Dirichlet weights:")
    aug = Augmenter("augmix", seed=7, k=3, depth=2, m=8.0)
    mixed, _, _ = aug(x)
    assert np.allclose(aug.weights.sum(axis=1), 1.0)
    stack = np.stack(aug.branches)
    assert np.all(mixed >= stack.min(axis=0) - 1e-12)
    assert np.all(mixed <= stack.max(axis=0) + 1e-12)
    probs = np.stack([branch.reshape(branch.shape[0], -1).mean(axis=1)
                      for branch in aug.branches], axis=-1)
    probs = np.exp(probs) / np.exp(probs).sum(axis=-1, keepdims=True)
    same = Augmenter.jsd([probs[..., 0]] * 3)
    apart = Augmenter.jsd([probs[..., index] for index in range(3)])
    assert np.allclose(same, 0.0, atol=1e-12) and apart.max() > 0.0
    print("  three chains, weights summing to one, so every mixed pixel")
    print("  lies between the branches; the JSD of identical branches is")
    print(f"  zero and of these three is {apart.mean():.4f}")

    print("augmentation. colour jitter is four ordered factors:")
    rgb = _rgb(seed=8)
    quiet = Augmenter("color_jitter", seed=9, brightness=0.0, contrast=0.0,
                      saturation=0.0, hue=0.0)
    assert np.allclose(quiet(rgb)[0], rgb)
    grey = np.repeat(rgb.mean(axis=-1, keepdims=True), 3, axis=-1)
    jittered = Augmenter("color_jitter", seed=9, brightness=0.2,
                         contrast=0.2, saturation=0.5, hue=0.05)(grey)[0]
    spread = float(np.abs(jittered - jittered.mean(axis=-1, keepdims=True)).max())
    assert spread < 1e-12
    print("  zero magnitudes are the identity; a grey image stays grey under")
    print("  every factor, because the hue matrix has rows summing to one")
    print(f"  largest channel deviation on grey: {spread:.1e}")

    print("augmentation. autoaugment samples one policy per image:")
    aug = Augmenter("auto_augment", seed=10)
    changed, _, _ = aug(x)
    fraction = float(np.mean(np.any(changed != x, axis=(1, 2, 3))))
    assert aug.counts["policies"] == x.shape[0]
    assert fraction > 0.5 and changed.min() >= 0.0 and changed.max() <= 1.0
    print(f"  {aug.counts['policies']} policies drawn, {aug.counts['ops']} ops")
    print(f"  applied, {fraction:.0%} of the images changed, all inside [0, 1]")

    print("augmentation. translation supplies an invariance the data lacks:")
    train_x, train_y = _images(seed=11, per_class=3)
    test_x, test_y = _images(seed=12, per_class=50)
    flat_train = train_x.reshape(train_x.shape[0], -1)
    flat_test = test_x.reshape(test_x.shape[0], -1)
    onehot = _onehot(train_y, 2)
    recipes = (
        ("plain", None),
        ("mixup", lambda run: Augmenter("mixup", seed=30 + run, alpha=0.4)),
        ("cutmix", lambda run: Augmenter("cutmix", seed=30 + run, alpha=1.0)),
        ("shift", lambda run: Augmenter("randaugment", seed=30 + run, n=1,
                                        m=10.0, ops=("translate",))),
    )
    scores = {}
    for label, factory in recipes:
        runs = []
        for run in range(5):
            model = _MLP(flat_train.shape[1], 16, 2, seed=20 + run)
            aug = None if factory is None else factory(run)
            for _ in range(600):
                if aug is None:
                    model.step(flat_train, onehot, 0.3)
                else:
                    batch, target, _ = aug(train_x, train_y)
                    model.step(batch.reshape(batch.shape[0], -1), target, 0.3)
            runs.append(model.accuracy(flat_test, test_y))
        scores[label] = float(np.mean(runs))
    print(f"  six labelled images, {len(test_y)} held out, five seeds:")
    for label, score in scores.items():
        print(f"    {label:<7s} {score:.3f}")
    assert scores["shift"] > 0.95
    assert scores["shift"] > scores["plain"] + 0.4
    print("  six images cannot show every position, so the plain model")
    print("  memorises a few and scores at chance; translating each batch")
    print("  covers the shifts and the same model generalises")
    print("  mixing smooths the boundary but cannot invent that invariance")

    print("augmentation. malformed calls are refused:")
    guard = Augmenter("mixup", seed=14, classes=2)
    good_x, good_y = x[:4], y[:4]
    guard(good_x, good_y)
    bad = [
        lambda: Augmenter("mixup_v2"),
        lambda: Augmenter("cutout", channels=0),
        lambda: Augmenter("cutout", classes=1),
        lambda: Augmenter("mixup", alpha=0.0),
        lambda: Augmenter("cutout", coverage=0.0),
        lambda: Augmenter("cutout", coverage=1.0),
        lambda: Augmenter("random_erasing", p=1.5),
        lambda: Augmenter("random_erasing", erase_scale=(0.3, 0.1)),
        lambda: Augmenter("random_erasing", ratio=(0.0, 1.0)),
        lambda: Augmenter("random_erasing", fill="noise"),
        lambda: Augmenter("randaugment", n=0),
        lambda: Augmenter("randaugment", ops=()),
        lambda: Augmenter("randaugment", ops=("sharpen",)),
        lambda: Augmenter("randaugment", m=-1.0),
        lambda: Augmenter("augmix", k=0),
        lambda: Augmenter("augmix", depth=0),
        lambda: Augmenter("color_jitter", channels=1),
        lambda: Augmenter("color_jitter", hue=1.5),
        lambda: guard(good_x[0], good_y),
        lambda: guard(good_x[:, :, :, :2], good_y),
        lambda: guard(np.zeros((0, 8, 8, 3)), y[:0]),
        lambda: guard(np.full((2, 8, 8, 3), np.nan), y[:2]),
        lambda: guard(-good_x, good_y),
        lambda: guard(good_x, good_y[:2]),
        lambda: guard(good_x, good_y.astype(float)),
        lambda: guard(good_x, good_y + 5),
        lambda: guard(good_x),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero channel")
    print("  count, one class, a non-positive alpha, an impossible coverage,")
    print("  a probability outside [0, 1], an inverted interval, a bad fill,")
    print("  an empty or unknown op set, a zero op count, a negative")
    print("  severity, a two-channel colour jitter, a one-image input,")
    print("  wrong channels, an empty or non-finite batch, out-of-range")
    print("  pixels, wrong labels, float labels, a label outside the class")
    print("  range, and mixing without labels")
    print("augmentation demo passed")


if __name__ == "__main__":
    demo()
