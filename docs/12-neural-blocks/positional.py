"""Positional encoding — how a bag of tokens learns about order.

Chapter:   12-neural-blocks
Companion: positional.md
Tier 2 · 8 variants: sinusoidal, learned, rope, rope_ntk, alibi, relative_bias,
t5_bias, no_position
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("sinusoidal", "learned", "rope", "rope_ntk", "alibi",
            "relative_bias", "t5_bias", "no_position")

ROTARY = ("rope", "rope_ntk")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _frequencies(dim: int, base: float) -> np.ndarray:
    """The d/2 inverse wavelengths of the sinusoidal and rotary tables."""
    return base ** (-np.arange(0, dim, 2) / dim)


def _rotations(length: int, dim: int, base: float) -> np.ndarray:
    """Angle of every position, one column per rotation plane."""
    return np.arange(length)[:, None] * _frequencies(dim, base)[None, :]


def _rotate(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate consecutive coordinate pairs by the given per-position angles."""
    shape = x.shape
    pairs = x.reshape(shape[:-1] + (shape[-1] // 2, 2))
    cosine = np.cos(angles)[None]
    sine = np.sin(angles)[None]
    out = np.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cosine - pairs[..., 1] * sine
    out[..., 1] = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return out.reshape(shape)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _buckets(signed: np.ndarray, buckets: int, max_distance: int) -> np.ndarray:
    """T5's relative-position buckets, split by sign and log-spaced far away."""
    half = buckets // 2
    exact = half // 2
    distance = np.minimum(np.abs(signed), max_distance)
    index = np.zeros_like(distance)
    small = distance < exact
    index[small] = distance[small]
    large = ~small
    index[large] = exact + (np.log(distance[large] / exact)
                            / np.log(max_distance / exact) * (half - exact))
    index = np.minimum(index.astype(int), half - 1)
    return index + (signed < 0) * half


# --------------------------------------------------------------------------- #
# The encoding
# --------------------------------------------------------------------------- #
class Positions:
    """Every positional scheme behind one interface.

    `table(length)` is an additive table to add to token features,
    `rotate(x, length)` is applied to queries and keys after their
    projection, and `bias(length)` is added to the attention scores.
    Variants use one of the three; the others return zeros.
    """

    def __init__(self, variant: str = "sinusoidal", *, dim: int = 8,
                 heads: int = 2, max_positions: int = 32, base: float = 1e4,
                 scale: float = 8.0, buckets: int = 8,
                 max_distance: int = 16, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if dim < 2 or dim % 2:
            raise ValueError("dim must be even and at least two")
        if heads < 1:
            raise ValueError("heads must be positive")
        if max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not np.isfinite(base) or base <= 0.0:
            raise ValueError("base must be finite and positive")
        if not np.isfinite(scale) or scale < 1.0:
            raise ValueError("scale must be finite and at least one")
        if buckets < 4 or buckets % 2:
            raise ValueError("buckets must be even and at least four")
        if max_distance <= buckets // 4:
            raise ValueError("max_distance must exceed the exact buckets")
        self.variant = variant
        self.dim = dim
        self.heads = heads
        self.max_positions = max_positions
        self.base = base
        self.scale = scale
        self.buckets = buckets
        self.max_distance = max_distance
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.parameters: dict[str, np.ndarray] = {}
        if variant == "learned":
            self.parameters["table"] = rng.normal(
                0.0, 0.02, (max_positions, dim))
        if variant == "relative_bias":
            self.parameters["bias"] = rng.normal(
                0.0, 0.02, (heads, 2 * max_distance + 1))
        if variant == "t5_bias":
            self.parameters["bias"] = rng.normal(0.0, 0.02, (buckets,))
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    # -- the three channels -------------------------------------------------- #
    def table(self, length: int) -> np.ndarray:
        if length < 1:
            raise ValueError("length must be positive")
        if self.variant == "sinusoidal":
            angles = _rotations(length, self.dim, self.base)
            table = np.zeros((length, self.dim))
            table[:, 0::2] = np.sin(angles)
            table[:, 1::2] = np.cos(angles)
            return table
        if self.variant == "learned":
            if length > self.max_positions:
                raise ValueError("length exceeds max_positions")
            return self.parameters["table"][:length]
        return np.zeros((length, self.dim))

    def rotate(self, x: np.ndarray, length: int | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.dim:
            raise ValueError("the last axis must equal dim")
        steps = x.shape[-2] if x.ndim >= 2 else 1
        if length is not None and length != steps:
            raise ValueError("length does not match the sequence axis")
        if steps < 1:
            raise ValueError("length must be positive")
        if self.variant not in ROTARY:
            return x
        base = self.base if self.variant == "rope" else self._ntk_base()
        return _rotate(x, _rotations(steps, self.dim, base))

    def bias(self, length: int) -> np.ndarray:
        if length < 1:
            raise ValueError("length must be positive")
        if self.variant == "alibi":
            slopes = 2.0 ** (-8.0 * np.arange(1, self.heads + 1)
                             / self.heads)
            distance = np.abs(np.arange(length)[:, None]
                              - np.arange(length)[None, :])
            return -slopes[:, None, None] * distance[None]
        if self.variant == "relative_bias":
            table = self.parameters["bias"]
            signed = np.arange(length)[:, None] - np.arange(length)[None, :]
            signed = np.clip(signed, -self.max_distance, self.max_distance)
            return table[:, signed + self.max_distance]
        if self.variant == "t5_bias":
            signed = np.arange(length)[:, None] - np.arange(length)[None, :]
            index = _buckets(signed, self.buckets, self.max_distance)
            rows = self.parameters["bias"][index]
            return np.broadcast_to(rows, (self.heads, length, length))
        return np.zeros((self.heads, length, length))

    # -- helpers ------------------------------------------------------------- #
    def _ntk_base(self) -> float:
        """The NTK-aware base stretch for a context-extension factor."""
        return self.base * self.scale ** (self.dim / (self.dim - 2))

# --------------------------------------------------------------------------- #
# A single attention readout, used to test what survives the encoding
# --------------------------------------------------------------------------- #
def _projections(dim: int, seed: int):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(dim)
    return (rng.normal(0.0, scale, (dim, dim)),
            rng.normal(0.0, scale, (dim, dim)),
            rng.normal(0.0, scale, (dim, dim)))


def _readout(pos: Positions, x: np.ndarray, weights) -> np.ndarray:
    """Attention at position zero over a sequence, with the encoding applied."""
    wq, wk, wv = weights
    length = x.shape[0]
    hidden = x + pos.table(length)
    query = pos.rotate(hidden @ wq)
    key = pos.rotate(hidden @ wk)
    value = hidden @ wv
    scores = query[0] @ key.T / np.sqrt(pos.dim)
    weights_ = _softmax(scores + pos.bias(length)[0, 0])
    return weights_ @ value


def _readout_backward(pos: Positions, x: np.ndarray, weights,
                      target: np.ndarray):
    """Loss, its gradient with respect to the learned table, and the loss."""
    wq, wk, wv = weights
    length, dim = x.shape
    hidden = x + pos.table(length)
    query = pos.rotate(hidden @ wq)
    key = pos.rotate(hidden @ wk)
    value = hidden @ wv
    scores = query[0] @ key.T / np.sqrt(dim)
    weights_ = _softmax(scores + pos.bias(length)[0, 0])
    out = weights_ @ value
    residual = out - target
    loss = 0.5 * float(np.sum(residual ** 2))
    d_scores = weights_ * (value @ residual
                           - float(weights_ @ (value @ residual)))
    d_query = d_scores @ key / np.sqrt(dim)
    d_key = d_scores[:, None] * query[0][None, :] / np.sqrt(dim)
    d_value = weights_[:, None] * residual[None, :]
    gradient = np.zeros_like(hidden)
    gradient[0] += d_query @ wq.T
    gradient += d_key @ wk.T
    gradient += d_value @ wv.T
    return loss, gradient


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _marker_pair(length: int, rng: np.random.Generator):
    """Two markers with values +1 and -1, in one order or the other.

    The marker rows are identical apart from the value, so exchanging the two
    values is exactly a permutation of two rows.
    """
    x = rng.normal(0.0, 0.05, (length, 8))
    places = rng.choice(np.arange(1, length), 2, replace=False)
    values = rng.permutation(np.array([-1.0, 1.0]))
    x[places] = 0.0
    x[places, 0] = 1.0
    x[places, 1] = values
    swapped = x.copy()
    swapped[places, 1] = values[::-1]
    return x, swapped


def demo() -> None:
    print("positional. the sinusoidal table is a sum of cosines:")
    pos = Positions("sinusoidal", dim=8, base=1e4)
    table = pos.table(16)
    theta = _frequencies(8, 1e4)
    gaps = np.arange(16)[:, None] - np.arange(16)[None, :]
    expected = np.cos(gaps[..., None] * theta).sum(axis=-1)
    assert np.allclose(table @ table.T, expected, atol=1e-12)
    norms = np.sqrt((table ** 2).sum(axis=1))
    assert np.allclose(norms, np.sqrt(4.0), atol=1e-12)
    assert np.allclose(table[0], [0.0, 1.0] * 4)
    print(f"  row 0 is {np.round(table[0], 3).tolist()}, every row has norm")
    print(f"  {norms[0]:.6f}, and the dot product of rows p and q is")
    print("  sum_i cos((p - q) * theta_i), which is stationary in p")

    print("positional. a learned table is a parameter, not a formula:")
    learned = Positions("learned", dim=8, max_positions=16, seed=3)
    before = learned.parameters["table"].copy()
    learned.table(5)[...] += 1.0
    assert np.allclose(learned.parameters["table"][:5], before[:5] + 1.0)
    assert np.allclose(learned.parameters["table"][5:], before[5:])
    learned.parameters["table"][...] = before
    rows = learned.table(16)
    gaps = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=-1)
    assert gaps[~np.eye(16, dtype=bool)].min() > 1e-6
    print("  writing through `table(k)` writes the first k rows of the")
    print("  parameter, and the remaining rows are untouched")

    print("positional. rotary attention depends only on the distance:")
    rope = Positions("rope", dim=8, base=1e4)
    rng = np.random.default_rng(2)
    query = rng.normal(size=8)
    key = rng.normal(size=8)
    length = 16
    rotated_query = rope.rotate(np.tile(query, (length, 1)))
    rotated_key = rope.rotate(np.tile(key, (length, 1)))
    for gap in (1, 3, 7):
        scores = [rotated_query[p] @ rotated_key[p + gap]
                  for p in range(length - gap)]
        assert np.allclose(scores, scores[0], atol=1e-12)
    assert np.allclose(np.linalg.norm(rotated_query, axis=1),
                       np.linalg.norm(query))
    angles = _rotations(8, 8, 1e4)
    reference = query @ _rotate(key, angles[4])
    assert abs(rotated_query[3] @ rotated_key[7] - reference) < 1e-12
    print("  q at 3 dotted with k at 7 equals q at 0 with k at 4:")
    print(f"  {float(rotated_query[3] @ rotated_key[7]):.12f}, and rotation")
    print("  preserves the norm of every row")

    print("positional. NTK scaling slows the fast dimensions only:")
    ntk = Positions("rope_ntk", dim=8, base=1e4, scale=8.0)
    ratio = ntk._ntk_base() ** (-np.arange(0, 8, 2) / 8) / theta
    assert abs(ratio[0] - 1.0) < 1e-12
    assert abs(ratio[-1] - 1.0 / 8.0) < 1e-12
    assert np.all(np.diff(ratio) < 0.0)
    print(f"  base {ntk.base:g} becomes {ntk._ntk_base():.0f}, and the")
    print("  frequency ratios are "
          + ", ".join(f"{value:.3f}" for value in ratio))
    print("  so the longest wave is unchanged and the fastest is stretched")

    print("positional. ALiBi is a Toeplitz distance penalty:")
    alibi = Positions("alibi", dim=8, heads=4)
    bias = alibi.bias(12)
    slopes = 2.0 ** (-8.0 * np.arange(1, 5) / 4)
    distance = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
    assert np.allclose(bias, -slopes[:, None, None] * distance[None])
    assert np.allclose(slopes[1:] / slopes[:-1], 2.0 ** (-8.0 / 4))
    assert bias[0, 0, 0] == 0.0 and bias[0, 0, 5] < 0.0
    print(f"  slopes {np.round(slopes, 4).tolist()}, ratio "
          f"{float(slopes[1] / slopes[0]):.4f} per head, and the first row")
    print(f"  is {(np.round(bias[0, 0, :6], 3) + 0.0).tolist()}")

    print("positional. a learned relative bias is constant on diagonals:")
    relative = Positions("relative_bias", dim=8, heads=3, max_distance=4)
    bias = relative.bias(10)
    for gap in range(-4, 5):
        entries = [bias[:, row, row + gap] for row in range(10)
                   if 0 <= row + gap < 10]
        assert np.allclose(entries, entries[0])
    assert bias.shape == (3, 10, 10)
    assert relative.parameters["bias"].shape == (3, 9)
    print(f"  {relative.parameters['bias'].size} parameters, one per head and")
    print("  relative distance in [-4, 4], clipped beyond that")

    print("positional. the T5 table buckets far distances together:")
    t5 = Positions("t5_bias", dim=8, heads=2, buckets=8, max_distance=16)
    signed = np.arange(12)[:, None] - np.arange(12)[None, :]
    index = _buckets(signed, 8, 16)
    forward = [int(index[row, 0]) for row in range(12)]
    backward = [int(index[0, row]) for row in range(12)]
    assert forward[:2] == [0, 1]
    assert forward[-1] == 3
    assert np.all(np.diff(forward) >= 0)
    assert backward[0] == 0 and backward[-1] == 7
    assert index.min() >= 0 and index.max() < 8
    assert np.allclose(t5.bias(12)[0], t5.bias(12)[1])
    print(f"  forward 0 to 11 use buckets {forward}")
    print(f"  backward 0 to 11 use {backward}; the heads share it")

    print("positional. no position is exactly permutation equivariant:")
    nothing = Positions("no_position", dim=8)
    assert not np.any(nothing.table(8))
    assert not np.any(nothing.bias(8))
    probe = rng.normal(size=(8, 8))
    assert np.allclose(nothing.rotate(probe), probe)
    print("  its table and bias are zero and `rotate` returns its input")

    print("positional. order sensitivity of one fixed attention readout:")
    numbers = _projections(8, 11)
    report = {}
    for variant in VARIANTS:
        pos = Positions(variant, dim=8, seed=4)
        rng = np.random.default_rng(9)
        deltas = []
        for _ in range(48):
            x, swapped = _marker_pair(8, rng)
            deltas.append(float(np.linalg.norm(
                _readout(pos, x, numbers) - _readout(pos, swapped, numbers))))
        report[variant] = (min(deltas), float(np.mean(deltas)),
                           sum(value < 1e-12 for value in deltas))
    assert report["no_position"][2] == 48
    assert report["t5_bias"][2] > 0
    for variant in ("sinusoidal", "learned", "rope", "rope_ntk", "alibi",
                    "relative_bias"):
        assert report[variant][2] == 0
    for variant in VARIANTS:
        low, mean, same = report[variant]
        print(f"    {variant:<14s} min {low:.3e}  mean {mean:.3e}"
              f"  unchanged {same:2d}")
    print("  swapping the two marker values leaves every `no_position` pair")
    print("  bit-identical: attention over a set cannot see which came first")
    print("  `t5_bias` also collides, but only when both distances fall in one")
    print("  bucket; the exact-distance tables never do")

    print("positional. the learned table follows its gradient:")
    pos = Positions("learned", dim=6, max_positions=10, seed=5)
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 0.3, (7, 6))
    weights = _projections(6, 7)
    values = x @ weights[2]
    target = 0.5 * (values[3] + values[4])
    loss, gradient = _readout_backward(pos, x, weights, target)
    numeric = np.zeros_like(pos.parameters["table"])
    for row in range(7):
        for column in range(6):
            old = pos.parameters["table"][row, column]
            pos.parameters["table"][row, column] = old + 1e-6
            up = _readout_backward(pos, x, weights, target)[0]
            pos.parameters["table"][row, column] = old - 1e-6
            down = _readout_backward(pos, x, weights, target)[0]
            pos.parameters["table"][row, column] = old
            numeric[row, column] = (up - down) / 2e-6
    error = float(np.abs(gradient[:7] - numeric[:7]).max())
    assert error < 1e-6
    assert not np.any(numeric[7:])
    start = loss
    for _ in range(200):
        loss, gradient = _readout_backward(pos, x, weights, target)
        pos.parameters["table"][:7] -= 0.3 * gradient[:7]
    end = loss
    assert end < 0.05 * start
    print(f"  the table gradient matches differences to {error:.1e}; rows")
    print("  past the sequence get zero, and 200 steps cut the loss from")
    print(f"  {start:.4f} to {end:.4f}")

    print("positional. malformed calls are refused:")
    bad = [
        lambda: Positions("sinusoid"),
        lambda: Positions("rope", dim=3),
        lambda: Positions("rope", dim=0),
        lambda: Positions("rope", heads=0),
        lambda: Positions("rope", max_positions=0),
        lambda: Positions("rope", base=0.0),
        lambda: Positions("rope", base=np.nan),
        lambda: Positions("rope", scale=0.5),
        lambda: Positions("rope", buckets=3),
        lambda: Positions("rope", buckets=2),
        lambda: Positions("t5_bias", buckets=8, max_distance=2),
        lambda: Positions("learned", max_positions=4).table(5),
        lambda: Positions("learned", max_positions=4).table(0),
        lambda: Positions("sinusoidal").table(0),
        lambda: Positions("sinusoidal").bias(0),
        lambda: Positions("sinusoidal").rotate(np.zeros((4, 3))),
        lambda: Positions("sinusoidal").rotate(np.zeros((4, 8)), length=3),
        lambda: Positions("sinusoidal").rotate(np.zeros((4, 8)), length=0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, an odd or empty")
    print("  width, zero heads or positions, a zero or non-finite base, a")
    print("  scale below one, an odd or tiny bucket count, a tiny maximum")
    print("  distance, a table longer than the learned parameter or empty,")
    print("  an empty bias, a rotation whose width is not the dimension, and")
    print("  a length that disagrees with the sequence axis")
    print("positional demo passed")


if __name__ == "__main__":
    demo()
