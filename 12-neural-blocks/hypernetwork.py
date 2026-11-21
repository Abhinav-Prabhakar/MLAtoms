"""Hypernetworks — weights that are functions of a conditioning vector.

Chapter:   12-neural-blocks
Companion: hypernetwork.md
Tier 2 · 4 variants: hypernet, lora_hypernet, conditioned_weights,
weight_generation
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("hypernet", "lora_hypernet", "conditioned_weights",
            "weight_generation")

_GELU = 0.7978845608028654
_CUBIC = 0.044715


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _gelu(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    tanh = np.tanh(inner)
    return (0.5 * (1.0 + tanh)
            + 0.5 * x * (1.0 - tanh ** 2)
            * _GELU * (1.0 + 3.0 * _CUBIC * x ** 2))


# --------------------------------------------------------------------------- #
# The hypernetwork
# --------------------------------------------------------------------------- #
class HyperNetwork:
    """A small network that emits the parameters used by a target network.

    `forward(x, c)` conditions on `c` and evaluates the target on `x`. Each
    batch element gets its own generated weights; the demo checks that a batch
    is exactly a loop over its conditions.
    """

    def __init__(self, variant: str = "hypernet", *, in_dim: int = 4,
                 out_dim: int = 3, cond_dim: int = 2, hidden: int = 6,
                 width: int = 5, rank: int = 2, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(in_dim, out_dim, cond_dim, hidden, width, rank) < 1:
            raise ValueError("all sizes must be positive")
        self.variant = variant
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.width = width
        self.rank = rank
        self.seed = seed
        rng = np.random.default_rng(seed)
        if variant == "hypernet":
            emitted = in_dim * out_dim + out_dim
        elif variant == "weight_generation":
            emitted = (in_dim * width + width) + (width * out_dim + out_dim)
        elif variant == "lora_hypernet":
            emitted = in_dim * rank + out_dim * rank + out_dim
        else:
            emitted = 2 * out_dim
        self.emitted = emitted
        p: dict[str, np.ndarray] = {}
        p["trunk.W"] = rng.normal(0.0, 1.0 / np.sqrt(cond_dim),
                                  (cond_dim, hidden))
        p["trunk.b"] = np.zeros(hidden)
        p["head.W"] = rng.normal(0.0, 1.0 / np.sqrt(hidden),
                                 (hidden, emitted))
        p["head.b"] = np.zeros(emitted)
        frozen = set()
        if variant in ("lora_hypernet", "conditioned_weights"):
            p["base.W"] = rng.normal(0.0, 1.0 / np.sqrt(in_dim),
                                     (out_dim, in_dim))
            frozen.add("base.W")
        self.parameters = p
        self.frozen = frozen
        self.gradients = {key: np.zeros_like(value)
                          for key, value in p.items()
                          if key not in frozen}

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @property
    def n_trainable(self) -> int:
        return int(sum(value.size for key, value in self.parameters.items()
                       if key not in self.frozen))

    # -- validation ---------------------------------------------------------- #
    def _check(self, x: np.ndarray, c: np.ndarray):
        x = np.asarray(x, dtype=float)
        c = np.asarray(c, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError("x must be (batch, in_dim)")
        if c.ndim != 2 or c.shape[1] != self.cond_dim:
            raise ValueError("c must be (batch, cond_dim)")
        if x.shape[0] != c.shape[0]:
            raise ValueError("x and c must have the same batch size")
        if x.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)):
            raise ValueError("inputs must be finite")
        return x, c

    # -- forward ------------------------------------------------------------- #
    def _trunk(self, c: np.ndarray):
        p = self.parameters
        pre = c @ p["trunk.W"] + p["trunk.b"]
        hidden = _gelu(pre)
        return hidden, {"c": c, "pre": pre, "hidden": hidden}

    def forward(self, x: np.ndarray, c: np.ndarray):
        x, c = self._check(x, c)
        batch = x.shape[0]
        trunk, trunk_cache = self._trunk(c)
        raw = trunk @ self.parameters["head.W"] + self.parameters["head.b"]
        p = self.parameters
        cache: dict = {"x": x, "trunk": trunk_cache, "raw": raw}
        if self.variant == "hypernet":
            split = self.in_dim * self.out_dim
            weight = raw[:, :split].reshape(batch, self.out_dim, self.in_dim)
            bias = raw[:, split:]
            out = np.einsum("bi,boi->bo", x, weight) + bias
            cache.update(weight=weight, bias=bias)
            return out, cache

        if self.variant == "weight_generation":
            one = self.in_dim * self.width
            two = one + self.width
            three = two + self.width * self.out_dim
            first = raw[:, :one].reshape(batch, self.width, self.in_dim)
            first_bias = raw[:, one:two]
            second = raw[:, two:three].reshape(batch, self.out_dim,
                                               self.width)
            second_bias = raw[:, three:]
            pre_hidden = np.einsum("bi,bwi->bw", x, first) + first_bias
            hidden = _gelu(pre_hidden)
            out = np.einsum("bw,bow->bo", hidden, second) + second_bias
            cache.update(first=first, first_bias=first_bias,
                         second=second, second_bias=second_bias,
                         pre_hidden=pre_hidden, hidden=hidden)
            return out, cache

        if self.variant == "lora_hypernet":
            one = self.in_dim * self.rank
            two = one + self.out_dim * self.rank
            down = raw[:, :one].reshape(batch, self.in_dim, self.rank)
            up = raw[:, one:two].reshape(batch, self.out_dim, self.rank)
            bias = raw[:, two:]
            base = x @ p["base.W"].T
            reduced = np.einsum("bi,bir->br", x, down)
            out = base + np.einsum("br,bor->bo", reduced, up) + bias
            cache.update(down=down, up=up, bias=bias, base=base,
                         reduced=reduced)
            return out, cache

        gamma = raw[:, :self.out_dim]
        beta = raw[:, self.out_dim:]
        base = x @ p["base.W"].T
        out = base * (1.0 + gamma) + beta
        cache.update(gamma=gamma, beta=beta, base=base)
        return out, cache

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache):
        gradient = np.asarray(gradient, dtype=float)
        x = cache["x"]
        if gradient.shape != (x.shape[0], self.out_dim):
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        p, g = self.parameters, self.gradients
        batch = x.shape[0]
        d_raw = np.zeros_like(cache["raw"])
        if self.variant == "hypernet":
            weight = cache["weight"]
            d_raw[:, :self.in_dim * self.out_dim] = np.einsum(
                "bo,bi->boi", gradient, x).reshape(batch, -1)
            d_raw[:, self.in_dim * self.out_dim:] = gradient
            d_x = np.einsum("bo,boi->bi", gradient, weight)
        elif self.variant == "weight_generation":
            second = cache["second"]
            hidden = cache["hidden"]
            d_hidden = np.einsum("bo,bow->bw", gradient, second)
            d_pre = d_hidden * _gelu_grad(cache["pre_hidden"])
            one = self.in_dim * self.width
            two = one + self.width
            three = two + self.width * self.out_dim
            d_raw[:, :one] = np.einsum("bw,bi->bwi", d_pre,
                                       x).reshape(batch, -1)
            d_raw[:, one:two] = d_pre
            d_raw[:, two:three] = np.einsum(
                "bo,bw->bow", gradient, hidden).reshape(batch, -1)
            d_raw[:, three:] = gradient
            d_x = np.einsum("bw,bwi->bi", d_pre, cache["first"])
        elif self.variant == "lora_hypernet":
            one = self.in_dim * self.rank
            two = one + self.out_dim * self.rank
            down, up = cache["down"], cache["up"]
            d_reduced = np.einsum("bo,bor->br", gradient, up)
            d_raw[:, :one] = np.einsum(
                "bi,br->bir", x, d_reduced).reshape(batch, -1)
            d_raw[:, one:two] = np.einsum(
                "bo,br->bor", gradient, cache["reduced"]).reshape(batch, -1)
            d_raw[:, two:] = gradient
            d_x = gradient @ p["base.W"]
            d_x += np.einsum("br,bir->bi", d_reduced, down)
        else:
            base = cache["base"]
            d_raw[:, :self.out_dim] = gradient * base
            d_raw[:, self.out_dim:] = gradient
            d_x = (gradient * (1.0 + cache["gamma"])) @ p["base.W"]
        g["head.W"] += cache["trunk"]["hidden"].T @ d_raw
        g["head.b"] += d_raw.sum(axis=0)
        d_hidden = d_raw @ p["head.W"].T
        trunk = cache["trunk"]
        d_pre = d_hidden * _gelu_grad(trunk["pre"])
        g["trunk.W"] += trunk["c"].T @ d_pre
        g["trunk.b"] += d_pre.sum(axis=0)
        d_c = d_pre @ p["trunk.W"].T
        return d_x, d_c


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4))
    c = rng.normal(size=(3, 2))

    print("hypernet. four ways to generate a target from a condition:")
    for variant in VARIANTS:
        model = HyperNetwork(variant, in_dim=4, out_dim=3, cond_dim=2,
                             hidden=6, width=5, rank=2, seed=2)
        out, _ = model.forward(x, c)
        assert out.shape == (3, 3)
        print(f"    {variant:<19s} {model.n_parameters:4d} total"
              f"  {model.n_trainable:4d} trainable")
    assert HyperNetwork("hypernet", in_dim=4, out_dim=3, cond_dim=2,
                        hidden=6).n_parameters == 123
    assert HyperNetwork("conditioned_weights", in_dim=4, out_dim=3,
                        cond_dim=2, hidden=6).n_parameters == 72
    print("  a full generated matrix costs 123 numbers; a modulation costs")
    print("  72, of which 12 are the frozen base")

    print("hypernet. the same input maps differently under two conditions:")
    model = HyperNetwork("hypernet", in_dim=4, out_dim=3, cond_dim=2,
                         hidden=6, seed=3)
    out, cache = model.forward(x, c)
    first = model.forward(x, c)[1]["weight"]
    other = model.forward(x, c + 1.0)[1]["weight"]
    assert not np.allclose(first, other)
    same = model.forward(x, c)[1]["weight"]
    assert np.allclose(first, same)
    print("  the generated 3x4 matrix is a function of the condition only;")
    print("  shifting c changes every entry")

    print("hypernet. each example gets its own weights:")
    conditions = rng.normal(size=(3, 2))
    model = HyperNetwork("hypernet", in_dim=4, out_dim=3, cond_dim=2,
                         hidden=6, seed=4)
    batched, _ = model.forward(x, conditions)
    looped = np.stack([model.forward(x[i:i + 1], conditions[i:i + 1])[0][0]
                       for i in range(3)])
    assert np.allclose(batched, looped)
    print("  a batch of three conditions equals three single-example runs,")
    print("  so the hypernetwork never shares a weight matrix across rows")

    print("hypernet. the LoRA update is low rank and the base is frozen:")
    model = HyperNetwork("lora_hypernet", in_dim=4, out_dim=3, cond_dim=2,
                         hidden=6, rank=2, seed=5)
    out, cache = model.forward(x, c)
    update = cache["up"] @ cache["down"].transpose(0, 2, 1)
    for row in range(3):
        singular = np.linalg.svd(update[row], compute_uv=False)
        rank = int((singular > 1e-9).sum())
        assert rank <= 2
    assert "base.W" not in model.gradients
    before = model.parameters["base.W"].copy()
    model.backward(np.ones_like(out), cache)
    for key in model.gradients:
        model.parameters[key] -= 0.01 * model.gradients[key]
    assert np.array_equal(model.parameters["base.W"], before)
    print(f"  rank 2 updates on a 3x4 matrix; {model.n_trainable} of")
    print(f"  {model.n_parameters} parameters train, and the base is untouched")

    print("hypernet. conditioned weights are exactly an affine modulation:")
    model = HyperNetwork("conditioned_weights", in_dim=4, out_dim=3,
                         cond_dim=2, hidden=6, seed=6)
    out, cache = model.forward(x, c)
    manual = cache["base"] * (1.0 + cache["gamma"]) + cache["beta"]
    assert np.allclose(out, manual)
    assert np.allclose(cache["base"], x @ model.parameters["base.W"].T)
    print("  scale and shift are generated per example, so one frozen matrix")
    print("  becomes a family of affine maps")

    print("hypernet. finite differences agree with the analytic gradients:")
    xs = rng.normal(size=(2, 3))
    cs = rng.normal(size=(2, 3))
    target = rng.normal(size=(2, 2))
    for variant in VARIANTS:
        model = HyperNetwork(variant, in_dim=3, out_dim=2, cond_dim=3,
                             hidden=4, width=3, rank=2, seed=7)
        out, cache = model.forward(xs, cs)
        d_x, d_c = model.backward(target, cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            if key in model.frozen:
                continue
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(xs, cs)[0]
                value[index] = old - 1e-6
                minus = model.forward(xs, cs)[0]
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        for index in np.ndindex(xs.shape):
            old = xs[index]
            xs[index] = old + 1e-6
            plus = model.forward(xs, cs)[0]
            xs[index] = old - 1e-6
            minus = model.forward(xs, cs)[0]
            xs[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_x[index]))
        for index in np.ndindex(cs.shape):
            old = cs[index]
            cs[index] = old + 1e-6
            plus = model.forward(xs, cs)[0]
            cs[index] = old - 1e-6
            minus = model.forward(xs, cs)[0]
            cs[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_c[index]))
        assert worst < 1e-6
        print(f"  {variant:<19s} worst error {worst:.1e}")

    print("hypernet. a short fit learns a family of linear maps:")
    model = HyperNetwork("hypernet", in_dim=2, out_dim=2, cond_dim=1,
                         hidden=8, seed=8)
    first = rng.normal(size=(2, 2))
    second = rng.normal(size=(2, 2))
    points = rng.normal(size=(3, 2))
    conditions = rng.normal(size=(3, 1))
    goal = np.einsum("bi,boi->bo", points, first + conditions[:, 0, None, None]
                     * second)

    def loss() -> float:
        return 0.5 * float(np.sum((model.forward(points, conditions)[0]
                                   - goal) ** 2))

    start = loss()
    for _ in range(400):
        out, cache = model.forward(points, conditions)
        d_x, d_c = model.backward(out - goal, cache)
        for key in model.gradients:
            model.parameters[key] -= 0.02 * model.gradients[key]
    end = loss()
    assert end < 0.05 * start
    print("  the target is x @ (A0 + c A1); 400 steps take the loss from")
    print(f"  {start:.3f} to {end:.4f}")

    print("hypernet. malformed calls are refused:")
    guard = HyperNetwork("hypernet", in_dim=4, out_dim=3, cond_dim=2,
                         hidden=6, seed=9)
    good, good_cache = guard.forward(x, c)
    bad = [
        lambda: HyperNetwork("hypernet_v2"),
        lambda: HyperNetwork("hypernet", in_dim=0),
        lambda: HyperNetwork("hypernet", out_dim=0),
        lambda: HyperNetwork("hypernet", cond_dim=0),
        lambda: HyperNetwork("hypernet", hidden=0),
        lambda: HyperNetwork("weight_generation", width=0),
        lambda: HyperNetwork("lora_hypernet", rank=0),
        lambda: guard.forward(x[0], c),
        lambda: guard.forward(x, c[0]),
        lambda: guard.forward(x, c[:2]),
        lambda: guard.forward(np.zeros((0, 4)), np.zeros((0, 2))),
        lambda: guard.forward(np.full_like(x, np.nan), c),
        lambda: guard.forward(x, np.full_like(c, np.nan)),
        lambda: guard.backward(np.zeros((3, 2)), good_cache),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, a")
    print("  one-dimensional input or condition, mismatched batch sizes, an")
    print("  empty batch, non-finite inputs, and a wrong gradient shape")
    assert good.shape == (3, 3)
    print("hypernet demo passed")


if __name__ == "__main__":
    demo()
