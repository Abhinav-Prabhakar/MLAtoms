"""Gradient checkpointing — trade compute for memory in a deep chain.

Chapter:   13-training
Companion: checkpointing.md
Tier 2 · 5 variants: full, selective, sqrt, offload, reversible
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("full", "selective", "sqrt", "offload", "reversible")


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
class Chain:
    """A residual chain of `layers` blocks with switchable checkpointing.

    `forward(x)` returns the output and a cache; `backward(gradient, cache)`
    returns the input gradient and fills `self.gradients`. `self.counts`
    records the memory ledger: device states stored, host states, peak states,
    layer evaluations recomputed, and states transferred.
    """

    def __init__(self, variant: str = "full", *, layers: int = 24,
                 dim: int = 8, segment: int = 4, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if layers < 1 or dim < 1 or segment < 1:
            raise ValueError("layers, dim and segment must be positive")
        if variant == "reversible" and (layers % 2 or dim % 2):
            raise ValueError("reversible needs even layers and dim")
        self.variant = variant
        self.layers = layers
        self.dim = dim
        self.segment = segment
        self.reversible = variant == "reversible"
        self.width = dim // 2 if self.reversible else dim
        rng = np.random.default_rng(seed)
        self.parameters = {
            "W": rng.normal(0.0, 0.6 / np.sqrt(self.width),
                            (layers, self.width, self.width)),
            "b": np.zeros((layers, self.width)),
        }
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}
        self.counts = self._zero_counts()

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @staticmethod
    def _zero_counts() -> dict[str, int]:
        return {"stored": 0, "host": 0, "peak": 0, "recomputed": 0,
                "transferred": 0}

    # -- validation ---------------------------------------------------------- #
    def _check(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must be (batch, dim)")
        if x.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        return x

    # -- one block ----------------------------------------------------------- #
    def _block(self, h: np.ndarray, layer: int):
        pre = h @ self.parameters["W"][layer] + self.parameters["b"][layer]
        return np.tanh(pre), pre

    # -- checkpoints --------------------------------------------------------- #
    def _bounds(self) -> list[int]:
        if self.variant == "full":
            return list(range(self.layers + 1))
        if self.variant == "offload":
            return list(range(self.layers + 1))
        step = (max(1, int(round(np.sqrt(self.layers))))
                if self.variant == "sqrt" else self.segment)
        bounds = list(range(0, self.layers + 1, step))
        if bounds[-1] != self.layers:
            bounds.append(self.layers)
        return bounds

    # -- forward ------------------------------------------------------------- #
    def forward(self, x):
        x = self._check(x)
        self.counts = self._zero_counts()
        if self.reversible:
            self.counts.update(stored=1, peak=1, recomputed=self.layers)
            width = self.width
            first = x[:, :width].copy()
            second = x[:, width:].copy()
            for layer in range(self.layers):
                if layer % 2 == 0:
                    first = first + self._block(second, layer)[0]
                else:
                    second = second + self._block(first, layer)[0]
            out = np.concatenate([first, second], axis=1)
            return out, {"x": x, "out": out}

        hidden = [x]
        current = x
        for layer in range(self.layers):
            current = current + self._block(current, layer)[0]
            hidden.append(current)
        bounds = self._bounds()
        states = {index: hidden[index] for index in bounds}
        largest = max(right - left for left, right in zip(bounds, bounds[1:],
                                                          strict=False))
        if self.variant == "offload":
            self.counts.update(stored=1, host=len(hidden), peak=1,
                               transferred=len(hidden))
        else:
            self.counts.update(stored=len(bounds),
                               peak=len(bounds) + largest - 1)
        cache = {"x": x, "bounds": bounds, "states": states}
        if self.variant == "full":
            cache["hidden"] = hidden
        if self.variant == "offload":
            cache["host"] = hidden
        return current, cache

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        expected = (cache["x"].shape[0], self.dim)
        if gradient.shape != expected:
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        if self.reversible:
            return self._reversible_backward(gradient, cache)
        bounds = cache["bounds"]
        adjoint = gradient
        for left, right in reversed(list(zip(bounds, bounds[1:],
                                            strict=False))):
            if self.variant == "full":
                segment = [cache["hidden"][index]
                           for index in range(left, right + 1)]
            elif self.variant == "offload":
                segment = [cache["host"][index]
                           for index in range(left, right + 1)]
            else:
                segment = [cache["states"][left]]
                current = cache["states"][left]
                for layer in range(left, right):
                    current = current + self._block(current, layer)[0]
                    segment.append(current)
                self.counts["recomputed"] += right - left
            for offset in range(right - left - 1, -1, -1):
                layer = left + offset
                hidden = segment[offset]
                tanh, _ = self._block(hidden, layer)
                dpre = adjoint * (1.0 - tanh ** 2)
                self.gradients["W"][layer] += hidden.T @ dpre
                self.gradients["b"][layer] += dpre.sum(axis=0)
                adjoint = adjoint + dpre @ self.parameters["W"][layer].T
        return adjoint

    def _reversible_backward(self, gradient, cache) -> np.ndarray:
        width = self.width
        first = cache["out"][:, :width].copy()
        second = cache["out"][:, width:].copy()
        d_first = gradient[:, :width].copy()
        d_second = gradient[:, width:].copy()
        for layer in reversed(range(self.layers)):
            if layer % 2 == 0:
                tanh, _ = self._block(second, layer)
                dpre = d_first * (1.0 - tanh ** 2)
                self.gradients["W"][layer] += second.T @ dpre
                self.gradients["b"][layer] += dpre.sum(axis=0)
                d_second = d_second + dpre @ self.parameters["W"][layer].T
                first = first - tanh
            else:
                tanh, _ = self._block(first, layer)
                dpre = d_second * (1.0 - tanh ** 2)
                self.gradients["W"][layer] += first.T @ dpre
                self.gradients["b"][layer] += dpre.sum(axis=0)
                d_first = d_first + dpre @ self.parameters["W"][layer].T
                second = second - tanh
        return np.concatenate([d_first, d_second], axis=1)

    # -- inversion ----------------------------------------------------------- #
    def invert(self, out) -> np.ndarray:
        """Recover the input of a reversible chain from its output."""
        if not self.reversible:
            raise ValueError("only the reversible variant is invertible")
        out = np.asarray(out, dtype=float)
        if out.shape[1] != self.dim:
            raise ValueError("out has the wrong width")
        width = self.width
        first = out[:, :width].copy()
        second = out[:, width:].copy()
        for layer in reversed(range(self.layers)):
            if layer % 2 == 0:
                first = first - self._block(second, layer)[0]
            else:
                second = second - self._block(first, layer)[0]
        return np.concatenate([first, second], axis=1)

    # -- convenience --------------------------------------------------------- #
    def loss(self, x, target) -> float:
        prediction, cache = self.forward(x)
        return 0.5 * float(np.mean((prediction - target) ** 2)), cache


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8))
    target = rng.normal(size=(8, 8))

    print("checkpointing. one chain, five storage strategies:")
    counts = {}
    for variant in VARIANTS:
        model = Chain(variant, layers=24, dim=8, segment=4, seed=2)
        prediction, cache = model.forward(x)
        assert prediction.shape == x.shape
        counts[variant] = dict(model.counts)
        row = model.counts
        print(f"    {variant:<11s} device {row['stored']:3d}"
              f"  host {row['host']:3d}  peak {row['peak']:3d}"
              f"  recompute {row['recomputed']:3d}"
              f"  transfer {row['transferred']:3d}")
    assert counts["full"]["stored"] == 25
    assert counts["sqrt"]["stored"] == 6
    assert counts["offload"]["host"] == 25
    assert counts["reversible"]["stored"] == 1
    print("  full keeps every one of the 24 intermediate states; sqrt keeps")
    print("  six boundaries and rebuilds each segment during the backward")

    print("checkpointing. every strategy produces the same gradient:")
    reference = None
    for variant in VARIANTS[:4]:
        model = Chain(variant, layers=24, dim=8, segment=4, seed=3)
        prediction, cache = model.forward(x)
        gradient = (prediction - target) / prediction.size
        d_x = model.backward(gradient, cache)
        block = {key: value.copy() for key, value in model.gradients.items()}
        if reference is None:
            reference = (d_x, block)
            continue
        assert np.array_equal(d_x, reference[0])
        assert all(np.array_equal(block[key], reference[1][key])
                   for key in block)
    print("  recomputation reruns exactly the same arithmetic, so the")
    print("  gradients are bit-identical, not merely close")

    print("checkpointing. sqrt minimises the peak checkpoint budget:")
    best = None
    for k in range(1, 25):
        peak = k + -(-24 // k)
        if best is None or peak < best[0]:
            best = (peak, k)
        if k in (1, 2, 4, 5, 6, 8, 12, 24):
            print(f"    {k:2d} segments -> peak {peak:2d} states")
    root = max(1, int(round(np.sqrt(24))))
    assert root + -(-24 // root) == best[0]
    print(f"  the minimum is {best[0]} states at k = 4 to 6, straddling")
    print("  sqrt(24) = 4.9, which is the square-root rule")

    print("checkpointing. the compute bill is one extra forward pass:")
    for variant in VARIANTS:
        model = Chain(variant, layers=24, dim=8, segment=4, seed=4)
        prediction, cache = model.forward(x)
        model.backward((prediction - target) / prediction.size, cache)
        print(f"    {variant:<11s} recomputed {model.counts['recomputed']:3d}"
              f"  of 24 layer evaluations")
    print("  every segment is rebuilt once, so the total is the depth and")
    print("  not the depth times the number of checkpoints")

    print("checkpointing. reversible layers store one state and invert:")
    model = Chain("reversible", layers=24, dim=8, seed=5)
    prediction, cache = model.forward(x)
    recovered = model.invert(prediction)
    error = float(np.abs(recovered - x).max())
    assert error < 1e-12
    print("  24 coupling layers recovered from the output alone, error")
    print(f"  {error:.1e}; the backward pass inverts as it goes, so the")
    print("  peak stays at one state instead of twenty-five")

    print("checkpointing. finite differences confirm the reversible pass:")
    small = rng.normal(size=(3, 4))
    goal = rng.normal(size=(3, 4))
    model = Chain("reversible", layers=6, dim=4, seed=6)
    prediction, cache = model.forward(small)
    d_x = model.backward((prediction - goal) / prediction.size, cache)
    analytic = {key: value.copy() for key, value in model.gradients.items()}
    worst = 0.0
    for key, value in model.parameters.items():
        for index in np.ndindex(value.shape):
            old = value[index]
            value[index] = old + 1e-6
            plus = model.loss(small, goal)[0]
            value[index] = old - 1e-6
            minus = model.loss(small, goal)[0]
            value[index] = old
            numeric = (plus - minus) / 2e-6
            worst = max(worst, abs(numeric - analytic[key][index]))
    for index in np.ndindex(small.shape):
        old = small[index]
        small[index] = old + 1e-6
        plus = model.loss(small, goal)[0]
        small[index] = old - 1e-6
        minus = model.loss(small, goal)[0]
        small[index] = old
        numeric = (plus - minus) / 2e-6
        worst = max(worst, abs(numeric - d_x[index]))
    assert worst < 1e-6
    print(f"  worst error {worst:.1e} over every parameter and the input")

    print("checkpointing. checkpointing is invisible to the optimizer:")
    goal = rng.normal(size=(8, 8))
    results = {}
    for variant in ("full", "sqrt", "reversible"):
        model = Chain(variant, layers=8, dim=8, segment=4, seed=7)
        start = model.loss(x, goal)[0]
        for _ in range(300):
            prediction, cache = model.forward(x)
            gradient = (prediction - goal) / prediction.size
            model.backward(gradient, cache)
            for key in model.gradients:
                model.parameters[key] -= 0.1 * model.gradients[key]
        results[variant] = model.loss(x, goal)[0]
        print(f"    {variant:<11s} {start:.3f} -> {results[variant]:.6f}")
    assert np.isclose(results["full"], results["sqrt"])
    print("  full and sqrt follow the same trajectory to the last digit,")
    print("  because the gradient does not know how it was stored")

    print("checkpointing. malformed calls are refused:")
    guard = Chain("full", layers=4, dim=8, seed=8)
    good = guard.forward(x)[1]
    bad = [
        lambda: Chain("segment"),
        lambda: Chain("full", layers=0),
        lambda: Chain("full", dim=0),
        lambda: Chain("selective", segment=0),
        lambda: Chain("reversible", layers=5, dim=8),
        lambda: Chain("reversible", layers=4, dim=7),
        lambda: guard.forward(x[0]),
        lambda: guard.forward(x[:, :7]),
        lambda: guard.forward(np.zeros((0, 8))),
        lambda: guard.forward(np.full_like(x, np.nan)),
        lambda: guard.backward(np.zeros((8, 7)), good),
        lambda: guard.invert(x),
        lambda: Chain("reversible", layers=4, dim=8).invert(x[:, :7]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown strategy, a zero size, an")
    print("  odd reversible shape, a one-dimensional input, the wrong")
    print("  width, an empty or non-finite batch, a wrong gradient shape,")
    print("  and inversion on a chain that does not have it")
    print("checkpointing demo passed")


if __name__ == "__main__":
    demo()
