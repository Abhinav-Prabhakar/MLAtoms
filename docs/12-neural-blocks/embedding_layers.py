"""Embedding layers — tables, compositions and the weights they share.

Chapter:   12-neural-blocks
Companion: embedding_layers.md
Tier 2 · 7 variants: token, position, type, factorized, weight_tying,
tied_embeddings, adapter_embedding
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("token", "position", "type", "factorized", "weight_tying",
            "tied_embeddings", "adapter_embedding")

POSITIONED = ("position", "type")

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
# The layer
# --------------------------------------------------------------------------- #
class Embedding:
    """Integer ids in, vectors out, with optional tables and tying.

    The token table is the only mandatory parameter. Position and type tables
    are added before the output; the factorized variant replaces the table by
    a product of two smaller ones; the adapter variant freezes the table and
    learns a bottleneck correction; the tying variants share one table with a
    second use.
    """

    def __init__(self, variant: str = "token", *, vocab: int = 20,
                 dim: int = 8, max_positions: int = 16, segments: int = 2,
                 rank: int = 3, adapter_rank: int = 2, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if vocab < 1 or dim < 1:
            raise ValueError("vocab and dim must be positive")
        if max_positions < 1 or segments < 1:
            raise ValueError("max_positions and segments must be positive")
        if rank < 1 or adapter_rank < 1:
            raise ValueError("rank and adapter_rank must be positive")
        self.variant = variant
        self.vocab = vocab
        self.dim = dim
        self.max_positions = max_positions
        self.segments = segments
        self.rank = rank
        self.adapter_rank = adapter_rank
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.parameters: dict[str, np.ndarray] = {}
        if variant == "factorized":
            self.parameters["token.left"] = rng.normal(
                0.0, 1.0 / np.sqrt(dim), (vocab, rank))
            self.parameters["token.right"] = rng.normal(
                0.0, 1.0 / np.sqrt(rank), (rank, dim))
        else:
            self.parameters["token.table"] = rng.normal(
                0.0, 1.0 / np.sqrt(dim), (vocab, dim))
        if variant in POSITIONED:
            self.parameters["position.table"] = rng.normal(
                0.0, 0.02, (max_positions, dim))
        if variant == "type":
            self.parameters["type.table"] = rng.normal(
                0.0, 0.02, (segments, dim))
        if variant == "adapter_embedding":
            self.parameters["adapter.down"] = rng.normal(
                0.0, 1.0 / np.sqrt(dim), (dim, adapter_rank))
            self.parameters["adapter.up"] = rng.normal(
                0.0, 1.0 / np.sqrt(adapter_rank), (adapter_rank, dim))
        self.frozen = {"token.table"} if variant == "adapter_embedding" else set()
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()
                          if key not in self.frozen}

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @property
    def n_trainable(self) -> int:
        return int(sum(value.size for key, value in self.parameters.items()
                       if key not in self.frozen))

    # -- validation ---------------------------------------------------------- #
    def _check_ids(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu":
            raise ValueError("ids must be integers")
        if ids.ndim < 1 or ids.size == 0:
            raise ValueError("ids must be a non-empty array")
        if np.any(ids < 0) or np.any(ids >= self.vocab):
            raise ValueError("an id is out of vocabulary")
        return ids.astype(int)

    def _check_indices(self, values, limit: int, name: str,
                       shape: tuple[int, ...]) -> np.ndarray:
        values = np.asarray(values)
        if values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be integers")
        try:
            broadcast = np.broadcast_to(values, shape)
        except ValueError as error:
            raise ValueError(f"{name} does not broadcast to the ids") from error
        if np.any(broadcast < 0) or np.any(broadcast >= limit):
            raise ValueError(f"{name} out of range")
        return values

    # -- forward ------------------------------------------------------------- #
    def _lookup(self, ids: np.ndarray):
        p = self.parameters
        if self.variant == "factorized":
            out = p["token.left"][ids] @ p["token.right"]
            return out, {"ids": ids, "kind": "factorized"}
        if self.variant == "adapter_embedding":
            base = p["token.table"][ids]
            pre = base @ p["adapter.down"]
            hidden = _gelu(pre)
            out = base + hidden @ p["adapter.up"]
            return out, {"ids": ids, "kind": "adapter", "base": base,
                         "pre": pre, "hidden": hidden}
        return p["token.table"][ids], {"ids": ids, "kind": "table"}

    def forward(self, ids: np.ndarray, *, positions=None, segments=None,
                stream: int = 0):
        ids = self._check_ids(ids)
        if stream not in (0, 1) or (stream != 0
                                    and self.variant != "tied_embeddings"):
            raise ValueError("only tied_embeddings has a second stream")
        if positions is not None and self.variant not in POSITIONED:
            raise ValueError("this variant has no position table")
        if segments is not None and self.variant != "type":
            raise ValueError("this variant has no type table")
        if positions is None and self.variant in POSITIONED:
            positions = np.arange(ids.shape[-1])
        if positions is not None:
            positions = self._check_indices(
                positions, self.max_positions, "positions", ids.shape)
        if segments is None and self.variant == "type":
            segments = np.zeros(ids.shape[-1], dtype=int)
        if segments is not None:
            segments = self._check_indices(
                segments, self.segments, "segments", ids.shape)
        output, cache = self._lookup(ids)
        if positions is not None:
            output = output + self.parameters["position.table"][positions]
        if segments is not None:
            output = output + self.parameters["type.table"][segments]
        cache.update(positions=positions, segments=segments, stream=stream)
        return output, cache

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache) -> None:
        g = self.gradients
        for key in g:
            g[key][...] = 0.0
        gradient = np.asarray(gradient, dtype=float)
        ids = cache["ids"]
        if gradient.shape != ids.shape + (self.dim,):
            raise ValueError("gradient must have the shape of the output")
        flat = gradient.reshape(-1, self.dim)
        if cache["kind"] == "factorized":
            p = self.parameters
            np.add.at(g["token.left"], ids.ravel(),
                      flat @ p["token.right"].T)
            g["token.right"] += p["token.left"][ids.ravel()].T @ flat
        elif cache["kind"] == "adapter":
            p = self.parameters
            pre = cache["pre"].reshape(-1, self.adapter_rank)
            hidden = cache["hidden"].reshape(-1, self.adapter_rank)
            base = cache["base"].reshape(-1, self.dim)
            d_pre = (flat @ p["adapter.up"].T) * _gelu_grad(pre)
            g["adapter.down"] += base.T @ d_pre
            g["adapter.up"] += hidden.T @ flat
        else:
            np.add.at(g["token.table"], ids.ravel(), flat)
        if cache.get("positions") is not None:
            positions = np.broadcast_to(cache["positions"], ids.shape)
            np.add.at(g["position.table"], positions.ravel(), flat)
        if cache.get("segments") is not None:
            segments = np.broadcast_to(cache["segments"], ids.shape)
            np.add.at(g["type.table"], segments.ravel(), flat)

    # -- the output side of tying -------------------------------------------- #
    def logits(self, hidden: np.ndarray) -> np.ndarray:
        if self.variant != "weight_tying":
            raise ValueError("only weight_tying exposes tied logits")
        hidden = np.asarray(hidden, dtype=float)
        if hidden.shape[-1] != self.dim:
            raise ValueError("the hidden width must equal dim")
        return hidden @ self.parameters["token.table"].T


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    ids = np.array([[1, 2, 1], [3, 0, 1]])

    print("embedding. a token table is a gather:")
    model = Embedding("token", vocab=10, dim=4, seed=1)
    out, cache = model.forward(ids)
    table = model.parameters["token.table"]
    assert out.shape == (2, 3, 4)
    assert np.allclose(out, table[ids])
    assert model.n_parameters == 40
    model.backward(np.ones_like(out), cache)
    counts = np.bincount(ids.ravel(), minlength=10)
    assert np.allclose(model.gradients["token.table"],
                       counts[:, None] * np.ones((1, 4)))
    print(f"  {model.n_parameters} parameters; a backward pass scatters the")
    print(f"  rows, so id 1 receives {int(counts[1])} counts and id 9 none")

    print("embedding. a position table is added to the token table:")
    model = Embedding("position", vocab=10, dim=4, max_positions=6, seed=2)
    out, _ = model.forward(ids)
    table = model.parameters["token.table"]
    position = model.parameters["position.table"]
    assert np.allclose(out, table[ids] + position[np.arange(3)])
    shifted, _ = model.forward(ids, positions=np.array([2, 3, 5]))
    assert np.allclose(shifted - out,
                       position[[2, 3, 5]] - position[[0, 1, 2]])
    assert model.n_parameters == 40 + 24
    print(f"  {model.n_parameters} parameters, and moving the positions by two")
    print("  adds exactly the difference of the two position rows")

    print("embedding. type adds a third table for the segment:")
    model = Embedding("type", vocab=10, dim=4, max_positions=6,
                      segments=2, seed=3)
    segments = np.array([[0, 0, 0], [1, 1, 1]])
    out, _ = model.forward(ids, segments=segments)
    base = model.parameters["token.table"][ids]
    base = base + model.parameters["position.table"][np.arange(3)]
    types = model.parameters["type.table"]
    assert np.allclose(out[0], base[0] + types[0])
    assert np.allclose(out[1], base[1] + types[1])
    same_a, _ = model.forward(ids, segments=np.zeros((2, 3), dtype=int))
    same_b, _ = model.forward(ids, segments=np.ones((2, 3), dtype=int))
    assert np.allclose(same_b - same_a, types[1] - types[0])
    assert model.n_parameters == 40 + 24 + 8
    print(f"  {model.n_parameters} parameters: token plus position plus two")
    print("  segment vectors, and changing only the segment shifts every token")

    print("embedding. factorizing a table trades rank for parameters:")
    model = Embedding("factorized", vocab=10, dim=4, rank=2, seed=4)
    out, cache = model.forward(ids)
    left = model.parameters["token.left"]
    right = model.parameters["token.right"]
    assert np.allclose(out, left[ids] @ right)
    assert model.n_parameters == 10 * 2 + 2 * 4
    assert model.n_parameters < 40
    target = rng.normal(size=out.shape)
    model.backward(target, cache)
    analytic = {key: value.copy() for key, value in model.gradients.items()}
    worst = 0.0
    for key, value in model.parameters.items():
        for index in np.ndindex(value.shape):
            old = value[index]
            value[index] = old + 1e-6
            plus = model.forward(ids)[0]
            value[index] = old - 1e-6
            minus = model.forward(ids)[0]
            value[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - analytic[key][index]))
    assert worst < 1e-6
    print(f"  {model.n_parameters} parameters instead of 40; both factors")
    print(f"  match finite differences to {worst:.1e}")

    print("embedding. weight tying reuses the table as the output matrix:")
    model = Embedding("weight_tying", vocab=10, dim=4, seed=5)
    out, _ = model.forward(ids)
    hidden = rng.normal(size=(3, 4))
    logits = model.logits(hidden)
    table = model.parameters["token.table"]
    assert np.allclose(logits, hidden @ table.T)
    assert model.n_parameters == 40
    before = logits.copy()
    table[2] += 1.0
    after = model.logits(hidden)
    # the column for token 2 moves, no other column does
    assert np.allclose(after[:, 2] - before[:, 2],
                       hidden @ np.ones(4))
    assert np.allclose(np.delete(after - before, 2, axis=1), 0.0)
    print(f"  {model.n_parameters} parameters; a separate output matrix would")
    print("  add 40, and editing one table row moves one logit column")

    print("embedding. tied embeddings share one table across two streams:")
    model = Embedding("tied_embeddings", vocab=10, dim=4, seed=6)
    first, first_cache = model.forward(ids, stream=0)
    second, second_cache = model.forward(ids, stream=1)
    assert np.allclose(first, second)
    assert model.n_parameters == 40
    model.backward(np.ones_like(first), first_cache)
    first_gradient = model.gradients["token.table"].copy()
    model.backward(np.ones_like(second), second_cache)
    assert np.allclose(model.gradients["token.table"], first_gradient)
    print(f"  {model.n_parameters} parameters instead of 80; both streams")
    print("  read and write the one table")

    print("embedding. an adapter learns a correction to a frozen table:")
    model = Embedding("adapter_embedding", vocab=10, dim=4, adapter_rank=2,
                      seed=7)
    out, cache = model.forward(ids)
    base = model.parameters["token.table"][ids]
    pre = base @ model.parameters["adapter.down"]
    delta = _gelu(pre) @ model.parameters["adapter.up"]
    assert np.allclose(out, base + delta)
    assert model.n_parameters == 40 + 8 + 8
    assert model.n_trainable == 16
    target = 0.5 * base
    model.backward(out - target, cache)
    analytic = {key: value.copy() for key, value in model.gradients.items()}
    worst = 0.0
    for key, value in analytic.items():
        parameter = model.parameters[key]
        for index in np.ndindex(value.shape):
            old = parameter[index]
            parameter[index] = old + 1e-6
            plus = model.forward(ids)[0]
            parameter[index] = old - 1e-6
            minus = model.forward(ids)[0]
            parameter[index] = old
            numeric = float(np.sum((plus - minus) * (out - target))) / 2e-6
            worst = max(worst, abs(numeric - analytic[key][index]))
    assert worst < 1e-7
    frozen = model.parameters["token.table"].copy()
    start = 0.5 * float(np.sum((out - target) ** 2))
    for _ in range(300):
        out, cache = model.forward(ids)
        residual = out - target
        model.backward(residual, cache)
        for key in model.gradients:
            model.parameters[key] -= 0.2 * model.gradients[key]
    end = 0.5 * float(np.sum((out - target) ** 2))
    assert end < 0.1 * start
    assert np.array_equal(model.parameters["token.table"], frozen)
    print(f"  {model.n_trainable} trainable of {model.n_parameters}; the loss")
    print(f"  falls from {start:.4f} to {end:.4f} and the base table is untouched")

    print("embedding. parameter counts at a realistic vocabulary:")
    for variant in VARIANTS:
        model = Embedding(variant, vocab=1000, dim=16, max_positions=32,
                          segments=2, rank=4, adapter_rank=8)
        print(f"    {variant:<18s} {model.n_parameters:6d} total"
              f"  {model.n_trainable:6d} trainable")
    assert (Embedding("factorized", vocab=1000, dim=16,
                      rank=4).n_parameters == 4064)
    assert (Embedding("position", vocab=1000, dim=16,
                      max_positions=32).n_parameters == 16512)

    print("embedding. malformed calls are refused:")
    guard = Embedding("token", vocab=10, dim=4, seed=8)
    bad = [
        lambda: Embedding("lookup"),
        lambda: Embedding("token", vocab=0),
        lambda: Embedding("token", dim=0),
        lambda: Embedding("position", max_positions=0),
        lambda: Embedding("type", segments=0),
        lambda: Embedding("factorized", rank=0),
        lambda: Embedding("adapter_embedding", adapter_rank=0),
        lambda: guard.forward(np.array([[1.0, 2.0]])),
        lambda: guard.forward(np.array([[-1, 2]])),
        lambda: guard.forward(np.array([[10, 2]])),
        lambda: Embedding("token").forward(np.array([], dtype=int)),
        lambda: Embedding("position", vocab=10).forward(
            ids, positions=np.zeros((2, 2), dtype=int)),
        lambda: Embedding("position", vocab=10, max_positions=3).forward(
            ids, positions=np.array([0, 1, 9])),
        lambda: Embedding("position", vocab=10).forward(
            ids, segments=np.zeros((2, 3), dtype=int)),
        lambda: Embedding("type", vocab=10).forward(
            ids, segments=np.array([[0, 0, 2], [0, 0, 0]])),
        lambda: guard.forward(ids, stream=1),
        lambda: guard.logits(np.zeros((2, 4))),
        lambda: guard.backward(np.zeros((2, 3, 5)), {"ids": ids,
                                                     "kind": "table"}),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, empty sizes, a")
    print("  fractional, negative, too-large or empty id, a position table")
    print("  on a token lookup, a position of the wrong shape or out of range,")
    print("  a segment on a position variant or out of range, a second stream")
    print("  without tying, tied logits without weight tying, and a wrong")
    print("  gradient shape")
    print("embedding demo passed")


if __name__ == "__main__":
    demo()
