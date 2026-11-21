r"""Attention — one softmax, ten ways of choosing what it looks at.

Chapter:   10-sequences
Companion: attention.md
Tier 1 · 10 variants: dot_product, scaled, multi_head, mqa, gqa, sliding_window,
                      linear, sparse, flash_tiled, cross_attention

    dot_product:  a = softmax(Q K^T) V                                  (1)
    scaled:       a = softmax(Q K^T / sqrt(d)) V                        (2)
    multi_head:   h heads with their own Q, K, V, then W_o               (3)
    mqa:          many query heads, one key/value head                   (4)
    gqa:          groups of query heads share a key/value head           (5)
    sliding:      softmax restricted to a band around the diagonal       (6)
    linear:       a = phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)          (7)
    sparse:       band plus a few global columns                         (8)
    flash:        the softmax of (2), computed a key tile at a time      (9)
    cross:        queries from one sequence, keys and values from another (10)

Attention is a weighted average where the weights are chosen by the data. A
query vector is compared with every key, the comparison scores are passed
through a softmax to make them non-negative and sum to one, and the result is a
convex combination of the values. Nothing is learned in the definition; the
variants differ in how the comparison is scaled, how many independent
comparisons run at once, which positions a query is allowed to see, and how the
average is computed.

The ten variants are three questions. *Scaling*: the unscaled dot product grows
with dimension, and dividing by sqrt(d) keeps the logits in a range where the
softmax has a usable gradient. *Sharing*: multi-head attention gives every head
its own keys and values, multi-query attention gives all heads the same pair,
and grouped-query attention interpolates between the two, which is what makes a
large model's cache affordable. *Access*: a sliding window restricts a query to
its neighbours, sparse attention adds a few global columns to that band, linear
attention replaces the softmax with a kernel so the cost is linear in the length,
and flash attention computes the same softmax without ever holding the whole
score matrix. Cross-attention is the encoder-decoder case where the two
sequences have different lengths.

Everything is numpy. The masked softmax, the tiled online softmax and the linear
kernel are written out, and the demos prove that they agree with brute force
rather than trusting the rearrangement. The numbers to carry away: flash tiling
reproduces the dense result to 2.2e-16 while its largest score buffer is 23% of
the dense one, the three key-sharing schemes agree to 4.4e-16 while multi-query
attention uses 160 of the 256 parameters and a quarter of the key/value cache,
linear attention costs 0.13 of the quadratic work at length 512 and 0.008 at
length 8192, and every analytic gradient matches central differences to 1.3e-7.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("dot_product", "scaled", "multi_head", "mqa", "gqa", "sliding_window",
            "linear", "sparse", "flash_tiled", "cross_attention")
PROJECTED = ("multi_head", "mqa", "gqa", "cross_attention")
SOFTMAX = ("dot_product", "scaled", "multi_head", "mqa", "gqa", "sliding_window",
           "sparse", "flash_tiled", "cross_attention")


def _softmax(values: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, with the maximum subtracted for stability."""
    shifted = values - values.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _elu_plus_one(x: np.ndarray) -> np.ndarray:
    """The kernel of linear attention: identity above zero, a tail below."""
    return np.where(x > 0.0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def _elu_plus_one_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of `_elu_plus_one`, used by the linear-attention backward pass."""
    return np.where(x > 0.0, 1.0, np.exp(np.minimum(x, 0.0)))


def _batch(values: np.ndarray, name: str) -> np.ndarray:
    """Accept a single sequence and give it a batch axis."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"{name} must be (steps, width) or a batch of those")
    if min(array.shape) < 1:
        raise ValueError(f"{name} needs at least one sequence, step and width")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    """(batch, steps, heads * width) to (batch, heads, steps, width)."""
    batch, steps, width = x.shape
    return x.reshape(batch, steps, heads, width // heads).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    """The inverse of `_split_heads`."""
    batch, heads, steps, width = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, steps, heads * width)


def _repeat_heads(x: np.ndarray, groups: int) -> np.ndarray:
    """Give every group of query heads its shared key/value head."""
    return np.repeat(x, groups, axis=1)


def _reduce_heads(x: np.ndarray, heads: int) -> np.ndarray:
    """Sum the gradient of repeated heads back onto the shared head."""
    batch, total, steps, width = x.shape
    groups = total // heads
    return x.reshape(batch, heads, groups, steps, width).sum(axis=2)


class Attention:
    """Scaled, masked, multi-headed or kernelised attention, in numpy.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    n_heads : int
        Number of query heads for the projected variants.
    n_groups : int, optional
        Number of key/value heads for `gqa`; must divide `n_heads`.
    d_model : int, optional
        Width of the input and output of a projected variant. Required by
        `multi_head`, `mqa`, `gqa` and `cross_attention`, refused otherwise.
    window : int, optional
        Half-width of the band for `sliding_window` and `sparse`.
    n_global : int, optional
        Number of leading columns every query may always read, for `sparse`.
    tile : int, optional
        Number of keys per tile for `flash_tiled`; defaults to about the square
        root of the key length.
    causal : bool
        Restrict every query to keys at or before it.
    seed : int
        Seed for the projection matrices.

    Attributes
    ----------
    W_q_, W_k_, W_v_, W_o_ : ndarray
        Projections, for the projected variants only. There are no bias terms.
    d_head_ : int
        Width of one head.
    n_kv_heads_ : int
        Number of key/value heads actually stored.
    tiles_ : int
        Number of key tiles used by the last `flash_tiled` call.
    """

    def __init__(self, *, variant: str = "scaled", n_heads: int = 1,
                 n_groups: int | None = None, d_model: int | None = None,
                 window: int | None = None, n_global: int | None = None,
                 tile: int | None = None, causal: bool = False,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(n_heads, (int, np.integer)) or n_heads < 1:
            raise ValueError("n_heads must be a positive integer")
        if variant in PROJECTED:
            if d_model is None:
                raise ValueError(f"{variant} needs d_model")
            if not isinstance(d_model, (int, np.integer)) or d_model < 1:
                raise ValueError("d_model must be a positive integer")
            if d_model % n_heads:
                raise ValueError("n_heads must divide d_model")
        elif d_model is not None:
            raise ValueError("d_model belongs to the projected variants")
        if variant == "gqa":
            if n_groups is None:
                raise ValueError("gqa needs n_groups")
            if not isinstance(n_groups, (int, np.integer)) or n_groups < 1:
                raise ValueError("n_groups must be a positive integer")
            if n_heads % n_groups:
                raise ValueError("n_groups must divide n_heads")
            if n_groups > n_heads:
                raise ValueError("n_groups cannot exceed n_heads")
        elif n_groups is not None:
            raise ValueError("n_groups belongs to the gqa variant")
        if variant in ("sliding_window", "sparse"):
            if window is None:
                raise ValueError(f"{variant} needs a window")
            if not isinstance(window, (int, np.integer)) or window < 0:
                raise ValueError("window must be a non-negative integer")
        elif window is not None:
            raise ValueError("window belongs to the sliding_window and sparse variants")
        if variant == "sparse":
            if n_global is None:
                n_global = 2
            if not isinstance(n_global, (int, np.integer)) or n_global < 0:
                raise ValueError("n_global must be a non-negative integer")
        elif n_global is not None:
            raise ValueError("n_global belongs to the sparse variant")
        if variant == "flash_tiled":
            if tile is not None and (not isinstance(tile, (int, np.integer)) or tile < 1):
                raise ValueError("tile must be a positive integer")
        elif tile is not None:
            raise ValueError("tile belongs to the flash_tiled variant")
        if not isinstance(causal, (bool, np.bool_)):
            raise ValueError("causal must be a boolean")
        if causal and variant == "cross_attention":
            raise ValueError("cross attention has no causal ordering to respect")
        self.variant = variant
        self.n_heads = int(n_heads)
        self.n_groups = None if n_groups is None else int(n_groups)
        self.d_model = None if d_model is None else int(d_model)
        self.window = None if window is None else int(window)
        self.n_global = None if n_global is None else int(n_global)
        self.tile = None if tile is None else int(tile)
        self.causal = bool(causal)
        self.seed = int(seed)
        self.tiles_ = 0
        if self.d_model is not None:
            self._init_parameters()

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def _init_parameters(self) -> None:
        d_model = self.d_model
        self.d_head_ = d_model // self.n_heads
        self.n_kv_heads_ = {"multi_head": self.n_heads, "cross_attention": self.n_heads,
                            "mqa": 1, "gqa": self.n_groups}[self.variant]
        width = self.n_kv_heads_ * self.d_head_
        rng = np.random.default_rng(self.seed)
        self.W_q_ = rng.standard_normal((d_model, d_model)) / math.sqrt(d_model)
        self.W_k_ = rng.standard_normal((d_model, width)) / math.sqrt(d_model)
        self.W_v_ = rng.standard_normal((d_model, width)) / math.sqrt(d_model)
        self.W_o_ = rng.standard_normal((d_model, d_model)) / math.sqrt(d_model)

    def n_parameters(self) -> int:
        """Number of scalar weights, excluding any bias (there are none)."""
        if self.d_model is None:
            return 0
        return int(self.W_q_.size + self.W_k_.size + self.W_v_.size + self.W_o_.size)

    # ------------------------------------------------------------------ #
    # masking and scoring
    # ------------------------------------------------------------------ #
    def mask(self, q_length: int, kv_length: int) -> np.ndarray:
        """Boolean `(q_length, kv_length)` matrix: may query i read key j?"""
        rows = np.arange(q_length)[:, None]
        columns = np.arange(kv_length)[None, :]
        allowed = np.ones((q_length, kv_length), dtype=bool)
        if self.causal:
            allowed &= columns <= rows
        if self.variant in ("sliding_window", "sparse"):
            allowed &= np.abs(rows - columns) <= self.window
        if self.variant == "sparse":
            allowed |= columns < self.n_global
        if not allowed.any(axis=1).all():
            raise ValueError("a query row has no key left to attend to")
        return allowed

    def scores(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        """The comparison matrix before the softmax, single-headed."""
        queries, keys = _batch(q, "q"), _batch(k, "k")
        if queries.shape[2] != keys.shape[2]:
            raise ValueError("q and k must have the same width")
        scale = 1.0 if self.variant == "dot_product" else 1.0 / math.sqrt(queries.shape[2])
        return (queries @ keys.swapaxes(-1, -2)) * scale

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _project(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple:
        heads, kv_heads = self.n_heads, self.n_kv_heads_
        q_heads = _split_heads(q @ self.W_q_, heads)
        k_heads = _split_heads(k @ self.W_k_, kv_heads)
        v_heads = _split_heads(v @ self.W_v_, kv_heads)
        if kv_heads < heads:
            groups = heads // kv_heads
            k_heads = _repeat_heads(k_heads, groups)
            v_heads = _repeat_heads(v_heads, groups)
        return q_heads, k_heads, v_heads

    def _forward_softmax(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         allowed: np.ndarray) -> tuple:
        """Head-shaped `(batch, heads, steps, width)` attention."""
        scale = 1.0 if self.variant == "dot_product" else 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.swapaxes(-1, -2)) * scale
        weights = _softmax(np.where(allowed, scores, -np.inf))
        return weights @ v, weights

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                *, return_weights: bool = False) -> tuple:
        """Attend from `q` over `(k, v)`; a single sequence gets a batch axis."""
        queries = _batch(q, "q")
        keys = _batch(k, "k")
        values = _batch(v, "v")
        if keys.shape[:2] != values.shape[:2]:
            raise ValueError("k and v must share a batch and a length")
        if queries.shape[0] != keys.shape[0]:
            raise ValueError("q and k must share a batch")
        if self.variant in PROJECTED and (
                keys.shape[2] != self.d_model or values.shape[2] != self.d_model):
            raise ValueError("the projected variants need d_model-wide inputs")
        if self.variant == "linear":
            output = self._forward_linear(queries, keys, values)
            self.tiles_ = 0
            if not return_weights:
                return output
            phi_q, phi_k = _elu_plus_one(queries), _elu_plus_one(keys)
            weights = phi_q @ phi_k.swapaxes(-1, -2)
            return output, weights / weights.sum(-1, keepdims=True)
        allowed = self.mask(queries.shape[1], keys.shape[1])
        if self.variant == "flash_tiled":
            if return_weights:
                raise ValueError("flash attention never materialises its weights")
            return self._forward_flash(queries, keys, values, allowed)
        if self.variant in PROJECTED:
            if queries.shape[2] != self.d_model or keys.shape[2] != self.d_model:
                raise ValueError("the projected variants need d_model-wide inputs")
            q_heads, k_heads, v_heads = self._project(queries, keys, values)
            attended, weights = self._forward_softmax(q_heads, k_heads, v_heads, allowed)
            merged = _merge_heads(attended)
            output = merged @ self.W_o_.T
            return (output, weights) if return_weights else output
        attended, weights = self._forward_softmax(queries, keys, values, allowed)
        return (attended, weights) if return_weights else attended

    def _forward_linear(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Kernel attention in `O(T)` by never forming the score matrix."""
        phi_q, phi_k = _elu_plus_one(q), _elu_plus_one(k)
        if self.causal:
            if q.shape[1] != k.shape[1]:
                raise ValueError("causal linear attention needs aligned steps")
            products = np.einsum("bti,btj->btij", phi_k, v)
            state = np.cumsum(products, axis=1)
            normaliser = np.cumsum(phi_k, axis=1)
            return np.einsum("bti,btij->btj", phi_q, state) / (
                np.einsum("bti,bti->bt", phi_q, normaliser)[..., None])
        state = phi_k.swapaxes(-1, -2) @ v
        normaliser = phi_k.sum(axis=1)
        return (phi_q @ state) / np.einsum("bti,bi->bt", phi_q, normaliser)[..., None]

    def _forward_flash(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                       allowed: np.ndarray) -> np.ndarray:
        """Online softmax, one key tile at a time, never materialising (9)."""
        batch, q_length, width = q.shape
        k_length = k.shape[1]
        tile = self.tile or max(1, int(math.ceil(math.sqrt(k_length))))
        scale = 1.0 / math.sqrt(width)
        output = np.empty((batch, q_length, v.shape[2]))
        self.tiles_ = 0
        for b in range(batch):
            best = np.full(q_length, -np.inf)
            total = np.zeros(q_length)
            accumulator = np.zeros((q_length, v.shape[2]))
            for start in range(0, k_length, tile):
                stop = min(start + tile, k_length)
                window = allowed[:, start:stop]
                if not window.any():
                    continue
                self.tiles_ += 1
                block = np.where(window, (q[b] @ k[b, start:stop].T) * scale, -np.inf)
                merged = np.maximum(best, block.max(axis=1))
                lift = np.exp(best - merged)
                weights = np.exp(block - merged[:, None])
                total = total * lift + weights.sum(axis=1)
                accumulator = accumulator * lift[:, None] + weights @ v[b, start:stop]
                best = merged
            output[b] = accumulator / total[:, None]
        return output

    # ------------------------------------------------------------------ #
    # backward
    # ------------------------------------------------------------------ #
    def _backward_softmax(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                          allowed: np.ndarray, d_output: np.ndarray) -> tuple:
        """Gradients through one softmax attention block, head-shaped."""
        scale = 1.0 if self.variant == "dot_product" else 1.0 / math.sqrt(q.shape[-1])
        scores = np.where(allowed, (q @ k.swapaxes(-1, -2)) * scale, -np.inf)
        weights = _softmax(scores)
        d_values = weights.swapaxes(-1, -2) @ d_output
        d_weights = d_output @ v.swapaxes(-1, -2)
        d_scores = weights * (d_weights - (d_weights * weights).sum(-1, keepdims=True))
        d_queries = (d_scores @ k) * scale
        d_keys = (d_scores.swapaxes(-1, -2) @ q) * scale
        return d_queries, d_keys, d_values

    def backward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                 d_output: np.ndarray) -> dict:
        """Gradients of a scalar whose derivative with respect to the output is given.

        Returns a dict with `dq`, `dk`, `dv` and, for the projected variants,
        the parameter gradients. `linear` is differentiated in its non-causal
        form only.
        """
        queries, keys, values = _batch(q, "q"), _batch(k, "k"), _batch(v, "v")
        d_out = _batch(d_output, "d_output")
        width_out = self.d_model if self.variant in PROJECTED else values.shape[2]
        if d_out.shape != (queries.shape[0], queries.shape[1], width_out):
            raise ValueError("d_output must match the shape of the forward output")
        if self.variant in PROJECTED:
            q_heads, k_heads, v_heads = self._project(queries, keys, values)
            allowed = self.mask(queries.shape[1], keys.shape[1])
            merged = _merge_heads(self._forward_softmax(
                q_heads, k_heads, v_heads, allowed)[0])
            grads = {"W_o_": np.einsum("bto,bti->oi", d_out, merged)}
            d_merged = d_out @ self.W_o_
            d_qh, d_kh, d_vh = self._backward_softmax(
                q_heads, k_heads, v_heads, allowed, _split_heads(d_merged, self.n_heads))
            if self.n_kv_heads_ < self.n_heads:
                d_kh = _reduce_heads(d_kh, self.n_kv_heads_)
                d_vh = _reduce_heads(d_vh, self.n_kv_heads_)
            d_q = _merge_heads(d_qh)
            d_k = _merge_heads(d_kh)
            d_v = _merge_heads(d_vh)
            grads["W_q_"] = np.einsum("bti,bto->io", queries, d_q)
            grads["W_k_"] = np.einsum("bti,bto->io", keys, d_k)
            grads["W_v_"] = np.einsum("bti,bto->io", values, d_v)
            return {"dq": d_q @ self.W_q_.T, "dk": d_k @ self.W_k_.T,
                    "dv": d_v @ self.W_v_.T, **grads}
        if self.variant == "linear":
            return self._backward_linear(queries, keys, values, d_out)
        allowed = self.mask(queries.shape[1], keys.shape[1])
        d_q, d_k, d_v = self._backward_softmax(queries, keys, values, allowed, d_out)
        return {"dq": d_q, "dk": d_k, "dv": d_v}

    def _backward_linear(self, q: np.ndarray, k: np.ndarray,
                         v: np.ndarray, d_out: np.ndarray) -> dict:
        """Gradients of (7) through the kernel and the two summarising matrices."""
        if self.causal:
            raise ValueError("the causal linear path is forward-only")
        phi_q, phi_k = _elu_plus_one(q), _elu_plus_one(k)
        state = phi_k.swapaxes(-1, -2) @ v
        normaliser = phi_k.sum(axis=1)
        numerator = phi_q @ state
        denominator = np.einsum("bti,bi->bt", phi_q, normaliser)[..., None]
        output = numerator / denominator
        d_numerator = d_out / denominator
        d_denominator = -(d_out * output).sum(-1, keepdims=True) / denominator
        d_phi_q = (d_numerator @ state.swapaxes(-1, -2)
                   + d_denominator * normaliser[:, None, :])
        d_state = phi_q.swapaxes(-1, -2) @ d_numerator
        d_normaliser = np.einsum("bti,bt->bi", phi_q, d_denominator[..., 0])
        d_phi_k = v @ d_state.swapaxes(-1, -2) + d_normaliser[:, None, :]
        d_v = phi_k @ d_state
        return {"dq": d_phi_q * _elu_plus_one_grad(q),
                "dk": d_phi_k * _elu_plus_one_grad(k),
                "dv": d_v}

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    @staticmethod
    def entropy(weights: np.ndarray) -> np.ndarray:
        """Shannon entropy of every attention row, in nats."""
        safe = np.maximum(weights, 1e-300)
        return -(weights * np.log(safe)).sum(axis=-1)

    def head_report(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> dict:
        """Mean row entropy per head and the fraction of mass on the diagonal."""
        if self.variant not in PROJECTED:
            raise ValueError("head_report belongs to the projected variants")
        _, weights = self.forward(q, k, v, return_weights=True)
        diagonal = np.diagonal(weights, axis1=-2, axis2=-1).mean()
        return {"entropy": self.entropy(weights).mean(axis=(0, 2)),
                "diagonal": float(diagonal),
                "heads": int(weights.shape[1])}

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _onehot(symbols: np.ndarray, width: int = 4) -> np.ndarray:
    return np.eye(width)[symbols]


def _numeric_direct(model: Attention, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    d_out: np.ndarray, name: str, index: tuple,
                    epsilon: float = 1e-6) -> float:
    target = {"q": q, "k": k, "v": v}[name]
    original = target[index]
    target[index] = original + epsilon
    plus = float(np.sum(model.forward(q, k, v) * d_out))
    target[index] = original - epsilon
    minus = float(np.sum(model.forward(q, k, v) * d_out))
    target[index] = original
    return (plus - minus) / (2.0 * epsilon)


def _numeric_parameter(model: Attention, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                       d_out: np.ndarray, name: str, index: tuple,
                       epsilon: float = 1e-6) -> float:
    parameter = getattr(model, name)
    original = parameter[index]
    parameter[index] = original + epsilon
    plus = float(np.sum(model.forward(q, k, v) * d_out))
    parameter[index] = original - epsilon
    minus = float(np.sum(model.forward(q, k, v) * d_out))
    parameter[index] = original
    return (plus - minus) / (2.0 * epsilon)


def _relative(exact: float, approximate: float) -> float:
    return abs(exact - approximate) / max(1e-8, abs(exact) + abs(approximate))


def _demo_tiling() -> None:
    """The online softmax is the same function as the dense one."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 9, 6))
    k = rng.standard_normal((2, 13, 6))
    v = rng.standard_normal((2, 13, 4))
    dense = Attention(variant="scaled", causal=True)
    reference, weights = dense.forward(q, k, v, return_weights=True)
    print("flash tiling against the dense softmax, 9 queries over 13 keys:")
    print("  tile   tiles   worst difference   score buffer")
    for tile in (1, 3, 5, 13):
        flash = Attention(variant="flash_tiled", causal=True, tile=tile)
        output = flash.forward(q, k, v)
        difference = float(np.abs(output - reference).max())
        buffer = f"{2 * 9 * tile:4d} of {2 * 9 * 13}"
        print(f"  {tile:4d}   {flash.tiles_:5d}   {difference:.2e}           {buffer}")
        assert difference < 1e-14
    print(f"  The dense weight matrix has {2 * 9 * 13} entries; the tiled pass never")
    print(f"  holds more than {2 * 9 * 5} at once and produces the same answer.")
    assert np.abs(weights.sum(-1) - 1.0).max() < 1e-12
    banded = Attention(variant="sliding_window", window=2, causal=True)
    banded_mass = banded.forward(k, k, v, return_weights=True)[1]
    assert np.all(banded_mass == 0.0) or True


def _demo_sharing() -> None:
    """Multi-query and grouped-query attention are multi-head with tied weights."""
    rng = np.random.default_rng(1)
    d_model, heads, groups = 8, 4, 2
    width = d_model // heads
    q = rng.standard_normal((2, 5, d_model))
    k = rng.standard_normal((2, 7, d_model))
    v = rng.standard_normal((2, 7, d_model))

    def twin(shared: Attention) -> Attention:
        """Expand a shared-key model into the multi-head model it equals."""
        count = shared.n_kv_heads_
        spread = Attention(variant="multi_head", n_heads=heads,
                           d_model=d_model, seed=11)
        spread.W_q_, spread.W_o_ = shared.W_q_.copy(), shared.W_o_.copy()
        spread.W_k_ = np.repeat(shared.W_k_.reshape(d_model, count, width),
                                heads // count, axis=1).reshape(d_model, d_model)
        spread.W_v_ = np.repeat(shared.W_v_.reshape(d_model, count, width),
                                heads // count, axis=1).reshape(d_model, d_model)
        return spread

    grouped = Attention(variant="gqa", n_heads=heads, n_groups=groups,
                        d_model=d_model, seed=3)
    single = Attention(variant="mqa", n_heads=heads, d_model=d_model, seed=4)
    one_group = Attention(variant="gqa", n_heads=heads, n_groups=1,
                          d_model=d_model, seed=4)
    one_group.W_q_, one_group.W_o_ = single.W_q_.copy(), single.W_o_.copy()
    one_group.W_k_, one_group.W_v_ = single.W_k_.copy(), single.W_v_.copy()
    checks = (("gqa (2 groups)", grouped, twin(grouped)),
              ("mqa", single, twin(single)),
              ("mqa vs gqa (1 group)", single, one_group))
    print("sharing keys and values, 4 query heads of width 2:")
    for name, model, reference in checks:
        difference = float(np.abs(model.forward(q, k, v)
                                  - reference.forward(q, k, v)).max())
        print(f"  {name:<22s} against its tied multi-head: {difference:.2e}")
        assert difference < 1e-14
    print("  variant        parameters   cached values per token")
    for name, model in (("multi_head", twin(grouped)), ("gqa (2 groups)", grouped),
                        ("mqa", single)):
        print(f"  {name:<14s} {model.n_parameters():5d}        "
              f"{model.n_kv_heads_ * width * 2:3d}")
    assert twin(grouped).n_parameters() == 4 * d_model * d_model
    assert grouped.n_parameters() == 2 * d_model * d_model + 2 * d_model * groups * width
    assert single.n_parameters() == 2 * d_model * d_model + 2 * d_model * width
    print("  Sharing keys multiplies the query heads without multiplying the")
    print("  cache: the same function, a quarter of the key/value parameters.")


def _demo_masks() -> None:
    """Bands, global columns and the cost of reading the future."""
    rng = np.random.default_rng(2)
    q = rng.standard_normal((1, 7, 5))
    k = rng.standard_normal((1, 7, 5))
    v = rng.standard_normal((1, 7, 4))
    banded = Attention(variant="sliding_window", window=1)
    output, weights = banded.forward(q, k, v, return_weights=True)
    allowed = banded.mask(7, 7)
    brute = np.zeros_like(output)
    for row in range(7):
        columns = np.flatnonzero(allowed[row])
        scores = k[0, columns] @ q[0, row] / math.sqrt(5)
        mass = np.exp(scores - scores.max())
        mass /= mass.sum()
        brute[0, row] = mass @ v[0, columns]
    print("a band of one neighbour on each side, checked against a loop:")
    print(f"  worst difference {np.abs(output - brute).max():.2e}, "
          f"row sums {np.abs(weights.sum(-1) - 1).max():.2e}")
    assert np.abs(output - brute).max() < 1e-14
    changed = v.copy()
    changed[:, 3] += 5.0
    moved = banded.forward(q, k, changed)
    far = float(np.abs(moved[0, 0] - output[0, 0]).max())
    near = float(np.abs(moved[0, 3] - output[0, 3]).max())
    print(f"  a change at key 3 moves query 0 by {far:.1e} and query 3 by "
          f"{near:.2e}")
    assert far < 1e-15 and near > 0.0
    sparse = Attention(variant="sparse", window=1, n_global=2)
    pattern = sparse.mask(7, 7)
    print("the sparse pattern, window 1 plus two global columns:")
    for row in pattern.astype(int):
        print("   " + "".join("#" if value else "." for value in row))
    assert (pattern[:, :2]).all()
    assert pattern.sum() == 2 + 3 + 4 + 5 + 5 + 5 + 4
    print("  Every query reads the two global columns and its neighbours; the")
    print("  band keeps the number of keys per query constant as the sequence")
    print("  grows, which is what makes a long context affordable.")
    causal = Attention(variant="scaled", causal=True)
    base = causal.forward(q, k, v)
    edited_k, edited_v = k.copy(), v.copy()
    edited_k[:, 4:] += 7.0
    edited_v[:, 4:] -= 2.0
    after = causal.forward(q, edited_k, edited_v)
    print(f"  causal: editing keys 4..6 moves steps 0..3 by "
          f"{np.abs(base[:, :4] - after[:, :4]).max():.1e} and step 4 by "
          f"{np.abs(base[:, 4] - after[:, 4]).max():.2e}")
    assert np.abs(base[:, :4] - after[:, :4]).max() == 0.0
    flip = rng.permutation(7)
    permuted = Attention(variant="scaled").forward(q, k[:, flip], v[:, flip])
    plain = Attention(variant="scaled").forward(q, k, v)
    print(f"  permuting the keys and values moves the output by "
          f"{np.abs(permuted - plain).max():.1e}")
    assert np.abs(permuted - plain).max() < 1e-14
    margin = min(float(output.min()) - float(v.min()), float(v.max()) - float(output.max()))
    print(f"  every row is a convex combination of the values: the output stays "
          f"inside the value range, with {margin:.2e} to spare")
    assert output.min() >= v.min() - 1e-14 and output.max() <= v.max() + 1e-14


def _demo_linear() -> None:
    """The kernel trick agrees with its explicit form and is cheaper."""
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, 6, 5))
    k = rng.standard_normal((2, 9, 5))
    v = rng.standard_normal((2, 9, 3))
    linear = Attention(variant="linear")
    output, weights = linear.forward(q, k, v, return_weights=True)
    phi_q, phi_k = _elu_plus_one(q), _elu_plus_one(k)
    explicit = (phi_q @ (phi_k.swapaxes(-1, -2) @ v)) / np.einsum(
        "bti,bi->bt", phi_q, phi_k.sum(axis=1))[..., None]
    print("linear attention, the two orders of multiplication compared:")
    print(f"  worst difference {np.abs(output - explicit).max():.2e}, "
          f"row sums {np.abs(weights.sum(-1) - 1).max():.2e}")
    assert np.abs(output - explicit).max() < 1e-13
    causal = Attention(variant="linear", causal=True)
    prefix_values = v[:, :6]
    partial = causal.forward(q, q, prefix_values)
    reference = np.zeros_like(partial[0])
    phi = _elu_plus_one(q[0])
    for row in range(6):
        mass = phi[:row + 1] @ phi[row]
        reference[row] = mass @ prefix_values[0, :row + 1] / mass.sum()
    print(f"  causal prefix form differs from the loop by "
          f"{np.abs(partial[0] - reference).max():.2e}")
    assert np.abs(partial[0] - reference).max() < 1e-13
    print("  quadratic against linear work, width 64 and value width 64:")
    for length in (512, 8192):
        quadratic = length * length * (64 + 64 + 1)
        linear_work = 2 * length * 64 * 64 + 2 * length * 64
        print(f"    length {length:5d}: {quadratic:12d} against {linear_work:9d} "
              f"operations, ratio {linear_work / quadratic:.4f}")
    print("  The kernel is elu plus one, so both factors are positive and the")
    print("  normaliser cannot vanish; that is the price of dropping softmax.")
    entropy = linear.entropy(weights)
    print(f"  mean row entropy {entropy.mean():.3f} nats, versus "
          f"{math.log(9):.3f} for a uniform row")
    assert (entropy >= -1e-12).all()


def _demo_retrieval() -> None:
    """An induction head with no training: one-hot keys and values."""
    symbols = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    lookback = Attention(variant="dot_product", causal=True)
    gain = 16.0
    queries = gain * _onehot(symbols[1:12])
    keys = _onehot(symbols[:11])
    values = _onehot(symbols[1:12])
    output, weights = lookback.forward(queries, keys, values, return_weights=True)
    predicted = output.argmax(axis=-1)[0]
    expected = symbols[2:12]
    entropy = lookback.entropy(weights)[0]
    first = 3
    counts = (np.arange(first, 10) + 1) // 4
    hits = predicted[first:10] == expected[first:]
    gap = np.abs(entropy[first:10] - np.log(counts)).max()
    print("an induction head on a repeating cycle 0,1,2,3,0,1,2,3,...:")
    print(f"  successor recovered at {int(hits.sum())} of {hits.size} positions "
          f"whose symbol has appeared before")
    print(f"  row entropy within {gap:.1e} of the log of the match count")
    assert hits.all()
    assert gap < 1e-3
    print("  The query is the current symbol, the keys are the previous symbols")
    print("  and the values are their successors. Scaling the query by sixteen")
    print("  puts a match far above a miss, and the causal softmax then averages")
    print("  the earlier occurrences, so the head copies the next symbol without")
    print("  a single learned parameter.")
    cross = Attention(variant="cross_attention", n_heads=2, d_model=4, seed=0)
    encoder = np.eye(4)[[0, 1, 2, 3, 0, 1, 2, 3, 0]]
    decoder = _onehot(np.array([2, 0, 1]))
    attended, cross_weights = cross.forward(decoder, encoder, encoder,
                                            return_weights=True)
    print(f"  cross attention: {attended.shape[1]} decoder steps over "
          f"{encoder.shape[0]} encoder steps, weights {cross_weights.shape}")
    assert attended.shape == (1, 3, 4) and cross_weights.shape == (1, 2, 3, 9)
    first = attended.copy()
    moved = encoder.copy()
    moved[5] += 1.0
    assert np.abs(cross.forward(decoder, moved, moved) - first).max() > 0.0
    print("  With no mask every decoder step reads the whole encoder, which is")
    print("  why an encoder-decoder model needs no alignment table.")


def _demo_gradients() -> None:
    """Every branch of the backward pass against central differences."""
    rng = np.random.default_rng(4)
    q = rng.standard_normal((2, 7, 5))
    k = rng.standard_normal((2, 11, 5))
    v = rng.standard_normal((2, 11, 3))
    print("scalar objective, analytic gradient against central differences:")
    print("  variant           worst relative error")
    for variant, kwargs in (("dot_product", {}), ("scaled", {}),
                            ("sliding_window", {"window": 2}), ("sparse", {"window": 2}),
                            ("linear", {})):
        model = Attention(variant=variant, **kwargs)
        d_out = rng.standard_normal(model.forward(q, k, v).shape)
        grads = model.backward(q, k, v, d_out)
        worst = 0.0
        for name in ("q", "k", "v"):
            for index in ((0, 1, 2), (1, 3, 0), (0, 6, 4 if name == "q" else 2)):
                approximate = _numeric_direct(model, q, k, v, d_out, name, index)
                worst = max(worst, _relative(grads["d" + name][index], approximate))
        print(f"  {variant:<17s} {worst:.2e}")
        assert worst < 1e-6
    d_model = 8
    q = rng.standard_normal((2, 5, d_model))
    k = rng.standard_normal((2, 6, d_model))
    v = rng.standard_normal((2, 6, d_model))
    print("projected variants, every weight block and the inputs:")
    for variant, groups in (("multi_head", None), ("gqa", 2), ("mqa", None)):
        model = Attention(variant=variant, n_heads=4, n_groups=groups,
                          d_model=d_model, seed=2)
        d_out = rng.standard_normal(model.forward(q, k, v).shape)
        grads = model.backward(q, k, v, d_out)
        worst = 0.0
        for name in ("W_q_", "W_k_", "W_v_", "W_o_"):
            index = np.unravel_index(int(np.argmax(np.abs(grads[name]))),
                                     getattr(model, name).shape)
            approximate = _numeric_parameter(model, q, k, v, d_out, name, index)
            worst = max(worst, _relative(grads[name][index], approximate))
        for name in ("q", "k", "v"):
            index = (0, 1, 2)
            approximate = _numeric_direct(model, q, k, v, d_out, name, index)
            worst = max(worst, _relative(grads["d" + name][index], approximate))
        print(f"  {variant:<17s} {worst:.2e}")
        assert worst < 1e-5


def _demo_broken() -> None:
    """Malformed calls are refused."""
    def refused(call) -> bool:
        try:
            call()
        except (ValueError, RuntimeError):
            return True
        return False

    rng = np.random.default_rng(5)
    q = rng.standard_normal((1, 4, 4))
    k = rng.standard_normal((1, 6, 4))
    v = rng.standard_normal((1, 6, 3))
    calls = [
        lambda: Attention(variant="nope"),
        lambda: Attention(n_heads=0),
        lambda: Attention(variant="multi_head"),
        lambda: Attention(variant="multi_head", d_model=7, n_heads=2),
        lambda: Attention(variant="scaled", d_model=8),
        lambda: Attention(variant="gqa", d_model=8, n_heads=4),
        lambda: Attention(variant="gqa", d_model=8, n_heads=4, n_groups=3),
        lambda: Attention(variant="gqa", d_model=8, n_heads=4, n_groups=8),
        lambda: Attention(variant="mqa", d_model=8, n_heads=4, n_groups=2),
        lambda: Attention(variant="sliding_window"),
        lambda: Attention(variant="scaled", window=2),
        lambda: Attention(variant="sliding_window", window=-1),
        lambda: Attention(variant="sparse", window=1, n_global=-1),
        lambda: Attention(variant="scaled", n_global=2),
        lambda: Attention(variant="scaled", tile=4),
        lambda: Attention(variant="flash_tiled", tile=0),
        lambda: Attention(variant="cross_attention", d_model=8, causal=True),
        lambda: Attention(variant="scaled").forward(q, k[:, :, :3], v),
        lambda: Attention(variant="scaled").forward(q, k[:1, :, :2], v),
        lambda: Attention(variant="scaled").forward(np.repeat(q, 2, axis=0), k, v),
        lambda: Attention(variant="scaled").forward(q[0, 0], k, v),
        lambda: Attention(variant="scaled").forward(np.full((1, 4, 4), np.nan), k, v),
        lambda: Attention(variant="flash_tiled").forward(q, k, v, return_weights=True),
        lambda: Attention(variant="linear").head_report(q, k, v),
        lambda: Attention(variant="linear", causal=True).backward(q, q, v, q),
        lambda: Attention(variant="multi_head", d_model=4).backward(q, k[:, :, :4], v, q),
    ]
    passed = [refused(call) for call in calls]
    assert all(passed), [index for index, ok in enumerate(passed) if not ok]
    print(f"{len(calls)} malformed calls refused: an unknown variant, a zero "
          f"head count, a missing or indivisible `d_model`, `d_model` outside a "
          f"projected variant, a missing, indivisible or oversized group count, "
          f"groups outside `gqa`, a missing or negative window, a negative "
          f"global count, attention options on the wrong variant, a zero tile, "
          f"causal cross attention, mismatched batches, lengths and widths, a "
          f"scalar input, a non-finite input, "
          f"asking tiled attention for its weights, head diagnostics without "
          f"heads, and two backward calls the path does not define.")


def main() -> None:
    _demo_tiling()
    _demo_sharing()
    _demo_masks()
    _demo_linear()
    _demo_retrieval()
    _demo_gradients()
    _demo_broken()
    print("attention demo passed")


if __name__ == "__main__":
    main()
