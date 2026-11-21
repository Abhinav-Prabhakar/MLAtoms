"""Latent bottlenecks — Perceiver, Set Transformer and the pieces between.

Chapter:   12-neural-blocks
Companion: perceiver.md
Tier 2 · 4 variants: perceiver, set_transformer, latent_array,
cross_attention_bottleneck
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("perceiver", "set_transformer", "latent_array",
            "cross_attention_bottleneck")

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


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv, {"x": x, "mean": mean, "inv": inv}


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    x, mean, inv = cache["x"], cache["mean"], cache["inv"]
    centred = x - mean
    shift = gradient.mean(axis=-1, keepdims=True)
    scale = (gradient * centred).mean(axis=-1, keepdims=True)
    return inv * (gradient - shift - centred * inv ** 2 * scale)


# --------------------------------------------------------------------------- #
# The bottleneck
# --------------------------------------------------------------------------- #
class Perceiver:
    """A fixed latent array that reads a variable-length sequence.

    `forward(tokens)` takes `(batch, tokens, dim)` and returns
    `(batch, latents, dim)`. Every projection and the FFN is shared across
    depth, so the latent array is the only thing that grows with the number of
    latents.
    """

    def __init__(self, variant: str = "perceiver", *, dim: int = 4,
                 latents: int = 3, heads: int = 2, hidden: int = 8,
                 depth: int = 2, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(dim, latents, heads, hidden, depth) < 1:
            raise ValueError("all sizes must be positive")
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        self.variant = variant
        self.dim = dim
        self.latents = latents
        self.heads = heads
        self.hidden = hidden
        self.depth = depth
        self.depth_per_head = dim // heads
        self.scale = 1.0 / np.sqrt(self.depth_per_head)
        self.seed = seed
        rng = np.random.default_rng(seed)
        p: dict[str, np.ndarray] = {}
        if variant != "set_transformer":
            self._attention(p, "cross", dim, rng)
        if variant in ("perceiver", "latent_array"):
            self._attention(p, "self", dim, rng)
        if variant == "set_transformer":
            self._attention(p, "sab", dim, rng)
            self._attention(p, "pma", dim, rng)
        p["ffn.in"] = rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, hidden))
        p["ffn.in.b"] = np.zeros(hidden)
        p["ffn.out"] = rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, dim))
        p["ffn.out.b"] = np.zeros(dim)
        p["latent"] = rng.normal(0.0, 0.02, (latents, dim))
        self.parameters = p
        self.gradients = {key: np.zeros_like(value)
                          for key, value in p.items()}

    def _attention(self, p, prefix: str, dim: int,
                   rng: np.random.Generator) -> None:
        for name in ("q", "k", "v", "o"):
            p[f"{prefix}.{name}.W"] = rng.normal(
                0.0, 1.0 / np.sqrt(dim), (dim, dim))
            p[f"{prefix}.{name}.b"] = np.zeros(dim)

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    # -- validation ---------------------------------------------------------- #
    def _check(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=float)
        if tokens.ndim != 3:
            raise ValueError("tokens must be (batch, tokens, dim)")
        if tokens.shape[2] != self.dim:
            raise ValueError("tokens have the wrong width")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("batch and token count must be positive")
        if not np.all(np.isfinite(tokens)):
            raise ValueError("tokens must be finite")
        return tokens

    # -- attention ----------------------------------------------------------- #
    def _heads(self, x: np.ndarray) -> np.ndarray:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads,
                         self.depth_per_head).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        batch, _, length, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)

    def _attend(self, queries: np.ndarray, keys: np.ndarray, prefix: str):
        p = self.parameters
        q = self._heads(queries @ p[f"{prefix}.q.W"] + p[f"{prefix}.q.b"])
        k = self._heads(keys @ p[f"{prefix}.k.W"] + p[f"{prefix}.k.b"])
        v = self._heads(keys @ p[f"{prefix}.v.W"] + p[f"{prefix}.v.b"])
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attention = _softmax(scores)
        context = attention @ v
        out = (self._merge(context) @ p[f"{prefix}.o.W"]
               + p[f"{prefix}.o.b"])
        cache = {"queries": queries, "keys": keys, "q": q, "k": k, "v": v,
                 "attention": attention, "context": context,
                 "merged": self._merge(context)}
        return out, cache

    def _attend_backward(self, gradient: np.ndarray, cache, prefix: str):
        p, g = self.parameters, self.gradients
        g[f"{prefix}.o.W"] += cache["merged"].reshape(
            -1, self.dim).T @ gradient.reshape(-1, self.dim)
        g[f"{prefix}.o.b"] += gradient.sum(axis=(0, 1))
        d_context = (gradient @ p[f"{prefix}.o.W"].T).reshape(
            gradient.shape[0], gradient.shape[1], self.heads,
            self.depth_per_head).transpose(0, 2, 1, 3)
        attention, v, q, k = (cache["attention"], cache["v"], cache["q"],
                              cache["k"])
        d_attention = d_context @ v.transpose(0, 1, 3, 2)
        d_v = attention.transpose(0, 1, 3, 2) @ d_context
        d_scores = attention * (
            d_attention - (attention * d_attention).sum(
                axis=-1, keepdims=True))
        d_scores = d_scores * self.scale
        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q
        d_q = self._merge(d_q)
        d_k = self._merge(d_k)
        d_v = self._merge(d_v)
        g[f"{prefix}.q.W"] += cache["queries"].reshape(
            -1, self.dim).T @ d_q.reshape(-1, self.dim)
        g[f"{prefix}.q.b"] += d_q.sum(axis=(0, 1))
        g[f"{prefix}.k.W"] += cache["keys"].reshape(
            -1, self.dim).T @ d_k.reshape(-1, self.dim)
        g[f"{prefix}.k.b"] += d_k.sum(axis=(0, 1))
        g[f"{prefix}.v.W"] += cache["keys"].reshape(
            -1, self.dim).T @ d_v.reshape(-1, self.dim)
        g[f"{prefix}.v.b"] += d_v.sum(axis=(0, 1))
        d_queries = d_q @ p[f"{prefix}.q.W"].T
        d_keys = (d_k @ p[f"{prefix}.k.W"].T
                  + d_v @ p[f"{prefix}.v.W"].T)
        return d_queries, d_keys

    # -- residual sublayers -------------------------------------------------- #
    def _ffn(self, x: np.ndarray):
        p = self.parameters
        norm, norm_cache = _layer_norm(x)
        pre = norm @ p["ffn.in"] + p["ffn.in.b"]
        hidden = _gelu(pre)
        out = hidden @ p["ffn.out"] + p["ffn.out.b"]
        return x + out, {"norm": norm, "norm_cache": norm_cache,
                         "pre": pre, "hidden": hidden}

    def _ffn_backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        p, g = self.parameters, self.gradients
        g["ffn.out"] += cache["hidden"].reshape(
            -1, self.hidden).T @ gradient.reshape(-1, self.dim)
        g["ffn.out.b"] += gradient.sum(axis=(0, 1))
        d_pre = (gradient @ p["ffn.out"].T) * _gelu_grad(cache["pre"])
        g["ffn.in"] += cache["norm"].reshape(
            -1, self.dim).T @ d_pre.reshape(-1, self.hidden)
        g["ffn.in.b"] += d_pre.sum(axis=(0, 1))
        return gradient + _layer_norm_backward(
            d_pre @ p["ffn.in"].T, cache["norm_cache"])

    def _cross_layer(self, x: np.ndarray, keys: np.ndarray):
        norm, norm_cache = _layer_norm(x)
        out, cache = self._attend(norm, keys, "cross")
        return x + out, {"norm": norm_cache, "attend": cache}

    def _cross_backward(self, gradient: np.ndarray, cache):
        d_norm, d_keys = self._attend_backward(gradient, cache["attend"],
                                               "cross")
        return gradient + _layer_norm_backward(d_norm, cache["norm"]), d_keys

    def _self_layer(self, x: np.ndarray, prefix: str):
        norm, norm_cache = _layer_norm(x)
        out, cache = self._attend(norm, norm, prefix)
        return x + out, {"norm": norm_cache, "attend": cache}

    def _self_backward(self, gradient: np.ndarray, cache, prefix: str):
        d_query, d_key = self._attend_backward(gradient, cache["attend"],
                                               prefix)
        return gradient + _layer_norm_backward(d_query + d_key, cache["norm"])

    # -- forward ------------------------------------------------------------- #
    def forward(self, tokens: np.ndarray):
        tokens = self._check(tokens)
        batch = tokens.shape[0]
        latent = np.broadcast_to(self.parameters["latent"],
                                 (batch, self.latents, self.dim)).copy()
        cache: dict = {"tokens": tokens, "steps": []}
        if self.variant == "set_transformer":
            current = tokens
            for _ in range(self.depth):
                current, sab = self._self_layer(current, "sab")
                current, ffn = self._ffn(current)
                cache["steps"].append({"sab": sab, "ffn": ffn})
            latent, cache["pool"] = self._pool_layer(latent, current)
            cache["encoded"] = current
            return latent, cache

        if self.variant == "perceiver":
            latent, cache["cross"] = self._cross_layer(latent, tokens)
            for _ in range(self.depth):
                latent, self_cache = self._self_layer(latent, "self")
                latent, ffn = self._ffn(latent)
                cache["steps"].append({"self": self_cache, "ffn": ffn})
            return latent, cache

        if self.variant == "latent_array":
            for _ in range(self.depth):
                latent, cross = self._cross_layer(latent, tokens)
                latent, self_cache = self._self_layer(latent, "self")
                latent, ffn = self._ffn(latent)
                cache["steps"].append({"cross": cross, "self": self_cache,
                                       "ffn": ffn})
            return latent, cache

        latent, cache["cross"] = self._cross_layer(latent, tokens)
        latent, cache["ffn"] = self._ffn(latent)
        return latent, cache

    def _pool_layer(self, queries: np.ndarray, keys: np.ndarray):
        """A sublayer whose query and key streams differ, as in PMA."""
        norm_q, cache_q = _layer_norm(queries)
        norm_k, cache_k = _layer_norm(keys)
        out, attention = self._attend(norm_q, norm_k, "pma")
        return queries + out, {"norm_q": cache_q, "norm_k": cache_k,
                               "attend": attention}

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        tokens = cache["tokens"]
        if gradient.shape != (tokens.shape[0], self.latents, self.dim):
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        if self.variant == "set_transformer":
            d_latent, d_encoded = self._pool_backward(
                gradient, cache["pool"])
            for step in reversed(cache["steps"]):
                d_encoded = self._ffn_backward(d_encoded, step["ffn"])
                d_encoded = self._same_backward(
                    d_encoded, step["sab"], "sab")
            self._scatter_latent(d_latent)
            return d_encoded

        d_tokens = np.zeros_like(tokens)
        if self.variant == "perceiver":
            d_latent = gradient
            for step in reversed(cache["steps"]):
                d_latent = self._ffn_backward(d_latent, step["ffn"])
                d_latent = self._same_backward(
                    d_latent, step["self"], "self")
            d_latent, d_tokens = self._cross_backward(
                d_latent, cache["cross"])
        elif self.variant == "latent_array":
            d_latent = gradient
            for step in reversed(cache["steps"]):
                d_latent = self._ffn_backward(d_latent, step["ffn"])
                d_latent = self._same_backward(
                    d_latent, step["self"], "self")
                d_latent, d_step = self._cross_backward(
                    d_latent, step["cross"])
                d_tokens += d_step
        else:
            d_latent = self._ffn_backward(gradient, cache["ffn"])
            d_latent, d_tokens = self._cross_backward(
                d_latent, cache["cross"])
        self._scatter_latent(d_latent)
        return d_tokens

    def _same_backward(self, gradient, cache, prefix):
        d_query, d_key = self._attend_backward(gradient, cache["attend"],
                                               prefix)
        return gradient + _layer_norm_backward(d_query + d_key, cache["norm"])

    def _pool_backward(self, gradient, cache):
        d_query, d_key = self._attend_backward(gradient, cache["attend"],
                                               "pma")
        return (gradient + _layer_norm_backward(d_query, cache["norm_q"]),
                _layer_norm_backward(d_key, cache["norm_k"]))

    def _scatter_latent(self, d_latent: np.ndarray) -> None:
        self.gradients["latent"] += d_latent.sum(axis=0)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    batch, tokens, dim = 2, 5, 4
    x = rng.normal(size=(batch, tokens, dim))

    print("perceiver. every variant maps tokens to a fixed latent array:")
    for variant in VARIANTS:
        model = Perceiver(variant, dim=dim, latents=3, heads=2, hidden=8,
                          depth=2, seed=2)
        out, _ = model.forward(x)
        assert out.shape == (batch, 3, dim)
        print(f"    {variant:<26s} {model.n_parameters:3d} parameters")

    print("perceiver. depth reuses the same weights:")
    shallow = Perceiver("perceiver", dim=dim, latents=3, heads=2, hidden=8,
                        depth=1, seed=3)
    deep = Perceiver("perceiver", dim=dim, latents=3, heads=2, hidden=8,
                     depth=6, seed=3)
    assert shallow.n_parameters == deep.n_parameters
    assert not np.allclose(shallow.forward(x)[0], deep.forward(x)[0])
    print(f"  depth 1 and depth 6 both own {shallow.n_parameters} parameters;")
    print("  only the number of passes changes")

    print("perceiver. the bottleneck ignores the order of the tokens:")
    perm = np.array([3, 0, 4, 1, 2])
    for variant in VARIANTS:
        model = Perceiver(variant, dim=dim, latents=3, heads=2, hidden=8,
                          depth=2, seed=4)
        out, _ = model.forward(x)
        moved, _ = model.forward(x[:, perm])
        error = float(np.abs(moved - out).max())
        assert error < 1e-12
        print(f"    {variant:<26s} error {error:.1e}")

    print("perceiver. cross attention is a convex combination:")
    model = Perceiver("perceiver", dim=dim, latents=1, heads=1, hidden=8,
                      depth=1, seed=5)
    out, cache = model.forward(x)
    attention = cache["cross"]["attend"]["attention"]
    values = cache["cross"]["attend"]["v"]
    assert attention.shape == (batch, 1, 1, tokens)
    assert np.allclose(attention.sum(axis=-1), 1.0)
    assert np.all(attention > 0.0)
    context = attention @ values
    assert np.allclose(cache["cross"]["attend"]["context"], context)
    print(f"  one latent, one head: {tokens} weights per example, each row")
    print("  summing to one; the readout is a weighted average of values")

    print("perceiver. permuting the latents permutes the output:")
    for variant in ("perceiver", "latent_array", "set_transformer"):
        model = Perceiver(variant, dim=dim, latents=4, heads=2, hidden=8,
                          depth=2, seed=6)
        out, _ = model.forward(x)
        order = np.array([2, 0, 3, 1])
        model.parameters["latent"] = model.parameters["latent"][order]
        moved, _ = model.forward(x)
        error = float(np.abs(moved - out[:, order]).max())
        assert error < 1e-12
        print(f"    {variant:<26s} error {error:.1e}")

    print("perceiver. finite differences agree with the analytic gradients:")
    small = rng.normal(size=(2, 4, 3))
    target = rng.normal(size=(2, 3, 3))
    for variant in VARIANTS:
        model = Perceiver(variant, dim=3, latents=3, heads=3, hidden=5,
                          depth=2, seed=7)
        out, cache = model.forward(small)
        d_tokens = model.backward(target, cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small)[0]
                value[index] = old - 1e-6
                minus = model.forward(small)[0]
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        for index in np.ndindex(small.shape):
            old = small[index]
            small[index] = old + 1e-6
            plus = model.forward(small)[0]
            small[index] = old - 1e-6
            minus = model.forward(small)[0]
            small[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_tokens[index]))
        assert worst < 1e-6
        print(f"  {variant:<26s} worst error {worst:.1e}")

    print("perceiver. a short fit reduces the loss:")
    model = Perceiver("cross_attention_bottleneck", dim=dim, latents=3,
                      heads=2, hidden=8, depth=1, seed=8)
    goal = rng.normal(size=(batch, 3, dim))
    start = 0.5 * float(np.sum((model.forward(x)[0] - goal) ** 2))
    for _ in range(300):
        out, cache = model.forward(x)
        model.backward(out - goal, cache)
        for key in model.gradients:
            model.parameters[key] -= 0.02 * model.gradients[key]
    end = 0.5 * float(np.sum((model.forward(x)[0] - goal) ** 2))
    assert end < 0.05 * start
    print(f"  300 steps take the loss from {start:.3f} to {end:.4f}")

    print("perceiver. malformed calls are refused:")
    guard = Perceiver("perceiver", dim=dim, latents=3, heads=2, depth=1,
                      seed=9)
    good, good_cache = guard.forward(x)
    bad = [
        lambda: Perceiver("bottleneck"),
        lambda: Perceiver("perceiver", dim=0),
        lambda: Perceiver("perceiver", latents=0),
        lambda: Perceiver("perceiver", heads=0),
        lambda: Perceiver("perceiver", hidden=0),
        lambda: Perceiver("perceiver", depth=0),
        lambda: Perceiver("perceiver", dim=5, heads=2),
        lambda: guard.forward(x[0]),
        lambda: guard.forward(x[..., :3]),
        lambda: guard.forward(np.zeros((0, tokens, dim))),
        lambda: guard.forward(np.zeros((2, 0, dim))),
        lambda: guard.forward(np.full_like(x, np.nan)),
        lambda: guard.backward(np.zeros((2, 4, dim)), good_cache),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, a")
    print("  width that does not divide into heads, a two-dimensional input,")
    print("  the wrong width, an empty batch or token set, non-finite")
    print("  values, and a wrong gradient shape")
    assert good.shape == (batch, 3, dim)
    print("perceiver demo passed")


if __name__ == "__main__":
    demo()
