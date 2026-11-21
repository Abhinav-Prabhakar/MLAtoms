"""Token-mixing blocks — Mixer, gMLP, ConvMixer and ResMLP.

Chapter:   12-neural-blocks
Companion: mixer.md
Tier 2 · 4 variants: mlp_mixer, gmlp, convmixer, resmlp
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("mlp_mixer", "gmlp", "convmixer", "resmlp")

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


def _layer_norm(x: np.ndarray, eps: float = 1e-5):
    """Normalise over the channel axis, the last one."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    out = (x - mean) * inv
    return out, {"x": x, "mean": mean, "inv": inv}


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    x, mean, inv = cache["x"], cache["mean"], cache["inv"]
    centred = x - mean
    shift = gradient.mean(axis=-1, keepdims=True)
    scale = (gradient * centred).mean(axis=-1, keepdims=True)
    return inv * (gradient - shift - centred * inv ** 2 * scale)


def _windows(x: np.ndarray, kernel: int):
    """Sliding windows over the token axis, shape (batch, tokens, dim, k)."""
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    tokens = x.shape[1]
    window = np.stack([padded[:, start:start + tokens, :]
                       for start in range(kernel)], axis=-1)
    return padded, window


# --------------------------------------------------------------------------- #
# The block stack
# --------------------------------------------------------------------------- #
class Mixer:
    """A stack of blocks that mix along tokens and then along channels.

    The input is `(batch, tokens, dim)` and every variant returns the same
    shape. `forward` caches every block, `backward` walks them in reverse.
    """

    def __init__(self, variant: str = "mlp_mixer", *, tokens: int = 8,
                 dim: int = 6, hidden: int = 12, token_hidden: int = 4,
                 depth: int = 2, kernel: int = 3, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(tokens, dim, hidden, token_hidden, depth) < 1:
            raise ValueError("all sizes must be positive")
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be a positive odd number")
        self.variant = variant
        self.tokens = tokens
        self.dim = dim
        self.hidden = hidden
        self.token_hidden = token_hidden
        self.depth = depth
        self.kernel = kernel
        self.seed = seed
        rng = np.random.default_rng(seed)
        p: dict[str, np.ndarray] = {}
        for block in range(depth):
            tag = f"b{block}"
            if variant == "mlp_mixer":
                p[f"{tag}.token.in"] = rng.normal(
                    0.0, 1.0 / np.sqrt(tokens), (tokens, token_hidden))
                p[f"{tag}.token.out"] = rng.normal(
                    0.0, 1.0 / np.sqrt(token_hidden), (token_hidden, tokens))
                self._channel(p, tag, dim, hidden, rng)
            elif variant == "gmlp":
                p[f"{tag}.channel"] = rng.normal(
                    0.0, 1.0 / np.sqrt(dim), (dim, 2 * hidden))
                p[f"{tag}.channel.b"] = np.zeros(2 * hidden)
                p[f"{tag}.gate"] = np.zeros((tokens, tokens))
                p[f"{tag}.gate.b"] = np.ones(tokens)
                p[f"{tag}.out"] = rng.normal(
                    0.0, 1.0 / np.sqrt(hidden), (hidden, dim))
                p[f"{tag}.out.b"] = np.zeros(dim)
            elif variant == "convmixer":
                p[f"{tag}.depth"] = rng.normal(
                    0.0, 1.0 / np.sqrt(kernel), (dim, kernel))
                p[f"{tag}.depth.b"] = np.zeros(dim)
                p[f"{tag}.point"] = rng.normal(
                    0.0, 1.0 / np.sqrt(dim), (dim, dim))
                p[f"{tag}.point.b"] = np.zeros(dim)
            else:  # resmlp
                p[f"{tag}.token.scale"] = np.ones(dim)
                p[f"{tag}.token.bias"] = np.zeros(dim)
                p[f"{tag}.token.mix"] = rng.normal(
                    0.0, 1.0 / np.sqrt(tokens), (tokens, tokens))
                p[f"{tag}.channel.scale"] = np.ones(dim)
                p[f"{tag}.channel.bias"] = np.zeros(dim)
                self._channel(p, tag, dim, hidden, rng)
        p["head.scale"] = np.ones(dim)
        p["head.bias"] = np.zeros(dim)
        self.parameters = p
        self.gradients = {key: np.zeros_like(value)
                          for key, value in p.items()}

    def _channel(self, p, tag, dim, hidden, rng) -> None:
        p[f"{tag}.channel.in"] = rng.normal(
            0.0, 1.0 / np.sqrt(dim), (dim, hidden))
        p[f"{tag}.channel.in.b"] = np.zeros(hidden)
        p[f"{tag}.channel.out"] = rng.normal(
            0.0, 1.0 / np.sqrt(hidden), (hidden, dim))
        p[f"{tag}.channel.out.b"] = np.zeros(dim)

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    # -- validation ---------------------------------------------------------- #
    def _check(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError("x must be (batch, tokens, dim)")
        if x.shape[1:] != (self.tokens, self.dim):
            raise ValueError("x has the wrong token or channel count")
        if x.shape[0] < 1 or not np.all(np.isfinite(x)):
            raise ValueError("x must be non-empty and finite")
        return x

    # -- variant blocks ------------------------------------------------------ #
    def _block(self, x: np.ndarray, block: int):
        tag = f"b{block}"
        p = self.parameters
        if self.variant == "mlp_mixer":
            norm, norm_cache = _layer_norm(x)
            flat = norm.transpose(0, 2, 1)
            inner = flat @ p[f"{tag}.token.in"]
            hidden = _gelu(inner)
            mixed = (hidden @ p[f"{tag}.token.out"]).transpose(0, 2, 1)
            first = x + mixed
            norm2, norm2_cache = _layer_norm(first)
            chan_pre = (norm2 @ p[f"{tag}.channel.in"]
                        + p[f"{tag}.channel.in.b"])
            chan = _gelu(chan_pre)
            mixed2 = (chan @ p[f"{tag}.channel.out"]
                      + p[f"{tag}.channel.out.b"])
            out = first + mixed2
            cache = {"x": x, "norm": norm, "norm_cache": norm_cache,
                     "inner": inner, "hidden": hidden, "mixed": mixed,
                     "first": first, "norm2": norm2,
                     "norm2_cache": norm2_cache, "chan_pre": chan_pre,
                     "chan": chan, "mixed2": mixed2}
            return out, cache

        if self.variant == "gmlp":
            norm, norm_cache = _layer_norm(x)
            pre_act = norm @ p[f"{tag}.channel"] + p[f"{tag}.channel.b"]
            pre = _gelu(pre_act)
            activated, values = pre[..., :self.hidden], pre[..., self.hidden:]
            gate = (values.transpose(0, 2, 1) @ p[f"{tag}.gate"]
                    + p[f"{tag}.gate.b"][None, None, :]).transpose(0, 2, 1)
            gated = activated * gate
            mixed = gated @ p[f"{tag}.out"] + p[f"{tag}.out.b"]
            out = x + mixed
            cache = {"x": x, "norm": norm, "norm_cache": norm_cache,
                     "pre_act": pre_act, "pre": pre,
                     "activated": activated, "values": values,
                     "gate": gate, "gated": gated, "mixed": mixed}
            return out, cache

        if self.variant == "convmixer":
            _, window = _windows(x, self.kernel)
            depth = np.einsum("btck,ck->btc", window,
                              p[f"{tag}.depth"]) + p[f"{tag}.depth.b"]
            first = x + depth
            norm = _gelu(first)
            point = norm @ p[f"{tag}.point"] + p[f"{tag}.point.b"]
            out = first + point
            cache = {"x": x, "window": window, "first": first,
                     "norm": norm, "point": point}
            return out, cache

        # resmlp
        scale, bias = p[f"{tag}.token.scale"], p[f"{tag}.token.bias"]
        norm = x * scale + bias
        mixed = (norm.transpose(0, 2, 1)
                 @ p[f"{tag}.token.mix"]).transpose(0, 2, 1)
        first = x + mixed
        scale2 = p[f"{tag}.channel.scale"]
        bias2 = p[f"{tag}.channel.bias"]
        norm2 = first * scale2 + bias2
        chan_pre = (norm2 @ p[f"{tag}.channel.in"]
                    + p[f"{tag}.channel.in.b"])
        chan = _gelu(chan_pre)
        mixed2 = chan @ p[f"{tag}.channel.out"] + p[f"{tag}.channel.out.b"]
        out = first + mixed2
        cache = {"x": x, "norm": norm, "mixed": mixed, "first": first,
                 "scale": scale, "bias": bias, "norm2": norm2,
                 "chan_pre": chan_pre, "chan": chan, "mixed2": mixed2,
                 "scale2": scale2, "bias2": bias2}
        return out, cache

    # -- forward ------------------------------------------------------------- #
    def forward(self, x: np.ndarray):
        x = self._check(x)
        caches = []
        for block in range(self.depth):
            x, cache = self._block(x, block)
            caches.append(cache)
        norm = x * self.parameters["head.scale"] + self.parameters["head.bias"]
        return norm, {"blocks": caches, "final": x, "head": norm}

    # -- backward ------------------------------------------------------------ #
    def _block_backward(self, gradient: np.ndarray, block: int, cache):
        tag = f"b{block}"
        g, p = self.gradients, self.parameters
        if self.variant == "mlp_mixer":
            g[f"{tag}.channel.out"] += cache["chan"].reshape(
                -1, self.hidden).T @ gradient.reshape(-1, self.dim)
            g[f"{tag}.channel.out.b"] += gradient.sum(axis=(0, 1))
            d_chan = (gradient @ p[f"{tag}.channel.out"].T) \
                * _gelu_grad(cache["chan_pre"])
            g[f"{tag}.channel.in"] += cache["norm2"].reshape(
                -1, self.dim).T @ d_chan.reshape(-1, self.hidden)
            g[f"{tag}.channel.in.b"] += d_chan.sum(axis=(0, 1))
            d_first = gradient + _layer_norm_backward(
                d_chan @ p[f"{tag}.channel.in"].T, cache["norm2_cache"])
            d_flat = d_first.transpose(0, 2, 1).reshape(-1, self.tokens)
            hidden = cache["hidden"]
            g[f"{tag}.token.out"] += hidden.reshape(
                -1, self.token_hidden).T @ d_flat
            d_hidden = (d_flat @ p[f"{tag}.token.out"].T) * _gelu_grad(
                cache["inner"]).reshape(-1, self.token_hidden)
            g[f"{tag}.token.in"] += cache["norm"].transpose(
                0, 2, 1).reshape(-1, self.tokens).T @ d_hidden
            d_norm = d_hidden @ p[f"{tag}.token.in"].T
            d_norm = d_norm.reshape(gradient.shape[0], self.dim,
                                    self.tokens).transpose(0, 2, 1)
            return d_first + _layer_norm_backward(d_norm, cache["norm_cache"])

        if self.variant == "gmlp":
            g[f"{tag}.out"] += cache["gated"].reshape(
                -1, self.hidden).T @ gradient.reshape(-1, self.dim)
            g[f"{tag}.out.b"] += gradient.sum(axis=(0, 1))
            d_gated = gradient @ p[f"{tag}.out"].T
            d_activated = d_gated * cache["gate"]
            d_gate = d_gated * cache["activated"]
            g[f"{tag}.gate.b"] += d_gate.sum(axis=(0, 2))
            flat = d_gate.transpose(0, 2, 1).reshape(-1, self.tokens)
            g[f"{tag}.gate"] += cache["values"].transpose(
                0, 2, 1).reshape(-1, self.tokens).T @ flat
            d_values = (flat @ p[f"{tag}.gate"].T).reshape(
                gradient.shape[0], self.hidden, self.tokens).transpose(0, 2, 1)
            d_pre = np.concatenate([d_activated, d_values], axis=-1)
            d_pre = d_pre * _gelu_grad(cache["pre_act"])
            g[f"{tag}.channel"] += cache["norm"].reshape(
                -1, self.dim).T @ d_pre.reshape(-1, 2 * self.hidden)
            g[f"{tag}.channel.b"] += d_pre.sum(axis=(0, 1))
            d_norm = d_pre @ p[f"{tag}.channel"].T
            return gradient + _layer_norm_backward(
                d_norm, cache["norm_cache"])

        if self.variant == "convmixer":
            g[f"{tag}.point"] += cache["norm"].reshape(
                -1, self.dim).T @ gradient.reshape(-1, self.dim)
            g[f"{tag}.point.b"] += gradient.sum(axis=(0, 1))
            d_first = gradient + _gelu_grad(cache["first"]) * (
                gradient @ p[f"{tag}.point"].T)
            d_window = d_first[..., None] * p[f"{tag}.depth"][None, None, :, :]
            pad = self.kernel // 2
            d_padded = np.zeros((gradient.shape[0],
                                 self.tokens + 2 * pad, self.dim))
            for start in range(self.kernel):
                d_padded[:, start:start + self.tokens, :] += \
                    d_window[..., start]
            g[f"{tag}.depth"] += np.einsum(
                "btck,btc->ck", cache["window"], d_first)
            g[f"{tag}.depth.b"] += d_first.sum(axis=(0, 1))
            return d_first + d_padded[:, pad:pad + self.tokens, :]

        # resmlp
        g[f"{tag}.channel.out"] += cache["chan"].reshape(
            -1, self.hidden).T @ gradient.reshape(-1, self.dim)
        g[f"{tag}.channel.out.b"] += gradient.sum(axis=(0, 1))
        d_chan = (gradient @ p[f"{tag}.channel.out"].T) \
            * _gelu_grad(cache["chan_pre"])
        g[f"{tag}.channel.in"] += cache["norm2"].reshape(
            -1, self.dim).T @ d_chan.reshape(-1, self.hidden)
        g[f"{tag}.channel.in.b"] += d_chan.sum(axis=(0, 1))
        d_norm2 = d_chan @ p[f"{tag}.channel.in"].T
        d_first = gradient + d_norm2 * cache["scale2"]
        g[f"{tag}.channel.scale"] += np.sum(d_norm2 * cache["first"],
                                            axis=(0, 1))
        g[f"{tag}.channel.bias"] += d_norm2.sum(axis=(0, 1))
        d_mixed = d_first
        flat = d_mixed.transpose(0, 2, 1).reshape(-1, self.tokens)
        g[f"{tag}.token.mix"] += cache["norm"].transpose(
            0, 2, 1).reshape(-1, self.tokens).T @ flat
        d_norm = (flat @ p[f"{tag}.token.mix"].T).reshape(
            gradient.shape[0], self.dim, self.tokens).transpose(0, 2, 1)
        g[f"{tag}.token.scale"] += np.sum(d_norm * cache["x"], axis=(0, 1))
        g[f"{tag}.token.bias"] += d_norm.sum(axis=(0, 1))
        return d_first + d_norm * cache["scale"]

    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        shape = cache["head"].shape
        if gradient.shape != shape:
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        d_norm = gradient
        self.gradients["head.scale"] += np.sum(d_norm * cache["final"],
                                               axis=(0, 1))
        self.gradients["head.bias"] += d_norm.sum(axis=(0, 1))
        d_x = d_norm * self.parameters["head.scale"]
        for block in range(self.depth - 1, -1, -1):
            d_x = self._block_backward(d_x, block, cache["blocks"][block])
        return d_x


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    batch, tokens, dim = 2, 8, 6
    x = rng.normal(size=(batch, tokens, dim))

    print("mixer. every variant keeps the shape and has a known size:")
    for variant in VARIANTS:
        model = Mixer(variant, tokens=tokens, dim=dim, hidden=12,
                      token_hidden=4, depth=2, kernel=3, seed=2)
        out, _ = model.forward(x)
        assert out.shape == x.shape
        print(f"    {variant:<9s} {model.n_parameters:4d} parameters")
    mix = Mixer("mlp_mixer", tokens=8, dim=6, hidden=12, token_hidden=4,
                depth=2).n_parameters
    conv = Mixer("convmixer", tokens=8, dim=6, depth=2).n_parameters
    print(f"  two blocks plus the head: Mixer {mix}, ConvMixer {conv}")

    print("mixer. mlp_mixer transposes the token axis for its first MLP:")
    model = Mixer("mlp_mixer", tokens=tokens, dim=dim, hidden=12,
                  token_hidden=4, depth=1, seed=3)
    out, cache = model.forward(x)
    p = model.parameters
    norm = _layer_norm(x)[0]
    inner = norm.transpose(0, 2, 1) @ p["b0.token.in"]
    mixed = (_gelu(inner) @ p["b0.token.out"]).transpose(0, 2, 1)
    first = x + mixed
    chan = _gelu(_layer_norm(first)[0] @ p["b0.channel.in"]
                 + p["b0.channel.in.b"])
    expected = first + chan @ p["b0.channel.out"] + p["b0.channel.out.b"]
    head = expected * p["head.scale"] + p["head.bias"]
    assert np.allclose(out, head)
    assert np.allclose(cache["blocks"][0]["mixed"], mixed)
    print(f"  token mix is {tokens}x{4} then {4}x{tokens}, shared by all")
    print("  channels but indexed by position")

    print("mixer. gmlp starts with an identity gate:")
    model = Mixer("gmlp", tokens=tokens, dim=dim, hidden=6, depth=1, seed=4)
    out, cache = model.forward(x)
    gate = cache["blocks"][0]["gate"]
    assert np.allclose(gate, 1.0)
    assert np.allclose(cache["blocks"][0]["gated"],
                       cache["blocks"][0]["activated"])
    assert model.parameters["b0.gate"].max() == 0.0
    print(f"  the {tokens}x{tokens} spatial matrix is zero and the bias one,")
    print("  so the block opens as a channel-only MLP")

    print("mixer. convmixer mixes only inside the kernel window:")
    model = Mixer("convmixer", tokens=tokens, dim=dim, hidden=dim, depth=2,
                  kernel=3, seed=5)
    base, _ = model.forward(x)
    moved = x.copy()
    moved[:, 2, :] += 1.0
    shifted, _ = model.forward(moved)
    difference = np.abs(shifted - base).sum(axis=(0, 2))
    assert np.all(difference[:5] > 0.0)
    assert np.allclose(difference[5:], 0.0)
    print(f"  one perturbed token reaches radius {2 * (3 // 2)} over two")
    print(f"  layers: tokens {np.flatnonzero(difference > 0).tolist()}, and no")
    print("  further")

    print("mixer. resmlp mixes tokens without a nonlinearity:")
    model = Mixer("resmlp", tokens=tokens, dim=dim, hidden=12, depth=1,
                  seed=6)
    out, cache = model.forward(x)
    p = model.parameters
    norm = x * p["b0.token.scale"] + p["b0.token.bias"]
    mixed = (norm.transpose(0, 2, 1)
             @ p["b0.token.mix"]).transpose(0, 2, 1)
    assert np.allclose(cache["blocks"][0]["mixed"], mixed)
    first = x + mixed
    chan = _gelu((first * p["b0.channel.scale"] + p["b0.channel.bias"])
                 @ p["b0.channel.in"] + p["b0.channel.in.b"])
    expected = first + chan @ p["b0.channel.out"] + p["b0.channel.out.b"]
    assert np.allclose(out, expected * p["head.scale"] + p["head.bias"])
    offset = np.arange(dim, dtype=float) + 1.0
    shifted, shifted_cache = model.forward(x + offset)
    assert np.allclose(shifted_cache["blocks"][0]["norm"]
                       - cache["blocks"][0]["norm"],
                       offset * p["b0.token.scale"])
    print("  the affine norm is not mean-centred: adding a constant vector")
    print("  shifts every token by that vector times the learned scale")

    print("mixer. the shared layer norm matches finite differences:")
    probe = rng.normal(size=(2, 3, 4))
    weight = rng.normal(size=(2, 3, 4))
    _, norm_cache = _layer_norm(probe)
    analytic = _layer_norm_backward(weight, norm_cache)
    worst = 0.0
    for index in np.ndindex(probe.shape):
        old = probe[index]
        probe[index] = old + 1e-6
        plus = _layer_norm(probe)[0]
        probe[index] = old - 1e-6
        minus = _layer_norm(probe)[0]
        probe[index] = old
        numeric = float(np.sum((plus - minus) * weight)) / 2e-6
        worst = max(worst, abs(numeric - analytic[index]))
    assert worst < 1e-8
    print(f"  worst channel error {worst:.1e} on a 2x3x4 probe")

    print("mixer. finite differences agree with the analytic gradients:")
    small_x = rng.normal(size=(2, 4, 3))
    target = rng.normal(size=(2, 4, 3))
    for variant in VARIANTS:
        model = Mixer(variant, tokens=4, dim=3, hidden=5, token_hidden=2,
                      depth=1, kernel=3, seed=7)
        out, cache = model.forward(small_x)
        d_x = model.backward(target, cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small_x)[0]
                value[index] = old - 1e-6
                minus = model.forward(small_x)[0]
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        for index in np.ndindex(small_x.shape):
            old = small_x[index]
            small_x[index] = old + 1e-6
            plus = model.forward(small_x)[0]
            small_x[index] = old - 1e-6
            minus = model.forward(small_x)[0]
            small_x[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_x[index]))
        assert worst < 1e-6
        print(f"  {variant:<9s} worst parameter or input error {worst:.1e}")

    print("mixer. a short fit reduces the loss:")
    model = Mixer("mlp_mixer", tokens=8, dim=6, hidden=12, token_hidden=4,
                  depth=2, seed=8)
    goal = rng.normal(size=(batch, tokens, dim))
    start = 0.5 * float(np.sum((model.forward(x)[0] - goal) ** 2))
    for _ in range(300):
        out, cache = model.forward(x)
        residual = out - goal
        model.backward(residual, cache)
        for key in model.gradients:
            model.parameters[key] -= 0.02 * model.gradients[key]
    end = 0.5 * float(np.sum((model.forward(x)[0] - goal) ** 2))
    assert end < 0.05 * start
    print(f"  300 steps of plain descent take the loss from {start:.3f} to")
    print(f"  {end:.4f}")

    print("mixer. malformed calls are refused:")
    guard = Mixer("mlp_mixer", tokens=8, dim=6, depth=1, seed=9)
    good, good_cache = guard.forward(x)
    bad = [
        lambda: Mixer("mixer"),
        lambda: Mixer("mlp_mixer", tokens=0),
        lambda: Mixer("mlp_mixer", dim=0),
        lambda: Mixer("gmlp", hidden=0),
        lambda: Mixer("mlp_mixer", token_hidden=0),
        lambda: Mixer("mlp_mixer", depth=0),
        lambda: Mixer("convmixer", kernel=0),
        lambda: Mixer("convmixer", kernel=4),
        lambda: guard.forward(x[0]),
        lambda: guard.forward(x[:, :4]),
        lambda: guard.forward(x[..., :3]),
        lambda: guard.forward(np.zeros((0, 8, 6))),
        lambda: guard.forward(np.full_like(x, np.nan)),
        lambda: guard.backward(np.zeros((2, 8, 5)), good_cache),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, an")
    print("  even or zero kernel, a two-dimensional input, the wrong token")
    print("  or channel count, an empty batch, non-finite values, and a")
    print("  wrong gradient shape")
    print("mixer demo passed")


if __name__ == "__main__":
    demo()
