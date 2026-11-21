"""LoRA and parameter-efficient fine-tuning — nine ways to train a few numbers.

Chapter:   13-training
Companion: lora.md
Tier 2 · 9 variants: lora, qlora, dora, adalora, rslora, loha, lokr,
prefix_tuning, prompt_tuning
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("lora", "qlora", "dora", "adalora", "rslora", "loha", "lokr",
            "prefix_tuning", "prompt_tuning")
MATRIX_NAMES = ("Wq", "Wk", "Wv", "W1", "W2")
WEIGHT_NAMES = MATRIX_NAMES + ("b1", "b2")
WEIGHT_VARIANTS = ("lora", "qlora", "dora", "adalora", "rslora", "loha",
                   "lokr")

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


# --------------------------------------------------------------------------- #
# Four-bit quantisation
# --------------------------------------------------------------------------- #
def nf4_pack(weight: np.ndarray, block: int = 64,
             double: bool = True) -> dict:
    """Quantise a matrix to NF4 codes with per-block scales."""
    flat = np.asarray(weight, dtype=float).reshape(-1)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    scales = np.abs(blocks).max(axis=1) / 1.0
    scales = np.where(scales > 0.0, scales, 1.0)
    normalised = blocks / scales[:, None]
    codes = np.abs(normalised[..., None] - NF4).argmin(axis=-1)
    packed = {"codes": codes.astype(np.uint8), "length": weight.size,
              "shape": tuple(weight.shape)}
    if double:
        step = scales.max() / 127.0
        packed["scale_step"] = step
        packed["scales"] = np.round(scales / step).astype(np.int8)
    else:
        packed["scales"] = scales
    return packed


def nf4_unpack(packed: dict, block: int = 64) -> np.ndarray:
    if "scale_step" in packed:
        scales = packed["scales"].astype(float) * packed["scale_step"]
    else:
        scales = packed["scales"].astype(float)
    blocks = NF4[packed["codes"]] * scales[:, None]
    flat = blocks.reshape(-1)[: packed["length"]]
    return flat.reshape(packed["shape"])


def nf4_bytes(packed: dict) -> int:
    codes = (packed["codes"].size + 1) // 2
    scales = packed["scales"].nbytes
    return codes + scales + 8


# --------------------------------------------------------------------------- #
# The frozen block
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    probs = _softmax(logits)
    batch = logits.shape[0]
    loss = -float(np.mean(np.log(probs[np.arange(batch), labels] + 1e-12)))
    grad = probs.copy()
    grad[np.arange(batch), labels] -= 1.0
    return loss, grad / batch


def _block_forward(mats: dict, x: np.ndarray, prefix=None, prompt=None):
    """One attention block plus a tanh MLP, with optional soft tokens."""
    batch, tokens, dim = x.shape
    width = prompt.shape[0] if prompt is not None else 0
    if prompt is not None:
        tokens_in = np.concatenate(
            [np.broadcast_to(prompt, (batch,) + prompt.shape), x], axis=1)
    else:
        tokens_in = x
    query = tokens_in @ mats["Wq"]
    keys = tokens_in @ mats["Wk"]
    values = tokens_in @ mats["Wv"]
    if prefix is not None:
        keys = np.concatenate(
            [np.broadcast_to(prefix["K"], (batch,) + prefix["K"].shape),
             keys], axis=1)
        values = np.concatenate(
            [np.broadcast_to(prefix["V"], (batch,) + prefix["V"].shape),
             values], axis=1)
    scale = 1.0 / np.sqrt(dim)
    scores = query @ keys.transpose(0, 2, 1) * scale
    attention = _softmax(scores)
    context = attention @ values
    residual = context[:, -tokens:] + x
    pre = residual @ mats["W1"] + mats["b1"]
    hidden = np.tanh(pre)
    pooled = hidden.mean(axis=1)
    logits = pooled @ mats["W2"] + mats["b2"]
    cache = {"x": x, "tokens_in": tokens_in, "width": width,
             "prefix": 0 if prefix is None else prefix["K"].shape[0],
             "query": query, "keys": keys, "values": values,
             "attention": attention, "context": context,
             "residual": residual, "pre": pre, "hidden": hidden,
             "pooled": pooled, "scale": scale}
    return logits, cache


def _block_backward(mats: dict, cache: dict, dlogits: np.ndarray):
    hidden, residual = cache["hidden"], cache["residual"]
    tokens = cache["x"].shape[1]
    grads = {}
    grads["W2"] = cache["pooled"].T @ dlogits
    grads["b2"] = dlogits.sum(axis=0)
    dpooled = dlogits @ mats["W2"].T
    dpre = (dpooled[:, None, :] / tokens) * (1.0 - hidden ** 2)
    grads["W1"] = np.einsum("bti,btj->ij", residual, dpre)
    grads["b1"] = dpre.sum(axis=(0, 1))
    dresidual = dpre @ mats["W1"].T
    dcontext = np.zeros_like(cache["context"])
    dcontext[:, -tokens:] = dresidual
    dvalues = cache["attention"].transpose(0, 2, 1) @ dcontext
    dattention = dcontext @ cache["values"].transpose(0, 2, 1)
    dscores = cache["attention"] * (
        dattention - (dattention * cache["attention"]).sum(
            axis=-1, keepdims=True))
    dscores = dscores * cache["scale"]
    dquery = dscores @ cache["keys"]
    dkeys = dscores.transpose(0, 2, 1) @ cache["query"]
    grads["Wq"] = np.einsum("bti,btj->ij", cache["tokens_in"], dquery)
    dtokens = dquery @ mats["Wq"].T
    dprefix = None
    left = cache["prefix"]
    if left:
        dprefix = {"K": dkeys[:, :left].sum(axis=0),
                   "V": dvalues[:, :left].sum(axis=0)}
        dkeys = dkeys[:, left:]
        dvalues = dvalues[:, left:]
    grads["Wk"] = np.einsum("bti,btj->ij", cache["tokens_in"], dkeys)
    grads["Wv"] = np.einsum("bti,btj->ij", cache["tokens_in"], dvalues)
    dtokens = dtokens + dkeys @ mats["Wk"].T + dvalues @ mats["Wv"].T
    dprompt = None
    if cache["width"]:
        dprompt = dtokens[:, :cache["width"]].sum(axis=0)
    return grads, dprompt, dprefix


def _kronecker_shapes(dim: int) -> tuple[int, int]:
    first = 2 if dim % 2 == 0 else 1
    return first, dim // first


# --------------------------------------------------------------------------- #
# The adapters
# --------------------------------------------------------------------------- #
class PEFT:
    """A frozen block plus one trainable adapter.

    `forward(x, y)` returns the loss and a state; `backward(state)` fills
    `self.gradients` for trainable parameters only, and `step(lr)` applies
    them. The frozen matrices are never updated.
    """

    def __init__(self, variant: str = "lora", *, dim: int = 16,
                 tokens: int = 4, hidden: int = 16, classes: int = 3,
                 rank: int = 2, alpha: float = 1.0, prefix: int = 2,
                 prompt: int = 2, targets=("Wq", "Wv", "W1", "W2"),
                 ortho: float = 1e-3, block: int = 64, double: bool = True,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(dim, tokens, hidden, classes) < 1:
            raise ValueError("all sizes must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if rank < 1 or prefix < 1 or prompt < 1:
            raise ValueError("rank, prefix and prompt must be positive")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        if ortho < 0.0 or block < 2:
            raise ValueError("ortho must be non-negative and block at least 2")
        if not targets or any(name not in MATRIX_NAMES for name in targets):
            raise ValueError("targets must name at least one matrix")
        self.variant = variant
        self.dim = dim
        self.tokens = tokens
        self.hidden = hidden
        self.classes = classes
        self.rank = rank
        self.alpha = float(alpha)
        self.prefix = prefix
        self.prompt = prompt
        self.targets = tuple(targets) if variant in WEIGHT_VARIANTS else ()
        self.ortho = float(ortho)
        self.block = block
        self.double = double
        self.scale = (alpha / np.sqrt(rank) if variant == "rslora"
                      else alpha / rank)
        rng = np.random.default_rng(seed)
        self.weights = {
            "Wq": rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, dim)),
            "Wk": rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, dim)),
            "Wv": rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, dim)),
            "W1": rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, classes)),
            "b2": np.zeros(classes),
        }
        self.packed = {}
        if variant == "qlora":
            for name in MATRIX_NAMES:
                self.packed[name] = nf4_pack(self.weights[name], block, double)
                self.weights[name] = nf4_unpack(self.packed[name], block)
        self.shapes: dict[str, tuple] = {}
        self.trainable = self._build(rng)
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.trainable.items()}
        self._cache: dict[str, dict] = {}
        self._mats = None
        self._state = None

    # -- construction -------------------------------------------------------- #
    def _build(self, rng) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        if self.variant in ("prefix_tuning", "prompt_tuning"):
            if self.variant == "prefix_tuning":
                params["prefix.K"] = rng.normal(
                    0.0, 0.1, (self.prefix, self.dim))
                params["prefix.V"] = rng.normal(
                    0.0, 0.1, (self.prefix, self.dim))
            else:
                params["prompt.P"] = rng.normal(
                    0.0, 0.1, (self.prompt, self.dim))
            return params
        for name in self.targets:
            out_dim, in_dim = self.weights[name].shape
            if self.variant in ("lora", "qlora", "rslora", "dora"):
                params[f"{name}.A"] = rng.normal(
                    0.0, 1.0 / np.sqrt(in_dim), (self.rank, in_dim))
                params[f"{name}.B"] = np.zeros((out_dim, self.rank))
                if self.variant == "dora":
                    params[f"{name}.m"] = np.linalg.norm(
                        self.weights[name], axis=0)
            elif self.variant == "adalora":
                left = rng.normal(size=(out_dim, self.rank))
                left /= np.linalg.norm(left, axis=0, keepdims=True)
                right = rng.normal(size=(self.rank, in_dim))
                right /= np.linalg.norm(right, axis=1, keepdims=True)
                params[f"{name}.P"] = left
                params[f"{name}.Q"] = right
                params[f"{name}.lam"] = np.zeros(self.rank)
            elif self.variant == "loha":
                params[f"{name}.A1"] = rng.normal(
                    0.0, 1.0 / np.sqrt(in_dim), (self.rank, in_dim))
                params[f"{name}.B1"] = np.zeros((out_dim, self.rank))
                params[f"{name}.A2"] = rng.normal(
                    0.0, 1.0 / np.sqrt(in_dim), (self.rank, in_dim))
                params[f"{name}.B2"] = rng.normal(
                    0.0, 1.0 / np.sqrt(self.rank), (out_dim, self.rank))
            elif self.variant == "lokr":
                rows = _kronecker_shapes(out_dim)
                cols = _kronecker_shapes(in_dim)
                self.shapes[name] = (rows, cols)
                params[f"{name}.A1"] = rng.normal(0.0, 0.5, (1, cols[0]))
                params[f"{name}.A2"] = rng.normal(0.0, 0.5, (1, cols[1]))
                params[f"{name}.B1"] = np.zeros((rows[0], 1))
                params[f"{name}.B2"] = rng.normal(0.0, 0.5, (rows[1], 1))
            else:
                raise ValueError(f"unknown variant {self.variant!r}")
        return params

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.trainable.values()))

    @property
    def n_base(self) -> int:
        return int(sum(value.size for value in self.weights.values()))

    # -- adapters ------------------------------------------------------------ #
    def _effective(self, name: str) -> np.ndarray:
        base = self.weights[name]
        if name not in self.targets:
            return base
        if self.variant in ("lora", "qlora", "rslora"):
            left = self.trainable[f"{name}.A"]
            right = self.trainable[f"{name}.B"]
            self._cache[name] = {"A": left, "B": right}
            return base + self.scale * (right @ left)
        if self.variant == "dora":
            left = self.trainable[f"{name}.A"]
            right = self.trainable[f"{name}.B"]
            value = base + self.scale * (right @ left)
            norm = np.maximum(np.linalg.norm(value, axis=0), 1e-12)
            unit = value / norm
            self._cache[name] = {"A": left, "B": right, "unit": unit,
                                 "norm": norm}
            return self.trainable[f"{name}.m"] * unit
        if self.variant == "adalora":
            left = self.trainable[f"{name}.P"]
            middle = self.trainable[f"{name}.lam"]
            right = self.trainable[f"{name}.Q"]
            self._cache[name] = {"P": left, "lam": middle, "Q": right}
            return base + (left * middle) @ right
        if self.variant == "loha":
            left = self.trainable[f"{name}.B1"] @ self.trainable[f"{name}.A1"]
            right = (self.trainable[f"{name}.B2"]
                     @ self.trainable[f"{name}.A2"])
            self._cache[name] = {"left": left, "right": right}
            return base + self.scale * (left * right)
        if self.variant == "lokr":
            rows, cols = self.shapes[name]
            self._cache[name] = {"rows": rows, "cols": cols}
            first = (self.trainable[f"{name}.B1"]
                     @ self.trainable[f"{name}.A1"])
            second = (self.trainable[f"{name}.B2"]
                      @ self.trainable[f"{name}.A2"])
            return base + self.scale * np.kron(first, second)
        raise ValueError(f"unknown variant {self.variant!r}")

    def _delta_backward(self, name: str, dweight: np.ndarray) -> None:
        cached = self._cache[name]
        if self.variant in ("lora", "qlora", "rslora"):
            self.gradients[f"{name}.A"] += self.scale * (
                cached["B"].T @ dweight)
            self.gradients[f"{name}.B"] += self.scale * (
                dweight @ cached["A"].T)
        elif self.variant == "dora":
            unit, norm = cached["unit"], cached["norm"]
            self.gradients[f"{name}.m"] += (dweight * unit).sum(axis=0)
            dunit = dweight * self.trainable[f"{name}.m"]
            dv = (dunit - unit * (unit * dunit).sum(axis=0)) / norm
            self.gradients[f"{name}.A"] += self.scale * (
                cached["B"].T @ dv)
            self.gradients[f"{name}.B"] += self.scale * (
                dv @ cached["A"].T)
        elif self.variant == "adalora":
            left, middle, right = cached["P"], cached["lam"], cached["Q"]
            self.gradients[f"{name}.lam"] += np.einsum(
                "ok,oi,ki->k", left, dweight, right)
            self.gradients[f"{name}.P"] += (dweight @ right.T) * middle
            self.gradients[f"{name}.Q"] += (left.T @ dweight) * middle[:, None]
            weight = self.ortho
            if weight:
                left_error = left.T @ left - np.eye(self.rank)
                right_error = right @ right.T - np.eye(self.rank)
                self.gradients[f"{name}.P"] += 4.0 * weight * (
                    left @ left_error)
                self.gradients[f"{name}.Q"] += 4.0 * weight * (
                    right_error @ right)
        elif self.variant == "loha":
            dleft = dweight * cached["right"]
            dright = dweight * cached["left"]
            self.gradients[f"{name}.A1"] += self.scale * (
                self.trainable[f"{name}.B1"].T @ dleft)
            self.gradients[f"{name}.B1"] += self.scale * (
                dleft @ self.trainable[f"{name}.A1"].T)
            self.gradients[f"{name}.A2"] += self.scale * (
                self.trainable[f"{name}.B2"].T @ dright)
            self.gradients[f"{name}.B2"] += self.scale * (
                dright @ self.trainable[f"{name}.A2"].T)
        elif self.variant == "lokr":
            rows, cols = cached["rows"], cached["cols"]
            shaped = dweight.reshape(rows[0], rows[1], cols[0], cols[1])
            first = (self.trainable[f"{name}.B1"]
                     @ self.trainable[f"{name}.A1"])
            second = (self.trainable[f"{name}.B2"]
                      @ self.trainable[f"{name}.A2"])
            dfirst = np.einsum("ijkl,jl->ik", shaped, second)
            dsecond = np.einsum("ijkl,ik->jl", shaped, first)
            self.gradients[f"{name}.A1"] += self.scale * (
                self.trainable[f"{name}.B1"].T @ dfirst)
            self.gradients[f"{name}.B1"] += self.scale * (
                dfirst @ self.trainable[f"{name}.A1"].T)
            self.gradients[f"{name}.A2"] += self.scale * (
                self.trainable[f"{name}.B2"].T @ dsecond)
            self.gradients[f"{name}.B2"] += self.scale * (
                dsecond @ self.trainable[f"{name}.A2"].T)
        else:
            raise ValueError(f"unknown variant {self.variant!r}")

    # -- training ------------------------------------------------------------ #
    def _check(self, x, y):
        values = np.asarray(x, dtype=float)
        if values.ndim != 3:
            raise ValueError("x must be (batch, tokens, dim)")
        if values.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if values.shape[1] != self.tokens or values.shape[2] != self.dim:
            raise ValueError("x has the wrong token or feature count")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must be finite")
        labels = np.asarray(y)
        if labels.shape != (values.shape[0],):
            raise ValueError("y must hold one label per example")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return values, labels

    def forward(self, x, y):
        x, labels = self._check(x, y)
        self._cache = {}
        mats = dict(self.weights)
        for name in self.targets:
            mats[name] = self._effective(name)
        prefix = prompt = None
        if self.variant == "prefix_tuning":
            prefix = {"K": self.trainable["prefix.K"],
                      "V": self.trainable["prefix.V"]}
        if self.variant == "prompt_tuning":
            prompt = self.trainable["prompt.P"]
        logits, cache = _block_forward(mats, x, prefix, prompt)
        loss, dlogits = _cross_entropy(logits, labels)
        if self.variant == "adalora" and self.ortho:
            loss += self.ortho * self._penalty()
        self._mats = mats
        self._state = {"cache": cache, "dlogits": dlogits}
        return loss, self._state

    def _penalty(self) -> float:
        total = 0.0
        for name in self.targets:
            left = self.trainable[f"{name}.P"]
            right = self.trainable[f"{name}.Q"]
            total += float(np.sum((left.T @ left - np.eye(self.rank)) ** 2))
            total += float(np.sum((right @ right.T - np.eye(self.rank)) ** 2))
        return total

    def backward(self, state=None) -> None:
        state = self._state if state is None else state
        if state is None or self._mats is None:
            raise ValueError("forward must run before backward")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        grads, dprompt, dprefix = _block_backward(
            self._mats, state["cache"], state["dlogits"])
        for name in self.targets:
            self._delta_backward(name, grads[name])
        if dprompt is not None:
            self.gradients["prompt.P"] += dprompt
        if dprefix is not None:
            self.gradients["prefix.K"] += dprefix["K"]
            self.gradients["prefix.V"] += dprefix["V"]

    def step(self, lr: float) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        for key, gradient in self.gradients.items():
            self.trainable[key] -= lr * gradient

    def loss(self, x, y) -> float:
        return self.forward(x, y)[0]

    def accuracy(self, x, y) -> float:
        x, labels = self._check(x, y)
        self._cache = {}
        mats = dict(self.weights)
        for name in self.targets:
            mats[name] = self._effective(name)
        prefix = prompt = None
        if self.variant == "prefix_tuning":
            prefix = {"K": self.trainable["prefix.K"],
                      "V": self.trainable["prefix.V"]}
        if self.variant == "prompt_tuning":
            prompt = self.trainable["prompt.P"]
        logits, _ = _block_forward(mats, x, prefix, prompt)
        return float(np.mean(np.argmax(logits, axis=-1) == labels))

    def fit(self, x, y, *, steps: int = 300, lr: float = 0.2) -> list[float]:
        if steps < 1:
            raise ValueError("steps must be positive")
        history = []
        for _ in range(steps):
            loss, state = self.forward(x, y)
            self.backward(state)
            self.step(lr)
            history.append(loss)
        return history


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _centres(seed: int, classes: int = 3, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(classes, dim)) * 1.4


def _task(seed: int, centres: np.ndarray, per_class: int = 40,
          tokens: int = 4, noise: float = 0.6):
    rng = np.random.default_rng(seed)
    classes, dim = centres.shape
    labels = np.repeat(np.arange(classes), per_class)
    x = centres[labels][:, None, :] + rng.normal(
        0.0, noise, (labels.size, tokens, dim))
    order = rng.permutation(labels.size)
    return x[order], labels[order]


def _train_base(weights: dict, x, y, steps: int, lr: float) -> None:
    for _ in range(steps):
        logits, cache = _block_forward(weights, x)
        _, dlogits = _cross_entropy(logits, y)
        grads, _, _ = _block_backward(weights, cache, dlogits)
        for name in weights:
            weights[name] -= lr * grads[name]


def demo() -> None:
    centres_a = _centres(1)
    centres_b = _centres(4)
    base_x, base_y = _task(3, centres_a)
    tune_x, tune_y = _task(4, centres_b, per_class=20)
    valid_x, valid_y = _task(5, centres_b, per_class=20)
    test_x, test_y = _task(6, centres_b, per_class=40)
    base_weights = PEFT("lora", seed=5).weights
    _train_base(base_weights, base_x, base_y, 300, 0.5)
    frozen = PEFT("lora", seed=5)
    frozen.weights = base_weights
    frozen_score = frozen.accuracy(test_x, test_y)

    print("lora. a frozen model and nine trainable adapters:")
    print(f"  the base block has {frozen.n_base} parameters and scores"
          f" {frozen_score:.3f}")
    print("  on a task with new class centres; every adapter below trains")
    print("  only the numbers in the last column")

    print("lora. adapters start at the base model:")
    for variant in WEIGHT_VARIANTS:
        model = PEFT(variant, seed=6)
        worst = max(float(np.abs(model._effective(name)
                                 - model.weights[name]).max())
                    for name in model.targets)
        if variant == "dora":
            assert worst < 1e-12
        else:
            assert worst == 0.0
        print(f"    {variant:<8s} largest initial change {worst:.1e}")

    print("lora. finite differences agree with every adapter gradient:")
    errors = {}
    for variant in VARIANTS:
        model = PEFT(variant, dim=4, tokens=3, hidden=4, classes=2,
                     rank=1, prefix=1, prompt=1, seed=7)
        small_x, small_y = _task(8, _centres(7, classes=2, dim=4),
                                 per_class=6, tokens=3)
        for _ in range(5):
            loss, state = model.forward(small_x, small_y)
            model.backward(state)
            model.step(0.2)
        worst = 0.0
        for key, value in model.trainable.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.loss(small_x, small_y)
                value[index] = old - 1e-6
                minus = model.loss(small_x, small_y)
                value[index] = old
                model.forward(small_x, small_y)
                model.backward()
                numeric = (plus - minus) / 2e-6
                worst = max(worst, abs(numeric - model.gradients[key][index]))
        errors[variant] = worst
        print(f"    {variant:<15s} worst error {worst:.1e}")
    assert max(errors.values()) < 1e-6

    print("lora. the rank is the parameter budget:")
    for rank in (1, 2, 4):
        dense = sum(frozen.weights[name].size
                    for name in ("Wq", "Wv", "W1", "W2"))
        lora = PEFT("lora", rank=rank).n_parameters
        loha = PEFT("loha", rank=rank).n_parameters
        print(f"    rank {rank}: lora {lora:3d}, loha {loha:3d},"
              f" dense {dense:3d}")
    assert PEFT("lora", rank=4).n_parameters > PEFT("lora", rank=1).n_parameters
    assert (PEFT("lora", rank=1).n_parameters
            < frozen.n_base)

    print("lora. rslora keeps the scale with the rank:")
    for rank in (1, 4, 16):
        plain = PEFT("lora", rank=rank, alpha=8.0).scale
        root = PEFT("rslora", rank=rank, alpha=8.0).scale
        print(f"    rank {rank:2d}: alpha/r {plain:.3f}  alpha/sqrt(r) "
              f"{root:.3f}")
    assert np.isclose(PEFT("rslora", rank=16, alpha=8.0).scale, 2.0)
    assert np.isclose(PEFT("lora", rank=16, alpha=8.0).scale, 0.5)

    print("lora. dora keeps the magnitude separate from the direction:")
    model = PEFT("dora", seed=9)
    model.fit(tune_x, tune_y, steps=200, lr=0.2)
    worst = 0.0
    for name in model.targets:
        effective = model._effective(name)
        norms = np.linalg.norm(effective, axis=0)
        worst = max(worst, float(np.abs(
            norms - model.trainable[f"{name}.m"]).max()))
        direction = effective / norms
        worst = max(worst, float(np.abs(
            np.linalg.norm(direction, axis=0) - 1.0).max()))
    assert worst < 1e-10
    print(f"  every column norm equals its magnitude to {worst:.1e}, and")
    print("  every column of the direction is a unit vector by construction")

    print("lora. adalora ranks the factors by importance:")
    model = PEFT("adalora", rank=4, seed=10)
    model.fit(tune_x, tune_y, steps=300, lr=0.2)
    floor = model.loss(valid_x, valid_y)
    scores = {}
    for name in model.targets:
        left = model.trainable[f"{name}.P"]
        right = model.trainable[f"{name}.Q"]
        middle = model.trainable[f"{name}.lam"]
        scores[name] = np.abs(middle) * np.linalg.norm(
            left, axis=0) * np.linalg.norm(right, axis=1)
    total = sum(scores.values())
    order = np.argsort(total)
    keep = list(order[len(order) // 2:])
    drop = list(order[:len(order) // 2])
    saved = {name: model.trainable[f"{name}.lam"].copy()
             for name in model.targets}

    def cost(ranks):
        for name in model.targets:
            model.trainable[f"{name}.lam"] = np.where(
                np.isin(np.arange(model.rank), ranks), saved[name], 0.0)
        value = model.loss(valid_x, valid_y) - floor
        for name in model.targets:
            model.trainable[f"{name}.lam"] = saved[name].copy()
        return value

    print(f"  the four rank scores are {np.round(total, 2)}")
    print(f"  dropping the two lowest costs {cost(keep):+.4f} of validation")
    print(f"  dropping the two highest costs {cost(drop):+.4f}")
    assert cost(keep) < cost(drop)

    print("lora. qlora stores the frozen base in four bits:")
    model = PEFT("qlora", seed=11)
    reference = PEFT("lora", seed=11).weights
    worst = max(float(np.abs(model.weights[name] - reference[name]).max())
                for name in MATRIX_NAMES)
    large = np.random.default_rng(0).normal(size=(256, 256))
    packed_bytes = nf4_bytes(nf4_pack(large, 64, True))
    ratio = 4 * large.size / packed_bytes
    bits = 8 * packed_bytes / large.size
    print(f"    a 256x256 matrix: {4 * large.size} bytes fp32,"
          f" {packed_bytes} bytes NF4")
    print(f"    that is {ratio:.2f}x smaller, {bits:.2f} bits per weight")
    print(f"    largest dequantisation error on the demo block {worst:.3f}")
    assert ratio > 7.5
    assert worst < 0.2

    print("lora. nine adapters, one training loop:")
    scores = {}
    params = {}
    for variant in VARIANTS:
        model = PEFT(variant, seed=12)
        model.weights = {name: value.copy()
                         for name, value in base_weights.items()}
        if variant == "dora":
            for name in model.targets:
                model.trainable[f"{name}.m"][...] = np.linalg.norm(
                    model.weights[name], axis=0)
        if variant == "qlora":
            for name in MATRIX_NAMES:
                model.packed[name] = nf4_pack(model.weights[name],
                                              model.block, model.double)
                model.weights[name] = nf4_unpack(model.packed[name],
                                                 model.block)
        model.fit(tune_x, tune_y, steps=300, lr=0.2)
        scores[variant] = model.accuracy(test_x, test_y)
        params[variant] = model.n_parameters
        print(f"    {variant:<15s} {scores[variant]:.3f} accuracy"
              f"  {params[variant]:4d} trainable")
    assert all(score > 0.6 for score in scores.values())
    assert max(params.values()) < frozen.n_base // 2
    print(f"  the frozen base scores {frozen_score:.3f}; the seven weight")
    print("  adapters recover the task completely, while the two token")
    print("  adapters reach 0.68: a soft prompt steers a frozen head, it")
    print("  cannot rewrite it")

    print("lora. malformed calls are refused:")
    guard = PEFT("lora", dim=4, tokens=3, hidden=4, classes=2, seed=13)
    good_x, good_y = tune_x[:4, :3, :4], tune_y[:4] % 2
    guard.forward(good_x, good_y)
    bad = [
        lambda: PEFT("lora_v2"),
        lambda: PEFT("lora", dim=0),
        lambda: PEFT("lora", tokens=0),
        lambda: PEFT("lora", hidden=0),
        lambda: PEFT("lora", classes=1),
        lambda: PEFT("lora", rank=0),
        lambda: PEFT("lora", alpha=0.0),
        lambda: PEFT("prefix_tuning", prefix=0),
        lambda: PEFT("prompt_tuning", prompt=0),
        lambda: PEFT("lora", targets=()),
        lambda: PEFT("lora", targets=("W9",)),
        lambda: PEFT("adalora", ortho=-1.0),
        lambda: PEFT("qlora", block=1),
        lambda: guard.forward(good_x[0], good_y),
        lambda: guard.forward(good_x[:, :2], good_y),
        lambda: guard.forward(good_x[:, :, :3], good_y),
        lambda: guard.forward(np.zeros((0, 3, 4)), good_y[:0]),
        lambda: guard.forward(np.full_like(good_x, np.nan), good_y),
        lambda: guard.forward(good_x, good_y[:2]),
        lambda: guard.forward(good_x, good_y.astype(float)),
        lambda: guard.forward(good_x, good_y + 5),
        lambda: guard.step(0.0),
        lambda: PEFT("lora").backward(),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, one")
    print("  class, a zero rank or weight, an empty prefix or prompt, an")
    print("  empty or unknown target set, a negative penalty, a one-entry")
    print("  quantisation block, a 2-d input, wrong tokens or features, an")
    print("  empty or non-finite batch, wrong or float or out-of-range")
    print("  labels, a non-positive learning rate, and a backward pass")
    print("  before any forward pass")
    print("lora demo passed")


if __name__ == "__main__":
    demo()
