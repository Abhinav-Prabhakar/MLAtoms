"""Mixture of experts — sparse capacity without sparse gradients.

Chapter:   12-neural-blocks
Companion: moe.md
Tier 2 · 8 variants: top1, topk, expert_choice, switch, soft_moe,
capacity_factor, load_balance_loss, shared_expert
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("top1", "topk", "expert_choice", "switch", "soft_moe",
            "capacity_factor", "load_balance_loss", "shared_expert")

TOP_ONE = ("top1", "switch")
AUXILIARY = ("switch", "load_balance_loss", "soft_moe")
CAPACITY = ("capacity_factor", "expert_choice")

_GELU = 0.7978845608028654
_CUBIC = 0.044715


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    tanh = np.tanh(inner)
    return (0.5 * (1.0 + tanh)
            + 0.5 * x * (1.0 - tanh ** 2) * _GELU * (1.0 + 3.0 * _CUBIC * x ** 2))


def _add_expert(parameters: dict[str, np.ndarray], name: str, dim: int,
                hidden: int, rng: np.random.Generator) -> None:
    parameters[f"{name}.W1"] = rng.normal(
        0.0, np.sqrt(2.0 / dim), (dim, hidden))
    parameters[f"{name}.b1"] = np.zeros(hidden)
    parameters[f"{name}.W2"] = rng.normal(
        0.0, np.sqrt(2.0 / hidden), (hidden, dim))
    parameters[f"{name}.b2"] = np.zeros(dim)


def _expert_forward(parameters: dict[str, np.ndarray], name: str,
                    x: np.ndarray):
    pre = x @ parameters[f"{name}.W1"] + parameters[f"{name}.b1"]
    hidden = _gelu(pre)
    out = hidden @ parameters[f"{name}.W2"] + parameters[f"{name}.b2"]
    return out, (x, pre, hidden)


def _expert_backward(parameters: dict[str, np.ndarray],
                     gradients: dict[str, np.ndarray], name: str,
                     cache, gradient: np.ndarray) -> None:
    x, pre, hidden = cache
    gradients[f"{name}.W2"] += hidden.T @ gradient
    gradients[f"{name}.b2"] += gradient.sum(axis=0)
    d_pre = (gradient @ parameters[f"{name}.W2"].T) * _gelu_grad(pre)
    gradients[f"{name}.W1"] += x.T @ d_pre
    gradients[f"{name}.b1"] += d_pre.sum(axis=0)


def _load_balance(probability: np.ndarray, fractions: np.ndarray) -> float:
    """Switch-style auxiliary loss, minimal at uniform routing."""
    experts = probability.shape[1]
    return float(experts * np.sum(fractions * probability.mean(axis=0)))


def _balance_gradient(probability: np.ndarray,
                      fractions: np.ndarray) -> np.ndarray:
    """Gradient of `_load_balance` with respect to the router logits."""
    experts, n = probability.shape[1], probability.shape[0]
    d_probability = experts * fractions[None, :] / n
    return probability * (d_probability
                          - (probability * d_probability).sum(axis=1,
                                                              keepdims=True))


# --------------------------------------------------------------------------- #
# The mixture
# --------------------------------------------------------------------------- #
class MoE:
    """A batch of small experts behind one router.

    Every variant splits the batch by a router logit matrix, so a token is a
    row and nothing in the layer mixes rows. `forward` returns the mixed
    output and the routing record; `backward` differentiates the task loss
    through the selected experts and the router logits.
    """

    def __init__(self, variant: str = "topk", *, dim: int = 6,
                 hidden: int = 8, experts: int = 4, top_k: int = 2,
                 capacity_factor: float = 1.0, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if dim < 1 or hidden < 1 or experts < 1:
            raise ValueError("dim, hidden and experts must be positive")
        if top_k < 1 or top_k > experts:
            raise ValueError("top_k must be between one and the expert count")
        if not np.isfinite(capacity_factor) or capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be finite and positive")
        self.variant = variant
        self.dim = dim
        self.hidden = hidden
        self.experts = experts
        self.top_k = 1 if variant in TOP_ONE else top_k
        self.capacity_factor = capacity_factor
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.parameters: dict[str, np.ndarray] = {}
        for expert in range(experts):
            _add_expert(self.parameters, f"expert{expert}", dim, hidden, rng)
        self.parameters["router.W"] = rng.normal(
            0.0, 1.0 / np.sqrt(dim), (dim, experts))
        self.parameters["router.b"] = np.zeros(experts)
        if variant == "shared_expert":
            _add_expert(self.parameters, "shared", dim, hidden, rng)
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    # -- helpers ------------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    def _check_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must be (tokens, dim)")
        if x.shape[0] < 1:
            raise ValueError("need at least one token")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        return x

    def _capacity(self, n: int) -> int:
        capacity = int(np.ceil(self.capacity_factor * n / self.experts))
        return int(min(max(capacity, 1), n))

    def _keep(self, n: int, width: int, indices: np.ndarray, capacity: int):
        """Keep at most `capacity` assignments per expert, in token order."""
        used = np.zeros(self.experts, dtype=int)
        kept = np.zeros((n, width), dtype=bool)
        for row in range(n):
            for slot in range(width):
                expert = int(indices[row, slot])
                if used[expert] < capacity:
                    kept[row, slot] = True
                    used[expert] += 1
        return kept, used

    def expert_output(self, expert: int | str, x: np.ndarray) -> np.ndarray:
        """Evaluate one expert without touching the router."""
        x = self._check_input(x)
        name = "shared" if expert == "shared" else f"expert{int(expert)}"
        if name not in {key.rsplit(".", 1)[0] for key in self.parameters}:
            raise ValueError("no such expert")
        return _expert_forward(self.parameters, name, x)[0]

    # -- forward ------------------------------------------------------------- #
    def forward(self, x: np.ndarray):
        x = self._check_input(x)
        n = x.shape[0]
        p = self.parameters
        logits = x @ p["router.W"] + p["router.b"]
        probability = _softmax(logits)
        output = np.zeros_like(x)
        caches: dict[int, tuple] = {}
        info = {"logits": logits, "probability": probability, "n": n,
                "capacity": None, "kept": None, "aux": None}

        if self.variant == "soft_moe":
            rows = np.arange(n)
            for expert in range(self.experts):
                out, cache = _expert_forward(p, f"expert{expert}", x)
                output += probability[:, expert, None] * out
                caches[expert] = (rows, None, cache, out)
            info["gates"] = probability
            info["counts"] = probability.sum(axis=0)
            info["fractions"] = probability.mean(axis=0)
            info["caches"] = caches

        elif self.variant == "expert_choice":
            capacity = self._capacity(n)
            chosen = np.zeros((n, self.experts), dtype=bool)
            for expert in range(self.experts):
                order = np.argsort(-logits[:, expert])
                chosen[order[:capacity], expert] = True
            mass = (probability * chosen).sum(axis=1, keepdims=True)
            gates = (probability * chosen) / np.maximum(mass, 1e-12)
            for expert in range(self.experts):
                rows = np.flatnonzero(chosen[:, expert])
                if rows.size == 0:
                    continue
                out, cache = _expert_forward(p, f"expert{expert}", x[rows])
                output[rows] += gates[rows, expert, None] * out
                caches[expert] = (rows, None, cache, out)
            counts = chosen.sum(axis=0)
            info.update(chosen=chosen, gates=gates, mass=mass,
                        counts=counts, capacity=capacity,
                        fractions=counts / (self.experts * capacity),
                        caches=caches)

        else:
            width = self.top_k
            indices = np.argsort(-logits, axis=1)[:, :width]
            selected = np.take_along_axis(logits, indices, axis=1)
            gates = _softmax(selected)
            if self.variant == "capacity_factor":
                capacity = self._capacity(n)
                kept, used = self._keep(n, width, indices, capacity)
                gates = gates * kept
                info.update(capacity=capacity, kept=kept, used=used)
            for expert in range(self.experts):
                rows, slots = np.nonzero(indices == expert)
                if rows.size == 0:
                    continue
                out, cache = _expert_forward(p, f"expert{expert}", x[rows])
                output[rows] += gates[rows, slots, None] * out
                caches[expert] = (rows, slots, cache, out)
            counts = np.bincount(indices.ravel(), minlength=self.experts)
            info.update(indices=indices, gates=gates, counts=counts,
                        fractions=counts / (n * width), caches=caches)

        if self.variant == "shared_expert":
            shared, shared_cache = _expert_forward(p, "shared", x)
            output += shared
            info.update(shared=shared, shared_cache=shared_cache)

        if self.variant in AUXILIARY:
            info["aux"] = _load_balance(probability, info["fractions"])
        return output, info

    # -- backward ------------------------------------------------------------ #
    def backward(self, x: np.ndarray, info, gradient: np.ndarray) -> None:
        x = self._check_input(x)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != x.shape:
            raise ValueError("gradient must have the shape of the output")
        p, g = self.parameters, self.gradients
        for key in g:
            g[key][...] = 0.0
        n = x.shape[0]
        probability = info["probability"]
        dz = np.zeros_like(probability)

        if self.variant == "soft_moe":
            d_probability = np.zeros_like(probability)
            for expert, (rows, _, cache, out) in info["caches"].items():
                _expert_backward(p, g, f"expert{expert}", cache,
                                 gradient[rows]
                                 * probability[rows, expert, None])
                d_probability[rows, expert] = (gradient[rows] * out).sum(axis=1)
            dz = probability * (d_probability
                                - (probability * d_probability).sum(
                                    axis=1, keepdims=True))

        elif self.variant == "expert_choice":
            gates, mass = info["gates"], info["mass"]
            d_gate = np.zeros_like(probability)
            for expert, (rows, _, cache, out) in info["caches"].items():
                _expert_backward(p, g, f"expert{expert}", cache,
                                 gradient[rows] * gates[rows, expert, None])
                d_gate[rows, expert] = (gradient[rows] * out).sum(axis=1)
            correction = (d_gate * gates).sum(axis=1, keepdims=True)
            d_probability = np.where(
                info["chosen"],
                (d_gate - correction) / np.maximum(mass, 1e-12), 0.0)
            dz = probability * (d_probability
                                - (probability * d_probability).sum(
                                    axis=1, keepdims=True))

        else:
            gates, indices = info["gates"], info["indices"]
            d_gate = np.zeros_like(gates)
            for expert, (rows, slots, cache, out) in info["caches"].items():
                _expert_backward(p, g, f"expert{expert}", cache,
                                 gradient[rows] * gates[rows, slots, None])
                d_gate[rows, slots] = (gradient[rows] * out).sum(axis=1)
            d_selected = gates * (d_gate - (gates * d_gate).sum(
                axis=1, keepdims=True))
            rows = np.arange(n)
            for slot in range(indices.shape[1]):
                np.add.at(dz, (rows, indices[:, slot]), d_selected[:, slot])

        if self.variant == "shared_expert":
            _expert_backward(p, g, "shared", info["shared_cache"], gradient)

        g["router.W"] += x.T @ dz
        g["router.b"] += dz.sum(axis=0)

    def aux_gradient(self, x: np.ndarray, info) -> dict[str, np.ndarray]:
        """Gradient of the auxiliary loss with respect to the router."""
        if info.get("aux") is None:
            raise ValueError("this variant has no auxiliary loss")
        x = self._check_input(x)
        probability = info["probability"]
        experts, n = probability.shape[1], probability.shape[0]
        if self.variant == "soft_moe":
            d_probability = 2.0 * experts * probability.mean(
                axis=0)[None, :] / n
        else:
            d_probability = experts * info["fractions"][None, :] / n
        dz = probability * (d_probability
                            - (probability * d_probability).sum(
                                axis=1, keepdims=True))
        return {"router.W": x.T @ dz, "router.b": dz.sum(axis=0)}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _assert_close(left: np.ndarray, right: np.ndarray, tol: float) -> float:
    error = float(np.abs(left - right).max())
    assert error < tol
    return error


def demo() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(24, 6))

    print("moe. a top-1 layer is an argmax plus a gather:")
    model = MoE("top1", dim=6, hidden=8, experts=4, seed=1)
    out, info = model.forward(x)
    chosen = info["indices"][:, 0]
    for expert in range(4):
        rows = chosen == expert
        if rows.any():
            _assert_close(out[rows], model.expert_output(expert, x[rows]),
                          1e-12)
    assert np.array_equal(info["indices"].ravel(), info["logits"].argmax(axis=1))
    assert np.allclose(info["gates"], 1.0)
    print(f"  load {np.bincount(chosen, minlength=4).tolist()}, and every row of")
    print("  the output is exactly one expert's output")

    print("moe. top-k renormalises over the selected logits:")
    model = MoE("topk", dim=6, hidden=8, experts=4, top_k=2, seed=1)
    out, info = model.forward(x)
    selected = np.take_along_axis(info["logits"], info["indices"], axis=1)
    _assert_close(info["gates"], _softmax(selected), 1e-12)
    _assert_close(info["gates"].sum(axis=1), np.ones(24), 1e-12)
    model.parameters["router.W"][...] = 0.0
    model.parameters["router.b"][...] = 0.0
    _, flat = model.forward(x)
    _assert_close(flat["gates"], np.full((24, 2), 0.5), 1e-12)
    top_one = MoE("top1", dim=6, hidden=8, experts=4, seed=1)
    one_of_k = MoE("topk", dim=6, hidden=8, experts=4, top_k=1, seed=1)
    _assert_close(top_one.forward(x)[0], one_of_k.forward(x)[0], 1e-12)
    print("  two equal logits give gates of one half; with k=1 the same seed")
    print("  reproduces the top-1 output bit for bit")

    print("moe. a capacity factor drops the overflow:")
    model = MoE("capacity_factor", dim=6, hidden=8, experts=4, top_k=1,
                capacity_factor=0.5, seed=3)
    model.parameters["router.b"][...] = np.array([6.0, 0.0, 0.0, 0.0])
    out, info = model.forward(x)
    kept = info["kept"][:, 0]
    assert info["capacity"] == 3 and kept.sum() == 3
    assert info["used"].tolist() == [3, 0, 0, 0]
    _assert_close(out[~kept], np.zeros_like(out[~kept]), 1e-12)
    _assert_close(info["gates"][~kept], np.zeros((21, 1)), 1e-12)
    print(f"  all 24 tokens want expert 0, capacity {info['capacity']} keeps")
    print("  three, and the other 21 outputs are exactly zero")

    print("moe. the auxiliary loss counts usage and probability:")
    experts, n = 4, 48
    uniform = np.full((n, experts), 1.0 / experts)
    assert abs(_load_balance(uniform, np.full(experts, 1.0 / experts)) - 1.0) \
        < 1e-12
    collapsed = np.zeros((n, experts))
    collapsed[:, 0] = 1.0
    assert abs(_load_balance(collapsed, np.eye(experts)[0]) - experts) < 1e-12
    features = rng.normal(size=(n, 6))
    router = rng.normal(0.0, 1.0 / np.sqrt(6), (6, experts))
    bias = np.array([4.0, 0.0, 0.0, 0.0])
    start = None
    for _ in range(600):
        logits = features @ router + bias
        probability = _softmax(logits)
        hard = np.bincount(logits.argmax(axis=1),
                           minlength=experts) / n
        aux = _load_balance(probability, hard)
        start = aux if start is None else start
        dz = _balance_gradient(probability, hard)
        router -= 0.5 * features.T @ dz
        bias -= 0.5 * dz.sum(axis=0)
    final = np.bincount(logits.argmax(axis=1), minlength=experts)
    assert start > 3.0 and aux < 1.2
    assert final.max() - final.min() <= 2
    print(f"  uniform routing gives 1.0 and one-hot routing {experts}. From")
    print(f"  {start:.3f}, 600 router steps reach {aux:.3f} with load"
          f" {final.tolist()}")

    print("moe. expert choice fills every expert to capacity:")
    model = MoE("expert_choice", dim=6, hidden=8, experts=4,
                capacity_factor=1.0, seed=5)
    out, info = model.forward(x)
    assert info["capacity"] == 6
    assert np.all(info["counts"] == 6)
    assert np.all(info["chosen"].sum(axis=0) == 6)
    empty = info["mass"][:, 0] == 0.0
    _assert_close(out[empty], np.zeros_like(out[empty]), 1e-12)
    print(f"  capacity {info['capacity']} and counts {info['counts'].tolist()}:")
    print(f"  {int(empty.sum())} tokens were skipped by all four experts and")
    print("  their output is zero, which is the price of a hard quota")

    print("moe. soft routing keeps every expert alive:")
    model = MoE("soft_moe", dim=6, hidden=8, experts=4, seed=6)
    out, info = model.forward(x)
    _assert_close(info["gates"], info["probability"], 1e-12)
    explicit = sum(info["gates"][:, expert, None]
                   * model.expert_output(expert, x) for expert in range(4))
    _assert_close(out, explicit, 1e-12)
    model.backward(x, info, np.ones_like(out))
    for expert in range(4):
        assert np.any(model.gradients[f"expert{expert}.W1"])
    print(f"  gate range {info['gates'].min():.4f} to {info['gates'].max():.4f};")
    print("  every expert contributes to every token and receives gradient")

    print("moe. a shared expert runs for every token:")
    shared = MoE("shared_expert", dim=6, hidden=8, experts=4, top_k=2, seed=8)
    plain = MoE("topk", dim=6, hidden=8, experts=4, top_k=2, seed=8)
    out, info = shared.forward(x)
    routed, _ = plain.forward(x)
    _assert_close(info["shared"], out - routed, 1e-12)
    before = info["shared"].copy()
    shared.parameters["router.b"] += 1.0
    _, changed = shared.forward(x)
    _assert_close(changed["shared"], before, 1e-12)
    print(f"  the shared branch is {np.abs(before).mean():.3f} on average and")
    print("  does not move when the router does")

    print("moe. top-1 starves the router unless the auxiliary loss feeds it:")
    model = MoE("top1", dim=6, hidden=8, experts=4, seed=12)
    model.parameters["router.b"][...] = np.array([8.0, -8.0, -8.0, -8.0])
    out, info = model.forward(x)
    assert np.all(info["indices"] == 0)
    model.backward(x, info, np.ones_like(out))
    for expert in (1, 2, 3):
        assert not np.any(model.gradients[f"expert{expert}.W1"])
        assert not np.any(model.gradients[f"expert{expert}.b2"])
    assert not np.any(model.gradients["router.W"])
    assert not np.any(model.gradients["router.b"])
    switch = MoE("switch", dim=6, hidden=8, experts=4, seed=12)
    switch.parameters["router.b"][...] = np.array([8.0, -8.0, -8.0, -8.0])
    _, switch_info = switch.forward(x)
    boost = switch.aux_gradient(x, switch_info)
    assert np.any(boost["router.b"])
    print("  experts 1 to 3 are unselected and their gradients are zero; the")
    print("  top-1 gate is the constant 1, so the router's task gradient is zero")
    print("  too, and only the auxiliary loss reaches it -- that is the Switch fix")

    print("moe. finite differences agree with the analytic gradient:")
    small = rng.normal(size=(7, 4))
    target = rng.normal(size=(7, 4))
    for variant in ("topk", "soft_moe", "expert_choice"):
        model = MoE(variant, dim=4, hidden=5, experts=3, top_k=2,
                    capacity_factor=2.0, seed=10)
        out, info = model.forward(small)
        model.backward(small, info, target)
        analytic = {key: value.copy()
                    for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus, _ = model.forward(small)
                value[index] = old - 1e-6
                minus, _ = model.forward(small)
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        assert worst < 1e-6
        print(f"  {variant:<14s} {model.n_parameters:3d} parameters, worst"
              f" error {worst:.1e}")

    print("moe. the auxiliary loss has an exact gradient too:")
    for variant in ("switch", "load_balance_loss", "soft_moe"):
        model = MoE(variant, dim=4, hidden=5, experts=3, top_k=2, seed=11)
        _, info = model.forward(small)
        analytic = model.aux_gradient(small, info)
        worst = 0.0
        for key in ("router.W", "router.b"):
            value = model.parameters[key]
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small)[1]["aux"]
                value[index] = old - 1e-6
                minus = model.forward(small)[1]["aux"]
                value[index] = old
                worst = max(worst, abs((plus - minus) / 2e-6
                                       - analytic[key][index]))
        assert worst < 1e-8
        print(f"  {variant:<18s} worst router error {worst:.1e}")

    print("moe. malformed calls are refused:")
    guard = MoE("topk", dim=6, hidden=8, experts=4, seed=13)
    guard_out, guard_info = guard.forward(x)
    bad = [
        lambda: MoE("mixture"),
        lambda: MoE("topk", dim=0),
        lambda: MoE("topk", hidden=0),
        lambda: MoE("topk", experts=0),
        lambda: MoE("topk", top_k=0),
        lambda: MoE("topk", experts=2, top_k=3),
        lambda: MoE("capacity_factor", capacity_factor=0.0),
        lambda: MoE("capacity_factor", capacity_factor=np.nan),
        lambda: guard.forward(np.zeros((3, 5))),
        lambda: guard.forward(np.zeros(3)),
        lambda: guard.forward(np.full((3, 6), np.nan)),
        lambda: guard.backward(x, guard_info, np.zeros((3, 6))),
        lambda: guard.aux_gradient(x, guard_info),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, an empty width,")
    print("  depth or expert count, an out-of-range k, a zero or non-finite")
    print("  capacity, a wrong or non-finite input, a wrong gradient shape,")
    print("  and an auxiliary gradient from a variant without that loss")
    print("moe demo passed")


if __name__ == "__main__":
    demo()
