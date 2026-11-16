"""Schedules — 13 ways to spend a learning-rate budget.

Chapter:   01-optimization
Companion: schedules.md
Tier 1 · 13 variants: constant, step, multistep, exponential, cosine,
cosine_restarts, linear_warmup, one_cycle, cyclical, plateau, polynomial,
inv_sqrt, wsd

Everything except `plateau` is a pure function of the step count, so it can be
evaluated ahead of time and logged before the run. `plateau` is stateful by
definition: it needs the validation metric.
"""

from __future__ import annotations

import numpy as np

STATELESS = frozenset(
    ["constant", "step", "multistep", "exponential", "cosine", "cosine_restarts",
     "linear_warmup", "one_cycle", "cyclical", "polynomial", "inv_sqrt", "wsd"]
)


def lr_multiplier(
    step: int,
    variant: str = "cosine",
    *,
    total_steps: int = 1000,
    warmup_steps: int = 0,
    gamma: float = 0.1,
    step_size: int = 100,
    milestones: tuple[int, ...] = (),
    power: float = 2.0,
    min_ratio: float = 0.0,
    cycle_steps: int | None = None,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    pct_start: float = 0.3,
) -> float:
    """Learning-rate multiplier in [0, 1] at global `step` (0-indexed).

    The returned value multiplies a base LR; `one_cycle` and `cyclical` interpret
    the base as the peak, and `warmup_steps` composes with every schedule except
    `plateau`.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if variant not in STATELESS:
        raise ValueError(f"unknown or stateful variant {variant!r}")

    # Warmup is a prefix, not an alternative: ramp linearly to 1, then apply the
    # chosen shape to the remaining budget.
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    t = max(step - warmup_steps, 0)
    T = max(total_steps - warmup_steps, 1)

    if variant == "constant":
        return 1.0
    if variant == "step":
        return gamma ** (step // step_size)
    if variant == "multistep":
        return gamma ** sum(1 for m in milestones if step >= m)
    if variant == "exponential":
        return gamma ** (step / max(step_size, 1))
    if variant == "cosine":
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * min(t / T, 1.0)))
    if variant == "cosine_restarts":
        period = cycle_steps or T
        t_cur = t % period
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * t_cur / period))
    if variant == "linear_warmup":
        return 1.0
    if variant == "polynomial":
        return min_ratio + (1.0 - min_ratio) * max(1.0 - t / T, 0.0) ** power
    if variant == "inv_sqrt":
        return 1.0 / np.sqrt(max(t, 1))
    if variant == "wsd":
        decay_start = int(T * 0.8)
        if t < decay_start:
            return 1.0
        frac = (t - decay_start) / max(T - decay_start, 1)
        return max(1.0 - np.sqrt(frac), 0.0)
    if variant == "cyclical":
        s = cycle_steps or max(step_size, 1)
        phase = (t % (2 * s)) / s
        tri = 1.0 - abs(phase - 1.0)
        return min_ratio + (1.0 - min_ratio) * tri
    if variant == "one_cycle":
        # Smith's 1cycle: warm up to the peak over pct_start, cosine down to a
        # floor 1/final_div_factor of the peak.
        peak = 1.0
        start = peak / div_factor
        end = peak / final_div_factor
        up = int(T * pct_start)
        if t < up:
            return start + (peak - start) * (t / max(up, 1))
        frac = (t - up) / max(T - up, 1)
        return end + (peak - end) * 0.5 * (1.0 + np.cos(np.pi * frac))
    raise ValueError(variant)


class Plateau:
    """ReduceLROnPlateau: stateful, metric-driven, the one schedule that cannot
    be precomputed. Waits `patience` evaluations without improvement, then
    multiplies by `factor`, never going below `min_lr`."""

    def __init__(
        self, lr: float = 1.0, factor: float = 0.1, patience: int = 3,
        min_lr: float = 0.0, threshold: float = 1e-4,
    ) -> None:
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best = np.inf
        self.bad = 0
        self.reductions = 0

    def step(self, metric: float) -> float:
        if metric < self.best - self.threshold:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.bad = 0
                self.reductions += 1
        return self.lr


def main() -> None:
    T = 1000

    # 1. Constant, step and multistep: the flat baseline and the two discrete
    #    decays. Step drops every step_size regardless of the loss; multistep
    #    drops at chosen milestones.
    assert lr_multiplier(500, "constant") == 1.0
    assert abs(lr_multiplier(299, "step", step_size=300, gamma=0.1) - 1.0) < 1e-12
    assert abs(lr_multiplier(300, "step", step_size=300, gamma=0.1) - 0.1) < 1e-12
    ms = (300, 600)
    assert abs(lr_multiplier(650, "multistep", gamma=0.1, milestones=ms) - 0.01) < 1e-12
    print("constant/step/multistep ok")

    # 2. Exponential and cosine: the two smooth defaults. Cosine hits exactly 1
    #    at the start and 0 at the end, and is 0.5 at the midpoint.
    assert abs(lr_multiplier(0, "exponential", gamma=0.9, step_size=10) - 1.0) < 1e-12
    assert abs(lr_multiplier(10, "exponential", gamma=0.9, step_size=10) - 0.9) < 1e-12
    assert abs(lr_multiplier(0, "cosine", total_steps=T) - 1.0) < 1e-12
    assert abs(lr_multiplier(T, "cosine", total_steps=T)) < 1e-12
    assert abs(lr_multiplier(T // 2, "cosine", total_steps=T) - 0.5) < 1e-9
    curve = [lr_multiplier(s, "cosine", total_steps=T) for s in range(T + 1)]
    assert all(curve[i] >= curve[i + 1] - 1e-12 for i in range(T))
    print(f"cosine: 1.0 -> {lr_multiplier(T, 'cosine', total_steps=T):.1e}, monotone")

    # 3. Cosine with restarts returns to full LR at the start of every cycle.
    for k in range(4):
        assert abs(lr_multiplier(k * 250, "cosine_restarts", cycle_steps=250) - 1.0) < 1e-12
    assert abs(lr_multiplier(125, "cosine_restarts", cycle_steps=250) - 0.5) < 1e-9
    print("cosine_restarts: back to 1.0 every cycle")

    # 4. Warmup composes with a shape: 1/warmup at step 0, full LR at warmup-1,
    #    then the cosine takes over on the remaining steps.
    w = 100
    assert abs(lr_multiplier(0, "cosine", total_steps=T, warmup_steps=w) - 0.01) < 1e-12
    assert abs(lr_multiplier(w - 1, "cosine", total_steps=T, warmup_steps=w) - 1.0) < 1e-12
    assert abs(lr_multiplier(T, "cosine", total_steps=T, warmup_steps=w)) < 1e-12
    print("linear warmup + cosine ok")

    # 5. 1cycle: rises to a peak then anneals to a floor 1/final_div_factor of it.
    oc = [lr_multiplier(s, "one_cycle", total_steps=T, div_factor=25, final_div_factor=1e4)
          for s in range(T + 1)]
    peak = int(np.argmax(oc))
    assert abs(oc[peak] - 1.0) < 1e-9 and 0.2 * T < peak < 0.4 * T
    assert oc[0] == 1 / 25 and abs(oc[-1] - 1e-4) < 1e-9
    print(f"one_cycle: peak at step {peak}, ends at {oc[-1]:.1e}")

    # 6. Cyclical (triangular): periodic between min_ratio and 1.
    cyc = [lr_multiplier(s, "cyclical", cycle_steps=100, min_ratio=0.1) for s in range(401)]
    assert abs(cyc[0] - 0.1) < 1e-9 and abs(cyc[100] - 1.0) < 1e-9
    assert abs(cyc[200] - 0.1) < 1e-9 and abs(cyc[300] - 1.0) < 1e-9
    print("cyclical: triangular, period 2*cycle_steps")

    # 7. Plateau: patience counts, then a single multiplicative drop.
    p = Plateau(lr=1.0, factor=0.1, patience=2)
    metrics = [10, 9, 9, 9, 8]     # improve, stall, stall (drop), improve
    lrs = [p.step(m) for m in metrics]
    assert lrs == [1.0, 1.0, 1.0, 0.1, 0.1], lrs
    assert p.reductions == 1
    print(f"plateau: lrs {lrs}")

    # 8. Polynomial and inverse-sqrt: the two shapes that decay without a hard end.
    poly = [lr_multiplier(s, "polynomial", total_steps=T, power=2.0) for s in range(T + 1)]
    assert poly[0] == 1.0 and poly[T] == 0.0 and abs(poly[T // 2] - 0.25) < 1e-9
    inv = [lr_multiplier(s, "inv_sqrt") for s in range(1, 101)]
    assert abs(inv[0] - 1.0) < 1e-12 and abs(inv[99] - 0.1) < 1e-12
    print("polynomial (1-t/T)^2 and 1/sqrt(t) ok")

    # 9. WSD (warmup-stable-decay): flat for 80% of the budget, then a 1-sqrt
    #    decay to zero. The shape used for most modern LLM pretraining runs.
    wsd = [lr_multiplier(s, "wsd", total_steps=T) for s in range(T + 1)]
    assert abs(wsd[100] - 1.0) < 1e-12 and abs(wsd[int(0.8 * T)] - 1.0) < 1e-12
    assert wsd[-1] == 0.0
    decreases = sum(1 for i in range(T) if wsd[i] > wsd[i + 1] + 1e-12)
    assert 0.15 * T < decreases < 0.25 * T
    print(f"wsd: flat to {int(0.8 * T)}, then decay over {decreases} steps")

    # 10. Every schedule is a valid multiplier on [0, T] and never goes negative.
    for v in sorted(STATELESS):
        vals = [lr_multiplier(s, v, total_steps=T, step_size=100, gamma=0.9,
                               milestones=(300, 600), cycle_steps=250) for s in range(T + 1)]
        assert min(vals) >= 0.0 and max(vals) <= 1.0 + 1e-12, (v, min(vals), max(vals))
    print("all 12 stateless schedules stay in [0, 1]")
    print("schedules demo passed")


if __name__ == "__main__":
    main()
