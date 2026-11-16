# Schedules

`schedules.py` · Optimization · Tier 1 · 13 variants

## Intuition

A schedule is a statement about how much you trust the current gradient. Early in
training the gradient is informative but the model is far from anything good, so
you take large steps (after a brief warmup). Late in training the gradient is
dominated by noise, and a large step means you never settle.

So a schedule has two jobs: **get through the transient safely** (warmup) and
**shrink the noise floor** (decay). Everything else — restarts, cycles, plateau —
is a bet about where the transient ends.

## Variants

| Variant | Shape | Signal it uses | Typical use |
|:--|:--|:--|:--|
| `constant` | flat | nothing | debugging, RL, very short runs |
| `step` | `γ^(t//k)` | step count | classic CNN training (30/60/90) |
| `multistep` | drops at milestones | step count | hand-tuned vision recipes |
| `exponential` | `γ^t` | step count | smooth, rarely optimal |
| `cosine` | `½(1+cos πt/T)` | total budget | transformer default |
| `cosine_restarts` | periodic cosine | cycle length | SGDR, ensembling snapshots |
| `linear_warmup` | ramp to 1, then flat | warmup length | prefix for any other shape |
| `one_cycle` | up then cosine down to a floor | total budget | fast single-run training |
| `cyclical` | triangle | cycle length | exploring several minima |
| `plateau` | `×γ` when the metric stalls | validation metric | classical, small-data |
| `polynomial` | `(1−t/T)^p` | total budget | BERT-style, `p=1` |
| `inv_sqrt` | `1/√t` | step count | the SGD theory default |
| `wsd` | flat then `1−√` decay | total budget | LLM pretraining |

## Derivation

**Why `1/√t` is the theoretical default.** For stochastic convex optimisation with
`O(1/√T)` convergence, the optimal *averaged* iterate uses `η_t ∝ 1/√t`. The
regret bound of online gradient descent is `O(GD√T)` with `η = D/(G√T)`, and the
anytime version is `η_t = η₀/√t`. Cosine and step decays are heuristics that beat
this in practice because deep learning is not convex; `1/√t` remains the shape to
compare against.

**Why warmup.** Two separate reasons, and conflating them causes bugs:

1. *Adam's variance estimate is unreliable for the first few steps.* `v̂_t` is
   computed from `t` samples; its reciprocal is biased upward, so the first
   adaptive steps are too large. Bias correction helps but cannot fix the variance.
2. *At initialisation the loss surface is often sharp.* With random weights, the
   largest curvature can be orders of magnitude above the bulk, so the stability
   threshold `2/L` is tiny. Warmup keeps `η` below it until the iterate moves into
   a flatter region.

`linear_warmup` here is a prefix: it ramps to 1 and then applies whichever shape
you asked for to the remaining budget. That composition is the one thing every
implementation must get right — the cosine should be computed over
`total − warmup`, not over `total`.

**Cosine.** `η(t) = η_min + (η_max − η_min)·½(1 + cos(πt/T))`. Its appeal is that
it spends most of the budget near the top (the derivative is flat at `t=0`), then
decelerates smoothly to the floor. It needs `T` up front, which is why
`t_total`-dependent schedules and checkpoint resumption interact badly: resuming
with a different `T` silently changes the whole curve.

**Restarts (SGDR).** Run cosine on a cycle of length `T_i`, then reset to
`η_max`. Each restart kicks the iterate out of a narrow minimum; the snapshots at
cycle ends are diverse and ensemble well. `T_i` is usually multiplied by 2 each
cycle.

**1cycle.** Warm up from `η_max/25` to `η_max` over the first 30% of the budget,
then cosine-anneal to `η_max/10⁴`. The large initial LR acts as regularisation
(Smith's "super-convergence"): the model spends the high-LR phase exploring and
the long anneal phase settling. It is the fastest known recipe for a *fixed*
compute budget and a poor choice for anything you might extend.

**WSD (warmup–stable–decay).** Keep the LR flat for 80% of the run, then decay
with `1−√((t−t_s)/(T−t_s))`. The flat phase means any checkpoint in it is a valid
model, so you can decide to stop late; the short decay recovers the quality that a
cosine schedule would have reached. This is why most LLM pretraining now uses it.

**Plateau.** The only non-anticipative schedule: it observes the validation metric
and multiplies by `γ` after `patience` evaluations without improvement. It cannot
be precomputed, it depends on the metric's noise level, and it is the right choice
when you do not know `T` — which is the normal case for small-data problems.

## Code walkthrough

`lr_multiplier` is a pure function of the step; `Plateau` is a small state machine.
The pure part begins with the composition rule that every implementation has to get
right:

```python
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    t = max(step - warmup_steps, 0)
    T = max(total_steps - warmup_steps, 1)
```

Warmup is a **prefix**, not an alternative shape: the selected schedule is evaluated
on the remaining `T = total − warmup` steps via the shifted `t`. Two details carry
the meaning. The `(step + 1)/warmup_steps` makes step 0 return `1/warmup_steps`
instead of exactly zero — a zero first LR wastes a step and breaks optimisers that
divide by it. And `T` is recomputed after the shift; a cosine over `total_steps`
instead of `total_steps − warmup_steps` would finish its half-cosine early and then
sit at the floor for the warmup duration.

The discrete schedules index the **global** step, not the shifted `t`:

```python
    if variant == "step":
        return gamma ** (step // step_size)
    if variant == "multistep":
        return gamma ** sum(1 for m in milestones if step >= m)
```

That is deliberate: milestones are absolute step numbers in every published recipe
("drop at 30/60/90 epochs"), so shifting them by the warmup would silently move the
drops. The continuous, budget-based schedules below use `t/T`, because their shape
depends on how much budget remains, not on an absolute calendar.

Cosine is written with a clamp on the ratio:

```python
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * min(t / T, 1.0)))
```

Without `min(t/T, 1.0)`, evaluating past the budget lets `cos` turn around and the
LR starts *increasing* again — which happens whenever a run is extended past its
planned `total_steps`. The clamp makes "past the end" mean "stay at the floor".
`min_ratio` is what keeps the final LR non-zero; the demo checks the endpoints at
`1.0` and `0.0` for the default `min_ratio=0`.

`cosine_restarts` is the same formula on `t mod period`:

```python
        period = cycle_steps or T
        t_cur = t % period
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * t_cur / period))
```

Every cycle has the same length here; SGDR's doubling (`T_i = T_0·2^i`) is a
one-line change to `period`, and the demo asserts the simple version returns to
exactly 1.0 at each multiple of `cycle_steps`.

`wsd` is the LLM schedule and the shape is entirely in the branch point:

```python
        decay_start = int(T * 0.8)
        if t < decay_start:
            return 1.0
        frac = (t - decay_start) / max(T - decay_start, 1)
        return max(1.0 - np.sqrt(frac), 0.0)
```

Flat for 80%, then `1 − √frac`. The `max(·, 0)` is needed because `frac` can exceed 1
by one step when `T` is not divisible; without it the multiplier goes negative and
the optimiser climbs. The demo counts strictly decreasing steps and expects roughly
20% of `T` — that is the invariant the schedule is sold on.

`cyclical` is a triangle, and the period is `2·cycle_steps`, not `cycle_steps`:

```python
        s = cycle_steps or max(step_size, 1)
        phase = (t % (2 * s)) / s
        tri = 1.0 - abs(phase - 1.0)
```

At `t=0`, `phase=0`, `tri=0` → `min_ratio`; at `t=s`, `phase=1`, `tri=1` → peak; at
`t=2s` it wraps. Using `t % s` instead gives a sawtooth that jumps from the peak
straight back to the floor, which is a different (and much worse) schedule.

`one_cycle` is two pieces joined at `pct_start`:

```python
        up = int(T * pct_start)
        if t < up:
            return start + (peak - start) * (t / max(up, 1))
        frac = (t - up) / max(T - up, 1)
        return end + (peak - end) * 0.5 * (1.0 + np.cos(np.pi * frac))
```

`start = 1/div_factor`, `end = 1/final_div_factor`. The first piece is linear from
`1/25` to `1`; the second is cosine from `1` to `1e-4`. The `max(up, 1)` and
`max(T − up, 1)` guard degenerate budgets (`pct_start=0`, or `T=1`), which is the
kind of input a unit test uses and a real run never does.

`Plateau` is the only stateful schedule:

```python
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
```

The strict `< best − threshold` means an improvement smaller than `threshold` counts
as a stall — that is the noise filter, and setting `threshold=0` makes it fire on
every wobble. `bad` resets to zero *after* a reduction, so the schedule needs another
full `patience` window before dropping again; without that reset it would collapse to
`min_lr` in `patience` consecutive bad evaluations. `min_lr` is a floor on the
multiplied value, not on the factor, so `factor=0.1` can never take it below it.

## Complexity

`O(1)` time and memory per step: every schedule is a closed-form function of the
step count (`plateau` is a few scalars of state). The real cost is that schedules
constrain batching: to hit a target `T` you must fix the number of steps, hence the
batch size and the dataset passes.

## Failure modes

1. **Warmup shorter than the sharp phase.** Loss spikes at step ~200 and never
   recovers. Symptom: grad norm spikes first, then the loss. Fix: longer warmup,
   or gradient clipping while you diagnose.
2. **Cosine to exactly zero.** The final steps have `η ≈ 0`, which is fine for a
   loss but not for BatchNorm running statistics or for an EMA that needs to track
   a moving model. Keep a floor of 1–5% of peak.
3. **Step decay with the wrong `total_steps`.** The drops land at the wrong place;
   accuracy is 1–2 points below the tuned recipe. If you change the budget,
   re-tune the milestones or switch to cosine.
4. **Restarts without resetting momentum.** The optimiser's velocity still points
   into the old minimum; the first steps of the new cycle are wasted. Reset the
   momentum buffer at a restart.
5. **Plateau on a noisy metric.** Validation loss jitters by more than `threshold`,
   so the schedule fires constantly and the LR collapses to `min_lr` in the first
   third of training. Smooth the metric or raise `patience`.
6. **Schedule/optimiser mismatch.** `one_cycle` assumes SGD-like dynamics; with
   Adam, the peak LR is effectively multiplied again by the adaptive denominator
   and the run diverges. Scale the peak down by ~10× for Adam.

## Applications

- **LLM pretraining:** linear warmup (0.1–1% of steps) + cosine or WSD; `η_peak`
  from a small sweep, often `3e-4` for ~1B parameters.
- **Vision CNNs:** step decay at 50/75% of the budget, or cosine; SGD + momentum.
- **Small tabular data:** `plateau` with `patience=5–10`, `factor=0.3`, because
  the number of useful epochs is unknown.
- **Fine-tuning:** short warmup (~5% of steps) then linear or cosine decay to a
  small floor; never restart from a large LR.
- **Snapshot ensembles:** cosine restarts, keep the checkpoint at each cycle end.

## References

- Smith, *Cyclical Learning Rates for Training Neural Networks*, 2017. arXiv:1506.01186
- Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*, 2018. arXiv:1708.07120
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2017. arXiv:1608.03983
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017. arXiv:1706.02677
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models* (WSD), 2024. arXiv:2404.06395
- Robbins & Monro, *A Stochastic Approximation Method*, 1951 (the `1/√t` root).
