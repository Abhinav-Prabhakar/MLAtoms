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
