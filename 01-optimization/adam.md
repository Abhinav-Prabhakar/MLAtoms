# Adam

`adam.py` · Optimization · Tier 1 · 8 variants

## Intuition

Adam gives every coordinate its own learning rate, derived from the history of
that coordinate's gradient:

$$
\theta_i \leftarrow \theta_i - \eta\,\frac{\hat m_i}{\sqrt{\hat v_i} + \epsilon}
$$

`m` is a smoothed gradient (direction), `v` is a smoothed squared gradient
(scale). Dividing by `√v` makes the step roughly scale-free: a coordinate whose
gradients are consistently 100× larger gets a 100× smaller step. In the first
steps, where `v` is tiny, this behaves like sign descent — `|Δθ| ≈ η` per
coordinate — which is why Adam is so forgiving about the initial LR and so bad at
actually converging to high precision.

## Variants

| Variant | Change from Adam | When it wins |
|:--|:--|:--|
| `adam` | — | default for transformers, sparse gradients, fast prototyping |
| `adamw` | weight decay applied to `θ`, not `g` | any regularised transformer; the modern default |
| `amsgrad` | denominator uses `max(v_t)` over time | non-stationary objectives; fixes Adam's non-convergence counterexample |
| `nadam` | Nesterov-style momentum on `m` | slight speed-up on vision tasks |
| `adamax` | `v` replaced by running `max |g|` | `∞`-norm variant, robust to rare huge gradients |
| `radam` | variance rectification, no update while `v` is unreliable | early training, warmup-free recipes |
| `adan` | second statistic is `(g_t − g_{t−1})`, Nesterov update | faster convergence at low LR; used in some diffusion code |
| `eps_inside_sqrt` | `√(v̂ + ε)` instead of `√v̂ + ε` | fp16/BF16, and any case where `v` underflows |

## Derivation

**Why `m/√v`.** With independent coordinates, the natural gradient is
`H⁻¹g`; the diagonal of the Hessian is approximately the second moment of the
gradient, `E[g_i²]`. So `m_i/√v_i` is diagonal natural gradient descent with a
smoothed estimate of the metric. That is the honest justification; "it normalises
the gradients" is the intuition.

**Bias correction.** With `m_0 = v_0 = 0`:

$$
m_t = (1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}g_k
\;\Longrightarrow\;
\mathbb{E}[m_t] = (1-\beta_1^t)\,\mathbb{E}[g]
$$

so dividing by `1−β₁ᵗ` (and `1−β₂ᵗ`) makes the moments unbiased. Without it the
first step is smaller by `(1−β₁)/√(1−β₂)`: the demo measures 0.316η instead of
η at `β = (0.9, 0.999)`. That factor is also why "Adam needs warmup" is only half
true — the bias correction already handles the start; warmup handles the large
early *variance*.

**AMSGrad.** Reddi et al. showed a convex problem where Adam's effective learning
rate `η/√v̂_t` increases, which breaks convergence. Replacing `v̂_t` with
`v̂_max,t = max_{s≤t} v̂_s` makes the rate non-increasing and restores the
`O(1/√T)` bound.

**AdamW.** For SGD, `g + λθ` and the decoupled `θ ← θ(1−ηλ)` coincide. For Adam
they do not, because the decay is divided by `√v`:

$$
\text{coupled: } \Delta\theta = \eta\,\frac{\hat m + \lambda\theta}{\sqrt{\hat v}+\epsilon},
\qquad
\text{decoupled: } \Delta\theta = \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \eta\lambda\theta
$$

Coordinates with small `v` get their weight decay amplified enormously by the
coupled form — the reason L2-Adam under-regularises some layers and blows up
others. AdamW's fix is one line and is now the default in every transformer
training code.

**RAdam.** Early on, `v̂` is estimated from very few samples, so its reciprocal is
biased and unstable. RAdam computes `ρ_t = ρ_∞ − 2tβ₂ᵗ/(1−β₂ᵗ)` with
`ρ_∞ = 2/(1−β₂) − 1`, the effective number of samples behind `v̂`, and:

- if `ρ_t ≤ 4` (usually the first ~5 steps): take a plain momentum step, no
  adaptive scaling;
- otherwise: scale the adaptive step by
  `r_t = √(((ρ_t−4)(ρ_t−2)ρ_∞)/((ρ_∞−4)(ρ_∞−2)ρ_t))`, which → 1 as `t → ∞`.

So RAdam *is* Adam plus a smooth, automatic warmup — no LR schedule needed for
stability.

**Adan.** The second moment of the gradient measures magnitude but not
*agreement*. Adan instead smooths the difference `d_t = g_t − g_{t−1}`:

$$
m_t = (1-\beta_1)m_{t-1} + \beta_1 g_t,\quad
v_t = (1-\beta_2)v_{t-1} + \beta_2 d_t,\quad
n_t = (1-\beta_3)n_{t-1} + \beta_3 d_t^2
$$

$$
\theta \leftarrow \theta - \eta\,\frac{\hat m + \beta_2\hat v}{\sqrt{\hat n}+\epsilon}
$$

This is a Nesterov-style acceleration in which the denominator shrinks exactly when
the gradient stops changing — i.e. when the iterate is near a stationary point with
consistent curvature. The flip side, visible in the demo, is that the stationary
error scales with `η`: `n` vanishes like `d²` while `m` vanishes like `d`, so the
step does not go to zero. Adan wants a decaying LR at the end.

## Complexity

Time `O(P)` per step. Memory `3P` for Adam (`θ, m, v`), `4P` for AMSGrad
(`+v_max`), `5P` for Adan (`m, v, n, g_prev`). That ratio is why Adan and AMSGrad
are rarely used at LLM scale even when they converge faster per step. All variants
vectorise to a handful of fused elementwise kernels; on a GPU the memory traffic,
not the arithmetic, is the cost.

## Failure modes

1. **Non-convergence at the end.** Adam's step does not vanish as `g → 0` (the
   `ε` floor and the EMA keep `m/√v` at `O(1)`). Use a decaying LR, or switch to
   SGD+momentum for the last 10% of training.
2. **Coupled weight decay.** `adam(weight_decay=0.1)` is not AdamW and the effect
   depends on the gradient scale of each layer. Use `adamw` and tune `λ` directly.
3. **`ε` in the wrong place.** `√v + ε` vs `√(v + ε)`: in fp16, `v` can underflow
   to 0 and the first form produces a division by `ε = 1e-8`, i.e. a gigantic step.
   Post-LN transformers trained in fp16 crashed on exactly this; BF16 plus
   `eps_inside_sqrt` is the robust combination.
4. **`β₂` too high for small batches.** `v` averages over `1/(1−β₂) = 1000` steps;
   with 32-sample batches that is 32k examples, which is fine, but with highly
   non-stationary data it lags. `β₂ = 0.95–0.98` is common in RL for this reason.
5. **Bias correction with a resumed checkpoint.** `t` must be restored too;
   restarting at `t=0` reintroduces the transient (usually harmless, occasionally
   a visible loss spike).

## Applications

- **Transformers:** AdamW, `β=(0.9, 0.95)`, `ε=1e-8`, decoupled `λ=0.1`, warmup
  + cosine decay. This is the canonical LLM recipe.
- **Diffusion models:** Adam or Adan with `β₂=0.999` and a long warmup; the
  gradient noise is large and the second moment is genuinely informative.
- **Sparse features (recommenders, NLP with rare tokens):** Adam-family updates
  are per-coordinate, so a feature seen once still gets a sensible step — the
  main reason it beat SGD there.
- **Fine-tuning with tiny data:** AdamW + low LR is more robust than SGD, because
  the per-coordinate scaling is effectively a per-layer LR.

## References

- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015. arXiv:1412.6980
- Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond* (AMSGrad), 2018. arXiv:1904.09237
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2019. arXiv:1711.05101
- Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond* (RAdam), 2019. arXiv:1908.03265
- Dozat, *Incorporating Nesterov Momentum into Adam* (Nadam), ICLR workshop 2016.
- Xie et al., *Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models*, 2022. arXiv:2208.06677
- Kunstner, Hennig & Balles, *Limitations of the empirical Fisher approximation*, 2019 (why `√v` is not the Hessian).
