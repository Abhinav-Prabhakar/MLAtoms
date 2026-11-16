# Losses

`losses.py` · Primitives · Tier 1 · 20 variants

## Intuition

A loss is not a score — it is the *estimator*. Minimising squared error gives the
conditional mean, absolute error gives the median, pinball loss gives any quantile
you ask for, cross-entropy gives the conditional distribution. Changing the loss
changes what you are predicting, even when the model is identical.

Two practical axes decide the choice: **what the errors cost** (symmetric vs
asymmetric, bounded vs unbounded) and **what the labels look like** (balanced,
imbalanced, soft, pairwise, sets).

## Variants

| Variant | Input | Estimator / behaviour | Use when |
|:--|:--|:--|:--|
| `mse` | preds | conditional mean | Gaussian noise, no outliers |
| `mae` | preds | conditional median | outliers, robust regression |
| `huber` | preds | mean near 0, median far | outliers but smooth gradients wanted |
| `log_cosh` | preds | like Huber, C² everywhere | second-order methods |
| `quantile` | preds | conditional q-quantile | intervals, inventory, SLA targets |
| `bce` | logits | Bernoulli MLE | binary classification |
| `cross_entropy` | logits | categorical MLE | multi-class |
| `focal` | logits | down-weights easy examples | extreme imbalance (>1:100) |
| `dice` | probs | overlap, scale-free | segmentation with tiny objects |
| `tversky` | probs | precision/recall trade-off | when FN and FP differ in cost |
| `hinge` | scores | max-margin | SVM, ranking thresholds |
| `squared_hinge` | scores | smooth margin | least-squares SVM |
| `triplet` | embeddings | relative distance | retrieval, face verification |
| `contrastive` | distances | pair similarity | siamese networks |
| `infonce` | embeddings | mutual information lower bound | CLIP-style pretraining |
| `kl` | distributions | forward divergence | distillation, VAEs |
| `js` | distributions | symmetric, bounded | GAN objectives, drift tests |
| `wasserstein` | samples | earth-mover distance | distribution matching |
| `cosine_embedding` | embeddings | angular margin | metric learning |
| `label_smoothing` | logits | CE with soft targets | large models, calibration |

## Derivation

**Cross-entropy is maximum likelihood.** For one-hot `y` and logits `z`:

$$
-\log p_y = -z_y + \log\sum_k e^{z_k}
\qquad\Longrightarrow\qquad
\frac{\partial}{\partial z_k} = p_k - y_k
$$

The `logsumexp` form is why the implementation never exponentiates the logits
directly: subtract the row max first, then everything stays in `[0, log K]`.

**Focal loss.** With `p_t` the probability of the true class,

$$
L = -(1-p_t)^\gamma \log p_t
$$

and since `dp_t/dz = (2y-1)\,p_t(1-p_t)`,

$$
\frac{\partial L}{\partial z} = (2y-1)\left[\gamma\,p_t(1-p_t)^\gamma \log p_t - (1-p_t)^{\gamma+1}\right]
$$

At `γ=0` this collapses to `p−y`, which is the check that the derivation is right.

**Dice and Tversky.** With `I = Σ p_i y_i`, `P = Σ p_i`, `T = Σ y_i`:

$$
\text{Dice} = \frac{2I+\epsilon}{P+T+\epsilon},\qquad
\text{Tversky} = \frac{I+\epsilon}{I + \alpha(P-I) + \beta(T-I) + \epsilon}
$$

`α` penalises false positives, `β` false negatives. `α=β=0.5` is Dice; `α=0.3,
β=0.7` is the standard "recall matters more" setting for lesion segmentation.
Both are *not* averaged per-pixel — the gradient is the shape of the whole mask,
which is why `p.size` must not appear in the gradient.

**InfoNCE.** With `a_i, b_i` L2-normalised and `s_ij = a_i·b_j / τ`:

$$
L = -\tfrac12\left[\frac1n\sum_i \log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} + \frac1n\sum_i \log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\right]
$$

The gradient is one matrix product per direction:

$$
\frac{\partial L}{\partial a} = \frac{1}{2\tau}\left(G_1 + G_2^\top\right)b,
\qquad
\frac{\partial L}{\partial b} = \frac{1}{2\tau}\left(G_1^\top + G_2\right)a,
\qquad
G_k = \frac{\text{softmax}(S_k) - I}{n}
$$

then back through the normalisation with `(I − aaᵀ)/‖z‖`. The `1/2` and the
`/τ` are where hand-written implementations usually go wrong; the demo
gradient-checks both.

## Complexity

All are `O(n)` elementwise except InfoNCE, which builds an `n×n` similarity
matrix: `O(n²d)` time, `O(n²)` memory — the reason large-batch contrastive
training uses memory queues or sharded negatives.

## Failure modes

1. **Loss/activation mismatch.** `cross_entropy` on `softmax` output double-applies
   the softmax and silently trains a worse model; `bce` on probabilities instead of
   logits underflows to `log(0)`. This file takes logits on purpose.
2. **Dice on empty masks.** When `y` is all zeros, Dice is `1 - ε/(P+ε) ≈ 1` for
   every prediction, so the gradient vanishes exactly where the model is wrong
   on background. Use per-image Dice plus CE, or Tversky with `β > α`.
3. **Focal over-correction.** At `γ=2` on *balanced* data, focal loss underfits the
   hard-but-learnable tail. It is an imbalance tool, not a general upgrade.
4. **`mae` at zero.** The subgradient `sign(0)` is arbitrary; training can
   oscillate around the median. Smooth it (`huber` with small `δ`) or use a
   proximal step.
5. **KL asymmetry.** `KL(p‖q)` punishes `q` for missing mass where `p` has it, but
   tolerates `q` putting mass where `p` does not — that is why forward KL gives
   *mode-covering* fits and reverse KL gives *mode-seeking* ones.

## Applications

- Regression with asymmetric cost (delivery time, inventory, latency): `quantile`
  at the relevant `q`, trained once per quantile or with a multi-head output.
- Segmentation with 1:1000 class ratios: `dice + 0.5·cross_entropy`, or `tversky`
  with `β=0.7`.
- Retrieval and embedding models: `triplet` with hard-negative mining; `infonce`
  when the batch itself supplies the negatives.
- Distillation: `kl(student_logits / T ‖ teacher_logits / T) · T²` — the `T²`
  factor restores the gradient scale that temperature flattening removes.

## References

- Lin et al., *Focal Loss for Dense Object Detection*, 2017. arXiv:1708.02002
- Salehi, Erdogmus & Gholipour, *Tversky Loss Function for Image Segmentation*, 2017. arXiv:1706.05721
- Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive Coding*, 2018. arXiv:1807.03748
- Müller, Kornblith & Hinton, *When Does Label Smoothing Help?*, 2019. arXiv:1906.02629
