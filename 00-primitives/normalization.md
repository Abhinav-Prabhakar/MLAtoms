# Normalisation

`normalization.py` · Primitives · Tier 1 · 8 variants

## Intuition

Normalisation removes a degree of freedom you did not want: the scale and offset of
each activation. After it, every layer sees inputs with a known mean and variance,
so one learning rate works across layers and depth stops multiplying scale error.

The textbook reason ("internal covariate shift") is largely wrong — Santurkar et al.
showed that BN helps even when the input distribution is held fixed. The mechanism
that survives scrutiny is **conditioning**: normalisation makes the loss landscape
smoother (smaller Lipschitz constant of the gradients), which permits larger steps.
That is why the layer still helps when the statistics are already stable, and why
the *placement* (before or after the residual add) matters more than the variant.

## Variants

| Variant | Statistics over | Depends on batch? | Use when |
|:--|:--|:--|:--|
| `batch` | batch + spatial, per channel | yes | CNNs, large batches |
| `layer` | all features, per sample | no | transformers, RNNs, any batch size |
| `group` | channel groups + spatial, per sample | no | detection/segmentation at small batch |
| `instance` | spatial, per channel, per sample | no | style transfer, generative |
| `rms` | RMS over features, no centring | no | LLMs (cheaper than LayerNorm) |
| `weight_std` | weight rows | no | BN-free reparameterisation |
| `batch_renorm` | batch, clipped to running stats | yes, bounded | small/biased batches with BN |
| `power_norm` | l_p norm of the vector | no | very deep nets without BN |

## Derivation

**Batch norm.** For a feature `j` with batch mean `μ_j` and variance `σ_j²`:

$$
\hat x_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}},
\qquad y_j = \gamma_j \hat x_j + \beta_j
$$

`γ, β` are learned, so the layer can undo the normalisation if that is better — the
network chooses its own scale rather than inheriting it from the initialisation.
At inference the batch statistics are replaced by running averages; that switch is
the source of most BN bugs.

**Layer norm** applies the same formula per sample over all features. It commutes
with the batch, which is why it is the default in sequence models where batch
composition must not leak across examples.

**Group norm** reshapes `(N, C, H, W)` to `(N, G, C/G, H, W)` and normalises over
`(C/G, H, W)`. `G=1` is LayerNorm, `G=C` is InstanceNorm. It exists because BN's
batch statistics are unusable at batch size 2 (detection), and LayerNorm loses the
channel structure that convolution encodes.

**RMS norm** drops the mean:

$$
y = \frac{x}{\sqrt{\frac1d\sum_i x_i^2 + \epsilon}}\cdot\gamma
$$

Two fewer reductions than LayerNorm, no `β`, and empirically no loss in quality at
LLM scale. The reason it works: in a transformer, the mean of a residual stream is
carried by the residual path anyway, so re-centring each block is redundant.

**Batch renormalisation.** With `r = clip(σ_B/σ, 1/r_max, r_max)` and
`d = clip((μ_B − μ)/σ, −d_max, d_max)`:

$$
y = \frac{x - \mu_B}{\sigma_B}\,r + d
$$

The clip bounds how much a single odd batch can move the activations, which is what
lets BN survive small or non-i.i.d. batches.

## Complexity

All are `O(n·d)` — the same order as the layer they follow, with one extra pass over
the activations. `group` adds a reshape; `batch` adds a synchronisation point across
devices (the real cost at scale). Memory: BN stores `2·d` running statistics, LayerNorm
`2·d` parameters; neither stores activations beyond the forward pass.

## Failure modes

1. **Train/eval mismatch.** Using batch statistics at inference (or running
   statistics during training) changes the function being evaluated. The classic
   symptom: validation accuracy collapses the moment you stop training.
2. **Batch size 1.** BN with `n=1` normalises to zero — every activation becomes
   `β`. LayerNorm/GroupNorm or batch renorm are the fixes.
3. **Normalising before the residual add.** Pre-LN (`x + f(LN(x))`) trains stably
   at depth; post-LN (`LN(x + f(x))`) needs warmup and often diverges at 24+ layers.
   The variant is identical; the wiring is not.
4. **`γ` initialised to zero everywhere.** Some "zero-init" recipes kill the whole
   block at step 0. Zero is right only for the *last* layer of a residual branch.
5. **Normalising the wrong axis.** In `(N, C, H, W)` tensors, `axis=1` normalises
   channels; using the last axis normalises pixels and destroys the image. Shape
   bugs here are silent — they still train, just worse.

## Applications

- Transformers: `rms_norm` pre-norm, one per sub-layer.
- CNNs at batch ≥ 32: `batch_norm`; at batch < 8 (detection, video): `group_norm`
  with `G=32`.
- GANs and style transfer: `instance_norm` (it removes content statistics, which is
  exactly what you want when only style should vary).
- BN-free residual nets: `weight_std` on every conv plus `scaled_variance` init.

## References

- Ioffe & Szegedy, *Batch Normalization*, 2015. arXiv:1502.03167
- Ba, Kiros & Hinton, *Layer Normalization*, 2016. arXiv:1607.06450
- Wu & He, *Group Normalization*, 2018. arXiv:1803.08494
- Santurkar et al., *How Does Batch Normalization Help Optimization?*, 2018. arXiv:1805.11604
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019. arXiv:1910.07467
- Ioffe, *Batch Renormalization*, 2017. arXiv:1702.03275
