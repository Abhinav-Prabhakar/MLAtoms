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

## Code walkthrough

Every function returns `(value, grad)`: `value` is a scalar **mean** over the batch,
`grad` has the shape of the prediction. That single convention is why every
gradient in the file ends in `/ y.size`, `/ n` or `/ r.size` — the mean's `1/n` is
propagated by hand rather than by a wrapper, because three functions (triplet,
contrastive, infonce) return a tuple of gradients and cannot share a wrapper.

The four logit-based losses never see probabilities. Binary cross-entropy is
written as `log(1+e^z) − yz` instead of `−y log σ(z)`:

```python
def bce(y: np.ndarray, logits: np.ndarray) -> tuple[float, np.ndarray]:
    """Binary cross-entropy from logits: log(1+e^z) - y z, stable."""
    loss = np.logaddexp(0.0, logits) - y * logits
    p = 0.5 * (1.0 + np.tanh(0.5 * logits))
    return float(np.mean(loss)), (p - y) / y.size
```

`σ(z)` underflows to exactly `0` for `z < −745`, and `log(0)` is `−inf`; the loss is
then `inf` and the gradient `nan`, even though the true loss at `z = −800, y = 0` is
`0`. `logaddexp` keeps it at `8e-348`. The gradient is `σ(z) − y` *without* the
`σ'(z)` factor — that cancellation is the entire reason to fuse the sigmoid into the
loss.

`cross_entropy` uses the same cancellation for `K` classes, and avoids one-hot:

```python
    logp = z - lse
    n = z.shape[0]
    loss = -(logp[np.arange(n), labels]).mean()
    p = np.exp(logp)
    p[np.arange(n), labels] -= 1.0
    return float(loss), p / n
```

`p` starts as the softmax and then `−1` is subtracted *only at the true class*, so the
returned gradient is `softmax(z) − onehot(y)` in place. Building the one-hot and
subtracting costs an `n×K` temporary for no reason; on a 50k-vocab LM head that
temporary is the whole activation budget.

`focal` is the loss where a plausible-looking implementation is subtly wrong:

```python
    log_pt = y * -np.logaddexp(0.0, -logits) + (1.0 - y) * -np.logaddexp(0.0, logits)
    pt = np.exp(log_pt)
    loss = -((1.0 - pt) ** gamma) * log_pt
    sign = 2.0 * y - 1.0  # d(pt)/d(logit) has this factor on top of pt(1-pt)
    grad = sign * (gamma * pt * (1.0 - pt) ** gamma * log_pt - (1.0 - pt) ** (gamma + 1.0))
```

`pt` is the probability of the *true* class, so for `y=0` it is `σ(−z)` and
`dpt/dz = −pt(1−pt)`. The `(2y−1)` factor carries that sign flip; omit it and the
loss is still correct, the gradient check on `y=1` still passes, and only the
negative class trains backwards. `log_pt` is computed from logits in both directions
rather than as `log(pt)`, for the same underflow reason as `bce`.

Dice and Tversky are whole-mask functions, so their gradients look nothing like the
elementwise ones:

```python
    a = float((p * y).sum())
    b = float(p.sum() + y.sum())
    loss = 1.0 - (2.0 * a + eps) / (b + eps)
    grad = -(2.0 * y * (b + eps) - (2.0 * a + eps)) / (b + eps) ** 2
```

Quotient rule: `d/dp_i [1 − (2a+ε)/(b+ε)] = −[2y_i(b+ε) − (2a+ε)]/(b+ε)²`. Note the
`y_i` and the absence of any `1/p.size`: this gradient is the derivative of a
*ratio of sums*, so every pixel feels the whole mask. Dividing by `p.size` — the
reflex for a "mean loss" — scales the gradient by `1/(HW)` and quietly turns Dice
into a very slow loss. The demo checks this against finite differences on a 2×3 map,
which is small enough that the error is obvious if the normalisation is wrong.

`triplet` is the cleanest place to see why the three gradients are not independent:

```python
    ga = 2.0 * (neg - pos) * active[:, None] / n
    gp = -2.0 * (a - pos) * active[:, None] / n
    gn = 2.0 * (a - neg) * active[:, None] / n
```

With `m = ‖a−p‖² − ‖a−n‖² + margin`, `∂m/∂a = 2(a−p) − 2(a−n) = 2(n−p)`: the anchor
gradient is the *difference* of the two other directions. Writing `2(a−p)` there is
the standard bug — it is the gradient of a loss that only pulls positives together.

`infonce` factorises the symmetric loss into one helper applied twice:

```python
    def rows(logit):
        z = logit - logit.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -np.mean(np.diag(logp))
        return loss, (np.exp(logp) - eye) / n

    l1, g1 = rows(logits)
    l2, g2 = rows(logits.T)
    da = 0.5 * (g1 @ b + g2.T @ b) / tau
```

`rows` is ordinary cross-entropy with labels `0..n−1`, so its gradient is
`softmax(S) − I`, already divided by `n`. The second call transposes the similarity
matrix, so its gradient arrives transposed — hence `g2.T @ b`. And because
`logits = a bᵀ/τ`, each `@ b` carries the `/τ`. The `0.5` averages the two
directions. The final step is the chain rule through `a = z/‖z‖`:

```python
    gz1 = (da - a * (a * da).sum(axis=1, keepdims=True)) / na
```

`(I − aaᵀ)/‖z‖` applied to `da`, expanded. Skipping the projection term gives a
gradient that is correct on the sphere and wrong off it — the loss still falls,
because the radial component is not the one that matters, but the encoder drifts
in norm.

The divergences have one trap each. `kl` returns the gradient with respect to `q`,
the *parameterised* distribution, and `q` is clipped before the log but not before
the divide:

```python
    q = np.clip(q, eps, None)
    loss = float((p * np.log(np.clip(p, eps, None) / q)).sum(axis=-1).mean())
    return loss, -p / (q * p.shape[0])
```

The gradient of `p log(p/q)` in `q` is `−p/q` — no `log` survives. In `js` the
mixture `m` also depends on `q`, which adds a second term:

```python
    grad = 0.5 * np.log(q / m) + 0.5 * (1.0 - q / m)
```

`d/dq [½ q log(q/m)] = ½ log(q/m) + ½`, and `d/dq [½ p log(p/m)] = −½ p/m`; the two
combine to `½ log(q/m) + ½(1 − q/m)`. Keeping only the first term is the usual
mistake, and it still points downhill, so only a gradient check catches it.

`wasserstein` honestly returns `None` for its gradient: the 1-D closed form is a
sort, and `np.sort` is piecewise linear with a permutation derivative that is not
worth encoding. It is in the file as a *metric* to monitor, not a training loss.

Finally, `_numgrad` is the file's own test harness and worth understanding because it
explains why every loss takes its input as an argument rather than closing over it:

```python
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = val()
        x[i] = old - h
        fm = val()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
```

It perturbs `x` **in place**, so the closure inside `val()` must read `x` at call
time. Every check in `main` is written as `lambda: fn(...)` for exactly this reason;
writing `lambda: fn(y, p.copy())` would gradient-check a constant and pass trivially.

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
