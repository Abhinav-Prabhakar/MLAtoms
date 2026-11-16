# Adaptive methods

`adaptive.py` · Optimization · Tier 1 · 7 variants

## Intuition

Adam asked "how big have this coordinate's gradients been?" and divided by the
answer. Every method here is a different answer to the same question, and the
differences are not cosmetic:

- **How long is the memory?** Adagrad never forgets; RMSProp forgets at rate `β`.
- **Mean or second moment?** Centred RMSProp divides by the *variance*, not the
  raw second moment.
- **What is the increment?** Yogi adds `(1−β₂)|g|²` only when `|g|²` exceeds the
  accumulator — an absolute, not relative, update.
- **Do you need a denominator at all?** SignSGD says no: the sign is the whole
  normalisation.

## Variants

| Variant | Denominator | State per parameter | Good at | Bad at |
|:--|:--|:--|:--|:--|
| `adagrad` | `√(Σg²)` | 1 | sparse features, convex problems | long runs (LR → 0) |
| `adadelta` | `RMS[Δ] / RMS[g]` | 2 | no LR to tune | slow first steps |
| `rmsprop` | `√EMA[g²]` | 1 | RNNs, non-stationary data | no bias correction |
| `rmsprop_centered` | `√(EMA[g²] − EMA[g]²)` | 2 | stochastic gradients | constant gradients (denominator → 0) |
| `yogi` | `√v` with an additive `v` | 2 | avoiding LR blow-ups after a spike | little theory beyond the original paper |
| `signsgd` | `sign(g)` | 0 | huge batches, communication-efficient training | fine convergence, small batches |
| `sm3` | `min` over factorised accumulators | `Σdᵢ` | giant embedding tables | accuracy of the denominator |

## Derivation

**Adagrad is a regret bound.** For a convex problem with `G = max‖g‖`, the
AdaGrad regret is `O(G·√T)` against the best fixed point, which is optimal for
online convex optimisation. The proof requires `η_t = η/√t`, and the accumulator
`Σ g²` implements exactly that when gradients are stationary: after `t` steps, the
denominator is `√t·σ`, so the step is `η/(σ√t)`. The price is that the LR decays
monotonically even when the problem does not need it.

**RMSProp forgets.** Replace the sum with an EMA:

$$
v_t = \beta v_{t-1} + (1-\beta)g_t^2,
\qquad \theta \leftarrow \theta - \eta\,\frac{g_t}{\sqrt{v_t}+\epsilon}
$$

which tracks a local scale. There is no bias correction, so the first steps are
too large by `1/√(1−β)` — irrelevant in practice, because warmup covers it.

**Centred RMSProp divides by the variance.** `v − m²` estimates `Var[g]`; with
independent noise, dividing by the *variance* rather than the second moment gives
a step that grows as the noise shrinks. On a deterministic gradient `Var[g] → 0`,
so the denominator collapses to `ε` and the step explodes — see the demo, where the
centred variant travels 35× further after 100 unit steps. That is not a bug in the
demonstration; it is the reason centred RMSProp is a niche choice.

**Adadelta is dimensionless.** The numerator is the RMS of recent *updates*, the
denominator the RMS of recent *gradients*:

$$
\Delta_t = \frac{\sqrt{E[\Delta^2]_{t-1}+\epsilon}}{\sqrt{E[g^2]_t+\epsilon}}\,g_t
$$

Both sides carry gradient units, so their ratio is scale-free: changing the units
of the loss does not change the trajectory. That is why `lr=1` converges in the
demo, and why the first steps are tiny (the numerator starts at `√ε`).

**Yogi is additive.** `v_t = v_{t−1} + (1−β₂)\,g_t^2\,\text{sign}(g_t^2 − v_{t−1})`.
When gradients are large the accumulator grows by `(1−β₂)g²`; when they are small
it shrinks by the *same absolute* amount, so it cannot collapse geometrically the
way Adam's can. In the demo, after one large gradient followed by 200 tiny ones,
Adam's `v` sits at `10⁻⁴` (it has adapted to the quiet regime) while Yogi's stays
at `0.098` — Adam's effective LR has grown 31×, Yogi's has not.

**SignSGD.** `θ ← θ − η·sign(g)` moves every coordinate by exactly `η`. For
convex problems with bounded gradient noise it matches SGD's rate up to constants
and is `O(d)` cheaper to communicate (1 bit per coordinate). It cannot converge to
better than `O(η)` from the optimum, which is why it is used with large batches,
momentum, and a decaying LR.

**SM3.** A full accumulator for an `m×n` tensor costs `mn`; SM3 keeps `m+n`, one
per mode, and sets `v̂_ij = min(rowᵢ, colⱼ)`. Since `rowᵢ = Σ_j g_ij² ≥ g_ij²`,
the minimum is still an upper bound — a valid, slightly conservative denominator.
The memory saving is the whole point: a `10⁶ × 10³` embedding table goes from
`10⁹` floats to `10⁶+10³`.

## Code walkthrough

One class, seven branches inside `step`. Nothing is shared except the parameter
list and the optional momentum buffer, because the methods disagree about *what
state* is needed — that is the point of the file.

The simplest branch is SignSGD:

```python
            if self.variant == "signsgd":
                d = np.sign(g)
```

`d` is the *direction* that the rest of the method multiplies by `lr`. Every branch
computes a `d`, and `p -= self.lr * d` at the bottom is common; keeping that shape
makes the differences comparable line by line. Note `np.sign(0) = 0`, so a
coordinate with a zero gradient freezes rather than dithering.

Adagrad accumulates, RMSProp forgets:

```python
            elif self.variant == "adagrad":
                self.acc[i] += g * g
                d = g / (np.sqrt(self.acc[i]) + self.eps)
```

```python
            elif self.variant == "rmsprop":
                self.acc[i] = self.beta * self.acc[i] + (1 - self.beta) * g * g
                d = g / (np.sqrt(self.acc[i]) + self.eps)
```

The `+=` is what makes Adagrad's rate `η/√t` under a constant gradient and what
makes it stall eventually; the EMA is what makes RMSProp stationary. They differ by
one operator, and the demo's `2√T + ζ(½)` check is what distinguishes them
quantitatively rather than by eye.

Adadelta's ordering is the algorithm:

```python
                self.acc[i] = self.rho * self.acc[i] + (1 - self.rho) * g * g
                rms_d = np.sqrt(self.delta[i] + self.eps)
                rms_g = np.sqrt(self.acc[i] + self.eps)
                d = (rms_d / rms_g) * g
                self.delta[i] = self.rho * self.delta[i] + (1 - self.rho) * d * d
```

`rms_d` is read from the *previous* `delta`, before this step's `d` is known. That
is not a bug or a lag: the method is defined with the previous update RMS, which is
what makes it implicit and removes the learning rate. Updating `delta` before
computing `d` (the "obvious" reordering) gives a different, unstable method.

Centred RMSProp subtracts the squared mean and needs the clamp:

```python
                var = np.maximum(self.acc[i] - self.acc2[i] ** 2, 0.0)
                d = g / (np.sqrt(var) + self.eps)
```

`E[g²] − E[g]²` is non-negative in exact arithmetic but can be `−1e-17` after two
EMAs, and `sqrt` of that is `nan`. The `maximum(·, 0)` is not defensive programming;
it is required for the method to run at all. When the gradient is constant, `var → 0`
and `d ≈ g/ε`, which is the 35× overshoot the doc describes — a real property, not a
numerical artefact.

Yogi's update is written in the `sign` form, which is the part people misread:

```python
                self.acc[i] += (1 - self.beta2) * np.sign(g * g - self.acc[i]) * g * g
```

If `g² > v`, this adds `(1−β)g²`; if `g² < v`, it *subtracts* `(1−β)g²`. It is not
`(1−β)(g² − v)` — that would be an EMA and would decay geometrically. The additive
form is what keeps `v` near the largest recent scale instead of collapsing to the
latest quiet one. The first moment is a separate EMA and *is* bias-corrected:

```python
                mhat = self.acc2[i] / (1 - self.beta ** self.t)
                vhat = self.acc[i] / (1 - self.beta2 ** self.t)
                d = mhat / (np.sqrt(vhat) + self.eps)
```

`acc2` is reused from the centred-RMSProp branch; the two variants are mutually
exclusive, so one buffer serves both.

SM3 is the only branch that does not have a full-size accumulator:

```python
        accs = self.sm3[i]
        for k, acc in enumerate(accs):
            axes = tuple(ax for ax in range(g.ndim) if ax != k)
            acc += (g * g).sum(axis=axes)
        v = None
        for k, acc in enumerate(accs):
            shape = [1] * g.ndim
            shape[k] = g.shape[k]
            shaped = acc.reshape(shape)
            v = shaped if v is None else np.minimum(v, shaped)
```

For a matrix, mode 0 accumulates over rows into a `(cols,)` vector and mode 1 into a
`(rows,)` vector. Reducing over `ax != k` is the non-obvious indexing: it keeps the
`k`-th dimension and sums everything else, so `acc` has length `g.shape[k]`. The
broadcast then reshapes `acc` to `(1, cols)` or `(rows, 1)` and takes the elementwise
minimum. Since `row_i = Σ_j g_ij² ≥ g_ij²`, the min is still an upper bound on the
true second moment — conservative, never optimistic. The memory is `Σ dᵢ` instead of
`Π dᵢ`, which is the entire reason the method exists; for a `2×3` tensor the demo
checks `5 < 6`, and the ratio improves with every added dimension.

Finally, momentum acts on the *normalised* direction, not on the raw gradient:

```python
            if self.momentum:
                self.vel[i] = self.momentum * self.vel[i] + d
                d = self.vel[i]
            p -= self.lr * d
```

For Adam-family methods this is the correct place (momentum of the adaptive update),
and moving it above the branch — `v ← μv + g` — changes the method to one with two
coupled scales. The demo does not use momentum, so this line is only exercised by a
reader who passes the flag; it is still the one that decides whether "Adam with
momentum" means what they think.

## Complexity

| Variant | Extra memory | Time per step |
|:--|:--|:--|
| `adagrad`, `rmsprop` | `P` | `O(P)` |
| `adadelta`, `rmsprop_centered`, `yogi` | `2P` | `O(P)` |
| `signsgd` | `0` (or `P` with momentum) | `O(P)` |
| `sm3` | `Σ dᵢ` | `O(P)` for the min and the update |

All are elementwise, so the cost is memory bandwidth. The memory column, not the
time column, is what decides whether a model fits.

## Failure modes

1. **Adagrad stalls.** After `10⁵` steps the accumulated denominator is huge and
   the LR is effectively zero. It is a convex/online method; for deep nets use
   Adagrad only with a schedule that resets the accumulator, or switch to RMSProp.
2. **Centred RMSProp on clean gradients.** The variance goes to zero and the step
   explodes. Keep a floor on the denominator, or use plain RMSProp.
3. **SignSGD with small batches.** The sign of a noisy gradient is not a descent
   direction; the theory needs large batches or variance reduction. With batch
   size 8 it often oscillates instead of descending.
4. **SM3 on dense, correlated gradients.** `min(row, col)` overestimates the
   per-coordinate second moment when the gradient matrix is dense, so the
   effective LR is smaller than intended.
5. **Mixing `ε` conventions.** `√(v+ε)` vs `√v+ε` changes behaviour by orders of
   magnitude when `v` is tiny; pick one and keep it consistent with the optimizer
   state you load.

## Applications

- **Adagrad:** sparse bag-of-words, CTR models, any problem where most features
  appear rarely — the per-feature LR is exactly what you want and the total step
  count is modest.
- **RMSProp:** RNNs, RL (where the value loss is non-stationary), and as the
  denominator inside RMSProp-style optimisers in JAX/Optax.
- **Adadelta:** quickly trying a model without tuning the LR; it is a reasonable
  default when you have no time for a sweep.
- **SignSGD:** federated learning and distributed training where upload bandwidth
  dominates, combined with error feedback to recover accuracy.
- **SM3:** recommendation models with vocabulary in the millions, where a full
  Adam state would dominate memory.

## References

- Duchi, Hazan & Singer, *Adaptive Subgradient Methods* (Adagrad), 2011. JMLR.
- Zeiler, *ADADELTA: An Adaptive Learning Rate Method*, 2012. arXiv:1212.5701
- Tieleman & Hinton, *Lecture 6.5 — RMSProp*, Coursera course notes, 2012.
- Zaheer et al., *Adaptive Methods for Nonconvex Optimization* (Yogi), NeurIPS 2018.
- Bernstein et al., *signSGD: Compressed Optimisation for Non-Convex Problems*, ICML 2018.
- Anil, Gupta, Koren & Singer, *Memory-Efficient Adaptive Optimization* (SM3), NeurIPS 2019. arXiv:1901.11150
