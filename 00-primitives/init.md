# Initialisation

`init.py` · Primitives · Tier 1 · 10 variants

## Intuition

A network is a sequence of linear maps and nonlinearities. Each map multiplies the
input variance by `fan_in · Var(w)`; each ReLU halves it. If the product per layer
is not ≈ 1, the signal decays geometrically going forward and the gradient decays
going backward. Initialisation is how you make that product equal one *before any
training happens*.

## Variants

| Variant | Variance | Matched to | Note |
|:--|:--|:--|:--|
| `zeros` | 0 | — | Only correct for biases and residual-branch outputs. |
| `xavier_uniform` | `2/(fan_in+fan_out)` | linear, tanh, sigmoid | The original Glorot & Bengio rule. |
| `xavier_normal` | `2/(fan_in+fan_out)` | same | Same variance, unbounded support. |
| `he_uniform` | `2/fan_in` | ReLU family | Factor 2 compensates the ReLU halving. |
| `he_normal` | `2/fan_in` | ReLU family | Default for conv/MLP. |
| `orthogonal` | spectrum = 1 | RNNs, deep residual | Gradient norm preserved exactly. |
| `lsuv` | data-dependent | any, per layer | Orthogonal then rescale on a real batch. |
| `fixup` | `1/fan_in · 1/(depth−1)` | residual nets without BN | Variance grows linearly, not exponentially. |
| `truncated_normal` | `2/fan_in`, clipped at 2σ | ReLU, small nets | Removes rare large weights that destabilise. |
| `scaled_variance` | `1/fan_in` | normalised residual streams | Recent default when LN already controls scale. |

## Derivation

**Xavier.** Want `Var(y) = Var(x)` for `y = Wx` with `W ∈ ℝ^{d_out×d_in}` and
independent zero-mean weights:

$$
\operatorname{Var}(y_j) = d_{in}\,\operatorname{Var}(w)\,\operatorname{Var}(x)
\;\overset{!}{=}\;\operatorname{Var}(x)
\quad\Longrightarrow\quad
\operatorname{Var}(w) = \frac{1}{d_{in}}
$$

The backward pass imposes the symmetric condition `Var(w) = 1/d_out`. Averaging
the two gives `2/(d_in + d_out)`, which is the published Xavier variance.

**He.** With ReLU, `Var(relu(z)) = ½ Var(z)` because the negative half is zeroed.
To keep the second moment equal, the weights must be twice as large:

$$
\operatorname{Var}(w) = \frac{2}{d_{in}}
$$

This is also why He init does *not* preserve variance after ReLU — ReLU output has
a positive mean, so `Var = E[y²] − E[y]² < E[y²]`. The invariant is the second
moment, which is what the demo checks. Getting this wrong is the most common
"my unit test for He init fails" experience.

**Orthogonal.** Draw a Gaussian `A`, take `A = QR`, and fix the sign of each column
by `sign(diag(R))` so the distribution is uniform over the orthogonal group. Then
`WᵀW = I`: every singular value is 1, so no direction is amplified or attenuated —
the strongest possible statement about gradient norm preservation.

**LSUV.** Orthogonal init fixes the spectrum, not the scale on your data.
Compute `y = act(Wx)` on one real batch, set `W ← W · target/std(y)`, repeat.
Because `act` is not linear, one step is not enough; the implementation iterates.
This beats closed-form init when the activation is unusual or the inputs are not
standardised.

## Code walkthrough

`initialize` is one dispatch on `(fan_out, fan_in)` — never `(fan_in, fan_out)`.
That convention is stated in the signature docstring because every rule below is
asymmetric in the two and the wrong order produces a plausible-looking matrix that
is silently mis-scaled.

The uniform variants derive their bound from `Var[U(−a,a)] = a²/3`:

```python
    if variant == "xavier_uniform":
        a = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-a, a, size=shape)
    ...
    if variant == "he_uniform":
        a = gain * np.sqrt(6.0 / fan_in)
        return rng.uniform(-a, a, size=shape)
```

Setting `a²/3 = 2/(fan_in+fan_out)` gives `a = √(6/(fan_in+fan_out))`; setting
`a²/3 = 2/fan_in` gives `√(6/fan_in)`. The `6` is not a fudge — it is `3 × 2`, and
an implementation that copies `√(2/fan_in)` into the uniform branch (a very common
copy-paste) has one third of the intended variance. When porting between frameworks
check the distribution, not just the name: PyTorch's `kaiming_uniform_` uses the
same `6` and its `a` parameter is a *negative slope*, not a scale.

The orthogonal path builds a tall matrix first, then transposes:

```python
    a = rng.normal(size=(max(fan_out, fan_in), min(fan_out, fan_in)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if fan_out < fan_in:
        q = q.T
    return gain * q[:fan_out, :fan_in]
```

`np.linalg.qr` on a tall `m×n` matrix gives `Q` of shape `m×n` with orthonormal
columns, i.e. `QᵀQ = I`. When `fan_out ≥ fan_in` that is exactly the shape we need
and `WᵀW = I`; when `fan_out < fan_in` the slice must come from `Qᵀ`, so that
`WWᵀ = I`. The sign fix `sign(diag(r))` is what makes the distribution uniform over
the orthogonal group: without it, QR biases the first column toward the positive
orthant. Note `sign` is applied to the *columns* via broadcasting, matching `diag(r)`
against the columns of `q`.

LSUV is the variant that cannot be written as a closed form:

```python
    q = _orthogonalize(W)
    for _ in range(iters):
        y = x @ q.T
        if activation is not None:
            y = activation(y)
        q = q * (target / max(float(np.std(y)), 1e-12))
```

One rescale is not enough because `act(cWx) ≠ c·act(Wx)` for a nonlinearity: after
scaling, the ReLU threshold moves relative to the data and the output std is no
longer `c` times the old one. The loop converges in a handful of iterations. The
`max(..., 1e-12)` guards a dead layer whose activation is exactly zero — otherwise
the scale is `inf` and the next multiply produces `nan` that survives training
forever. `_orthogonalize` is separate from `_orthogonal` because LSUV starts from a
*provided* weight matrix and must not draw a new one.

The two depth-aware schemes are one line each and the line is the whole idea:

```python
        scale = 1.0 / np.sqrt(max(depth - 1, 1))
        return rng.normal(0.0, gain * np.sqrt(1.0 / fan_in), size=shape) * scale
```

Fixup's residual branch is scaled by `1/√(depth−1)` so that `depth` branches add in
quadrature to unit variance — the sum grows *linearly* in variance, which is what
lets a bias-free, BN-free residual net reach depth 10k. Removing the scale gives
exponential blow-up; using `depth` instead of `depth−1` is a small, systematic
under-scale that costs accuracy but never crashes, which is why it survives review
in real codebases.

`truncated_normal` clips rather than resamples:

```python
        w = rng.normal(0.0, sd, size=shape)
        return np.clip(w, -2.0 * sd, 2.0 * sd)
```

Clipping leaves a point mass at `±2σ` instead of the smooth truncated density, so
the realised variance is slightly *below* `2/fan_in` — the demo allows 12%. The
alternative (rejection sampling) is exact but has an unbounded loop; in a weight
initialiser the point mass is harmless and the vectorisation is worth it. This is
worth knowing because the same shortcut in a *sampler* is not harmless.

`scaled_variance` is deliberately `1/fan_in`, not `2/fan_in`:

```python
        return rng.normal(0.0, 1.0, size=shape) / np.sqrt(fan_in)
```

It is paired with a residual stream that already has LayerNorm, so the missing
factor of two is absorbed downstream. Reading it as "He init with a bug" is the
usual mistake; the invariant being targeted is `E[‖Wx‖²] = ‖x‖²` for `x` with unit
per-coordinate variance, and LN re-establishes that at the next block anyway.

## Complexity

`O(d_in·d_out)` for all schemes. `orthogonal` costs `O(d³)` for the QR of a square
matrix — negligible next to training. `lsuv` costs one forward pass per layer per
iteration (8 here).

## Failure modes

1. **Init/activation mismatch.** Xavier + ReLU halves the signal every layer;
   He + tanh doubles it. The symptom is a loss that is flat (too small) or NaN
   within 100 steps (too large).
2. **Forgetting the bias.** Biases start at zero, except in the last layer of a
   classifier with imbalanced classes: `b = log(p/(1−p))` from the prior saves
   several thousand steps.
3. **Resetting pretrained weights.** Fine-tuning code that calls `initialize` on
   every module destroys the pretrained model. Initialise on construction, load
   after.
4. **Orthogonal on non-square matrices.** `WᵀW ≠ I` when `d_out < d_in`; you get
   `WWᵀ = I` instead. Both are fine, but the conservation property differs.
5. **Init as a substitute for normalisation.** Fixup exists because zero-init +
   depth scaling *can* replace BN — but only with the matching architecture. He
   init without BN in a 100-layer net still explodes.

## Applications

- Default recipe: `he_normal` for conv/MLP with ReLU/GELU, `xavier` for tanh,
  `orthogonal` for RNNs and very deep residual stacks, `scaled_variance` when the
  block already has LayerNorm.
- Diagnostics: if adding one more layer halves your activation variance, your init
  is wrong — measure `mean(y²)` per layer before blaming the optimiser.
- Transfer learning: init matters only for the new head.

## References

- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, 2010.
- He et al., *Delving Deep into Rectifiers*, 2015. arXiv:1502.01852
- Saxe, McClelland & Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear networks*, 2013. arXiv:1312.6120
- Mishkin & Matas, *All you need is a good init* (LSUV), 2015. arXiv:1511.06422
- Zhang, Dauphin & Ma, *Fixup Initialization*, 2019. arXiv:1901.09321
