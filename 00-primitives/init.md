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
