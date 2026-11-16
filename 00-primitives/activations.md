# Activations

`activations.py` · Primitives · Tier 1 · 17 variants

## Intuition

An activation is the only source of nonlinearity in a feed-forward network. Without
it, any stack of matrices collapses to one matrix: `W₂W₁x = Wx`. So the activation
controls three things at once — the function class, the gradient scale, and what
the network does at the origin.

The whole design space is one question: *what do you do for negative inputs?*
Kill them (ReLU), leak a little (LeakyReLU, ELU), keep a smooth S-curve (GELU,
SiLU, Mish), or let the gate modulate a separate value branch (GLU family).

## Variants

| Variant | `f(x)` for `x < 0` | Why pick it |
|:--|:--|:--|
| `relu` | `0` | Default for CNNs. Cheapest, sparse, but dead units. |
| `leaky_relu` | `0.01x` | One-line fix for dead units. |
| `prelu` | `αx`, `α` learned | Learned slope, per-channel; small accuracy gain over ReLU. |
| `elu` | `α(eˣ−1)` | Smooth at 0, saturating negativity; mean activations closer to 0. |
| `selu` | `λα(eˣ−1)` with fixed `λ, α` | Self-normalising: keeps unit variance across depth without BN, if weights are also initialised correctly. |
| `gelu_tanh` | `≈ 0` but smooth | Transformer default. Tanh approximation of the erf form. |
| `gelu_erf` | `≈ 0` but smooth | Exact GELU; the form in the paper. |
| `silu` | `≈ −0.278` at `x=−1.28` | Smooth, non-monotone; the `Swish` of `SwiGLU`. |
| `mish` | small negative dip | Smooth non-monotone, better than SiLU on some vision tasks. |
| `softplus` | `log(1+eˣ)` | Smooth ReLU without a hard zero; used as a variance/rate head. |
| `glu` | gate `σ(b)` | Multiplicative interaction, not just a bend. |
| `swiglu` | `SiLU(a)·b` | The FFN of modern LLMs: better than `ReLU(Wx)` at equal params. |
| `geglu` | `GELU(a)·b` | Paired with `gelu` models; same idea. |
| `softmax` | dense distribution | Output layer for single-label classification. |
| `log_softmax` | dense log-probabilities | Numerically stable training loss (`NLLLoss` input). |
| `sparsemax` | exact zeros | Sparse attention/classification; Euclidean projection onto the simplex. |
| `entmax` | sparse, `α=1.5` | Interpolates softmax ↔ sparsemax with one parameter. |

## Derivation

**GELU.** Multiply the input by the probability that a standard normal is below it:

$$
\text{GELU}(x) = x\,\Phi(x) = x \cdot \tfrac12\left(1 + \mathrm{erf}\!\left(\frac{x}{\sqrt2}\right)\right)
$$

The derivative uses $\Phi'(x) = \phi(x) = e^{-x^2/2}/\sqrt{2\pi}$:

$$
\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x\,\phi(x)
$$

The tanh form replaces $\Phi$ with $\tfrac12(1+\tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$, which is the same curve to ~1e-3 and cheaper before `erf` was fast.

**SELU.** `selu(x) = λ·elu(x)` with `λ = 1.0507`, `α = 1.6733`. These are not tuned: they are the fixed point of the variance recursion for unit-mean, unit-variance inputs, so `Var[output] ≈ Var[input]` for any depth. That is why `selu` is advertised as self-normalising — and why the guarantee evaporates the moment your data is not standardised.

**Sparsemax.** Project the logits onto the probability simplex:

$$
\underset{p \in \Delta^{K-1}}{\arg\min} \; \|p - z\|_2^2
\quad\Longrightarrow\quad
p_i = \max(0,\; z_i - \tau)
$$

where $\tau$ is the threshold that makes the sum equal 1. Sort descending, find the largest $k$ with $z_{(k)} - \frac{1}{k}\left(\sum_{j\le k} z_{(j)} - 1\right) > 0$, and set $\tau$ from that $k$. The output has exactly $k$ non-zeros.

**Entmax(α).** Replace the Euclidean projection with a Tsallis entropy regulariser. For `α = 1.5` the closed form is

$$
p_i = \max\!\left(0,\; \tfrac{z_i}{2} - \tau\right)^2
$$

with $\tau$ again fixed by $\sum_i p_i = 1$, found here by bisection. `α = 1` recovers softmax, `α = 2` recovers sparsemax.

## Code walkthrough

The file is three dispatch functions and a handful of shared kernels. `activation`
and `activation_grad` are deliberately parallel `if`-chains: every variant appears
in both, in the same order, so a new activation is added by writing two adjacent
branches and nothing else. `POINTWISE`, `SIMPLEX` and `GATED` are sets used by the
demo to iterate over every variant, not by the call path.

Everything starts from two overflow-safe kernels:

```python
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free
```

`1/(1+exp(-x))` overflows to `inf` at `x ≈ -710`, and `inf` in the denominator
gives `0`, which happens to be right — but the gradient path later multiplies
`s*(1-s)` and a `0 * inf` there yields `nan`. The tanh identity is exact and never
saturates the intermediate. Same trick for softplus:

```python
def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)
```

`np.log(1 + np.exp(x))` is the formula; `logaddexp` is the implementation, because
it factors out the larger term internally.

The negative branch of `elu` and `selu` is where a naive version breaks:

```python
    if variant == "elu":
        return np.where(x > 0.0, x, alpha * (np.expm1(np.minimum(x, 0.0))))
```

`np.where` evaluates *both* branches on the whole array. Without the
`np.minimum(x, 0.0)`, positive inputs still go through `np.exp` and a pre-activation
of 800 returns `inf` — which then poisons the backward pass even though the forward
value looked correct. `expm1` is the other half: `exp(x) - 1` loses all precision for
`x → 0⁻`, where `elu` must match `relu` to first order.

Gradients are hand-derived per variant; `gelu_tanh` is the one worth reading because
it is a chain rule through a cubic:

```python
    if variant == "gelu_tanh":
        t = np.tanh(GELU_C * (x + GELU_K * x**3))
        du = GELU_C * (1.0 + 3.0 * GELU_K * x**2)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du
```

`0.5*(1+t)` is `Φ(x)` in the tanh approximation, `0.5*x*(1-t²)` is `x·φ(x)`, and `du`
is the derivative of the *inner* argument `√(2/π)(x + 0.044715x³)`. Forgetting `du`
is the classic bug: the loss still decreases, just to the wrong place, so it never
shows up as a crash. Every gradient in the file is checked against central
differences in `main`, which is what catches that class of error.

The simplex family switches from elementwise maps to projections, and `sparsemax`
is the sort-and-threshold form of the derivation:

```python
    zs = -np.sort(-z, axis=axis)
    css = np.cumsum(zs, axis=axis) - 1.0
    ...
    support = zs - css / k > 0
    k_z = support.sum(axis=axis, keepdims=True)
    tau = np.take_along_axis(css, (k_z - 1).astype(int), axis=axis) / k_z
    return np.maximum(z - tau, 0.0)
```

`css` is the running sum *minus the unit mass of the simplex* — that `- 1.0` is the
entire constraint `Σp = 1` folded into a prefix sum, and dropping it produces a
projection onto `{p ≥ 0}` that does not sum to one. The support test is the
paper's condition `z₍ₖ₎ > cssₖ/k`, and `k_z ≥ 1` always holds because the largest
coordinate beats its own mean.

`entmax15` cannot vectorise the threshold, because each row has its own `τ`, so it
loops rows and bisects:

```python
        lo = zz.max() / 2.0 - 1.0
        hi = zz.max() / 2.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            p = np.maximum(zz / 2.0 - mid, 0.0) ** 2
            if p.sum() > 1.0:      # tau too small: masses still too large
                lo = mid
            else:
                hi = mid
```

The bracket is not arbitrary: at `τ = max(z)/2` every term is zero (sum 0), and at
`τ = max(z)/2 − 1` the largest term alone contributes 1. The direction of the
update is the line that is easy to invert — mass is *decreasing* in `τ`, so too
much mass means move `lo` up. Getting it backwards still converges, to a bracket
that contains no root, and the output is silently wrong by a factor.

Finally the gated units fix a convention that is a recurring source of shape bugs:

```python
def gated(a: np.ndarray, b: np.ndarray, variant: str = "swiglu") -> np.ndarray:
    """Gated linear unit. a is the value branch, b is the gate branch."""
```

`swiglu` is `SiLU(a)·b`, not `a·σ(b)` — Shazeer's paper writes the gate first, many
implementations call the *first* half the gate. The demo asserts the identity
`gated(a, b, "swiglu") == silu(a) * b` so the convention is pinned by a test rather
than by a comment.

## Complexity

All pointwise variants are `O(n)` time and memory, one elementwise op. `sparsemax` is `O(n log n)` per row from the sort. `entmax15` as implemented is `O(n · iters)` per row with 40 bisection steps — fine for attention over a few thousand keys, not for million-way softmax. Gated variants double the parameter count of the layer (`2·d·d_ff` instead of `d·d_ff`).

## Failure modes

1. **Dead ReLU.** A unit whose pre-activation is negative for every input has zero gradient forever; it never recovers. Diagnostic: fraction of units with all-zero output over a batch > 30% → switch to `leaky_relu` or lower the learning rate.
2. **Exploding `exp`.** `elu`/`selu` with a large negative pre-activation underflow, and a naive `softmax` overflows. Both are handled here (`expm1` on clipped input, max-subtraction), but a hand-rolled version is the classic NaNs-at-step-3000 bug.
3. **Activation/initialisation mismatch.** `selu` assumes unit-variance inputs and its own init; `gelu` assumes the variance-preserving He/Xavier init. Pairing them wrongly silently costs accuracy.
4. **`sparsemax` in attention.** Zero probabilities give zero gradient, so a key that is never selected never learns. `entmax` with `α ∈ (1, 1.5]` is the safer sparse choice.

## Applications

- **FFN blocks:** `swiglu`/`geglu` at `d_ff ≈ 8/3·d` to match parameter count of a 4d ReLU FFN.
- **Output heads:** `log_softmax` + NLL for classification; `softplus` for positive rates (Poisson GLM, variance heads); `sigmoid` for multi-label.
- **Sparse structured prediction:** `sparsemax`/`entmax` where a hard top-k decision is needed and you want gradients (attention, keyphrase extraction, structured outputs).
- **Normalising-free training:** `selu` is a legitimate alternative to BatchNorm in small MLPs, but only with standardised inputs and the matching init.

## References

- Hendrycks & Gimpel, *Gaussian Error Linear Units*, 2016. arXiv:1606.08415
- Klambauer et al., *Self-Normalizing Neural Networks*, 2017. arXiv:1706.02515
- Shazeer, *GLU Variants Improve Transformer*, 2020. arXiv:2002.05202
- Martins & Astudillo, *From Softmax to Sparsemax*, 2016. arXiv:1602.02068
- Peters, Niculae & Martins, *Sparse Sequence-to-Sequence Models* (entmax), 2019. arXiv:1905.05702
