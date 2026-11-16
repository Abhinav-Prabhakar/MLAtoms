# Kernel ridge

`kernel_ridge.py` · Kernels & instance-based · Tier 1 · 6 variants

## Intuition

Ridge regression is least squares with a penalty on the coefficients. Do the
same thing in a feature space of infinite dimension and something remarkable
happens: the penalty keeps the solution finite, and the solution provably lives
in the span of the training points. So the infinite-dimensional problem
collapses to an `n × n` linear system that you can write down in one line.

$$
(K + λI)α = y
$$

No optimiser, no support vectors, no slack. Kernel ridge is what a kernel
machine looks like when the loss is squared instead of hinged: the same
hypothesis space, the same kernel, a strictly easier algebra — and, as the price,
every training point participates in the answer, so the model is not sparse.

It is also the *only* kernel method whose leave-one-out error is free. Because
the fit is linear in `y`, deleting one point scales that point's residual by a
known factor; the entire LOO curve over a `λ` grid costs a handful of solves
instead of `n` refits per candidate.

## Variants

| Variant | What changes | Why you would pick it |
|:--|:--|:--|
| `linear` | `K = X Zᵀ` | reproduces primal ridge exactly; the dual is a sanity check |
| `rbf` | `K = exp(−γ‖x−z‖²)` | smooth, translation invariant, the default |
| `laplacian` | `K = exp(−γ‖x−z‖₁)` | steeper sample paths; piecewise-constant targets |
| `polynomial` | `K = (γ X Zᵀ + r)^p` | exact degree-`p` interactions, finite-dimensional |
| `nystrom` | `K ≈ ZW⁻¹Zᵀ` on `m` landmarks | `n` large, `m ≪ n`; `O(nm²)` instead of `O(n³)` |
| `leave_one_out` | `λ` chosen by the closed-form LOO | tuning with no cross-validation folds |

## Derivation

### From the RKHS to a linear system

Let `H` be a reproducing-kernel Hilbert space with kernel `k`, and consider

$$
\min_{f ∈ H}\ \tfrac12\sum_i \big(y_i - f(x_i)\big)^2 + \tfrac{λ}{2}\|f\|_H^2 .
$$

The **representer theorem** says the minimiser has the form
`f = Σ_j α_j k(x_j, ·)`. The proof is a projection: write any `f` as its
projection onto `span{k(x_j,·)}` plus an orthogonal residual. The residual
changes no `f(x_j)` — the evaluation functionals are inner products with
`k(x_j, ·)` — but it strictly increases `‖f‖_H²`. So the minimiser has none.

That single move turns a problem over functions into a problem over `n` numbers.
Substituting `f` gives `‖f‖_H² = αᵀKα` and `f(x_i) = (Kα)_i`, hence

$$
\min_{α}\ \tfrac12\|y - Kα\|^2 + \tfrac{λ}{2} α^T K α ,
$$

and setting the gradient to zero:

$$
-K(y - Kα) + λKα = 0 \;\Longrightarrow\; (K + λI)α = y .
$$

`λ > 0` is what makes this solvable even when `K` is singular — and it must be,
because for many kernels `K` is rank-deficient the moment `n > d`.

### Two identities worth memorising

The normal equations give the residual identity

$$
r \equiv y - Kα = λα ,
$$

so the residual is proportional to the dual weights: points the model cannot fit
are exactly the ones with large `α`. Multiplying the normal equations by `αᵀ`:

$$
α^T K α + λ α^T α = y^T α .
$$

The left side is `‖f‖_H² + λ‖α‖²`, and since `‖r‖² = λ²‖α‖²`, the primal
objective at the optimum reduces to

$$
\tfrac12\|r\|² + \tfrac{λ}{2}\|f\|_H^2 = \tfrac{λ}{2}\,y^T α .
$$

Both identities are exact, cheap to check, and the demo asserts them.

### Classification by regression

For `y ∈ {±1}` this is the *least-squares SVM*. It is a perfectly usable
classifier and it is not a margin maximiser: the hinge penalises only points
inside the margin, the square penalises every point, so the solution is pulled
by easy points that the SVM would ignore. Expect KRR to lose to a real SVM when
the classes overlap, and to match it when they are clean.

### Nyström: a rank-`m` kernel

Sample `m` landmark points, let `Z = K(X, U)` be `n × m` and `W = K(U, U)` be
`m × m`. Then

$$
K \approx \tilde K = Z W^{-1} Z^T ,
$$

which is exact whenever `K` has rank at most `m` — true for a linear kernel with
`m ≥ d`, and a genuinely good approximation for smooth kernels whose spectrum
decays. The naive fit still forms an `n × n` matrix, but Woodbury removes even
that:

$$
(\tilde K + λI)^{-1} y = \frac{y - Zc}{λ},
\qquad c = (λW + Z^T Z)^{-1} Z^T y ,
$$

an `m × m` solve. Fitting and prediction become `O(nm²)` time and `O(nm)`
memory, so `n = 10⁵` with `m = 500` is a laptop problem.

### Leave-one-out, free

The fitted values are `ŷ = Hy` with `H = K(K+λI)^{-1}` — a *linear smoother*,
because `α` is linear in `y`. For any linear smoother, refitting without point
`i` and predicting it gives

$$
y_i - f_{-i}(x_i) = \frac{e_i}{1 - H_{ii}},
\qquad e_i = y_i - \hat y_i ,
$$

which is exact, not asymptotic. One solve produces `H` and therefore all `n`
LOO residuals; a grid of `λ` costs one `O(n³)` factorisation each, versus `n`
refits per candidate under brute force.

The intercept deserves a sentence, because it is where the identity is usually
misapplied. `fit_intercept` centres `X` and `y` once; the closed form treats
that mean as *known*, which is the standard kernel-ridge convention (the RKHS
of an RBF kernel does not contain the constant function). If you re-estimate the
mean inside each fold, you are fitting a slightly different estimator and the
numbers move by `O(1/n)`.

## Code walkthrough

The kernel table is shared with `svm.py`; only the Laplacian entry is new, and
it is the reason the wrapper exists rather than a bare import:

```python
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}")
    if kind != "laplacian":
        return _gram(X, Z, kind=kind, degree=degree, gamma=gamma, coef0=coef0)
    Z = X if Z is None else Z
    gamma = 1.0 / X.shape[1] if gamma is None else float(gamma)
    d1 = np.abs(X[:, None, :] - Z[None, :, :]).sum(axis=2)
    return np.exp(-gamma * d1)
```

The Laplacian kernel is *conditionally* positive definite, so its Gram matrix
can carry small negative eigenvalues. That is harmless with `λ > 0` and fatal
without: it is one more reason the ridge term is not optional.

The whole fit is one `solve`. There is no iteration to get subtly wrong:

```python
        if self.variant == "nystrom":
            self._fit_nystrom(Xc, yc)
        else:
            K = self._k(Xc)
            self.X_ = Xc
            self.alpha_ = np.linalg.solve(
                K + self.lam * np.eye(K.shape[0]), yc)
```

`self.X_` is the *centred* training set, and `predict` subtracts the same mean,
so the kernel is evaluated on the data the model was actually fitted to.

The LOO variant is four lines, and the comment earns its keep — `A⁻¹K` is
symmetric only in exact arithmetic, so inverting it in place would be the kind
of bug that shows up as a 1e-9 drift:

```python
        n = yc.size
        K = self._k(Xc)
        A = K + lam * np.eye(n)
        alpha = np.linalg.solve(A, yc)
        h = np.diag(np.linalg.solve(A, K))     # symmetric in exact arithmetic
        return (yc - K @ alpha) / (1.0 - h)
```

The Nyström path is Woodbury, transcribed. Note that the returned `alpha_` lives
in *landmark* space, so prediction only ever touches `m` columns:

```python
        c = np.linalg.solve(self.lam * W + Z.T @ Z, Z.T @ yc)
        self.alpha_ = np.linalg.solve(W, Z.T @ ((yc - Z @ c) / self.lam))
```

`W` is symmetrised and nudged by `1e-10` before use: duplicated landmarks make
it singular, and a rank-`m` approximation with a singular `W` throws away the
entire approximation rather than one direction.

## Complexity

| Path | Time | Memory |
|:--|:--|:--|
| build `K` | `O(n²d)` | `O(n²)` |
| solve | `O(n³)` | `O(n²)` |
| LOO over `g` values of `λ` | `g·O(n³)` | `O(n²)` |
| predict | `O(ns)` for `s` test points | — |
| Nyström (`m` landmarks) | `O(nm² + m³)` | `O(nm)` |

The `n³` solve is the ceiling. Two escapes: reduce `n` with Nyström or random
features (`random_features.py`), or reduce the *effective* rank by using a
kernel whose spectrum decays — the two are the same idea seen from different
sides.

## Failure modes

**`λ` too small.** `(K + λI)` inherits the conditioning of `K`. With duplicated
or near-duplicated points and `λ = 1e-10`, the solve is dominated by round-off
and test predictions explode between the training points. The diagnostic is
`cond(K + λI)`; the LOO curve separates the stabilising range from the
overfitting range: too small and LOO is erratic, too large and it rises
smoothly.

**`λ` chosen by training error.** Training error decreases monotonically in
`λ → 0` and says nothing. Use LOO or a fold-based CV; the LOO variant makes this
cheap enough that there is no excuse.

**Extrapolation.** An RBF kernel decays to zero, so far from the data the
prediction returns to the mean. That is correct Bayesian behaviour for a GP
prior with zero mean, and it is often reported as a bug when the model is used
to extrapolate.

**Non-sparse models.** Every training point has a non-zero `α` (only exactly
duplicated points vanish). Prediction costs `O(n)` per point, no matter how
simple the function is; the SVM's sparsity is a hinge property, not a kernel
property.

**Classification by regression.** Squared loss on `±1` labels is not calibrated
and not robust: a badly mislabelled point contributes a residual that grows
linearly, where the hinge stops at the margin. Use `svm.py` when the margin
matters.

## Applications

Kernel ridge is the mean of a Gaussian process with a zero prior mean: fit
kernel ridge, and the GP posterior *mean* is what you get, with `λ = σ²/σ_f²`
the noise-to-signal ratio. That is why KRR is the standard way to strip the
linear algebra out of a GP story.

It is the right baseline whenever `n` is between a few hundred and a few
thousand and the geometry matters: spectral data, molecular kernels,
interpolation of simulation output, and as the readout layer of a two-stage
kernel model (`rbf_network.py` uses exactly that trick).

It is a trap when `n` is in the tens of thousands without an approximation, when
you need a probability rather than a mean, and when the loss is not squared —
in that last case the closed form disappears, and you are back in the saddle
point of the dual, which is the SVM's territory.

## References

- Kimeldorf, G. & Wahba, G. (1971). Some results on Tchebycheffian spline functions. *J. Math. Anal. Appl.* 33.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
- Golub, G., Heath, M. & Wahba, G. (1979). Generalized cross-validation as a method for choosing a good ridge parameter. *Technometrics* 21(2).
- Williams, C. & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. *NeurIPS*.
- Suykens, J. & Vandewalle, J. (1999). Least squares support vector machine classifiers. *Neural Processing Letters* 9(3).
- Rasmussen, C. & Williams, C. (2006). *Gaussian Processes for Machine Learning*, ch. 2 and 5. MIT Press.
