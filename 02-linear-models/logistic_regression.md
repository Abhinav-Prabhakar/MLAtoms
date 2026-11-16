# Logistic regression

`logistic_regression.py` · Linear models · Tier 1 · 8 variants

## Intuition

A linear model predicts a number; logistic regression turns that number into a
probability by squashing it. For two classes,

$$
P(y=1\mid x) = \sigma(x^\top\beta + \beta_0),
\qquad
\log\frac{P(y=1\mid x)}{P(y=0\mid x)} = x^\top\beta + \beta_0
$$

so the model is *linear in the log-odds*. That is the whole assumption: each feature
adds a constant amount to the log-odds, and the coefficients are interpretable as
log odds ratios. Fitting is maximum likelihood, which is a smooth convex problem —
the reason every solver in this file reaches the same answer.

The eight variants are not eight models. They are the three likelihoods people
actually need (binary, multi-class, ordered), three ways to move toward the optimum
(gradient, Newton/IRLS, L-BFGS), and two practical fixes (a sparsity penalty, and a
reweighting for imbalance). Keeping them in one file makes the shared likelihood
visible: the solvers are interchangeable because they all minimise the same
`(loss, grad)` closure.

## Variants

| Variant | Likelihood | Solver | Use when |
|:--|:--|:--|:--|
| `binomial` | Bernoulli | gradient descent with backtracking | reference / debugging; small `d` |
| `multinomial` | softmax over `K` classes | L-BFGS | more than two unordered classes |
| `ordinal` | cumulative logit | L-BFGS | classes are ordered (ratings, stages) |
| `irls` | Bernoulli | Newton / Fisher scoring | small or medium `d`; the fastest exact solver |
| `lbfgs` | Bernoulli | limited-memory BFGS | the default for large `d` |
| `elastic_net` | Bernoulli + `L1`/`L2` | proximal gradient (ISTA) | sparse coefficients, `p ≫ n` |
| `class_weight` | Bernoulli, reweighted | gradient descent | imbalance; equalise class influence |
| `separable_fix` | Bernoulli + a tiny `L2` | IRLS | separable data, where the MLE diverges |

The naming mixes likelihood and solver, which is honest about how people use it:
"logistic regression" in practice means "binomial with whatever solver my library
uses". The table is the map; `binomial` and `lbfgs` fit the *same* model and return
the same coefficients to `1e-6`, only with different iteration counts.

## Derivation

**Bernoulli likelihood.** With `z = xᵀβ + β₀` and `y ∈ {0,1}`:

$$
\ell(\beta) = -\sum_i \left[y_i z_i - \log(1+e^{z_i})\right]
= \sum_i \left[\mathrm{softplus}(z_i) - y_i z_i\right]
$$

The gradient and Hessian are

$$
\nabla = X^\top(\sigma(z) - y),
\qquad
\nabla^2 = X^\top \mathrm{diag}\!\big(\sigma(z)(1-\sigma(z))\big) X
$$

The Hessian is positive semidefinite, so the problem is convex; it is positive
definite unless the classes are separable, which is exactly when the Hessian
degenerates and the MLE runs off to infinity.

**IRLS = Newton with the Fisher information.** At each iterate, solve

$$
\left(X^\top W X + \lambda D\right)\Delta = X^\top(\sigma(z) - y),
\qquad W = \mathrm{diag}(p(1-p))
$$

and take a damped step `β ← β − tΔ`. Because `W` is recomputed each iteration, IRLS
is Newton's method on the log-likelihood, with quadratic convergence near the
optimum. `D` is the identity except at the intercept, which is never penalised.

**Multinomial.** With `W ∈ ℝ^{(d+1)×K}` and `z = XW`, the softmax NLL is

$$
\ell(W) = -\sum_i \sum_k Y_{ik}\log p_{ik},
\qquad
p_{ik} = \frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}},
\qquad
\nabla_W = X^\top(P - Y)
$$

The parameterisation is redundant: adding the same vector to every column of `W`
leaves `P` unchanged. The `L2` penalty on `W` (excluding the intercept row) picks a
representative and makes the Hessian definite; with `λ=0` the problem is still
convex but has a flat direction, and different solvers can return shifted `W` that
predict identically.

**Ordinal (cumulative logit).** For ordered classes `0…K−1`, model the cumulative
probabilities with shared slopes and `K−1` cut points `θ₀ < … < θ_{K−2}`:

$$
P(y \le k \mid x) = \sigma(\theta_k - x^\top\beta),
\qquad
P(y = k \mid x) = P(y \le k) - P(y \le k-1)
$$

with the conventions `P(y ≤ −1) = 0` and `P(y ≤ K−1) = 1`. To keep the thresholds
ordered by construction, the code parameterises

$$
\theta_0 = c,\qquad \theta_k = \theta_{k-1} + \mathrm{softplus}(\delta_k)
$$

so `δ ∈ ℝ^{K−2}` is unconstrained and `θ` is strictly increasing. This is the
proportional-odds model: one `β` for all cut points, so the effect of a feature is to
slide the whole distribution along the ordered scale.

**Elastic net.** The `L1` term is handled by proximal gradient:

$$
\beta \leftarrow S_{t\lambda_1}\!\left(\beta - t\nabla\ell(\beta)\right),
\qquad
t = 1/L,
\qquad
L \le \tfrac14 \max_i w_i\,\|X\|_2^2 / n + \lambda_2
$$

The `1/4` is the maximum of `p(1−p)`, and the intercept coordinate is copied through
without thresholding.

**Class weights.** For counts `n₀, n₁`, the weight for class `c` is `n/(2n_c)`, so
the two classes contribute equal total weight. The weights are then normalised to
mean 1, which keeps `λ` on the same scale.

**Separability.** If a hyperplane separates the classes, the likelihood has no
finite maximiser: scaling `β` up by `t` sends the loss to zero as `t → ∞`. A tiny
`L2` term makes the objective strictly convex and the solution finite; the demo's
unpenalised IRLS reaches `‖β‖ ≈ 1531` with loss `1.4e-08`, while a `1e-3` penalty
settles at `‖β‖ ≈ 8.8` with loss `0.13`.

## Code walkthrough

`sigmoid` is branchwise for a reason:

```python
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
```

For `z = −800`, `exp(-z)` overflows to `inf` and the first branch gives `0` correctly
only by accident of `1/inf`; for `z = +800` the second branch is the safe one.
Evaluating `1/(1+exp(-z))` on the whole array produces `inf/inf = nan` at one end, and
the `nan` survives into the gradient. The two masks are exact for all finite `z`, and
the same identity is used by `softplus`:

```python
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
```

`max(z,0) + log1p(exp(-|z|))` equals `log(1+e^z)` and only exponentiates a
non-positive number, so neither branch overflows.

`_design` appends the intercept **last**, and that convention is threaded through the
entire file:

```python
    if fit_intercept:
        return np.column_stack([X, np.ones(len(X))])
    return X
```

So `theta[:-1]` is always the coefficient vector and `theta[-1]` the intercept, and
every penalty is written as `theta[:-1]`:

```python
    loss += 0.5 * l2 * float(theta[:-1] @ theta[:-1])
    grad[:-1] += l2 * theta[:-1]
```

Putting the column first would work too, but then every `[:-1]` becomes `[1:]` and a
single missed slice penalises the intercept — which shrinks the model toward
`P(y=1) = 1/2` regardless of the data. The convention is the guard.

`gd` is backtracking gradient descent with an Armijo test:

```python
            xn = x + t * (-g)
            fn, gn = fun(xn)
            if fn <= f + 1e-4 * t * float(g @ (-g)) or t < 1e-14:
                break
            t *= 0.5
```

`g @ (-g) = −‖g‖²`, so the sufficient-decrease condition is `fn ≤ f − 1e-4·t‖g‖²`.
The `t < 1e-14` escape prevents an infinite loop when the gradient is zero but
floating-point noise makes `fn` slightly larger. After a step, `lr = min(2.0*t, 1.0)`
lets the step grow again — without it, one hard iteration would permanently shrink
the learning rate.

`lbfgs` is the two-loop recursion, and the reverse order is the whole algorithm:

```python
        for s, yv in zip(reversed(S), reversed(Y), strict=True):
            rho = 1.0 / float(yv @ s)
            a = rho * float(s @ q)
            alphas.append((a, rho))
            q -= a * yv
        ...
        for (a, rho), s, yv in zip(reversed(alphas), S, Y, strict=True):
            b = rho * float(yv @ r)
            r += s * (a - b)
```

The first loop walks the history backwards, storing `α` and subtracting `αy`; the
second walks it forwards, adding `s(α−β)`. `gamma = sᵀy / yᵀy` scales the initial
inverse-Hessian guess, which is what makes the first step have the right magnitude.
If `g @ d ≥ 0` the recursion produced an ascent direction (possible after a bad
curvature pair), and the code resets to steepest descent instead of taking it.

`_binomial_fun` is the likelihood, vectorised:

```python
    z = Xa @ theta
    loss = float(np.sum(sw * (softplus(z) - y * z))) / n
    grad = Xa.T @ (sw * (sigmoid(z) - y)) / n
```

`softplus(z) − y z` is `log(1+e^z) − yz`, the stable form of `−[y log p + (1−y)
log(1−p)]`; computing it from probabilities would take `log(0)` at saturation. The
gradient is `Xᵀ(σ(z) − y)/n`, with no `σ'(z)` factor — the same cancellation as the
`bce` loss in `00-primitives`. Sample weights multiply the per-row loss and gradient
before the mean, which is why `sw` is normalised to mean 1 upstream.

`_irls_binomial` builds the Fisher information and damps the Newton step:

```python
        s = sw * p * (1.0 - p)
        H = (Xa * s[:, None]).T @ Xa / n + np.diag(penalty)
        ...
            step = np.linalg.solve(H + 1e-10 * np.eye(d1), grad)
```

`(Xa * s[:,None]).T @ Xa` is `Xᵀ diag(s) X` without forming the diagonal matrix.
`penalty[-1] = 0.0` leaves the intercept unpenalised, and the `1e-10 I` is a numerical
floor for a component whose `p(1−p)` is zero — without it, `solve` raises for
separable data before the line search ever runs. The damping loop then halves `t`
until the loss does not increase; plain Newton can overshoot on flat likelihood
regions, and the guard costs at most 20 function evaluations.

`_ista_binomial` is the proximal method:

```python
    L = 0.25 * float(sw.max()) * float(np.linalg.norm(Xa, 2) ** 2) / n + l2
    t = 1.0 / max(L, 1e-12)
    ...
        z = theta - t * grad
        new = np.sign(z) * np.maximum(np.abs(z) - t * l1, 0.0)
        new[-1] = z[-1]                   # the intercept is never penalised
```

`0.25·max(sw)·‖X‖²/n` bounds the largest eigenvalue of the Hessian because
`p(1−p) ≤ 1/4`; `t = 1/L` is therefore a safe step. `new[-1] = z[-1]` is the line that
must not be forgotten: without it the intercept is soft-thresholded toward zero,
which is a different model (and makes the fit systematically under-confident).
`hist` is sampled every 25 iterations because storing the full 3000-entry loss curve
is pointless.

`_multinomial_fun` mirrors the binary case with a matrix of weights:

```python
        z = Xa @ W
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -float(np.sum(sw * (Y * logp).sum(axis=1))) / n
        G = Xa.T @ (sw[:, None] * (np.exp(logp) - Y)) / n
```

The max subtraction is mandatory before `exp`; without it a logit of 1000 gives
`inf` and a `nan` loss. `np.exp(logp)` is the softmax, recomputed instead of stored,
and `G = Xᵀ(P − Y)/n` is one matmul. `G.ravel()` matches the flattened parameter
vector that `lbfgs` expects, and `W = flat.reshape(d1, K)` is its inverse — the reshape
order is C-order, so `flat` must be laid out column by column, which is what
`flat.reshape` guarantees.

The ordinal likelihood is the most intricate function in the file:

```python
        theta = np.concatenate([[c], c + np.cumsum(sp)])
        ...
        p = sigmoid(theta[None, :] - z[:, None])       # (n, K-1)
        p_ext = np.concatenate([np.zeros((n, 1)), p, np.ones((n, 1))], axis=1)
        prob = p_ext[idx, y + 1] - p_ext[idx, y]
```

The sentinels `0` and `1` at the two ends encode `P(y ≤ −1) = 0` and `P(y ≤ K−1) = 1`
without a special case for the first and last class. `y + 1` shifts the label so that
`p_ext[:, y+1] = P(y ≤ y)` and `p_ext[:, y] = P(y ≤ y−1)`. The probability of class
`y` is their difference. If the sentinels are placed at the wrong end, the first and
last classes get probabilities that do not sum to one — the demo's `predict_proba`
rows would still sum to 1 for interior classes, so only a full check catches it.

The threshold gradient uses the chain rule through `softplus`:

```python
        tail = np.cumsum(g_t[::-1])[::-1]              # tail[j] = Σ_{k ≥ j} gθ_k
        grad_delta = tail[1:] * sigmoid(delta)         # d softplus/dδ = σ(δ)
```

`θ_k` depends on `δ_j` for every `j ≤ k`, so the derivative of `δ_j` is the sum of
the threshold gradients from `j` onward — a reverse cumulative sum. `tail[1:]` drops
the derivative of `θ₀ = c`, which does not depend on any `δ`. The `sigmoid(delta)` is
`d softplus/dδ`; omitting it still descends, but to a different parameterisation and
with a different `δ` scale.

`logistic_regression` normalises the class weights before fitting:

```python
            counts = np.array([np.sum(sw[yb == 0]), np.sum(sw[yb == 1])])
            per_class = n / (2.0 * counts)          # balanced: equal total weight
            sw = sw * np.where(yb == 1, per_class[1], per_class[0])
        sw = sw / sw.mean()
```

`n/(2n_c)` makes each class contribute `n/2` total weight. Dividing by `sw.mean()`
afterwards keeps the *average* weight at 1, so the effective sample size — and hence
the scale of `λ` and the gradient — is unchanged. Without that final normalisation,
the weighted loss is `n/2` times larger or smaller and `λ` silently means something
else.

`separable_fix` is not a different solver; it is a lower bound on the penalty:

```python
            theta, hist = _irls_binomial(Xa, yb, sw, n, max(l2, 1e-3),
                                         iters=min(iters, 100))
```

`max(l2, 1e-3)` means the fix cannot be accidentally disabled by passing `l2=0`. The
demo's contrast between `‖β‖ = 1531` and `8.8` is the same fit at two penalties: the
unpenalised loss is smaller, but only because the coefficients have run away.

`predict_proba` dispatches on the stored variant, not on the coefficient shape:

```python
    if fit["variant"] == "ordinal":
        ...
        return np.diff(p, axis=1)
    ...
    if theta.ndim == 1:
        p1 = sigmoid(X @ theta + float(b))
        return np.column_stack([1.0 - p1, p1])
```

Ordinal probabilities are differences of cumulative probabilities, so they are
non-negative by monotonicity and sum to one by telescoping — no clipping is needed.
The binary branch returns `[1−p, p]` in the **class order stored in `fit["classes"]`**,
which is why the fit records `classes` and why `classes[1]` is the positive class.
Multinomial uses a max-subtracted softmax, so the rows sum to one even when the
logits are large.

One parameterisation detail is worth stating explicitly: the ordinal design appends
the same ones column as the binary model, so `β` contains an intercept that is
redundant with the location of the thresholds (`θ₀`). With `l2 > 0` the penalty
identifies the split by shrinking that coefficient; with the default `l2 = 0` the
direction is flat and the optimiser returns whichever representative it reaches
from the zero start. Predictions are unaffected, but `coef[-1]` and `thresholds[0]`
should not be interpreted separately — only their combination is identified.

## Complexity

| Variant | Per iteration | Iterations in practice |
|:--|:--|:--|
| `binomial` / `class_weight` (GD) | `O(nd)` | hundreds to thousands |
| `irls` | `O(nd² + d³)` | 5–10 |
| `lbfgs` | `O(nd + m·d)` | 10–100 |
| `elastic_net` (ISTA) | `O(nd)` | hundreds to thousands |
| `multinomial` | `O(ndK + m·dK)` | 20–200 |
| `ordinal` | `O(ndK)` | 20–200 |

IRLS has the best iteration count and the worst per-iteration cost; L-BFGS is the
compromise that wins for `d` in the thousands. Memory is `O(d)` for GD/ISTA, `O(d²)`
for IRLS (the Hessian), and `O(m·d)` for L-BFGS with memory `m=10`. The multinomial
and ordinal models multiply both by `K`.

## Failure modes

1. **Complete separation.** The MLE does not exist; coefficients diverge and
   `‖β‖` grows with the iteration budget. Symptom: loss below `1e-6`, enormous
   coefficients, standard errors that explode. Fix: a small `L2` (`separable_fix`),
   or Firth's penalised likelihood.
2. **Penalising the intercept.** Shrinking `β₀` toward zero biases the predicted
   prevalence. Every penalty here slices `theta[:-1]`; custom code must do the same.
3. **Unscaled features.** The `L2`/`L1` penalties are in coefficient units, so a
   feature in thousands is penalised far less than one in fractions. Standardise, or
   use per-feature penalties.
4. **Multinomial shift invariance.** Without `L2`, `W` and `W + c1ᵀ` give identical
   probabilities. Comparing coefficients across runs requires fixing a reference
   class (drop one column) or accepting that only differences are identified.
5. **Ordinal thresholds out of order.** Fitting `θ` directly can produce
   `θ₀ > θ₁`, which makes some class probabilities negative. The `softplus`
   parameterisation prevents this by construction; if you fit `θ` freely, check
   `np.diff(θ) > 0` and reparameterise if not.
6. **Class weights changing the meaning of `λ`.** Reweighting without
   renormalising changes the effective sample size, so a penalty tuned before the
   reweight is no longer the same regulariser. Normalise the weights to mean 1.
7. **IRLS on a singular Hessian.** When `p(1−p) = 0` for every row, `W` is zero and
   the normal equations are singular. The `1e-10 I` floor keeps `solve` alive, but
   the resulting step is arbitrary; that is the signal that the data is separable.
8. **Reading probabilities as decisions.** `argmax` at 0.5 is a decision rule, not
   part of the model. With imbalance, tune the threshold on a validation set; the
   `class_weight` variant changes the model, not just the rule.

## Applications

- **Credit scoring and risk.** Coefficients are log odds ratios with confidence
  intervals; regulators accept the model because it is transparent. `irls` or
  `lbfgs` with `L2` is the standard.
- **Epidemiology.** Odds ratios adjusted for confounders — the multivariate
  logistic model is the workhorse, and `separable_fix` matters for small strata.
- **Clinical staging and surveys.** Ordered outcomes (stage I–IV, Likert scales)
  call for the ordinal variant, not a sequence of binary fits: it uses one slope per
  feature and respects the ordering.
- **Text classification.** Sparse bag-of-words with `elastic_net`; the `L1` term
  selects tokens, the `L2` term shares weight among correlated n-grams.
- **Baselines for neural models.** A one-layer softmax classifier is logistic
  regression; if a deep model cannot beat it, the features or the training are the
  problem.
- **Calibration.** Logistic regression on a single score is the standard
  post-hoc calibrator (Platt scaling); the `binomial` variant with one feature is
  exactly that.

## References

- Cox, *The Regression Analysis of Binary Sequences*, 1958. JRSS-B.
- McCullagh, *Regression Models for Ordinal Data*, 1980. JRSS-B (cumulative logit).
- Nelder & Wedderburn, *Generalized Linear Models*, 1972. JRSS-A.
- Green, *Iteratively Reweighted Least Squares for Maximum Likelihood Estimation*, 1984. JRSS-B.
- Liu & Nocedal, *On the Limited Memory BFGS Method for Large Scale Optimization*, 1989. Math. Programming.
- Firth, *Bias Reduction of Maximum Likelihood Estimates*, 1993. Biometrika (separation).
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, 2011 (the API conventions this file follows).
