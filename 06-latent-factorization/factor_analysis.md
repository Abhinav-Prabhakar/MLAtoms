# Factor analysis

`factor_analysis.py` · Latent structure · Tier 2 · 5 variants

## Intuition

Probabilistic PCA says every feature is measured with the same noise. Factor
analysis says each feature has its own noise level, and that is the entire
difference:

$$x = \Lambda f + \mu + \varepsilon, \qquad f \sim \mathcal{N}(0, \Phi), \qquad \varepsilon \sim \mathcal{N}(0, \Psi), \qquad (1)$$

$$x \sim \mathcal{N}(\mu, \Sigma), \qquad \Sigma = \Lambda \Phi \Lambda^\top + \Psi, \qquad \Psi = \mathrm{diag}(\psi_1, \ldots, \psi_d). \qquad (2)$$

PPCA is the case $\Phi = I$, $\Psi = \sigma^2 I$. Dropping the second constraint
is what makes the model honest about instruments: a microphone has one noise
floor, a thermometer another, and a model that averages them fits neither. The
loadings then have to arrange themselves around a *fixed* noise profile instead
of absorbing it.

The cost is one parameter per feature, which is exactly the parameter that is
hardest to estimate from a small sample.

| | PCA | PPCA | Factor analysis |
|:--|:--|:--|:--|
| parameters in the covariance | $dk - k(k-1)/2$ | $dk + 1 - k(k-1)/2$ | $dk + d - k(k-1)/2$ |
| noise | none | one $\sigma^2$ | one $\psi_j$ per feature |
| likelihood | no | yes | yes |
| a partial row | cannot project it | $E[x_{\text{miss}} \mid x_{\text{obs}}]$ | no (see below) |
| a rotation worth choosing | no | no | yes, $k \ge 3$ |

The last row is the second half of the story. For **any** orthogonal $R$,

$$\Lambda R R^\top \Lambda^\top = \Lambda \Lambda^\top, \qquad (3)$$

so the likelihood is completely flat along the $k(k-1)/2$ rotation directions.
With $k = 2$ that is a single angle and the fitted basis is arbitrary; with
$k = 3$ it is three angles. The model is identified in $\Sigma$ and not in
$\Lambda$, and a loading matrix is only as interpretable as the rotation you
chose. That is why two of the five variants are not estimators at all.

```
    the same Σ, three bases

      factor 2                        factor 2                       factor 2
         │   ●                           │                              │
         │  ●  ●                         │  ●  ●                        │ ●
         │ ●     ●                       │ ●     ●                      │  ●
         │●        ●                     │●        ●                    │   ●
         └──────────── factor 1          └──────────── factor 1         └──────── factor 1
       unrotated (ML, PCA)             varimax (orthogonal)          promax (oblique)
       every variable on               one factor per                one factor per
       every factor                    variable, Φ = I               variable, Φ ≠ I
```

## Variants

| Variant | Optimises | Algorithm | What it is for |
|:--|:--|:--|:--|
| `ml_fa` | the likelihood | Jöreskog fixed point on $\psi$ | the estimate, and the likelihood |
| `pca_fa` | nothing | one eigendecomposition | the fast moment estimator |
| `minres` | off-diagonal residual | iterated principal axes | the least-squares alternative |
| `varimax` | loadings simplicity | polar/SVD iterations | an orthogonal, readable basis |
| `promax` | fit to a simple target | Procrustes to $\lvert \Lambda \rvert^p$ | an oblique, readable basis |

**`ml_fa`** is the maximum-likelihood estimator, and it is a fixed point rather
than a gradient ascent: at fixed $\psi$ the optimal $\Lambda$ is available in
closed form from an eigendecomposition of the scaled sample covariance, and at
fixed $\Lambda$ the optimal $\psi$ is the diagonal residual. Alternating the two
exact steps gives a monotone iteration that converges in 30–40 sweeps on the
demo's data. Every claim in the demo is one of its stationarity conditions.

**`pca_fa`** is the principal-component method: take the top $k$ eigenvectors of
$S$, scale them by the square roots of the eigenvalues, and set the uniquenesses
to whatever the diagonal leaves over. It is one eigendecomposition, it never
iterates, and it is not a likelihood estimator — its $\psi$ can come out
negative and has to be clipped, which is why its fitted covariance is not an
optimum of anything. It exists because it is what people ran before computers,
and because it makes a clean baseline: on the demo's wide-noise data it loses
0.20 nats per row and leaves an off-diagonal residual ten times larger.

**`minres`** minimises the off-diagonal residual directly. Given $\psi$, the
best $\Lambda$ is the top $k$ eigenpairs of the reduced matrix $S -
\mathrm{diag}(\psi)$ — the sample covariance with its diagonal replaced by
communalities — and $\psi$ is then updated from the new diagonal residual. It is
the same iteration as `pca_fa` with the diagonal replaced at every step, and it
converges to the least-squares solution. The demo's measured result is worth
stating plainly: **`ml_fa` and `minres` are the same estimator in effect** on
this data. Their held-out likelihoods agree to $10^{-4}$ and their residuals to
0.015. They are different objectives with the same answer, which is the
asymptotic equivalence the theory predicts.

**`varimax`** fits `ml_fa` and then rotates. Kaiser's criterion is the variance
of the squared loadings within each column, summed over columns:

$$V = \sum_{j=1}^{k} \mathrm{Var}_i\left( (\Lambda R)_{ij}^2 \right), \qquad (4)$$

which is maximised exactly when the squared loadings within a column are as
spread out as possible — a few large, the rest near zero. That is the formal
version of "simple structure". It is a local method: it ascends a non-convex
quartic, and the demo verifies local optimality (200 random rotations, none
improves) rather than global optimality.

**`promax`** relaxes orthogonality. It fits the varimax solution to its own
fourth power — a Procrustes problem with a closed-form answer — and reads the
result as an oblique pattern $\Lambda = P$ together with the factor correlation
$\Phi$:

$$\Sigma = P \Phi P^\top + \Psi, \qquad P \Phi P^\top = \Lambda_v \Lambda_v^\top. \qquad (5)$$

The identity on the right is exact, and it is the reason an oblique rotation is
a reparameterisation rather than a new model: the same $\Sigma$, the same
likelihood, a different basis. What $\Phi$ is *not* is an estimate of the
population factor correlation. It is a diagnostic of how far the orthogonal
solution had to be pushed.

## Derivation

### 1. Why the diagonal is the whole difference

Both models match the off-diagonals with $\Lambda \Lambda^\top$, so any
difference between them is confined to the diagonal — and the diagonal is
exactly where a covariance has to stay positive definite. Write $h_j^2 =
\sum_i \lambda_{ij}^2$ for the communality of feature $j$, the variance it shares
with the others. Then

$$S_{jj} = h_j^2 + \psi_j, \qquad (6)$$

so $\psi_j$ is the part of feature $j$'s variance that no factor explains. In
PPCA all $d$ of those residuals are tied to one number; in factor analysis they
are free. That is why factor analysis is the model you reach for when the
features have visibly different noise, and why it needs $d$ more parameters.

### 2. Profiling $\Lambda$ out: the scaled spectrum

Fix $\psi$. Maximising the likelihood over $\Lambda$ is a regression problem in
disguise. Whiten the data by $\Psi^{-1/2}$, which makes the noise the identity:

$$\Psi^{-1/2} S \Psi^{-1/2} = G\, \Theta\, G^\top, \qquad \theta_1 \ge \theta_2 \ge \cdots \ge \theta_d. \qquad (7)$$

The transformed model is $\tilde S = \tilde\Lambda \tilde\Lambda^\top + I$, and
the top-$k$ eigenpairs of $\tilde S$ are the answer:

$$\tilde\Lambda = G_k\left(\Theta_k - I\right)^{1/2}, \qquad \Lambda = \Psi^{1/2} \tilde\Lambda. \qquad (8)$$

This is the same statement as PPCA's closed form with $\sigma^2 = 1$ in the
whitened coordinates. The $\theta_j - 1$ is the signal *above* the noise: a
direction whose scaled eigenvalue is below 1 has less variance than the noise
floor, so its loading is clipped to zero (the code's `np.maximum(..., 0.0)`).

Substituting (8) back gives the update for the noise,

$$\psi_j = S_{jj} - \sum_{i=1}^{k} \lambda_{ij}^2, \qquad (9)$$

which is (6) rearranged. Iterating (7)–(9) is Jöreskog's algorithm, and it is
what `_fit_ml` does.

### 3. The three stationarity conditions

The demo does not assert (9) — it could only recompute it. It asserts the deeper
conditions, each of which is a different expression:

**(i) $\Sigma$ is optimal over $\Lambda$ at fixed $\psi$.** This is where
$\mathrm{tr}(\Sigma^{-1}S) = d$ comes from, and it makes the profile likelihood
possible:

$$\ell(\psi) = -\frac{n}{2}\left[ d\log 2\pi + \sum_j \log \psi_j + \sum_{j \le k} \log \theta_j + d \right]. \qquad (10)$$

The determinant identity behind it is clean: $\lvert \Sigma \rvert =
\lvert \Psi \rvert \cdot \lvert I + \tilde\Lambda\tilde\Lambda^\top \rvert =
\prod_j \psi_j \cdot \prod_{j\le k}\theta_j$, because the whitened loadings have
eigenvalues $\theta_j - 1$ on $k$ directions and zero elsewhere.

**(ii) $\psi$ is optimal given $\Lambda$.** Differentiating (10) gives

$$\frac{\partial \ell}{\partial \psi_j} = \tfrac{1}{2}\left[ (\Sigma^{-1} S \Sigma^{-1})_{jj} - (\Sigma^{-1})_{jj} \right], \qquad (11)$$

and the demo measures its max-abs at $3.7 \times 10^{-8}$ over the eight
features. That is a genuine derivative check: `score_samples` forms $\Sigma^{-1}$
and never differentiates anything.

**(iii) the scaled spectrum keeps its weight.** From (8),
$\mathrm{tr}(\Lambda^\top \Psi^{-1} \Lambda) = \sum_{j\le k}(\theta_j - 1)$, so

$$\mathrm{tr}\!\left(\Psi^{-1/2} S \Psi^{-1/2}\right) = \sum_{j\le k}\theta_j + (d - k) \quad\Longrightarrow\quad \sum_{j>k}\theta_j = d - k. \qquad (12)$$

The tail eigenvalues are near 1 but not equal to it — the demo's are 1.081,
1.040, 1.012, 0.999, 0.942, 0.926 — and yet their sum is exactly $d - k =
6.000000$. That is a sharper check than "they are all around 1", and it is an
identity, not a tolerance.

### 4. The fitted covariance is the identity in units of the noise

Multiply (2) by $\Psi^{-1/2}$ on both sides. With $\Phi = I$,

$$\Psi^{-1/2}\Sigma\Psi^{-1/2} = I + \tilde\Lambda \tilde\Lambda^\top, \qquad (13)$$

whose eigenvalues are $\theta_1, \ldots, \theta_k$ on the factor directions and
exactly 1 on the other $d - k$. The demo prints them: $(10.7806, 6.4830)$ then
`1.000` six times. This is the geometric statement of the whole model — **the
factors are the only directions in which a feature's variance exceeds its
noise**, and everything else sits at the noise level by construction.

### 5. Rotations are reparameterisations

Because (3) holds, the likelihood cannot prefer $R$. So pick $R$ by a criterion
about the loadings. Varimax's is (4), and the standard algorithm is a sequence
of exact orthogonal Procrustes steps. Write $B = A R$ for the row-normalised
loadings at the current rotation. The derivative of the criterion with respect
to $R$, evaluated and then projected onto the orthogonal group, says the next
rotation is the orthogonal polar factor of

$$M = B^\top\left(B^{\circ 3} - B\,\mathrm{diag}\!\left(\overline{B^2}\right)\right), \qquad M = U \Sigma V^\top \;\Longrightarrow\; R \leftarrow R\,(UV^\top). \qquad (14)$$

$B^{\circ 3}$ is the elementwise cube and $\overline{B^2}$ the column means of
the squared loadings. Each step increases the criterion; because the criterion
is bounded above on a compact group, it converges. It can converge to a local
maximum, which is why the demo checks local and not global optimality.

The row normalisation matters for a reason that is easy to miss. Without it, a
feature with communality 0.9 dominates the variance being maximised and the
rotation is decided by the noisiest variables. Dividing each row by its
communality turns the criterion into a statement about *shares*, and since an
orthogonal rotation preserves row norms, the communalities are unchanged when
the rotation is applied to the unnormalised matrix — the demo asserts that to
$10^{-10}$.

Promax starts from the varimax solution $\Lambda_v$, chooses a target
$T = \lvert \Lambda_v \rvert^p$ with $p = 4$, and solves the least-squares
problem $\Lambda_v A \approx T$ with the constraint $\mathrm{diag}(A) = 1$.
With $A$ in hand,

$$P = \Lambda_v A^{-\top}, \qquad \Phi = A^\top A, \qquad P\Phi P^\top = \Lambda_v A^{-\top} A^\top A A^{-1} \Lambda_v^\top = \Lambda_v\Lambda_v^\top. \qquad (15)$$

$A$ has a unit diagonal by construction but $\Phi = A^\top A$ does not, so a
final rescaling makes $\Phi$ a correlation matrix:

$$P \leftarrow P\,\mathrm{diag}(s), \qquad \Phi \leftarrow \mathrm{diag}(1/s)\,\Phi\,\mathrm{diag}(1/s), \qquad s_j = \sqrt{\Phi_{jj}}, \qquad (16)$$

and the two rescalings cancel inside $P\Phi P^\top$. That cancellation is
exactly what a wrong transpose in (15) destroys: the first version of this file
computed $P = \Lambda A^{-1}$, and the identity failed at the third decimal
while everything else looked fine.

### 6. Why a pattern loading can exceed 1

In the orthogonal model a loading is a correlation-like quantity: $\lambda_{ij}^2
\le h_j^2 \le S_{jj}$ after standardisation. In the oblique model the pattern
entry $p_{ij}$ is a **regression weight** — the effect of factor $i$ on variable
$j$ holding the other factors fixed. Collinearity among the factors inflates
regression weights, so $p_{ij}$ can exceed 1 while the communality stays below
$S_{jj}$. The demo's promax pattern reaches 1.05 where the varimax solution
stopped at 0.93, and that is not a bug; it is the difference between a
coefficient and a correlation. The structural loading is $(P\Phi)_{ij}$, and
*that* is bounded.

### 7. The Heywood boundary

Maximise (10) and nothing stops $\psi_j$ from reaching 0. At $\psi_j = 0$ the
log-likelihood has a $\log 0$ in it, so the boundary is not a finite optimum —
the likelihood is unbounded as $\psi_j \to 0$ whenever the data can be fitted
exactly at that feature. That happens when one feature is (nearly) a linear
combination of the others, or when $k$ is too large for the sample. The estimate
is then said to be a **Heywood case**: a feature with no unique variance, an
improper solution, and $d$-dimensional standard errors that do not exist.

The code treats it as a boundary event rather than a numerical accident:

- the iteration stops as soon as a $\psi_j$ falls below $10^{-6} S_{jj}$,
- the offending entries are set to that floor so the arithmetic stays finite,
- and `heywood_` is set so a caller can find out.

The floor is relative to each feature's own variance, which makes the test
scale-free: a feature measured in millimetres and one in metres are compared on
the same footing.

## Code walkthrough

`FactorAnalysis(n_components, variant=..., max_iter=..., tol=..., power=...)`
with `fit`, `transform` ($E[f \mid x]$), `fit_transform`, `reconstruct`,
`score_samples`, and the attributes `loadings_` $(d,k)$, `noise_variance_`
($\psi$), `uniquenesses_` (the same array under its factor-analysis name),
`communalities_`, `covariance_`, `rotation_`, `phi_`, `heywood_`, `loglik_`,
`loglik_curve_`, `n_iter_`.

The orientation is `(d, k)` — feature by factor — because a loading *is* a
regression coefficient of a feature on a factor, and `pca.PCA.components_` uses
the transpose of that. Stating which one is meant is not pedantry: every
transpose bug in this file was an orientation bug.

### 1. The fixed point, in seven lines

```python
        for it in range(1, self.max_iter + 1):
            # Ψ^{-1/2} S Ψ^{-1/2}: the likelihood in coordinates where the
            # uniquenesses are 1, which is where the closed form lives
            C = S / np.sqrt(np.outer(psi, psi))
            theta, G = _eigh_desc(C)
            keep = np.maximum(theta[:k] - 1.0, 0.0)     # the signal above the noise
            L = np.sqrt(psi)[:, None] * (G[:, :k] * np.sqrt(keep)[None, :])
            psi_new = diag - np.sum(L * L, axis=1)
```

That is (7), (8), (9) in order. `np.outer(psi, psi)` broadcast through a square
root is the whitening matrix without forming it: dividing entry $(j,l)$ by
$\sqrt{\psi_j\psi_l}$ is exactly $\Psi^{-1/2} S \Psi^{-1/2}$. `_eigh_desc` sorts
descending because `np.linalg.eigh` does not, and `theta[:k] - 1.0` is the
whitened signal. The `np.maximum` is the clip that makes a factor with less
variance than the noise contribute nothing rather than a `nan`.

Note what is *not* built: no $d \times d$ inverse of $\Sigma$, no gradient, no
step size, no line search. Each sweep is one symmetric eigendecomposition and two
matrix products.

### 2. The profile likelihood, which never inverts $\Sigma$

```python
        d = S.shape[0]                                # S is a covariance: (d, d)
        psi = np.maximum(psi, 1e-12)
        theta, _ = _eigh_desc(S / np.sqrt(np.outer(psi, psi)))
        keep = np.maximum(theta[:self.n_components], 1.0)
        return float(-0.5 * (d * math.log(2.0 * math.pi) + np.sum(np.log(psi))
                             + np.sum(np.log(keep)) + d))
```

(10), per row of data. `keep = max(theta, 1)` rather than `theta - 1` because
this is $\log \lvert \Sigma \rvert$, where a clipped factor contributes
$\log 1 = 0$: the model does not claim that direction. `score_samples` computes
the same quantity by inverting $\Sigma$ and forming a quadratic form. The two
agree to $3 \times 10^{-10}$ relative on the demo's fit, which is the point of
having both: a wrong constant in either one shows up immediately.

There is a trap here worth spelling out, because the first version of this file
fell into it. (10) is valid **only at the fixed point**, since it uses
$\mathrm{tr}(\Sigma^{-1}S) = d$ to replace a quadratic form it never computes.
Evaluating it at a perturbed $\psi$ therefore returns a number that is not the
likelihood of anything, and a "perturb and check it does not improve" test built
on it is meaningless. The demo tests stationarity with (11) instead.

### 3. The two moment estimators

```python
        lam, V = _eigh_desc(S)
        L = V[:, :k] * np.sqrt(np.maximum(lam[:k], 0.0))
        self.loadings_ = L
        self.noise_variance_ = S.diagonal() - np.sum(L * L, axis=1)
```

This is `pca_fa`: the principal-component method, which is PPCA's closed form
with $\sigma^2 = 0$. It is one line of real work, and its $\psi$ is whatever the
diagonal has left over — possibly negative when $k$ is large relative to $d$ or
the data has redundant features, which is why `_finish` exists.

```python
            lam, V = _eigh_desc(S - np.diag(psi))
            L = V[:, :k] * np.sqrt(np.maximum(lam[:k], 0.0))
            psi_new = diag - np.sum(L * L, axis=1)
```

This is `minres`: the same two lines with $S$ replaced by the reduced matrix
$S - \mathrm{diag}(\psi)$. Given $\psi$, that is the exact least-squares
solution for $\Lambda$ against the off-diagonals — the diagonal of the reduced
matrix holds the communalities, which is precisely the part of $S$ that
$\Lambda\Lambda^\top$ is supposed to reproduce on the diagonal. Substituting
the diagonal back in and iterating is principal-axis factoring, whose fixed
point is the MINRES solution.

The difference between the two is one subtraction, and it is the difference
between a consistent estimator and a biased one: `pca_fa` never updates $\psi$,
so its loadings are computed as if the noise were zero on the diagonal, and the
bias grows with the noise-to-signal ratio. On the demo's 40× noise spread that
is 0.20 nats per row.

### 4. The model, assembled from its parts

```python
        return self.loadings_ @ self._phi() @ self.loadings_.T + np.diag(self.noise_variance_)
```

(2) with $\Phi = I$ for the orthogonal variants and the fitted $\Phi$ for
promax. `np.diag` rather than a diagonal matrix stored as an attribute keeps the
memory at $O(d^2 + d)$ and makes the structure of the model visible in one line:
low rank plus diagonal.

Everything downstream — `score_samples`, `covariance_`, the residual checks —
goes through this expression, so a wrong orientation in `loadings_` cannot hide.
It is also why promax's identity (15) is checkable: both variants must return
the same matrix to $10^{-10}$, and they do.

### 5. Factor scores are a ridge regression

```python
        L, psi = self.loadings_, self.noise_variance_
        # E[f|x] = (I + Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹ (x − μ)   -- Thomson's scores
        A = np.eye(L.shape[1]) + (L.T / psi) @ L
        return np.linalg.solve(A, ((X - self.mean_) / psi) @ L).T
```

The posterior of $f$ given $x$ under (1). Dividing by `psi` is $\Psi^{-1}$ on
both sides, and `np.linalg.solve` keeps the $k \times k$ system on the left
instead of forming its inverse. This is Thomson's score; the alternative,
Bartlett's, is the weighted least-squares estimate that ignores the prior on
$f$ and is unbiased but has a larger variance. Thomson's is the conditional
mean, which is the one that belongs in a probabilistic model.

Note that the scores are well defined even when $\Psi$ has a tiny entry. The
ridge term $I$ is the prior on $f$, so the system is always positive definite;
what is *not* well defined is the interpretation of a score on a factor that a
nearly noise-free feature determines exactly.

### 6. Varimax, and why the rotation accumulates

```python
        B = A @ R
        crit = float(np.sum(np.var(B * B, axis=0)))
        if crit - best <= tol * max(abs(crit), 1.0):
            break
        best = crit
        M = B.T @ (B ** 3 - B * (B * B).mean(axis=0))
        u, _, vt = np.linalg.svd(M)
        R = R @ (u @ vt)                     # accumulate: B was already rotated
```

(14). `B ** 3` is the elementwise cube, `(B * B).mean(axis=0)` is the per-column
mean of the squared loadings, and the SVD of the resulting $k \times k$ matrix
gives its orthogonal polar factor $UV^\top$.

The accumulation is the subtle line. $B$ already contains $R$, so the step from
$B$ is a *further* rotation; applying it to $R$ rather than replacing $R$ is
what makes the iteration converge instead of oscillate. Returning `R` as well as
the rotated loadings costs nothing and makes the whole thing reproducible: the
demo uses `rotation_` to check the criterion under arbitrary extra rotations,
and a later fit can be aligned to an earlier one with it.

### 7. Promax in nine lines

```python
    target = np.abs(L) ** power
    A = np.linalg.solve(L.T @ L, L.T @ target)          # least squares, L A ≈ target
    A = A @ np.diag(1.0 / np.diag(A))                   # unit diagonal, by column
    P = np.linalg.solve(A, L.T).T                       # P = L A^{-T}
    phi = A.T @ A
    # rescale so that Φ is a correlation matrix and P Φ Pᵀ is untouched:
    # P -> P diag(s) and Φ -> diag(1/s) Φ diag(1/s) cancel exactly
    s = np.sqrt(np.diag(phi))
    return P * s[None, :], phi / np.outer(s, s)
```

Line 1 is the target, line 2 the unconstrained least-squares solution of
$\Lambda A \approx T$, line 3 the column rescaling that gives $A$ a unit
diagonal — last because rescaling columns of $A$ is what moves the diagonal.
Line 4 is (15) for the pattern, and at this point $\Phi = A^\top A$ is a
covariance rather than a correlation. Lines 6–9 are (16): rescale so the
diagonal is 1, in the only way that leaves $P\Phi P^\top$ alone.

The transpose in `np.linalg.solve(A, L.T).T` is the whole reason this file had a
bug. `P = L A^{-\top}` wants to solve with $A$; writing
`np.linalg.solve(A.T, L.T).T` computes $L A^{-1}$, which is a different matrix
whenever $A$ is not symmetric, and the identity in (15) then fails at the third
decimal while the loadings still look plausible. The demo's assertion on
$P\Phi P^\top$ is what catches it.

### 8. Which variants rotate, and in what order

```python
        if self.variant == "varimax":
            self.loadings_, self.rotation_ = _varimax(self.loadings_)
            self.phi_ = None
        else:
            varimax_L, rot = _varimax(self.loadings_)
            self.loadings_, self.phi_ = _promax(varimax_L, self.power)
            self.rotation_ = rot
```

`varimax` and `promax` fit by maximum likelihood and then rotate, so
`variant="varimax"` is **not** a different estimator: its `covariance_` and
`score_samples` are identical to `ml_fa`'s to machine precision. That is the
design, and the demo asserts it. Only `loadings_` changes, by an orthogonal map
that leaves $\Sigma$ alone.

Promax rotates twice — varimax first, then the oblique Procrustes — which is why
`rotation_` records the *orthogonal* part. The oblique map is not a rotation of
the loadings in the same sense; it is the pair $(P, \Phi)$.

### 9. The clipping, and the flag

```python
        floor = 1e-6 * S.diagonal()             # per feature, so the test is scale-free
        self.heywood_ = self.heywood_ or bool(np.any(self.noise_variance_ <= floor))
        self.noise_variance_ = np.maximum(self.noise_variance_, floor)
```

Two lines: detect, then clip. The detection is against each feature's own
variance rather than a global threshold, so the test means the same thing after
a unit change. `or` rather than assignment because `_finish` is called after
`_fit_ml` has already had a chance to set the flag on the boundary.

Clipping is the smaller half of the job. The fitted $\Sigma$ is then positive
definite by construction and `score_samples` returns a finite number — which is
exactly the danger: a Heywood solution *looks* like a good fit and has a
slightly better likelihood than the honest one. `heywood_` is the only warning.

### 10. Two refusals that keep the API honest

```python
        if self.n_components > d - 1:
            k = self.n_components
            n_par = d * k + d - k * (k - 1) // 2     # loadings + uniquenesses − rotations
```

At $k = d$ the parameter count is $d^2 + d - d(d-1)/2 = (d^2 + 3d)/2$ against
$d(d+1)/2$ entries in $\Sigma$: exactly $d$ more parameters than the covariance
has numbers, so $\psi$ is not identified and no amount of data fixes it.
Refusing with a pointer to `probabilistic_pca` — which does have a $d$-component
model, with $\sigma^2 = 0$ — is more useful than returning something.

```python
        if np.isnan(X).any():
            raise ValueError("factor analysis needs a complete matrix; "
                             "use probabilistic_pca with variant='missing_data'")
```

Missing data is not a missing feature here: the M-step needs $\psi$ per feature
and the E-step needs a complete row. Silently dropping rows changes the sample
without saying so; silently imputing biases $\psi$ towards zero and manufactures
Heywood cases. The sibling atom has the correct machinery, so the error message
names it.

## Complexity

$n$ rows, $d$ features, $k$ factors, $T$ sweeps.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `ml_fa` | $O(nd^2 + Td^3)$ | $O(d^2)$ | one covariance, then $T$ eigendecompositions |
| `pca_fa` | $O(nd^2 + d^3)$ | $O(d^2)$ | one eigendecomposition, no iteration |
| `minres` | $O(nd^2 + Td^3)$ | $O(d^2)$ | same cost as `ml_fa` |
| `_varimax` | $O(T_r d k^2 + T_r k^3)$ | $O(dk)$ | SVD of a $k \times k$ matrix per step |
| `_promax` | $O(dk^2 + k^3)$ | $O(dk)$ | one Procrustes solve, no iteration |
| `transform` | $O(mdk + k^3)$ | $O(mk)$ | Thomson's scores for $m$ rows |
| `score_samples` | $O(d^3 + md^2)$ | $O(d^2)$ | inverts $\Sigma$ once |

Two things follow. First, factor analysis is linear in $n$ only through the
covariance: everything after that is $O(d^3)$ per sweep, so the method's cost is
set by $d$, not by the number of rows — the opposite of what a gradient method
would give, and the reason it is feasible on $d = 10^3$ response matrices but not
on $d = 10^5$. Second, the rotation costs are negligible compared with the
fitting, which is why `varimax` and `promax` are offered as variants rather than
as a separate post-processing atom.

The demo fits $d = 8$ in 35 sweeps and the whole six-section script runs in
0.13 s.

## Failure modes

**1. ML and MINRES are not distinguishable at this sample size.** The demo's
numbers, on a 40× noise spread: held-out likelihood $-11.8504$ for `ml_fa`
against $-11.8503$ for `minres`, off-diagonal residual 0.142 against 0.127. The
earlier version of this demo asserted `ml > minres` on a narrower noise profile
and it is false at the third digit — the two criteria are asymptotically
equivalent, and reporting a winner from a $10^{-4}$ gap is reporting noise.
What *is* distinguishable is `pca_fa`: 0.20 nats per row and a residual ten
times larger. Comparisons between the two proper estimators need either much
more data or a criterion-specific argument.

**2. Varimax cannot undo a correlated $\Phi$.** On the demo's oblique truth the
recovered simple structure is off by 0.24 at the largest entry — better than
the unrotated 0.61 share of mid-range loadings, but nowhere near exact. An
orthogonal rotation is a rotation within the same $k$-space, and a simple
structure generated by correlated factors is simply not reachable by one. If
the factors are really correlated, the honest options are promax, a targeted
rotation, or a confirmatory model; treating varimax output as the truth
systematically overstates the distinctness of the factors.

**3. Promax's $\Phi$ is a rotation artefact.** It is real as a
reparameterisation — the demo verifies $P \Phi P^\top = \Lambda_v \Lambda_v^\top$
to $10^{-10}$ — and it is not an estimate of the population factor correlation.
On the demo's truth $(\lvert \phi_{12} \rvert, \lvert \phi_{13} \rvert, \lvert
\phi_{23} \rvert) = (0.40, 0.20, 0.00)$, promax returns $(0.40, 0.23, 0.07)$
sorted: right in magnitude, wrong in correspondence, and with a small correlation
invented where the truth has none. The magnitudes come back because the
Procrustes target is built from the data; the correspondence does not, because
which factor is which is decided by the rotation. Report $\Phi$ as "how oblique
the simple structure is", never as "the correlation between factor 1 and factor
2".

**4. The rotation is local, and the basis is arbitrary.** Varimax ascends a
non-convex quartic. The demo checks local optimality — 200 random rotations of
the fitted basis, none improves the criterion — which is the strongest statement
available without an exhaustive search. Different starting rotations, different
implementations and different software can land on different local optima, and
factor order and sign are arbitrary in every one of them: two fits of the same
model are only comparable after a Procrustes match. The `_match` helper in the
demo exists for exactly this, and the atom exposes `rotation_` so a fit can be
reproduced exactly rather than approximately.

**5. Heywood cases are not errors, and that is the problem.** The demo builds
one deliberately: 11 features, one of which is an exact linear combination of two
others, fitted with $k = 4$. One of 12 uniquenesses is driven to the floor
($3.0 \times 10^{-6}$) and `heywood_` is set. No exception is raised, no
convergence warning fires, the likelihood is finite, and the model is not a
model. The tell is the flag; the sanity check is that $\psi_j$ should be
positive and not absurdly small relative to $S_{jj}$. In published work the
Heywood case is often a sign that $k$ is too large, that two variables are
collinear, or that the sample is too small for the parameter count.

**6. $k$ is bounded by the data, not by ambition.** The code refuses
$k \ge d$, but the real bound is Ledermann's: the parameter count must not
exceed the number of distinct covariance entries,

$$d k + d - \frac{k(k-1)}{2} \le \frac{d(d+1)}{2} \qquad\Longleftrightarrow\qquad (d-k)^2 \ge d + k, \qquad (17)$$

which for $d = 12$ allows at most $k = 7$, and for $d = 8$ at most $k = 4$.
Beyond that the model is not identified however well the iteration converges —
a $d = 8$, $k = 5$ fit will return eight numbers for $\psi$ and they will all be
meaningless. The code's guard only refuses $k \ge d$, so the counting bound is
not enforced; it is stated here so that a fit inside the guard but outside (17)
is not mistaken for an estimate.

**7. It needs complete data.** The E-step is a conditional mean over a complete
row and the M-step updates $\psi$ per feature; there is no observed-block
version of either. `probabilistic_pca`'s `missing_data` variant has the
machinery, at the cost of the per-feature noise. Silently mean-imputing instead
biases $\psi$ towards zero — imputation *creates* a feature with no unique
variance — and manufactures Heywood cases out of missingness.

**8. Scale changes the answer.** Loadings, uniquenesses and communalities are in
the units of the data. A feature in metres and the same feature in millimetres
produce the same $\Sigma$ up to the change of basis and visibly different
loadings, and the varimax criterion, which averages over features, is sensitive
to that. Standardise before fitting unless the units are meaningful, and if they
are meaningful, say so — the whole reason to prefer factor analysis to PPCA is
that the per-feature noise is interpretable.

## Applications

- **Psychometrics and educational testing.** The origin of the method: a battery
  of test items, a handful of latent abilities, and loadings you want to read.
  Varimax is the default rotation in that literature precisely because the goal
  is a table a human can interpret.
- **Finance.** Barra-style risk models are factor models of returns with
  correlated style factors; the oblique parameterisation, not the orthogonal
  one, is the one that matches how exposures actually co-move.
- **Genomics.** Expression modules: genes that load on the same factor are
  co-regulated, and the uniqueness is the measurement noise of that probe. The
  Heywood case is common with near-duplicate probes, and flagging it is part of
  standard QC.
- **Sensor fusion and metrology.** Per-instrument noise is the model's stated
  purpose. If you know the noise levels from calibration, fixing $\psi$ and
  solving for $\Lambda$ is a smaller, better-conditioned problem, and the
  likelihood in (10) gives a test of whether the calibrated values are
  consistent with the data.
- **Structural equation modelling.** Factor analysis is the measurement model
  half of SEM; the identification condition $k \le d - 1$, the rotation problem
  and the Heywood case all reappear there with different names.
- **Dimension reduction before regression.** Factor scores are a lower-variance
  summary than principal components when the features have unequal noise, which
  is why they are used as predictors in econometrics and chemometrics.
- **A testbed for identifiability.** The rotation non-identifiability is the
  cleanest example in statistics of a likelihood that is flat along a manifold of
  parameters — the same phenomenon as label switching in mixtures and sign
  flips in matrix factorisation, in its most tractable form.

Where it is a trap: reading an unrotated ML solution as interpretable (it is
not — the basis is whatever the eigendecomposition returned); reading a rotated
solution as a hypothesis test (it is a description, chosen after seeing the
data); and reading $\psi$ as measurement noise when it is really everything the
factors do not explain, including model misspecification.

## References

- **Spearman (1904),** *General intelligence, objectively determined and
  measured* — the two-factor model this all descends from.
- **Lawley & Maxwell (1971),** *Factor Analysis as a Statistical Method* — the
  likelihood derivation, the identification conditions, and the Heywood case,
  stated as statistics rather than as a computational recipe.
- **Jöreskog (1967),** *Some contributions to maximum likelihood factor
  analysis* — the fixed point on $\Psi^{-1/2} S \Psi^{-1/2}$ that `_fit_ml`
  implements, and the convergence argument.
- **Kaiser (1958),** *The varimax criterion for analytic rotation in factor
  analysis* — the criterion (4) and the iterative algorithm.
- **Hendrickson & White (1964),** *Promax: a quick method for rotation to
  oblique simple structure* — the power target and the Procrustes step.
- **Harman (1976),** *Modern Factor Analysis* — the principal-axis and MINRES
  iterations, and the classical notation this file follows.
- **Anderson & Rubin (1956),** *Statistical inference in factor analysis* — the
  identifiability conditions, including the bound (17), and the asymptotics that
  make the likelihood usable.
- **Bartlett (1937)** and **Thomson (1934)** — the two factor-score estimators,
  and why the conditional mean is the one with the smaller variance.
