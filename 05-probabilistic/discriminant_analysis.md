# Discriminant analysis

`discriminant_analysis.py` · Probabilistic models · Tier 1 · 6 variants

## Intuition

Assume each class is a Gaussian. Then the Bayes-optimal classifier is a
formula, not an optimisation: plug in the class means and covariances and read
off the posterior. The entire family is a decision about **how many covariance
matrices to estimate** — one shared by every class, one per class, or something
in between.

$$\delta_c(x) \;=\; \log \pi_c \;-\; \tfrac12 \log\lvert \Sigma_c \rvert
\;-\; \tfrac12 (x-\mu_c)^\top \Sigma_c^{-1} (x-\mu_c) \qquad (1)$$

That is all of it. LDA is (1) with `Σ_c = Σ` for every class, which cancels
the quadratic term and leaves a hyperplane. QDA keeps a separate `Σ_c`, which
keeps a quadratic boundary. Shrinkage and RDA are the two ways of writing
"estimate fewer numbers than QDA but more than LDA", because the parameter
count is the whole story:

| Rule | Covariance parameters | Boundary |
|:--|--:|:--|
| LDA | `d(d+1)/2` | linear |
| QDA | `k·d(d+1)/2` | quadratic |
| shrinkage LDA | `d(d+1)/2` + one scalar | linear |
| RDA | between the two, by `α` | interpolates |

With `d = 100` and `k = 10`, QDA estimates 50 500 covariance entries against
5 050 for LDA. The extra 45 450 numbers are only worth having if every class
has many more than 100 examples.

The connection to the rest of the book: Gaussian naive Bayes estimates only
the `kd` diagonal entries of `Σ_c`; LDA estimates the full shared `Σ`; QDA
estimates every `Σ_c`. The three sit on one axis of "how much covariance to
believe".

## Variants

| Variant | `Σ` used for class `c` | Pick it when |
|:--|:--|:--|
| `lda` | one pooled `Σ` | `n_c` small, classes genuinely similar in shape |
| `qda` | `Σ_c` | `n_c \gg d`, classes differ in spread or correlation |
| `shrinkage_lda` | `(1−γ)Σ + γ ν I`, `ν = tr Σ / d` | `d` close to `n` and `Σ̂` is ill-conditioned |
| `rda` | `(1−γ)[α Σ_c + (1−α)Σ] + γ ν I` | you want to tune both dials on a validation set |
| `fisher_lda` | `Σ` (binary only) | you want the projection, or the classical threshold rule |
| `multiclass` | `Σ` | you want at most `k−1` coordinates, e.g. to plot or denoise |

`lda` and `fisher_lda` produce the same boundary — Fisher's derivation is a
different route to the same hyperplane. `multiclass` is Fisher generalised to
`k` classes, which is a *projection* rather than a scoring rule, and is the one
variant whose output is features as well as labels.

## Derivation

### From the Gaussian to the score

For `x \mid c \sim N(\mu_c, \Sigma_c)`,

$$\log p(x \mid c) = -\tfrac12 \log\lvert 2\pi\Sigma_c \rvert
- \tfrac12 (x-\mu_c)^\top \Sigma_c^{-1} (x-\mu_c), \qquad (2)$$

and adding `\log \pi_c` gives (1). Dividing by `p(x)` to get a posterior is a
normalisation; the classifier never needs it, only the argmax of (1).

### LDA: the quadratic term is common, so it cancels

With one `Σ`, expand the Mahalanobis term:

$$x^\top \Sigma^{-1} x \;-\; 2\,\mu_c^\top \Sigma^{-1} x
\;+\; \mu_c^\top \Sigma^{-1}\mu_c. \qquad (3)$$

The first term has no `c` in it, so it drops out of the comparison, as does
`-\tfrac12\log\lvert\Sigma\rvert`. What survives is linear in `x`:

$$\delta_c(x) \;=\; \underbrace{\mu_c^\top \Sigma^{-1} x}_{\text{weights}}
\;-\; \tfrac12 \mu_c^\top \Sigma^{-1}\mu_c \;+\; \log \pi_c. \qquad (4)$$

So LDA is a linear classifier with `w_c = \Sigma^{-1}\mu_c` and bias
`b_c = \log \pi_c - \tfrac12\mu_c^\top\Sigma^{-1}\mu_c`. Two classes, subtract:
the boundary is `w^\top(x - m) + \log(\pi_1/\pi_0) = 0` with `w = \Sigma^{-1}(\mu_1-\mu_0)`
and `m` the midpoint of the means. The demo checks exactly this identity
against (4), using the *fitted* parameters so the test is of the algebra, not
of the sample.

LDA is also invariant to any invertible linear map of the input. Apply `A` to
every point: the new class Gaussians are `N(A\mu_c, A\Sigma A^\top)`, which is
again a shared-covariance Gaussian model, and the decision boundary is the
image of the old one. This is why LDA is the canonical "linear" classifier:
it depends on the coordinates only through their equivalence class.

### Estimation

$$\hat\mu_c = \frac{1}{n_c}\sum_{i \in c} x_i, \qquad
\hat\Sigma = \frac{1}{n}\sum_{c}\sum_{i \in c}
(x_i - \hat\mu_c)(x_i - \hat\mu_c)^\top, \qquad (5)$$

which is the pooled MLE (`ddof = 0`), not the unbiased estimator — the factor
`n/(n-k)` is irrelevant once the same covariance is used for every class, and
standardising it away is what the code does.

The plug-in rule is not the Bayes rule: `(\hat\mu_c, \hat\Sigma)` are noisy, and
the noise enters through `\Sigma^{-1}`, which amplifies it along the *smallest*
eigen-directions of `Σ`. Those are precisely the directions LDA's
`w = \Sigma^{-1}\Delta` weights most heavily. When `d` is comparable to `n`,
this is the dominant error term and no amount of data per class fixes it —
only shrinkage does.

### Shrinkage LDA

Replace `\hat\Sigma` by a convex combination with the most boring covariance
matrix there is, `\nu I` with `\nu = \mathrm{tr}\,\hat\Sigma/d`:

$$\hat\Sigma(\gamma) = (1-\gamma)\hat\Sigma + \gamma\,\nu I,
\qquad \nu = \frac{1}{d}\sum_j \hat\Sigma_{jj}. \qquad (6)$$

The target is chosen to preserve the average variance, so the trace — hence the
overall scale — is untouched. `\gamma` trades bias for variance; the Ledoit–Wolf
optimum minimises `E\lVert\hat\Sigma(\gamma) - \Sigma\rVert_F^2`, whose
derivative is zero at

$$\gamma^\star = \frac{\hat\pi}{n\,\hat\delta}, \qquad
\hat\pi = \frac{1}{n}\sum_i \lVert x_ix_i^\top - S\rVert_F^2, \qquad
\hat\delta = \lVert S - \nu I\rVert_F^2, \qquad (7)$$

with `S = \tfrac1n\sum x_ix_i^\top` on centred rows. Both `\hat\pi` and
`\hat\delta` are plain sums: one `O(nd^2)` pass, no cross-validation, no
assumption about the truth. That is the whole appeal of the analytic estimate;
it is a bias-variance calculation, not a hyperparameter search.

### RDA: two dials instead of one

Friedman (1989) interpolates *between LDA and QDA* first, then shrinks:

$$\Sigma_c(\alpha,\gamma) = (1-\gamma)\bigl[\alpha\Sigma_c
+ (1-\alpha)\Sigma_{\text{pooled}}\bigr] + \gamma\,\nu_c I,
\qquad \nu_c = \frac{\mathrm{tr}\,\Sigma_c(\alpha,0)}{d}. \qquad (8)$$

`\alpha = 0` pools (LDA), `\alpha = 1` splits (QDA), and `\gamma` then moves the
result toward isotropy. The two corners of the `\alpha` axis are exact, which is
what the demo asserts to `1e-12` — if the interpolation is implemented correctly
it must reproduce both classics at the ends. RDA is the honest answer for
medium-sized problems: you have enough data to estimate *some* per-class shape
but not all of it.

### Fisher: the same hyperplane, derived as a projection

Fisher (1936) asked for the direction `w` that maximises the ratio of
between-class to within-class scatter of the projected data:

$$J(w) = \frac{\bigl(w^\top(\mu_1-\mu_0)\bigr)^2}{w^\top\Sigma_w w},
\qquad \Sigma_w = \Sigma_{\text{pooled}}. \qquad (9)$$

`J` is invariant to the scale of `w`, and its stationary point is

$$w \;\propto\; \Sigma_w^{-1}(\mu_1-\mu_0), \qquad (10)$$

the same direction as LDA's. The classical rule assigns class 1 when
`w^\top(x - m) > 0`. LDA adds the prior and assigns class 1 when
`w^\top(x-m) + \log(\pi_1/\pi_0) > 0` — the same hyperplane with the threshold
shifted so that the prior odds are honoured. The demo asserts the two produce
identical predictions.

### Multiclass: `k−1` directions from a generalised eigenproblem

With `k > 2` there is no single direction, but there is a canonical set: solve

$$\Sigma_b\, w = \lambda\, \Sigma_w\, w, \qquad
\Sigma_b = \sum_c \pi_c (\mu_c - \bar\mu)(\mu_c - \bar\mu)^\top, \qquad (11)$$

whose non-zero eigenvalues number exactly `k−1`, because `\Sigma_b` is a sum of
`k` rank-one terms centred on `\bar\mu`. A generalised symmetric problem is
solved by whitening: with `\Sigma_w = LL^\top`,

$$M = L^{-1}\Sigma_b L^{-\top}, \qquad M = V\Lambda V^\top, \qquad
W = L^{-\top}V. \qquad (12)$$

`M` is symmetric, so `\mathrm{eigh}` gives real eigenvalues and orthogonal
vectors; `W` then satisfies `W^\top\Sigma_w W = I`, i.e. it is
`\Sigma_w`-orthonormal. Project and classify by nearest centroid in that
space, `-\tfrac12\lVert W^\top x - W^\top\mu_c\rVert^2 + \log\pi_c`.

Discarding the directions with `\lambda = 0` is free *only* when `\Sigma_w` is
isotropic — then the discarded coordinates contribute the same distance to
every class and the projection is lossless for classification. With a general
`\Sigma_w` the discarded directions still carry within-class variance and the
rule is an approximation. That distinction is exactly what the demo pins: it
constructs `\Sigma_w = I` by whitening and then asserts `multiclass` reproduces
`lda` bit for bit, and shows that one component instead of two costs accuracy.

### The bridge to naive Bayes

Gaussian naive Bayes is (1) with `\Sigma_c = \mathrm{diag}(\sigma^2_{c1},
\ldots, \sigma^2_{cd})`. LDA is the same expression with the off-diagonal
entries believed. The intermediate — a block-diagonal or sparse `Σ` — is a
graphical model over the features, which is where this chapter meets
`pgm.py`.

## Code walkthrough

Everything is estimated once, in `fit`, before any variant dispatch. Note
`Xc.T @ Xc / counts[i]`: the biased MLE, and the per-class centred rows are
kept because the Ledoit–Wolf estimate needs the raw sample, not just its
scatter:

```python
        covs = np.empty((k, d, d))
        self._centred = []
        for i, c in enumerate(self.classes_):
            Xc = X[y == c] - self.means_[i]
            covs[i] = Xc.T @ Xc / counts[i]
            self._centred.append(Xc)

        pooled = np.einsum("i,ijk->jk", counts / counts.sum(), covs)
        self.class_covs_ = covs
        self.pooled_ = pooled
```

`np.einsum("i,ijk->jk", ...)` is the pooled covariance in one line: weight
each `Σ_c` by its class frequency and sum the matrices. Because the sum is
weighted by `n_c/n`, pooling is equivalent to centring every row on its own
class mean and taking one big scatter matrix.

RDA is two lines of algebra, and its structure is the reason it nests both
classics: at `α = 0` the first line returns `pooled` for every class, at
`α = 1` it returns `covs`, and the second line is a no-op at `γ = 0`:

```python
        elif self.variant == "rda":
            mixed = np.array([
                (1.0 - self.alpha) * pooled + self.alpha * C for C in covs
            ])
            self.cov_ = np.array([
                self._stabilise((1.0 - self.gamma) * M
                                + self.gamma * _isotropic(M))
                for M in mixed
            ])
            self.kind_ = "quadratic"
```

The auto shrinkage is the analytic estimate of equation (7). `q` is
`\lVert x_i\rVert^2`, so `q*q` is `\lVert x_ix_i^\top\rVert_F^2`, and
`np.einsum("ij,jk,ik->i", ...)` is `x_i^\top S x_i` — the only cross term in
the expansion. Written that way the whole estimate is three vectors, not an
`n×d×d` array:

```python
    n, d = Xc.shape
    S = Xc.T @ Xc / n
    mu = np.trace(S) / d
    sq = (S ** 2).sum()
    q = (Xc ** 2).sum(axis=1)
    r = np.einsum("ij,jk,ik->i", Xc, S, Xc)
    pi = float(np.mean(q * q - 2.0 * r + sq))
    delta = float(sq - d * mu * mu)
```

The linear branch is equation (4) as a matmul. `W = Σ^{-1}M^\top` stacks the
`k` weight vectors as columns, and the bias uses `(self.means_ * W.T).sum(1)`
to form `\mu_c^\top\Sigma^{-1}\mu_c` without a per-class loop:

```python
        if self.kind_ == "linear":
            # shared Σ: the quadratic term is common to every class, so the
            # whole decision collapses to a dot product with Σ⁻¹μ_c
            W = np.linalg.solve(self.cov_, self.means_.T)
            b = self.class_log_prior_ - 0.5 * (self.means_ * W.T).sum(axis=1)
            return X @ W + b
```

Quadratic scoring never forms `Σ^{-1}`. A Cholesky factor costs half the
flops of an inverse, and `solve(L, ...)` gives the Mahalanobis distance with
one triangular solve; `logdet` comes from the product of `L`'s diagonal, which
cannot underflow the way `np.linalg.det` can for `d = 1000`:

```python
    @staticmethod
    def _log_gauss(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Cholesky log-density: no explicit inverse, no `det` underflow."""
        L = np.linalg.cholesky(cov)
        sol = np.linalg.solve(L, (X - mu).T)
        maha = (sol ** 2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return -0.5 * (logdet + maha + X.shape[1] * np.log(2.0 * np.pi))
```

Fisher stores the direction and the threshold separately, because the two
things it is used for — project, or classify — want different halves. The
offset is `\log(\pi_1/\pi_0)`, the only place the prior enters:

```python
    def _fit_fisher(self) -> None:
        """Binary Fisher: one direction, threshold from the midpoint + priors."""
        delta = self.means_[1] - self.means_[0]
        self.direction_ = np.linalg.solve(self._stabilise(self.pooled_), delta)
        self.midpoint_ = 0.5 * (self.means_[0] + self.means_[1])
        self.offset_ = float(self.class_log_prior_[1] - self.class_log_prior_[0])
```

The multiclass eigenproblem is solved on the whitened matrix. The two nested
solves compute `L^{-1}\Sigma_b L^{-\top}`, and the explicit symmetrisation
`0.5 * (M + M.T)` is not cosmetic: `eigh` reads only one triangle, so a
floating-point asymmetry there silently changes the answer:

```python
        L = np.linalg.cholesky(within)
        M = np.linalg.solve(L, np.linalg.solve(L, between).T).T
        M = 0.5 * (M + M.T)
        vals, vecs = np.linalg.eigh(M)
        order = np.argsort(vals)[::-1]
        self.discriminants_ = vals[order]
```

`scalings_` is `L^{-\top}V`, which is what makes the projection
`\Sigma_w`-orthonormal; the centroids are centred on the same `mean_` that
`transform` subtracts, so the two stay consistent:

```python
        V = vecs[:, order[:m]]
        self.scalings_ = np.linalg.solve(L.T, V)      # Σ_w-orthonormal
        self.centroids_ = (self.means_ - mean) @ self.scalings_
```

`_stabilise` is the guard that makes the file usable at all when `d ≥ n`,
where `Σ̂` is singular by construction and `cholesky` would raise:

```python
    def _stabilise(self, cov: np.ndarray) -> np.ndarray:
        if self.reg == 0.0:
            return cov
        return cov + self.reg * _isotropic(cov)
```

## Complexity

`n` rows, `d` features, `k` classes, `n_c` rows in class `c`.

| Stage | LDA / Fisher | QDA | multiclass |
|:--|:--|:--|:--|
| `fit` | `O(nd^2 + d^3)` | `O(nd^2 + k d^3)` | `O(nd^2 + d^3 + kd^2)` |
| `predict` | `O(ndk)` | `O(nd^2k)` | `O(ndkm)` |
| memory | `O(d^2 + kd)` | `O(kd^2)` | `O(d^2 + dm)` |

The `d^3` is one Cholesky, done once at fit time. Prediction is where the
family separates: LDA is a `d×k` matmul per row, QDA is a triangular solve per
row per class, and that `d^2` factor is why QDA is rarely used above a few
hundred dimensions. When `n < d` the `nd^2` term is dominated by the `d^3`
factorisation and the model cannot be fitted without `reg > 0` or shrinkage.

## Failure modes

**`d > n` makes `Σ̂` singular.** The pooled scatter matrix has rank at most
`n − k`, so `cholesky` raises `LinAlgError` for `d > n − k`. This is not a
numerical annoyance but the actual statistics: there are more parameters than
data points. Diagnose with `np.linalg.matrix_rank(Σ̂) < d`; fix with `reg` or
`shrinkage_lda`.

**QDA overfits before it helps.** With `k = 10`, `d = 100` and 50 examples per
class, each `Σ_c` is estimated from 50 points in 100 dimensions and the
training accuracy is 1.0 while test accuracy is chance. The diagnostic is the
gap between the two, and the fix is `rda(alpha<1)` — which is exactly the
interpolation this family provides.

**Micro-collinearity, amplified by the inverse.** Two near-duplicate features
make `Σ` ill-conditioned; `Σ^{-1}` then places enormous weight on a direction
that is almost pure noise. `np.linalg.cond(Σ̂) > 10^6` is the tell, and it is
worse than it looks because LDA's weight vector points along the *smallest*
eigenvectors — the noise is not distributed evenly but concentrated in the
coefficients that matter most. Shrinkage is the cure, and `γ = 1` is the
limiting case where every feature is trusted equally in its own units.

**Non-Gaussian classes.** A class with two modes gets one `\mu_c` between them
and a covariance that spans the gap; LDA then draws a boundary through the
empty space, and QDA fits an ellipse that contains neither mode. Diagnose with
a per-class histogram or a normality test on the Mahalanobis distances —
`\chi^2_d` quantiles are the reference. The fix is a mixture model
(`gmm.py`), not a better covariance estimate.

## Applications

Face recognition is the historical home: Fisherfaces is LDA on pixel vectors,
and `multiclass` is the projection that made "eigenfaces vs fisherfaces"
a fair comparison. Medical decision rules, speaker identification and
sensor-fusion classifiers all use LDA as the calibrated, interpretable
baseline, because the coefficients `\Sigma^{-1}\mu_c` are readable as
per-feature evidence.

`multiclass` doubles as supervised dimensionality reduction: project to `k−1`
coordinates, then run anything else — a tree, an SVM, a plot. If the classes
are Gaussian this loses nothing; otherwise it is a cheap, decent
preprocessing step, and it is the standard "LDA before the classifier" recipe.

Where it is a trap: any `d > n` problem without shrinkage; any class that is
multimodal; and any dataset whose features are on wildly different scales — the
covariance is not scale-free, so `\Sigma^{-1}` will amplify whichever column
happens to be measured in the smallest units. Standardise first, always.

## References

- R. A. Fisher, *The use of multiple measurements in taxonomic problems*,
  Annals of Eugenics 7 (1936) — the original two-group discriminant.
- J. H. Friedman, *Regularized discriminant analysis*, JASA 84 (1989) — RDA,
  including the `(\alpha, \gamma)` parametrisation and the case for it.
- O. Ledoit and M. Wolf, *A well-conditioned estimator for large-dimensional
  covariance matrices*, Journal of Multivariate Analysis 88 (2004) — the
  analytic `\gamma^\star` of equation (7).
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical
  Learning*, §4.3 — LDA and QDA with the parameter counts and the worked
  examples.
- G. J. McLachlan, *Discriminant Analysis and Statistical Pattern Recognition*,
  Wiley 2004 — the reference monograph, including the plug-in error rates.
- P. J. Bickel and E. Levina, *Some theory for Fisher's linear discriminant
  function*, Bernoulli 10 (2004) — why LDA breaks when `d/n` does not go to
  zero, which is the modern reason shrinkage exists.
