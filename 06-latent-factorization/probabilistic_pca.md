# Probabilistic PCA

`probabilistic_pca.py` · Latent structure · Tier 2 · 4 variants

## Intuition

PCA answers one question — which $k$-plane is closest to the data — and refuses
to answer any other. There is no likelihood, so two different $k$ cannot be
compared; there is no noise parameter, so the distance from the plane has no
units; and a row with one missing coordinate cannot be projected at all.

Probabilistic PCA keeps the plane and adds a story that generated the data:

$$x = W z + \mu + \varepsilon, \qquad z \sim \mathcal{N}(0, I_k), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_d), \qquad (1)$$

which marginalises to a single Gaussian with a low-rank-plus-diagonal covariance:

$$x \sim \mathcal{N}(\mu, C), \qquad C = W W^\top + \sigma^2 I_d. \qquad (2)$$

`z` is the position on the plane, `W z` puts it in the observed space, and
`ε` is the part of the row the plane does not explain.

```
                        σ  ┌─ the fog: isotropic, thickness σ
                     ╭──╮   │
        · · ·   ·  · │··│ · │ ·        ·  scattered around a 2-plane
      ·    ·   ·   · ╰──╯   │
        ·   ·    ·  ·   ·   │  ·  ·
     ────────────────────────────────  the retained subspace (k = 2)
```

The quiet fact that makes this worth doing: **the maximum-likelihood `W` spans
exactly the PCA subspace.** Tipping and Bishop showed that

$$W_{\mathrm{ML}} = V_k \left(\Lambda_k - \sigma^2 I\right)^{1/2} R \qquad (3)$$

for any orthogonal $R$, where $V_k \Lambda_k V_k^\top$ is the top-$k$ part of the
sample covariance. So PPCA does not find better directions than PCA. It finds the
*same* directions and additionally reports how far off them the data sits.

That extra number buys three things.

| | PCA | PPCA |
|:--|:--|:--|
| rank selection | a scree plot, read by eye | a held-out likelihood |
| distance to the plane | unnormalised | $\sigma^2$, in the data's units |
| a partly observed row | cannot project it | $E[x_{\text{miss}} \mid x_{\text{obs}}]$ |
| new data | no generative story | sample from the fitted model |

The picture that makes (2) concrete is the eigenvalue spectrum. PPCA's covariance
has the top $k$ sample eigenvalues and then $d - k$ copies of $\sigma^2$:

```
    eigenvalue
      │  ●                                      sample covariance
      │   ●
      │    ●  ●                                 ●
      │         ●  ●  ●  ●  ●  ●  ●             ●  ●  ●  ●  ●  ●
      │  ─────────────────────────── σ²  ───────────────────────
      └────────────────────────────────────────────────────▶ index
         ╰── retained ──╯   ╰────── flattened tail ──────╯
```

So $\sigma^2$ is the **mean of the discarded eigenvalues**, and PPCA is the
statement "everything beyond $k$ looks like isotropic noise". When that statement
is false — the tail decays smoothly instead of flattening — the fit is still the
best rank-$k$ Gaussian, but $\sigma^2$ is an average of things that are not equal,
and the residual structure is real.

One consequence deserves its own sentence, because it is what makes the
posterior useful: the model's Gaussian fog means the reconstruction
$\mu + W E[z \mid x]$ is a **denoiser**, not a projection. Along the retained
directions the noise is almost entirely subtracted; along the discarded ones it
is kept. The demo measures the result: mean squared reconstruction error is
$(d - k)\sigma^2$, not $d\sigma^2$.

## Variants

| Variant | Fit | Algorithm | What it adds |
|:--|:--|:--|:--|
| `ppca_em` | observed likelihood | EM | the default, and the one that generalises |
| `ppca_closed` | same optimum | eigendecomposition | exact, one SVD, no iterations |
| `missing_data` | observed-block likelihood | EM with per-row conditioning | NaN-tolerant fitting |
| `mixture_ppca` | mixture likelihood | EM on responsibilities | several subspaces at once |

**`ppca_em`** is the workhorse. The E-step computes the posterior of `z` given
the row, the M-step solves for `W` and $\sigma^2$ in closed form given the
posterior second moments. Both steps are exact, so the likelihood is monotone by
construction, and the demo checks that over the whole curve. It converges in a
few dozen iterations from the closed-form initialisation and is the variant that
extends to missing data and to mixtures.

**`ppca_closed`** is (3) directly: one eigendecomposition of the sample
covariance, $\sigma^2$ = mean of the discarded eigenvalues, and a square root.
It is exact, deterministic, and needs no tolerances — which is why it is the
oracle in the demo: EM's answer has to match it in $C$ (not in $W$, which is only
identified up to a rotation) or one of the two is wrong.

**`missing_data`** conditions on the observed coordinates of each row instead of
the full row. Every row gets its own $k \times k$ system, the M-step for `W`
becomes one least-squares problem *per feature* over the rows where that feature
is present, and $\sigma^2$ normalises by observed cells rather than all cells.
With nothing missing it reduces exactly to `ppca_em` — the demo asserts that — so
the extra machinery costs nothing when it is not needed.

**`mixture_ppca`** replaces "the data is near one $k$-plane" with "the data is a
mixture of $m$ of them", each with its own centre, loading and noise:

$$p(x) = \sum_{c=1}^{m} \pi_c \, \mathcal{N}(x;\, \mu_c,\, W_c W_c^\top + \sigma_c^2 I). \qquad (4)$$

Responsibilities are computed with the same density as the single model, so the
E-step is one `_logpdf` call per component, and the M-step is the closed form on
a weighted scatter matrix plus one EM refinement of `W`. It is the natural
generalisation of PCA to a union of subspaces, and it is the entry point to
mixtures of factor analysers.

## Derivation

### The density without a $d \times d$ inverse

For $n$ rows and $d$ features, forming $C$ costs $O(d^2)$ memory and inverting it
$O(d^3)$. With $k \ll d$ both are avoidable by the matrix determinant lemma and
Woodbury:

$$\log \lvert C \rvert = (d - k) \log \sigma^2 + \log \lvert W^\top W + \sigma^2 I_k \rvert, \qquad (5)$$

$$x^\top C^{-1} x = \frac{1}{\sigma^2}\left( \lVert x \rVert^2 - (W^\top x)^\top (W^\top W + \sigma^2 I_k)^{-1} (W^\top x) \right). \qquad (6)$$

Both right-hand sides involve only $k \times k$ objects. This is not a
micro-optimisation: it is what lets PPCA be applied at $d = 10^4$ where $C$ alone
would be 800 MB, and it is the same identity that gives the posterior below.

### E-step: the posterior of the latent

With $M = W^\top W + \sigma^2 I_k$,

$$z \mid x \;\sim\; \mathcal{N}\!\left(M^{-1} W^\top (x - \mu),\; \sigma^2 M^{-1}\right). \qquad (7)$$

Every row of a complete matrix shares the same posterior covariance — it is
independent of $x$ — which is why EM here is so cheap: the covariance is computed
once per iteration, and only the means are per-row.

### M-step: the second moment, not the covariance

Maximising the expected complete-data log-likelihood over $W$ with the posterior
held fixed gives a weighted least-squares problem whose normal equations are

$$W = \left( \sum_i (x_i - \mu)\, E[z_i]^\top \right)\left( \sum_i E[z_i z_i^\top] \right)^{-1}, \qquad (8)$$

and the second factor is the second moment, not the covariance:

$$E[z_i z_i^\top] = E[z_i]E[z_i]^\top + \mathrm{Cov}(z_i). \qquad (9)$$

Dropping the $\mathrm{Cov}$ term is the classic mistake: it makes the fit
systematically over-confident in directions the data constrains weakly, because
it treats a posterior mean as if it were an observed value. The demo's fit would
still look plausible; it just would not agree with the closed form.

For the noise,

$$\sigma^2 = \frac{1}{nd}\left( \lVert X_c - E[Z] W^\top \rVert_F^2 + n\,\mathrm{tr}\!\left( \sigma^2 M^{-1} W^\top W \right) \right). \qquad (10)$$

The second term is the expected residual of the *latent*, and for a mixture that
term shows up weighted by the responsibilities.

> **The factor that must appear once.** $\mathrm{Cov}(z) = \sigma^2 M^{-1}$ already
> carries a $\sigma^2$. Multiplying the trace term by $\sigma^2$ a second time —
> which is easy to do while transcribing (10) — makes every M-step shrink
> $\sigma^2$ a bit too fast. EM still converges — to $\sigma^2 = 0.123$ against
> the closed form's $0.158$, 62 nats worse in likelihood — and every loss curve
> is still monotone, so nothing looks wrong until a quantity that does not depend
> on the path checks it. That was a real bug in this file, and the closed form is
> what caught it.

### The closed form

Stationarity of the log likelihood with respect to $W$ means $W$'s columns have
to be eigenvectors of the sample covariance in the retained subspace and the
residual has to be isotropic. Writing $S = V \Lambda V^\top$ with $\lambda_1 \ge
\cdots \ge \lambda_d$:

$$W = V_k \left(\Lambda_k - \sigma^2 I\right)^{1/2} R, \qquad \sigma^2 = \frac{1}{d - k}\sum_{j > k} \lambda_j, \qquad (11)$$

and the resulting covariance is the sample spectrum with its tail replaced by its
mean:

$$C = V_k \Lambda_k V_k^\top + \sigma^2 \left(I - V_k V_k^\top\right). \qquad (12)$$

Two limits are worth remembering. As $\sigma^2 \to 0$, (11) becomes
$W = V_k \Lambda_k^{1/2}$, the PCA projection, and the posterior mean becomes the
orthogonal projection — PPCA contains PCA. And when $k = d$ there is no tail, so
the formula divides by zero and $\sigma^2$ hits the boundary at 0; that is why
the code guards with `max(..., 1e-12)` rather than trusting the expression.

### Why the in-sample likelihood cannot choose $k$

Substituting (11) and (12) into the log likelihood gives the profile

$$\ell_k = -\frac{n}{2}\left[ d\log 2\pi + \sum_{j \le k} \log \lambda_j + (d - k)\log \hat\sigma_k^2 + d \right], \qquad \hat\sigma_k^2 = \frac{1}{d-k}\sum_{j>k}\lambda_j, \qquad (13)$$

which is non-decreasing in $k$ for the trivial reason that a bigger model contains
the smaller one. Setting $k = d$ gives $\ell = \infty$ at $\hat\sigma^2 = 0$.
So rank selection needs either held-out data or an explicit test; the classical
route is Anderson's likelihood-ratio test for the sphericity of the tail, whose
null distribution is a mixture of $\chi^2$ because $\sigma^2 \ge 0$ sits on a
boundary.

The demo uses held-out log likelihood and shows both sides of it. On 2000
training rows against 2000 held out, the curve peaks at the true $k = 2$
($-14.316$, $-8.627$, $-8.631$, $-8.632$ for $k = 1 \ldots 4$). On 400 training
rows it peaks at $k = 3$ by $10^{-3}$ — with that little data the sample tail is
not yet spherical enough for the correct model to win.

### Missing data changes the M-step, not just the E-step

For a row with observed set $O_i$, the model says

$$x_{O_i} \sim \mathcal{N}\!\left(\mu_{O_i},\; W_{O_i} W_{O_i}^\top + \sigma^2 I_{|O_i|}\right), \qquad (14)$$

with $W_{O_i}$ the rows of $W$ in $O_i$. The E-step is (7) with $W$ replaced by
$W_{O_i}$ and $M_i = W_{O_i}^\top W_{O_i} + \sigma^2 I_k$, which is one $k
\times k$ system per row. The M-step for `W` is where the structure changes:
each **row** $j$ of $W$ has its own normal equations, over the rows where feature
$j$ is present,

$$W_j = \left[ \sum_{i \in O_j} (x_{ij} - \mu_j) E[z_i]^\top \right] \left[ \sum_{i \in O_j} E[z_i z_i^\top] \right]^{-1}, \qquad (15)$$

because the inner product that produced the shared system no longer exists. And
$\sigma^2$ normalises by the number of **observed** cells: a missing cell
contributes $E[\epsilon^2] = \sigma^2$ to the objective, a term that does not
depend on $W$ and so does not move the optimum.

Filling the NaNs with column means and running ordinary PPCA is not the same
estimator, and the difference is not small. Mean imputation shrinks each column's
variance by the missing fraction and biases every correlation towards zero; on
this demo's data it recovers the hidden entries with RMSE 1.962 against 0.522 for
the conditional mean, and the model that comes out of it is not the one that
generated the data.

### A mixture: weighted versions of the same three equations

With responsibilities $r_{ic} \propto \pi_c\,\mathcal{N}(x_i; \mu_c, C_c)$,

$$W_c = \left[ \sum_i r_{ic} (x_i - \mu_c) E[z_i^{(c)}]^\top \right]\left[ \sum_i r_{ic} E[z_i^{(c)} z_i^{(c)\top}] \right]^{-1}, \qquad (16)$$

and likewise $\mu_c$ is the responsibility-weighted mean, $\pi_c$ the mean
responsibility, and $\sigma_c^2$ gets (10) with every sum weighted by $r_{ic}$.
The only genuinely new ingredient is the initialisation, and it is where mixtures
actually fail: two centres drawn uniformly at random can land in the same
cluster, both components then fit the same half of the data, and EM has no
symmetry-breaking mechanism to escape — it converges to "everything is 50/50"
and stops. The demo's earlier uniform initialisation did exactly that, at 53%
purity, on data that a quadratic discriminant separates perfectly.

## Code walkthrough

`PPCA(n_components, variant=..., n_mixtures=..., max_iter=..., tol=..., seed=...)`
with `fit`, `transform` ($E[z \mid x]$), `reconstruct` ($E[x \mid x_{\text{obs}}]$),
`fit_transform`, `score_samples`, `sample`, and `predict_proba` for the mixture.
`W_` is $(d, k)$ — latent to observed — and `components_` is its transpose, the
same orientation as `pca.PCA.components_`, so the two atoms can be read side by
side. `covariance_` is $C$; for the mixture the per-component tuples are
`means_`, `Ws_`, `sigma2s_` and `weights_`, and the single-model attributes stay
unset.

### 1. The density, without ever touching a $d \times d$ matrix

```python
    M = W.T @ W + sigma2 * np.eye(k)
    _, logdet_M = np.linalg.slogdet(M)
    logdet = (d - k) * math.log(sigma2) + logdet_M
    P = Xc @ W                                   # (n, k)
    quad = (np.sum(Xc * Xc, axis=1) - np.sum((P @ np.linalg.inv(M)) * P, axis=1)) / sigma2
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)
```

That is (5) and (6) with `P = Xc @ W` standing in for $W^\top x$ for every row at
once. `slogdet` rather than `log(det(...))` because $\lvert C \rvert$ for $d = 50$
and $\sigma^2 = 0.01$ is $10^{-100}$: the determinant underflows to exactly zero
in floating point and the likelihood becomes `-inf` for reasons that have nothing
to do with the fit.

Note what is *not* here: no $d \times d$ matrix, no Cholesky factor, no inverse of
$C$. The whole model is expressed through $k \times k$ objects, and that is the
form in which PPCA scales.

### 2. The EM loop

```python
            M = W.T @ W + sigma2 * np.eye(k)
            Minv = np.linalg.inv(M)
            Ez = Xc @ W @ Minv                          # (n, k)
            Covz = sigma2 * Minv                        # same for every row
            Ezz = Ez.T @ Ez + n * Covz                  # Σ_i E[z_i z_iᵀ]
            W_new = np.linalg.solve(Ezz, Ez.T @ Xc).T   # (d, k)
            resid = Xc - Ez @ W_new.T
            # Covz is already σ²M⁻¹, so the posterior spread enters once
            ssr = (float(np.sum(resid * resid))
                   + n * float(np.trace(Covz @ (W_new.T @ W_new))))
            sigma2_new = max(ssr / (n * d), 1e-12)
```

Line by line this is (7), (9), (8) and (10). Three details carry the derivation.
`Ezz = Ez.T @ Ez + n * Covz` is the sum of second moments — the `n *` is there
because the covariance is shared across rows but the sum is not. `np.linalg.solve`
with `Ez.T @ Xc` transposed afterwards solves for $W^\top$ rather than $W$, which
avoids forming an inverse and keeps the $(k,k)$ system on the left. And the
comment on `ssr` is load-bearing: `Covz` already equals $\sigma^2 M^{-1}$, so the
trace enters once. Writing `n * sigma2 * trace(...)` — the natural reading of
"posterior spread times $\sigma^2$" — silently changes the fixed point, and the
only symptom is that EM and the closed form disagree at the third digit.

### 3. The closed form, which is also the oracle

```python
        lam, V = _top_eigs(Xc.T @ Xc / n, k)
        sigma2 = float(lam[k:].mean()) if d > k else 0.0
        self.W_ = V[:, :k] * np.sqrt(np.maximum(lam[:k] - sigma2, 0.0))
        self.sigma2_ = max(sigma2, 1e-12)
```

`Xc.T @ Xc / n` is the **biased** covariance. Using $n - 1$ is the standard
sample-covariance convention and it is wrong here: the likelihood is a function of
the second moment about the fitted mean, so dividing by $n - 1$ makes the closed
form disagree with EM. On the demo's 600 rows that is a factor of 0.998 in
$\sigma^2$ — small, systematic, and exactly the kind of discrepancy that makes
someone go looking for a bug in EM.

`np.maximum(lam[:k] - sigma2, 0.0)` guards the case where a retained eigenvalue is
smaller than the noise level; that happens when $k$ is too large for the data, and
the unclipped value would put a `nan` in `W` via the square root.

### 4. Conditioning on the observed block, batched

```python
        M = sigma2 * np.eye(k) + np.einsum("dj,id,dk->ijk", W, m, W)
        Minv = np.linalg.inv(M)                        # (n, k, k)
        P = Xf @ W                                     # Wᵀ x_i, on observed coords
        Ez = np.einsum("ijk,ik->ij", Minv, P)
        Cov = sigma2 * Minv
```

With a per-row observed mask `m`, the system is $M_i = W^\top \mathrm{diag}(m_i)
W + \sigma^2 I$, and `einsum("dj,id,dk->ijk", W, m, W)` reads exactly that: sum
over features $d$, keep row $i$, land on the $j,k$ block. `np.linalg.inv` on an
$(n,k,k)$ stack is a batched inverse, so the whole E-step is three array
operations rather than $n$ Python iterations.

`Xf` is the centred data with the missing entries set to zero, which is what makes
`P = Xf @ W` equal $W_{O_i}^\top (x_{O_i} - \mu_{O_i})$ without any slicing: the
missing coordinates contribute exactly $0 \cdot W_j = 0$. The same `M` then gives
the observed log density

```python
        logdet = np.log(sigma2) * (nobs - k) + logdet_M
        quad = (np.sum(Xf * Xf, axis=1) - np.einsum("ij,ik,ijk->i", P, P, Minv)) / sigma2
        logp = -0.5 * (nobs * math.log(2.0 * math.pi) + logdet + quad)
```

which is (5) and (6) with $d$ replaced by the per-row observed count `nobs`. That
one substitution is the whole of "condition on the observed block": a row with 3
of 8 coordinates observed is a 3-dimensional Gaussian with the corresponding
$3 \times 3$ marginal, and its density must not be diluted by the 5 coordinates
nobody measured.

The batching is not cosmetic. The straightforward per-row loop took 10.9 s for
this demo's missing-data fit, 371 iterations of $n = 600$ small inverses; the
batched version takes 0.67 s for the same answer — 16 times faster, and the
reason the whole demo runs in under a second.

### 5. The missing-data M-step has one system per feature

```python
            for j in range(d):
                rows = mask[:, j]
                if not rows.any():
                    continue                     # never observed: W_j stays put
                Aj = second[rows].sum(axis=0)
                bj = ((X[rows, j] - mean[j])[:, None] * Ez[rows]).sum(axis=0)
                W_new[j] = np.linalg.solve(Aj, bj)
                resid = X[rows, j] - mean[j] - Ez[rows] @ W_new[j]
                ssr += float(resid @ resid) + float(np.sum(W_new[j] @ Cov[rows] @ W_new[j]))
```

This is (15). `second` is the per-row $E[z_i z_i^\top]$ from (9), and `rows` is
the set of rows where feature $j$ exists — the thing that cannot be shared, since
two features generally have different missing patterns. `Aj` is the sum of second
moments over *those* rows only; using all rows instead is the single easiest way
to get a confidently wrong `W` row for a feature with many gaps.

The last line accumulates both halves of the residual, exactly as in (10): the
squared error on observed cells and the posterior spread $W_j \mathrm{Cov}(z_i)
W_j^\top$. `if not rows.any(): continue` leaves an entirely unobserved feature's
loading at its initialisation rather than dividing by zero — honest, since there
is no information about it in the data.

Measured on 14% hidden entries: the fit recovers $\sigma^2 = 0.1587$ against
0.1583 for the complete matrix, and the true loading subspace to 0.026 rad, in 371
iterations.

### 6. The two conditionals, and the difference between them

```python
        Ez, _ = self._posterior(X, self.W_, self.sigma2_, self.mean_)
        return self.mean_ + Ez @ self.W_.T
```

`transform` returns `Ez` — the position on the plane, shape $(m, k)$.
`reconstruct` returns `mean_ + Ez @ W_.T` — the position *and* the plane, shape
$(m, d)$, with the missing coordinates filled in by their conditional mean.

That second line is the denoiser from the introduction. Because
$P = W(W^\top W + \sigma^2 I)^{-1}W^\top$ has eigenvalues $\lambda_j/(\lambda_j +
\sigma^2)$ near 1 on the retained directions and 0 elsewhere, the residual
operator $I - P$ kills the noise inside the subspace and passes the rest through
unchanged. The demo verifies both halves: the mean squared error is exactly
$\mathrm{tr}((I-P)^2 S)$ — a plumbing identity that pins the arithmetic — and it
equals $(d - k)\sigma^2 = 0.950$ to within a percent, which is the statistical
statement.

### 7. A mixture needs a symmetry-breaking start

```python
        centres = [X[rng.integers(n)]]
        for _ in range(m - 1):
            gap = np.min([((X - c) ** 2).sum(axis=1) for c in centres], axis=0)
            centres.append(X[rng.choice(n, p=gap / gap.sum())])
```

k-means++ seeding, followed by ten Lloyd iterations before the first EM step.
Each new centre is drawn with probability proportional to its squared distance
from the nearest existing centre, so the second one lands far away by
construction. Two uniform draws — the obvious initialisation — put both centres
in the same cluster about half the time when the clusters are unbalanced, and
when that happens the responsibilities come out near $1/m$ for every row and
nothing downstream can recover.

```python
                A = (Ez.T * resp[:, c]) @ Ez + Nc[c] * Cov
                Wc = np.linalg.solve(A, ((xc * resp[:, c][:, None]).T @ Ez).T).T
                resid = xc - Ez @ Wc.T
                ssr = (float(np.sum(resp[:, c] * np.sum(resid * resid, axis=1)))
                       + Nc[c] * float(np.trace(Cov @ (Wc.T @ Wc))))
```

(16) with the responsibility as a weight. `(Ez.T * resp[:, c]) @ Ez` is
$E[Z]^\top \mathrm{diag}(r_c) E[Z]$, and `Nc[c] * Cov` is `Cov` scaled by the
component's effective sample size — the mixture's stand-in for the `n * Covz` in
the single-model M-step. The `ssr` line is (10) with the per-row squared error
weighted by `resp[:, c]`, which is what makes $\sigma_c^2$ the noise *within* a
component rather than across the mixture.

### 8. Sampling, and the check it enables

```python
        Z = rng.standard_normal((n, self.n_components))
        return (self.mean_ + Z @ self.W_.T
                + math.sqrt(self.sigma2_) * rng.standard_normal((n, self.W_.shape[0])))
```

(1) literally: draw the latent, push it through the plane, add the fog. Having
this is what makes the model falsifiable in a way PCA is not — the demo draws
20000 rows and compares their sample covariance against $C$. That comparison has
to be made on the right scale: a covariance entry has Monte Carlo error
$\sqrt{(C_{ij}^2 + C_{ii}C_{jj})/n}$, which is 0.02 on the demo's 25.0 entry and
0.001 on its 0.16 ones. A single absolute tolerance either passes everything or
fails on the largest entry for no reason; normalising by
$\sqrt{C_{ii}C_{jj}}$ puts every entry on the same footing, and 20000 draws land
within 1.9%.

### 9. The refusals that keep the API honest

```python
        if has_nan and self.variant != "missing_data":
            raise ValueError(
                f"{self.variant} needs a complete matrix; variant='missing_data' is the "
                f"one that conditions on the observed block, and it treats NaN as missing")
```

A NaN reaching the closed form produces a `nan` covariance and a `nan` likelihood
that compares `False` against everything, so the model silently "converges" on the
first iteration and reports `loglik_ = nan`. Failing at `fit` with the name of the
variant that handles it is the difference between a five-minute fix and an
afternoon.

```python
        if self.variant == "mixture_ppca":
            raise NotImplementedError("a mixture has one covariance per component")
```

A mixture has no single $\mu$, $W$ or $\sigma^2$, so `covariance_`, `components_`,
`reconstruct` and `sample` raise for it rather than returning the largest
component's parameters — which would be a plausible-looking number answering a
question nobody asked. The per-component lists are there for anyone who wants
them.

## Complexity

$n$ rows, $d$ features, $k$ components, $T$ iterations, $m$ mixture components.

| Variant | Time | Per-iteration dominant term | Memory |
|:--|:--|:--|:--|
| `ppca_em` | $O(T(ndk + dk^2 + k^3))$ | `Xc @ W @ Minv` | $O(nd + dk)$ |
| `ppca_closed` | $O(nd\min(n,d) + dk^2)$ | one eigendecomposition | $O(nd + d^2)$ |
| `missing_data` | $O(T(ndk + nk^3 + dk^3))$ | the batched $(n,k,k)$ inverse | $O(nd + nk^2)$ |
| `mixture_ppca` | $O(Tm(ndk + dk^2 + k^3))$ | $m$ densities per E-step | $O(nd + mdk)$ |
| `_logpdf` | $O(ndk + k^3)$ | `Xc @ W` | $O(nk + k^2)$ |

Three notes. The closed form's one eigendecomposition beats $T$ EM iterations on
a complete matrix, which is why `ppca_closed` exists and why it is the fastest
route to the same optimum; EM's advantage is that it survives missing data and
mixtures, where no eigendecomposition is available. `missing_data` is more
expensive than its $O(ndk)$ suggests because the $(n,k,k)$ posterior covariance
tensor is materialised — at $k = 20$ and $n = 10^5$ that is 320 MB, so it is the
one variant where $k$ has to stay small or the loop has to be chunked. And the
cost of `mixture_ppca` is linear in $m$ with no sharing, because each component
has its own centre and its own eigenbasis.

## Failure modes

**1. `W` is not identified; `C` is.** For any orthogonal $R$, $(WR)(WR)^\top =
WW^\top$, so the loadings are only defined up to a rotation — infinite solutions
with identical likelihood. The demo asserts the invariance to $10^{-10}$ and
checks the *subspace* against PCA's ($2.1 \times 10^{-8}$ rad). Two consequences:
never compare `W` entries across fits, seeds or packages, because they will
differ; and if you need comparable loadings, align them (Procrustes) or
canonicalise them (varimax or similar, which is `factor_analysis`'s job).

**2. The in-sample likelihood cannot choose $k$.** It is non-decreasing by
construction, and $k = d$ sends it to infinity. The demo's numbers show the other
trap: held-out selection works but needs data. On 400 training rows the peak was
at $k = 3$ by $10^{-3}$ on data whose true rank is 2; on 2000 rows the correct
model wins by $0.007$. Reporting "the model chose $k$" from a small sample is
reporting noise.

**3. $\sigma^2$ collapses at the boundary.** When $k \ge d$ or $n \le d$ the fit
wants $\sigma^2 = 0$, and then $\log \sigma^2 \to -\infty$, `logdet` is `-inf`,
and every comparison against another model is meaningless. The guard in the code
(`max(..., 1e-12)`) keeps the arithmetic finite but does not make the model valid.
Check `k < min(n, d)` and treat a $\sigma^2$ at the floor as "this model is too
big for this data", not as a very good fit.

**4. Mean imputation is a different estimator.** Filling NaNs with column means
shrinks variance by the missing fraction, biases every correlation towards zero,
and destroys exactly the low-rank structure PPCA is looking for. The demo's
numbers: RMSE 1.962 for imputation against 0.522 for the conditional mean. The
same applies to "fit on complete rows only", which throws away all the partially
observed ones and biases the sample towards its easy part.

**5. Missing-not-at-random is not covered.** The E-step conditions on the
observed values, which is valid when the missingness is independent of the
unobserved value (MCAR, or MAR given the model). If a sensor drops out *because*
the reading was extreme, the conditional mean is biased and no amount of EM
fixes it — that is a model of the missingness process, which PPCA does not have.
All-missing rows fall back to the prior ($E[z] = 0$), which is the right answer
and also a silent one; count them before trusting the fit.

**6. Mixtures have local optima, label switching and $m$ more parameters.** The
demo's uniform initialisation stalled at 53% purity on data that is 100%
separable, and the fixed version reaches 100%. Component labels are arbitrary —
the demo matches them before scoring, because "component 0" means nothing across
runs. And $\sigma_c^2 \to 0$ on a component with few points is the classic
singularity: the likelihood diverges while the model becomes useless. A minimum
noise floor per component is not a hack; it is the constraint that makes the
likelihood bounded.

**7. The biased covariance is the right one.** $n - 1$ is the convention for
estimating a covariance and the wrong divisor for this likelihood: it is exactly
a factor of $n/(n-1)$ in $\sigma^2$, 0.17% at $n = 600$, and it shows up as a
small standing disagreement between EM and the closed form that is easy to
misdiagnose as a bug in either.

**8. Isotropic noise is a strong assumption.** PPCA says the noise is the same in
every direction. Factor analysis says it is $\Psi = \mathrm{diag}(\psi_1, \ldots,
\psi_d)$, which is one parameter per feature; if the features have different
units or different measurement quality, PPCA's single $\sigma^2$ is an average
that fits neither, and the loadings absorb the mismatch. The symptom is a
residual covariance that is visibly non-spherical at $k = d - 1$. PPCA is factor
analysis with $\Psi = \sigma^2 I$ — one noise parameter instead of $d$ — so
choosing it is a statement about the noise, not a simplification.

## Applications

- **Missing data imputation.** The conditional mean $E[x_{\text{miss}} \mid
  x_{\text{obs}}]$ is the reason `missing_data` exists; it is the low-rank
  imputer behind a lot of "fill the gaps in the sensor matrix" tooling, and it
  degrades gracefully as the missing fraction grows.
- **Collaborative filtering.** The user-item rating matrix is sparse and roughly
  low rank; PPCA is the probabilistic version of the truncated-SVD baseline, and
  its $\sigma^2$ is the "how noisy are ratings" parameter that the plain SVD
  baseline has to set by hand.
- **Denoising and compression.** Reconstruction is a denoiser whose error is
  exactly $(d - k)\sigma^2$ — image patches, spectra, sensor arrays. The rank $k$
  is a tunable knob with a likelihood attached, instead of a guess.
- **Anomaly detection.** `score_samples` is a density, so a row far from the
  subspace *and* far along it both score low; the Mahalanobis-style decomposition
  into "distance off the plane" and "distance along it" is standard in process
  monitoring.
- **Model comparison as a building block.** Because the model has a likelihood,
  it composes: mixtures of PPCAs, PPCA with a prior on $W$, PPCA inside a larger
  graphical model all need the marginal density that PCA cannot provide.
- **Nonlinear limits.** Replace $W z$ with $f(z)$ for a GP and the same marginal
  likelihood gives the Gaussian process latent variable model; PPCA is its
  linear special case, which is why the GP-LVM literature starts here.
- **Subspace identification.** In control, the state-space model $x_{t+1} = A
  x_t + w_t$, $y_t = C x_t + v_t$ has a stationary covariance of exactly the form
  (2) plus a second factor; PPCA and its two-block cousin CCA are the static
  versions of the subspace methods used to identify such systems.

Where PPCA is a trap: reading $\sigma^2$ as measurement noise when it is really
*everything the plane does not explain*, including the tail of a spectrum that was
never noise. And treating the loadings as interpretable factors — they are
identified only up to rotation, and the rotation that makes them interpretable is
a separate problem (`factor_analysis`'s varimax, or a sparse prior).

## References

- **Tipping & Bishop (1999),** *Probabilistic principal component analysis* —
  the model, the EM algorithm, and the theorem that the MLE spans the PCA
  subspace with $\sigma^2$ the mean of the discarded eigenvalues.
- **Roweis (1998),** *EM algorithms for PCA and SPCA* — the same EM derived from
  the projection side, with the posterior and the second-moment M-step.
- **Tipping & Bishop (1999),** *Mixtures of probabilistic principal component
  analysers* — the mixture in (4) and its EM, including the local-optima problem.
- **Ghahramani & Beal (2000),** *Variational inference for Bayesian mixtures of
  factor analysers* — what to do when $m$ and $k$ both have to be chosen, and the
  variational bound that makes the mixture identifiable in practice.
- **Ilin & Raiko (2010),** *Practical approaches to principal component analysis
  in the presence of missing values* — the per-feature M-step (15), the
  normalisation by observed cells, and the stabilisations that make missing-data
  EM converge in practice.
- **Anderson (1963),** *Asymptotic theory for principal component analysis* — the
  likelihood-ratio test for the rank of a covariance matrix, i.e. the classical
  answer to "how many components".
- **Bishop (2006),** *Pattern Recognition and Machine Learning*, §12.2 — the
  textbook derivation, including the Woodbury identities used for the density.
- **Lawrence (2005),** *Probabilistic non-linear principal component analysis
  with Gaussian process latent variable models* — the nonlinear generalisation
  that keeps PPCA's marginal likelihood and replaces $W z$ with a GP.
