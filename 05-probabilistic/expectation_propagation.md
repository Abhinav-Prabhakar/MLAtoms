# Expectation propagation

`expectation_propagation.py` · Probabilistic models · Tier 3 · 3 variants

## Intuition

[`gp_classification.py`](gp_classification.py) treated EP as a fact about
Gaussian processes. It is not a fact about Gaussian processes. It is a recipe
for any model that factorises into a Gaussian prior and a product of scalar
likelihoods:

$$p(\theta) \propto N(\theta;0,\Sigma_0)\prod_{i=1}^{n} f_i(a_i^\top\theta) \qquad (1)$$

Bayesian logistic regression, a Gaussian process with a robust likelihood, a
probit model with correlated coefficients, a factor graph whose edges are
observations — all of them are (1) with a different `a_i` and a different `f_i`.
EP is one algorithm that runs on all of them.

The recipe is: **replace each factor with a Gaussian, adjust the Gaussians until
the resulting posterior has the right marginals.** A single Gaussian site in
front of factor $i$,

$$t_i(a_i^\top\theta) = \tilde Z_i\,N(a_i^\top\theta;\,\nu_i,\,1/\tau_i) \qquad (2)$$

is a claim that factor $i$ carries a certain amount of information ($\tau_i$),
centred at a certain value ($\nu_i$), with a certain total mass ($\tilde Z_i$).
Multiply them all onto the prior and the posterior is a Gaussian, exactly, by
completing the square. The only question is what those numbers should be.

```
        Σ₀, {aᵢ, yᵢ}
             │
             ▼
      q(θ) = N(μ, Σ) ──────────────────────────────┐
             │                                     │
             │ divide out site i (a subtraction)   │
             ▼                                     │
        cavity q₋ᵢ ──── × fᵢ(yᵢ | aᵢᵀθ) ───► tilted q̂ᵢ
             │                                     │
             │ more uncertain than q,              │
             │ and pulled by the likelihood        │
             ▼                                     │
        match mean and variance of q̂ᵢ              │
             │                                     │
             ▼                                     │
        new site (τᵢ, νᵢ) ── damped ────────────────┘
```

The reason to match the **marginal** rather than the joint is that the marginal
is what you report. A posterior over forty weights is forty one-dimensional
summaries plus a covariance; getting the joint KL right while getting every
marginal wrong is not a win in practice, and that is exactly the failure mode of
the mean-field bound in [`variational_inference.py`](variational_inference.py).

The three Gaussian approximations in this chapter are distinguished by *what*
they match and *where*:

| | matches | cost | guarantee |
|:--|:--|:--|:--|
| Laplace | the joint log-density, at its mode | one Hessian | exact in the $n\to\infty$ limit |
| mean-field VI | a factorised $q$, by $\mathrm{KL}(q\lVert p)$ | an optimiser | a lower bound on $\log Z$ |
| EP | every marginal, by moments | a fixed-point loop | none |

There is no bound here, no Lyapunov function, and no promise that the loop
converges. What EP has instead is accuracy on the quantity people ask for, which
is why it is the default in production systems — TrueSkill's matchmaking model
is EP, and so is the layerwise Bayesian neural network in
[Hernández-Lobato & Adams (2015)](#references).

Two numbers from the demo set the scale. On Bayesian logistic regression with
40 points and two weights, EP's posterior mean is within 0.001 of an exact 2-D
grid integral and its covariance within 0.003, its evidence within 0.0047 nats,
and it gets there in 9 sweeps. And with a Gaussian likelihood it
reproduces GP regression to $2\times10^{-13}$ — the invariant that catches an
error in the site algebra, because in that case EP is *not* approximating
anything.

## Variants

| Variant | Factor | `a_i` | Scheduling | Reading |
|:--|:--|:--|:--|:--|
| `ep` | logistic | a data row $x_i$ | full steps | Bayesian logistic regression |
| `damped_ep` | logistic | a data row $x_i$ | $\delta = 0.7$ | the same fit, stabilised |
| `ep_for_gp` | Student-t | a basis vector $e_i$ | full steps | robust GP regression |

`ep` and `damped_ep` are the same model. Damping interpolates between the old
site and the new one, so it changes the path and not the destination — the demo
runs both on four nearly collinear points and they reach the same mean
$[1.246, 0.065]$ and the same $\log Z = -1.8650$. What damping costs is
iterations, and the cost is measurable:

| damping | 1.0 | 0.7 | 0.5 | 0.3 |
|:--|--:|--:|--:|--:|
| sweeps | 8 | 25 | 41 | 75 |

Roughly $1/\delta$, because a shorter step has to cover the same distance. On
this problem full steps happen to converge, so damping buys nothing but
robustness — the point of the variant is that on the problems where full steps
oscillate, it is the difference between an answer and a loop that never ends. It
is a safety net, not an accelerator, and a table that presented it as faster
would be lying.

The third variant is where EP stops being about regression coefficients:

```
parameter space    a_i = x_i    the factor sees one row of data
function space     a_i = e_i    the factor sees one coordinate of θ
```

With $a_i = e_i$ the prior covariance is the kernel matrix $K$ and $\theta$ is
the vector of latent function values at the training points. That is GP
regression, and `ep_for_gp` is the GP-classification machinery of
[`gp_classification.py`](gp_classification.py) with the likelihood swapped for a
Student-t. The code does not branch on which one it is; `A=None` means "one
factor per coordinate", and `A` being the data matrix means the other.

Two more likelihoods are reachable through the `likelihood=` kwarg and are not
variants, because they are controls rather than models:

- `gaussian` — the exactness test. With $f_i(y_i\mid\eta) = N(y_i;\eta,\sigma^2)$
  the site is the likelihood, EP is not an approximation, and the demo asserts
  the sites come out as $\tau_i = 1/\sigma^2$, $\nu_i = y_i$ to $10^{-10}$.
- `probit` — the closed forms. Probit sites have analytic tilted moments, so it
  is the one non-Gaussian likelihood where nothing in the loop is quadrature.

## Derivation

### 1. A Gaussian prior times Gaussian sites

Start from (1) with a Gaussian prior $N(\theta;0,\Sigma_0)$ and the $n$ sites (2).
Then

$$q(\theta) \propto N(\theta;0,\Sigma_0)\prod_{i=1}^{n}N(a_i^\top\theta;\nu_i,1/\tau_i) \qquad (3)$$

Each site is an exponential of a quadratic in the projection, so the product is
an exponential of a quadratic in $\theta$:

$$\Lambda = \Sigma_0^{-1}+\sum_i\tau_i a_i a_i^\top, \qquad \Sigma=\Lambda^{-1}, \qquad \mu=\Sigma b, \qquad b=\sum_i\tau_i\nu_i a_i \qquad (4)$$

$$q(\theta)=N(\theta;\mu,\Sigma) \qquad (5)$$

That is the whole of the easy half: given the sites, (4) is two matrix products
and one solve. Every hard part of EP is in choosing $\nu$ and $\tau$.

### 2. The cavity, by subtraction

To decide site $i$'s value, EP asks what the *rest of the model* believes about
$a_i^\top\theta$ — the posterior with site $i$ removed. In natural parameters
that is a subtraction of one rank-one precision and one linear term:

$$\Lambda_{-i}=\Lambda-\tau_i a_i a_i^\top, \qquad \Sigma_{-i}=\left(\Lambda_{-i}\right)^{-1} \qquad (6)$$

Sherman–Morrison on the rank-one removal gives everything in terms of the
current $q$, with no matrix built. With $s_i = a_i^\top\Sigma a_i$:

$$\Sigma_{-i}=\Sigma+\frac{\tau_i\,\Sigma a_i a_i^\top\Sigma}{1-\tau_i s_i}, \qquad s_{-i}^2=\frac{s_i}{1-\tau_i s_i} \qquad (7)$$

$$m_{-i}=s_{-i}^2\left(\frac{a_i^\top\Sigma b}{s_i}-\tau_i\nu_i\right) \qquad (8)$$

Equation (7) is why the code never forms a cavity covariance: the mean and the
variance of the projection are three dot products, and $1/(1-\tau_i s_i)$ is the
only scalar that can misbehave.

### 3. The tilted distribution, and the site update

Multiply the cavity by the true factor and renormalise:

$$\hat q_i(\eta)=\frac{1}{\hat Z_i}\,N(\eta;\,m_{-i},\,s_{-i}^2)\,f_i(y_i\mid\eta), \qquad \eta=a_i^\top\theta \qquad (9)$$

$q$ will be $q_{-i}$ times the site $t_i$, so requiring $q$'s projection to have
$\hat q_i$'s first two moments fixes the site uniquely:

$$\tau_i^{\text{new}}=\frac{1}{\hat\sigma_i^2}-\frac{1}{s_{-i}^2}, \qquad \nu_i^{\text{new}}=\frac{1}{\tau_i^{\text{new}}}\left(\frac{\hat\mu_i}{\hat\sigma_i^2}-\frac{m_{-i}}{s_{-i}^2}\right) \qquad (10)$$

The bracket in (10) is the site's **natural** linear parameter
$\eta_i=\tau_i\nu_i$; $\nu_i=\eta_i/\tau_i$ is its mean. Storing $\eta$ where
the code expects $\nu$ leaves the posterior covariance exactly right and the
mean wrong by a factor of $\tau_i$ — a bug that survives every variance check
and is caught only by a case with a closed form.

Putting the site back with a rank-one update is also Sherman–Morrison:

$$\Sigma\leftarrow\Sigma-\frac{\Delta\tau_i}{1+\Delta\tau_i s_i}\,\Sigma a_i a_i^\top, \qquad b\leftarrow b+\Delta(\tau_i\nu_i)\,a_i \qquad (11)$$

so a site update costs $O(d^2)$ and a sweep costs $O(nd^2)$, with no $O(d^3)$
anywhere after the initial Cholesky.

### 4. Damping

EP as described is a fixed-point iteration with no descent property. In practice
the update is shortened:

$$\tau_i\leftarrow(1-\delta)\tau_i+\delta\,\tau_i^{\text{new}}, \qquad \nu_i\leftarrow(1-\delta)\nu_i+\delta\,\nu_i^{\text{new}} \qquad (12)$$

With $\delta=1$ this is the textbook algorithm and it can oscillate; $\delta=0.7$
is the usual compromise. Because a damped step is a shorter step in the same
direction, the fixed points of the map do not change — only the route. That is a
claim worth asserting rather than believing, and the demo does.

### 5. The evidence

The point of EP is the posterior, but its normaliser is a usable by-product. The
unnormalised $q$ in (3) integrates to

$$Z_q=\left(\prod_i\tilde Z_i\right)\int N(\theta;0,\Sigma_0)\prod_iN(a_i^\top\theta;\nu_i,1/\tau_i)\,d\theta \qquad (13)$$

The product of sites is a Gaussian in $\theta$ with precision
$\sum_i\tau_i a_ia_i^\top=\Lambda-\Sigma_0^{-1}$ and linear term $b$, so

$$\int N(\theta;0,\Sigma_0)\,e^{-\frac12\theta^\top(\Lambda-\Sigma_0^{-1})\theta+b^\top\theta}\,d\theta=\frac{\lvert\Sigma\rvert^{1/2}}{\lvert\Sigma_0\rvert^{1/2}}\,e^{\frac12b^\top\Sigma b} \qquad (14)$$

and with the $(2\pi)^{-n/2}\prod_i\tau_i^{1/2}$ from the site normalisations:

$$\log Z_{\text{EP}}=\sum_i\log\tilde Z_i+\frac12\sum_i\log\tau_i-\frac{n}{2}\log2\pi-\frac12\log\lvert\Sigma_0\rvert+\frac12\log\lvert\Sigma\rvert+\frac12b^\top\Sigma b-\frac12\sum_i\tau_i\nu_i^2 \qquad (15)$$

Three constants in (15) are easy to get wrong and the demo found all three: the
$d$-dimensional $(2\pi)^{-d/2}$ of the prior and the $(2\pi)^{d/2}$ of the
Gaussian integral cancel, leaving one $-(n/2)\log2\pi$; $\lvert\Lambda\rvert$ is
$\lvert\Sigma\rvert^{-1}$, so the determinant term is $+\tfrac12\log\lvert\Sigma\rvert$;
and the $-\tfrac12\sum_i\tau_i\nu_i^2$ is not optional.

### 6. The amplitude, which moment matching leaves free

Equation (10) fixes the **shape** of the site and says nothing about its mass.
The condition that fills the gap is that the site must carry the same mass as
the factor it stands in for, measured under the cavity:

$$\int q_{-i}(\eta)\,t_i(\eta)\,d\eta=\tilde Z_i\,N(\nu_i;\,m_{-i},\,s_{-i}^2+1/\tau_i)\overset{!}{=}\hat Z_i \qquad (16)$$

$$\tilde Z_i=\frac{\hat Z_i}{N(\nu_i;\,m_{-i},\,s_{-i}^2+1/\tau_i)} \qquad (17)$$

Without (17) the evidence is not even exact in the Gaussian case, which is the
single cheapest test for it. With (17) it is, and the one-site case becomes exact
for a second reason: with $n=1$ the cavity is the prior, so $\hat Z_1=p(y_1)$
and the two Gaussian terms in (15) cancel by symmetry.

### 7. The tilted moments, for four factors

| $f_i(y\mid\eta)$ | $\hat\mu_i$ | $\hat\sigma_i^2$ | $\hat Z_i$ |
|:--|:--|:--|:--|
| $N(y;\eta,\sigma^2)$ | $\hat\sigma^2\left(\frac{m}{s^2}+\frac{y}{\sigma^2}\right)$ | $\left(\frac{1}{s^2}+\frac{1}{\sigma^2}\right)^{-1}$ | $N(y;m,s^2+\sigma^2)$ |
| $\Phi(y\eta)$ | $m+\frac{y s^2 r}{\sqrt{1+s^2}}$ | $s^2-\frac{s^4 r(z+r)}{1+s^2}$ | $\Phi(z)$ |
| $\sigma(y\eta)$ | quadrature | quadrature | quadrature |
| $t_\nu\!\left(\frac{y-\eta}{\sigma}\right)$ | quadrature | quadrature | quadrature |

with, for the probit row,

$$z=\frac{ym}{\sqrt{1+s^2}}, \qquad r=\frac{\varphi(z)}{\Phi(z)} \qquad (18)$$

and $m,s^2$ standing for the cavity mean and variance. The probit row is the
truncated-normal formulas: a Gaussian times a Gaussian CDF is not Gaussian, but
its first two moments are elementary, which is why `probit` has no quadrature in
it anywhere. The logistic and Student-t rows have no closed form in the mean
either, so the code integrates them with 40-point Gauss–Hermite.

The Gaussian row deserves a second look. There the tilted variance is
$(1/s^2+1/\sigma^2)^{-1}$, so the new site precision is

$$\tau^{\text{new}}=\frac{1}{\hat\sigma^2}-\frac{1}{s^2}=\frac{1}{\sigma^2} \qquad (19)$$

independent of the cavity, and the new site mean is $y$. **The site is the
likelihood, exactly.** EP recovers GP regression because in that case it is not
approximating: the first sweep writes down the right sites, the second confirms
they do not move, and the answer is exact. The demo checks the sites against
$1/\sigma^2$ to $10^{-10}$ and the posterior against the closed form to
$2\times10^{-13}$.

### 8. When the site is not a Gaussian

Everything above assumed $\hat q_i$ has a moment-matched Gaussian site, which
means assuming $\hat\sigma_i^2\le s_{-i}^2$ so that (10) gives $\tau_i\ge0$. A
log-concave factor guarantees that. A Student-t does not:

```
      cavity N(η; m, s²)            tilted ∝ cavity × t((y−η)/σ)
      ╭────╮                        ╭─╮        ╭─╮
     ╱      ╲                      ╱   ╲      ╱   ╲
    ────────────                  ────────────────────
        m                        m          y
```

When the residual is large the tilt has two bumps — one where the prior is, one
where the observation is — and the Gaussian that matches its first two moments is
*wider* than the cavity. Then $\tau_i<0$: the site is not a probability
distribution, only an algebraic object that widens $q$ along $a_i$. That is
precisely the mechanism that makes the fit robust: a negative site precision is
the algebra's way of saying "this point is not evidence, ignore it". But the
factorisation in (13) no longer describes a product of densities, so the evidence
is undefined. The code reports the posterior and sets `logZ_` to `nan`.

## Code walkthrough

The Gaussian tilt is (19) in code, and its lack of dependence on the cavity is
visible: `s2` appears only in the sum `1/s2 + 1/noise**2`.

```python
        if self.likelihood == "gaussian":
            # the one case with a closed form in every direction, and the
            # reason EP must reproduce GP regression exactly
            prec = 1.0 / s2 + 1.0 / self.noise**2
            mean = (m / s2 + y / self.noise**2) / prec
            lz = (-0.5 * LOG2PI - 0.5 * math.log(s2 + self.noise**2)
                  - 0.5 * (y - m) ** 2 / (s2 + self.noise**2))
            return mean, 1.0 / prec, lz
```

The probit tilt is (18) plus the truncated-normal moments. `r` is
$\varphi/\Phi$, computed as an exponential of a difference because both factors
underflow in the left tail.

```python
        if self.likelihood == "probit":
            z = y * m / math.sqrt(1.0 + s2)
            lz = math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
            r = math.exp(-0.5 * z * z - 0.5 * LOG2PI - lz)
            mean = m + y * s2 * r / math.sqrt(1.0 + s2)
            var = s2 - s2 * s2 * r * (z + r) / (1.0 + s2)
            return mean, var, lz
```

Everything else — logistic, Student-t — goes through one quadrature branch. The
two likelihoods share every line except the one that evaluates the factor, which
is why `likelihood="student_t"` versus `"logistic"` is a one-word change.

```python
        x, w = np.polynomial.hermite.hermgauss(40)
        eta = m + math.sqrt(2.0 * s2) * x
        if self.likelihood == "logistic":
            lik = sigmoid(y * eta)
        else:
            # Student-t on the residual: p(y|η) ∝ (1 + z²/ν)^(−(ν+1)/2)
            z = (y - eta) / self.noise
            lik = (1.0 + z * z / self.dof) ** (-0.5 * (self.dof + 1.0))
        den = float((w * lik).sum()) * INV_SQRT_PI
```

The sweep is equations (7)–(12) in order: the cavity variance and mean as two
scalars, the tilt, the site in natural form, damping, then a rank-one update of
$\Sigma$ instead of a new factorisation.

```python
                a = A[i]
                s = float(a @ Sigma @ a)                    # Var(aᵀθ) under q
                denom = 1.0 - tau[i] * s
                if denom <= 1e-12:                          # cavity would be improper
                    delta = max(delta, 1.0)
                    continue
                s2 = s / denom
                m = s2 * (float(a @ (Sigma @ b)) / s - tau[i] * nu[i])
                m_t, s2_t, _ = self._tilted(m, s2, y[i])
                tau_new = 1.0 / s2_t - 1.0 / s2
                eta_new = m_t / s2_t - m / s2
                nu_new = eta_new / tau_new

                d_ = self.damping
                tau_i = (1.0 - d_) * tau[i] + d_ * tau_new
                nu_i = (1.0 - d_) * nu[i] + d_ * nu_new
```

`nu_new` divides by `tau_new`: the subtraction produces the natural parameter
$\eta$, and the stored parameter is the mean. Dropping that division is the
invisible bug described under (10).

The rank-one update is (11), and the guard `scale > 1e-12` is the same
impropriety check as above — a negative $\Delta\tau$ is legal, a step that would
make $\Lambda$ singular is not.

```python
                # rank-one update of Σ in O(d²), Sherman–Morrison with a
                # negative semi-definite step; b = Σ τ_iν_i a_i tracks the mean
                dt = tau_i - tau[i]
                scale = 1.0 + dt * s
                if abs(dt) > 0.0 and scale > 1e-12:
                    Sig_a = Sigma @ a
                    Sigma = Sigma - (dt / scale) * np.outer(Sig_a, Sig_a)
                b = b + (tau_i * nu_i - tau[i] * nu[i]) * a
```

The evidence is (15) and (17). The amplitude loop rebuilds each cavity once more
— the sites have all moved since the sweep — and the final expression is one
line per term of (15).

```python
                v = s2 + 1.0 / tau[i]
                log_nu = -0.5 * LOG2PI - 0.5 * math.log(v) - 0.5 * (nu[i] - m) ** 2 / v
                z_hat += lz - log_nu
            # the `(2π)^{-d/2}` of the prior and the `(2π)^{d/2}` of the Gaussian
            # integral cancel, leaving one `−(n/2)log2π` for the n sites
            self.logZ_ = (z_hat - 0.5 * self.n * LOG2PI
                          + 0.5 * float(np.log(tau).sum())
                          + 0.5 * logdet - 0.5 * logdet0
                          + 0.5 * float(b @ self.mu_)
                          - 0.5 * float((tau * nu**2).sum()))
```

Negative precision is handled by refusing to compute the evidence at all, rather
than by returning a number that came from a factorisation that does not apply.

```python
        self.logZ_valid_ = bool(np.all(tau > 0.0))
        if self.logZ_valid_:
```

Prediction is a projection and a variance, and then either the link on the mean
or the link averaged under the Gaussian. As in
[`gp_classification.py`](gp_classification.py), the averaged version is the
honest one: the plug-in is systematically overconfident.

```python
        A_new = np.atleast_2d(np.asarray(A_new, dtype=float))
        mean = A_new @ self.mu_
        var = np.einsum("ij,jk,ik->i", A_new, self.Sigma_, A_new)
        return mean, np.maximum(var, 0.0)
```

The demo's reference is a grid integral of (1) — no EP anywhere in it. It is 241
points per axis on a two-dimensional parameter, which is cheap enough to be the
ground truth in every run.

```python
    g = np.linspace(lo, hi, n)
    W0, W1 = np.meshgrid(g, g, indexing="ij")
    W = np.stack([W0, W1], axis=-1)
    a = W @ X.T
    logp = -0.5 * np.sum(W**2, axis=-1) + np.sum(-np.logaddexp(0.0, -y * a), axis=-1)
    top = logp.max()
    w = np.exp(logp - top)
```

The damping claim is checked rather than assumed: four schedules, one fixed
point, four iteration counts.

```python
    sched = (1.0, 0.7, 0.5, 0.3)
    runs = [EP(np.eye(d), stiff, variant="ep", damping=g, n_iter=500).fit(y_s)
            for g in sched]
    assert all(r.converged_ for r in runs)
```

## Complexity

$n$ factors, $d$ parameters, $T$ sweeps, one factor revisited per step:

| Step | Time | Memory |
|:--|:--|:--|
| prior Cholesky | $O(d^3)$ | $O(d^2)$ |
| cavity + tilt | $O(d^2)$ per site | $O(d)$ |
| rank-one site update | $O(d^2)$ per site | $O(d^2)$ |
| one sweep | $O(nd^2)$ | $O(d^2)$ |
| full fit | $O(nd^2T)$ | $O(d^2+nd)$ |
| prediction on $m$ points | $O(md^2)$ | $O(md)$ |

In parameter space $d\ll n$ and the fit is linear in the data: 40 points with two
weights takes 9 sweeps in milliseconds. In function space $d=n$ and one sweep is
$O(n^3)$ — the same cost as one Newton step in
[`gp_classification.py`](gp_classification.py), and that file's 25 damped sweeps
are the same $Tn^3$ budget. Nothing here improves on the cubic; the sparse
variants (inducing points, `A` replaced by a low-rank factor) are what change it.

The one expensive operation that is *avoided* is the reason the update is written
as Sherman–Morrison rather than as a refactorisation: with $d=n$, a naive
implementation would do $n$ Cholesky factorisations per sweep instead of one
matrix–vector update per site.

## Failure modes

**Non-convergence, and multiple fixed points.** The loop is a fixed-point
iteration with no objective being decreased, so it can oscillate, and where it
converges it can converge to different answers from different orderings. The
demo's schedules land on the same fixed point but that is a property of the
problem, not a theorem. *Diagnostic:* `history_` — a `delta` that plateaus above
`tol` is oscillation, not slowness. Lower the damping before raising `n_iter`.

**A negative site precision.** A non-log-concave factor (Student-t, a mixture, a
bimodal observation model) can make $\hat\sigma_i^2>s_{-i}^2$, so $\tau_i<0$ and
the site is not a density. The posterior can still be a legitimate robust fit;
the evidence cannot be computed and the code returns `nan`. Treat any negative
$\tau_i$ as a statement about the factor, not about the data. *Diagnostic:*
`logZ_valid_`, and which sites are negative — a single one out of many is a
different situation from half of them.

**A vanishing cavity variance.** $s_{-i}^2=s_i/(1-\tau_i s_i)$ has $1-\tau_i s_i$
in a denominator, and for a site that has become near-deterministic the two terms
agree to within rounding. The code clamps and skips; too many skips means a
silently truncated fit. *Diagnostic:* count how often `denom <= 1e-12` fires by
instrumenting the loop; if it is not zero, the result is not EP.

**Sites that should be zero.** With a logistic or probit factor and a point that
is well explained, $\tau_i$ legitimately shrinks toward zero and the site becomes
uninformative. With a Gaussian likelihood it never does. The failure is not the
small precision but `1/tau[i]` in the amplitude step, which is why that step is
inside the `logZ_valid_` guard.

**The evidence is not a bound.** Laplace and EP both report a number for
$\log p(y)$, and neither is variational, so neither is guaranteed to sit below
the truth. On the demo's two-point problem Laplace is 0.014 nats off and EP
0.00001; on a misspecified model they can disagree by many nats in either
direction. Never read a 0.5-nat gap between two EP runs as evidence for
anything.

**Overconfidence under misspecification.** EP matches moments of $\hat q_i$, not
of $p$, and when the factor is wrong (an outlier, a mislabelled point, the wrong
kernel) the matched moments are the wrong target and the posterior variance comes
out too small. The Student-t variant exists because that is the usual fix:
change the factor until a Gaussian site can describe it.

**The invisible parameterisation bug.** (10) has one division that
moment-matching tests cannot see, because it does not affect the covariance. Any
EP implementation should be tested on the Gaussian likelihood first, where the
correct sites are known in closed form, and on a one-site problem, where EP is
exact by theorem.

**$n$ in the tens of thousands.** Function-space EP is $O(Tn^3)$ and 25 sweeps
make that unusable past a few thousand points. Sparse EP, like sparse GP
regression, replaces the kernel matrix with inducing points; the sites survive
the substitution unchanged.

## Applications

**Bayesian logistic and probit regression.** (1) with $a_i=x_i$ is the model,
and EP's posterior mean agrees with the exact grid to 0.001 on the demo's data.
The practical payoff over a Laplace fit is calibrated uncertainty on small data —
the same situation where [`laplace_approximation.py`](laplace_approximation.py)
is most overconfident.

**Gaussian process classification.** EP is the standard inference for GP
classification and the one that
[`gp_classification.py`](gp_classification.py) implements natively; this file
shows that it is a special case of a general loop rather than a GP-specific
derivation.

**Robust regression.** A Student-t likelihood on the residual gives a fit where
one bad point becomes a negative-precision site instead of a large residual — the
demo's contaminated series goes from RMSE 0.423 to 0.238 with one outlier moved.
Heavy-tailed observation models are the natural home of this file.

**Skill rating and matchmaking.** TrueSkill models a player as a Gaussian and one
game as a difference of two Gaussians plus a probit; the update is EP's
one-factor update, and a ladder of millions of players is a factor graph that
gets swept. This is EP as production software, not as statistics.

**Layerwise Bayesian neural networks.** Probabilistic backpropagation treats each
layer's weights as independent Gaussian sites and propagates moments forward and
backward; the backward pass is EP on the layer's output factor. It is the reason
EP is in the toolbox of people who would otherwise say "we use variational
inference".

**Decoding and factor graphs.** Turbo codes, LDPC decoding, and Gaussian belief
propagation are EP with different factors; the "approximate message passing"
family in signal processing is EP's fixed-point analysis applied to the same
updates. Minka's 2001 paper frames all of it as one divergence-minimisation
procedure.

**Where it is a trap.** Factors that are not unimodal or not close to Gaussian
(mixtures, hard constraints, combinatorial likelihoods), models where the
evidence is the deliverable and a bound is required, and anything where the site
parameterisation is not checked against a closed-form case first.

## References

- Minka (2001). *Expectation propagation for approximate Bayesian inference.*
  UAI. The original: EP as iterative moment matching, with the KL
  interpretation of the site update.
- Minka (2005). *Divergence measures and message passing.* MSR-TR-2005-173. The
  unified view — EP minimises a local $\mathrm{KL}(p\lVert q)$, VI minimises the
  reverse — and the reason EP's variances are larger.
- Opper & Winther (2000). *Gaussian processes for classification: mean-field
  algorithms.* Neural Computation 12, 2655. The mean-field counterpart, and the
  comparison the GP version of this file is built on.
- Seeger (2005). *Expectation propagation for exponential families.* Technical
  report. The amplitude condition (17) and the evidence (15), derived rather than
  stated.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.* MIT
  Press, Chapter 3.6. The function-space reading used by `ep_for_gp`, and the
  sparse EP extension.
- Bishop (2006). *Pattern Recognition and Machine Learning.* Springer, §10.7. The
  parameter-space reading, worked through on a Bayesian logistic regression.
- Herbrich, Minka & Graepel (2007). *TrueSkill: a Bayesian skill rating system.*
  NeurIPS. EP at scale, in a system that ran on live traffic.
- Hernández-Lobato & Adams (2015). *Probabilistic backpropagation for scalable
  learning of Bayesian neural networks.* ICML. Layerwise EP, and the argument
  that it beats mean-field VI on calibrated uncertainty.
- Kuss & Rasmussen (2005). *Assessing approximate inference for binary Gaussian
  process classification.* JMLR 6, 1679. What EP gets right and wrong, measured
  against MCMC rather than against other approximations.
- Jylänki, Vanhatalo & Vehtari (2011). *Robust Gaussian process regression with a
  Student-t likelihood.* JMLR 12, 3227. The Student-t variant, including the
  negative-site-precision behaviour the demo reproduces.
- Vehtari, Gelman, Sivula, Jylänki & Tran (2020). *Expectation propagation as a
  way of life.* JMLR 21. EP as a computational framework for leave-one-out and
  for parallelising Bayesian inference, with the convergence analysis this file
  only warns about.
