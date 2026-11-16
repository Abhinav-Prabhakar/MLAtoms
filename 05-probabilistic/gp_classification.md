# Gaussian process classification

`gp_classification.py` · Probabilistic models · Tier 3 · 5 variants

## Intuition

Regression with a Gaussian process is a matrix solve because a Gaussian prior
times a Gaussian likelihood is Gaussian. `gaussian_process.py` gets a
closed-form posterior for free, and the whole atom is one Cholesky factorisation.

Classification breaks that, and it breaks it completely. The likelihood is
$\sigma(f_i)$ or $\Phi(f_i)$ — bounded, and not exponential of a quadratic — so
the posterior over the $n$ latent values is an $n$-dimensional distribution with
no closed form, and the predictive probability is an integral over it. Nothing
about the prior's conjugacy survives.

What survives is the shape. The posterior over $f$ is *approximately* Gaussian:
it is a Gaussian times a product of sigmoids, and a sigmoid is close to a
Gaussian in log space over the region where the data pin the function down.

```
   prior p(f)              likelihood             posterior p(f|y)
   ─────────               ──────────             ─────────────────
     N(0, K)               ∏ σ(yᵢfᵢ)              Gaussian × sigmoids
   ╭─────╮                 ╭──╮                    ╭──╮
  ╱       ╲               ╱    ╲                  ╱    ╲___
 ───────────             ────────                ──────────
   symmetric               saturates              skewed, heavier
   exactly Gaussian        at both ends           on the uncertain side
```

So both approximations here return a Gaussian, and they differ in *how* they
choose it:

- **Laplace** matches the log-density quadratically at its maximum. One point,
  second-order there, nothing else. Cheap, local, and exact in the limit of
  lots of data — the Bernstein–von Mises argument from
  [`laplace_approximation.py`](laplace_approximation.py) applies unchanged.
- **EP** matches the *marginal* mean and variance of every latent, by
  iteratively replacing the likelihood with one Gaussian factor per data point.
  Global in the sense that every site's update feels every other site, but it
  optimises nothing and carries no guarantee.

Neither is uniformly better, and the demo is built to show that rather than
assert otherwise. On two data points, against a 121×121 grid:

| | posterior mean error | evidence error |
|:--|--:|--:|
| Laplace | $4.5\times10^{-2}$ | $1.4\times10^{-2}$ nats |
| EP | $3.8\times10^{-6}$ | $1.2\times10^{-5}$ nats |

EP wins decisively there. On a 60-point problem the two produce predictions
that differ by at most 0.021 in probability and score identically on held-out
data — which is the usual situation, and the reason nobody rewrites a working
Laplace-based model.

The three other variants are the parts people get wrong. `probit` and `logistic`
are the two likelihoods; the probit one has closed-form derivatives, closed-form
EP moments and a closed-form predictive, while the logistic one needs
quadrature in both places. `one_vs_rest` is what you actually have to do for
more than two classes, because neither approximation generalises to $k$ classes
without a new derivation.

## Variants

| Variant | Likelihood | Inference | Predictive | Use |
|:--|:--|:--|:--|:--|
| `laplace` | logistic | Laplace | plug-in $\sigma(\mu)$ | the textbook baseline; fast, slightly overconfident |
| `logistic` | logistic | Laplace | integrate over $q(f_*)$ | the same fit, honest predictive |
| `probit` | probit | EP | analytic $\Phi(\mu/\sqrt{1+v})$ | closed forms everywhere; the classic pairing |
| `expectation_propagation` | logistic | EP | integrate | best marginals, needs damping |
| `one_vs_rest` | logistic | Laplace | softmax over log-odds | more than two classes |

The `predictive` axis is not cosmetic. At the demo's test point the exact
predictive is 0.5519, the plug-in reports **0.5607** — overconfident, always, in
the direction of the mode — integrating over the same Gaussian gives 0.5508, and
EP gives 0.5520. Plugging the mode in is what most implementations do and it is
the one choice that is systematically wrong.

`probit` bundles two independent facts that happen to line up: the probit
likelihood has closed-form EP moments (truncated-Gaussian integrals), and its
predictive given a Gaussian posterior is exact, not an approximation. Its
drawback is the same as everyone's: $\Phi$ is not in numpy.

`laplace` and `logistic` share every line of the fit. They are separate atoms in
the table because the predictive is a different function, and because a table
that hides that difference teaches the wrong thing.

## Derivation

### 1. The model, and the integral that is not there

$$f \sim \mathcal{GP}(0, k), \qquad y_i \mid f_i \sim p(y_i \mid f_i), \qquad p(f) = N(f; 0, K) \qquad (1)$$

with $K_{ij} = k(x_i,x_j)$ and $y_i \in \{-1,+1\}$. Bayes gives

$$p(f\mid y) = \frac{1}{p(y)}\,N(f;0,K)\prod_{i=1}^{n}p(y_i\mid f_i), \qquad p(y) = \int N(f;0,K)\prod_{i=1}^{n}p(y_i\mid f_i)\,df \qquad (2)$$

The product of $n$ sigmoids is not the exponential of a quadratic in $f$, so (2)
has no closed form and no conjugate update exists. Every method below is a
Gaussian ansatz $q(f) = N(f;\mu,\Sigma)$ plus a rule for choosing it — and the
predictive is

$$p(y_*=1\mid y) = \int \sigma(f_*)\,p(f_*\mid y)\,df_* \qquad (3)$$

which is one more integral, again not available.

### 2. Laplace: quadratic at the mode

Write $\Psi(f) = \log p(f,y) = -\tfrac12 f^\top K^{-1}f + \sum_i \log p(y_i\mid f_i)$.
Then

$$\nabla\Psi = -K^{-1}f + t, \qquad \nabla^2\Psi = -K^{-1} - W, \qquad t_i = \frac{\partial \log p(y_i\mid f_i)}{\partial f_i}, \quad W_{ii} = -\frac{\partial^2 \log p(y_i\mid f_i)}{\partial f_i^2} \qquad (4)$$

with $W$ positive because each log-likelihood is concave in $f_i$. Newton's
method on $\nabla\Psi = 0$ is

$$(K^{-1} + W)\,\delta = t - K^{-1}f, \qquad f \leftarrow f + \delta \qquad (5)$$

and at convergence $\hat f$ the posterior is approximated by

$$q(f) = N\!\left(\hat f,\ (K^{-1}+W)^{-1}\right) \qquad (6)$$

The evidence follows from the same expansion as
[`laplace_approximation.py`](laplace_approximation.py) (3), one dimension per
latent:

$$\log p(y) \approx -\tfrac12 \hat f^\top K^{-1}\hat f + \sum_i \log p(y_i\mid \hat f_i) - \tfrac12\log\lvert I + W^{1/2}KW^{1/2}\rvert \qquad (7)$$

The predictive needs two things from (6). The mean uses an identity that is easy
to miss: at the mode $K^{-1}\hat f = t$, so

$$\mathbb{E}[f_*] = k_*^\top K^{-1}\hat f = k_*^\top t \qquad (8)$$

— no inverse of $K$ required, just the gradient vector the optimiser already
has. The variance uses the Woodbury form

$$k_{**} - k_*^\top (K + W^{-1})^{-1}k_* = k_{**} - \big\lVert L^{-1}W^{1/2}k_*\big\rVert^2, \qquad B = I + W^{1/2}KW^{1/2} = LL^\top \qquad (9)$$

### 3. The numerical trick that makes (5) work

$W_{ii}$ can be enormous — for a well-separated point $p(1-p) \to 0$, so
$W^{-1}$ is fine but $W$ is nearly singular, and solving with $K^{-1}+W$ directly
loses precision. The code keeps everything on the $B$ side:

$$a = (K+W^{-1})^{-1}\big(Wf + t\big), \qquad \hat f = Ka \qquad (10)$$

and evaluates (10) as $\;a = b - W^{1/2}L^{-\top}L^{-1}W^{1/2}Kb$ with
$b = Wf + t$. Every solve is triangular, no inverse of $K$ or $W$ is formed, and
the determinant in (7) is $\sum_i \log L_{ii}$. That is
[Rasmussen & Williams (2006)](https://gaussianprocess.org/gpml/) Algorithm 3.1,
and it is the difference between a method that converges and one that does not
on separable data.

### 4. EP: a Gaussian site per data point

EP replaces each likelihood factor by an unnormalised Gaussian in $f_i$:

$$t_i(f_i) = \tilde Z_i\,N(f_i;\,\nu_i,\,1/\tau_i), \qquad q(f) \propto N(f;0,K)\prod_i t_i(f_i) \qquad (11)$$

This is a Gaussian, so its parameters are two matrix formulas:

$$\Sigma = (K^{-1} + T)^{-1}, \qquad \mu = \Sigma\,(\tau \odot \nu), \qquad T = \mathrm{diag}(\tau) \qquad (12)$$

Sites are updated one at a time. Removing site $i$ from $q$ gives the *cavity*,
which in natural parameters is a subtraction of one precision and one linear
term. With Sherman–Morrison on the rank-one difference:

$$\sigma_{-i}^2 = \frac{1}{1/\Sigma_{ii} - \tau_i}, \qquad \mu_{-i} = \sigma_{-i}^2\left(\frac{\mu_i}{\Sigma_{ii}} - \tau_i\nu_i\right) \qquad (13)$$

The **tilted** distribution is the cavity times the true likelihood,

$$\hat q_i(f_i) \propto q_{-i}(f_i)\,p(y_i\mid f_i) \qquad (14)$$

and EP chooses the new site so that $q$ — which is $q_{-i}$ times the site — has
$\hat q_i$'s mean and variance. Solving for the site parameters:

$$\tau_i^{\text{new}} = \frac{1}{\tilde\sigma_i^2} - \frac{1}{\sigma_{-i}^2}, \qquad \nu_i^{\text{new}} = \frac{1}{\tau_i^{\text{new}}}\left(\frac{\tilde\mu_i}{\tilde\sigma_i^2} - \frac{\mu_{-i}}{\sigma_{-i}^2}\right) \qquad (15)$$

**The division by $\tau$ in (15) is where implementations go wrong.** The bracket
is the site's *natural* linear parameter $\eta_i$; $\nu_i = \eta_i/\tau_i$ is its
mean. Storing $\eta$ where the code expects $\nu$ leaves the posterior variance
right and the mean wrong by a factor of $\tau_i$ — precisely the bug the demo
found, and it looked plausible because the variances matched.

For a probit likelihood, (14) is a Gaussian times a Gaussian CDF, and its moments
are the truncated-normal formulas:

$$z_i = \frac{y_i\mu_{-i}}{\sqrt{1+\sigma_{-i}^2}}, \quad \hat Z_i = \Phi(z_i), \quad \tilde\mu_i = \mu_{-i} + \frac{y_i\sigma_{-i}^2\,\varphi(z_i)}{\Phi(z_i)\sqrt{1+\sigma_{-i}^2}}, \quad \tilde\sigma_i^2 = \sigma_{-i}^2 - \frac{\sigma_{-i}^4\,\varphi(z_i)\big(z_i + \varphi(z_i)/\Phi(z_i)\big)}{\Phi(z_i)(1+\sigma_{-i}^2)} \qquad (16)$$

One site, one closed form, no quadrature. For the logistic likelihood there is no
such formula and the demo computes the same three moments with 40-point
Gauss–Hermite quadrature.

### 5. The EP evidence, and the amplitude nobody remembers

Moment matching in (15) fixes $\tau_i$ and $\nu_i$ — the *shape* of the site — but
not $\tilde Z_i$. The missing condition is that the site must carry the same
mass as the likelihood it replaces, measured under the cavity:

$$\tilde Z_i = \frac{\hat Z_i}{N(\nu_i;\,\mu_{-i},\,\sigma_{-i}^2 + 1/\tau_i)} \qquad (17)$$

With that, the approximate evidence is the integral of (11):

$$\log Z_{\text{EP}} = \sum_i \log \tilde Z_i + \log N\!\big(0;\,\nu,\,K + S\big), \qquad S = \mathrm{diag}(1/\tau) \qquad (18)$$

Drop (17) and the estimate is not even exact for a Gaussian likelihood, which is
the check that catches it. Keep it, and the one-point case becomes exact for a
different reason: with $n=1$ the cavity is the prior, so $\hat Z_1 = p(y_1)$ and
the two Gaussian terms in (18) cancel by symmetry. The demo asserts this to
$10^{-13}$.

### 6. Two likelihoods, one difference

$$\text{logistic: } \log p(y\mid f) = -\log(1+e^{-yf}), \qquad t = y\sigma(-yf), \qquad W = \sigma(f)\sigma(-f) \qquad (19)$$

$$\text{probit: } \log p(y\mid f) = \log\Phi(yf), \qquad t = \frac{y\varphi(z)}{\Phi(z)}, \qquad W = \frac{\varphi(z)}{\Phi(z)}\left(z + \frac{\varphi(z)}{\Phi(z)}\right) \qquad (20)$$

Both are concave, so both Laplace and EP apply. The probit ratios
$\varphi(z)/\Phi(z)$ have to be computed as $\exp(\log\varphi - \log\Phi)$, not
as a quotient of two small numbers; the demo's `log_ndtr` does that and falls
back to the asymptotic series in the far tail. The logistic $W$ uses
$\sigma(f)\sigma(-f) = \sigma(f)(1-\sigma(f))$, which is the same quantity and is
what the Fisher information of a Bernoulli is.

The predictive is where they differ most:

$$p(y_*=1\mid y) \approx \begin{cases} \Phi\!\left(\mu_*/\sqrt{1+v_*}\right) & \text{probit} \\ \int\sigma(\mu_* + \sqrt{v_*}z)\,\varphi(z)\,dz & \text{logistic} \end{cases} \qquad (21)$$

The probit one is *exact given* the Gaussian $q$ — the convolution of a Gaussian
with a Gaussian CDF is a Gaussian CDF — so `probit:predictive` is exact under the
approximation, not a second approximation on top. The logistic one is a genuine
integral, done with Gauss–Hermite. Both pull the prediction toward $\tfrac12$
relative to the plug-in, because uncertainty in $f_*$ cannot make the model more
confident.

### 7. More than two classes

Neither approximation extends to $k$ classes by itself; the multiclass probit
posterior is a product of $k$ coupled Gaussians and the site structure changes
shape. The cheap route is $k$ one-vs-rest binary problems, and then a combination
rule. Normalising the $k$ probabilities is wrong — they are not calibrated
against each other and the sum is an artefact — so the code combines log-odds:

$$P(c\mid x) = \frac{\exp(r_c)}{\sum_{c'}\exp(r_{c'})}, \qquad r_c = \log\frac{p_c(x)}{1-p_c(x)} \qquad (22)$$

$r_c$ is the latent mean of classifier $c$ up to the link, so (22) is a softmax
over the $k$ latent functions. It is still not a coherent probability model — the
$k$ classifiers were fitted independently and can all be confident about
different classes — which is why multinomial EP and one-vs-one exist. It works
well enough that it is what most libraries do.

## Code walkthrough

The probit site function is (20), and the only interesting line is the ratio of
two exponentially small numbers, which has to be an exponential of a difference:

```python
def _probit_site(y: np.ndarray, f: np.ndarray):
    """Same three quantities for `p(y|f) = Φ(y f)`."""
    z = y * f
    ll = log_ndtr(z)
    r = np.exp(-0.5 * z * z - 0.5 * LOG2PI - ll)          # φ(z)/Φ(z)
    return ll, y * r, r * (z + r)
```

`log_ndtr` is a hand-rolled $\log\Phi$, because numpy has no `erfc`:

```python
def ndtr(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF. numpy has no `erf`, so this goes through `math`."""
    z = np.asarray(z, dtype=float)
    flat = np.array([0.5 * math.erfc(-v / SQRT2) for v in z.ravel()], dtype=float)
    return flat.reshape(z.shape)
```

The Newton step is (10). Two triangular solves, and the `K_ @ b` inside the
first one is the only matrix-matrix product in the loop:

```python
            sW = np.sqrt(W)
            L = np.linalg.cholesky(np.eye(n) + sW[:, None] * self.K_ * sW[None, :])
            b = W * f + t
            # a = (K + W⁻¹)⁻¹ b, and the mode is K a. Doing it this way keeps
            # every solve triangular; forming K + W⁻¹ would need an inverse.
            v = np.linalg.solve(L, sW * (self.K_ @ b))
            a = b - sW * np.linalg.solve(L.T, v)
            new = self.K_ @ a
```

The evidence is (7), and `slogdet` is not used: the Cholesky factor is already
there, so the log-determinant is the sum of its log-diagonal in the same pass
that the predictive needs.

```python
        self.a_ = np.linalg.solve(self.K_, f)
        self.logZ_ = (-0.5 * float(f @ self.a_) + float(self._site(f)[0].sum())
                      - float(np.log(np.diag(self.L_)).sum()))
        # predictive: mean = k*ᵀ t, var = k** − ‖L⁻¹ √W k*‖²
        self._coef, self._chol, self._scale = self.t_, self.L_, self.sqrtW_
```

EP's posterior is (12), computed without ever forming $K^{-1}$: the shape is
$K - V^\top V$ with $V = L^{-1}T^{1/2}K$, which is the Woodbury identity applied
to $(K^{-1}+T)^{-1}$.

```python
        sT = np.sqrt(np.maximum(tau, 0.0))
        L = np.linalg.cholesky(np.eye(n) + sT[:, None] * self.K_ * sT[None, :])
        V = np.linalg.solve(L, sT[:, None] * self.K_)
        Sigma = self.K_ - V.T @ V                      # (K⁻¹ + diag τ)⁻¹
        return Sigma, Sigma @ (tau * nu)
```

The cavity and the site update are (13) and (15). The comment on `nu_i` is the
one that matters — it is the line the demo's first version had wrong.

```python
                s2 = 1.0 / max(1.0 / Sigma[i, i] - tau[i], 1e-12)
                m = s2 * (mu[i] / Sigma[i, i] - nu[i] * tau[i])
                m_t, s2_t, _ = self._tilted(m, s2, self.y_[i])
                tau_i = 1.0 / max(s2_t, 1e-300) - 1.0 / s2
                # natural-form site: precision `tau_i`, linear term `eta_i`.
                # `nu` is stored as the *mean*, eta/tau, which is what makes
                # `mu = Sigma @ (tau * nu)` one matrix-vector product.
                nu_i = (m_t / max(s2_t, 1e-300) - m / s2) / tau_i
```

The probit tilt is (16) written out. `r` is again $\varphi/\Phi$, and the
variance is the truncated-normal expression with the $z + r$ correction that
makes the site *narrower* than the cavity — the term without it is the naive
answer and it is wrong by a factor that grows in the tail.

```python
            z = y * m / math.sqrt(1.0 + s2)
            lz = float(log_ndtr(np.array([z]))[0])
            r = math.exp(-0.5 * z * z - 0.5 * LOG2PI - lz)
            mean = m + y * s2 * r / math.sqrt(1.0 + s2)
            var = s2 - s2 * s2 * r * (z + r) / (1.0 + s2)
            return mean, var, lz
```

The amplitude fix (17) is the loop at the end of `_ep`: the tilted mass against
the mass the site would have under the cavity. `log_zq` is the second term of
(18), one Cholesky of $K + S$.

```python
            lz = self._tilted(m, s2, self.y_[i])[2]
            v = s2 + S[i]                                  # variance of q_{-i} * site
            log_nu = -0.5 * LOG2PI - 0.5 * math.log(v) - 0.5 * (nu[i] - m) ** 2 / v
            log_amp += lz - log_nu
        self.logZ_ = log_amp + log_zq
```

Prediction is (21), three branches, and this is the whole of the `laplace` versus
`logistic` variant difference:

```python
        mean, var = self._latent(X)
        if self.predictive == "plug":
            return sigmoid(mean)
        if self.predictive == "analytic":       # probit: exact integral, no quadrature
            return ndtr(mean / np.sqrt(1.0 + var))
        x, w = np.polynomial.hermite.hermgauss(20)
        nodes = mean[:, None] + np.sqrt(2.0 * var)[:, None] * x
        return (w * sigmoid(nodes)).sum(axis=1) * INV_SQRT_PI
```

The multiclass wrapper is (22), and the only subtlety is the clip: a classifier
that says 0.9999 would otherwise produce an infinite log-odds.

```python
            P = np.stack([m.predict_proba(X) for m in self.models_], axis=1)
            logit = np.log(np.clip(P, 1e-12, 1.0)) - np.log(np.clip(1.0 - P, 1e-12, 1.0))
            logit -= logit.max(axis=1, keepdims=True)
            E = np.exp(logit)
            return E / E.sum(axis=1, keepdims=True)
```

## Complexity

With $n$ training points, $m$ test points, and $T$ EP sweeps:

| Step | Time | Memory |
|:--|:--|:--|
| kernel matrix | $O(n^2d)$ | $O(n^2)$ |
| one Newton step | $O(n^3)$ | $O(n^2)$ |
| full Laplace fit | $O(I n^3)$, $I \approx 5$ | $O(n^2)$ |
| one EP sweep | $O(n^3 + n^2)$ | $O(n^2)$ |
| full EP fit | $O(Tn^3)$, $T \approx 25$ | $O(n^2)$ |
| predictive | $O(n^2m)$ | $O(nm)$ |

EP costs roughly five Laplace fits, and both are cubic for the same reason as in
regression: the bottleneck is a Cholesky of an $n\times n$ dense matrix. The
demo's numbers are 5 Newton steps and 25 damped EP sweeps for 60 points.

The predictive is the part that scales badly in the wrong place. Every test point
needs a row of $K_*$ and a triangular solve, so $m$ test points cost
$O(n^2m)$ and $O(nm)$ memory — and at $m = 10^5$ that is the whole budget. The
usual fix is the same as in regression: inducing points (a sparse GP), which
replaces $K$ with $K_{nm}K_{mm}^{-1}K_{mn}$ everywhere here, and turns the cost
into $O(nm^2)$. With a non-Gaussian likelihood the sparse version needs the
sites as well, and that is the sparse EP of Kuss (2006).

## Failure modes

**EP not converging.** The site updates are a fixed-point iteration with no
Lyapunov function; on some problems they oscillate, and the code's `damping = 0.7`
is what makes the demo converge in 25 sweeps instead of never. EP's fixed points
can also be multiple. *Diagnostic:* watch `delta` across sweeps; if it plateaus
above `tol`, lower the damping and raise `n_iter`. A converged EP on a
*misspecified* model is still wrong, and the failure is silent.

**The cavity variance going negative.** (13) has $1/\Sigma_{ii} - \tau_i$ in a
denominator, and for a nearly-deterministic site the two terms are equal to
within rounding. The code clamps at $10^{-12}$; without a guard the sign flips
and the next site update receives a negative precision. *Diagnostic:* any site
with $\tau_i \le 0$ after `fit`, or `nan` in `Sigma_`.

**EP being overconfident on the wrong model.** EP matches moments of the
approximation, not of the truth, and its variance estimates are famously too
small when the likelihood does not fit the data — outliers, label noise, a
Gaussian process whose kernel is wrong. Laplace errs in the same direction but by
a different amount. *Diagnostic:* EP versus Laplace on the same data; a large
disagreement in variance (not in mean) is the signature, and neither is
trustworthy in that case.

**Treating the evidence as a Bayes factor you can trust.** (7) and (18) are both
approximations to the same quantity, and they disagree by more than their
individual errors would suggest. On the demo's two points they are 0.014 nats
apart; on a misspecified model they can differ by tens. *Diagnostic:* compare
with a nested estimate or a short MCMC run before reporting a Bayes factor
difference below about 1 nat.

**The plug-in predictive.** `sigmoid(mean)` is the mode's opinion, and the mode's
opinion is more extreme than the posterior's average, always. On the demo it is
0.0087 too high, and the error grows with the predictive variance — so it is
largest exactly where the model is least sure. *Diagnostic:* `integrate=True`
versus `integrate=False`; if the two differ by more than the uncertainty you are
reporting, the plug-in one is the wrong number.

**Separable data.** If the classes are separable, the likelihood has no maximum
at any finite $f$ and the posterior is improper along a direction. Newton walks
off, `W` collapses, and `B` becomes the identity — which is numerically stable
and statistically meaningless. The kernel and the prior are what prevent this;
there is no `prior_prec` knob here because the GP prior is the regulariser.
*Diagnostic:* $\lVert\hat f\rVert$ growing without bound and `logZ_` increasing
monotonically as the step count rises.

**One-vs-rest incoherence.** The $k$ binary classifiers are independent, so the
combined probabilities are not those of any single generative model; a point can
be far from all $k$ class means and still receive a confident label. This is not
a bug to be fixed by better normalisation — normalising probabilities is just as
incoherent — but a property of the approach. *Diagnostic:* the maximum
probability on points far outside the training region, which should be near
$1/k$ and will not be.

**$n$ above a few thousand.** Dense $O(n^3)$ with $n = 10^4$ is $10^{12}$ flops
per Newton step and 800 MB for the kernel matrix. Both approximations are exact
algorithms with an exact cost; they do not degrade, they stop. Use inducing
points or a variational bound.

## Applications

**Any binary classification with calibrated uncertainty.** GP classification is
the default when the answer is "how likely" rather than "which" — medical
screening, credit risk, fault detection. The predictive variance is a real
quantity here, not a by-product, and it is the reason to pay $O(n^3)$ instead of
logistic regression.

**Spatial and spatiotemporal data.** Geo-referenced presence/absence records,
species distribution modelling, air-quality threshold exceedance: a kernel on
coordinates is a more honest prior than feature engineering, and the latent
surface is the interpretable object. The regression version is
[`gaussian_process.py`](gaussian_process.py); this file is the same thing for
counts that have been thresholded.

**Bayesian optimisation with constraints.** Constrained BO keeps two GPs — the
objective and the feasibility classifier — and evaluates only where the
expected improvement is high *and* the probability of feasibility is high. That
probability is exactly `predict_proba` from this file.

**Ordinal regression.** Ratings, severity scores, Likert data: probit GP with
$K-1$ ordered thresholds is the standard model, and it reduces to this file for
$K=2$.

**Active learning.** The predictive variance is the acquisition function. Points
with $v_*$ large and $\lvert\mu_*\rvert$ near zero are where the model does not
know, and the EP version's better-calibrated variances make it the better
acquisition.

**Where it is a trap.** Multiclass problems where a coherent joint matters
(use multinomial EP or a proper softmax GP), datasets where the labels are noisy
in a way the likelihood does not model, and anything above a few thousand points
without a sparse approximation. Also: using the evidence to compare a GP
classifier with a non-probabilistic model, where the comparison is between two
different notions of likelihood.

## References

- Opper & Winther (2000). *Gaussian processes for classification: mean-field
  algorithms.* Neural Computation 12, 2655. EP for GP classification, and the
  mean-field alternative.
- Minka (2001). *Expectation propagation for approximate Bayesian inference.*
  UAI. EP as moment matching, and the fixed-point structure of (14)–(15).
- Williams & Barber (1998). *Bayesian classification with Gaussian processes.*
  IEEE TPAMI 20, 1342. The Laplace approximation for GP classification, and the
  predictive formulas (8)–(9).
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.* MIT
  Press, Chapter 3. Algorithm 3.1 (the stable Newton step), 3.5 (EP), and the
  evidence formulas (7) and (18).
- Nickisch & Rasmussen (2008). *Approximations for binary Gaussian process
  classification.* JMLR 9, 2035. A systematic comparison of Laplace, EP and
  variational bounds — the source of the "EP is better on marginals, Laplace is
  often as good on the evidence" finding the demo reproduces.
- Seeger (2005). *Expectation propagation for exponential families.* Technical
  report. The general EP derivation, including the amplitude condition (17).
- Kuss & Rasmussen (2005). *Assessing approximate inference for binary Gaussian
  process classification.* JMLR 6, 1679. What EP gets right and wrong, measured
  against MCMC.
- Kim & Ghahramani (2006). *Bayesian Gaussian process classification with the
  EM-EP algorithm.* JMLR 7, 1949. Multiclass GP classification, and why
  one-vs-rest is an approximation of an approximation.
- Hensman, Matthews & Ghahramani (2015). *Scalable variational Gaussian process
  classification.* AISTATS. The sparse version that makes $n = 10^6$ possible,
  and the bound that EP does not have.
- Kuss (2006). *Gaussian Process Models for Robust Regression, Classification,
  and Reinforcement Learning.* PhD thesis, TU Darmstadt. Sparse EP, and the
  practical damping schedules.
