# VAR — one autoregression for the whole system

`var.py` · Time series · Tier 2 · 6 variants

## Intuition

A VAR is what you get when you stop asking "how does $y$ depend on its own past"
and start asking "how does each variable depend on everyone's past". Every
equation has the same right-hand side — a constant and $p$ lags of all $k$
series — so the system is just $k$ regressions stacked side by side. That
symmetry is the whole reason the model is easy to estimate: equation-by-equation
least squares is not an approximation, it is exactly the joint MLE.

The interesting content is in what the fitted matrices let you ask:

- **Stability.** Stack the $p$ coefficient matrices into one companion matrix.
  If every eigenvalue is inside the unit circle, shocks die out; if not, the
  fitted system implies an explosive path and forecasts are meaningless beyond a
  few steps.
- **Granger causality.** Does adding lags of $x$ improve the forecast of $y$?
  That is an $F$-test on a block of coefficients. It is a statement about
  predictability inside this sample and this lag order, not about intervention.
- **Impulse responses.** The MA representation turns the fitted recursion into a
  sequence of matrices $\Psi_h$: what a one-unit shock at time $t$ does to the
  system at $t+h$. Orthogonalising with a Cholesky factor makes the shocks
  uncorrelated, at the cost of imposing a causal ordering.
- **Forecast error variance decomposition.** At each horizon, split each
  variable's forecast uncertainty into shares attributable to each orthogonal
  shock. It answers "what moves $y$" with the same ordering caveat.

The variants here are the standard extensions: exogenous regressors (`varx`),
the three diagnostics as first-class variants (`granger`, `irf`, `fevd`), and a
ridge-penalised fit for systems where $k^2p$ is large relative to the sample.

## Variants

| variant | adds | demo result |
|:--|:--|:--|
| `var` | the plain reduced-form system | $A$ recovered, stable |
| `varx` | contemporaneous regressors $x_t$ | $B=\mathrm{diag}(0.78,-0.54)$ |
| `granger` | $F$-tests for block exclusion | $p=2.7\times10^{-22}$ |
| `irf` | orthogonalised responses | $\Psi_1=\hat A$ |
| `fevd` | variance shares | $[0.80,0.20]$ |
| `var_ridge` | L2 penalty on coefficients | norm 0.494 vs 0.870 |

**`var`** is the base model. On 400 points of a bivariate VAR(1) with
$A=[[0.6,0],[0.35,0.45]]$ it estimates $[[0.6264,0.0343],[0.4187,0.4336]]$,
with companion eigenvalues $0.6838$ and $0.3761$ — comfortably stable. The demo
also fits a VAR(2) and checks that the one-step forecast equals
$c+A_1y_T+A_2y_{T-1}$.

**`varx`** appends $x_t$ to every equation. This is the multivariate analogue of
ARIMAX: the coefficients on $x$ are estimated jointly with the autoregressive
part, so a persistent $y$ does not masquerade as an effect of $x$. On the demo's
$y_t=Ay_{t-1}+Bx_t+u_t$ with $B=\mathrm{diag}(0.8,-0.5)$, the fit recovers
$[[0.7802,-0.0634],[0.0391,-0.5427]]$. It is still a regression, not an
identification strategy: if $x_t$ responds to $y_t$ within the same period, the
coefficient is not causal.

**`granger`** computes the $F$-statistic for each ordered pair. In the demo, $y_0$
strongly predicts $y_1$ ($F=106.64$, $p=2.7\times10^{-22}$) while $y_1$ does not
predict $y_0$ ($F=0.74$, $p=0.390$), exactly as the true zero in $A_{01}$
implies. `granger_pvalues_` is a $k\times k$ matrix with the effect in the row
and the cause in the column, and $1$ on the diagonal because self-causation is
not tested.

**`irf`** stores the orthogonalised responses at the requested horizon. For a
VAR(1) the reduced-form weights are powers of $\hat A$: the demo checks
$\Psi_1=\hat A$ and $\Psi_2=\hat A^2$ to machine precision, and checks that the
$h=0$ response times its transpose reproduces $\hat\Sigma$.

**`fevd`** turns those responses into variance shares. It is the diagnostic that
makes the ordering assumption visible: on data with correlated shocks, the
second variable's variance is 80% attributed to the first shock, because the
Cholesky factor lets shock one move both variables at impact and shock two move
only the second.

**`var_ridge`** replaces OLS with a ridge solve, leaving the intercept
unpenalised. The demo's coefficient norm drops from 0.8700 to 0.4942 at
$\lambda=1$, and `ridge=0` reproduces OLS exactly. The penalty is scaled by the
trace of the Gram matrix so that $\lambda$ means something across data sets.

## Derivation

### The model and its stacked form

With $k$ variables, $p$ lags and $n$ observations,

$$
Y_t=c+\sum_{i=1}^{p}A_iY_{t-i}+u_t,\qquad u_t\sim\mathcal N(0,\Sigma).
\qquad (1)
$$

Collect the regressors for row $t$ into
$z_t=(1,Y_{t-1}^\top,\dots,Y_{t-p}^\top)^\top\in\mathbb R^{d}$ with
$d=1+kp$, and stack rows $t=p+1,\dots,n$ into $Z\in\mathbb R^{n_0\times d}$ and
$Y\in\mathbb R^{n_0\times k}$. Then (1) is the matrix regression

$$
Y=ZB+U,\qquad \hat B=(Z^\top Z)^{-1}Z^\top Y .
\qquad (2)
$$

Because every equation shares the same $Z$, (2) is numerically identical to $k$
separate least-squares fits; the code uses one `lstsq`, and the demo verifies the
equality by refitting each equation by hand. Writing $B$ as
$[c^\top;B_1;\dots;B_p]$ and transposing the lag block gives `coef_[i]` =
$A_{i+1}$. The residual covariance is
$\hat\Sigma=\hat U^\top\hat U/(n_0-d)$.

### Stability

Define the state $\xi_t=(Y_t^\top,\dots,Y_{t-p+1}^\top)^\top$ and the companion
matrix

$$
F=\begin{bmatrix}
A_1&A_2&\cdots&A_{p-1}&A_p\\
I&0&\cdots&0&0\\
0&I&\cdots&0&0\\
\vdots&&\ddots&&\vdots\\
0&0&\cdots&I&0
\end{bmatrix}.
\qquad (3)
$$

Then $\xi_t=F\xi_{t-1}+v_t$, and the system is stable exactly when
$\rho(F)=\max_j\lvert\lambda_j(F)\rvert<1$. For $p=1$, $F=A_1$. The demo prints
$\rho$ for both fitted systems: $0.6838$ for the VAR(1) and $0.8035$ for the
VAR(2), whose companion has four eigenvalues because the state is four
dimensional.

### Impulse responses

Repeated substitution of (1) gives the MA representation

$$
Y_t=\mu+\sum_{h\ge0}\Psi_h u_{t-h},\qquad
\Psi_0=I,\quad
\Psi_h=\sum_{i=1}^{\min(p,h)}A_i\Psi_{h-i}.
\qquad (4)
$$

The reduced-form shocks are correlated, so a unit shock to $u_{1t}$ also moves
$u_{2t}$ on impact. The Cholesky factor $P$ with $PP^\top=\Sigma$ (lower
triangular) defines orthogonal shocks $\varepsilon_t=P^{-1}u_t$, and

$$
\Theta_h=\Psi_hP
\qquad (5)
$$

is the response to a one-standard-deviation shock in $\varepsilon_j$ alone. The
triangularity of $P$ is what imposes the ordering: $\varepsilon_1$ affects every
variable at impact, $\varepsilon_2$ affects variables $2,\dots,k$, and so on. A
different ordering produces a different $\Theta_0$ and different long-run
shares; the data do not choose between them.

### Forecast error variance decomposition

The $H$-step forecast error is
$\sum_{h=0}^{H-1}\Theta_h\varepsilon_{t+H-h}$, so the contribution of shock $j$
to variable $i$ is the sum of squares of the $(i,j)$ entries:

$$
\mathrm{FEVD}_{ij}(H)=
\frac{\sum_{h=0}^{H-1}\Theta_{h,ij}^2}
{\sum_{j'=1}^{k}\sum_{h=0}^{H-1}\Theta_{h,ij'}^2}.
\qquad (6)
$$

Each row sums to one. At $H=1$ the shares are $\Theta_0^2$ normalised, which is
not the identity when $\Sigma$ is non-diagonal; as $H$ grows they approach the
long-run shares implied by the MA weights. The demo's $[0.8002,0.1998]$ for the
second variable is that limit, not a property of the data alone.

### Granger causality

For the equation of variable $e$, the unrestricted residual sum of squares is
$SSE_u$. The restricted model drops the $p$ lag blocks of variable $c$ and
refits, giving $SSE_r$. Under the null that those blocks are zero,

$$
F=\frac{(SSE_r-SSE_u)/p}{SSE_u/\nu},\qquad
\nu=n_0-d,
\qquad (7)
$$

and $F$ has an $F(p,\nu)$ distribution. The code builds the restricted design by
deleting the relevant columns from the already-stacked `design_`, so exogenous
columns and all other lags stay in both models. Rejecting (7) says the lags help
prediction; it does not say that changing $c$ would change $e$.

### Ridge

For large $k$ and $p$, $d=1+kp$ can approach $n_0$ and OLS becomes unstable.
`var_ridge` minimises

$$
\lVert Y-ZB\rVert_F^2+\lambda\tau\lVert B_{[1:]}\rVert_F^2,\qquad
\tau=\frac{\mathrm{tr}(Z^\top Z)}{d},
\qquad (8)
$$

where $B_{[1:]}$ excludes the intercept row. The normal equations are
$(Z^\top Z+\lambda\tau I)B=Z^\top Y$ with the $(1,1)$ penalty entry zeroed, which
is the `shrink[0, 0] = 0.0` line in the code. Scaling by $\tau$ makes $\lambda$
dimensionless: $\lambda=1$ shrinks towards zero by an amount comparable to the
average curvature of the loss.

### Exogenous regressors

`varx` extends (1) to

$$
Y_t=c+\sum_{i=1}^{p}A_iY_{t-i}+Bx_t+u_t,
\qquad (9)
$$

with $x_t\in\mathbb R^m$ and $B\in\mathbb R^{m\times k}$. Estimation is
unchanged — $x_t$ is simply more columns of $Z$ — and forecasting requires a
future path for $x$. The demo checks the first forecast against the manual
expression $c+\hat A y_T+\hat Bx_{T+1}$.

### Likelihood and information criteria

With $\hat\Sigma_{MLE}=\hat U^\top\hat U/n_0$, the Gaussian log-likelihood is

$$
\ell=-\frac{n_0}{2}\Bigl(k\log2\pi+\log\lvert\hat\Sigma_{MLE}\rvert+k\Bigr),
\qquad (10)
$$

and $\mathrm{AIC}=-2\ell+2k d$, $\mathrm{BIC}=-2\ell+kd\log n_0$. The factor $k$
multiplies the parameter count because each of the $d$ regressors has one
coefficient per equation. The demo's VAR(2) has $\mathrm{AIC}=2765.92$; the
number is comparable across lag orders of the same variables, which is how lag
length is normally chosen.

## Code walkthrough

Everything begins by stacking the lag blocks. Adding the intercept first and
each lag in order means the coefficient slice for lag $i$ is contiguous:

```python
        rows = np.arange(p, n)
        blocks = [np.ones((rows.size, 1))]
        for lag in range(1, p + 1):
            blocks.append(Y[rows - lag])
        if exog is not None:
            blocks.append(exog[rows])
        design = np.column_stack(blocks)
        target = Y[rows]
        penalty = 0.0 if self.variant != "var_ridge" else self.ridge
        if penalty > 0.0:
            gram = design.T @ design
            scale = np.trace(gram) / gram.shape[0]
            shrink = np.eye(gram.shape[0]) * penalty * scale
            shrink[0, 0] = 0.0
            beta = np.linalg.solve(gram + shrink, design.T @ target)
        else:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
```

`Y[rows - lag]` is the lagged matrix aligned with `target`, so no shifting is
needed later. The ridge branch solves the normal equations directly instead of
`lstsq`, because the penalty is a modification of the normal equations rather
than of the design; `shrink[0, 0] = 0.0` keeps the intercept free. With
`penalty == 0` the two branches coincide, which the demo checks.

The flat coefficient matrix is then folded back into a stack of $k\times k$
matrices, transposing because `beta` stores one column per equation:

```python
        self.coef_ = np.empty((p, k, k))
        for lag in range(p):
            self.coef_[lag] = beta[1 + lag * k:1 + (lag + 1) * k].T
```

The transpose is the step that is easy to get wrong: row $r$ of `beta` holds
regressor $r$'s coefficients for all equations, whereas `coef_[lag][a, b]` must
mean "effect of variable $b$ at lag $lag+1$ on variable $a$".

The companion matrix is assembled from blocks, with identity blocks below the
diagonal:

```python
    def _companion(self) -> np.ndarray:
        k, p = self.k_, self.p_
        top = np.column_stack([self.coef_[lag] for lag in range(p)])
        if p == 1:
            return top
        lower = np.zeros((k * (p - 1), k * p))
        for lag in range(p - 1):
            lower[lag * k:(lag + 1) * k, lag * k:(lag + 1) * k] = np.eye(k)
        return np.vstack([top, lower])
```

This is (3) written block by block. The `p == 1` shortcut avoids allocating a
$k(p-1)\times kp$ zero block with zero rows for the most common case.

Forecasting shifts a rolling window of the last $p$ vectors. The stored history
is reversed, so lag $1$ is entry $0$:

```python
        history = [self.history_[lag - 1] for lag in range(1, self.p_ + 1)]
        out = np.empty((horizon, self.k_))
        for h in range(horizon):
            value = self.intercept_.copy()
            for lag, past in enumerate(history):
                value = value + self.coef_[lag] @ past
            if self.exog_coef_ is not None:
                value = value + self.exog_coef_.T @ exog[h]
            out[h] = value
            history = [value] + history[:-1]
```

`history = [value] + history[:-1]` slides the window without copying the whole
path, and the exog term is added at each step from the supplied future values.
The `coefficient @ past` orientation matches (1): the coefficient matrix acts on
a column vector.

The MA weights are the same recursion with an identity seed:

```python
        k = self.k_
        weights = np.zeros((horizon, k, k))
        weights[0] = np.eye(k)
        for h in range(1, horizon):
            value = np.zeros((k, k))
            for lag in range(1, min(self.p_, h) + 1):
                value += self.coef_[lag - 1] @ weights[h - lag]
            weights[h] = value
        return weights
```

This is (4). The `min(self.p_, h)` bound is what stops the recursion from
reaching for lags that do not exist yet; for $h<p$ the missing terms are zero by
definition.

Orthogonalisation is one matrix factor and one multiplication:

```python
    def irf(self, horizon: int) -> np.ndarray:
        """Orthogonalised impulse responses, shape `(horizon, k, k)`."""
        self._check_fitted()
        weights = self.ma_coefficients(horizon)
        shock = np.linalg.cholesky(self.sigma_)
        return weights @ shock
```

`np.linalg.cholesky` returns the lower-triangular $P$ of (5), so `weights @ shock`
right-multiplies every horizon's $\Psi_h$ by $P$. The first column of the result
is the response to a shock in the first variable, which is why variable order
matters.

The variance decomposition is a cumulative sum of squares followed by a
row-normalisation:

```python
        responses = self.irf(horizon)
        cumulative = np.cumsum(responses ** 2, axis=0)
        total = cumulative.sum(axis=2, keepdims=True)
        return cumulative / total
```

`axis=0` accumulates over horizons, `axis=2` sums over shocks, and `keepdims`
makes the division broadcast against the $(k,k)$ share matrix at every horizon.
This is (6) with the $H$-th entry containing the first $H$ terms.

Granger's restricted fit reuses the full design and deletes columns:

```python
        dropped = [1 + (lag - 1) * k + cause for lag in range(1, p + 1)]
        keep = [c for c in range(self.design_.shape[1]) if c not in dropped]
        target = self.fitted_ + self.residuals_
        subset = self.design_[:, keep]
        restricted, *_ = np.linalg.lstsq(subset, target[:, effect], rcond=None)
        residual = target[:, effect] - subset @ restricted
        sse_r = float(residual @ residual)
```

The column formula follows from the layout: column $0$ is the intercept, then
lag $1$'s $k$ variables, and variable `cause` sits at offset `cause` inside each
block. `target = fitted_ + residuals_` reconstructs $Y$ on the estimation rows
without storing it separately, and the unrestricted SSE comes from the residuals
already in the model. Deleting columns rather than rebuilding the design is what
keeps exogenous regressors in the restricted model automatically.

## Complexity

With $n$ observations, $k$ variables, $p$ lags, $m$ exogenous columns and horizon
$H$:

- **Design:** $O(n(kp+m))$ memory and time to build.
- **Fit:** one least squares on an $n\times d$ design with $k$ responses,
  $O(nd^2+nkd)$; the ridge branch solves a $d\times d$ system, $O(d^3)$.
- **Companion eigenvalues:** $O((kp)^3)$.
- **Forecast:** $O(Hk^2p)$ for $k$ equations, since each step multiplies $p$
  matrices by vectors.
- **MA weights and IRF:** $O(Hkp\,k^2)=O(Hpk^3)$ in the worst case, because
  every step multiplies $k\times k$ matrices.
- **Granger test:** one restricted least squares per ordered pair, so
  $O(k^2nd^2)$ if all pairs are computed.
- **FEVD:** $O(Hk^3)$ after the IRF.

The demo's largest job is the 400-point bivariate VAR(2) fit, which is three
least-squares calls on a $498\times5$ design; the $F$-tests dominate nothing.
Memory is $O(nd)$ for the design, which is the price of keeping it for the
diagnostics.

## Failure modes

**Granger causality is not causation.** The test asks whether lags of $x$
improve the forecast of $y$ given lags of $y$ and the other variables. A common
driver, a policy reaction function, or an anticipation effect can all produce a
significant $F$ with no intervention behind it. The demo's null case is clean
because the data were simulated with an exact zero; real data rarely are.

**The Cholesky ordering is an assumption.** `fevd` and `irf` depend on the
column order of $Y$, because $P$ is triangular. On the demo's data, reversing
the columns moves the second variable's own share from 0.20 to 0.33. The demo's
$[0.8002,0.1998]$ is a statement about the ordering as much as about the
dynamics, which is why it prints that caveat next to the number. Structural VARs
replace the Cholesky factor with an identified impact matrix, and that is a
different modelling exercise.

**Spurious stability.** A fitted companion matrix can look stable while one
eigenvalue is $0.99$, which produces forecasts that look fine for five steps and
then explode. Check $\rho(F)$ against the horizon you actually need, not just
against one. A near-unit root also makes the Granger tests liberal, because the
asymptotic $F$ approximation assumes stationarity.

**Lag length eats the sample.** A VAR($p$) with $k$ variables spends $1+kp$
degrees of freedom per equation. With $k=5$ and $p=4$ that is 21 parameters per
equation, so a 100-point series leaves 79 residual degrees of freedom and very
wide confidence bands. Ridge helps the variance but biases every coefficient
towards zero, including the ones that are real; the demo's norm drop from 0.870
to 0.494 is a warning, not a recommendation.

**Exogenous forecasts are not free.** `varx` conditions on a future path for
$x$, and its intervals ignore the uncertainty in that path. A scenario forecast
built on an assumed policy rate is a conditional statement, not a predictive
distribution.

**Correlated residuals are not a modelling error.** They are the normal case,
and they are exactly why the reduced-form IRF needs orthogonalisation. The
mistake is to read the orthogonalised result as if the data had identified the
ordering.

## Applications

VARs are the standard workhorse of empirical macroeconomics. Sims (1980) argued
that a small system of macroeconomic aggregates, estimated with few restrictions,
forecasts better and says more than the large structural models then in use, and
the VAR-plus-IRF workflow has been the default ever since. Central banks use
them to trace the response of output and inflation to a policy shock, and the
variance decomposition is a standard way to ask how much of a variable's
movement is attributable to each shock.

Granger tests are used well beyond economics: in neuroscience, lagged
predictability between brain regions is a common connectivity screen; in
genomics, it is applied to expression time series; in industrial forecasting, it
is a cheap way to decide whether a downstream sensor belongs in a model. The
ridge variant matters when the system is wide — dozens of series, a few hundred
observations — which is now common in macro nowcasting and in sensor networks.

The model is the wrong tool when the relationships are nonlinear, when the
series have structural breaks, or when the question is explicitly about a
counterfactual intervention. In those cases a VAR is a baseline and a diagnostic,
not the final answer.

## References

- Sims, C. A. (1980). *Macroeconomics and reality.* Econometrica 48(1), 1–48.
  The paper that made unrestricted VARs and impulse responses standard.
- Granger, C. W. J. (1969). *Investigating causal relations by econometric
  models and cross-spectral methods.* Econometrica 37(3), 424–438. The original
  predictability test behind (7).
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis.*
  Springer. The reference for companion-form stability, MA weights and the
  asymptotic theory of VAR diagnostics.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press.
  Chapters 10–11 cover VAR estimation, Granger causality and IRFs.
- Bernanke, B. S. (1986). *Alternative explanations of the money-income
  correlation.* Carnegie-Rochester Conference Series 25, 49–99. The structural
  alternative to a Cholesky ordering.
- Hoerl, A. E., Kennard, R. W. (1970). *Ridge regression: biased estimation for
  nonorthogonal problems.* Technometrics 12(1), 55–67. The penalty in (8).
