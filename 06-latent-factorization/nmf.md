# Non-negative matrix factorisation

`nmf.py` · Latent structure · Tier 1 · 7 variants

## Intuition

Two matrices multiply to a data matrix. The only constraint is that all three are
non-negative:

$$X \approx WH, \qquad X \in \mathbb{R}^{n \times d}_{\ge 0},\quad W \in \mathbb{R}^{n \times k}_{\ge 0},\quad H \in \mathbb{R}^{k \times d}_{\ge 0}. \qquad (1)$$

That looks like a small tweak on the SVD, and it changes the answer completely.
The SVD minimises the same squared error without the constraint, and its solution
is the truncated SVD — signed, orthogonal, and *cancelling*. Allow subtraction
and the first component can be a contrast, `feature_a − feature_b`, which is a
perfectly good direction for explaining variance and a meaningless one for
describing what the data is made of.

Remove the option to subtract and each row of $X$ becomes a **sum** of the rows
of $H$, weighted by the corresponding row of $W$:

$$x_i \approx \sum_{c=1}^{k} W_{ic}\, h_c, \qquad W_{ic} \ge 0. \qquad (2)$$

Nothing can cancel, so every part pays its own way. This is the whole reason the
method is used: on a face dataset the rows of $H$ come out as eyes, a nose, a
mouth, and each face is a weighted sum of the parts at their positions; on a
document-term matrix they come out as topics, and each document is a mixture.

```
        X          ≈        W         H
   ┌─────────┐          ┌───┐   ┌───────────────┐
   │ 0 1 1 0 │  = 0.9 ·  │ ▪ │ ⊗ │ ████          │  ∧ part 1
   │ 1 0 0 1 │  + 0.4 ·  │ ▪ │   │         ████  │  ∧ part 2
   │ 0 1 0 1 │  + 0.7 ·  │ ▪ │   │      ████     │  ∧ part 3
   └─────────┘          └───┘   └───────────────┘
      data               codes        parts
```

The same picture explains the failure mode: if a part never fires, no
multiplicative update can start it firing (zero times anything is zero). Parts
are switched on, never grown from nothing, which is why initialisation is a real
part of the algorithm and not a formality.

The objective also has to be chosen, and the choice is about variance, not
taste. For counts with variance equal to the mean, squared error is the wrong
geometry; the right one is the generalised Kullback–Leibler divergence:

$$D_{KL}(X \,\|\, WH) = \sum_{ij} \left( X_{ij}\log\frac{X_{ij}}{(WH)_{ij}} - X_{ij} + (WH)_{ij} \right). \qquad (3)$$

Both objectives are non-convex in $(W,H)$ jointly, both have simple monotone
updates, and both have the same two ambiguities — scale and permutation — which
is what makes the output interpretable and the parameters unidentifiable at the
same time.

## Variants

| Variant | Objective | Update | Exact? | When |
|:--|:--|:--|:--|:--|
| `frobenius_mu` | $\|X-WH\|_F^2$ | multiplicative | monotone, sublinear | the default |
| `kl_mu` | $D_{KL}(X\|WH)$ | multiplicative | monotone, sublinear | counts, rates, intensities |
| `als` | $\|X-WH\|_F^2$ | exact solve per factor | alternating, no monotonicity proof | small $k$, fast per sweep |
| `hals` | $\|X-WH\|_F^2$ | coordinate descent | monotone, fast | the practical default |
| `sparse_nmf` | Frobenius + L0 screening | threshold the parts | not an objective, a constraint | interpretable supports |
| `online_nmf` | Frobenius, minibatches | decayed sufficient statistics | stochastic | $n$ too large for memory |
| `semi_nmf` | $\|X-WH\|_F^2$ | Ding's square-root rule | monotone in $H$ | signed $X$, non-negative basis |

**`frobenius_mu`** is Lee and Seung's original pair of updates. Each factor is
updated by multiplying it by the ratio of the negative to the positive part of
the gradient, which is non-negative by construction and therefore needs no
projection step. The demo checks that the loss falls at every sweep and that it
lands on the same value as the other two routes.

**`kl_mu`** is the same idea for the divergence (3). The update is not the
Frobenius one with a different loss plugged in — the denominator is a *sum* of
$W$ rather than a product with $H$, which is what makes it the right
generalisation, and it reduces to the Frobenius update only in the small-residual
limit.

**`als`** alternates exact least-squares solves: with $H$ fixed, the optimal $W$
is $XH^\top(HH^\top)^{-1}$, and the non-negativity is restored by clipping. The
clip is not the constrained optimum, so there is no monotonicity guarantee; in
practice it converges faster per sweep than MU and needs the same
regularisation.

**`hals`** is coordinate descent taken literally: hold all of $W$ and $H$ fixed
except one column of $W$ or one row of $H$, and solve that exactly. Because the
objective is quadratic, each such solve is one division, and because the
subproblem is one-dimensional with a positivity constraint, the solution is the
unconstrained one clipped at zero. It is monotone, has no step size, and — the
reason it is used — it converges in tens of sweeps where MU needs hundreds.

**`sparse_nmf`** switches off loadings below a fraction of their part's maximum.
This is a projection, not a penalty, and the distinction matters: the L1 route on
$H$ alone is scale-unbounded and drifts to a degenerate solution (the demo
measures it — $\|H\|$ collapses by six orders of magnitude while the fit
*improves*, because $H \to 0$, $W \to \infty$ with $WH$ fixed is free).

**`online_nmf`** replaces the dataset with sufficient statistics. Keep
$A = \sum_i w_i w_i^\top$ and $B = \sum_i x_i w_i^\top$ over minibatches with
exponential forgetting, and update $H$ from those. Memory is $O(kd)$ instead of
$O(nd)$, and the demo's 16-row-per-step fit is within 1% of the batch answer on
200 rows.

**`semi_nmf`** drops the constraint on $W$ and keeps it on $H$: signed weights
over non-negative parts. That is a different model class — a *conical*
combination with negative coefficients — and it fits data that non-negative NMF
cannot represent at all, such as a similarity matrix with negative entries.

## Derivation

### Multiplicative updates come from an auxiliary function

Take the Frobenius objective with $W$ fixed, and write $h$ for one row of $H$:

$$F(h) = \|X - WH\|_F^2. \qquad (4)$$

Its gradient is $\nabla F = 2\,W^\top(WH - X)$, which splits into a
non-negative and a non-positive part: $W^\top WH \ge 0$ and $W^\top X \ge 0$. The
descent direction is $-\nabla F$, and a gradient step would need a step size to
stay non-negative. Lee and Seung's observation is that the *ratio* of the two
parts is a valid step of exactly the right size:

$$H \leftarrow H \odot \frac{W^\top X}{W^\top W H}, \qquad (5)$$

which is non-negative for any $W \ge 0$, $X \ge 0$, and $H > 0$. The proof of
monotonicity is an auxiliary-function argument: $F$ is majorised by
$G(h, h^t)$ built from $W^\top W H$ at the current point, $G(h^t,h^t) = F(h^t)$,
and minimising $G$ in closed form gives exactly (5). Two consequences follow.

- Every fixed point of (5) satisfies the KKT conditions of the constrained
  problem, because at a fixed point $W^\top X = W^\top W H$, which is the
  stationarity condition with a non-negative multiplier.
- The convergence rate is sublinear, and slow. This is not a defect of the
  implementation; MU takes hundreds of sweeps because it is a majorise–minimise
  scheme with a very loose majoriser.

The $W$ update is (5) transposed, using $X H^\top$ and $W H H^\top$.

### The KL update is the same manoeuvre with a different split

For (3) with $W$ fixed, the gradient is
$\nabla = \mathbf{1} - W^\top (X / WH)$ where the division is elementwise. The
non-negative part is $W^\top(X/WH)$ and the non-positive part is the constant
row vector $\mathbf{1}$ times $W$ column sums, giving

$$H \leftarrow H \odot \frac{W^\top (X / WH)}{W^\top \mathbf{1}}, \qquad (6)$$

and the $W$ update is its transpose. The structure is identical — a ratio, not a
difference — which is why both fits are monotone and both are slow.

The choice between them is a statement about the noise. If $X_{ij}$ is a count,
its variance is its mean, so the squared error over-weights the large entries;
the KL divergence is the Bregman divergence whose Hessian is $1/(WH)$, which is
the variance of a Poisson. The demo makes the difference visible: on Poisson
counts, factors fitted by KL have KL divergence 7911 against 8030 for factors
fitted by Frobenius — a 1.5% difference that is consistent across seeds and would
be larger on sparser counts.

### Coordinate descent: one row at a time

HALS applies the obvious idea. Write the objective as a sum over parts:

$$\|X - WH\|_F^2 = \left\| R_c - w_c h_c \right\|_F^2 + \text{const}, \qquad R_c = X - \sum_{j \ne c} w_j h_j. \qquad (7)$$

Minimising over $h_c$ with $w_c$ fixed is a one-dimensional least-squares problem
in each of the $d$ entries, because $w_c h_c$ is rank one:

$$h_c \leftarrow \max\left(0,\ h_c + \frac{w_c^\top R_c}{\|w_c\|^2}\right), \qquad
w_c \leftarrow \max\left(0,\ w_c + \frac{R_c h_c^\top}{\|h_c\|^2}\right). \qquad (8)$$

Both are exact, both are cheap ($O(nd)$ per part), and the update to $R_c$ after
each is a rank-one correction, so a full sweep is $O(ndk)$ with no matrix
inversion. The clipping is exact rather than a projection: the subproblem is
one-dimensional and the unconstrained optimum is *the* optimum when positive.

The demo's numbers: after 50 sweeps HALS is at 0.0743 relative error where MU is
at 0.0866, and at 400 sweeps all three sit between 0.0741 and 0.0745 — the noise
floor of that construction, since the data was built with 1% noise.

### NNDSVD: start where the SVD wants to be

Random non-negative initialisation works, but it seeds the zero-locking problem
and it wastes sweeps. NNDSVD takes the first singular triple and, for each
subsequent one, keeps whichever of the positive or negative parts carries more
energy:

$$x = u_j,\quad y = v_j, \qquad
\sigma^+ = \|x^+\|\,\|y^+\|, \quad \sigma^- = \|x^-\|\,\|y^-\|, \qquad (9)$$

then sets $w_j = \sqrt{s_j \sigma}\, \hat{x}$, $h_j = \sqrt{s_j \sigma}\, \hat{y}$
for the winning sign. Because each $s_j$ multiplies the retained fraction, the
result reproduces the SVD's energy ordering and is non-negative everywhere.

Two details are load-bearing and both are in the code. Entries that come out as
exactly zero must be seeded — under (5) a zero is absorbing — but the seed has to
be small *relative to that component*, not borrowed from component 1, or a weak
component gets inflated by a factor of four and NNDSVD ends up worse than noise.
With the proportional seed the init costs 1.31 in Frobenius norm on the demo's
data and reaches 0.68 after ten sweeps, against 5.03 and 1.02 for the borrowed
seed.

### Online: the objective you can actually compute

With $n$ too large to hold, minimise the expected loss over the empirical
distribution instead. Mairal's scheme keeps

$$A_t = (1-\rho)A_{t-1} + \rho\, \tilde{W}_t^\top \tilde{W}_t, \qquad
B_t = (1-\rho)B_{t-1} + \rho\, X_t^\top \tilde{W}_t, \qquad (10)$$

where $\tilde{W}_t$ are the codes of the current minibatch (with $H$ fixed, so
it is a linear least-squares problem per batch). The basis is then updated with
the same multiplicative rule, using the statistics in place of the full matrix:

$$H \leftarrow H \odot \frac{B_t^\top}{A_t H}. \qquad (11)$$

Note the orientation: $A$ is $k \times k$ and $B$ is $d \times k$, so both are
independent of $n$. Forgetting ($\rho < 1$) is what makes the scheme able to
track a drifting basis, and it is also the only hyper-parameter that matters:
$\rho = 1$ is plain stochastic descent on the current batch, $\rho \to 0$ freezes
the basis.

### Semi-NMF: splitting the sign

Drop the constraint on $W$ and the objective is still quadratic; the trick is
that every matrix splits into a non-negative and a non-positive part,
$A = A^+ - A^-$ with $A^\pm \ge 0$, so the same ratio construction applies with
a square root because the update is now *multiplicative in both directions*:

$$H \leftarrow H \odot \sqrt{\frac{(W^\top X)^+ + (W^\top W)^- H}{(W^\top X)^- + (W^\top W)^+ H}}. \qquad (12)$$

Why a square root? For a signed $W$ the cross term $W^\top W H$ can point either
way, so the gradient split is no longer into two non-negative matrices and the
linear ratio can go negative. Ding, Li and Jordan's derivation uses a
majoriser that is a *geometric* mean of the two terms, and the square root is
what makes it a valid majoriser. The $W$ update is an exact least-squares solve,
because with $H$ fixed there is no constraint to project onto.

## Code walkthrough

`NMF(n_components, variant=..., seed=..., lam=..., init=..., n_iter=..., batch_size=..., forget=...)`
is the surface: `fit`, `transform` (codes for new rows against the fixed basis),
`inverse_transform`, `fit_transform`, and `loss_curve_` for the diagnostic.
`W_` is $(n,k)$ and `components_` is $(k,d)$ — the same convention as
`sklearn.decomposition.NMF`, chosen so that `transform(X)` and
`inverse_transform(codes)` compose to the identity on the fitted subspace.

### 1. The two Frobenius updates

```python
    def _update_h(self, X, W, H) -> np.ndarray:
        return np.maximum(H * (W.T @ X) / np.maximum(W.T @ W @ H, EPS), EPS)

    def _update_w(self, X, W, H) -> np.ndarray:
        num = X @ H.T
        den = W @ H @ H.T
        return np.maximum(W * num / np.maximum(den, EPS), EPS)
```

That is (5) and its transpose, and the `np.maximum(..., EPS)` does double duty:
it is the guard against a zero denominator, and it is what stops an entry that
has been driven to zero from becoming `nan` on the next pass. `EPS = 1e-12` is
deliberately not a "small number" in the numerical-analysis sense — the entries
here are $O(10^{-1})$, so it behaves as an exact zero and the demo's sparsity
count (`< 1e-10`) treats it as one.

### 2. The KL updates, where the denominator is a sum

```python
    def _update_h_kl(self, X, W, H) -> np.ndarray:
        ratio = X / np.maximum(W @ H, EPS)
        return np.maximum(H * (W.T @ ratio) / np.maximum(W.sum(axis=0)[:, None], EPS), EPS)
```

`X / WH` is the elementwise ratio of data to model; `W.T @ ratio` is its
non-negative gradient part; and `W.sum(axis=0)` is $W^\top\mathbf{1}$ — the sum
down the columns of $W$, broadcast across features. The tell that this is the
KL update and not the Frobenius one is that the denominator does not involve $H$.

The `np.maximum(W @ H, EPS)` is not cosmetic. A count matrix has exact zeros, and
`X / 0` is `inf`, one `inf` spoils the whole product, and `inf * 0` is `nan`; the
clip keeps $X_{ij}/(WH)_{ij}$ at a large finite number instead, which the
subsequent multiplication by $H$ recovers.

### 3. Exact solves, and the projection that is not an optimum

```python
    def _solve_w(self, X, H, nonneg: bool = True) -> np.ndarray:
        G = H @ H.T + 1e-10 * np.eye(H.shape[0])
        W = np.linalg.solve(G, H @ X.T).T
        return np.maximum(W, EPS) if nonneg else W
```

This is the $W$ half of ALS: with $H$ fixed, minimise over $W$ and get
$XH^\top(HH^\top)^{-1}$. Three details. The solve is written transposed
(`solve(G, H @ X.T).T`) because `G` is symmetric and it avoids forming an
inverse. The `1e-10 * np.eye` ridge is what keeps a rank-deficient $H$ — a part
that has died — from making the system singular. And `np.maximum(W, EPS)` is a
*projection*, not the constrained optimum: the true non-negative least-squares
solution needs an active-set method, and clipping is what makes `als` the one
Frobenius variant with no monotonicity guarantee.

The `nonneg=False` path is `semi_nmf`, where $W$ may be signed, so the unclipped
solution is the right answer and the ridge is the only regularisation.

The loss that is reported in `loss_curve_` is a separate function, because the
KL variant has to be measured in its own divergence:

```python
    def _loss(self, X) -> float:
        if self.variant == "kl_mu":
            R = X / np.maximum(self.W_ @ self.components_, EPS)
            return float(np.sum(X * np.log(np.maximum(R, EPS)) - X + self.W_ @ self.components_))
        return float(np.linalg.norm(X - self.W_ @ self.components_) ** 2)
```

The `np.maximum(..., EPS)` inside the log matters for the same reason it did in
the update: a zero count against a zero model entry would otherwise produce
`log(0)` and a `nan` that silently poisons the monotonicity check the demo runs
over the whole curve.

### 4. HALS, and the sign that has to be right

```python
        for c in range(W.shape[1]):
            denom = float(H[c] @ H[c])
            if denom < EPS:                          # a dead component stays dead
                continue
            w_old = W[:, c].copy()
            W[:, c] = np.maximum(W[:, c] + (R @ H[c]) / denom, EPS)
            # R is X − WH, so raising column c *subtracts* its contribution:
            # R += outer(w_old − w_new, H[c]). Getting this sign wrong makes the
            # textbook HALS look like a divergent algorithm.
            R += np.outer(w_old - W[:, c], H[c])
```

`R` is the residual $X - WH$, maintained incrementally so that a sweep costs
$O(ndk)$ rather than $k$ full recomputations. The subtlety is the direction:
raising column $c$ of $W$ *reduces* the residual, so the correction to $R$ has the
sign of $w_{old} - w_{new}$. Written the other way round — which is how the
identity is usually printed, with the residual defined as $WH - X$ — HALS looks
like it oscillates between two losses and a component dies on the first sweep.
That is exactly what happened here before the sign was fixed, and it produced a
final loss eleven times worse than MU with every individual step looking
plausible.

The `denom < EPS` guard is for a component whose row of $H$ is entirely zero: the
division would be by $10^{-12}$ and the column would explode. Skipping it keeps
the component dead, which is the honest behaviour — a part with no loading
anywhere cannot be updated to have one, under either update rule.

### 5. Initialisation, and the seed that must not be borrowed

```python
        if mp >= mn:
            u, v, sigma = xp / max(np.linalg.norm(xp), EPS), yp / max(np.linalg.norm(yp), EPS), mp
        else:
            u, v, sigma = xn / max(np.linalg.norm(xn), EPS), yn / max(np.linalg.norm(yn), EPS), mn
        scale = math.sqrt(max(s[j] * sigma, 0.0))
        W[:, j], H[j, :] = scale * u, scale * v
        # NNDSVDa: a zero entry locks for ever under a multiplicative update,
        # so seed it. The seed has to be small compared with *this* component —
        # borrowing component 1's mean inflates a weak component by 4x here,
        # which is how NNDSVD turns into a worse starting point than noise
        W[W[:, j] == 0.0, j] = 1e-2 * W[:, j].max()
        H[j, H[j, :] == 0.0] = 1e-2 * H[j].max()
```

This is (9). `mp` and `mn` are the energies of the two sign choices, and picking
the larger is what keeps the component's contribution as close to $s_j$ as the
non-negativity allows. The seed line is the difference between a useful
initialisation and a harmful one: with `W[:, 0].mean()` — the published NNDSVDa
recipe, which assumes components of comparable scale — a weak component's zeros
were filled with values four times its own loading, and the init loss was 5.03
against a data norm of 4.25. Starting from that, ten sweeps got to 1.02; with
the proportional seed they get to 0.68.

The demo does not rely on the initialisation for its conclusions: every
"reaches the same place" assertion holds with `init="random"` too, and the
agreement between MU, ALS and HALS is a statement about the objective, not about
the start.

### 6. Screening, which is a projection

```python
        if v in ("frobenius_mu", "sparse_nmf"):
            H = self._update_h(X, W, H)
            W = self._update_w(X, W, H)
            if v == "sparse_nmf":
                # a loading below lam × its part's largest is supported by
                # another part; switch it off. Because a zero is absorbing
                # under a multiplicative update, the support only shrinks.
                cut = self.lam * H.max(axis=1, keepdims=True)
                H[cut > H] = 0.0
```

`keepdims=True` is what makes `cut` a column vector, so each part is thresholded
against its own maximum instead of against the global one — otherwise a part with
small overall magnitude is deleted wholesale the moment another part dominates.
The threshold is applied after the MU step and before the next one, so the
update that follows sees a sparse $H$ and reallocates mass onto the surviving
entries. The demo's fit is 69% zeros with support 19/19/18 against a true 20 per
part, at 1.16× the dense reconstruction error; the cost is a real one and is
reported rather than hidden.

### 7. Online: two statistics and a decay

```python
            rho = self.forget
            A = (1.0 - rho) * A + rho * (Wb.T @ Wb)
            B = (1.0 - rho) * B + rho * (Xb.T @ Wb)
            H = np.maximum(H * (B.T / np.maximum(A @ H, EPS)), EPS)
```

Three lines that are (10) and (11). `A` accumulates the Gram matrix of the
minibatch codes and `B` accumulates their cross-product with the data, so
`B.T / (A @ H)` is the ratio of the gradient parts for the *whole* dataset, not
just the batch. The codes `Wb` are themselves fitted against the current $H$ by
30 MU steps on the batch alone, which is the linear subproblem.

`forget` is the only knob that matters. The demo uses 0.3 with 16 rows per step
and lands 0.6% below the batch answer — below, not equal, because a decayed
average over minibatches weights recent rows more and behaves slightly like a
bootstrapped ensemble.

### 8. Semi-NMF, and why the square root

```python
    def _update_h_semi(self, X, W, H) -> np.ndarray:
        WtX = W.T @ X
        WtW = W.T @ W
        # positive and negative parts: A = A⁺ − A⁻ with A⁺, A⁻ ≥ 0
        num = 0.5 * (np.abs(WtX) + WtX) + 0.5 * (np.abs(WtW) - WtW) @ H
        den = 0.5 * (np.abs(WtX) - WtX) + 0.5 * (np.abs(WtW) + WtW) @ H
        return np.maximum(H * np.sqrt(np.maximum(num, EPS) / np.maximum(den, EPS)), EPS)
```

`0.5*(|A| + A)` is $A^+$ and `0.5*(|A| - A)` is $A^-$; writing them this way
keeps the code to two lines instead of four temporaries. Note that the sign split
is taken on $W^\top W$ as well as on $W^\top X$ — that is the whole point of the
variant, since a signed $W$ makes the Gram matrix indefinite and the cross term
can subtract. The outer `np.sqrt` is (12).

Because $W$ is solved exactly rather than updated multiplicatively, this variant's
sweep is `_solve_w` followed by one pass of (12). The demo's fit on data with
signed weights reaches 0.016 relative error, which is the noise level; the two
ways of forcing ordinary NMF onto the same data — clipping the negatives, or
shifting everything up — cost 0.72 and 0.25.

### 9. The two refusals that keep the API honest

```python
        if self.variant != "semi_nmf" and X.min() < 0.0:
            raise ValueError(f"{self.variant} needs a non-negative X; use semi_nmf for "
                             f"signed data, or shift the data and record the offset")
```

A negative entry in $X$ makes the objective unbounded below for the
non-negative variants: with $W$ free to grow, a negative $x_{ij}$ can be explained
by driving one factor negative and the other positive, which the constraint
forbids — so the optimiser is being asked for something that does not exist. The
error names the alternative, because the failure at run time is a `nan` in the
loss curve three hundred sweeps later.

```python
    def transform(self, X) -> np.ndarray:
        """Codes for new rows against the learned basis: `(m, k)`."""
        if self.W_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X)
        if self.variant == "semi_nmf":
            return self._solve_w(X, self.components_, nonneg=False)
        W = np.full((X.shape[0], self.n_components), 1.0 / self.n_components)
        for _ in range(200):
            W = self._update_w(X, W, self.components_)
        return W
```

`transform` refits only the codes, holding `components_` fixed: that is the
difference between projecting new rows onto a learned basis and refitting the
model, and it is the only defensible way to score held-out data. The initial
`1/k` is a constant start rather than random, so `transform` is deterministic
given the fit — for `semi_nmf` there is nothing to iterate, since the codes are
an unconstrained least-squares solve.

## Complexity

$n$ rows, $d$ columns, $k$ components, $T$ sweeps, $b$ rows per minibatch.

| Variant | Time | Memory | Dominant term |
|:--|:--|:--|:--|
| `frobenius_mu` | $O(Tndk)$ | $O(nd + k(n+d))$ | two matrix products per sweep |
| `kl_mu` | $O(Tndk)$ | $O(nd)$ | the ratio $X/WH$, recomputed |
| `als` | $O(T(nk^2 + k^3 + dk^2))$ | $O(nd)$ | the $k \times k$ solve |
| `hals` | $O(Tndk)$ | $O(nd)$ | rank-one residual updates |
| `sparse_nmf` | $O(Tndk)$ | $O(nd)$ | MU plus a threshold |
| `online_nmf` | $O(Tbk^2 + Tbk d)$ | $O(kd + bk)$ | batch codes and $B$ |
| `semi_nmf` | $O(T(nk^2 + k^3))$ | $O(nd)$ | the exact $W$ solve |
| `_nndsvd` | $O(nd\min(n,d))$ | $O(nd)$ | one SVD |

Two notes. `als` looks cheap per sweep and is not: $HH^\top$ is $k \times k$ but
forming $XH^\top$ is $O(ndk)$, the same as MU, and the $k \times k$ inverse is
in the noise only because $k$ is small. And the $O(nd)$ memory of every batch
variant is the point of `online_nmf`: at $n = 10^7$ and $d = 10^4$ the matrix is
800 GB, while $A$ and $B$ are $10^8$ floats — 0.8 GB — and that is the whole
algorithm.

## Failure modes

**1. Zero-locking, and the part that never fires.** Under (5) an entry that is
zero stays zero for ever. So a part initialised at zero never appears, and a part
driven to zero by a bad sweep is dead. The symptom is a `components_` row that is
exactly zero while the reconstruction error is far above the noise floor. The
diagnostic is `np.sum(components_ == 0, axis=1)`; the fixes are a better
initialisation (NNDSVD), a random restart, or re-seeding the dead row.

**2. Scale ambiguity: the factorisation is not unique.** For any positive
diagonal $D$, $(WD)(D^{-1}H) = WH$, so a part can be scaled up and its codes down
with no change to the fit. Nothing is wrong with this — the demo asserts it to
$10^{-12}$ — but it means loadings are not comparable across components without
normalising, and "part 2 has 40% of the mass" is a statement about the
parametrisation unless the parts are normalised first. Normalise (rows of $H$ to
unit norm, columns of $W$ scaled up) before reading anything off.

**3. Permutation, and the ranking that is not there.** Components come out in
whatever order the initialisation and the numerical path produce; the demo's
three parts come out in reverse. There is no canonical order, so any comparison
across fits (a bootstrap, a cross-validation, two seeds) needs a matching step —
Hungarian, or cosine similarity against a reference — or the "component 1" of the
two fits will be different objects.

**4. Rank selection.** $k$ is a hyper-parameter and the objective is
non-increasing in it, so more components always fit better. There is no
information criterion that is not ad hoc here; the usable diagnostics are a
held-out reconstruction error, the stability of the components across
bootstraps, or — for the sparse variant — the number of parts you can name.
Setting $k$ to the true rank and reporting an exact fit is a test of the code,
not of the model.

**5. The L1 penalty on one factor is not a well-posed objective.** The demo
measures it: 200 sweeps take $\|H\|$ from 2.41 to $7\times 10^{-6}$ and $\|W\|$
from 2.5 to $10^{6}$, with the fit *improving* from 1.02 to 0.36. The reason is
that $\|X - WH\|^2 + \lambda\|H\|_1$ has no minimiser when $W$ is free — the
infimum is reached by $H \to 0$, $W \to \infty$ at fixed product. Penalise both
factors, normalise one of them, or use the screening projection; do not tune
$\lambda$ against a metric, because the metric will be satisfied.

**6. KL on data that is not counts.** The divergence (3) requires $X$ and $WH$ to
be positive and has its scale set by the data. Applying it to standardised
features (which have negative entries) is a type error, and applying it to counts
with a very different magnitude per column silently weights the large columns
entirely. If the data is continuous and positive — intensities, rates — KL is
still defensible; if it is mean-centred, it is not.

**7. Online forgetting that forgets the wrong thing.** $\rho$ weights recent
minibatches. Too small and the basis never moves (the statistics are dominated
by the initial transient); too large and it tracks noise. The diagnostic is the
batch loss on a held-out block over time: it should fall and then stay flat, not
fall and rise. The demo's 0.3 with 16-row batches is a choice that works at
$n = 200$; at $n = 10^7$ the same $\rho$ would be far too fast.

**8. Semi-NMF's sign convention is easy to get backwards.** The constraint is on
the *basis*, not the codes: $W$ signed, $H \ge 0$. Data whose true model is
"signed basis, positive codes" is not semi-NMF data, and semi-NMF will fit it
badly no matter how long it runs — the demo's earlier version did exactly that
and converged to 0.20 relative error on data with an exact rank-3 representation.

## Applications

- **Topic models and document-term matrices.** The rows of $H$ are topics —
  non-negative distributions over words — and the rows of $W$ are the mixtures.
  pLSI and LDA add a probabilistic model on top; NMF gets most of the
  interpretability with none of the inference.
- **Hyperspectral unmixing.** Each pixel is a mixture of endmember spectra, and
  the mixing proportions are non-negative by physics. NMF with a sum-to-one
  constraint on $W$ recovers the endmembers and their abundances, which is the
  canonical remote-sensing use.
- **Audio source separation.** A magnitude spectrogram is non-negative, and its
  rows of $H$ are spectral templates of notes or instruments; $W$ is their
  activation over time. `kl_mu` is the right divergence here (the spectrogram is
  a magnitude, the KL is the Itakura–Saito-adjacent loss), and `sparse_nmf` is
  what makes the templates look like single notes.
- **Face parts and image patches.** Lee and Seung's original demonstration: the
  parts come out as noses, eyes and mouths, where PCA's components come out as
  whole-face contrasts. The reason is (2) — a part can only add.
- **Community detection on networks.** The adjacency matrix, or its
  symmetrised non-negative version, factorises into overlapping communities, with
  $W$ giving membership and $H$ giving the block structure. Overlap is the point:
  NMF assigns a node to several communities with weights, unlike a partition.
- **Recommender systems with implicit feedback.** A user-item interaction matrix
  of counts is non-negative and sparse; the factorisation is the classical
  baseline that matrix factorisation with explicit ratings replaced.
- **Semi-NMF on similarity matrices.** Cosine or correlation similarity has
  negative entries, and its clusters are non-negative parts; semi-NMF is the
  variant that fits the signed matrix directly rather than clipping it.

Where NMF is a trap: reading the parts as a *unique* decomposition. Scale and
permutation are exact symmetries, and even after fixing them the decomposition is
not identified — different initialisations find different local optima with the
same loss and different parts. If the parts have to be comparable across fits, the
non-negativity is not enough on its own; you need separability assumptions (each
part has a feature only it uses) or a stability check across restarts.

## References

- **Lee & Seung (1999),** *Learning the parts of objects by non-negative matrix
  factorization* — the algorithm, the parts-versus-wholes argument, and the
  face demonstration.
- **Lee & Seung (2001),** *Algorithms for non-negative matrix factorization* —
  the multiplicative updates, the auxiliary-function proof of monotonicity, and
  the KL divergence version.
- **Paatero & Tapper (1994),** *Positive matrix factorization* — the same model
  in chemometrics, with the weighted least-squares objective and a different
  naming convention.
- **Cichocki, Zdunek, Phan & Amari (2009),** *Nonnegative Matrix and Tensor
  Factorizations* — the reference for HALS, the sparse and regularised
  extensions, and the taxonomy of divergences.
- **Boutsidis & Gallopoulos (2008),** *SVD based initialization: a head start for
  nonnegative matrix factorization* — NNDSVD and NNDSVDa, including the
  zero-seeding that has to be rescaled per component.
- **Mairal, Bach, Ponce & Sapiro (2010),** *Online learning for matrix
  factorization and sparse coding* — the sufficient statistics (10) and the
  convergence argument for the stochastic scheme.
- **Ding, Li & Jordan (2010),** *Convex and semi-nonnegative matrix
  factorizations* — semi-NMF, the sign split, and the square-root update (12).
- **Donoho & Stodden (2003),** *When does non-negative matrix factorization give
  a correct decomposition of a matrix?* — the separability condition under which
  the parts *are* identified, and the counterexamples when they are not.
