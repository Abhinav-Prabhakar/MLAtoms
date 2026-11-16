# Sparse coding

`sparse_coding.py` · Latent structure · Tier 3 · 4 variants

## Intuition

One signal, a few atoms, from a dictionary bigger than the signal is long.

```
        d = 40                          K = 80
   ┌─────────────┐            ┌───────────────────────┐
   │             │            │ ▮   ▮ ▮     ▮   ▮     │   atoms (columns of D)
   │      x      │     =      │ ▮   ▮ ▮     ▮   ▮     │
   │             │            └───────────────────────┘
   └─────────────┘                  only 4 of 80 carry weight
```

An orthogonal basis of $\mathbb{R}^{40}$ has 40 atoms and can represent anything
with 40 non-zero coefficients. Drop the orthogonality requirement and the same
40-dimensional space can be spanned by 80, or 8000, atoms — and a natural signal
then needs only a handful of them. That is the whole trade: overcompleteness buys
sparsity, and an overcomplete dictionary has no inverse, so "the coefficients"
are whatever your solver decides.

Fitting a single signal is a convex program once you replace the count of
non-zeros with its $\ell_1$ norm:

$$\min_{\alpha}\; \tfrac12\lVert x - D\alpha\rVert^2 + \lambda\lVert\alpha\rVert_1. \qquad (1)$$

The figure below is the whole reason $\ell_1$ works. The $\ell_1$ ball is a
polytope with corners on the coordinate axes, and the first point where an
expanding level set touches it is generically a vertex — which is a sparse
vector. The $\ell_2$ ball is round and the contact point is dense.

```
        ℓ1 ball (corners ⇒ sparsity)          ℓ2 ball (round ⇒ dense)
             ▲                                      ▲
        ─────┼─────                             ╭───┼───╮
       ╱     │     ╲                           │    │    │
      ╱      │      ╲                          │    │    │
   ───┼──────┼──────┼───                    ───┼────┼────┼───
      ╲      │      ╱                          │    │    │
       ╲     │     ╱                           ╰───┼───╯
             ▼                                      ▼
```

Everything else in this file is a solver for (1), plus one greedy alternative
that is not a solver for (1) at all. The four variants exist because "solve (1)"
has four genuinely different costs: `lasso_sc` is exact and forms a $K\times K$
Gram, `fista_sc` is exact-ish and never forms it, `group_sc` solves a different
program with a block structure, and `omp` abandons convexity for a greedy fit
that is exact on the model it assumes and unpredictable off it.

The one number that governs all of it is the dictionary's **coherence**, the
largest absolute inner product between two distinct atoms. The demo takes a
dictionary whose atoms are all near-orthogonal and recovers a 2-sparse code from
40-dimensional signals every time; then it adds a single atom at cosine $0.994$
to another and support recovery falls to $0.86$, with **every** one of the 27
misses substituting the twin. The signal is still explained to within $0.11$.
Nothing in the output says anything is wrong — that is what a coherent dictionary
does to a sparse code, and it is not a bug in the solver.

## Variants

| Variant | Solves | Cost per signal | Where it wins |
|:--|:--|:--|:--|
| `omp` | not (1): greedy, then least squares | $O(k^2d)$ | tiny $k$; no penalty to tune |
| `lasso_sc` | (1) exactly | $O(K^2d)$ once, $O(K^2)$ per sweep | $K$ in the hundreds |
| `fista_sc` | (1) to tolerance | $O(Kd)$ per step | $K$ in the thousands |
| `group_sc` | (1) with $\ell_{2,1}$ in place of $\ell_1$ | same as FISTA | atoms that come in families |

**`omp`** is orthogonal matching pursuit. At each step take the atom most
correlated with the current residual, add it to the support, and re-solve a least
squares problem on the support so far. On the demo's 20 noiseless signals built
from 2 of 80 atoms it recovers the exact support 20 times out of 20 and leaves
$5\times10^{-16}$ of the signal unexplained. It is not solving (1): with no noise
and an exactly $k$-sparse signal, OMP recovers the sparsest solution, and with
noise it does something for which no objective is being maximised. The
$2\%$-noise case still recovers the support exactly and leaves a residual of
$0.106$, which is the noise, not an error.

**`lasso_sc`** is (1) solved exactly by cyclic coordinate descent. It builds
$G = D^{\mathsf T}D$ once, $K\times K$, and then each coordinate update is a
closed-form soft threshold. On the demo's signal — 4 non-zeros out of 80 atoms —
the sweep over $\lambda$ walks the support $5 \to 5 \to 3 \to 2 \to 0$ and the
residual $0.12 \to 0.48 \to 0.66 \to 1.25 \to 1.28$. Two of those points are
exact, not empirical: at $\lambda = 0$ the residual is zero to $2\times10^{-8}$
(the solver's tolerance) but the code has **all 80 coordinates non-zero** and
$\ell_2$ norm $1.255$ against the minimum-norm solution's $1.030$, and at
$\lambda \ge \lVert D^{\mathsf T}x\rVert_\infty$ the solution is exactly
$\alpha = 0$.

**`fista_sc`** is the same optimum reached from matvecs alone. It reaches
coordinate descent's answer to $3\times10^{-17}$ and is **not** monotone — the
objective rises on 9 of the first 40 steps, because momentum overshoots and the
convergence guarantee is on the smoothed sequence, not on every iterate. Which
solver is faster is purely a question of $K$: at $K = 80$ one coordinate sweep
costs $K^2 = 6400$ against $2Kd = 6400$ for a FISTA step, so they are the same
price and exactness wins; at $K = 1024$, $d = 64$ a sweep is $1.0$M against $131$k,
one sweep buys 8 steps, and at a matched budget of 10.5M multiply-adds FISTA is
$0.12$ lower on the objective.

**`group_sc`** replaces the penalty with $\lambda\sum_g\lVert\alpha_g\rVert_2$,
the sum of group norms. The prox is a block soft threshold — shrink a whole group
by a factor, or zero it entirely — so a family switches on or off together. On
the demo's 30 atoms in 10 families of 3 near-parallel members (mean cosine
$0.92$ inside a family) with a signal built from families 1 and 6, the group
penalty selects exactly those two families and keeps all 6 coefficients; the
lasso at the same $\lambda$ keeps 5 and silently drops atom 4, which costs it
almost nothing because atoms 3 and 5 reproduce it. With one atom per group the
two objectives agree to $2\times10^{-13}$ — the group penalty *is* the $\ell_1$
penalty, plus a decision about what counts as a coordinate.

## Derivation

### 1. Why $\ell_1$ and not $\ell_0$

The honest objective counts non-zeros, $\min \tfrac12\lVert x - D\alpha\rVert^2 +
\lambda\lVert\alpha\rVert_0$, and it is $\mathsf{NP}$-hard with a non-convex
feasible set that no local method can navigate. The relaxation is
$\lVert\alpha\rVert_1 = \sum_j|\alpha_j|$, the largest convex function below the
count on $[-1, 1]$. It is the tightest convex relaxation in a precise sense, and
its geometry is what buys the sparsity: the $\ell_1$ ball is the convex hull of
the signed coordinate vectors, so its extreme points are sparse, and minimising a
smooth function over it lands on faces of low dimension.

Two consequences to keep straight. First, $\ell_1$ sparsity and $\ell_0$
sparsity are different answers: on the demo's own data the $\ell_1$ solution at
$\lambda = 0.05$ keeps 5 atoms where the truth has 4. Second, the relaxation is
exact only under conditions on $D$ — §3 gives the one that is checkable — and the
demo's twin experiment is what failure looks like.

### 2. The optimality conditions, and the two exact limits

(1) is convex and non-smooth at the origin. A minimiser $\alpha^\star$ is
characterised by the subdifferential condition

$$D^{\mathsf T}(x - D\alpha^\star) \in \lambda\,\partial\lVert\alpha^\star\rVert_1,
\qquad \partial\lVert\alpha\rVert_1 = \{g : g_j = \mathrm{sign}(\alpha_j)
\ \text{if}\ \alpha_j \ne 0,\ \ |g_j| \le 1\ \text{otherwise}\}. \qquad (2)$$

Read coordinate by coordinate, with $G = D^{\mathsf T}D$ and
$b = D^{\mathsf T}x$:

$$\alpha_j \ne 0 \;\Rightarrow\; (G\alpha)_j - b_j = -\lambda\,\mathrm{sign}(\alpha_j),
\qquad \alpha_j = 0 \;\Rightarrow\; |(G\alpha)_j - b_j| \le \lambda. \qquad (3)$$

Two exact statements fall out, and the demo asserts both.

**$\lambda = 0$ gives an exact representation.** Then (3) says
$G\alpha = b$, i.e. $D^{\mathsf T}(x - D\alpha) = 0$: the residual is orthogonal
to every atom, which for $K > d$ is a system with infinitely many solutions — a
$K$-dimensional affine space of them, $K - d$ of whose dimensions are free. The
residual is zero and the coefficients are one point in that space, selected by
the solver's update order. The demo's $\lambda = 0$ fit has all $80$ coordinates
non-zero, $\ell_2$ norm $1.255$, against the minimum-norm solution $D^{+}x$ at
$1.030$: coordinate descent from a zero start is nowhere near a sparse answer,
and nothing in the output says so.

**$\lambda \ge \lVert b\rVert_\infty$ gives $\alpha = 0$.** With $\alpha = 0$
the condition is $|b_j| \le \lambda$ for every $j$, so it holds exactly when
$\lambda$ is at least the largest correlation any atom has with the signal. The
demo computes $\lVert D^{\mathsf T}x\rVert_\infty = 0.741$ and fits at
$0.741 \times 1.001$, getting exact zeros rather than small numbers.

### 3. Coordinate descent, and why it is exact here

Fixing every coefficient except $j$, (1) is a one-dimensional problem and its
minimiser is a soft threshold:

$$\alpha_j \leftarrow \frac{1}{G_{jj}}\,
\mathcal{S}\!\left(b_j - \sum_{k\ne j} G_{jk}\alpha_k,\ \lambda\right),
\qquad \mathcal{S}(z,\lambda) = \mathrm{sign}(z)\max(|z| - \lambda, 0). \qquad (4)$$

This is (3) solved for $\alpha_j$: the partial residual scaled by the curvature of
that coordinate. Each sweep is $O(K^2)$, the iterates are monotone in objective
because each coordinate step minimises the exact objective along a coordinate
line, and convergence to the unique minimiser (for $\lambda > 0$) is linear. The
demo's 80-atom problem converges in 22 sweeps to tolerance; the large one needs
tens more.

The one thing that makes this a *cheap* method rather than a nice one is that
$G$ is formed once. The alternative — recomputing $D\alpha$ and $D^{\mathsf
T}(\cdot)$ per coordinate — is the same arithmetic at $K$ times the cost, and it
is the standard first implementation.

### 4. Why the Gram is the problem

$G$ is $K\times K$. For $K = 1024$ that is $8$ MB and $K^2d = 6.7\times10^7$
multiply-adds to build; for $K = 10^5$ it is 80 GB and the method is unavailable,
not merely slow. FISTA avoids it by touching only $D$ and $D^{\mathsf T}$, and
pays a rate.

Split (1) into a smooth part $f(\alpha) = \tfrac12\lVert x - D\alpha\rVert^2$ and
a non-smooth part $g(\alpha) = \lambda\lVert\alpha\rVert_1$. The proximal
gradient step is

$$\alpha_{k+1} = \mathrm{prox}_{g/L}\!\left(\alpha_k - \tfrac{1}{L}\nabla
f(\alpha_k)\right), \qquad L = \lVert D\rVert_2^2, \qquad (5)$$

and the prox of $\lambda\lVert\cdot\rVert_1$ *is* the soft threshold of (4),
applied elementwise with threshold $\lambda/L$. $L$ is the smallest step that
makes the step a contraction; anything smaller converges, anything larger
diverges, so it is the one hyper-parameter and it has a formula. FISTA adds
Nesterov momentum,

$$y_{k+1} = \alpha_{k+1} + \frac{t_k - 1}{t_{k+1}}(\alpha_{k+1} - \alpha_k),
\qquad t_{k+1} = \tfrac12\left(1 + \sqrt{1 + 4t_k^2}\right), \qquad (6)$$

and the step is taken from $y$ rather than $\alpha$. The objective bound improves
from $O(1/k)$ to $O(1/k^2)$, which is the entire difference between the two
solver rows in the table.

Momentum is also why FISTA's objective is not monotone: the bound is on
$f(\alpha_k) + g(\alpha_k) - f^\star$ being $O(1/k^2)$, not on every step being a
descent step. The demo measures 9 increases in the first 40 steps. A monotone
variant (MFISTA, or adaptive restart) fixes this by rejecting a step that
increases the objective, and it is what you use when the iterates are being
watched rather than just the final point.

### 5. Group sparsity

Atoms often arrive in families — three shifted copies of a wavelet, the one-hot
columns of a categorical variable, the RGB of a patch. The penalty that encodes
"a family is on or off" is the sum of group norms,

$$\min_\alpha\; \tfrac12\lVert x - D\alpha\rVert^2 +
\lambda\sum_{g}\lVert\alpha_g\rVert_2, \qquad (7)$$

with prox

$$\mathrm{prox}_{t\lVert\cdot\rVert_2}(v) = \max\!\left(0,\ 1 -
\frac{t}{\lVert v\rVert_2}\right)v, \qquad (8)$$

a *block* soft threshold: the whole vector is either zeroed or scaled towards
zero along its own direction. There is no coordinate-wise thinning, which is the
modelling statement — within a family, the relative sizes are never questioned.

Two facts make (7) more than a variant. With singleton groups, $\lVert\alpha_g
\rVert_2 = |\alpha_g|$ and (7) *is* (1); the demo asserts the two objectives
agree to $2\times10^{-13}$. And the group penalty is invariant to rotations
*within* a group, so its answer does not depend on how a redundant family is
parameterised, which is exactly what the lasso's answer does depend on: in the
demo it drops one of three atoms at cosine $0.92$ to the others and there is no
principled reason it is atom 4 rather than atom 3.

### 6. Greedy: OMP, and the condition it needs

OMP picks $j_k = \arg\max_j |d_j^{\mathsf T}r_{k-1}|$ among unused atoms, then
sets $\alpha_S = (D_S^{\mathsf T}D_S)^{-1}D_S^{\mathsf T}x$ on the support and
recomputes the residual. It is exact on the model — a $k$-sparse signal, no noise
— provided the dictionary is incoherent enough.

The checkable condition is in terms of the coherence

$$\mu(D) = \max_{i\ne j} |d_i^{\mathsf T}d_j|. \qquad (9)$$

If $k < \tfrac12(1 + 1/\mu)$ then a $k$-sparse representation is the *unique*
sparsest one and OMP finds it; basis pursuit finds it too. For the demo's random
$40\times80$ dictionary $\mu \approx 0.55$ gives a bound of $k < 1.4$, and the
experiment runs at $k = 2$ and succeeds anyway — the bound is sufficient, not
necessary. What breaks OMP is not a threshold but an ambiguity: two atoms at
cosine $0.994$ are interchangeable, so the greedy choice between them is a coin
flip that the objective cannot see. All 27 failures in the demo are that coin
flip, and the residual afterwards is $0.11$ because the substitute atom explains
almost the same signal.

The standard fix is a *dictionary* fix, not a solver fix: merge the twins, or
penalise the pair jointly with `group_sc`. This is why dictionary learning keeps
atoms apart explicitly.

### 7. Identification, or what "the" sparse code means

With $K > d$ and $\lambda = 0$ the solution set is the affine space
$\{\alpha : D\alpha = x\}$, of dimension $K - d$. Every point in it is a perfect
fit, and the objective cannot distinguish them. As $\lambda$ grows the solution
becomes unique (strict convexity for $\lambda > 0$), and the path is continuous
and piecewise linear in $\lambda$.

The sentence to carry away: a sparse code is a statement about **atoms**, not
about the signal. Change the dictionary and the code changes, including its
support, and no amount of residual-checking detects it. That is why the twin
experiment is in the demo, and why `dictionary_learning` has to exist.

## Code walkthrough

`SparseCoding(nnz=5, *, variant="lasso_sc", lam=0.1, n_groups=4, n_iter=1000,
tol=1e-8)` with `fit(X, D)`, `transform`, `fit_transform`, `reconstruct`, and the
attributes `codes_` $(n, K)$, `dictionary_` $(d, K)$, `groups_`, `n_iter_`.

`fit` normalises the dictionary once, then dispatches one signal at a time. The
object is a solver: it stores no learned statistics, and fitting a new dictionary
means fitting again.

### 1. Normalisation, once per dictionary

```python
    nrm = np.linalg.norm(D, axis=0)
    if np.any(nrm < EPS):
        raise ValueError("a dictionary atom is all zeros — drop it or replace it")
    return D / nrm
```

Every penalty in this file is a threshold on coefficients, so it is only
comparable across atoms of equal norm. A dictionary handed in with columns of
wildly different scale produces a code that is sparse in *number* and not in
*information*: the short atoms get penalised 10× as hard as the long ones and are
dropped arithmetically. Normalising here rather than asking the caller to do it
means `lam` means one thing everywhere in the file.

The zero-atom check is not politeness. `G_jj = 0` makes the coordinate update in
(4) a division by zero, which produces an inf, which quietly poisons the whole
vector and then the whole batch of codes.

### 2. The objective everyone shares

```python
    fit = 0.5 * float(np.sum((x - D @ a) ** 2))
    if groups is None:
        return fit + lam * float(np.abs(a).sum())
    return fit + lam * float(sum(np.linalg.norm(a[g]) for g in groups))
```

Written in terms of the residual, not in terms of the Gram, so the same function
scores a code produced by any of the four variants — including `omp`, which never
builds a Gram. The demo uses it to compare `group_sc` against `lasso_sc` at
$2\times10^{-13}$ on singleton groups and to compare FISTA against coordinate
descent to $3\times10^{-17}$, which are the two checks that keep four solvers
honest about solving anything at all.

### 3. Greedy selection and the least-squares refit

```python
    for _ in range(min(nnz, D.shape[1])):
        corr = np.abs(D.T @ res)
        corr[support] = -1.0                             # never re-select an atom
        support.append(int(np.argmax(corr)))
        sol, *_ = np.linalg.lstsq(D[:, support], x, rcond=None)
        res = x - D[:, support] @ sol
```

The selection step of §6 with one detail that matters: `corr[support] = -1.0`
rather than skipping the index. Without it a correlated dictionary can select the
same atom twice, the support matrix becomes rank-deficient, and `lstsq` returns a
*solution* anyway, so the loop terminates with a coefficient vector of the right
shape and the wrong content.

`np.linalg.lstsq` rather than an explicit inverse of $D_S^{\mathsf T}D_S$: for a
coherent dictionary the Gram of the support is ill-conditioned, and a
pseudo-inverse minimum-norm answer is a much better thing to return than a
blow-up. It is also the one place `solver=` would matter: `lstsq` on a $40\times4$
system is dominated by the SVD, and a Cholesky solve on the normal equations is
faster and less stable.

### 4. The coordinate update

```python
    G, b = D.T @ D, D.T @ x
    d = np.diag(G).copy()
    a = np.zeros(G.shape[0])
    used = n_iter
    for it in range(n_iter):
        a_old = a.copy()
        for j in range(G.shape[0]):
            partial = b[j] - (G[j] @ a - d[j] * a[j])
            a[j] = _soft(partial, lam) / d[j]
```

(4), with the partial residual written as `b[j] - (G[j] @ a - d[j] * a[j])`,
because `G[j] @ a` already contains the $j$-th term and subtracting it is cheaper
than forming the sum over $k \ne j$. `d = np.diag(G).copy()` is hoisted: the
diagonal is constant and, after normalisation, positive.

`a_old = a.copy()` then a max-absolute-difference test rather than a norm test:
after a few sweeps only a handful of coordinates move, and the max difference is
the quantity that converges linearly. The relative tolerance divides by the
largest coefficient so that a signal with amplitude 1000 and one with amplitude
$10^{-3}$ converge at the same relative accuracy.

### 5. The step size, by power iteration

```python
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(D.shape[1])
    v /= np.linalg.norm(v)
    top = 0.0
    for _ in range(n_iter):
        u = D.T @ (D @ v)
        top = float(v @ u)
        nrm = float(np.linalg.norm(u))
```

$\lVert D\rVert_2^2$ is the Lipschitz constant of $\nabla f$ and therefore the
step size in (5). The obvious way to get it is `eigvalsh(D.T @ D)[-1]`, which
costs $O(K^2 d + K^3)$ and forms the very matrix FISTA exists to avoid. Power
iteration costs 60 matvecs — four orders of magnitude less at $K = 10^4$ — and is
accurate here because the eigenvalue ratio of a random dictionary's Gram is far
from 1.

The `rng` is seeded, so the step size is deterministic across runs. It is a
stochastic *estimate* of a fixed quantity, and an unseeded one would make two
runs of the demo produce different iteration counts, which is exactly the kind of
non-determinism the repository's convention forbids.

### 6. The momentum loop

```python
        a_new = _soft(y - (D.T @ (D @ y - x)) / L, lam / L)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = a_new + ((t - 1.0) / t_new) * (a_new - a)
        step = float(np.max(np.abs(a_new - a)))
        a, t = a_new, t_new
        if step <= tol * max(1.0, float(np.max(np.abs(a)))):
            used = it + 1
            break
```

(5) and (6) in five lines. `D.T @ (D @ y - x)` is the gradient in two matvecs and
never forms $D^{\mathsf T}D$; `lam / L` folds the step size into the threshold so
the prox is a plain `_soft`.

The subtle part is the convergence test: it measures the movement of
$\alpha$, not the size of the gradient, because the gradient at the optimum is
`Dᵀ(x − Dα)` which vanishes only in the $\lambda = 0$ case. For $\lambda > 0$ the
gradient at the optimum is exactly $\pm\lambda$ on the support, so a gradient
test would need to know the support. The coefficient movement is the right proxy
and it is what the monotone variants test too.

### 7. The block threshold

```python
    nrm = float(np.linalg.norm(v))
    if nrm < EPS:
        return np.zeros_like(v)
    return v * max(0.0, 1.0 - t / nrm)
```

(8), and the difference between it and `_soft` is the whole variant: `_soft`
scales each coefficient by a different factor, this scales a whole block by one
factor along its own direction. The `nrm < EPS` branch handles an exactly zero
block, where $t/\lVert v\rVert$ is infinite and the `max` would return `nan`.

Note what is *not* here: no normalisation by group size. A group of 10 and a
group of 2 are penalised by the same $\lambda$ applied to their norms, so the
larger group is easier to keep switched on. Group-size weighting
($\lambda\sqrt{|g|}$) is the standard fix and it changes the model, which is why
it is not silently built in.

### 8. The group solver

```python
    for it in range(n_iter):
        trial = a - (D.T @ (D @ a - x)) / L
        a_new = trial.copy()
        for g in groups:
            a_new[g] = _block_soft(trial[g], lam / L)
        step = float(np.max(np.abs(a_new - a)))
        a = a_new
```

Plain ISTA with the block prox, no momentum. The `trial` vector is computed once
and every group is thresholded from it; writing `a_new[g] = _block_soft(a_new[g],
...)` in the loop instead would compound the shrinkage across groups, since each
update would then read a partially-updated vector. That bug produces a solution
that is sparse, converges, and is wrong by a factor that depends on the group
order — which is the definition of a hard bug.

### 9. The dispatch

```python
        if self.variant == "omp":
            return _omp(D, x, self.nnz, self.tol)
        if self.variant == "lasso_sc":
            return _lasso_cd(D, x, self.lam, self.n_iter, self.tol)
        if self.variant == "fista_sc":
            return _fista(D, x, self.lam, self.n_iter, self.tol)
        return _group_pg(D, x, self.groups_, self.lam, self.n_iter, self.tol)
```

One signal at a time, because that is the only shape all four solvers share:
coordinate descent would batch naturally (it is the same Gram for every signal),
and this is the one place the file gives up a factor of $n$ for uniformity. Each
`_one` returns the code and the inner-iteration count, and `fit` keeps the
maximum over the batch in `n_iter_` so that a fit which hit the iteration cap is
visible rather than silently truncated.

### 10. The demo's generator

```python
    a = np.zeros(K)
    a[rng.choice(K, nnz, replace=False)] = np.linspace(1.0, 0.4, nnz)
    return D, D @ a, a
```

A signal from 4 atoms with coefficients from 1.0 down to 0.4. The decay matters:
with equal coefficients the $\lambda$ path drops all four at once and the
staircase the demo reports disappears. A graded signal is what makes "5 → 5 → 3 →
2 → 0" a measurement of the path rather than of a threshold.

## Complexity

$d$ ambient dimension, $K$ atoms, $k$ the greedy support size, $T$ iterations.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_normalise` | $O(dK)$ | $O(dK)$ | once per dictionary |
| `_omp` | $O(k^2d)$ | $O(kd)$ | the support least squares dominates |
| `DᵀD` in `_lasso_cd` | $O(K^2d)$ | $O(K^2)$ | the row that rules out large $K$ |
| one CD sweep | $O(K^2)$ | $O(K)$ | 22 sweeps here at $K = 80$ |
| `_lipschitz` | $O(60Kd)$ | $O(K)$ | 60 matvecs, replaces an $O(K^3)$ eigendecomposition |
| one FISTA step | $O(Kd)$ | $O(K)$ | no Gram |
| one group step | $O(Kd + K)$ | $O(K)$ | the prox is $O(K)$ in total |
| `fit`, 20 signals, $d{=}40$, $K{=}80$ | — | — | 2 ms `omp`, 46 ms `fista`, 56 ms `lasso`, 113 ms `group` |

The single row that matters is `DᵀD` against one FISTA step: $K^2d$ versus $Kd$,
a factor of $K$. At $K = 1024$ the measured wall-clock is 105 ms for coordinate
descent and 17 ms for FISTA on the same signal to the same tolerance — the 6×
that survives the constant factors. At $K = 80$ the two are within noise of one
another and `lasso_sc` is the better choice because it is exact, and at $K = 10$
neither matters.

Memory is the constraint that arrives first, and it is quadratic in `lasso_sc`:
$K = 10^5$ is 80 GB for the Gram. FISTA's footprint is $O(K)$, which is what
makes it the default in the dictionary-learning literature, where $K$ is chosen
to be large.

## Failure modes

**1. `lam = 0` returns an arbitrary point of an affine space.** §2: residual zero,
coefficients chosen by the coordinate order. The demo's $\lambda = 0$ fit is
*dense* — 80 non-zeros, norm $1.255$ against the minimum-norm $1.030$ — and a
different update order gives a different vector with the same zero residual. If a
pipeline uses `lam=0` "to get the exact fit", it is reading a least-squares
solution and calling it sparse.

**2. A coherent dictionary makes the support unidentifiable, silently.** The twin
experiment: cosine $0.994$, recovery $0.86$, all misses substituting the twin,
signal still explained to $0.11$. There is no residual check that catches this,
because both codes are equally good fits. The diagnostic is
$\mu(D)$ — compute it — and the remedy is a different dictionary or a group
penalty over the collinear set.

**3. $\ell_1$ is not $\ell_0$, and it is biased.** At $\lambda = 0.05$ on data
whose truth is 4-sparse, the solution has 5 atoms. The extra atom is not noise:
$\ell_1$ shrinks all coefficients towards zero, so the large ones are
underestimated as well, a bias that debiasing (refit by least squares on the
support) removes at the cost of no longer being the solution of (1).

**4. $\lambda$ has no scale-free setting.** Its useful range is
$[0, \lVert D^{\mathsf T}x\rVert_\infty]$, which depends on the signal's norm,
the dictionary's norm and the noise. `lam=0.05` on unit-normalised atoms and
unit-scale signals is a different model from `lam=0.05` on data scaled by 1000.
The only portable parameterisation is $\lambda/\lVert D^{\mathsf T}x\rVert_\infty$.

**5. FISTA's objective is not monotone, and its iterates can look divergent.**
9 rises in the first 40 steps on the demo's own problem. Iterating until the
objective stops improving therefore stops at the wrong place, and monitoring the
objective to detect a bug will detect a real property instead. Monitor the
coefficient movement, or use a monotone variant.

**6. `K²` is a wall, not a slowdown.** Coordinate descent at $K = 10^5$ is not
slow, it is unavailable. Code that "works" on the demo's 80 atoms and is pointed
at an 80000-atom dictionary will allocate 51 GB and die — inside `D.T @ D`, a
long way from the line that chose the solver.

**7. Group structure is a claim about the dictionary, and a wrong claim is
invisible.** Grouping contiguous columns assumes the atoms were ordered
meaningfully. With a permuted dictionary the same code clusters noise, and the
result is a sparse, stable, reproducible answer to a question nobody asked. The
demo's families are contiguous by construction and shuffled in practice.

**8. OMP's budget is not a sparsity guarantee.** `nnz=1` on a 2-sparse signal
leaves $0.45$ of it unexplained and no error; `nnz=10` on a 2-sparse signal
returns 10 atoms that fit noise. The budget has to come from the problem's
physics or from cross-validation, not from a default.

**9. Every solver here assumes the dictionary is right.** A bad atom is fitted as
faithfully as a good one, and the residual is the only warning — which is zero
when $K > d$ and $\lambda = 0$, and small whenever the dictionary spans the
signal. That assumption is exactly what `dictionary_learning` relaxes.

## Applications

- **Image and audio compression.** JPEG's DCT and MP3's filterbank are fixed
  dictionaries; matching pursuit over an overcomplete one (wavelets plus
  sinusoids plus transients) is the standard for sparse audio coding, and it is
  where `omp` earns its keep: a few dozen atoms per second of signal, chosen
  greedily, is fast enough to be a codec.
- **Denoising and inpainting.** Fit (1) to a corrupted signal, keep
  `reconstruct()`. The reason it works is §2's optimality condition: the residual
  of a sparse fit is uncorrelated with every atom, so it looks like noise, and a
  structured corruption does not.
- **Compressed sensing.** If $\alpha$ is $k$-sparse and $D$ obeys (9), then
  $m \gtrsim k\log(K/k)$ measurements determine $x$ exactly, and (1) is the
  recovery algorithm. This is the application that made the $\ell_1$ relaxation
  famous, and the whole theory is a statement about $\mu$ or its cousins.
- **Feature selection as a special case.** With $D = X$ and $x = y$, (1) is the
  lasso, and `sparse_cca`'s inner loop is this file's `_lasso_cd` verbatim. The
  same solver, a different story about what the columns are.
- **Hierarchical and structured penalties.** Gene pathways, wavelet trees,
  multi-task rows (the $\ell_{2,1}$ norm applied along the wrong axis): every one
  of these is (7) with a different group definition, and the block prox of (8) is
  the only thing that changes.
- **A building block for sparse PCA and dictionary learning.** Sparse PCA is
  (1) inside an eigenvector iteration; `dictionary_learning` alternates (1) with a
  dictionary update. Both call exactly this solver, which is why it is an atom
  and not a step inside one.

Where it is a trap: reading a support as a set of "true" variables when the
dictionary is coherent; using `lam=0` and calling the result sparse; choosing
$\lambda$ on the training residual; and pointing `lasso_sc` at a dictionary with
$10^5$ atoms and expecting a slowdown rather than a memory error.

## References

- **Chen, Donoho & Saunders (1998),** *Atomic decomposition by basis pursuit* —
  the paper that named the $\ell_1$ relaxation and made the case for it over
  greedy pursuit. (1) with $D$ a general overcomplete dictionary.
- **Tibshirani (1996),** *Regression shrinkage and selection via the lasso* — the
  same program in regression form, including the soft-threshold reading of the
  optimality conditions in §2.
- **Mallat & Zhang (1993),** *Matching pursuits with time-frequency dictionaries*
  — matching pursuit, of which `omp` is the least-squares variant; the source of
  the greedy view.
- **Pati, Rezaiifar & Krishnaprasad (1993),** *Orthogonal matching pursuit* — the
  refit step that makes OMP exact on the $k$-sparse model.
- **Donoho & Elad (2003),** *Optimally sparse representation in general
  (non-orthogonal) dictionaries via $\ell_1$ minimization* — the coherence
  condition (9) and the uniqueness argument behind it.
- **Tropp (2004),** *Greed is good: algorithmic results for sparse approximation*
  — the OMP guarantee under coherence, and the honest statement of when greedy
  matches basis pursuit.
- **Daubechies, Defrise & De Mol (2004),** *An iterative thresholding algorithm
  for linear inverse problems with a sparsity constraint* — ISTA, the
  proximal-gradient step of (5).
- **Beck & Teboulle (2009),** *A fast iterative shrinkage-thresholding algorithm
  for linear inverse problems* — FISTA and the $O(1/k^2)$ rate of (6).
- **Friedman, Hastie & Tibshirani (2010),** *Regularization paths for generalized
  linear models via coordinate descent* — the coordinate update of §3 as it is
  actually implemented, including the active-set trick that makes large-$K$
  paths affordable.
- **Yuan & Lin (2006),** *Model selection and estimation in regression with
  grouped variables* — the group lasso of (7) and the block prox of (8).
- **Bach, Jenatton, Mairal & Obozinski (2012),** *Optimization with sparsity
  inducing penalties* — the survey that unifies $\ell_1$, $\ell_{2,1}$ and the
  proximal view; the right reference when the penalty is not in this file.
- **Elad (2010),** *Sparse and Redundant Representations* — the book-length
  treatment of everything above, including the dictionary-learning loop this
  atom feeds.
