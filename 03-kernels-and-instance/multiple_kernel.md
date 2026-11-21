# Multiple kernel learning — combine kernels and learn the weights

`multiple_kernel.py` · Kernels and instance · Tier 3 · 4 variants

## Intuition

A kernel is a prior about similarity. On one dataset the right prior is smooth,
on another it is polynomial, and on a third the labels live in two columns while
six columns are noise. Choosing one kernel by cross-validation throws away the
fact that different kernels see different structure — and on a single split the
choice is noisy.

Multiple kernel learning keeps all of them and learns a convex combination:

$$
K(w)=\sum_{k=1}^{q} w_k K_k,\qquad
w\in\Delta_q=\Big\{w:\ w_k\ge 0,\ \sum_k w_k=1\Big\}.
\qquad (1)
$$

The constraint is not decoration. A non-negative sum of PSD matrices is PSD, so
$K(w)$ is a kernel for every $w\in\Delta_q$; the simplex makes the weights a
budget, one unit of similarity divided among $q$ experts, and keeps the problem
bounded. Learn $w$ on training data, then train one ordinary SVM — in fact the
two happen together, because the SVM's dual is the only place the labels enter.

The motivating failure is in the demo. Eight kernels are built from a 2-d
XOR problem buried in six noise axes: two kernels see only the signal axes, two
see only the noise, four see everything and drown in the noise. Alone, the best
kernel gets 0.9435 and the noise-only kernels are at chance. Uniform averaging
gets 0.9155 — adding seven kernels made things worse. Learned weights recover
most of the gap, and how they do it is the subject of this file.

## Variants

| variant | weights | optimises | SVM solves |
|:--|:--|:--|--:|
| `average` | $1/q$ each | nothing | 1 |
| `alignment` | $\propto\max(A_k,0)$ | kernel–target similarity | 1 |
| `mkl_svm` | projected gradient | the SVM dual itself | $T$, one per step |
| `easy_mkl` | quadratic pair mass | an $\ell_2$-relaxed margin | 1 |

**`average`** is the baseline that people forget to run. With many informative
kernels it is close to optimal and it cannot overfit the weights, because there
are none. It is also the only variant whose cost is one SVM.

**`alignment`** scores each kernel by how well its centred Gram matrix points in
the same direction as the centred label matrix $yy^{\top}$, then normalises the
positive scores. It is a one-shot heuristic: no SVM is trained to choose $w$, so
it is the cheapest learned variant, and it is the one that can be fooled by a
high-capacity kernel that mimics $yy^{\top}$ on the training points.

**`mkl_svm`** takes the SVM's own value function seriously. For fixed $w$ the
dual gives a number; the outer problem is to make that number as small as
possible over the simplex. Each step needs one SVM solve to score a candidate
$w$, so it is the expensive variant and the one that provably descends. On
$\ell_1$ MKL it usually ends near a vertex: a few kernels, the rest at zero.

**`easy_mkl`** replaces the max-margin inner problem with a quadratic one over a
distribution on training pairs. That makes the weight step a projected gradient
on a quadratic with no SVM in the loop, which is what makes it scale. Its weights
are dense — every kernel keeps some mass — and it is the variant to prefer when
$q$ is large.

Two variants that are *not* here: the semidefinite program of Lanckriet et al.,
which is the historical formulation but needs an SDP solver, and multiclass MKL,
which needs one weight vector per binary machine. Both are honest exclusions
rather than gaps.

## Derivation

### Scale first

Kernels have units. An RBF with $\gamma=100$ has diagonal one, a polynomial of
degree three on unnormalised features has diagonal in the thousands, and a
linear kernel on six noisy axes has diagonal equal to the noise variance times
six. Averaging those is averaging numbers with different units, and the largest
one wins. So each block is divided by its training mean diagonal:

$$
\tilde K_k=\frac{K_k}{s_k},\qquad s_k=\frac1n\sum_{i=1}^n (K_k)_{ii}.
\qquad (2)
$$

The same $s_k$ is applied to test blocks; recomputing it from the test diagonal
would make the train and test kernels disagree. This is the one preprocessing
step that changes the answer by more than the choice of variant.

### The SVM on a combined kernel

With labels coded as $y\in\{-1,+1\}^{n}$ and $Y=\mathrm{diag}(y)$, the dual of
the C-SVM on $K(w)$ is

$$
\max_{\alpha}\ \mathbf{1}^{\top}\alpha-\tfrac12\alpha^{\top}
\big(YK(w)Y\big)\alpha
\quad\text{s.t.}\quad 0\le\alpha_i\le C,\ \ y^{\top}\alpha=0 .
\qquad (3)
$$

The solver in this file is pairwise coordinate ascent: it picks the pair that
violates KKT most, moves $\alpha_i$ up and $\alpha_j$ down along
$y_i\Delta\alpha_i+y_j\Delta\alpha_j=0$, and clips the step so both stay in
their boxes. Because $K(w)$ is linear in $w$, nothing about the solver changes
when $w$ does; only the matrix it reads.

### The outer problem for `mkl_svm`

Define the value of the inner problem as a function of the weights:

$$
J(w)=\max_{\alpha\in\mathcal{A}}\ \mathbf{1}^{\top}\alpha
-\tfrac12\sum_k w_k\,\alpha^{\top}\big(YK_kY\big)\alpha .
\qquad (4)
$$

$J$ is a maximum of affine functions of $w$, so it is convex, and the outer
problem is $\min_{w\in\Delta_q}J(w)$. Danskin's theorem gives the gradient at the
current maximiser $\alpha^\star$:

$$
\frac{\partial J}{\partial w_k}=-\tfrac12 q_k,\qquad
q_k=(\alpha^\star\odot y)^{\top}K_k(\alpha^\star\odot y).
\qquad (5)
$$

So the descent direction is $+q/2$, and one projected-gradient step is
$w\leftarrow\Pi_\Delta(w+\eta q/2)$. The projection is the Euclidean one:

$$
\Pi_\Delta(v)=\max(v-\theta,0),\qquad
\theta\ \text{solves}\ \sum_k\max(v_k-\theta,0)=1,
\qquad (6)
$$

found by sorting $v$ once and testing which prefix is active. The step size
starts at $2/\max_k q_k$ and is halved until $J$ decreases, then inflated by
$1.5$ after each success — a standard backtracking line search, and the reason a
step costs a full SVM solve.

The gradient is worth reading twice. $q_k$ is the squared norm of the SVM's
weight vector in feature space $k$, so the update pushes mass towards kernels on
which the current classifier has large margin volume. That is the training
criterion, not a validation criterion, and it is exactly why `mkl_svm` will buy
a kernel that memorises.

### Centred alignment

For `alignment`, let $H=I-\mathbf{1}\mathbf{1}^{\top}/n$ be the centring matrix.
The centred kernel-target alignment is a cosine between matrices:

$$
A(K,yy^{\top})=
\frac{\langle HKH,\ Hyy^{\top}H\rangle_F}
{\lVert HKH\rVert_F\,\lVert Hyy^{\top}H\rVert_F}\in[-1,1],
\qquad (7)
$$

and the weights are the positive part, renormalised:

$$
w_k=\frac{\max(A_k,0)}{\sum_l\max(A_l,0)} .
\qquad (8)
$$

Centring removes the constant part of the kernel, the part that cannot carry
label information; without it every kernel with a large mean diagonal looks
aligned. For balanced classes $Hy=y$, so (7) reduces to
$\langle HKH,yy^{\top}\rangle_F$ over the norms: $A=1$ exactly for
$K=yy^{\top}$ and $A=0$ for $K=\mathbf{1}\mathbf{1}^{\top}$, both asserted in the
demo. `lin/all` and a degree-one polynomial with `coef0=1` differ only by the
constant 1, so (7) gives them the same alignment to four decimals and (8) the
same weight: an affine shift of a kernel is invisible to centring, which is the
invariance the centring was for. If every alignment is non-positive the code
falls back to uniform weights, because the alternative is a division by zero.

### EasyMKL's quadratic

`easy_mkl` swaps the max-margin inner problem for a quadratic over a
distribution $\gamma$ on the training points, with one simplex block per class:

$$
\min_{\gamma}\ \gamma^{\top}\Big(Y\bar KY+\lambda I\Big)\gamma,
\qquad \gamma\ge0,\ \ \sum_{i:y_i=+1}\gamma_i=
\sum_{i:y_i=-1}\gamma_i=1,
\qquad (9)
$$

with $\bar K=\frac1q\sum_kK_k$ and $\lambda$ a ridge that keeps the quadratic
strictly convex. The optimum is the mass the method places on the hardest
same-class pairs; the kernel weights are the same quadratic evaluated per block:

$$
\beta_k=\frac{\max\big(\gamma^{\top}YK_kY\gamma,\,0\big)}
{\sum_l\max\big(\gamma^{\top}YK_lY\gamma,\,0\big)} .
\qquad (10)
$$

The gradient of (9) is $2(Y\bar KY+\lambda I)\gamma$, so the Lipschitz constant
is $2\lambda_{\max}$, estimated with 30 power iterations on the matrix-free
operator $\gamma\mapsto y\odot(\bar K(y\odot\gamma))+\lambda\gamma$. Per-class
projection is applied after each step, and the loop stops when the largest
coordinate move falls below `tol`. In the demo that takes 572 steps and drives
the quadratic from 0.0863 to 0.0315.

### The classifier

Once $w$ is fixed the classifier is an ordinary SVM on $K(w)$, and the KKT
conditions of (3) are the check that the solver converged:

$$
\alpha_i=0\Rightarrow y_if_i\ge1,\qquad
0<\alpha_i<C\Rightarrow y_if_i=1,\qquad
\alpha_i=C\Rightarrow y_if_i\le1 .
\qquad (11)
$$

The demo asserts all three on the fitted model. The offset $b$ is the mean of
$y_i-f_i$ over the free support vectors, which is the only place where the
equality case is available.

## Code walkthrough

`kernel_matrices` builds the blocks; every learner below consumes them and
never sees the features again. The dispatch keeps `columns`, which is how the
demo constructs a "signal axes only" expert:

```python
    kind = spec.get("kind", "rbf")
    if kind not in KINDS:
        raise ValueError(f"unknown kernel kind {kind!r}; expected one of {KINDS}")
    columns = spec.get("columns")
    if columns is not None:
        columns = np.asarray(columns, dtype=int)
        A, B = A[:, columns], B[:, columns]
    gamma = float(spec.get("gamma", 1.0 / A.shape[1]))
```

The non-obvious line is the default bandwidth: `1 / A.shape[1]` is computed after
column selection, so a two-column expert gets $\gamma=0.5$ rather than
$1/8$. A default taken from the raw width would make every expert too flat.

`_svm_fit` is the inner solver of (3). The pair update is three lines, and the
third is the one that makes it $O(n)$ per step instead of $O(n^2)$:

```python
        alpha[i] += y[i] * step
        alpha[j] -= y[j] * step
        f += step * (K[i] - K[j])
        grad = y * f - 1.0
```

Because $f=K(\alpha\odot y)$ and only two coordinates change, the change in $f$
is $\Delta\alpha_i y_i K_i+\Delta\alpha_jy_jK_j$, which collapses to
$\text{step}\cdot(K_i-K_j)$ by the equality constraint. Recomputing `K @ (alpha*y)`
would give the same answer for $n$ times the work.

Weights live on the simplex, so every update needs the projection of (6):

```python
    u = np.sort(v)[::-1]
    total = np.cumsum(u) - 1.0
    index = np.arange(1, v.size + 1)
    active = u - total / index > 0.0
    rho = index[active][-1]
    theta = total[active][-1] / rho
    return np.maximum(v - theta, 0.0)
```

The last active prefix gives $\theta$ in one pass. Taking the *last* rather than
the first active index is what handles ties and the degenerate vector that
projects onto a vertex.

Alignment is four lines of linear algebra plus a guard, and the guard matters:
a constant kernel has zero centred norm, and the naive ratio would be `nan`.

```python
    Kc = H @ K @ H
    Tc = H @ target @ H
    denominator = np.linalg.norm(Kc) * np.linalg.norm(Tc)
    if denominator < EPS:
        return 0.0
    return float(np.sum(Kc * Tc) / denominator)
```

`fit` is where scaling happens, and the two blocks after it are the whole
learner: normalise, learn weights, form the combination, solve once.

```python
        self.scales_ = np.array(scales if self.normalize else np.ones(n_blocks))
...
        blocks = [K / s for K, s in zip(blocks, self.scales_, strict=True)]
...
        self.weights_ = weights
        self.combined_ = sum(w * K for w, K
                             in zip(weights, blocks, strict=True))
        self.alpha_, self.intercept_, self.solve_iter_ = _svm_fit(
            self.combined_, codes, self.C, tol=self.tol)
```

`combined_` is stored rather than recomputed, because it is the object every
diagnostic wants: the KKT check, the trace, the alignment. Note that it holds
the *normalised* combination; `scales_` is what converts back.

`_fit_mkl_svm` is the line search of (5). The candidate is scored by a full SVM
solve, and the acceptance test is strict:

```python
            for _ in range(12):
                trial = _project_simplex(weights + 0.5 * step_size * direction)
                shift = float(np.max(np.abs(trial - weights)))
                if shift < 1e-10:
                    break
                done, _ = self._svm_solve(self._mixed(blocks, trial))
                tq = self._quadratic(blocks, done, codes)
                tvalue = float(done.sum() - 0.5 * np.dot(trial, tq))
                if tvalue <= value - 1e-12 * abs(value):
```

A candidate whose projected step is essentially zero is the signal that the
iterate is a vertex and the gradient points out of the simplex; breaking there
is what makes the loop terminate instead of grinding out twelve solves per
iteration. The `1e-12 * abs(value)` slack is smaller than any real decrease but
large enough to absorb the solver's own tolerance.

`easy_mkl` never forms $Y\bar KY$. It applies it:

```python
        def apply(vec):
            return codes * (mean_block @ (codes * vec)) + self.reg * vec
...
        for _ in range(self.steps):
            relaxed = g - rate * apply(g)
            new = np.empty_like(g)
            new[positive] = _project_simplex(relaxed[positive])
            new[~positive] = _project_simplex(relaxed[~positive])
```

Two projections, one per class block, not one over all $2n$ coordinates: the
constraint in (9) is a product of simplices, and a single global projection
would let one class take all the mass. The matrix-free `apply` keeps the memory
at one $n\times n$ block even though $q$ of them exist — but only if you pass
the mean, which is why (9) uses $\bar K$.

Prediction is the weighted sum, with the training scales folded into the
coefficients:

```python
        v = self.alpha_ * self.codes_
        scores = np.full(blocks[0].shape[0], self.intercept_)
        for weight, scale, K in zip(self.weights_, self.scales_, blocks,
                                    strict=True):
            scores += (weight / scale) * (K @ v)
```

Each test block is $m\times n$ against the training set, so `K @ v` is
$O(mn)$ — no test kernel against itself is ever built. Dividing by the stored
training `scale` is the half of (2) that a naive implementation forgets, and it
silently rescales the margin.

## Complexity

With $q$ kernels, $n$ training points, $m$ test points, $d$ features and
$T$ weight steps:

- **Kernels:** $O(q\,n^2d)$ to build, $O(q\,n^2)$ to store. This is the real
  limit: $q=20$ with $n=10^4$ is 16 GB of `float64` before anything is learned.
- **One SVM solve:** $O(n^2)$ per working-set step and $O(n^2)$ memory; the
  iteration count is data dependent, typically tens to a few hundred.
- **`average`, `alignment`:** one solve, plus $q$ Frobenius products
  $O(q\,n^2)$ for the alignments.
- **`mkl_svm`:** $T$ outer steps, each one accepted solve plus up to 12
  rejected ones, so the constant is what hurts, not the asymptotics. The demo
  runs $T=25$ solves plus a few rejected candidates.
- **`easy_mkl`:** $O(n^2)$ per step, 30 power iterations for the step size, and
  572 steps at $n=150$ — the largest operation count in the file.

The demo is deliberately small so that all four variants can be compared; at
$n=10^4$ you would precompute the blocks once, cache them, and use `easy_mkl`
or `average`.

## Failure modes

**Kernel scale silently decides the average.** If the mean diagonals of the
blocks differ by more than a factor of two, uniform averaging is a weighted
average in disguise and the largest kernel wins. Diagnose by printing
`np.mean(np.diag(K))` per block; the fix is `normalize=True` (the default) or
explicit standardisation of the features before building the kernels.

**`mkl_svm` buys memorisation.** In the demo it gives weight 0.226 to a
100-bandwidth RBF on pure noise, a kernel that scores 0.5075 alone. Nothing in
(4) penalises that: a kernel on which the SVM can carve a large-margin
memorisation of the training set has large $q_k$, so the gradient rewards it.
The combined classifier still scores 0.9445 because the signal kernel holds
0.518, but the weights are not a statement about which kernels generalise.
Diagnose by evaluating each $K_k$ alone and comparing with its weight; if a
chance-level kernel has non-trivial weight, the weights need a validation split.

**Alignment prefers capacity.** $A_k$ in (7) is computed on the training
labels, and a high-capacity kernel can align with $yy^{\top}$ by interpolating
it. With $q$ large and $n$ small the alignment weights can concentrate on a
kernel that overfits, and unlike `mkl_svm` there is no $C$ to restrain it. This
is the price of skipping the SVM: the diagnostic is to compare the aligned
combination with the uniform one on held-out data, and to keep the uniform one
unless the gap is larger than the split's noise.

**The uniform average is hard to beat.** 0.9155 against 0.9365, 0.9445 and
0.9395 in the demo — real, but obtained with eight hand-built kernels and a
2 000-point test set. With $q=3$ generic kernels the gap is usually inside the
noise. Report the average as a baseline and the learned variant as an
improvement only when it clears it on more than one split.

## Applications

Multiple kernel learning is the right tool when the features are **already
heterogeneous**: a string kernel for sequences, an RBF on measured columns, a
linear kernel on counts, and no principled way to concatenate them. Classical
uses are in computational biology, where a gene or protein is described by a
spectrum kernel, a mismatch kernel, a phylogenetic kernel and an expression
kernel at once, and in multi-view learning, where each sensor gets its own
similarity.

It is also the honest way to encode "I do not know which bandwidth is right":
build an RBF at five bandwidths and let the weights choose. That is a
cross-validation over $\gamma$ compressed into one convex problem, and it
usually beats picking $\gamma$ on a single split.

It is a trap when $q$ or $n$ is large enough that the blocks do not fit in
memory, when the base kernels are near-duplicates (the weights are then
unidentifiable and unstable), and when a single well-tuned kernel is within
noise of the combination — which, outside of multi-view data, is most of the
time. The variant table is the decision: start with `average`, try `alignment`
because it is nearly free, and move to `mkl_svm` or `easy_mkl` only when the
kernels disagree about which examples are similar.

## References

- Lanckriet, Cristianini, Bartlett, El Ghaoui, Jordan (2004). *Learning the
  kernel matrix with semidefinite programming.* JMLR 5. The formulation that
  started the field, and the reason "MKL" and "SDP" were once synonyms.
- Bach, Lanckriet, Jordan (2004). *Multiple kernel learning, conic duality, and
  the SMO algorithm.* ICML. The $\ell_1$ MKL dual and the SMO-style algorithm
  that `mkl_svm` descends.
- Cortes, Mohri, Rostamizadeh (2012). *Algorithms for learning kernels based on
  centered alignment.* JMLR 13. Where (7) and its guarantees come from, and an
  honest account of when alignment is only a heuristic.
- Aiolli, Donini (2015). *EasyMKL: a scalable multiple kernel learning
  algorithm.* Neurocomputing 169. The quadratic relaxation of (9).
- Gönen, Alpaydın (2011). *Multiple kernel learning algorithms.* JMLR 12. The
  survey to read before choosing a variant.
