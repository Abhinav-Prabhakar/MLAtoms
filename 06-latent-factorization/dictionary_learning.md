# Dictionary learning

`dictionary_learning.py` · Latent structure · Tier 3 · 4 variants

## Intuition

Sparse coding takes the dictionary as given. This atom learns it from the data,
and the only reason that is possible is that the problem splits in half.

```
        X  (n × d)          A  (n × K)          Dᵀ (K × d)
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │              │ ≈  │  ·  ·   ·    │  · │              │
   │    signals   │    │  ·   ·  ·    │    │    atoms     │
   │              │    │ ·   ·    ·   │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘
                            ≤ s per row         unit columns

   given D → A is sparse coding (a convex problem, per row)
   given A → D is least squares (a closed form)
```

Alternate the two halves and each step is easy; the pair is not, because the
factorisation is non-convex and the two halves can cooperate on a bad local
optimum. That is the honest summary of every dictionary-learning algorithm: a
well-understood coordinate descent on an objective with no global guarantee.

What it buys is the thing fixed dictionaries cannot do. A DCT is a good basis for
a photograph and a bad one for a spectrogram; a dictionary learned from the data
in hand adapts, and the demo shows it: 500 signals of 20 dimensions, built from 3
of 40 unknown atoms, and after 12 alternations the learned atoms match the true
ones at mean cosine $0.926$. Sign and order are arbitrary — nothing in the
objective fixes them — so that number is the strongest statement available.

The four variants are four ways to spend the second half. `mod` solves it exactly
in one matrix inverse. `ksvd` updates one atom at a time with an SVD that also
revises that atom's coefficients. `online_dl` never forms the joint problem and
keeps $O(K^2)$ state instead of the whole code matrix. `sparse_dict` constrains
each atom to be sparse in a fixed base, so the learned atoms are readable.

The result that matters and that the textbooks get wrong: **`mod` and `ksvd` are
a draw.** On three problems the demo measures MOD at $0.114/0.106/0.112$ and
K-SVD at $0.115/0.102/0.116$ — MOD wins 2 of 3, and the means are $0.111$ and
$0.111$. Given fixed codes, MOD's update *is* the joint least-squares optimum
while K-SVD's is one Gauss-Seidel sweep, and K-SVD's famous coefficient update is
discarded as soon as the alternation re-codes. Neither dominates, and any claim
that one is $x\%$ better is a claim about an initialisation.

## Variants

| Variant | Dictionary update | Cost per pass | Where it wins |
|:--|:--|:--|:--|
| `ksvd` | one atom at a time, rank-one SVD | $O(K\,m d \min(m,d))$ | interleaved with expensive coding |
| `mod` | joint least squares, one inverse | $O(K^2 d + K^3)$ | when coding is cheap and exact |
| `online_dl` | per-batch, from two accumulators | $O(K^2 + dK)$ state | more signals than fit in memory |
| `sparse_dict` | per atom, a lasso in a fixed base | $O(P^2 + dP)$ per atom | when atoms must be interpretable |

**`ksvd`** is K-SVD: for each atom, restrict to the signals that use it, form the
residual without that atom's contribution, and take the leading singular pair of
that residual. Atom and coefficients update together, which is the trick — but
with greedy coding re-run every alternation the demo finds no advantage over
MOD's joint solve (below). Its own curve is honest: on 500 signals the error
falls $0.174 \to 0.124 \to 0.097$ over 12 alternations and all 40 atoms stay
alive.

**`mod`** is the method of optimal directions, and it is one line: given the
codes, the best dictionary is $D^{\mathsf T} = (A^{\mathsf T}A)^{-1}A^{\mathsf
T}X$. Exactly the least-squares optimum of the second half, so the alternation's
objective decreases monotonically and by more per step than K-SVD's. It is also
the reason K-SVD exists: $K^3$ for the inverse is fine at $K = 40$ and hopeless
at $K = 10^4$.

**`online_dl`** processes the data in batches and keeps only
$A = \sum\alpha\alpha^{\mathsf T}$, $B = \sum x\alpha^{\mathsf T}$. One dictionary
update per batch, so 64 updates for 800 signals against 8 full passes, and the
state is 19 kB against the 256 kB the batch methods hold. It reaches $0.0366$
against K-SVD's $0.0852$ — better, because it got eight times as many updates out
of the same data. The caveat is in the variant's own design: the accumulators are
unweighted, so the first batches dominate forever, and the standard fix is a
forgetting factor that turns them into a running average.

**`sparse_dict`** writes each atom as a sparse combination of a fixed base,
$D = \Phi C$, and updates $C$ with a lasso instead of an SVD. With $\Phi = I$ and
$\gamma = 0$ the constraint is off and every atom is dense (20 of 20
coefficients); with $\gamma = 0.15$ the atoms keep $5.4$ coefficients on average
(at most 8) and the error rises $0.0995 \to 0.1888$. Four times fewer parameters
per atom for $90\%$ more reconstruction error — that ratio is the entire argument
for double sparsity.

## Derivation

### 1. The objective, and why the constraints are both load-bearing

$$\min_{A,\,D}\; \lVert X - AD^{\mathsf T}\rVert_F^2 \quad\text{subject to}\quad
\lVert a_i\rVert_0 \le s \ \ \forall i, \qquad \lVert d_j\rVert_2 = 1 \ \ \forall j. \qquad (1)$$

Both constraints are needed and neither is cosmetic.

Without the sparsity budget the problem is ordinary PCA-type factorisation, and
with $K > d$ it has a continuum of exact solutions — the dictionary can be any
basis of anything. The budget is what makes the decomposition identifiable at
all, and it is also what makes the problem hard: $\ell_0$ makes each half
non-convex, which is why the coding step in the code is greedy rather than
convex.

Without the unit-norm constraint there is a trivial degeneracy: rescale atom $j$
by $c$ and its coefficients by $1/c$ and the product is unchanged, so the
objective is invariant along a ray and any optimiser drifts along it. Fixing
$\lVert d_j\rVert = 1$ picks one representative per ray. The code enforces it by
rescaling both factors (`_rescale`), which is exact — rescaling only $D$ would
change the fit.

### 2. Given the codes, the dictionary is a least-squares problem

With $A$ fixed, (1) is a quadratic in $D$ with no constraints other than the
norms, and the closed form is

$$D^{\mathsf T} = (A^{\mathsf T}A)^{-1}A^{\mathsf T}X = \left(\sum_i a_ia_i^{\mathsf T}\right)^{-1}\sum_i a_i x_i^{\mathsf T}, \qquad (2)$$

which is `mod`. The inner matrix $A^{\mathsf T}A$ is what makes the method
$O(K^3)$ and it is where the ridge in the code comes from: an atom used by no
signal gives a zero row and column, so the system is singular, and the ridge turns
"undefined" into "leave that direction alone".

### 3. K-SVD: the same problem, one atom at a time

Split the objective by atom. For atom $j$, let $E_j$ be the residual with its
contribution removed, $E_j = X - \sum_{k \ne j} a_k d_k^{\mathsf T}$. Then

$$\lVert X - AD^{\mathsf T}\rVert_F^2 = \lVert E_j - a_jd_j^{\mathsf T}\rVert_F^2
= \lVert E_j\rVert_F^2 - 2\,a_j^{\mathsf T}E_jd_j + \lVert a_j\rVert^2\lVert d_j\rVert^2,
\qquad (3)$$

so minimising over the pair $(a_j, d_j)$ is exactly the rank-one approximation of
$E_j$, and the solution is the leading singular triple:

$$E_j = \sum_r s_r u_r v_r^{\mathsf T} \;\Longrightarrow\;
d_j = v_1, \qquad a_j = s_1 u_1. \qquad (4)$$

The support of $a_j$ is respected by restricting (3) to the rows where $a_j \ne 0$
— that is the only place sparsity enters the dictionary step, and forgetting it
is the classic K-SVD bug: the SVD of the *full* residual gives an atom that tries
to explain signals that do not use it.

Two properties follow. Each atom update decreases the objective (it is an exact
minimisation over that pair), so the sweep is monotone. And the sweep is
Gauss-Seidel: atom $j+1$ sees the coefficients atom $j$ just changed, so it is not
the joint optimum that (2) computes. That is exactly the gap the demo measures,
and it is why the ordering of atoms within a sweep is arbitrary and its effect
shows up in the third decimal.

### 4. Online: a surrogate with two sufficient statistics

Holding $A = \sum_i \alpha_i\alpha_i^{\mathsf T}$ and $B = \sum_i x_i\alpha_i^{\mathsf T}$,
expand (2) per column:

$$d_j = \frac{1}{A_{jj}}\left(b_j - D\,a_{:j}\right) + d_j, \qquad (5)$$

with $a_{:j}$ the $j$-th column of $A$ with its own diagonal entry zeroed. This is
the coordinate-wise minimiser of a quadratic surrogate, and the point of the
rearrangement is that it uses only $A$, $B$ and the current $D$ — no data. So the
algorithm can be run streaming: code a batch, add its contributions to $A$ and
$B$, update every column once, forget the batch.

In finite arithmetic the two accumulators are not statistics of a fixed
distribution; they are weighted averages that never forget, which is why the
published algorithm adds a decay. The demo does not, and says so: the last batches
barely move the dictionary. It still works — better than 8 batch passes,
because 64 updates is 64 updates — but the unbiased version is
$A_t = \sum_{i\le t}\beta_i\alpha_i\alpha_i^{\mathsf T}$ with $\beta_i \to 0$.

### 5. Double sparsity

Fix a base $\Phi \in \mathbb{R}^{d\times P}$ (a DCT, a wavelet frame, the identity)
and write $D = \Phi C$. Then (1) becomes a problem in $C$, and the atom update for
column $j$ with $\alpha = a_j$, $R = E_j$ is

$$\min_{c}\; \tfrac12 c^{\mathsf T}\Phi^{\mathsf T}\Phi\, c -
c^{\mathsf T}\frac{\Phi^{\mathsf T}R\alpha}{\alpha^{\mathsf T}\alpha}
+ \gamma\lVert c\rVert_1, \qquad (6)$$

which is a lasso in $P$ dimensions with Gram $\Phi^{\mathsf T}\Phi$ — the
`sparse_coding` solver, called $K$ times. The $\alpha^{\mathsf T}\alpha$ divisor is
not cosmetic: it is the curvature of (3) in that direction, because a signal with
small coefficients gets a small weight in the total residual.

With $\Phi = I$ the constraint is "atoms are sparse in the ambient basis", which
is the case the demo runs, and $\gamma = 0$ removes the constraint entirely,
leaving a per-atom ordinary least-squares step. That is the same *model* as MOD;
it reaches a different point ($0.0995$ against $0.0885$) because per-atom
Gauss-Seidel is not the joint solve. The distinction matters when reading the
literature: "double sparsity" is a constraint on the parameterisation, and its
benefit is interpretability, not accuracy.

### 6. What is identifiable, and what is not

The learned dictionary is determined only up to

$$D \mapsto D P, \qquad A \mapsto A P^{-{\mathsf T}}, \qquad (7)$$

for any signed permutation $P$ — and, when $K > d$, any invertible right factor
that keeps the code's sparsity intact. So there is no such thing as "the"
dictionary: two runs from different seeds produce different atoms in different
orders with different signs, and both are correct answers. That is why the demo's
quality measure is `atom_recovery`, the mean over learned atoms of the best
absolute cosine with a true atom, rather than a matrix norm.

The same fact is what makes the choice of $K$ a modelling decision. $K$ smaller
than the truth and atoms merge; $K$ larger and the surplus atoms are dead (used
by no signal), which the code revives from the worst-reconstructed signal —
`_revive` — because a dead atom contributes nothing and never recovers on its own.

### 7. Convergence

Alternating minimisation on (1) decreases the objective at every step and is
bounded below, so it converges — to a stationary point, which is all that can be
said. In practice the curve is steep for two or three passes and then shallow:
the demo's $0.174 \to 0.124 \to 0.097$ is most of the total movement in the first
three of twelve. The stopping rule that matters is the one nobody implements:
the objective can be flat while the atoms rotate within the degenerate subspace of
(7), so "objective unchanged" is not "dictionary unchanged".

## Code walkthrough

`DictionaryLearning(n_components=20, *, variant="ksvd", nnz=3, n_iter=12,
lam=0.1, batch=128, gamma=0.1, base=None, seed=0)` with `fit(X)`,
`dictionary_` $(d, K)$, `codes_` $(n, K)$, `rmse_` (one entry per alternation),
`base_`, and the module-level `atom_recovery`.

The coding step is `sparse_coding` itself, imported rather than reimplemented:
`_codes` calls `SparseCoding(nnz=..., variant="omp")`. That is the composition the
chapter is built on — this atom is the alternation, not the coder.

### 1. Initialisation on real signals

```python
        if n >= self.n_components:
            # initialise on real signals, not on noise: a Gaussian dictionary has
            # no relation to the data's structure, so the first coding pass spends
            # its budget fitting the atoms it was given
            D = _normalise(X[rng.choice(n, self.n_components, replace=False)].T)
        else:
            D = _normalise(rng.standard_normal((d, self.n_components)))
```

The single most consequential line in the file has nothing to do with the
algorithm. On the demo's 500 signals, the same 12 alternations starting from
Gaussian atoms end at $0.1015$ with mean cosine $0.888$; starting from 40 actual
signals they end at $0.0975$ with $0.926$, and the curves start at $0.248$ and
$0.174$ respectively. The gap is not in the destination, it is in the budget
needed to get there: a Gaussian dictionary's atoms are unrelated to the data, so
the first coding pass spends its budget fitting noise and the alternation has to
undo that before it learns anything.

`rng.choice` without replacement, so no atom starts as a duplicate of another —
which matters because two identical atoms make the first coding pass arbitrary
between them and the pair can stay locked for the whole run.

### 2. The exact joint update

```python
        G = A.T @ A
        D = np.linalg.solve(G + 1e-8 * np.eye(G.shape[0]), A.T @ X).T
        return D, A
```

(2), written as a solve rather than an inverse: same arithmetic, better
conditioning, and `np.linalg.solve` on a $K\times K$ system is what makes this
$O(K^3)$ rather than $O(K^4)$.

The ridge is the dead-atom problem. If atom $j$ is used by no signal then row and
column $j$ of $G$ are zero, `G` is singular, and the solve either raises or
returns garbage — but only for the components along that atom, so the failure is
silent in every aggregate metric. `1e-8` makes those directions return zero, i.e.
the atom keeps whatever it had when multiplied back through $A^{\mathsf T}X$
(which is zero for it). The alternative — reviving dead atoms explicitly — is
what K-SVD's `_revive` does and what MOD needs if $K$ is chosen too large.

### 3. The rank-one sweep

```python
        for j in range(D.shape[1]):
            rows = np.nonzero(A[:, j])[0]
            if rows.size == 0:
                D[:, j], A[:, j] = self._revive(X, A, D)
                continue
            R = X - A @ D.T + np.outer(A[:, j], D[:, j])
            U, s, Vt = np.linalg.svd(R[rows], full_matrices=False)
            D[:, j] = Vt[0]
            A[rows, j] = s[0] * U[:, 0]
```

(3)–(4). `R = X - A @ D.T + np.outer(A[:, j], D[:, j])` is $E_j$ computed from the
current full residual: one matmul plus one rank-one correction, rather than a sum
over the other $K-1$ atoms. Recomputing the whole residual $K$ times per sweep is
the obvious implementation and it is $K$ times slower, which at $K = 40$ is the
difference between 0.4 s and 15 s.

`R[rows]` is the restriction that makes the update respect sparsity, and it is
also the shape trap: the matrix is $|{\rm support}|\times d$, so its SVD gives
$U$ of shape $(m, m)$ and $V^{\mathsf T}$ of shape $(m, d)$. The atom is a row of
$V^{\mathsf T}$ — length $d$ — and the coefficients are a column of $U$ scaled by
$s$. Swapping the two is the bug that produces a broadcast error, and the version
of it that survives is transposing a square case.

### 4. Reviving a dead atom

```python
        resid = X - A @ D.T
        worst = int(np.argmax(np.sum(resid ** 2, axis=1)))
        vec = resid[worst]
        nrm = float(np.linalg.norm(vec))
        if nrm < EPS:
            return D[:, 0], np.zeros(A.shape[0])
        coef = np.zeros(A.shape[0])
        coef[worst] = nrm
        return vec / nrm, coef
```

A dead atom is a coordinate nobody uses, and no amount of alternating will bring
it back: its gradient is zero, so its update is zero. The standard revival gives
it the signal the current dictionary explains worst, with the coefficient set to
that signal's residual norm — which is the exact least-squares fit for a
single-atom dictionary on that signal.

`nrm < EPS` cannot happen for real data but is the difference between an error and
a division by zero on degenerate input, and returning atom 0 unchanged is the
harmless choice: it stays dead for this sweep, and the next coding pass decides
again.

### 5. The online update

```python
            A_acc += Ab.T @ Ab
            B_acc += Xb.T @ Ab
            for j in range(K):
                if A_acc[j, j] < 1e-9:
                    D[:, j], _ = self._revive(Xb, Ab, D)
                    continue
                col = A_acc[:, j].copy()
                col[j] = 0.0
                D[:, j] = (B_acc[:, j] - D @ col) / A_acc[j, j]
                nrm = float(np.linalg.norm(D[:, j]))
                if nrm > EPS:
                    D[:, j] /= nrm
```

(5), and `col[j] = 0.0` is the line that makes it correct: the closed form
excludes atom $j$'s own contribution, because the term $A_{jj}d_j$ appears on
both sides of the normal equation. Leaving it in doubles the atom's weight on
every update, which converges to a dictionary whose atoms are too large — and
then the renormalisation hides it, so the fit still looks plausible.

The renormalisation is not the exact rescaling of `_rescale`; it is a projection
onto the unit ball, which is the published algorithm's constraint handling, and
the coefficients in `Ab` were computed before it. That inconsistency is why the
online path's reported error uses freshly computed codes at the end of the epoch.

### 6. Double sparsity's atom update

```python
            R = X - A @ D.T + np.outer(aj, D[:, j])
            b = base.T @ (R.T @ aj) / w
            c = _lasso_cd(G, b, self.gamma, 200, 1e-10)[0]
            nrm = float(np.linalg.norm(base @ c))
            if nrm < 1e-8:
                continue
            C[:, j] = c / nrm
            A[:, j] = aj * nrm
```

(6). `w = aj @ aj` is the curvature divisor, `G = baseᵀbase` is formed once
outside the atom loop, and `_lasso_cd` is imported from `sparse_coding` — the
same exact coordinate descent that solves (1) there, handed a different Gram.

`if nrm < 1e-8: continue` is the case where the penalty zeroes the whole atom. The
right behaviour is to leave the atom and its coefficients alone rather than to
divide by a near-zero norm, because a zeroed atom is a statement that this
direction carries nothing at this γ — not a licence to produce infinities.

### 7. Measuring two dictionaries against each other

```python
    G = np.abs(_normalise(learned).T @ _normalise(truth))
    return float(np.mean(np.max(G, axis=1)))
```

(7) made measurable. Each learned atom is matched to its best true atom in
absolute cosine, and the mean over learned atoms is the score. Absolute value
because the sign is free; max over the true dictionary because the order is free;
mean (not max) so that a dictionary with 20 good atoms and 20 junk cannot hide
behind the good ones.

What it does not measure is duplication: two learned atoms can both match the
same true atom and the score stays high while the dictionary is degenerate. The
complementary statistic is the fraction of atoms actually used — the demo reports
it as `alive`.

### 8. The error the curve reports

```python
    @staticmethod
    def _error(X: np.ndarray, A: np.ndarray, D: np.ndarray) -> float:
        """Root-mean-square reconstruction error per entry — comparable across variants."""
        return float(np.sqrt(np.mean((X - A @ D.T) ** 2)))
```

Per entry, not per signal and not summed, so that runs with different $n$, $d$ or
$K$ are comparable — the only reason the demo can put four variants in one
sentence. It is also the quantity that hides everything: a dictionary can have a
good RMSE and uninterpretable atoms, which is why `atom_recovery` exists.

The loop recomputes the codes after each update and *then* records the error, so
`rmse_[t]` is always the error of a dictionary with codes that were fitted to it.
Recording before the update instead would report a lagged quantity that is not
the objective of anything, and it would break the monotonicity that makes the
curve readable.

## Complexity

$n$ signals, $d$ dimensions, $K$ atoms, $s$ non-zeros per code, $T$ alternations,
$m$ = signals using a given atom.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_codes` (OMP, all rows) | $O(n s^2 d)$ | $O(nK)$ | the dominant cost per alternation |
| `_mod` | $O(nK^2 + K^3 + nKd)$ | $O(K^2)$ | the $K^3$ is why MOD stops scaling |
| `_ksvd` | $O(\sum_j m_j d \min(m_j,d))$ | $O(md)$ | one SVD per atom per pass |
| `_online` | $O(ns^2d + \frac{n}{b}K^2d)$ | $O(K^2 + dK)$ | bounded state, no code matrix |
| `_sparse_dict` | $O(K\,T_{\rm lin}(P^2 + dP))$ | $O(PK)$ | a lasso per atom |
| `atom_recovery` | $O(dK_{\rm true}K)$ | $O(K_{\rm true}K)$ | once, at the end |
| full demo | — | — | $2.5$ s, four variants, 15 fits |

The row that decides the method is the first: coding is $O(ns^2d)$ per
alternation and it is the same for every variant, so the dictionary update is
usually not the bottleneck at all — which is precisely why the MOD/K-SVD
distinction is worth so little in practice, and why the online variant (which
amortises coding over more updates) wins on the demo's data.

Memory separates them properly. `mod` and `ksvd` hold the code matrix, $nK$
floats — 256 kB for 800 signals and 40 atoms, 25 MB for a million signals.
`online_dl` holds $K^2 + dK$ — 19 kB in the same problem, and independent of $n$.
That is the variant's justification, and it is the reason dictionary learning is
routinely run on corpora that do not fit in RAM.

## Failure modes

**1. The comparison you read in a paper is a comparison of initialisations.**
MOD $0.114/0.106/0.112$, K-SVD $0.115/0.102/0.116$ on three seeds: the winner
alternates and the means are equal to three decimals. Reporting one run of each is
reporting the seed. Any honest table has several, and the spread is the finding.

**2. Blind initialisation costs iterations, and iteration budgets are finite.**
$0.1015$ against $0.0975$ at the *end* of the same 12 alternations, but $0.248$
against $0.174$ at the start: the Gaussian run spends its first four passes
recovering what the data initialisation got for free. When the budget is small —
which is the usual case, since each pass costs an $O(ns^2d)$ coding step — the
initialisation is the difference between a usable dictionary and a partially
converged one.

**3. A dead atom never comes back on its own.** Its gradient is zero, so its
update is zero, so it stays dead. MOD's ridge quietly freezes it; K-SVD revives
it from the worst-reconstructed signal. Choosing $K$ larger than the data supports
therefore does not add capacity, it adds dead columns — and the RMSE will not tell
you, because dead atoms contribute nothing.

**4. The objective is flat in the directions of (7).** Permuting atoms, flipping
their signs, or mixing two atoms that are never used together changes the
dictionary and leaves the objective bit-for-bit identical. Reporting "the learned
dictionary" from one run is reporting one representative; averaging dictionaries
across runs is averaging different bases and produces atoms that were in no run.

**5. Greedy coding makes the alternation's curve non-monotone.** Each half
decreases its own objective, but OMP re-coding is not the exact minimiser of the
coding half, so a dictionary update can raise the error measured with fresh codes.
The demo's curve is monotone here; on harder data it is not, and a rising RMSE is
not necessarily a bug.

**6. Online with unweighted accumulators forgets nothing and learns less.**
$A$ and $B$ grow without bound, so a batch seen at step 3 still has full weight at
step 300, and the dictionary ends up fitted to the beginning of the stream. It
looks fine on i.i.d. data, which is why the demo's numbers are as good as they
are; on a stream whose statistics drift, the failure is total and invisible.

**7. Sparsity of the atoms is not interpretability of the atoms.** `sparse_dict`
with $\Phi = I$ gives atoms with 5 non-zeros out of 20 and $90\%$ more error; that
is a readable atom only if the 20 coordinates are themselves meaningful. On raw
pixels, a 5-sparse atom is a five-pixel picture — sparse, uninterpretable, and
strictly worse than the dense one. The base has to be chosen so that its
coefficients mean something.

**8. Everything here assumes the codes are good enough.** The dictionary update
inherits whatever the coder got wrong: with $\text{nnz} = 1$ the learned atoms are
cluster centres, not dictionary atoms, and with a huge budget the atoms fit noise.
The coding budget is the hyper-parameter that matters most and it is usually
chosen by looking at the reconstruction error, which is precisely the quantity
that rewards a large budget.

## Applications

- **Image denoising and inpainting.** Learn a dictionary from patches of the
  corrupted image itself, then code each patch sparsely and reconstruct. The
  reason it beats a fixed wavelet basis is that the atoms are the image's own
  texture, and the reason it works with one image is the patch redundancy.
- **Compressed sensing with a learned basis.** Every recovery guarantee is stated
  relative to a dictionary; learning it is how the guarantee is made applicable
  to a specific signal class.
- **Audio: source separation and coding.** A learned dictionary of note-like atoms
  is how sparse coding separates instruments, and how low-bitrate codecs get their
  atoms.
- **Medical and scientific imaging.** Dictionary learning on MRI k-space
  reconstructs from fewer samples than a fixed transform requires; the learned
  atoms are the anatomical structures that repeat.
- **As the first layer of a network.** Convolutional sparse coding and the
  dictionary-learning layers of early deep models are this atom; the "feature
  learning" story of deep nets starts with exactly this alternation.
- **Anomaly detection.** Fit a dictionary to normal data; a new signal with a
  large residual is anomalous. The residual is the interpretable quantity — a
  sparse code that needs large coefficients to explain something is itself the
  explanation.

Where it is a trap: reporting one seed's dictionary; choosing $K$ above what the
data supports and calling the dead atoms "features"; using the reconstruction
error as the only diagnostic when interpretability was the goal; and expecting
online learning to track a distribution that drifts while the accumulators never
decay.

## References

- **Olshausen & Field (1996),** *Emergence of simple-cell receptive field
  properties by learning a sparse code for natural images* — the paper that made
  this a vision result, and still the cleanest statement of why the sparsity
  budget is the model.
- **Engan, Aase & Husøy (1999),** *Method of optimal directions for frame design*
  — MOD, equation (2), and the observation that the dictionary step is an
  ordinary least-squares problem.
- **Aharon, Elad & Bruckstein (2006),** *K-SVD: an algorithm for designing
  overcomplete dictionaries for sparse representation* — the rank-one sweep of
  (3)–(4), including the support restriction that makes it work.
- **Elad & Aharon (2006),** *Image denoising via sparse and redundant
  representations over learned dictionaries* — the application that made K-SVD
  famous, and the patch-based setup everything since uses.
- **Mairal, Bach, Ponce & Sapiro (2010),** *Online learning for matrix
  factorization and sparse coding* — the accumulators of §4, the block update of
  (5), and the streaming regime that makes dictionary learning scale.
- **Rubinstein, Zibulevsky & Elad (2010),** *Double sparsity: learning sparse
  dictionaries for sparse signal approximation* — (6) and the $\Phi C$
  factorisation; the variant that trades fit for interpretability on purpose.
- **Skretting & Engan (2010),** *Recursive least squares dictionary learning
  algorithm* — RLS-DLA, the variant that replaces the accumulators with a
  forgetting-factor recursion; the principled fix for failure mode 6.
- **Sahoo & Makur (2015),** *Dictionary training for sparse representation as
  generalization of K-means clustering* — SGK, which shows the alternation is
  K-means with a sparsity budget rather than a single atom, and is the cheapest
  way to see why the method works at all.
- **Tosic & Frossard (2011),** *Dictionary learning: what is the right
  representation for your signals?* — the survey that compares the variants on the
  criteria that matter, including the identification group (7).
- **Bao, Ji, Quan & Shen (2015),** *L0 norm based dictionary learning by proximal
  methods with global convergence* — what changes when the budget is $\ell_0$
  throughout instead of $\ell_0$ coding with $\ell_2$ dictionary updates.
