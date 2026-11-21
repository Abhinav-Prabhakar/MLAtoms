# Dynamic time warping — align sequences that do not share a clock

`dtw.py` · Sequences · Tier 1 · 7 variants

## Intuition

Two recordings of the same word are not the same length, and no single shift
makes them line up. One speaker stretched the vowels, the other paused in the
middle, and the samples therefore sit at different times. Comparing them point
by point measures the clock, not the content. Dynamic time warping measures the
content by allowing a monotone reparameterisation of time: point $i$ of one
sequence may correspond to point $j$ of the other, as long as the correspondence
never runs backwards.

The object is a grid. Rows are the points of the first sequence, columns are the
points of the second, and each cell holds the cost of pairing those two points.
A path from the top-left corner to the bottom-right corner, moving only right,
up or diagonally, is a valid alignment; the cost of the alignment is the sum of
the cells it visits. DTW is the minimum over paths, and it is found by the same
dynamic program as any shortest path on a grid.

```
        y:  0    1    2    3    4
      x ┌────┬────┬────┬────┬────┐
      0 │ 0  │ 1  │ 4  │ 9  │ 16 │
      1 │ 1  │ 0  │ 1  │ 4  │ 9  │
      2 │ 4  │ 1  │ 0  │ 1  │ 4  │
      3 │ 9  │ 4  │ 1  │ 0  │ 1  │
        └────┴────┴────┴────┴────┘
             cost of pairing x_i with y_j
```

That grid is the whole algorithm. Everything else is a decision about which
cells are legal. The unconstrained grid allows a path to match the beginning of
one sequence with the end of the other, which is sometimes right (a query inside
a reference) and usually wrong (two recordings of the same word). A band forbids
the far cells, a parallelogram forbids the steep slopes, free endpoints on one
axis allow the subsequence case, and a soft minimum replaces the hard choice
with a differentiable one.

The variants are those decisions, plus two operations built on top of the grid:
averaging a set of sequences in the warped geometry, and approximating the grid
on a coarser one.

The demos make the trade-offs numerical. On a shape sampled through a known
monotone warp, the recovered path pairs points that are a hundredth of a cycle
apart at worst, and the distance is $0.0675$. On a stronger warp, a Sakoe–Chiba
band of three cells costs $7.8$ times the unconstrained distance, while a band
of ten costs nothing because it contains the optimal path. Soft DTW's gradient
matches finite differences to $5\times10^{-10}$, and as the smoothing shrinks
the soft distance converges to the hard one. The barycenter of twelve warped
copies of a shape recovers it to a correlation of $0.99987$ and beats the
pointwise mean, $0.2577$ against $0.2839$ in total distance.

The one number to carry away is the cost of a wrong constraint. On the strongly
warped pair the unconstrained path is worth $0.100$; a band of three returns
$0.778$. A band is not an optimisation. It is a statement about which pairings
are physically possible, and it is the part of the model that has to be right.

## Variants

| variant | grid | endpoints | returns |
|:--|:--|:--|:--|
| `standard` | all cells | fixed | exact path and cost |
| `sakoe_chiba` | band of radius $r$ | fixed | exact path and cost |
| `itakura` | parallelogram | fixed | exact path and cost |
| `subsequence` | all cells | free on the reference | best matching interval |
| `soft_dtw` | all cells | fixed | smoothed cost and gradient |
| `dtw_barycenter` | all cells | fixed | an average sequence |
| `fast_dtw` | progressively refined band | fixed | approximate path and cost |

**`standard`** is the classical algorithm: squared Euclidean local cost, all
cells legal, both endpoints fixed. It is exact for its model and the reference
against which the others are measured.

**`sakoe_chiba`** keeps a band of half-width `radius` around the diagonal. It is
the right constraint when the two sequences are known to be roughly aligned —
same gesture, same word, same heartbeat — and it reduces the work from $nm$ to
about $n\,r$.

**`itakura`** keeps a parallelogram defined by a maximum local slope. Instead of
a fixed width, it allows the path to drift in the middle and forces it back at
both ends, which is a different statement: the alignment may be locally steep
but the global clocks cannot diverge.

**`subsequence`** frees the start and the end on the reference axis. The query
must be matched completely, but it may begin and end anywhere in the longer
sequence, which is the query-by-example problem: find this word in that
recording, this pattern in that trace.

**`soft_dtw`** replaces `min` with a soft minimum controlled by a temperature
$\gamma$. The result is a smooth, differentiable function of both sequences,
which is what makes it usable as a loss or a layer. It is not a distance: it is
negative for large $\gamma$ and it depends on the path length.

**`dtw_barycenter`** computes DTW barycenter averaging: align every member to a
running average, then average the points that landed at the same index. It is
the correct notion of mean when the samples are misaligned in time.

**`fast_dtw`** aligns a coarsened copy of the pair, projects the coarse path
back as a band, and refines. It is approximate, and its distance is never below
the exact one because the band is a restriction of the grid.

## Derivation

### The objective and the recursion

Let $x_{0:n-1}$ and $y_{0:m-1}$ be the two sequences, and let
$C(i,j)=\lVert x_i-y_j\rVert^2$ be the local cost. A warping path is a sequence
of cells $\pi=(\pi_0,\dots,\pi_L)$ with $\pi_0=(0,0)$, $\pi_L=(n-1,m-1)$, and
each step in $\{(1,0),(0,1),(1,1)\}$. The distance is

$$
D(n-1,m-1) = \min_{\pi}\ \sum_{(i,j)\in\pi} C(i,j).
\qquad (1)
$$

The path set is exponential, but the objective is additive and the constraint is
local, so the principle of optimality applies: the best path to $(i,j)$ ends
with the best path to one of its three predecessors. Hence

$$
D(i,j) = C(i,j) + \min\big(D(i-1,j),\,D(i,j-1),\,D(i-1,j-1)\big),
\qquad (2)
$$

with $D(0,0)=C(0,0)$ and the first row and column filled by a single
predecessor each. The table is $n\times m$, every cell is $O(1)$, and the
backtracking pass recovers the path that achieved the minimum.

The code is (2) with the mask folded in, and with the free-start case handled by
initialising the whole first row instead of its first cell:

```python
        if free_start:
            table[0] = np.where(mask[0], cost[0], np.inf)
        else:
            table[0, 0] = cost[0, 0]
        for i in range(1, n):
            for j in range(m):
                if not mask[i, j]:
                    continue
                best, source = np.inf, (-1, -1)
                for di, dj in STEPS:
                    pi, pj = i - di, j - dj
                    if pi < 0 or pj < 0:
                        continue
                    if table[pi, pj] < best:
                        best, source = table[pi, pj], (pi, pj)
                if np.isfinite(best):
                    table[i, j] = cost[i, j] + best
                    back[i, j] = source
```

The `np.isfinite(best)` test is not defensive noise: a cell whose three
predecessors are all outside the band must stay at infinity, and adding the cost
to infinity would poison the comparison. The stored `source` is what makes
backtracking possible without re-deriving the argmin.

### Bands and parallelograms

The Sakoe–Chiba band of half-width $r$ keeps the cells within $r$ of the
diagonal after accounting for the length ratio:

$$
|j - i\,(m-1)/(n-1)| \le r.
\qquad (3)
$$

In code the diagonal is rounded to the nearest integer cell and the row is
filled in one slice:

```python
        slope = (m - 1) / max(n - 1, 1)
        for i in range(n):
            centre = int(round(i * slope))
            low = max(0, centre - radius)
            high = min(m - 1, centre + radius)
            mask[i, low:high + 1] = True
```

The Itakura parallelogram is a different shape. In normalised coordinates
$u=i/(n-1)$, $v=j/(m-1)$, it requires the local slope to lie between $1/s$ and
$s$ at both ends of the path:

$$
v \le s u, \qquad v \ge u/s, \qquad 1-v \le s(1-u), \qquad 1-v \ge (1-u)/s.
$$

The first two bound the path near the start, the last two near the end, and
their intersection is the parallelogram. It allows a band that is wide in the
middle and narrow at the corners, which is the shape you want when the two
clocks are allowed to drift but not to swap order.

### Free endpoints

For the subsequence variant, the query $x$ must be matched in full and the
reference $y$ may be entered and left anywhere. That is one line of
initialisation and one line at the end:

$$
D(0,j) = C(0,j)\ \ \forall j, \qquad
\text{distance} = \min_j D(n-1,j).
\qquad (4)
$$

Freeing the whole first row lets the path start at any reference index; taking
the minimum over the last row lets it stop at any reference index. The path's
first cell carries the start index, which is the answer to "where does this
pattern begin". The suffix of the reference after the match is ignored, which is
the point; the prefix before it is skipped by construction.

The implementation keeps the start in the backtracked path rather than in a
separate variable, so the interval is read off the alignment:

```python
        end = int(np.argmin(ends))
        path = self._backtrack(back, (x.shape[0] - 1, end), free_start=True)
```

### Soft DTW

Hard DTW is a minimum, and a minimum is piecewise constant in the inputs, so its
gradient is zero almost everywhere. Soft DTW replaces the minimum with the
smooth minimum

$$
\mathrm{softmin}_\gamma(a_1,\dots,a_k) = -\gamma\log\sum_i e^{-a_i/\gamma},
$$

which tends to $\min_i a_i$ as $\gamma\to0$ and to the mean as $\gamma\to\infty$.
The recursion is otherwise identical:

$$
D(i,j) = C(i,j) + \mathrm{softmin}_\gamma\big(D(i-1,j), D(i,j-1), D(i-1,j-1)\big).
\qquad (5)
$$

The implementation subtracts the smallest predecessor before exponentiating,
which is the same log-sum-exp guard as everywhere else in this chapter:

```python
                previous = table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
                lowest = min(previous)
                total = sum(math.exp(-(p - lowest) / gamma) for p in previous)
                table[i, j] = cost[i, j] + lowest - gamma * math.log(total)
```

The gradient follows from the same recursion run backwards. Differentiating (5)
with respect to each predecessor gives the softmax of the negative predecessor
values, so a reverse sweep accumulates an adjoint at each cell and the
derivative with respect to a point of $x$ is twice the adjoint row times the
residual:

$$
\frac{\partial D}{\partial x_i}
= 2\sum_j \bar D(i,j)\,(x_i-y_j), \qquad
\bar D(p) = \sum_{q} \bar D(q)\,\frac{\partial D(q)}{\partial D(p)}.
$$

That is the entire backward pass, and the demo checks it against finite
differences.

### Barycenters

The mean of a set of sequences is not their pointwise mean unless they are
already aligned. DTW barycenter averaging alternates two steps: align every
member to the current average, then set each average point to the mean of the
member points that were warped onto it:

$$
\bar z_j = \frac{1}{|S_j|}\sum_{s\in S_j} z^{(s)}_i,
\qquad S_j = \{(s,i) : (i,j)\in\pi^{(s)}\}.
\qquad (6)
$$

The length of the barycenter is inherited from the initial average, which is the
medoid — the member with the smallest total distance to the others. The
iteration is not guaranteed to decrease the total distance monotonically, but it
does in practice on well-behaved data, and the demo's history is non-increasing.

### FastDTW

The exact table is $O(nm)$, which is too much for sequences of ten thousand
points. FastDTW exploits the fact that alignments are smooth at coarse scales:
halve both sequences by averaging adjacent points, align the short pair, project
the coarse path back to a band of radius $r$, and run the exact recursion inside
that band. Recurse until the sequences fit in a handful of points.

Each level costs $O(n_i r)$ and the lengths halve, so the total is $O(nr)$ for a
fixed radius, and the result is exact whenever the true path stays inside the
projected band. When it does not, the answer is an upper bound, because the band
is a subset of the grid.

## Code walkthrough

### Backtracking

```python
    @staticmethod
    def _backtrack(back: np.ndarray, end: tuple, free_start: bool) -> np.ndarray:
        path = [end]
        while True:
            i, j = path[-1]
            if free_start and i == 0:
                break
            if not free_start and path[-1] == (0, 0):
                break
            pi, pj = back[i, j]
            if pi < 0:
                raise ValueError("the backtracking table is incomplete")
            path.append((int(pi), int(pj)))
        path.reverse()
        return np.asarray(path, dtype=int)
```

The two stopping conditions are the two endpoint conventions: a fixed-start path
ends at the origin, a free-start path ends anywhere in the first row. The
`pi < 0` guard catches the case where the table was filled with an inconsistent
mask, which would otherwise walk to `(-1, -1)` and index out of bounds.

### The band mask

```python
        mask = np.zeros((n, m), dtype=bool)
        slope = (m - 1) / max(n - 1, 1)
        for i in range(n):
            centre = int(round(i * slope))
```

The mask is built once per pair and passed to the same `_fill` routine that the
unconstrained variant uses. That separation is what keeps seven variants from
becoming seven copies of the recursion: only the mask, the endpoint convention,
and the min-versus-softmin choice change.

### The coarse pass

```python
        coarse_x = self._reduce(x)
        coarse_y = self._reduce(y)
        coarse_mask = self._fast_mask(coarse_x, coarse_y, radius)
        coarse_cost = self._costs(coarse_x, coarse_y)
        coarse_path, _ = self._solve(coarse_cost, coarse_mask)
```

`_reduce` averages adjacent pairs and duplicates the last point when the length
is odd, so every level halves the length exactly. The recursion returns a mask,
not a distance: the distance is only ever computed at full resolution, which is
what keeps the approximation consistent with the exact algorithm's cost model.

### Accumulating a barycenter

```python
                path, value = self._exact(member, average)
                total += value
                for i, j in path:
                    accumulator[j] += member[i]
                    counts[j] += 1.0
```

Every cell of the path contributes one member point to one average index, and
the division by `counts` is the average. The `np.maximum(counts, 1.0)` in the
next line protects an index that no path visited, which happens when the average
is longer than the members; those indices keep their previous value instead of
becoming NaN.

## Complexity

Let $n,m$ be the sequence lengths, $r$ the band radius, $L$ the path length,
$S$ the number of members in a barycenter, and $I$ the DBA iterations.

| operation | time | memory |
|:--|:--|:--|
| `standard`, `subsequence` | $O(nm)$ | $O(nm)$ |
| `sakoe_chiba` | $O(nr)$ | $O(nr)$ |
| `itakura` | $O(nm)$ worst case | $O(nm)$ |
| `soft_dtw` | $O(nm)$ | $O(nm)$ |
| `soft_dtw` gradient | $O(nm)$ extra | $O(nm)$ |
| `dtw_barycenter` | $O(I\,S\,nm)$ | $O(nm)$ |
| `fast_dtw` | $O(nr)$ | $O(nr)$ |

The table is the memory bottleneck, and for long sequences it is the reason the
banded variants exist: a radius of ten on a pair of length ten thousand is four
orders of magnitude less work than the full grid. Both the table and the
backpointer array are materialised here because backtracking needs them; if only
the distance is wanted, two rows suffice and the memory drops to $O(m)$.

## Failure modes

**Unconstrained warping can match anything.** With no band, a path may map one
point of $x$ to a long stretch of $y$ — or the reverse — whenever that lowers the
sum. On sequences with slowly varying amplitude this produces alignments that
are optimal under (1) and meaningless as correspondences. The constraint is the
model; if the domain says the clocks cannot diverge by more than ten percent,
the band should say so too.

**A constraint can be infeasible on the integer grid.** The Itakura
parallelogram with a maximum slope close to the length ratio leaves less than one
cell per row, and the rounding can then leave no monotone path at all. The code
raises `no monotone path satisfies the constraint` rather than returning
infinity. In the demo, a slope of $1.3$ on a $60\times80$ pair is refused while
$1.5$ is accepted, and this is a property of the grid, not a bug.

**The band can be tight enough to be wrong.** The same demo's band of three
costs $7.8$ times the unconstrained distance. A band is a bias-variance choice
like any other: wide enough to contain the true path, narrow enough to exclude
the spurious ones, and there is no way to know which without a validation set.

**FastDTW is approximate, and coarsening is a low-pass filter.** On smooth data
a radius of two reproduces the exact answer and a radius of one is already
$3.5$ times worse. On piecewise-constant data the coarse path is actively wrong:
$75$ against an exact $50$, a $50\%$ error. Averaging neighbours destroys the
feature the alignment depends on. Use it when the sequences are smooth and the
grid is large, and check it against exact DTW on a sample.

**Soft DTW is not a distance.** It is below the hard value for every positive
$\gamma$, it can be negative, and it grows with the path length, so two pairs of
different lengths are not comparable. The demo's $\gamma=10$ value is $-1159$ on
a pair whose hard distance is $0.100$. Normalise by length, or compare only at
matched lengths and a fixed $\gamma$.

**Squared Euclidean cost is scale-sensitive.** A feature measured in thousands
dominates one measured in fractions, and the alignment will be driven entirely
by the first. Standardise the features before warping, or use a cost that is
invariant to the units; there is no such thing as an unscaled DTW distance
across heterogeneous channels.

**DTW is not a metric.** It is symmetric and non-negative, but the triangle
inequality fails: $d(x,z)$ can exceed $d(x,y)+d(y,z)$ because the optimal
alignment of $x$ to $z$ need not pass through the alignment of $y$. Anything
that assumes a metric — vantage-point trees, Lipschitz bounds, some clustering
guarantees — must use the lower bound (LB_Keogh) or a different representation
instead.

**The barycenter depends on its starting length and its members.** DBA cannot
change the length of the average, so a set of sequences of very different
lengths produces an average stuck at the medoid's length. It is also a local
method: the medoid and the iteration order matter, and a bad initial average can
trap the iteration. The demo's history is monotone, but that is a property of
this data, not a guarantee.

**Subsequence matching finds the best interval, not all of them.** If the query
occurs twice, only the cheaper occurrence is returned. Repeated patterns need
either a sliding-window scan or a different formulation; the single-interval
answer is what (4) optimises.

## Applications

**Speech recognition before HMMs.** Isolated-word recognisers aligned a test
utterance to a stored template with DTW and classified by the nearest template.
The `sakoe_chiba` and `itakura` constraints come from that literature, and the
`hmm` atom in this chapter is what replaced them once enough data existed to
estimate a generative model.

**Handwriting and shape matching.** Online handwriting is a sequence of pen
positions, and two copies of the same character differ by a monotone speed
change, which is exactly the DTW model. The same machinery matches contours and
skeleton curves in shape retrieval.

**Human activity and gesture recognition.** Accelerometer streams from wearables
are warped by speed and style; DTW with a band is the classical nearest-neighbour
classifier, and the barycenter is how prototypes are averaged.

**Time-series classification and clustering.** One-nearest-neighbour with DTW is
a famously strong baseline on the UCR archive, and DTW barycenters are the
centroids in DTW k-means. The `fast_dtw` variant exists because the archive's
sequences are long enough for the quadratic cost to matter.

**Query by humming and music retrieval.** A hummed melody is a warped, noisy
version of a score; subsequence DTW finds where it occurs. The `subsequence`
variant is the direct implementation of that query.

**Biomedical signal alignment.** ECG beats, EEG epochs and gait cycles are
compared after warping, both for clustering patients and for aligning trials
before averaging evoked responses. The barycenter is the template that a patient
is compared against.

**Imitation learning and trajectory matching.** A demonstrated trajectory and a
reproduction are compared under DTW to score the imitation without penalising a
different speed; the gradient of soft DTW is what lets the score be optimised
through.

## References

- Itakura, F. (1975). Minimum prediction residual principle applied to speech
  recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*
  23(1), 67–72. The parallelogram constraint.
- Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for
  spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal
  Processing* 26(1), 43–49. The band and the classical algorithm.
- Myers, C., Rabiner, L. & Rosenberg, A. (1980). Performance tradeoffs in dynamic
  time warping algorithms for isolated word recognition. *IEEE Transactions on
  Acoustics, Speech, and Signal Processing* 28(6), 623–635. Local constraints and
  their costs.
- Berndt, D. J. & Clifford, J. (1994). Using dynamic time warping to find
  patterns in time series. *KDD Workshop*, 359–370. Subsequence matching.
- Ratanamahatana, C. A. & Keogh, E. (2004). Making time-series classification
  more accurate using learned constraints. *SIAM International Conference on Data
  Mining*, 250–258. Learned band widths and LB_Keogh.
- Salvador, S. & Chan, P. (2007). Toward accurate dynamic time warping in linear
  time and space. *Intelligent Data Analysis* 11(5), 561–580. FastDTW.
- Müller, M. (2007). *Information Retrieval for Music and Motion*. Springer.
  The music and motion applications, with the constraint taxonomy.
- Petitjean, F., Ketterlin, A. & Gançarski, P. (2011). A global averaging method
  for dynamic time warping, with applications to clustering. *Pattern
  Recognition* 44(3), 678–693. DTW barycenter averaging.
- Cuturi, M. & Blondel, M. (2017). Soft-DTW: a differentiable loss function for
  time-series. *ICML*, 894–903. The smoothed recursion and its gradient.
