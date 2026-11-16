# Graphical models

`pgm.py` · Probabilistic models · Tier 2 · 6 variants

## Intuition

Every algorithm in this file is two operations: **multiply two potentials**, and
**sum one variable out**. That is the whole subject.

Take three variables and three factors:

```
  f₁(x₁)      f₂(x₁,x₂)      f₃(x₂,x₃)
     │             │              │
    (x₁)─────────(x₂)──────────(x₃)
```

The joint is the product of the factors divided by a constant,
$p(x) = \frac{1}{Z}\prod_a f_a(x_a)$. To find $p(x_3)$, you could evaluate all
$k^3$ assignments and add up the ones with each value of $x_3$. Or you could
notice that the multiplication distributes over addition:

$$p(x_3) \propto \sum_{x_2} f_3(x_2,x_3)\underbrace{\sum_{x_1} f_1(x_1)f_2(x_1,x_2)}_{g(x_2)}$$

The inner sum does not know about $x_3$, so it is computed once and reused. That
is variable elimination, and for a chain it turns $k^3$ into $O(k^2)$.

The catch is the intermediate factor $g(x_2)$: it exists only because $x_2$ was
shared by two factors. Eliminate a variable that appears in many factors and
this new factor spans *all* of their other variables. On a graph with treewidth
$w$, the largest intermediate has $w+1$ variables and holds $k^{w+1}$ numbers.
Inference is not exponential in the number of variables; it is exponential in
the width of the graph, and the width depends on the order you eliminate them
in.

That single sentence explains the whole file. Variable elimination picks an
order and pays once. Belief propagation reuses every message twice, which is
exact on trees and wrong on anything with a cycle. A junction tree merges
variables into cliques until the graph *is* a tree, and is exact everywhere it
can afford to build one. Max-product swaps the sum for a max, at the same cost.
Brute-force enumeration is the reference that cannot be wrong.

The demo's cost comparison is the point in one line: on an 8-node fan, the
natural order peaks at a clique of 8 (510 table entries), min-fill at 3 (54).
Same answer, 10× less work. On the 3×3 Ising grid, loopy BP converges to a fixed
point whose marginals are off by 0.064, while the junction tree is exact.

## Variants

| Variant | Entry point | Returns | Exact? | Cost |
|:--|:--|:--|:--|:--|
| `factor_graph` | `marginals`, `maximum` | everything, by enumeration | yes | $O(k^n)$ |
| `variable_elimination` | `marginals` | one marginal per query | yes | $O(nk^{w+1})$ |
| `sum_product` | `marginals` | all marginals, two passes | yes, trees only | $O(Ek^2)$ |
| `belief_propagation` | `marginals` | all marginals, iterated | no | $O(TEk^2)$ |
| `junction_tree` | `marginals` | all marginals, calibrated | yes | $O(\sum_C k^{\lvert C\rvert})$ |
| `max_product` | `maximum` | MAP assignment + score | yes | $O(nk^{w+1})$ |

The two exact all-marginal methods differ in what they are willing to assume.
`sum_product` is the cheapest thing in the file and refuses a graph with a
cycle, because on a cycle it would return a number without a meaning. The
junction tree costs what the width costs and never refuses.

`belief_propagation` is the same message updates with the tree assumption
dropped. It is the only approximate method here, and the only one whose output
you should sanity-check against something exact on a smaller instance.

`factor_graph` is not an algorithm anyone would use: it is the definition of the
model, evaluated the obvious way. Every assertion in the demo is ultimately
against it.

## Derivation

### 1. The model and the two operations

A factor graph over variables $x_1,\dots,x_n$ with cardinalities $k_i$ is a set
of factors, each a non-negative table over a scope:

$$p(x) = \frac{1}{Z}\prod_{a=1}^{m} f_a(x_{a}), \qquad Z = \sum_{x}\prod_{a=1}^{m} f_a(x_{a}) \qquad (1)$$

A factor is not a distribution and its entries do not sum to one; the
unnormalised joint is the product of all of them. Two operations are enough to
do everything:

- **product**: $h(x_{A\cup B}) = f(x_A)g(x_B)$ — align the two tables over the
  union of their scopes and multiply pointwise;
- **sum-out**: $h(x_{A\setminus\{v\}}) = \sum_{x_v} f(x_A)$ — add along one axis.

Marginals are a product followed by sums, the partition function is sums all the
way down, and the MAP is the same with $\max$ in place of $\sum$.

### 2. Variable elimination and the induced width

Eliminating $v$ from a set of factors means: collect every factor that mentions
$v$ (its *bucket*), multiply them, and sum $v$ out of the result.

$$g_v\big(B_v\big) = \sum_{x_v}\prod_{a \in \mathcal{B}(v)} f_a, \qquad B_v = \bigcup_{a\in\mathcal{B}(v)}\mathrm{scope}(a)\setminus\{v\} \qquad (2)$$

The new factor lives on $B_v$: every neighbour of $v$ in the *primal graph*
(two variables joined when some factor contains both). Eliminating $v$ therefore
makes all of $B_v$ a clique, whether or not those edges existed before — the
fill-in. The cost of the step is proportional to $k^{\lvert B_v\rvert+1}$, and
the largest $\lvert B_v\rvert$ over the sweep is the **induced width** $w$ of
that order. Total time is $O(nk^{w+1})$, memory $O(k^{w+1})$.

The smallest achievable $w$ over all orders is the treewidth of the graph.
Finding it is NP-hard, so real systems use a greedy heuristic; this file uses
min-fill, which at each step eliminates the variable whose elimination adds the
fewest new edges:

$$\text{choose } v = \arg\min_{v}\ \#\{(a,b) : a,b \in N(v) \cap U,\ (a,b) \notin E\} \qquad (3)$$

It is a heuristic, and on a fan it is dramatic: the hub touches everything, so
eliminating it first makes its neighbours a clique of size $n$; eliminating the
leaves first never creates anything.

To get $p(x_q)$, eliminate every variable except $x_q$ and normalise the
leftover factor. To get $Z$, eliminate everything and keep the scalar.

### 3. Sum-product on a tree

On a tree, the same sums can be organised so that each one is computed once and
sent in both directions. Factor $a$ with scope $S_a$ sends to $x \in S_a$:

$$m_{a\to x}(x) = \sum_{x_{S_a\setminus x}} f_a(x_{S_a})\prod_{y \in S_a\setminus x} m_{y\to a}(y) \qquad (4)$$

and variable $x$ sends to factor $a$:

$$m_{x\to a}(x) = \prod_{b \in N(x)\setminus a} m_{b\to x}(x), \qquad b(x) \propto \prod_{a\in N(x)} m_{a\to x}(x) \qquad (5)$$

Exactness is an induction. Root the tree anywhere. In (4), every message
$m_{y\to a}$ is a sum over the subtree hanging off $y$, which does not touch the
subtree hanging off $x$ — so it is the exact partial sum of (1) over that
subtree, and the full belief at the root is the exact sum over everything. The
induction only works because "the subtree hanging off $y$" and "the subtree
hanging off $x$" are disjoint, which is a statement about the graph having no
cycles.

Two passes suffice: one from the leaves to the root, one back. That is
`sum_product`.

### 4. Loopy belief propagation

The equations (4)–(5) do not mention trees. Run them anyway on a graph with
cycles, iterating until the messages stop changing, and you get a fixed point.
It is not the answer to (1) in general. It is the stationary point of the Bethe
free energy,

$$F_{\text{Bethe}}(b) = \sum_a \sum_{x_{S_a}} b_a(x_{S_a})\log\frac{b_a(x_{S_a})}{f_a(x_{S_a})} - \sum_i (d_i-1)\sum_{x_i} b_i(x_i)\log b_i(x_i) \qquad (6)$$

where $d_i$ is the degree of variable $i$. On a tree the Bethe free energy is
exact (the correction terms are zero by the running-intersection property), so
its stationary point is the true posterior. On a cycle the approximation treats
the two paths between a pair of variables as if they were independent, which is
exactly the double-counting the correction term was supposed to remove.

The error is not small in general and is not detectably small from the outside:
on the demo's 3×3 Ising grid (coupling 1.4, field 0.3) the fixed point is off by
0.064 per marginal, in probability, after 145 damped sweeps — and the messages
converged cleanly. Convergence is not correctness. That is the whole reason
`sum_product` raises instead of returning a number on a cyclic graph.

### 5. Junction trees

Here is the fix for loops: **make the graph a tree by merging variables into
cliques**. Run the elimination order symbolically and keep, for each variable,
the clique $C_v = \{v\}\cup B_v$ from (2). The set of these cliques, connected so
that each one hangs off the first later clique containing its separator
$C_v \setminus \{v\}$, has the **running intersection property**:

$$x \in C_i \cap C_k \implies x \in C_j \text{ for every } C_j \text{ on the path from } C_i \text{ to } C_k \qquad (7)$$

which is precisely the property that made the tree induction in §3 work. To
calibrate, attach each original factor to a clique containing its scope (the one
where it was first eliminated), multiply the factors in each clique to get
$\psi_C$, and run the *same* sum-product messages between cliques. After
calibration, the belief of clique $C$ is $\prod \psi \times$ all incoming
messages, and marginalising a variable is a sum over its clique.

The result is exact for any graph, at the price of tables of size
$k^{\lvert C\rvert}$. The junction tree is variable elimination with the
intermediate factors reused instead of thrown away.

### 6. Max-product and the traceback

$$\log p(x) = \sum_a \log f_a(x_{a}) - \log Z \qquad (8)$$

so the MAP is the assignment maximising a sum, and $Z$ is irrelevant to it. The
elimination of §2 with $\max$ instead of $\sum$ works because $\max$ distributes
over multiplication of positive numbers, and because in log space a product is a
sum:

$$\max_{x}\ \prod_a f_a = \exp\Big(\max_x \sum_a \log f_a\Big) \qquad (9)$$

In code this is one boolean: combine factors by adding when logged, and take
`max` along the eliminated axis instead of `sum`. Storing the `argmax` at each
step gives the assignment back. Backtracking works in reverse elimination order
because a variable's argmax depends only on variables eliminated *after* it, so
by the time you need its scope, every value in it is known.

The score returned is the joint at the assignment, not $Z$ plus a marginal. On
the demo's grid the MAP is all-ones at $\log p = 19.50$, and $\log Z = 19.58$.

### 7. Evidence

Clamping $x_e = e$ is not conditioning on a marginal; it is restricting every
factor that mentions $x_e$:

$$p(x \mid x_e = e) \propto \prod_a f_a(x_a)\,\mathbf{1}[x_e = e] \qquad (10)$$

In the code that is a single axis dropped from each affected factor, and the
normalising constant changes accordingly — which is why the demo's neighbour
marginal moves from 0.936 to 0.9995 when the centre of the grid is pinned. A
unary factor can collapse to a scalar; it still scales $Z$, so its log is
accumulated rather than discarded.

This is a different operation from "divide by $p(x_e)$". Dividing only the
unary factor of $x_e$ is the common bug, and it leaves the model with the wrong
normalisation; the correct restriction touches every factor in the variable's
scope.

## Code walkthrough

The representation is a frozen dataclass: a scope, and a table whose shape *is*
`cardinality[list(scope)]`. Everything downstream trusts that invariant, so
`add` checks it and rejects negative entries.

```python
        scope = tuple(int(v) for v in scope)
        table = np.asarray(table, dtype=float)
        want = tuple(int(k) for k in self.cardinality[list(scope)]) if scope else ()
        if table.shape != want:
            raise ValueError(
                f"factor over {scope} needs shape {want}, got {table.shape}"
            )
```

Aligning two factors is the one piece of numpy that is easy to get subtly
wrong. The table's axes follow its own scope, but the product's axes must follow
the union, so the table is transposed into the shared order and then reshaped
with `1`s in the missing slots:

```python
    shared = [v for v in scope if v in f.scope]
    t = f.table.transpose([f.scope.index(v) for v in shared]) if shared else f.table
    shape, k = [], 0
    for v in scope:
        if v in f.scope:
            shape.append(t.shape[k])
            k += 1
        else:
            shape.append(1)
    return t.reshape(shape)
```

`_combine` is then two lines, and the `logged` flag is the entire difference
between sum-product and max-product:

```python
        scope = tuple(sorted(set(a.scope) | set(b.scope)))
        left, right = _align(a, scope), _align(b, scope)
        return Factor(scope, left + right if logged else left * right)
```

The elimination sweep is the atom's core. Note that `pairs` is rebuilt without
the bucket *before* the new factor is pushed back — the new factor is on a
smaller scope and must not be re-eliminated:

```python
        for v in order:
            if v in ev:
                continue
            bucket = [(i, f) for i, f in pairs if v in f.scope]
            if not bucket:
                continue
            pairs = [(i, f) for i, f in pairs if v not in f.scope]
            prod = reduce(lambda a, b: self._combine(a, b, logged=logged),
                          [f for _, f in bucket])
            axis = prod.scope.index(v)
```

An isolated variable — one no factor mentions — has an empty bucket and a
uniform marginal; skipping it is right, and it is why the guard is there rather
than an assertion.

For max-product every table is logged once, up front. Logging the bucket product
instead is the bug that looks fine on the first variable: the intermediate you
push back would be in a different space from everything else in `pairs`.

```python
            with np.errstate(divide="ignore"):
                pairs = [(i, Factor(f.scope, np.log(f.table))) for i, f in pairs]
```

Min-fill is (3) in six lines. Ties break on the index, so the order is
deterministic, and the fill edges are added as they are created — the graph that
the next choice sees is the one after the current elimination, not the original.

```python
            best = min(left, key=lambda v: (self._fill(adj, v, left),
                                            len(adj[v] & left), v))
            nbrs = sorted(adj[best] & left)
            for a, b in combinations(nbrs, 2):
                adj[a].add(b)
                adj[b].add(a)
```

A factor-to-variable message is (4): multiply the factor by every incoming
variable message except the target's, then sum out every axis except the
target's. The `reshape` with `-1` in one slot is the same alignment trick as
`_align`, specialised to a 1-D message:

```python
            f = pairs[src - n][1]
            out = f.table
            for u in f.scope:
                if u != dst:
                    out = out * msg[(u, src)].reshape(
                        [-1 if w == u else 1 for w in f.scope])
            axis = f.scope.index(dst)
            return out.sum(axis=tuple(i for i in range(len(f.scope)) if i != axis))
```

The two-pass schedule is what makes `sum_product` linear in the number of edges.
A depth-first order from a root gives the tree; the reverse order computes every
message pointing at the root, and the forward order every message pointing away:

```python
        for node in reversed(order[1:]):
            msg[(node, parent[node])] = self._message(node, parent[node], msg, pairs,
                                                      var_nbrs)
        for node in order[1:]:
            msg[(parent[node], node)] = self._message(parent[node], node, msg, pairs,
                                                      var_nbrs)
```

The cycle check is the DFS itself: a node reached twice, other than through its
parent, closes a cycle. Refusing loudly is the point — the same code would
happily return a plausible wrong number.

```python
                if nxt in parent:
                    raise ValueError("sum_product needs a tree; this graph has a cycle "
                                     "— use belief_propagation or junction_tree")
```

The junction tree's structure is (7), built by asking, for each clique, which
earlier-created clique first swallows its separator. Because cliques are created
in elimination order, that question is answered by scanning forward:
satisfying it guarantees running intersection, so no verification pass is
needed.

```python
        for i in range(len(cliques)):
            if not sep[i]:
                continue
            for j in range(i + 1, len(cliques)):
                if set(sep[i]) <= set(cliques[j]):
                    parent[i] = j
                    break
```

Calibration messages must not include the message the destination already sent.
That one condition is the difference between a fixed point and a feedback loop,
and it is the classic junction-tree bug:

```python
            for (a, b), m in msg.items():
                if b == src and a != dst:
                    prod = prod * _align(Factor(_join(cliques[a], cliques[b]), m),
                                         cliques[src])
```

The traceback reverses the max-product sweep: the argmax arrays recorded during
elimination are indexed by the values of variables eliminated later, all of which
are already in `values` when the reverse loop reaches them.

```python
        values = dict(evidence or {})
        for v, scope, arg in reversed(trace):
            values[v] = int(arg[tuple(values[u] for u in scope)])
```

## Complexity

With $n$ variables of cardinality $k$, $m$ factors, induced width $w$, and $T$
BP sweeps:

| Variant | Time | Memory |
|:--|:--|:--|
| `factor_graph` | $O(k^n)$ | $O(k^n)$ |
| `variable_elimination` | $O(nk^{w+1})$ | $O(k^{w+1})$ |
| `sum_product` | $O(\sum_a \lvert S_a\rvert k^{\lvert S_a\rvert})$ | $O(mk)$ |
| `belief_propagation` | $O(Tmk^2)$ | $O(mk)$ |
| `junction_tree` | $O(2\sum_C k^{\lvert C\rvert})$ | $O(\sum_C k^{\lvert C\rvert})$ |
| `max_product` | $O(nk^{w+1})$ | $O(nk^{w+1})$ + traceback |

The width is a property of the graph, not of the algorithm, and it is the only
thing that matters at scale:

| Graph | $n$ | Treewidth $w$ | Largest table |
|:--|--:|--:|--:|
| chain | $n$ | 1 | $k^2$ |
| fan | $n$ | 2 | $k^3$ |
| $s\times s$ grid | $s^2$ | $s$ | $k^{s+1}$ |
| complete | $n$ | $n-1$ | $k^n$ |

The grid row is why exact inference stops being an option: a 100×100 Ising model
has treewidth 100, so no elimination order produces a table smaller than
$k^{101}$. Loopy BP is what people use there, and its error is what §4 is about.

For pairwise factors, `sum_product` is $O(nk^2)$ — the same as forward-backward
on a chain — and that is the reason HMMs were tractable in 1960 and general
graphical models were not.

## Failure modes

**Treewidth blow-up.** The first thing to check on a new model is `cost()`: it
returns the largest clique and the number of table entries for an order, before
any time is spent. A model whose min-fill width is 25 is not slow, it is
impossible — $2^{26}$ entries is 500 MB per factor. *Fix:* approximate (BP), or
find structure (the graph is really a chain with a few long edges, so cut them
and run exact inference per cut — the "cutset conditioning" trick).

**Loopy BP converging to the wrong answer.** On the 3×3 grid it converges in 145
sweeps and is off by 0.064 per marginal; on a frustrated model it can oscillate
and never converge at all. Convergence is a property of the iteration, not of
the answer. *Diagnostic:* damping (the demo uses 0.3), the iteration count, and
— always — a comparison against the junction tree on an instance small enough to
solve exactly. On a tree, BP is exact and any error is a bug in the code.

**Linear-space underflow.** A 100-variable chain of factors with entries around
0.1 multiplies down to $10^{-100}$ and then to zero, and every marginal becomes
`nan`. The demo's models are small enough to hide this entirely. *Fix:* work in
log space and normalise each message to sum to one — normalisation is free
because beliefs are renormalised at the end, and it keeps the products at $O(1)$.

**Conditioning on a marginal instead of clamping.** Dividing by $p(x_e)$ and
leaving the factors intact is an easy mistake that gives a plausible but wrong
normalisation, because the restriction has to touch every factor in the scope of
$x_e$. *Diagnostic:* the clamped variable's marginal should be a point mass, and
the others should renormalise — the demo asserts both.

**Confusing the MAP with the marginal mode.** `max_product` returns the most
probable *joint* assignment, not the assignment whose per-variable marginals are
largest. The two differ whenever the joint is multimodal, and the MAP can put a
variable at a value its own marginal dislikes. *Diagnostic:* compare
`best[v]` against `argmax(marginals(v)[v])`; a disagreement is the phenomenon,
not a bug.

**Zeros in the potentials.** A deterministic factor has zeros, `log(0)` is
$-\infty$, and a message whose every entry is $-\infty$ poisons the arithmetic.
The code suppresses the warning (`np.errstate`) but not the consequence.
*Diagnostic:* check for `nan` or `-inf` in the returned marginals; if a
constraint is genuinely hard, keep the potentials strictly positive and let the
numbers speak.

**A disconnected graph under `sum_product`.** A variable no factor mentions has
no edge to anything, so the DFS never reaches it and the pass would silently
skip it. The traversal counts its visits against $n + m$ and raises instead.

## Applications

**Hidden Markov models.** A chain of latent states with transition and emission
factors: `sum_product` is forward-backward, `max_product` is Viterbi, and the
junction tree is the same algorithm with wider cliques for higher-order
Markov chains. [`kalman.py`](kalman.py) is the continuous version of the same
message passing, where Gaussian factors make every message a mean and a variance.

**Conditional random fields and structured prediction.** Sequence labelling,
segmentation, and parsing are CRFs: the partition function is `log_partition`,
the marginals are per-token posteriors, and the MAP is the decoded label
sequence. The whole training loop is gradient descent on $-\log Z$ plus a
per-token loss built from the marginals.

**Error-correcting codes.** LDPC and turbo decoding are loopy BP on a factor
graph, run for a fixed number of iterations. The graph is sparse and loopy by
construction, the messages are log-likelihood ratios, and the decoder works
because the loops are long — the same approximation §4 describes, tuned until it
is good enough.

**Statistical physics.** The Ising model is a pairwise factor graph, the
partition function is the free energy, and loopy BP is the Bethe approximation
of it. The demo's 3×3 grid is the smallest case where the approximation is
visibly wrong, which is why it is there.

**Causal inference and probabilistic programming.** Variable elimination with
observed variables clamped is the engine behind do-calculus, and it is the
default inference backend in probabilistic programming systems — exact where the
graph is small, with BP or sampling substituted when `cost()` says otherwise.

**Where it is a trap.** Dense graphs (a full covariance over 50 variables has
treewidth 49), models whose factors span many variables (a clique factor over
10 variables is a table of $k^{10}$), and any graph where someone has already
written "we'll just use BP" without checking the treewidth. The order matters
more than the implementation: two implementations of VE differ by a constant,
two orders differ by an exponent.

## References

- Pearl (1988). *Probabilistic Reasoning in Intelligent Systems.* Morgan
  Kaufmann. Belief propagation, and the tree case where it is exact.
- Lauritzen & Spiegelhalter (1988). *Local computations with probabilities on
  graphical models and their application to expert systems.* JRSS-B 50, 157. The
  junction tree, and the running-intersection property (7).
- Dechter (1999). *Bucket elimination: a unifying framework for reasoning.*
  Artificial Intelligence 113, 41. Variable elimination as the one primitive
  behind dynamic programming, Viterbi and adaptive consistency.
- Kschischang, Frey & Loeliger (2001). *Factor graphs and the sum-product
  algorithm.* IEEE Transactions on Information Theory 47, 498. The message
  equations (4)–(5), and why the same update covers decoding, HMMs and Kalman
  filtering.
- Yedidia, Freeman & Weiss (2003). *Understanding belief propagation and its
  generalizations.* In *Exploring Artificial Intelligence in the New
  Millennium.* Bethe and Kikuchi free energies, and when BP is a variational
  approximation.
- Wainwright & Jordan (2008). *Graphical models, exponential families, and
  variational inference.* Foundations and Trends in Machine Learning 1, 1. The
  Bethe free energy (6) in its proper variational setting, and the tree
  exactness proof.
- Koller & Friedman (2009). *Probabilistic Graphical Models: Principles and
  Techniques.* MIT Press. Chapters 9–11: variable elimination, the clique tree,
  and the induced-width analysis.
- Bodlaender & Koster (2010). *Treewidth computations I and II.* Information and
  Computation. Why the optimal order is hard, and what the heuristics achieve.
- Viterbi (1967). *Error bounds for convolutional codes and an asymptotically
  optimum decoding algorithm.* IEEE Transactions on Information Theory 13, 260.
  Max-product on a chain, four years before the general formulation.
