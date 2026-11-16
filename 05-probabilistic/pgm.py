"""Graphical models — factor graphs, variable elimination, belief propagation
and junction trees.

Chapter:   05-probabilistic
Companion: pgm.md
Tier 2 · 6 variants: factor_graph, variable_elimination, sum_product,
belief_propagation, junction_tree, max_product

    p(x) = (1/Z) ∏_a f_a(x_a),      Z = ∑_x ∏_a f_a(x_a)

Every algorithm here is the same two operations — multiply factors, sum one
variable out — under a different schedule. That is the subject. Variable
elimination picks an order and pays for it once. Sum-product on a tree chooses
the order that reuses every message twice. A junction tree merges variables
into cliques so that a loopy graph *becomes* a tree. Max-product swaps the sum
for a max and carries a traceback. `factor_graph` is brute-force enumeration:
exponential, exact, and the only one of the six that cannot be wrong.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import combinations

import numpy as np

VARIANTS = frozenset(
    ["factor_graph", "variable_elimination", "sum_product", "belief_propagation",
     "junction_tree", "max_product"]
)


@dataclass(frozen=True)
class Factor:
    """A potential over `scope`, with `table.shape == cardinality[list(scope)]`."""

    scope: tuple[int, ...]
    table: np.ndarray


def _align(f: Factor, scope: tuple[int, ...]) -> np.ndarray:
    """`f.table` transposed and reshaped to broadcast against `scope`.

    Transposing first is what makes the reshape legal: axes of the result must
    follow `scope`, and `f.table`'s axes follow `f.scope`.
    """
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


class FactorGraph:
    """A discrete factor graph over `n_vars` variables with known cardinalities.

    Factors are potentials, not normalised distributions; a unary factor is a
    perfectly good factor, and so is a pairwise table of shape `(k, k)`. The
    product of all factors is the unnormalised joint.
    """

    def __init__(self, cardinality: int | Sequence[int] = 2):
        self.cardinality = np.atleast_1d(np.asarray(cardinality, dtype=int))
        self.n_vars = int(self.cardinality.size)
        self.factors: list[Factor] = []

    # -- construction --------------------------------------------------- #
    def add(self, scope: Sequence[int], table) -> FactorGraph:
        """Add a factor. `table.shape` must equal `cardinality[list(scope)]`."""
        scope = tuple(int(v) for v in scope)
        table = np.asarray(table, dtype=float)
        want = tuple(int(k) for k in self.cardinality[list(scope)]) if scope else ()
        if table.shape != want:
            raise ValueError(
                f"factor over {scope} needs shape {want}, got {table.shape}"
            )
        if np.any(table < 0):
            raise ValueError("potentials must be non-negative")
        self.factors.append(Factor(scope, table))
        return self

    # -- factor algebra -------------------------------------------------- #
    @staticmethod
    def _combine(a: Factor, b: Factor, *, logged: bool) -> Factor:
        """Multiply two factors, or add them if they are already logged.

        This one flag is the whole difference between sum-product and
        max-product: in log space a product is a sum, and `max` commutes with
        adding a constant, so the same elimination runs on either semiring.
        """
        scope = tuple(sorted(set(a.scope) | set(b.scope)))
        left, right = _align(a, scope), _align(b, scope)
        return Factor(scope, left + right if logged else left * right)

    @staticmethod
    def _restrict(f: Factor, evidence: dict[int, int]) -> Factor:
        """Clamp the evidence variables: drop their axes, keep the rest."""
        hits = [i for i, v in enumerate(f.scope) if v in evidence]
        if not hits:
            return f
        index = [slice(None)] * len(f.scope)
        for i in hits:
            index[i] = evidence[f.scope[i]]
        return Factor(tuple(v for v in f.scope if v not in evidence), f.table[tuple(index)])

    def _prepare(self, evidence: dict[int, int] | None):
        """Restrict to the evidence, returning `(factors, evidence, log const)`.

        A factor can collapse to a scalar under evidence. It still multiplies
        `Z`, so its log is accumulated rather than dropped.
        """
        ev = {} if evidence is None else {int(k): int(v) for k, v in evidence.items()}
        for v, value in ev.items():
            if not 0 <= value < self.cardinality[v]:
                raise IndexError(f"evidence {v}={value} outside cardinality {self.cardinality[v]}")
        pairs, const = [], 0.0
        for i, f in enumerate(self.factors):
            g = self._restrict(f, ev)
            if g.scope:
                pairs.append((i, g))
            else:
                const += float(np.log(g.table))
        return pairs, ev, const

    # -- the primitive: eliminate one variable ---------------------------- #
    def _eliminate(self, order: Sequence[int], *, op: str = "sum",
                   evidence: dict[int, int] | None = None):
        """Run one elimination sweep. Returns `(factors, log const, trace)`."""
        logged = op == "max"
        pairs, ev, const = self._prepare(evidence)
        if op == "max":
            # max-product is max-sum on logs: log(max) = max(log), and the
            # product of every factor becomes a sum. Every intermediate factor
            # stays in log space, which is the part a naive version gets wrong.
            with np.errstate(divide="ignore"):
                pairs = [(i, Factor(f.scope, np.log(f.table))) for i, f in pairs]
        trace = []
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
            if op == "sum":
                table = prod.table.sum(axis=axis)
            else:
                table = prod.table.max(axis=axis)
                trace.append((v, tuple(u for u in prod.scope if u != v),
                              prod.table.argmax(axis=axis)))
            scope = tuple(u for u in prod.scope if u != v)
            if scope:
                pairs.append((bucket[0][0], Factor(scope, table)))
            else:
                const += float(table) if op == "max" else float(np.log(table))
        return pairs, const, trace

    # -- orders and their cost -------------------------------------------- #
    def _primal(self) -> dict[int, set[int]]:
        adj = {v: set() for v in range(self.n_vars)}
        for f in self.factors:
            for a in f.scope:
                adj[a] |= {u for u in f.scope if u != a}
        return adj

    def _min_fill_order(self) -> list[int]:
        """Greedy min-fill: eliminate the variable that creates the fewest edges."""
        adj = self._primal()
        order, left = [], set(range(self.n_vars))
        while left:
            best = min(left, key=lambda v: (self._fill(adj, v, left),
                                            len(adj[v] & left), v))
            nbrs = sorted(adj[best] & left)
            for a, b in combinations(nbrs, 2):
                adj[a].add(b)
                adj[b].add(a)
            left.remove(best)
            order.append(best)
        return order

    @staticmethod
    def _fill(adj: dict[int, set[int]], v: int, left: set[int]) -> int:
        nbrs = sorted(adj[v] & left)
        return sum(1 for a, b in combinations(nbrs, 2) if b not in adj[a])

    def cost(self, order: Sequence[int] | None = None) -> tuple[int, int]:
        """`(largest clique, table entries touched)` for this elimination order.

        Both are exponential in the induced width: the largest clique decides
        whether the model is solvable at all, and the entry count decides how
        long it takes.
        """
        order = list(self._min_fill_order() if order is None else order)
        adj = self._primal()
        largest, ops, left = 1, 0, set(range(self.n_vars))
        for v in order:
            nbrs = sorted(adj[v] & left)
            largest = max(largest, len(nbrs) + 1)
            ops += int(np.prod([int(self.cardinality[u]) for u in nbrs + [v]]))
            for a, b in combinations(nbrs, 2):
                adj[a].add(b)
                adj[b].add(a)
            left.discard(v)
        return largest, ops

    # -- exact quantities -------------------------------------------------- #
    def log_partition(self, *, order: Sequence[int] | None = None,
                      evidence: dict[int, int] | None = None) -> float:
        """`log Z`, exactly, by summing every variable out."""
        order = self._min_fill_order() if order is None else order
        pairs, const, _ = self._eliminate(order, op="sum", evidence=evidence)
        if any(f.scope for _, f in pairs):
            raise ValueError("the order did not eliminate every variable")
        return const

    def _brute(self, evidence: dict[int, int] | None = None):
        """Enumerate the joint. Exponential, and the reference for everything."""
        shape = tuple(int(k) for k in self.cardinality)
        if int(np.prod(shape)) > 200_000:
            raise ValueError(
                f"brute force would enumerate {int(np.prod(shape))} states; "
                "use variable_elimination or junction_tree"
            )
        idx = np.indices(shape)
        joint = np.ones(shape)
        for f in self.factors:
            joint = joint * (f.table if not f.scope
                             else f.table[tuple(idx[v] for v in f.scope)])
        for v, value in (evidence or {}).items():
            keep = np.zeros(shape[v])
            keep[value] = 1.0
            joint = joint * keep.reshape([-1 if u == v else 1 for u in range(self.n_vars)])
        total = float(joint.sum())
        if total <= 0.0:
            raise ValueError("the model has zero mass under this evidence")
        return joint, total

    # -- marginals --------------------------------------------------------- #
    def marginals(self, variant: str = "variable_elimination", *,
                  query: Sequence[int] | None = None, order: Sequence[int] | None = None,
                  evidence: dict[int, int] | None = None,
                  max_iter: int = 200, tol: float = 1e-10,
                  damping: float = 0.5) -> dict[int, np.ndarray]:
        """Posterior marginal of each requested variable, `{v: (k_v,)}`."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        qs = list(range(self.n_vars) if query is None else query)
        if variant == "factor_graph":
            joint, total = self._brute(evidence)
            axes = tuple(u for u in range(self.n_vars))
            return {q: joint.sum(axis=tuple(u for u in axes if u != q)) / total
                    for q in qs}
        if variant == "variable_elimination":
            order = self._min_fill_order() if order is None else list(order)
            return {q: self._ve_marginal(q, order, evidence) for q in qs}
        if variant == "sum_product":
            return self._tree_bp(evidence)
        if variant == "belief_propagation":
            return self._loopy_bp(evidence, max_iter=max_iter, tol=tol,
                                  damping=damping)
        if variant == "junction_tree":
            return self._junction_tree(evidence, order)
        raise ValueError("`max_product` is the MAP variant; call .maximum()")

    def _ve_marginal(self, q: int, order: Sequence[int],
                     evidence: dict[int, int] | None) -> np.ndarray:
        if evidence and q in evidence:
            out = np.zeros(int(self.cardinality[q]))
            out[evidence[q]] = 1.0
            return out
        pairs, _, _ = self._eliminate([v for v in order if v != q], op="sum",
                                      evidence=evidence)
        table = np.ones(int(self.cardinality[q]))
        for _, f in pairs:
            if q in f.scope:
                table = table * f.table
            elif f.scope:
                raise ValueError("the order did not eliminate every non-query variable")
        total = table.sum()
        return table / total if total > 0 else np.full(table.shape, 1.0 / table.size)

    # -- belief propagation ------------------------------------------------ #
    def _topology(self, evidence: dict[int, int] | None):
        pairs, ev, _ = self._prepare(evidence)
        n = self.n_vars
        fac = {n + i: f for i, (_, f) in enumerate(pairs)}
        var_nbrs = {v: [] for v in range(n)}
        for node, f in fac.items():
            for v in f.scope:
                var_nbrs[v].append(node)
        return pairs, fac, var_nbrs

    def _message(self, src: int, dst: int, msg: dict, pairs,
                 var_nbrs: dict[int, list[int]]) -> np.ndarray:
        """One message, `src → dst`, in a factor graph whose nodes are ints."""
        n = self.n_vars
        if src >= n:                                   # factor → variable
            f = pairs[src - n][1]
            out = f.table
            for u in f.scope:
                if u != dst:
                    out = out * msg[(u, src)].reshape(
                        [-1 if w == u else 1 for w in f.scope])
            axis = f.scope.index(dst)
            return out.sum(axis=tuple(i for i in range(len(f.scope)) if i != axis))
        total = np.ones(int(self.cardinality[src]))    # variable → factor
        for g in var_nbrs[src]:
            if g != dst:
                total = total * msg[(g, src)]
        scale = total.sum()
        return total / scale if scale > 0 else total

    def _tree_bp(self, evidence: dict[int, int] | None) -> dict[int, np.ndarray]:
        """Two-pass sum-product on a tree: every message is computed once."""
        pairs, fac, var_nbrs = self._topology(evidence)
        n = self.n_vars

        def neighbours(node: int):
            return var_nbrs[node] if node < n else fac[node].scope

        parent, order = {0: None}, [0]
        for node in order:
            for nxt in neighbours(node):
                if nxt == parent[node]:
                    continue
                if nxt in parent:
                    raise ValueError("sum_product needs a tree; this graph has a cycle "
                                     "— use belief_propagation or junction_tree")
                parent[nxt] = node
                order.append(nxt)
        if len(order) != n + len(fac):
            raise ValueError("sum_product needs a connected graph; this one is disconnected")

        msg: dict[tuple[int, int], np.ndarray] = {}
        for node in reversed(order[1:]):
            msg[(node, parent[node])] = self._message(node, parent[node], msg, pairs,
                                                      var_nbrs)
        for node in order[1:]:
            msg[(parent[node], node)] = self._message(parent[node], node, msg, pairs,
                                                      var_nbrs)

        return self._beliefs(msg, var_nbrs)

    def _beliefs(self, msg: dict, var_nbrs: dict[int, list[int]]) -> dict[int, np.ndarray]:
        """A variable's belief is the product of every message arriving at it."""
        out = {}
        for v in range(self.n_vars):
            belief = np.ones(int(self.cardinality[v]))
            for g in var_nbrs[v]:
                belief = belief * msg[(g, v)]
            total = belief.sum()
            out[v] = belief / total if total > 0 else np.full(belief.shape, 1.0 / belief.size)
        return out

    def _loopy_bp(self, evidence: dict[int, int] | None, *, max_iter: int,
                  tol: float, damping: float) -> dict[int, np.ndarray]:
        """Iterate the same message updates on a graph with cycles.

        On a tree this converges to the exact answer in a sweep; on a loopy
        graph it converges to a fixed point of the *Bethe* approximation, which
        is a different distribution.
        """
        pairs, fac, var_nbrs = self._topology(evidence)
        n = self.n_vars
        edges = [(u, node) for node, f in fac.items() for u in f.scope]
        both = edges + [(b, a) for a, b in edges]
        var_of = lambda a, b: a if a < n else b  # noqa: E731
        msg = {(a, b): np.ones(int(self.cardinality[var_of(a, b)])) for a, b in both}

        self.converged_ = False
        self.n_iter_ = 0
        for it in range(1, max_iter + 1):
            self.n_iter_ = it
            new = {e: self._message(e[0], e[1], msg, pairs, var_nbrs) for e in both}
            delta = max(float(np.abs(_unit(new[e]) - _unit(msg[e])).max()) for e in both)
            for e in both:
                msg[e] = damping * new[e] + (1.0 - damping) * msg[e]
            if delta < tol:
                self.converged_ = True
                break

        return self._beliefs(msg, var_nbrs)

    # -- junction tree ------------------------------------------------------ #
    def _junction_tree(self, evidence: dict[int, int] | None,
                       order: Sequence[int] | None) -> dict[int, np.ndarray]:
        """Exact marginals on any graph: cliques from elimination, then calibrate."""
        pairs, ev, _ = self._prepare(evidence)
        order = [v for v in (self._min_fill_order() if order is None else order)
                 if v not in ev]
        n = self.n_vars
        size = [int(k) for k in self.cardinality]

        # 1. elimination, symbolically: each bucket becomes one clique, and the
        #    induced factor it leaves behind (the separator) joins later buckets
        live = [(fid, set(f.scope)) for fid, f in pairs]
        cliques: list[tuple[int, ...]] = []
        sep: list[tuple[int, ...]] = []
        owner: dict[int, int] = {}
        for v in order:
            bucket = [item for item in live if v in item[1]]
            if not bucket:
                continue
            scope = set().union(*[s for _, s in bucket])
            live = [item for item in live if v not in item[1]]
            for fid, _ in bucket:
                if fid is not None:
                    owner.setdefault(fid, len(cliques))
            cliques.append(tuple(sorted(scope)))
            sep.append(tuple(sorted(scope - {v})))
            if len(scope) > 1:
                live.append((None, scope - {v}))
        if not cliques:
            return {v: np.full(size[v], 1.0 / size[v]) for v in range(n)}
        if len(owner) != len(pairs):
            raise ValueError("the order does not eliminate every variable; a "
                             "junction tree needs a complete order")

        # 2. the elimination tree: a clique's parent is the first later clique
        #    containing its separator. Running intersection is then automatic.
        parent: list[int | None] = [None] * len(cliques)
        for i in range(len(cliques)):
            if not sep[i]:
                continue
            for j in range(i + 1, len(cliques)):
                if set(sep[i]) <= set(cliques[j]):
                    parent[i] = j
                    break
            if parent[i] is None:
                raise ValueError("could not build a junction tree for this order")

        # 3. each factor lives in the clique that first contained its scope
        psi = [np.ones(tuple(size[v] for v in c)) for c in cliques]
        for fid, f in pairs:
            c = owner[fid]
            psi[c] = psi[c] * _align(f, cliques[c])

        # 4. calibrate: up the tree, then down. A message from `src` to `dst`
        #    sums out everything outside their separator, and must not include
        #    the message `dst` sent to `src` — that would feed a node its own
        #    answer back.
        def depth(i: int) -> int:
            return 0 if parent[i] is None else depth(int(parent[i])) + 1

        msg: dict[tuple[int, int], np.ndarray] = {}

        def message(src: int, dst: int) -> np.ndarray:
            prod = psi[src].copy()
            for (a, b), m in msg.items():
                if b == src and a != dst:
                    prod = prod * _align(Factor(_join(cliques[a], cliques[b]), m),
                                         cliques[src])
            shared = _join(cliques[src], cliques[dst])
            axes = tuple(i for i, v in enumerate(cliques[src]) if v not in shared)
            return prod.sum(axis=axes) if axes else prod

        for i in sorted(range(len(cliques)), key=lambda c: -depth(c)):
            if parent[i] is not None:
                msg[(i, parent[i])] = message(i, parent[i])
        for i in sorted(range(len(cliques)), key=lambda c: depth(c)):
            if parent[i] is not None:
                msg[(parent[i], i)] = message(parent[i], i)

        # 5. marginalise each calibrated clique down to one variable
        out = {}
        for v in range(n):
            if v in ev:
                out[v] = np.zeros(size[v])
                out[v][ev[v]] = 1.0
                continue
            hit = next((c for c in range(len(cliques)) if v in cliques[c]), None)
            if hit is None:                       # a variable no factor mentions
                out[v] = np.full(size[v], 1.0 / size[v])
                continue
            belief = psi[hit].copy()
            for (a, b), m in msg.items():
                if b == hit:
                    belief = belief * _align(Factor(_join(cliques[a], cliques[b]), m),
                                             cliques[hit])
            axes = tuple(i for i, u in enumerate(cliques[hit]) if u != v)
            marg = belief.sum(axis=axes) if axes else belief
            total = marg.sum()
            out[v] = marg / total if total > 0 else np.full(marg.shape, 1.0 / marg.size)
        return out

    # -- MAP ---------------------------------------------------------------- #
    def maximum(self, variant: str = "max_product", *,
                order: Sequence[int] | None = None,
                evidence: dict[int, int] | None = None):
        """Most probable assignment, `(assignment, log p̂)`."""
        if variant == "factor_graph":
            joint, _ = self._brute(evidence)
            flat = int(np.argmax(joint))
            return (np.array(np.unravel_index(flat, joint.shape)),
                    float(np.log(joint.flat[flat])))
        if variant != "max_product":
            raise ValueError("`maximum` supports 'max_product' and 'factor_graph'")
        order = self._min_fill_order() if order is None else order
        pairs, best, trace = self._eliminate(order, op="max", evidence=evidence)
        if any(f.scope for _, f in pairs):
            raise ValueError("the order did not eliminate every variable")
        values = dict(evidence or {})
        for v, scope, arg in reversed(trace):
            values[v] = int(arg[tuple(values[u] for u in scope)])
        # an isolated variable is in no factor, so any value is a MAP for it
        return np.array([values.get(v, 0) for v in range(self.n_vars)]), best


def _join(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """The separator of two cliques, in a canonical order."""
    return tuple(sorted(set(a) & set(b)))


def _unit(m: np.ndarray) -> np.ndarray:
    total = m.sum()
    return m / total if total > 0 else m


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _chain(rng, n: int, card: int = 2) -> FactorGraph:
    g = FactorGraph([card] * n)
    for v in range(n):
        g.add((v,), rng.random(card) + 0.05)
    for v in range(n - 1):
        g.add((v, v + 1), rng.random((card, card)) + 0.05)
    return g


def _fan(rng, n: int) -> FactorGraph:
    """A hub wired to every node of a chain: trivial for min-fill, awful in order."""
    g = FactorGraph([2] * n)
    for v in range(n):
        g.add((v,), rng.random(2) + 0.05)
    for v in range(1, n - 1):
        g.add((v, v + 1), rng.random((2, 2)) + 0.05)
    for v in range(1, n):
        g.add((0, v), rng.random((2, 2)) + 0.05)
    return g


def _ising(coupling: float) -> np.ndarray:
    w = math.exp(coupling)
    return np.array([[w, 1.0 / w], [1.0 / w, w]])


def _grid(side: int, coupling: float, field: float) -> FactorGraph:
    g = FactorGraph([2] * (side * side))
    for i in range(side):
        for j in range(side):
            v = i * side + j
            g.add((v,), np.array([1.0, math.exp(field)]))
            if i + 1 < side:
                g.add((v, v + side), _ising(coupling))
            if j + 1 < side:
                g.add((v, v + 1), _ising(coupling))
    return g


def main() -> None:
    rng = np.random.default_rng(0)

    # --- a tree: every exact method must agree with enumeration ------------
    tree = _chain(rng, n=5)
    truth = tree.marginals("factor_graph")
    worst = 0.0
    for variant in ("variable_elimination", "sum_product", "junction_tree",
                    "belief_propagation"):
        got = tree.marginals(variant)
        err = max(float(np.abs(got[v] - truth[v]).max()) for v in truth)
        assert err < 1e-10, (variant, err)
        worst = max(worst, err)
    print(
        f"pgm: factor_graph — a 5-variable chain has {2 ** 5} states; elimination, "
        f"sum-product, junction tree and BP all match enumeration to {worst:.1e}"
    )

    # --- the tree is a tree, so sum_product has to refuse a loop -----------
    loop = FactorGraph([2] * 4)
    for v in range(4):
        loop.add((v,), np.array([1.0, 2.0]))
        loop.add((v, (v + 1) % 4), np.array([[2.0, 0.5], [0.5, 2.0]]))
    try:
        loop.marginals("sum_product")
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("sum_product accepted a cycle")
    print("pgm: sum_product — refuses a 4-cycle instead of returning a wrong answer")

    # --- a 3x3 Ising grid: loops, so BP is approximate and everything else is not
    grid = _grid(side=3, coupling=1.4, field=0.3)
    exact = grid.marginals("factor_graph")
    jt = grid.marginals("junction_tree")
    ve = grid.marginals("variable_elimination")
    for name, got in (("junction_tree", jt), ("variable_elimination", ve)):
        err = max(float(np.abs(got[v] - exact[v]).max()) for v in exact)
        assert err < 1e-10, (name, err)

    bp = grid.marginals("belief_propagation", max_iter=500, damping=0.3)
    err_bp = max(float(np.abs(bp[v] - exact[v]).max()) for v in exact)
    assert grid.converged_ and 1e-4 < err_bp < 0.2, (grid.converged_, err_bp)
    print(
        f"pgm: belief_propagation — on a 3x3 Ising grid the loopy fixed point is "
        f"off by {err_bp:.4f} per marginal after {grid.n_iter_} sweeps, while the "
        f"junction tree is exact"
    )

    # --- the elimination order is the cost of the algorithm ----------------
    # A hub wired to a chain. Eliminate the hub first and its neighbours become
    # a clique; eliminate the leaves first and nothing is ever created.
    fan = _fan(rng, n=8)
    natural = list(range(fan.n_vars))
    w_nat, ops_nat = fan.cost(natural)
    w_min, ops_min = fan.cost()
    assert w_min < w_nat and ops_min < ops_nat, (w_min, w_nat, ops_min, ops_nat)
    ref = fan.marginals("factor_graph")
    for order in (natural, None):
        got = fan.marginals("variable_elimination", order=order)
        assert max(float(np.abs(got[v] - ref[v]).max()) for v in ref) < 1e-10
    print(
        f"pgm: variable_elimination — on an 8-node fan, natural order peaks at a "
        f"clique of {w_nat} ({ops_nat} entries), min-fill at {w_min} "
        f"({ops_min} entries); same answer"
    )

    # --- evidence ---------------------------------------------------------
    # Clamping is not conditioning on a marginal: it restricts every factor that
    # mentions the variable, which changes the normalisation of everything else.
    ev = {4: 1}
    exact_ev = grid.marginals("factor_graph", evidence=ev)
    assert np.isclose(exact_ev[4][1], 1.0)
    for variant in ("variable_elimination", "junction_tree"):
        got = grid.marginals(variant, evidence=ev)
        err = max(float(np.abs(got[v] - exact_ev[v]).max())
                  for v in range(grid.n_vars) if v != 4)
        assert err < 1e-10, (variant, err)
    bp_ev = grid.marginals("belief_propagation", evidence=ev, max_iter=500, damping=0.3)
    err_bp_ev = max(float(np.abs(bp_ev[v] - exact_ev[v]).max())
                    for v in range(grid.n_vars) if v != 4)
    assert err_bp_ev < 1e-3, err_bp_ev
    print(
        f"pgm: junction_tree — clamping variable 4 to 1 moves its neighbour 3's "
        f"marginal from {exact[3][1]:.4f} to {exact_ev[3][1]:.4f}; loopy BP follows "
        f"it to {err_bp_ev:.0e}"
    )

    # --- MAP by max-product, against enumeration --------------------------
    joint, _ = grid._brute()
    best, score = grid.maximum("max_product")
    ref, ref_score = grid.maximum("factor_graph")
    assert np.array_equal(best, ref), (best, ref)
    assert np.isclose(score, ref_score, atol=1e-12)
    # the score is the *joint* at the argmax, not Z plus a marginal: the
    # traceback has to recover the whole assignment, not just its probability
    assert np.isclose(score, np.log(joint[tuple(best)]), atol=1e-12)
    print(
        f"pgm: max_product — MAP {''.join(str(b) for b in best)} at log p = "
        f"{score:.4f}, matching enumeration; log Z = {grid.log_partition():.4f}"
    )

    # --- evidence changes the MAP, and the traceback has to follow ---------
    cond, cond_score = grid.maximum("max_product", evidence={0: 1 - int(best[0])})
    assert cond[0] == 1 - int(best[0]) and cond_score < score
    forced = np.log(joint[tuple(cond)])
    assert np.isclose(cond_score, forced, atol=1e-12)
    print(
        f"pgm: max_product — forcing variable 0 to {1 - int(best[0])} drops the "
        f"best log p to {cond_score:.4f}, and the traceback still recovers a "
        f"consistent assignment"
    )

    print("pgm demo passed")


if __name__ == "__main__":
    main()
