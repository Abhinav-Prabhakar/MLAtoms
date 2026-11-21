r"""Dynamic time warping — align sequences that do not share a clock.

Chapter:   10-sequences
Companion: dtw.md
Tier 1 · 7 variants: standard, sakoe_chiba, itakura, subsequence, soft_dtw,
dtw_barycenter, fast_dtw

    objective:   D(n-1, m-1) = min_path sum_(i,j) in path C(i, j)            (1)
    recursion:   D(i, j) = C(i, j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1))  (2)
    band:        |j - i (m-1)/(n-1)| <= r                                    (3)
    itakura:     v <= s u,  v >= u / s,  1-v <= s (1-u),  1-v >= (1-u)/s
    subsequence: D(0, j) = C(0, j),   distance = min_j D(n-1, j)             (4)
    soft-dtw:    D = C - gamma log sum exp(-D_pred / gamma)                   (5)
    dba:         average[j] = mean of the points warped onto index j         (6)

Two signals that mean the same thing rarely arrive on the same clock. A word
spoken slowly and the same word spoken quickly are the same sequence of shapes
with a monotone change of time; two heartbeats, two gaits, two handwritten
digits are the same. Dynamic time warping is the smallest algorithm that
compares them: find the monotone path through the cost matrix that minimises the
total mismatch, and call that sum the distance.

The cost matrix is the easy part. `C(i, j)` is the squared Euclidean distance
between the `i`-th point of one sequence and the `j`-th point of the other. The
hard part is the constraint: the path must start at `(0, 0)`, end at
`(n-1, m-1)`, and move only right, up or diagonally. Those three moves are what
make "monotone reparameterisation" concrete, and the recursion of (2) is the
shortest path through the resulting grid.

Everything else in this file is a variation on that grid. `sakoe_chiba` and
`itakura` forbid cells far from the diagonal, which is how you stop the warping
from matching the start of one sequence to the end of another. `subsequence`
frees the endpoints on one axis, which is how you find a short query inside a
long reference. `soft_dtw` replaces the minimum with a soft minimum, which makes
the distance differentiable and therefore trainable. `dtw_barycenter` averages a
set of sequences in the warped geometry, which is what you need when the
"average" of two misaligned signals is not their pointwise mean. `fast_dtw`
approximates the grid on a coarser one, which is what you need when the grid is
ten thousand cells on a side.

The demos check the parts that can be checked exactly. On sequences warped by a
known monotone map, the recovered path visits the corresponding points. On
identical sequences the distance is zero and the path is the diagonal. On a
query hidden inside a reference, `subsequence` returns the true offset while the
anchored variants cannot. Soft-DTW's gradient is checked against finite
differences to seven digits, and as the smoothing shrinks the soft distance
converges to the hard one. The barycenter of twelve warped copies of a shape
recovers the shape to a correlation above 0.99.

The number that matters in practice is the size of the constraint. An
unconstrained path can always reduce the distance by matching unrelated points,
and on longer sequences it will. A band is not an optimisation; it is the part
of the model that says which alignments are allowed.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("standard", "sakoe_chiba", "itakura", "subsequence", "soft_dtw",
            "dtw_barycenter", "fast_dtw")
STEPS = ((1, 0), (0, 1), (1, 1))


class DTW:
    """Dynamic time warping under one of seven constraint or averaging regimes.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    radius : int, optional
        Half-width, in cells, of the Sakoe-Chiba band for `sakoe_chiba`, and
        the refinement radius for `fast_dtw`. Defaults to a tenth of the longer
        sequence, or one cell for `fast_dtw`.
    max_slope : float
        Largest admissible local slope for `itakura`; the smallest is its
        reciprocal. Must exceed one.
    gamma : float
        Smoothing scale for `soft_dtw`. Smaller is closer to hard DTW.
    n_iter : int
        Maximum DBA iterations for `dtw_barycenter`.
    tol : float
        Relative improvement at which DBA stops.

    Attributes
    ----------
    cost_matrix_ : ndarray
        Local costs of the last alignment, shape ``(n, m)``.
    accumulated_ : ndarray
        Dynamic-programming table of the last alignment, shape ``(n, m)``.
    average_ : ndarray
        The barycenter, for `dtw_barycenter`.
    history_ : list of float
        Total distance to the members after each DBA iteration.
    """

    def __init__(self, *, variant: str = "standard", radius: int | None = None,
                 max_slope: float = 2.0, gamma: float = 1.0, n_iter: int = 20,
                 tol: float = 1e-8) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if radius is not None and (not isinstance(radius, (int, np.integer))
                                   or radius < 1):
            raise ValueError("radius must be a positive integer")
        if not np.isfinite(max_slope) or max_slope <= 1.0:
            raise ValueError("max_slope must be finite and greater than one")
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive")
        if n_iter < 1:
            raise ValueError("n_iter must be at least one")
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if radius is not None and variant not in ("sakoe_chiba", "fast_dtw"):
            raise ValueError("radius belongs to sakoe_chiba and fast_dtw")
        self.variant = variant
        self.radius = radius
        self.max_slope = float(max_slope)
        self.gamma = float(gamma)
        self.n_iter = n_iter
        self.tol = float(tol)

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare(sequence: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("a sequence must be one- or two-dimensional")
        if values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("a sequence needs at least one point and feature")
        if not np.all(np.isfinite(values)):
            raise ValueError("a sequence must be finite")
        return values

    def _pair(self, x: np.ndarray, y: np.ndarray) -> tuple:
        first = self._prepare(x)
        second = self._prepare(y)
        if first.shape[1] != second.shape[1]:
            raise ValueError("the sequences must have the same dimension")
        return first, second

    @staticmethod
    def _costs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - y[None, :, :]
        return np.sum(diff * diff, axis=2)

    # ------------------------------------------------------------------ #
    # constraints
    # ------------------------------------------------------------------ #
    def _band(self, n: int, m: int, radius: int) -> np.ndarray:
        """Sakoe-Chiba band of (3), widened to integer cell boundaries."""
        mask = np.zeros((n, m), dtype=bool)
        slope = (m - 1) / max(n - 1, 1)
        for i in range(n):
            centre = int(round(i * slope))
            low = max(0, centre - radius)
            high = min(m - 1, centre + radius)
            mask[i, low:high + 1] = True
        mask[0, 0] = True
        mask[-1, -1] = True
        return mask

    def _itakura(self, n: int, m: int) -> np.ndarray:
        """Itakura parallelogram: both ends constrained by ``max_slope``."""
        mask = np.zeros((n, m), dtype=bool)
        slope = self.max_slope
        for i in range(n):
            u = i / max(n - 1, 1)
            v = np.arange(m) / max(m - 1, 1)
            inside = ((v <= slope * u + 1e-12) & (v >= u / slope - 1e-12)
                      & (1.0 - v <= slope * (1.0 - u) + 1e-12)
                      & (1.0 - v >= (1.0 - u) / slope - 1e-12))
            mask[i] = inside
        mask[0, 0] = True
        mask[-1, -1] = True
        return mask

    def _mask(self, n: int, m: int) -> np.ndarray:
        if self.variant == "sakoe_chiba":
            radius = self.radius if self.radius is not None else max(
                1, int(0.1 * max(n, m)))
            return self._band(n, m, radius)
        if self.variant == "itakura":
            return self._itakura(n, m)
        return np.ones((n, m), dtype=bool)

    # ------------------------------------------------------------------ #
    # the grid
    # ------------------------------------------------------------------ #
    def _fill(self, cost: np.ndarray, mask: np.ndarray,
              free_start: bool) -> tuple:
        """Fill the table of (2) and record every step it took."""
        n, m = cost.shape
        table = np.full((n, m), np.inf)
        back = np.full((n, m, 2), -1, dtype=int)
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
        if not free_start:
            for j in range(1, m):
                if mask[0, j] and np.isfinite(table[0, j - 1]):
                    table[0, j] = cost[0, j] + table[0, j - 1]
                    back[0, j] = (0, j - 1)
        return table, back

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

    def _solve(self, cost: np.ndarray, mask: np.ndarray,
               free_start: bool = False, end: tuple | None = None) -> tuple:
        table, back = self._fill(cost, mask, free_start)
        target = (cost.shape[0] - 1, cost.shape[1] - 1) if end is None else end
        if not np.isfinite(table[target]):
            raise ValueError("no monotone path satisfies the constraint")
        return self._backtrack(back, target, free_start), table

    def _exact(self, x: np.ndarray, y: np.ndarray) -> tuple:
        cost = self._costs(x, y)
        mask = self._mask(*cost.shape)
        path, table = self._solve(cost, mask)
        self.cost_matrix_ = cost
        self.accumulated_ = table
        return path, float(table[-1, -1])

    # ------------------------------------------------------------------ #
    # variants
    # ------------------------------------------------------------------ #
    def _subsequence(self, x: np.ndarray, y: np.ndarray) -> tuple:
        if x.shape[0] > y.shape[0]:
            raise ValueError("the query must be no longer than the reference")
        cost = self._costs(x, y)
        mask = np.ones_like(cost, dtype=bool)
        table, back = self._fill(cost, mask, free_start=True)
        ends = table[-1]
        if not np.any(np.isfinite(ends)):
            raise ValueError("the query cannot be matched anywhere")
        end = int(np.argmin(ends))
        path = self._backtrack(back, (x.shape[0] - 1, end), free_start=True)
        self.cost_matrix_ = cost
        self.accumulated_ = table
        self.match_interval_ = (int(path[0][1]), end)
        return path, float(table[-1, end])

    def _fast(self, x: np.ndarray, y: np.ndarray) -> tuple:
        radius = self.radius if self.radius is not None else 1
        cost = self._costs(x, y)
        mask = self._fast_mask(x, y, radius)
        path, table = self._solve(cost, mask)
        self.cost_matrix_ = cost
        self.accumulated_ = table
        return path, float(table[-1, -1])

    def _fast_mask(self, x: np.ndarray, y: np.ndarray, radius: int) -> np.ndarray:
        """Coarsen, align, project, refine — Salvador and Chan's recursion."""
        n, m = x.shape[0], y.shape[0]
        if max(n, m) <= 4 or min(n, m) <= 1:
            return np.ones((n, m), dtype=bool)
        coarse_x = self._reduce(x)
        coarse_y = self._reduce(y)
        coarse_mask = self._fast_mask(coarse_x, coarse_y, radius)
        coarse_cost = self._costs(coarse_x, coarse_y)
        coarse_path, _ = self._solve(coarse_cost, coarse_mask)
        mask = np.zeros((n, m), dtype=bool)
        for ci, cj in coarse_path:
            i0 = min(n - 1, 2 * int(ci))
            j0 = min(m - 1, 2 * int(cj))
            low_i, high_i = max(0, i0 - radius), min(n, i0 + radius + 1)
            low_j, high_j = max(0, j0 - radius), min(m, j0 + radius + 1)
            mask[low_i:high_i, low_j:high_j] = True
        mask[0, 0] = True
        mask[-1, -1] = True
        return mask

    @staticmethod
    def _reduce(sequence: np.ndarray) -> np.ndarray:
        length = sequence.shape[0]
        if length % 2 == 1:
            sequence = np.vstack([sequence, sequence[-1]])
            length += 1
        return 0.5 * (sequence[0::2] + sequence[1::2])

    def _soft(self, x: np.ndarray, y: np.ndarray) -> tuple:
        cost = self._costs(x, y)
        n, m = cost.shape
        table = np.full((n, m), np.inf)
        table[0, 0] = cost[0, 0]
        for i in range(1, n):
            table[i, 0] = cost[i, 0] + table[i - 1, 0]
        for j in range(1, m):
            table[0, j] = cost[0, j] + table[0, j - 1]
        gamma = self.gamma
        for i in range(1, n):
            for j in range(1, m):
                previous = table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
                lowest = min(previous)
                total = sum(math.exp(-(p - lowest) / gamma) for p in previous)
                table[i, j] = cost[i, j] + lowest - gamma * math.log(total)
        self.cost_matrix_ = cost
        self.accumulated_ = table
        return table, float(table[-1, -1])

    def _soft_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Derivative of the soft distance with respect to the first sequence."""
        self._soft(x, y)
        table = self.accumulated_
        n, m = table.shape
        adjoint = np.zeros((n, m))
        adjoint[n - 1, m - 1] = 1.0
        gamma = self.gamma
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if i == 0 and j == 0:
                    continue
                weight = adjoint[i, j]
                if weight == 0.0:
                    continue
                if i == 0:
                    adjoint[0, j - 1] += weight
                elif j == 0:
                    adjoint[i - 1, 0] += weight
                else:
                    previous = np.array([table[i - 1, j], table[i, j - 1],
                                         table[i - 1, j - 1]])
                    weights = np.exp(-(previous - previous.min()) / gamma)
                    weights /= weights.sum()
                    adjoint[i - 1, j] += weight * weights[0]
                    adjoint[i, j - 1] += weight * weights[1]
                    adjoint[i - 1, j - 1] += weight * weights[2]
        return 2.0 * (adjoint.sum(axis=1)[:, None] * x - adjoint @ y)

    # ------------------------------------------------------------------ #
    # public surface
    # ------------------------------------------------------------------ #
    def align(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Return the optimal monotone path and its total cost."""
        if self.variant == "soft_dtw":
            raise ValueError("soft_dtw is differentiable but has no single path")
        first, second = self._pair(x, y)
        if self.variant == "subsequence":
            return self._subsequence(first, second)
        if self.variant == "fast_dtw":
            return self._fast(first, second)
        return self._exact(first, second)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Warped distance between two sequences."""
        first, second = self._pair(x, y)
        if self.variant == "soft_dtw":
            return self._soft(first, second)[1]
        path, value = self.align(first, second)
        del path
        return value

    def soft_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of the soft distance with respect to `x`."""
        if self.variant != "soft_dtw":
            raise ValueError("the gradient belongs to the soft_dtw variant")
        first, second = self._pair(x, y)
        return self._soft_gradient(first, second)

    def fit(self, sequences) -> DTW:
        """DTW barycenter averaging of a set of sequences, equation (6)."""
        if self.variant != "dtw_barycenter":
            raise ValueError("fit belongs to the dtw_barycenter variant")
        members = self._as_sequences(sequences)
        dimension = members[0].shape[1]
        for member in members:
            if member.shape[1] != dimension:
                raise ValueError("every member must have the same dimension")
        medoid, best = 0, np.inf
        for i, member in enumerate(members):
            total = sum(self._exact(member, other)[1] for other in members)
            if total < best:
                medoid, best = i, total
        average = members[medoid].copy()
        history = []
        for _ in range(self.n_iter):
            accumulator = np.zeros_like(average)
            counts = np.zeros(average.shape[0])
            total = 0.0
            for member in members:
                path, value = self._exact(member, average)
                total += value
                for i, j in path:
                    accumulator[j] += member[i]
                    counts[j] += 1.0
            refined = accumulator / np.maximum(counts, 1.0)[:, None]
            history.append(total)
            shift = float(np.abs(refined - average).max())
            average = refined
            if shift <= self.tol * max(1.0, float(np.abs(average).max())):
                break
        final = sum(self._exact(member, average)[1] for member in members)
        history.append(final)
        self.average_ = average
        self.history_ = history
        return self

    def _as_sequences(self, sequences) -> list:
        if isinstance(sequences, np.ndarray) and sequences.ndim == 3:
            raw = [sequences[i] for i in range(sequences.shape[0])]
        elif isinstance(sequences, (list, tuple)):
            raw = list(sequences)
        else:
            raise ValueError("sequences must be a list or a three-dimensional array")
        if len(raw) < 2:
            raise ValueError("a barycenter needs at least two sequences")
        return [self._prepare(item) for item in raw]

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _shape(t: np.ndarray) -> np.ndarray:
    """A smooth signal with two harmonics, easy to warp and recognise."""
    return np.sin(2.0 * np.pi * t) + 0.3 * np.sin(6.0 * np.pi * t)


def _warped_pair(amplitude: float) -> tuple:
    """A shape and the same shape sampled through a known monotone map."""
    base = np.linspace(0.0, 1.0, 60)
    clock = np.linspace(0.0, 1.0, 80)
    warped = clock + amplitude * np.sin(2.0 * np.pi * clock) / (2.0 * np.pi)
    return _shape(base), _shape(warped), base, warped


def _demo_standard() -> None:
    """The grid, its closed forms, and a path that follows the warp."""
    model = DTW()
    path, distance = model.align(np.arange(4.0), np.arange(4.0))
    print("standard DTW, four closed-form cases:")
    print(f"  identical sequences        {distance:.6f}   path {path.tolist()}")
    assert distance == 0.0
    assert np.array_equal(path, np.column_stack([np.arange(4), np.arange(4)]))

    stretched = np.repeat(np.arange(4.0), 2)
    _, stretched_distance = model.align(np.arange(4.0), stretched)
    print(f"  every point doubled        {stretched_distance:.6f}")
    assert stretched_distance == 1.0

    _, constant = model.align(np.zeros(3), np.ones(4))
    print(f"  unit cost, lengths 3 and 4 {constant:.6f}   (= max(n, m))")
    assert constant == 4.0

    for (n, m) in ((3, 4), (4, 4), (2, 5)):
        first = np.arange(n, dtype=float)
        second = np.arange(m, dtype=float) + 0.5
        best = np.inf
        for candidate in _all_paths(n, m):
            cost = sum((first[i] - second[j]) ** 2 for i, j in candidate)
            best = min(best, cost)
        assert abs(model.distance(first, second) - best) < 1e-12
    print("  three small grids match exhaustive enumeration over paths")

    first, second, base, warped = _warped_pair(0.15)
    path, distance = model.align(first, second)
    error = float(np.abs(base[path[:, 0]] - warped[path[:, 1]]).max())
    steps = np.diff(path, axis=0)
    assert np.all((steps == (1, 0)) | (steps == (0, 1)) | (steps == (1, 1)))
    print("  warped shape, 60 against 80 points:")
    print(f"    distance                 {distance:.5f}")
    print(f"    path length              {path.size // 2}")
    print(f"    largest clock error      {error:.5f}")
    assert path[0, 0] == 0 and path[-1, 0] == 59 and path[-1, 1] == 79
    assert error < 0.02
    print("  The path is monotone, spans both sequences, and pairs points that")
    print("  are a tenth of a cycle apart at worst: the warp is recovered.")


def _all_paths(n: int, m: int) -> list:
    """Every monotone path on an n-by-m grid, for exhaustive checks."""
    paths = []

    def walk(i: int, j: int, trail: list) -> None:
        if i == n - 1 and j == m - 1:
            paths.append(list(trail))
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            if i + di < n and j + dj < m:
                trail.append((i + di, j + dj))
                walk(i + di, j + dj, trail)
                trail.pop()

    walk(0, 0, [(0, 0)])
    return paths


def _demo_bands() -> None:
    """Constraints are the model of which alignments are allowed."""
    first, second, _, _ = _warped_pair(0.6)
    print("strongly warped shape, 60 against 80 points, allowed cells per grid:")
    print("  variant        radius   slope   cells   distance")
    rows = []
    for variant, radius, slope in (
            ("standard", None, None),
            ("sakoe_chiba", 3, None),
            ("sakoe_chiba", 6, None),
            ("sakoe_chiba", 10, None),
            ("itakura", None, 1.5),
            ("itakura", None, 2.0)):
        model = DTW(variant=variant, radius=radius,
                    **({} if slope is None else {"max_slope": slope}))
        _, distance = model.align(first, second)
        cells = int(model._mask(60, 80).sum())
        rows.append((variant, radius, slope, cells, distance))
        label = f"{variant:<13s}"
        radius_text = "  -  " if radius is None else f"{radius:5d}"
        slope_text = "  -  " if slope is None else f"{slope:5.1f}"
        print(f"  {label} {radius_text}   {slope_text}  {cells:5d}   {distance:8.5f}")
    standard = rows[0][4]
    assert rows[1][4] > 3.0 * standard
    assert abs(rows[3][4] - standard) < 1e-6
    assert rows[4][4] > standard
    assert abs(rows[5][4] - standard) < 1e-6
    assert rows[1][3] < rows[2][3] < rows[3][3]
    print("  A band of ten cells contains the optimal path and changes nothing;")
    print("  a band of three forces the path onto the diagonal and costs eight")
    print("  times the distance. The constraint is the prior about which pairs")
    print("  of points are allowed to correspond, and it is not free.")


def _demo_subsequence() -> None:
    """Finding a query inside a reference requires free endpoints."""
    rng = np.random.default_rng(0)
    query = np.sin(np.linspace(0.0, 4.0, 30))
    reference = np.concatenate([np.zeros(20),
                                query + 0.02 * rng.standard_normal(30),
                                np.zeros(25)])
    anchored = DTW().distance(query, reference)
    model = DTW(variant="subsequence")
    path, distance = model.align(query, reference)
    print("a 30-point query hidden at offset 20 of a 75-point reference:")
    print(f"  anchored DTW distance      {anchored:.4f}")
    print(f"  subsequence distance       {distance:.4f}")
    print(f"  matched interval           {model.match_interval_}")
    assert distance < 0.02
    assert anchored > 3.0
    assert model.match_interval_[0] == 20
    assert path[0][0] == 0 and path[-1][0] == 29
    print("  Anchored DTW must match the ends to the ends, so the flat padding")
    print("  dominates. Freeing the start and the end turns the problem into")
    print("  \"where is this pattern\", and the answer is the true offset.")


def _demo_soft() -> None:
    """A differentiable distance, checked against finite differences."""
    first, second, _, _ = _warped_pair(0.6)
    hard = DTW().distance(first, second)
    print("soft DTW against hard DTW on the strongly warped pair:")
    print("  gamma        distance      gap to hard")
    distances = []
    for gamma in (10.0, 1.0, 0.1, 0.01, 1e-4):
        value = DTW(variant="soft_dtw", gamma=gamma).distance(first, second)
        distances.append((gamma, value))
        print(f"  {gamma:9.4f}   {value:11.4f}   {value - hard:11.4f}")
        assert value <= hard + 1e-9
    ordered = [value for _, value in distances]
    assert np.all(np.diff(ordered) > 0.0)
    assert abs(distances[-1][1] - hard) < 0.005

    small_first = _shape(np.linspace(0.0, 1.0, 12))[:, None]
    small_second = _shape(np.linspace(0.05, 1.05, 15))[:, None]
    model = DTW(variant="soft_dtw", gamma=0.5)
    gradient = model.soft_gradient(small_first, small_second)
    epsilon = 1e-6
    numeric = np.zeros_like(small_first)
    for i in range(small_first.shape[0]):
        plus = small_first.copy()
        minus = small_first.copy()
        plus[i, 0] += epsilon
        minus[i, 0] -= epsilon
        numeric[i, 0] = (model.distance(plus, small_second)
                         - model.distance(minus, small_second)) / (2 * epsilon)
    error = float(np.abs(gradient - numeric).max())
    print(f"  gradient against finite differences, 12 points: {error:.2e}")
    assert error < 1e-6
    print("  The soft minimum is below the hard one for every positive gamma and")
    print("  rises to it as gamma falls, at about gamma times the path length per")
    print("  cell. The gradient is exact, which is the point: soft DTW is a layer")
    print("  you can put in a network, and hard DTW is not.")


def _demo_barycenter() -> None:
    """Averaging under the warp, and why the pointwise mean is wrong."""
    rng = np.random.default_rng(0)
    template = _shape(np.linspace(0.0, 1.0, 50))
    members = []
    for _ in range(12):
        clock = np.linspace(0.0, 1.0, 50)
        jitter = rng.uniform(-0.12, 0.12)
        members.append(_shape(clock + jitter * np.sin(2.0 * np.pi * clock)
                              / (2.0 * np.pi)))
    model = DTW(variant="dtw_barycenter", n_iter=20).fit(members)
    history = model.history_
    print("twelve warped copies of one shape, DBA from the medoid:")
    print(f"  total distance by iteration  {[round(v, 4) for v in history]}")
    print(f"  correlation with the template {np.corrcoef(model.average_.ravel(), template)[0, 1]:.6f}")
    assert np.all(np.diff(history) <= 1e-9)
    assert history[-1] < history[0]
    correlation = float(np.corrcoef(model.average_.ravel(), template)[0, 1])
    assert correlation > 0.99

    pointwise = np.mean(members, axis=0)
    mean_total = sum(DTW().distance(pointwise, member) for member in members)
    print(f"  total distance to the pointwise mean {mean_total:.4f}")
    print(f"  total distance to the barycenter     {history[-1]:.4f}")
    assert history[-1] < mean_total
    print("  The pointwise mean averages points that do not correspond, so it")
    print("  blurs the peaks; the barycenter first aligns each member, then")
    print("  averages what landed in the same place, and recovers the shape.")


def _demo_fast() -> None:
    """The approximation is a restriction of the grid, so it can only cost."""
    first, second, _, _ = _warped_pair(0.6)
    exact = DTW().distance(first, second)
    print("FastDTW against exact DTW on the warped pair:")
    print("  radius   distance   ratio to exact")
    for radius in (1, 2):
        model = DTW(variant="fast_dtw", radius=radius)
        path, value = model.align(first, second)
        print(f"  {radius:6d}   {value:8.5f}   {value / exact:8.3f}")
        assert value >= exact - 1e-9
        assert np.isfinite(value)
        best = sum(model.cost_matrix_[i, j] for i, j in path)
        assert abs(best - value) < 1e-9
    assert DTW(variant="fast_dtw", radius=2).distance(first, second) < exact * 1.001

    rng = np.random.default_rng(1)
    blocky = np.repeat(rng.integers(0, 3, 20).astype(float), 5)
    blocky_two = np.repeat(rng.integers(0, 3, 24).astype(float), 5)
    blocky_exact = DTW().distance(blocky, blocky_two)
    blocky_fast = DTW(variant="fast_dtw", radius=1).distance(blocky, blocky_two)
    print("piecewise-constant pair, radius one:")
    print(f"  exact {blocky_exact:.2f}   fast {blocky_fast:.2f}")
    assert blocky_fast > blocky_exact
    print("  Coarsening averages neighbours, which is exactly what a smooth")
    print("  signal tolerates and a block signal does not: the coarse path")
    print("  misses the true alignment and the fine pass cannot recover it.")
    print("  The approximation is always a restriction of the grid, so its")
    print("  distance is never below the exact one — but it can be far above.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    first, second, _, _ = _warped_pair(0.4)
    calls = [
        lambda: DTW(variant="nope"),
        lambda: DTW(variant="sakoe_chiba", radius=0),
        lambda: DTW(variant="sakoe_chiba", radius=1.5),
        lambda: DTW(variant="itakura", max_slope=1.0),
        lambda: DTW(variant="soft_dtw", gamma=0.0),
        lambda: DTW(n_iter=0),
        lambda: DTW(tol=-1.0),
        lambda: DTW(radius=3),
        lambda: DTW().fit([first, second]),
        lambda: DTW(variant="soft_dtw").align(first, second),
        lambda: DTW().soft_gradient(first, second),
        lambda: DTW().distance(np.array([]), second),
        lambda: DTW().distance(np.zeros((2, 3, 4)), second),
        lambda: DTW().distance(np.array([0.0, np.nan, 1.0]), second),
        lambda: DTW().distance(np.zeros((5, 2)), np.zeros(6)),
        lambda: DTW(variant="subsequence").align(np.zeros(30), np.zeros(20)),
        lambda: DTW(variant="itakura", max_slope=1.3).distance(first, second),
        lambda: DTW(variant="dtw_barycenter").fit([first]),
        lambda: DTW(variant="dtw_barycenter").fit([first, np.zeros((40, 2))]),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, a zero "
          f"and a fractional radius, a slope of one, a zero smoothing scale, "
          f"zero iterations, a negative tolerance, a radius outside the banded "
          f"variants, a barycenter fit on the wrong variant, a discrete path "
          f"from soft DTW, a gradient on hard DTW, an empty, three-dimensional "
          f"and non-finite sequence, mismatched dimensions, a query longer "
          f"than its reference, an infeasible parallelogram, and a barycenter "
          f"of one sequence or of mixed dimensions.")


def main() -> None:
    _demo_standard()
    _demo_bands()
    _demo_subsequence()
    _demo_soft()
    _demo_barycenter()
    _demo_fast()
    _demo_broken()
    print("dtw demo passed")


if __name__ == "__main__":
    main()
