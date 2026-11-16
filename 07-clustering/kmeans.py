"""k-means — one objective, seven ways to reach a local optimum.

Chapter:   07-clustering
Companion: kmeans.md
Tier 1 · 7 variants: lloyd, kmeans_plusplus, mini_batch, elkan, bisecting,
spherical, empty_cluster_fix

k-means minimises the within-cluster sum of squares:

    J(C, z) = sum_i || x_i - c_{z_i} ||^2

For fixed centres the best assignment is the nearest centre; for fixed assignments
the best centres are the means. Alternating the two never increases J, so every
variant here is a descent method on the same objective — they differ in
initialisation, in how much of the objective they evaluate, and in the geometry
they assume.

    lloyd              alternate assign / mean until labels stop changing
    kmeans_plusplus    D^2 seeding, several restarts, keep the best inertia
    mini_batch         update centres from small batches (stochastic descent)
    elkan              use the triangle inequality to skip distance computations
    bisecting          recursive 2-means, splitting the cluster with the largest SSE
    spherical          cosine geometry: normalise centres, assign by angle
    empty_cluster_fix  re-seed empty clusters at the worst-fit point
"""

from __future__ import annotations

import itertools

import numpy as np

VARIANTS = frozenset(
    ["lloyd", "kmeans_plusplus", "mini_batch", "elkan", "bisecting",
     "spherical", "empty_cluster_fix"]
)

EPS = 1e-12


class KMeans:
    """Lloyd's algorithm and six variations.

    `init` is "kmeans++" or "random". `n_init` restarts are run and the lowest
    inertia wins. `tol` is the centre-shift threshold in Euclidean distance.
    """

    def __init__(
        self,
        variant: str = "lloyd",
        *,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 3,
        batch_size: int = 64,
        init: str | None = None,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if init is not None and init not in ("random", "kmeans++"):
            raise ValueError(f"unknown init {init!r}")
        self.variant = variant
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.batch_size = batch_size
        self.init = init or ("kmeans++" if variant == "kmeans_plusplus" else "random")
        self.rng = np.random.default_rng(random_state)
        self.n_evals_ = 0        # point-centre distances computed
        self.n_updates_ = 0      # centre-update steps taken
        self.n_rows_ = 0         # row visits (each costs one k-vector of distances)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        d = (X * X).sum(axis=1)[:, None] - 2.0 * X @ C.T + (C * C).sum(axis=1)[None]
        return np.maximum(d, 0.0)

    def _assign(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        self.n_rows_ += X.shape[0]
        if self.variant == "spherical":
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
            Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), EPS)
            self.n_evals_ += X.shape[0] * C.shape[0]
            return np.argmax(Xn @ Cn.T, axis=1)
        self.n_evals_ += X.shape[0] * C.shape[0]
        return np.argmin(self._sq_dists(X, C), axis=1)

    def _means(self, X: np.ndarray, labels: np.ndarray, prev: np.ndarray) -> np.ndarray:
        C = prev.copy()
        empty = []
        for j in range(self.k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
            elif self.variant == "empty_cluster_fix":
                empty.append(j)
        if empty:
            # Re-seed each empty cluster at a distinct worst-fit point.
            d = ((X - C[labels]) ** 2).sum(axis=1)
            order = np.argsort(-d, kind="mergesort")
            taken: set[int] = set()
            for j in empty:
                for i in order:
                    if int(i) not in taken:
                        taken.add(int(i))
                        C[j] = X[int(i)]
                        break
        return C

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _init_centres(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.init == "random":
            idx = self.rng.choice(n, size=self.k, replace=False)
            return X[idx].copy()
        # k-means++: sample proportional to squared distance to the nearest centre.
        C = [X[int(self.rng.integers(n))]]
        d = ((X - C[0]) ** 2).sum(axis=1)
        for _ in range(1, self.k):
            total = float(d.sum())
            if total <= EPS:
                C.append(X[int(self.rng.integers(n))])
            else:
                C.append(X[int(self.rng.choice(n, p=d / total))])
            d = np.minimum(d, ((X - C[-1]) ** 2).sum(axis=1))
        return np.array(C)

    # ------------------------------------------------------------------ #
    # Algorithms
    # ------------------------------------------------------------------ #
    def _lloyd(self, X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_iter):
            new_labels = self._assign(X, C)
            newC = self._means(X, new_labels, C)
            shift = float(np.sqrt(((newC - C) ** 2).sum()))
            C = newC
            self.n_updates_ += 1
            converged = np.array_equal(new_labels, labels)
            labels = new_labels
            if converged and shift <= self.tol:
                break
        return C, labels

    def _elkan(self, X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        d = self._sq_dists(X, C)
        self.n_evals_ += n * self.k
        labels = np.argmin(d, axis=1)
        u = np.sqrt(d[np.arange(n), labels])
        d[np.arange(n), labels] = np.inf
        lo = np.sqrt(d.min(axis=1))
        for _ in range(self.max_iter):
            newC = self._means(X, labels, C)
            delta = np.sqrt(((newC - C) ** 2).sum(axis=1))
            C = newC
            self.n_updates_ += 1
            u = u + delta[labels]
            lo = lo - float(delta.max())
            changed = False
            for i in range(n):
                if u[i] <= lo[i]:
                    continue
                dist = np.sqrt(((X[i] - C) ** 2).sum(axis=1))
                self.n_evals_ += self.k
                j = int(np.argmin(dist))
                u[i] = dist[j]
                dist[j] = np.inf
                lo[i] = float(dist.min())
                if j != labels[i]:
                    labels[i] = j
                    changed = True
            if not changed and float(delta.max()) <= self.tol:
                break
        return C, labels

    def _mini_batch(self, X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        counts = np.zeros(self.k)
        for _ in range(self.max_iter):
            idx = self.rng.choice(n, size=min(self.batch_size, n), replace=False)
            labels = self._assign(X[idx], C)
            for j in range(self.k):
                mask = labels == j
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                counts[j] += cnt
                eta = cnt / counts[j]
                C[j] += eta * (X[idx][mask].mean(axis=0) - C[j])
            self.n_updates_ += 1
        labels = self._assign(X, C)
        return C, labels

    def _bisecting(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clusters = [np.arange(X.shape[0])]

        def sse(idx):
            if idx.size == 0:
                return 0.0
            Z = X[idx]
            return float(((Z - Z.mean(axis=0)) ** 2).sum())

        while len(clusters) < self.k:
            sizes = np.array([sse(c) for c in clusters])
            i = int(np.argmax(sizes))
            idx = clusters.pop(i)
            if idx.size < 2:
                clusters.append(idx)
                break
            best = None
            for _ in range(3):
                sub = KMeans("lloyd", n_clusters=2, max_iter=self.max_iter,
                             tol=self.tol, init="kmeans++",
                             random_state=int(self.rng.integers(2**31 - 1)))
                C2, lab = sub._lloyd(X[idx], sub._init_centres(X[idx]))
                self.n_evals_ += sub.n_evals_
                self.n_updates_ += sub.n_updates_
                score = float(((X[idx] - C2[lab]) ** 2).sum())
                if best is None or score < best[0]:
                    best = (score, lab, C2)
            _, lab, _ = best
            clusters.extend([idx[lab == 0], idx[lab == 1]])
        labels = np.empty(X.shape[0], dtype=int)
        C = np.empty((len(clusters), X.shape[1]))
        for j, idx in enumerate(clusters):
            labels[idx] = j
            C[j] = X[idx].mean(axis=0)
        return C, labels

    def _spherical(self, X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(self.max_iter):
            labels = self._assign(X, C)
            newC = self._means(X, labels, C)
            newC /= np.maximum(np.linalg.norm(newC, axis=1, keepdims=True), EPS)
            self.n_updates_ += 1
            shift = float(np.sqrt(((newC - C) ** 2).sum()))
            C = newC
            if shift <= self.tol:
                break
        return C, self._assign(X, C)

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        self.n_evals_ = 0
        self.n_updates_ = 0
        self.n_rows_ = 0
        best = None
        runs = self.n_init
        for _ in range(runs):
            evals, updates, rows = self.n_evals_, self.n_updates_, self.n_rows_
            C = self._init_centres(X)
            if self.variant == "bisecting":
                C, labels = self._bisecting(X)
            elif self.variant == "mini_batch":
                C, labels = self._mini_batch(X, C)
            elif self.variant == "elkan":
                C, labels = self._elkan(X, C)
            elif self.variant == "spherical":
                C, labels = self._spherical(X, C)
            else:
                C, labels = self._lloyd(X, C)
            score = self._inertia(X, C, labels)
            if best is None or score < best[0]:
                best = (score, C, labels, self.n_evals_ - evals,
                        self.n_updates_ - updates, self.n_rows_ - rows)
        (_, self.cluster_centers_, self.labels_, self.n_evals_, self.n_updates_,
         self.n_rows_) = best
        self.inertia_ = float(best[0])
        return self

    def _inertia(self, X, C, labels) -> float:
        if self.variant == "spherical":
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
            Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), EPS)
            return float(1.0 - (Xn * Cn[labels]).sum(axis=1).mean())
        return float(((X - C[labels]) ** 2).sum())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(np.asarray(X, dtype=float), self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels: np.ndarray, truth: np.ndarray) -> float:
    """Best accuracy over label permutations (k! is tiny in the demos)."""
    k = int(max(labels.max(), truth.max())) + 1
    best = 0.0
    for perm in itertools.permutations(range(k)):
        acc = float(np.mean(np.asarray(perm)[labels] == truth))
        best = max(best, acc)
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    n = 450
    truth = rng.integers(0, 3, size=n)
    centers_true = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    X = centers_true[truth] + rng.normal(0.0, 0.55, size=(n, 2))

    # --- Lloyd: the reference solution ---
    km = KMeans("lloyd", n_clusters=3, n_init=10, random_state=1).fit(X)
    acc = _match_accuracy(km.labels_, truth)
    assert acc > 0.95, acc
    assert km.inertia_ < 300.0, km.inertia_
    print(f"kmeans: lloyd acc {acc:.3f}, inertia {km.inertia_:.2f}, "
          f"evals {km.n_evals_}, updates {km.n_updates_}")

    # --- k-means++ seeding beats random seeding when clusters are unbalanced ---
    k_u = 10
    sizes = [450] + [50] * 9
    u_labels = np.repeat(np.arange(k_u), sizes)
    u_centres = rng.uniform(-6.0, 6.0, size=(k_u, 2))
    u_centres[0] = 0.0
    Xu = u_centres[u_labels] + rng.normal(0.0, 0.45, size=(900, 2))
    plus, rand = [], []
    for seed in range(8):
        plus.append(KMeans("kmeans_plusplus", n_clusters=k_u, n_init=1,
                           random_state=seed).fit(Xu).inertia_)
        rand.append(KMeans("lloyd", n_clusters=k_u, init="random", n_init=1,
                           random_state=seed).fit(Xu).inertia_)
    assert np.mean(plus) < np.mean(rand), (np.mean(plus), np.mean(rand))
    print(f"kmeans: kmeans++ mean inertia {np.mean(plus):.0f} "
          f"vs random {np.mean(rand):.0f} (max {max(plus):.0f} vs {max(rand):.0f})")

    # --- Elkan: identical partition, fewer distance computations ---
    ll = KMeans("lloyd", n_clusters=3, n_init=1, init="kmeans++",
                random_state=7).fit(X)
    el = KMeans("elkan", n_clusters=3, n_init=1, init="kmeans++",
                random_state=7).fit(X)
    assert np.array_equal(ll.labels_, el.labels_), "Elkan must be exact"
    assert el.n_evals_ < ll.n_evals_, (el.n_evals_, ll.n_evals_)
    print(f"kmeans: elkan evals {el.n_evals_} vs lloyd {ll.n_evals_} "
          f"(same labels, inertia {el.inertia_:.2f})")

    # --- mini-batch: close to full-batch inertia with fewer distance evals ---
    big = centers_true[rng.integers(0, 3, size=3000)]
    big = big + rng.normal(0.0, 0.55, size=(3000, 2))
    full = KMeans("lloyd", n_clusters=3, n_init=1, init="kmeans++",
                  random_state=2).fit(big)
    mb = KMeans("mini_batch", n_clusters=3, n_init=3, max_iter=60,
                batch_size=64, init="kmeans++", random_state=2).fit(big)
    gap = (mb.inertia_ - full.inertia_) / full.inertia_
    assert gap < 0.06, gap
    assert mb.n_rows_ < full.n_rows_, (mb.n_rows_, full.n_rows_)
    print(f"kmeans: mini_batch inertia {mb.inertia_:.2f} (+{gap:.1%}) on 3000 rows, "
          f"row visits {mb.n_rows_} vs full {full.n_rows_}")

    # --- bisecting: a hierarchy of 2-means splits ---
    bis = KMeans("bisecting", n_clusters=3, n_init=1, random_state=3).fit(X)
    acc_b = _match_accuracy(bis.labels_, truth)
    assert acc_b > 0.9, acc_b
    assert bis.inertia_ < km.inertia_ * 1.6, (bis.inertia_, km.inertia_)
    print(f"kmeans: bisecting acc {acc_b:.3f}, inertia {bis.inertia_:.2f}")

    # --- spherical: directions matter, magnitudes do not ---
    group = rng.integers(0, 3, size=300)
    angles = group * (2.0 * np.pi / 3) + rng.normal(0.0, 0.25, size=300)
    radii = np.exp(rng.uniform(-1.6, 1.6, size=300))   # 0.2 .. 5.0
    Xs = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    sph = KMeans("spherical", n_clusters=3, n_init=5, random_state=4).fit(Xs)
    euc = KMeans("lloyd", n_clusters=3, n_init=5, random_state=4).fit(Xs)
    acc_s = _match_accuracy(sph.labels_, group)
    acc_e = _match_accuracy(euc.labels_, group)
    assert acc_s > 0.9 and acc_s > acc_e, (acc_s, acc_e)
    print(f"kmeans: spherical acc {acc_s:.3f} vs euclidean {acc_e:.3f} "
          f"when radius varies 25x")

    # --- empty clusters: force one, then check the repair ---
    fixed = KMeans("empty_cluster_fix", n_clusters=4, n_init=1,
                   random_state=5).fit(X)
    all_in_one = np.zeros(n, dtype=int)          # pretend cluster 0 took everything
    seeded = fixed._means(X, all_in_one, np.tile(X.mean(axis=0), (4, 1)))
    uniq = np.unique(np.round(seeded, 12), axis=0)
    assert uniq.shape[0] == 4, uniq
    assert np.isfinite(fixed.inertia_) and fixed.inertia_ > 0.0
    predict = fixed.predict(X)
    assert np.array_equal(predict, fixed.labels_)
    print(f"kmeans: empty_cluster_fix re-seeded {uniq.shape[0]} empty clusters "
          f"at distinct points, inertia {fixed.inertia_:.2f}")

    print("kmeans demo passed")


if __name__ == "__main__":
    main()
