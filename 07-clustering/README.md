# 07 · Clustering

> Grouping without labels: centroid, density, graph, hierarchy and mixtures.

Clustering has no single objective, which is why there are a dozen algorithms and no winner. Each file here states the objective it optimises so you can pick the one that matches your notion of a cluster.

<!-- catalog:begin -->
**16 atoms · 76 variants · 6 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ k-means | [`kmeans.py`](kmeans.py) | 1 | 7 | Lloyd's algorithm and the details that decide whether it works. |
| ✓ Gaussian mixture | [`gmm.py`](gmm.py) | 1 | 7 | Soft assignments with a likelihood you can compare across k. |
| ✓ DBSCAN | [`dbscan.py`](dbscan.py) | 1 | 4 | Density reachability, border points and the eps/minPts trade-off. |
| ✓ HDBSCAN | [`hdbscan.py`](hdbscan.py) | 1 | 5 | Mutual reachability, the condensed tree and cluster stability. |
| ✓ Spectral clustering | [`spectral_clustering.py`](spectral_clustering.py) | 1 | 5 | Clusters as cuts in a similarity graph. |
| ✓ Hierarchical clustering | [`agglomerative.py`](agglomerative.py) | 1 | 8 | Linkage criteria, dendrograms and cutting strategies. |
| · k-medoids | [`kmedoids.py`](kmedoids.py) | 2 | 5 | Medoids instead of means, so any distance works. |
| · OPTICS | [`optics.py`](optics.py) | 2 | 3 | Reachability plots for clusters of varying density. |
| · Mean shift | [`mean_shift.py`](mean_shift.py) | 2 | 4 | Mode seeking with a bandwidth-defined kernel. |
| · Fuzzy c-means | [`fuzzy_cmeans.py`](fuzzy_cmeans.py) | 2 | 4 | Soft memberships and the fuzzifier. |
| · Cluster validity | [`cluster_validity.py`](cluster_validity.py) | 2 | 6 | Choosing k without lying to yourself. |
| · BIRCH | [`birch.py`](birch.py) | 3 | 3 | Clustering-features tree for incremental, one-pass clustering. |
| · Affinity propagation | [`affinity_propagation.py`](affinity_propagation.py) | 3 | 3 | Exemplar selection by message passing. |
| · Density peaks | [`density_peaks.py`](density_peaks.py) | 3 | 4 | Centres are points with high density and far from denser points. |
| · Self-organising maps | [`som.py`](som.py) | 3 | 5 | Topology-preserving quantisation and neighbourhood updates. |
| · Subspace clustering | [`subspace_clustering.py`](subspace_clustering.py) | 3 | 3 | Clusters that only exist in some of the dimensions. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **k-means** ([`kmeans.py`](kmeans.py) · [kmeans.md](kmeans.md)) — `lloyd` · `kmeans_plusplus` · `mini_batch` · `elkan` · `bisecting` · `spherical` · `empty_cluster_fix`
- **Gaussian mixture** ([`gmm.py`](gmm.py) · [gmm.md](gmm.md)) — `full` · `diag` · `spherical` · `tied` · `variational` · `bic_selection` · `map_mixture`
- **DBSCAN** ([`dbscan.py`](dbscan.py) · [dbscan.md](dbscan.md)) — `standard` · `with_index` · `weighted` · `fuzzy_dbscan`
- **HDBSCAN** ([`hdbscan.py`](hdbscan.py) · [hdbscan.md](hdbscan.md)) — `hdbscan` · `hdbscan_star` · `soft_clustering` · `approximate` · `stability`
- **Spectral clustering** ([`spectral_clustering.py`](spectral_clustering.py) · [spectral_clustering.md](spectral_clustering.md)) — `normalized` · `unnormalized` · `ng_jordan_weiss` · `self_tuning` · `multiway`
- **Hierarchical clustering** ([`agglomerative.py`](agglomerative.py) · [agglomerative.md](agglomerative.md)) — `single` · `complete` · `average` · `ward` · `centroid` · `median` · `connectivity_constrained` · `cut_height`
- **k-medoids** ([`kmedoids.py`](kmedoids.py) · [kmedoids.md](kmedoids.md)) — `pam` · `voronoi_iteration` · `clara` · `clarans` · `alternating`
- **OPTICS** ([`optics.py`](optics.py) · [optics.md](optics.md)) — `optics` · `optics_xi` · `reachability`
- **Mean shift** ([`mean_shift.py`](mean_shift.py) · [mean_shift.md](mean_shift.md)) — `flat` · `gaussian` · `adaptive_bandwidth` · `merge`
- **Fuzzy c-means** ([`fuzzy_cmeans.py`](fuzzy_cmeans.py) · [fuzzy_cmeans.md](fuzzy_cmeans.md)) — `fcm` · `possibilistic` · `gustafson_kessel` · `noise_clustering`
- **Cluster validity** ([`cluster_validity.py`](cluster_validity.py) · [cluster_validity.md](cluster_validity.md)) — `silhouette` · `davies_bouldin` · `calinski_harabasz` · `gap` · `dunn` · `rand_index`
- **BIRCH** ([`birch.py`](birch.py) · [birch.md](birch.md)) — `birch` · `cf_tree` · `phase1_phase2`
- **Affinity propagation** ([`affinity_propagation.py`](affinity_propagation.py) · [affinity_propagation.md](affinity_propagation.md)) — `ap` · `damped` · `sparse_ap`
- **Density peaks** ([`density_peaks.py`](density_peaks.py) · [density_peaks.md](density_peaks.md)) — `dpc` · `adaptive` · `fuzzy` · `density_cut`
- **Self-organising maps** ([`som.py`](som.py) · [som.md](som.md)) — `kohonen` · `batch_som` · `growing_som` · `gsom` · `u_matrix`
- **Subspace clustering** ([`subspace_clustering.py`](subspace_clustering.py) · [subspace_clustering.md](subspace_clustering.md)) — `clique` · `proclus` · `orclus`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
