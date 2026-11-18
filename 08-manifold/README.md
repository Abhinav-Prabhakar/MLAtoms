# 08 · Manifolds & embeddings

> Trust the manifold: nonlinear embeddings for visualisation, structure and features.

Real data lies near a low-dimensional surface. These methods recover it — for plots (t-SNE, UMAP), for distances (Isomap, diffusion maps) or for features (autoencoders, random projections).

<!-- catalog:begin -->
**12 atoms · 47 variants · 8 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ t-SNE | [`tsne.py`](tsne.py) | 1 | 5 | Student-t affinities, perplexity and early exaggeration. |
| ✓ UMAP | [`umap.py`](umap.py) | 1 | 5 | Fuzzy simplicial sets, negative sampling and cross-entropy. |
| ✓ Isomap | [`isomap.py`](isomap.py) | 1 | 4 | Geodesic distances from a neighbour graph, then classical MDS. |
| ✓ Random projection | [`random_projection.py`](random_projection.py) | 1 | 4 | Johnson-Lindenstrauss embeddings that fit nothing. |
| ✓ PaCMAP | [`pacmap.py`](pacmap.py) | 2 | 4 | Three-term objective balancing local, mid-range and global structure. |
| ✓ Locally linear embedding | [`lle.py`](lle.py) | 2 | 4 | Local linear reconstruction weights and their eigenvectors. |
| ✓ Laplacian eigenmaps | [`laplacian_eigenmaps.py`](laplacian_eigenmaps.py) | 2 | 3 | Graph Laplacian spectrum as a smooth embedding. |
| ✓ Classical MDS | [`mds.py`](mds.py) | 2 | 5 | Distance-preserving embeddings by stress minimisation. |
| · Autoencoder embedding | [`autoencoder_dr.py`](autoencoder_dr.py) | 2 | 4 | Bottleneck autoencoders as learned nonlinear projections. |
| · TriMap | [`trimap.py`](trimap.py) | 3 | 3 | Triplet constraints that preserve global geometry. |
| · Diffusion maps | [`diffusion_maps.py`](diffusion_maps.py) | 3 | 3 | Markov-chain diffusion distances and spectral embeddings. |
| · PHATE | [`phate.py`](phate.py) | 3 | 3 | Potential distances and information-geometric visualisation. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **t-SNE** ([`tsne.py`](tsne.py) · [tsne.md](tsne.md)) — `exact` · `barnes_hut` · `fft` · `multiscale` · `parametric`
- **UMAP** ([`umap.py`](umap.py) · [umap.md](umap.md)) — `umap` · `supervised` · `parametric` · `densmap` · `fuzzy_simplicial`
- **Isomap** ([`isomap.py`](isomap.py) · [isomap.md](isomap.md)) — `isomap` · `landmark` · `conformal` · `sparse`
- **Random projection** ([`random_projection.py`](random_projection.py) · [random_projection.md](random_projection.md)) — `gaussian` · `sparse` · `achlioptas` · `very_sparse`
- **PaCMAP** ([`pacmap.py`](pacmap.py) · [pacmap.md](pacmap.md)) — `pacmap` · `localmap` · `supervised` · `knn_graph`
- **Locally linear embedding** ([`lle.py`](lle.py) · [lle.md](lle.md)) — `standard` · `modified` · `hessian` · `supervised`
- **Laplacian eigenmaps** ([`laplacian_eigenmaps.py`](laplacian_eigenmaps.py) · [laplacian_eigenmaps.md](laplacian_eigenmaps.md)) — `standard` · `normalized` · `diffusion_link`
- **Classical MDS** ([`mds.py`](mds.py) · [mds.md](mds.md)) — `classical` · `metric` · `nonmetric` · `weighted` · `smacof`
- **Autoencoder embedding** ([`autoencoder_dr.py`](autoencoder_dr.py) · [autoencoder_dr.md](autoencoder_dr.md)) — `ae` · `denoising_ae` · `vae_embed` · `contractive`
- **TriMap** ([`trimap.py`](trimap.py) · [trimap.md](trimap.md)) — `trimap` · `supervised` · `knn`
- **Diffusion maps** ([`diffusion_maps.py`](diffusion_maps.py) · [diffusion_maps.md](diffusion_maps.md)) — `diffusion_map` · `diffusion_potential` · `multiscale`
- **PHATE** ([`phate.py`](phate.py) · [phate.md](phate.md)) — `phate` · `potential` · `kernel_phate`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
