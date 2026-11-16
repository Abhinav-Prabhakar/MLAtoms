# 15 · Graphs

> Relational structure: centrality, communities and message passing.

When the unit of data is a relationship, the model has to respect it. This folder covers the classical graph algorithms that still run at scale and the GNN layers that generalise them.

<!-- catalog:begin -->
**12 atoms · 56 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Graph basics | [`graph_basics.py`](graph_basics.py) | 1 | 6 | Adjacency, Laplacian, degrees, components and spectral gap. |
| · PageRank | [`pagerank.py`](pagerank.py) | 1 | 5 | Power iteration, damping and personalised PageRank. |
| · Graph convolutional network | [`gcn.py`](gcn.py) | 1 | 5 | Spectral motivation and normalised-adjacency propagation. |
| · Label propagation | [`label_propagation.py`](label_propagation.py) | 2 | 4 | Harmonic functions and clamping for semi-supervised labels. |
| · Community detection | [`community.py`](community.py) | 2 | 6 | Partitions that maximise internal density. |
| · node2vec | [`node2vec.py`](node2vec.py) | 2 | 4 | Biased random walks plus skip-gram embeddings. |
| · GraphSAGE | [`graphsage.py`](graphsage.py) | 2 | 6 | Neighbour sampling with inductive aggregation. |
| · Graph attention network | [`gat.py`](gat.py) | 2 | 5 | Attention over neighbours with multi-head aggregation. |
| · Message passing (MPNN) | [`mpnn.py`](mpnn.py) | 2 | 4 | The general message-passing framework with edge features. |
| · DeepWalk | [`deepwalk.py`](deepwalk.py) | 3 | 3 | Truncated walks and the matrix-factorisation view. |
| · Graph isomorphism network | [`gin.py`](gin.py) | 3 | 4 | Sum aggregation and Weisfeiler-Lehman expressivity. |
| · Graph kernels | [`graph_kernels.py`](graph_kernels.py) | 3 | 4 | Kernels for whole graphs, without a GNN. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Graph basics** ([`graph_basics.py`](graph_basics.py) · [graph_basics.md](graph_basics.md)) — `adjacency` · `laplacian` · `normalized_laplacian` · `components` · `spectral_gap` · `degrees`
- **PageRank** ([`pagerank.py`](pagerank.py) · [pagerank.md](pagerank.md)) — `power_iteration` · `damping` · `personalized` · `weighted` · `topic_sensitive`
- **Graph convolutional network** ([`gcn.py`](gcn.py) · [gcn.md](gcn.md)) — `gcn` · `spectral_gcn` · `normalized_adjacency` · `self_loop` · `node_classification`
- **Label propagation** ([`label_propagation.py`](label_propagation.py) · [label_propagation.md](label_propagation.md)) — `harmonic` · `clamping` · `label_spreading` · `iterative`
- **Community detection** ([`community.py`](community.py) · [community.md](community.md)) — `louvain` · `leiden` · `girvan_newman` · `label_propagation` · `infomap` · `modularity`
- **node2vec** ([`node2vec.py`](node2vec.py) · [node2vec.md](node2vec.md)) — `node2vec` · `deepwalk` · `biased_walk` · `negative_sampling`
- **GraphSAGE** ([`graphsage.py`](graphsage.py) · [graphsage.md](graphsage.md)) — `mean` · `lstm` · `pool` · `gcn_agg` · `inductive` · `mini_batch`
- **Graph attention network** ([`gat.py`](gat.py) · [gat.md](gat.md)) — `gat` · `gatv2` · `multi_head` · `edge_features` · `sparse`
- **Message passing (MPNN)** ([`mpnn.py`](mpnn.py) · [mpnn.md](mpnn.md)) — `mpnn` · `edge_conv` · `interaction_network` · `message_passing`
- **DeepWalk** ([`deepwalk.py`](deepwalk.py) · [deepwalk.md](deepwalk.md)) — `deepwalk` · `truncated_walk` · `matrix_factorization`
- **Graph isomorphism network** ([`gin.py`](gin.py) · [gin.md](gin.md)) — `gin` · `sum_agg` · `mean_agg` · `wl_test`
- **Graph kernels** ([`graph_kernels.py`](graph_kernels.py) · [graph_kernels.md](graph_kernels.md)) — `wl_subtree` · `random_walk` · `shortest_path` · `graphlet`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
