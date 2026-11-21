"""Graph neural blocks — eight ways to pass messages along edges.

Chapter:   12-neural-blocks
Companion: gnn_blocks.md
Tier 2 · 8 variants: gcn, gat, gin, sage, mpnn, edge_conv, transformer_conv,
pna
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("gcn", "gat", "gin", "sage", "mpnn", "edge_conv",
            "transformer_conv", "pna")

ATTENTION = ("gat", "transformer_conv")
EDGE_VARIANTS = ("mpnn", "edge_conv", "transformer_conv")

_GELU = 0.7978845608028654
_CUBIC = 0.044715
_LEAK = 0.2


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _gelu(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    tanh = np.tanh(inner)
    return (0.5 * (1.0 + tanh)
            + 0.5 * x * (1.0 - tanh ** 2)
            * _GELU * (1.0 + 3.0 * _CUBIC * x ** 2))


def _leaky(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x, _LEAK * x)


def _leaky_grad(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0, _LEAK)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _adjacency(edges: np.ndarray, nodes: int, self_loops: bool) -> np.ndarray:
    adjacency = np.zeros((nodes, nodes))
    np.add.at(adjacency, (edges[:, 0], edges[:, 1]), 1.0)
    if self_loops:
        adjacency = adjacency + np.eye(nodes)
    return adjacency


def _normalized(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    return inverse[:, None] * adjacency * inverse[None, :]


def _row_mean(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(degree, 1e-12)


def _affine(parameters: dict[str, np.ndarray], name: str, inputs: int,
            outputs: int, rng: np.random.Generator, scale: float = 2.0):
    parameters[f"{name}.W"] = rng.normal(
        0.0, np.sqrt(scale / inputs), (inputs, outputs))
    parameters[f"{name}.b"] = np.zeros(outputs)


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #
class GNN:
    """One message-passing layer over a fixed graph.

    `forward(x, edges, edge_features)` returns the node states and a cache;
    `backward(gradient, cache)` fills parameter gradients and returns the
    gradient with respect to the input node states. Edges are directed
    `(source, target)` pairs, so a symmetric graph needs both directions.
    """

    def __init__(self, variant: str = "gcn", *, dim: int = 4,
                 hidden: int | None = None, heads: int = 2, edge_dim: int = 3,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if dim < 1 or edge_dim < 1:
            raise ValueError("dim and edge_dim must be positive")
        if heads < 1:
            raise ValueError("heads must be positive")
        if hidden is None:
            hidden = dim
        if hidden < 1:
            raise ValueError("hidden must be positive")
        if variant in ATTENTION and hidden % heads:
            raise ValueError("hidden must divide evenly into heads")
        self.variant = variant
        self.dim = dim
        self.hidden = hidden
        self.heads = heads
        self.edge_dim = edge_dim
        self.depth = hidden // heads if variant in ATTENTION else hidden
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.parameters: dict[str, np.ndarray] = {}
        if variant == "gcn":
            _affine(self.parameters, "gcn", dim, hidden, rng)
        elif variant == "gin":
            self.parameters["gin.eps"] = np.zeros(1)
            _affine(self.parameters, "gin.one", dim, hidden, rng)
            _affine(self.parameters, "gin.two", hidden, hidden, rng)
        elif variant == "sage":
            _affine(self.parameters, "sage.self", dim, hidden, rng)
            _affine(self.parameters, "sage.neigh", dim, hidden, rng)
            _affine(self.parameters, "sage.out", 2 * hidden, hidden, rng)
        elif variant == "mpnn":
            for tag in ("src", "dst"):
                self.parameters[f"mpnn.{tag}.W"] = rng.normal(
                    0.0, np.sqrt(2.0 / dim), (dim, hidden))
            self.parameters["mpnn.edge.W"] = rng.normal(
                0.0, np.sqrt(2.0 / edge_dim), (edge_dim, hidden))
            self.parameters["mpnn.message.b"] = np.zeros(hidden)
            _affine(self.parameters, "mpnn.self", dim, hidden, rng)
            _affine(self.parameters, "mpnn.update", hidden, hidden, rng)
        elif variant == "edge_conv":
            self.parameters["edge.W"] = rng.normal(
                0.0, 1.0 / np.sqrt(edge_dim), (edge_dim, dim))
            self.parameters["edge.b"] = np.zeros(dim)
            _affine(self.parameters, "edge.self", dim, hidden, rng)
            self.parameters["edge.out.W"] = rng.normal(
                0.0, np.sqrt(2.0 / dim), (dim, hidden))
        elif variant in ATTENTION:
            for head in range(heads):
                self.parameters[f"attn{head}.W"] = rng.normal(
                    0.0, np.sqrt(2.0 / dim), (dim, self.depth))
                self.parameters[f"attn{head}.src"] = rng.normal(
                    0.0, 0.1, self.depth)
                self.parameters[f"attn{head}.dst"] = rng.normal(
                    0.0, 0.1, self.depth)
            if variant == "transformer_conv":
                self.parameters["transformer.edge"] = rng.normal(
                    0.0, 0.1, (edge_dim, heads))
                _affine(self.parameters, "transformer.out", hidden, hidden,
                        rng)
            else:
                self.parameters["gat.b"] = np.zeros(hidden)
        else:  # pna
            _affine(self.parameters, "pna.self", dim, hidden, rng)
            _affine(self.parameters, "pna.out", 12 * dim, hidden, rng)
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    # -- properties and validation ------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    def _check(self, x: np.ndarray, edges: np.ndarray,
               edge_features: np.ndarray | None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must be (nodes, dim)")
        if x.shape[0] < 1 or not np.all(np.isfinite(x)):
            raise ValueError("x must be non-empty and finite")
        nodes = x.shape[0]
        edges = np.asarray(edges)
        if edges.dtype.kind not in "iu" or edges.ndim != 2 \
                or edges.shape[1] != 2 or edges.shape[0] < 1:
            raise ValueError("edges must be a non-empty (m, 2) integer array")
        if np.any(edges < 0) or np.any(edges >= nodes):
            raise ValueError("an edge endpoint is out of range")
        edges = edges.astype(int)
        if self.variant in EDGE_VARIANTS:
            if edge_features is None:
                raise ValueError("this variant needs edge features")
            edge_features = np.asarray(edge_features, dtype=float)
            if edge_features.shape != (edges.shape[0], self.edge_dim):
                raise ValueError("edge_features must be (m, edge_dim)")
            if not np.all(np.isfinite(edge_features)):
                raise ValueError("edge_features must be finite")
        elif edge_features is not None:
            raise ValueError("this variant has no edge features")
        return x, edges, edge_features

    # -- forward ------------------------------------------------------------- #
    def forward(self, x: np.ndarray, edges: np.ndarray,
                edge_features: np.ndarray | None = None):
        x, edges, edge_features = self._check(x, edges, edge_features)
        nodes = x.shape[0]
        p = self.parameters
        if self.variant == "gcn":
            adjacency = _normalized(_adjacency(edges, nodes, True))
            pre = adjacency @ x @ p["gcn.W"] + p["gcn.b"]
            out = _gelu(pre)
            return out, {"adjacency": adjacency, "x": x, "pre": pre}

        if self.variant == "gin":
            adjacency = _adjacency(edges, nodes, False)
            combined = (1.0 + p["gin.eps"][0]) * x + adjacency @ x
            pre = combined @ p["gin.one.W"] + p["gin.one.b"]
            hidden = _gelu(pre)
            out = hidden @ p["gin.two.W"] + p["gin.two.b"]
            return out, {"adjacency": adjacency, "x": x, "combined": combined,
                         "pre": pre, "hidden": hidden}

        if self.variant == "sage":
            adjacency = _row_mean(_adjacency(edges, nodes, False))
            own = x @ p["sage.self.W"]
            neighbour = (adjacency @ x) @ p["sage.neigh.W"]
            concat = np.concatenate([own, neighbour], axis=1)
            pre = concat @ p["sage.out.W"] + p["sage.out.b"]
            out = _gelu(pre)
            return out, {"adjacency": adjacency, "x": x, "own": own,
                         "neighbour": neighbour, "concat": concat,
                         "pre": pre}

        if self.variant == "mpnn":
            source, target = edges[:, 0], edges[:, 1]
            pre_message = (x[source] @ p["mpnn.src.W"]
                           + x[target] @ p["mpnn.dst.W"]
                           + edge_features @ p["mpnn.edge.W"]
                           + p["mpnn.message.b"])
            message = _gelu(pre_message)
            total = np.zeros((nodes, self.hidden))
            np.add.at(total, source, message)
            counts = np.bincount(source, minlength=nodes)
            mean = total / np.maximum(counts, 1)[:, None]
            pre_update = (x @ p["mpnn.self.W"] + mean @ p["mpnn.update.W"]
                          + p["mpnn.self.b"])
            out = _gelu(pre_update)
            return out, {"source": source, "target": target, "x": x,
                         "edge_features": edge_features,
                         "pre_message": pre_message, "message": message,
                         "counts": counts, "mean": mean,
                         "pre_update": pre_update}

        if self.variant == "edge_conv":
            source, target = edges[:, 0], edges[:, 1]
            raw_message = x[target] + edge_features @ p["edge.W"] + p["edge.b"]
            message = _gelu(raw_message)
            aggregate = np.zeros((nodes, self.dim))
            winner = np.full((nodes, self.dim), -1, dtype=int)
            for node in range(nodes):
                rows = np.flatnonzero(source == node)
                if rows.size == 0:
                    continue
                block = message[rows]
                choices = block.argmax(axis=0)
                aggregate[node] = block[choices, np.arange(self.dim)]
                winner[node] = rows[choices]
            pre_update = (x @ p["edge.self.W"] + aggregate @ p["edge.out.W"]
                          + p["edge.self.b"])
            out = _gelu(pre_update)
            return out, {"source": source, "target": target, "x": x,
                         "edge_features": edge_features,
                         "raw_message": raw_message, "winner": winner,
                         "aggregate": aggregate, "pre_update": pre_update}

        if self.variant == "gat":
            adjacency = _adjacency(edges, nodes, True)
            allowed = adjacency > 0.0
            parts, heads = [], []
            for head in range(self.heads):
                projected = x @ p[f"attn{head}.W"]
                scores = (projected @ p[f"attn{head}.src"])[:, None] \
                    + (projected @ p[f"attn{head}.dst"])[None, :]
                masked = np.where(allowed, _leaky(scores), -np.inf)
                attention = _softmax(masked)
                parts.append(attention @ projected)
                heads.append({"projected": projected, "scores": scores,
                              "attention": attention})
            concat = np.concatenate(parts, axis=1)
            pre = concat + p["gat.b"]
            out = _gelu(pre)
            return out, {"heads": heads, "concat": concat, "pre": pre,
                         "allowed": allowed, "x": x}

        if self.variant == "transformer_conv":
            adjacency = _adjacency(edges, nodes, True)
            allowed = adjacency > 0.0
            source, target = edges[:, 0], edges[:, 1]
            edge_score = edge_features @ p["transformer.edge"]
            edge_matrix = np.zeros((nodes, nodes, self.heads))
            np.add.at(edge_matrix, (source, target), edge_score)
            parts, heads = [], []
            for head in range(self.heads):
                projected = x @ p[f"attn{head}.W"]
                scores = ((projected @ p[f"attn{head}.src"])[:, None]
                          + (projected @ p[f"attn{head}.dst"])[None, :]
                          + edge_matrix[:, :, head] / np.sqrt(self.depth))
                masked = np.where(allowed, _leaky(scores), -np.inf)
                attention = _softmax(masked)
                parts.append(attention @ projected)
                heads.append({"projected": projected, "scores": scores,
                              "attention": attention})
            concat = np.concatenate(parts, axis=1)
            pre = concat @ p["transformer.out.W"] + p["transformer.out.b"]
            out = _gelu(pre)
            return out, {"heads": heads, "concat": concat, "pre": pre,
                         "allowed": allowed, "x": x, "source": source,
                         "target": target, "edge_score": edge_score,
                         "edge_features": edge_features}

        # pna
        adjacency = _adjacency(edges, nodes, False)
        degree = adjacency.sum(axis=1)
        neighbours = [np.flatnonzero(adjacency[node] > 0.0)
                      for node in range(nodes)]
        mean = np.zeros_like(x)
        maximum = np.zeros_like(x)
        minimum = np.zeros_like(x)
        deviation = np.zeros_like(x)
        amax = np.full((nodes, self.dim), -1, dtype=int)
        amin = np.full((nodes, self.dim), -1, dtype=int)
        for node, rows in enumerate(neighbours):
            if rows.size == 0:
                continue
            block = x[rows]
            mean[node] = block.mean(axis=0)
            maximum[node] = block.max(axis=0)
            minimum[node] = block.min(axis=0)
            deviation[node] = block.std(axis=0)
            amax[node] = rows[block.argmax(axis=0)]
            amin[node] = rows[block.argmin(axis=0)]
        logged = np.log1p(degree)
        middle = float(logged.mean()) if logged.size else 1.0
        amplify = logged / max(middle, 1e-6)
        attenuate = middle / np.maximum(logged, 1e-6)
        blocks = []
        for aggregate in (mean, maximum, minimum, deviation):
            blocks.extend([aggregate, aggregate * amplify[:, None],
                           aggregate * attenuate[:, None]])
        features = np.concatenate(blocks, axis=1)
        pre = (x @ p["pna.self.W"] + features @ p["pna.out.W"]
               + p["pna.self.b"])
        out = _gelu(pre)
        return out, {"adjacency": adjacency, "degree": degree,
                     "neighbours": neighbours, "mean": mean,
                     "maximum": maximum, "minimum": minimum,
                     "deviation": deviation, "amax": amax, "amin": amin,
                     "amplify": amplify, "attenuate": attenuate,
                     "features": features, "x": x, "pre": pre}

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != (cache["x"].shape[0], self.hidden):
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        g, p = self.gradients, self.parameters
        if self.variant == "gcn":
            d_pre = gradient * _gelu_grad(cache["pre"])
            g["gcn.W"] += cache["x"].T @ (cache["adjacency"].T @ d_pre)
            g["gcn.b"] += d_pre.sum(axis=0)
            return cache["adjacency"].T @ d_pre @ p["gcn.W"].T

        if self.variant == "gin":
            g["gin.two.W"] += cache["hidden"].T @ gradient
            g["gin.two.b"] += gradient.sum(axis=0)
            d_pre = (gradient @ p["gin.two.W"].T) * _gelu_grad(cache["pre"])
            g["gin.one.W"] += cache["combined"].T @ d_pre
            g["gin.one.b"] += d_pre.sum(axis=0)
            d_combined = d_pre @ p["gin.one.W"].T
            g["gin.eps"] += np.array([float(np.sum(d_combined * cache["x"]))])
            return ((1.0 + p["gin.eps"][0]) * d_combined
                    + cache["adjacency"].T @ d_combined)

        if self.variant == "sage":
            g["sage.out.W"] += cache["concat"].T @ (
                gradient * _gelu_grad(cache["pre"]))
            g["sage.out.b"] += (gradient * _gelu_grad(cache["pre"])).sum(axis=0)
            d_concat = (gradient * _gelu_grad(cache["pre"])) @ p["sage.out.W"].T
            d_own = d_concat[:, :self.hidden]
            d_neighbour = d_concat[:, self.hidden:]
            g["sage.self.W"] += cache["x"].T @ d_own
            g["sage.neigh.W"] += (cache["adjacency"] @ cache["x"]).T @ d_neighbour
            return (d_own @ p["sage.self.W"].T
                    + cache["adjacency"].T
                    @ (d_neighbour @ p["sage.neigh.W"].T))

        if self.variant == "mpnn":
            d_pre_update = gradient * _gelu_grad(cache["pre_update"])
            g["mpnn.self.W"] += cache["x"].T @ d_pre_update
            g["mpnn.update.W"] += cache["mean"].T @ d_pre_update
            g["mpnn.self.b"] += d_pre_update.sum(axis=0)
            d_mean = d_pre_update @ p["mpnn.update.W"].T
            d_total = d_mean / np.maximum(cache["counts"], 1)[:, None]
            d_message = d_total[cache["source"]]
            d_pre_message = d_message * _gelu_grad(cache["pre_message"])
            g["mpnn.src.W"] += cache["x"][cache["source"]].T @ d_pre_message
            g["mpnn.dst.W"] += cache["x"][cache["target"]].T @ d_pre_message
            g["mpnn.edge.W"] += cache["edge_features"].T @ d_pre_message
            g["mpnn.message.b"] += d_pre_message.sum(axis=0)
            d_x = d_pre_update @ p["mpnn.self.W"].T
            np.add.at(d_x, cache["source"],
                      d_pre_message @ p["mpnn.src.W"].T)
            np.add.at(d_x, cache["target"],
                      d_pre_message @ p["mpnn.dst.W"].T)
            return d_x

        if self.variant == "edge_conv":
            d_pre_update = gradient * _gelu_grad(cache["pre_update"])
            g["edge.self.W"] += cache["x"].T @ d_pre_update
            g["edge.out.W"] += cache["aggregate"].T @ d_pre_update
            g["edge.self.b"] += d_pre_update.sum(axis=0)
            d_aggregate = d_pre_update @ p["edge.out.W"].T
            d_raw = np.zeros_like(cache["raw_message"])
            for node in range(cache["x"].shape[0]):
                columns = np.flatnonzero(cache["winner"][node] >= 0)
                np.add.at(d_raw,
                          (cache["winner"][node, columns], columns),
                          d_aggregate[node, columns])
            d_message = d_raw * _gelu_grad(cache["raw_message"])
            g["edge.W"] += cache["edge_features"].T @ d_message
            g["edge.b"] += d_message.sum(axis=0)
            d_x = d_pre_update @ p["edge.self.W"].T
            np.add.at(d_x, cache["target"], d_message)
            return d_x

        if self.variant in ATTENTION:
            if self.variant == "gat":
                d_concat = gradient * _gelu_grad(cache["pre"])
                g["gat.b"] += d_concat.sum(axis=0)
            else:
                d_pre = gradient * _gelu_grad(cache["pre"])
                g["transformer.out.W"] += cache["concat"].T @ d_pre
                g["transformer.out.b"] += d_pre.sum(axis=0)
                d_concat = d_pre @ p["transformer.out.W"].T
            d_x = np.zeros_like(cache["x"])
            for head in range(self.heads):
                data = cache["heads"][head]
                width = self.depth
                d_head = d_concat[:, head * width:(head + 1) * width]
                attention, projected = data["attention"], data["projected"]
                d_attention = d_head @ projected.T
                d_projected = attention.T @ d_head
                d_scores = attention * (
                    d_attention - (attention * d_attention).sum(
                        axis=1, keepdims=True))
                d_scores = d_scores * _leaky_grad(data["scores"])
                d_projected += (d_scores.sum(axis=1)[:, None]
                                * p[f"attn{head}.src"][None, :])
                d_projected += (d_scores.sum(axis=0)[:, None]
                                * p[f"attn{head}.dst"][None, :])
                g[f"attn{head}.src"] += projected.T @ d_scores.sum(axis=1)
                g[f"attn{head}.dst"] += projected.T @ d_scores.sum(axis=0)
                g[f"attn{head}.W"] += cache["x"].T @ d_projected
                d_x += d_projected @ p[f"attn{head}.W"].T
            if self.variant == "transformer_conv":
                source, target = cache["source"], cache["target"]
                d_edge_scores = np.zeros((source.size, self.heads))
                for head in range(self.heads):
                    data = cache["heads"][head]
                    d_head = d_concat[:, head * self.depth:
                                      (head + 1) * self.depth]
                    d_attention = d_head @ data["projected"].T
                    d_scores = data["attention"] * (
                        d_attention - (data["attention"] * d_attention).sum(
                            axis=1, keepdims=True))
                    d_scores = d_scores * _leaky_grad(data["scores"])
                    d_edge_scores[:, head] = (
                        d_scores[source, target] / np.sqrt(self.depth))
                g["transformer.edge"] += cache["edge_features"].T @ d_edge_scores
            return d_x

        # pna
        d_pre = gradient * _gelu_grad(cache["pre"])
        g["pna.self.W"] += cache["x"].T @ d_pre
        g["pna.out.W"] += cache["features"].T @ d_pre
        g["pna.self.b"] += d_pre.sum(axis=0)
        d_features = d_pre @ p["pna.out.W"].T
        blocks = d_features.reshape(cache["x"].shape[0], 4, 3, self.dim)
        scales = np.stack([np.ones(cache["x"].shape[0]),
                           cache["amplify"], cache["attenuate"]], axis=1)
        d_aggregates = np.zeros((4, cache["x"].shape[0], self.dim))
        for kind in range(4):
            for scaler in range(3):
                d_aggregates[kind] += (blocks[:, kind, scaler, :]
                                       * scales[:, scaler, None])
        d_x = d_pre @ p["pna.self.W"].T
        for node, rows in enumerate(cache["neighbours"]):
            if rows.size == 0:
                continue
            d_x[rows] += d_aggregates[0, node] / rows.size
            columns = np.arange(self.dim)
            np.add.at(d_x, (cache["amax"][node], columns),
                      d_aggregates[1, node])
            np.add.at(d_x, (cache["amin"][node], columns),
                      d_aggregates[2, node])
            spread = cache["deviation"][node]
            if np.all(spread > 0.0):
                d_x[rows] += ((cache["x"][rows] - cache["mean"][node])
                              / (rows.size * spread)
                              * d_aggregates[3, node])
        return d_x


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _graph(nodes: int):
    pairs = [(node, (node + 1) % nodes) for node in range(nodes)]
    pairs += [(0, 2), (2, 4), (4, 0), (1, 3), (3, 5), (5, 1)]
    edges = []
    for left, right in pairs:
        edges.append((left, right))
        edges.append((right, left))
    return np.array(edges)


def demo() -> None:
    rng = np.random.default_rng(1)
    nodes = 6
    edges = _graph(nodes)
    x = rng.normal(size=(nodes, 4))
    edge_features = rng.normal(size=(edges.shape[0], 3))

    print("gnn. every variant is equivariant to a node permutation:")
    perm = np.array([3, 0, 5, 1, 4, 2])
    position = np.argsort(perm)
    for variant in VARIANTS:
        model = GNN(variant, dim=4, hidden=4, heads=2, edge_dim=3, seed=2)
        extra = edge_features if variant in EDGE_VARIANTS else None
        out, _ = model.forward(x, edges, extra)
        moved, _ = model.forward(x[perm], position[edges], extra)
        error = float(np.abs(moved - out[perm]).max())
        assert error < 1e-9
        print(f"    {variant:<15s} error {error:.1e}  "
              f"{model.n_parameters:3d} parameters")

    print("gnn. gcn normalises the adjacency with self loops:")
    model = GNN("gcn", dim=4, hidden=4, seed=3)
    out, cache = model.forward(x, edges)
    adjacency = _adjacency(edges, nodes, True)
    degree = adjacency.sum(axis=1)
    normalized = adjacency / np.sqrt(np.outer(degree, degree))
    assert np.allclose(cache["adjacency"], normalized)
    expected = _gelu(normalized @ x @ model.parameters["gcn.W"]
                     + model.parameters["gcn.b"])
    assert np.allclose(out, expected)
    eigenvalues = np.linalg.eigvalsh(normalized)
    assert eigenvalues.max() <= 1.0 + 1e-12
    assert eigenvalues.min() >= -1.0 - 1e-12
    print(f"  spectral range [{eigenvalues.min():.3f},"
          f" {eigenvalues.max():.3f}], so one layer cannot amplify")

    print("gnn. gat normalises attention over neighbours only:")
    model = GNN("gat", dim=4, hidden=4, heads=2, seed=4)
    out, cache = model.forward(x, edges)
    attention = cache["heads"][0]["attention"]
    allowed = cache["allowed"]
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert np.all(attention[~allowed] == 0.0)
    assert np.all(attention[allowed] > 0.0)
    assert np.allclose(attention, attention * allowed)
    assert out.shape == (nodes, 4)
    print(f"  {model.heads} heads, {model.depth} wide each; every row sums to")
    print(f"  one over {int(allowed.sum() / nodes)} allowed entries on average")

    print("gnn. gin sums neighbours, sage means and concatenates:")
    model = GNN("gin", dim=4, hidden=4, seed=5)
    out, cache = model.forward(x, edges)
    adjacency = _adjacency(edges, nodes, False)
    combined = (1.0 + model.parameters["gin.eps"][0]) * x + adjacency @ x
    hidden = _gelu(combined @ model.parameters["gin.one.W"]
                   + model.parameters["gin.one.b"])
    assert np.allclose(cache["combined"], combined)
    assert np.allclose(out, hidden @ model.parameters["gin.two.W"]
                       + model.parameters["gin.two.b"])
    assert np.allclose(combined - x, adjacency @ x)
    sage = GNN("sage", dim=4, hidden=4, seed=6)
    out, cache = sage.forward(x, edges)
    assert np.allclose(cache["neighbour"],
                       (_row_mean(adjacency) @ x)
                       @ sage.parameters["sage.neigh.W"])
    sums = adjacency @ x
    means = _row_mean(adjacency) @ x
    assert np.allclose(sums[0] / adjacency[0].sum(), means[0])
    print(f"  node 0 has {int(adjacency[0].sum())} neighbours; sum and mean")
    print("  differ by exactly that degree, which is the expressivity gap")

    print("gnn. mpnn computes an edge message then a node update:")
    model = GNN("mpnn", dim=4, hidden=4, edge_dim=3, seed=7)
    out, cache = model.forward(x, edges, edge_features)
    p = model.parameters
    source, target = edges[:, 0], edges[:, 1]
    message = _gelu(x[source] @ p["mpnn.src.W"] + x[target] @ p["mpnn.dst.W"]
                    + edge_features @ p["mpnn.edge.W"] + p["mpnn.message.b"])
    total = np.zeros((nodes, 4))
    np.add.at(total, source, message)
    counts = np.bincount(source, minlength=nodes)
    assert np.allclose(cache["message"], message)
    assert np.allclose(cache["mean"], total / counts[:, None])
    expected = _gelu(x @ p["mpnn.self.W"]
                     + (total / counts[:, None]) @ p["mpnn.update.W"]
                     + p["mpnn.self.b"])
    assert np.allclose(out, expected)
    print(f"  {edges.shape[0]} directed edges, mean aggregation, and a")
    print("  separate self path; the cache holds every edge message")

    print("gnn. edge conv keeps the maximum message per node:")
    model = GNN("edge_conv", dim=4, hidden=4, edge_dim=3, seed=8)
    out, cache = model.forward(x, edges, edge_features)
    message = _gelu(x[target] + edge_features @ model.parameters["edge.W"]
                    + model.parameters["edge.b"])
    rows = np.flatnonzero(source == 0)
    assert np.allclose(cache["aggregate"][0], message[rows].max(axis=0))
    assert np.allclose(_gelu(cache["raw_message"]), message)
    print(f"  node 0 takes the elementwise maximum over {rows.size} messages,")
    print("  with the winning edge recorded for the backward pass")

    print("gnn. transformer conv adds an edge score to attention:")
    model = GNN("transformer_conv", dim=4, hidden=4, heads=2, edge_dim=3,
                seed=9)
    out, cache = model.forward(x, edges, edge_features)
    flat, flat_cache = model.forward(x, edges, np.zeros_like(edge_features))
    attention = cache["heads"][0]["attention"]
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert not np.allclose(out, flat)
    assert not np.allclose(attention, flat_cache["heads"][0]["attention"])
    print("  edge features change both the scores and the output; the")
    print("  output projection mixes the concatenated heads")

    print("gnn. pna aggregates mean, max, min and standard deviation:")
    model = GNN("pna", dim=4, hidden=4, seed=10)
    out, cache = model.forward(x, edges)
    adjacency = _adjacency(edges, nodes, False)
    for node in range(nodes):
        rows = np.flatnonzero(adjacency[node] > 0)
        block = x[rows]
        assert np.allclose(cache["mean"][node], block.mean(axis=0))
        assert np.allclose(cache["maximum"][node], block.max(axis=0))
        assert np.allclose(cache["minimum"][node], block.min(axis=0))
        assert np.allclose(cache["deviation"][node], block.std(axis=0))
    logged = np.log1p(adjacency.sum(axis=1))
    middle = logged.mean()
    assert np.allclose(cache["amplify"], logged / middle)
    assert np.allclose(cache["attenuate"], middle / logged)
    print(f"  four aggregators times three degree scalers,"
          f" {cache['features'].shape[1]} features per node")

    print("gnn. finite differences agree with the analytic gradients:")
    small_edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
                            [3, 0], [0, 3]])
    small_x = rng.normal(size=(4, 3))
    small_edge = rng.normal(size=(8, 3))
    target = rng.normal(size=(4, 4))
    for variant in ("gcn", "gat", "gin", "sage", "mpnn", "edge_conv",
                    "transformer_conv", "pna"):
        model = GNN(variant, dim=3, hidden=4, heads=2, edge_dim=3, seed=11)
        extra = small_edge if variant in EDGE_VARIANTS else None
        out, cache = model.forward(small_x, small_edges, extra)
        d_x = model.backward(target, cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small_x, small_edges, extra)[0]
                value[index] = old - 1e-6
                minus = model.forward(small_x, small_edges, extra)[0]
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        for index in np.ndindex(small_x.shape):
            old = small_x[index]
            small_x[index] = old + 1e-6
            plus = model.forward(small_x, small_edges, extra)[0]
            small_x[index] = old - 1e-6
            minus = model.forward(small_x, small_edges, extra)[0]
            small_x[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_x[index]))
        assert worst < 1e-6
        print(f"  {variant:<6s} worst parameter or input error {worst:.1e}")

    print("gnn. malformed calls are refused:")
    guard = GNN("gcn", dim=4, hidden=4, seed=12)
    _, guard_cache = guard.forward(x, edges)
    bad = [
        lambda: GNN("message"),
        lambda: GNN("gcn", dim=0),
        lambda: GNN("gcn", hidden=0),
        lambda: GNN("gat", hidden=3, heads=2),
        lambda: GNN("gcn", heads=0),
        lambda: guard.forward(np.zeros((6, 3)), edges),
        lambda: guard.forward(np.zeros((6, 4)), np.array([[0, 1.0]])),
        lambda: guard.forward(np.zeros((6, 4)), np.array([[0, 7]])),
        lambda: guard.forward(np.zeros((6, 4)), np.array([[0, 1]]),
                              edge_features),
        lambda: GNN("mpnn", edge_dim=3).forward(
            x, edges, None),
        lambda: GNN("mpnn", edge_dim=3).forward(
            x, edges, np.zeros((4, 3))),
        lambda: guard.backward(np.zeros((5, 4)), guard_cache),
        lambda: guard.forward(np.full((6, 4), np.nan), edges),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, an empty width or")
    print("  head count, heads that do not divide the width, a wrong input")
    print("  width, fractional or out-of-range edges, edge features on a")
    print("  variant without them, missing or misplaced edge features, a")
    print("  wrong gradient shape, and non-finite features")
    print("gnn demo passed")


if __name__ == "__main__":
    demo()
