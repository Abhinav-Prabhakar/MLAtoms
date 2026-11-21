# 12 · Neural blocks

> Architectural atoms: the layers modern networks are assembled from.

A transformer is six of these files stacked. Each block is written forward and backward, so you can see exactly where the parameters, the FLOPs and the inductive bias sit.

<!-- catalog:begin -->
**14 atoms · 95 variants · 11 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ MLP | [`mlp.py`](mlp.py) | 1 | 6 | Dense layers, activations and the universal approximation argument. |
| ✓ Convolution | [`conv.py`](conv.py) | 1 | 10 | Sliding dot products in every shape they come in. |
| ✓ Pooling | [`pooling.py`](pooling.py) | 1 | 9 | Max, average, global, adaptive, attention and GeM pooling. |
| ✓ Residual blocks | [`residual.py`](residual.py) | 1 | 7 | Why depth works: identity paths and their descendants. |
| ✓ Transformer block | [`transformer.py`](transformer.py) | 1 | 8 | Encoder/decoder blocks with pre-LN and gated feed-forward. |
| ✓ Positional encoding | [`positional.py`](positional.py) | 2 | 8 | How a set of tokens learns about order. |
| ✓ Mixture of experts | [`moe.py`](moe.py) | 2 | 8 | Sparse capacity without sparse gradients. |
| ✓ Dropout family | [`dropout.py`](dropout.py) | 2 | 8 | Noise as regularisation, and the variants that fixed its bugs. |
| ✓ Embedding layers | [`embedding_layers.py`](embedding_layers.py) | 2 | 7 | Token, position and type tables, plus weight tying. |
| ✓ Graph neural blocks | [`gnn_blocks.py`](gnn_blocks.py) | 2 | 8 | The four GNN layers worth memorising. |
| ✓ MLP-Mixer | [`mixer.py`](mixer.py) | 3 | 4 | Token-mixing and channel-mixing without attention or convolution. |
| · Perceiver | [`perceiver.py`](perceiver.py) | 3 | 4 | Cross-attention bottlenecks and set transformers. |
| · ConvNeXt | [`convnext.py`](convnext.py) | 3 | 4 | Modern pure-conv block: large depthwise kernels, LN, GELU. |
| · Hypernetworks | [`hypernetwork.py`](hypernetwork.py) | 3 | 4 | Generating weights, including LoRA as a hypernetwork. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **MLP** ([`mlp.py`](mlp.py) · [mlp.md](mlp.md)) — `linear` · `mlp` · `residual_mlp` · `gated_mlp` · `swiglu_mlp` · `deep_mlp`
- **Convolution** ([`conv.py`](conv.py) · [conv.md](conv.md)) — `conv1d` · `conv2d` · `conv3d` · `dilated` · `transposed` · `depthwise` · `separable` · `group` · `causal` · `circular_pad`
- **Pooling** ([`pooling.py`](pooling.py) · [pooling.md](pooling.md)) — `max` · `avg` · `global_max` · `global_avg` · `adaptive` · `attention_pool` · `gem` · `lp_pool` · `stochastic`
- **Residual blocks** ([`residual.py`](residual.py) · [residual.md](residual.md)) — `basic` · `bottleneck` · `preactivation` · `densenet` · `highway` · `scaled_residual` · `stochastic_depth`
- **Transformer block** ([`transformer.py`](transformer.py) · [transformer.md](transformer.md)) — `post_ln` · `pre_ln` · `encoder` · `decoder` · `cross_attention` · `swiglu_ffn` · `rmsnorm_block` · `parallel_attention`
- **Positional encoding** ([`positional.py`](positional.py) · [positional.md](positional.md)) — `sinusoidal` · `learned` · `rope` · `rope_ntk` · `alibi` · `relative_bias` · `t5_bias` · `no_position`
- **Mixture of experts** ([`moe.py`](moe.py) · [moe.md](moe.md)) — `top1` · `topk` · `expert_choice` · `switch` · `soft_moe` · `capacity_factor` · `load_balance_loss` · `shared_expert`
- **Dropout family** ([`dropout.py`](dropout.py) · [dropout.md](dropout.md)) — `standard` · `spatial` · `gaussian` · `variational` · `dropout1d` · `droppath` · `stochastic_depth` · `alpha_dropout`
- **Embedding layers** ([`embedding_layers.py`](embedding_layers.py) · [embedding_layers.md](embedding_layers.md)) — `token` · `position` · `type` · `factorized` · `weight_tying` · `tied_embeddings` · `adapter_embedding`
- **Graph neural blocks** ([`gnn_blocks.py`](gnn_blocks.py) · [gnn_blocks.md](gnn_blocks.md)) — `gcn` · `gat` · `gin` · `sage` · `mpnn` · `edge_conv` · `transformer_conv` · `pna`
- **MLP-Mixer** ([`mixer.py`](mixer.py) · [mixer.md](mixer.md)) — `mlp_mixer` · `gmlp` · `convmixer` · `resmlp`
- **Perceiver** ([`perceiver.py`](perceiver.py) · [perceiver.md](perceiver.md)) — `perceiver` · `set_transformer` · `latent_array` · `cross_attention_bottleneck`
- **ConvNeXt** ([`convnext.py`](convnext.py) · [convnext.md](convnext.md)) — `convnext` · `convnext_v2` · `grn` · `patchify_stem`
- **Hypernetworks** ([`hypernetwork.py`](hypernetwork.py) · [hypernetwork.md](hypernetwork.md)) — `hypernet` · `lora_hypernet` · `conditioned_weights` · `weight_generation`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
