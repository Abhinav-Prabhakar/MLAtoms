# 19 · Tabular frontier

> The current state of the art on the most common data type in industry.

Tabular learning had a quiet revolution: prior-fitted networks, recursive feature machines and parameter-efficient MLP ensembles now rival tuned gradient boosting. These are the papers worth knowing from the last three years.

<!-- catalog:begin -->
**9 atoms · 36 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Feature engineering | [`feature_engineering.py`](feature_engineering.py) | 1 | 6 | Encodings and interactions that decide tabular performance. |
| · Explainable boosting machine | [`ebm.py`](ebm.py) | 2 | 4 | GAMs plus pairwise interactions fitted by round-robin boosting. |
| · FT-Transformer | [`ft_transformer.py`](ft_transformer.py) | 2 | 4 | Feature tokeniser plus transformer for tabular rows. |
| · TabPFN | [`tabpfn.py`](tabpfn.py) | 2 | 4 | Prior-fitted networks: in-context learning for small tables. |
| · NODE | [`node.py`](node.py) | 3 | 3 | Neural oblivious decision ensembles. |
| · TabM | [`tabm.py`](tabm.py) | 3 | 4 | Parameter-efficient deep ensemble in a single MLP. |
| · xRFM | [`xrfm.py`](xrfm.py) | 3 | 4 | Recursive feature machines with kernel-ridge leaves. |
| · TabNet | [`tabnet.py`](tabnet.py) | 3 | 4 | Sequential attention with sparse feature masks. |
| · SAINT | [`saint.py`](saint.py) | 3 | 3 | Self-attention plus intersample attention for tabular data. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Feature engineering** ([`feature_engineering.py`](feature_engineering.py) · [feature_engineering.md](feature_engineering.md)) — `target_encoding` · `count_encoding` · `hashing` · `interactions` · `missing_indicator` · `binning`
- **Explainable boosting machine** ([`ebm.py`](ebm.py) · [ebm.md](ebm.md)) — `ebm` · `pairwise_interactions` · `round_robin` · `monotone_constraints`
- **FT-Transformer** ([`ft_transformer.py`](ft_transformer.py) · [ft_transformer.md](ft_transformer.md)) — `ft_transformer` · `feature_tokenizer` · `numerical_embedding` · `cls_token`
- **TabPFN** ([`tabpfn.py`](tabpfn.py) · [tabpfn.md](tabpfn.md)) — `tabpfn` · `tabpfn_v2` · `in_context` · `calibration`
- **NODE** ([`node.py`](node.py) · [node.md](node.md)) — `node` · `oblivious_decision` · `ensemble`
- **TabM** ([`tabm.py`](tabm.py) · [tabm.md](tabm.md)) — `tabm` · `tabm_mini` · `tabm_packed` · `parameter_efficient`
- **xRFM** ([`xrfm.py`](xrfm.py) · [xrfm.md](xrfm.md)) — `xrfm` · `agop` · `feature_learning` · `kernel_leaves`
- **TabNet** ([`tabnet.py`](tabnet.py) · [tabnet.md](tabnet.md)) — `tabnet` · `attentive_transformer` · `sparse_mask` · `pretraining`
- **SAINT** ([`saint.py`](saint.py) · [saint.md](saint.md)) — `saint` · `intersample_attention` · `contrastive_pretrain`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
