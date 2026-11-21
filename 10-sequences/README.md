# 10 · Sequences

> Order matters: Markov structure, alignments, recurrences and state-space models.

Everything that reads left to right lives here: the classical probabilistic machinery (Markov chains, HMMs, CRFs, CTC), the distance notion for time series (DTW) and the neural sequence blocks (RNN, TCN, attention, SSMs).

<!-- catalog:begin -->
**12 atoms · 78 variants · 7 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Markov chains | [`markov_chain.py`](markov_chain.py) | 1 | 5 | Transition estimation, stationary distributions and mixing. |
| ✓ Hidden Markov models | [`hmm.py`](hmm.py) | 1 | 7 | Forward-backward, Viterbi and Baum-Welch in one file. |
| ✓ Dynamic time warping | [`dtw.py`](dtw.py) | 1 | 7 | Aligning sequences that move at different speeds. |
| ✓ Recurrent networks | [`rnn.py`](rnn.py) | 1 | 7 | Vanilla recurrence, backprop through time and clipping. |
| ✓ LSTM & GRU | [`lstm_gru.py`](lstm_gru.py) | 1 | 6 | Gated memory cells and why they fix vanishing gradients. |
| ✓ Attention | [`attention.py`](attention.py) | 1 | 10 | The one operation that replaced recurrence. |
| ✓ Conditional random fields | [`crf.py`](crf.py) | 2 | 6 | Linear-chain CRFs with forward-backward inference and training. |
| · Connectionist temporal classification | [`ctc.py`](ctc.py) | 2 | 4 | Marginalising over alignments for unsegmented labels. |
| · Temporal convnets | [`tcn.py`](tcn.py) | 2 | 4 | Dilated causal convolutions and receptive-field arithmetic. |
| · Beam search | [`beam_search.py`](beam_search.py) | 2 | 7 | Beam, greedy, top-k and nucleus decoding. |
| · N-gram language models | [`ngram_lm.py`](ngram_lm.py) | 2 | 8 | Counting, smoothing and the curse of dimensionality. |
| · State-space models | [`ssm.py`](ssm.py) | 3 | 7 | Linear recurrences that parallelise like convolutions. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Markov chains** ([`markov_chain.py`](markov_chain.py) · [markov_chain.md](markov_chain.md)) — `first_order` · `higher_order` · `absorbing` · `stationary` · `mcmc_chain`
- **Hidden Markov models** ([`hmm.py`](hmm.py) · [hmm.md](hmm.md)) — `discrete` · `gaussian` · `gmm_hmm` · `forward_backward` · `viterbi` · `baum_welch` · `scaling`
- **Dynamic time warping** ([`dtw.py`](dtw.py) · [dtw.md](dtw.md)) — `standard` · `sakoe_chiba` · `itakura` · `subsequence` · `soft_dtw` · `dtw_barycenter` · `fast_dtw`
- **Recurrent networks** ([`rnn.py`](rnn.py) · [rnn.md](rnn.md)) — `vanilla` · `elman` · `jordan` · `bptt` · `truncated_bptt` · `layer_norm_rnn` · `bidirectional`
- **LSTM & GRU** ([`lstm_gru.py`](lstm_gru.py) · [lstm_gru.md](lstm_gru.md)) — `lstm` · `gru` · `peephole` · `lstm_projection` · `cifg` · `bidirectional`
- **Attention** ([`attention.py`](attention.py) · [attention.md](attention.md)) — `dot_product` · `scaled` · `multi_head` · `mqa` · `gqa` · `sliding_window` · `linear` · `sparse` · `flash_tiled` · `cross_attention`
- **Conditional random fields** ([`crf.py`](crf.py) · [crf.md](crf.md)) — `linear_chain` · `forward_backward` · `viterbi` · `crf_sgd` · `crf_lbfgs` · `higher_order`
- **Connectionist temporal classification** ([`ctc.py`](ctc.py) · [ctc.md](ctc.md)) — `ctc_loss` · `greedy_decode` · `beam_decode` · `prefix_beam`
- **Temporal convnets** ([`tcn.py`](tcn.py) · [tcn.md](tcn.md)) — `dilated` · `causal` · `residual_tcn` · `noncausal`
- **Beam search** ([`beam_search.py`](beam_search.py) · [beam_search.md](beam_search.md)) — `greedy` · `beam` · `top_k` · `top_p` · `temperature` · `repetition_penalty` · `contrastive`
- **N-gram language models** ([`ngram_lm.py`](ngram_lm.py) · [ngram_lm.md](ngram_lm.md)) — `unigram` · `bigram` · `trigram` · `laplace` · `add_k` · `kneser_ney` · `stupid_backoff` · `interpolated`
- **State-space models** ([`ssm.py`](ssm.py) · [ssm.md](ssm.md)) — `s4` · `s4d` · `hippo` · `mamba_selective` · `s5` · `diagonal_ssm` · `parallel_scan`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
