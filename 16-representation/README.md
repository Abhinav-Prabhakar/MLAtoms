# 16 · Representation

> Turning raw tokens into vectors: counting, predicting and contrasting.

Before a model can learn, its input must be a vector — and how you build that vector often matters more than the model. Sparse counting, distributional embeddings, tokenizers, contrastive objectives and ANN indexes all live here.

<!-- catalog:begin -->
**12 atoms · 60 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · TF-IDF & BM25 | [`tfidf.py`](tfidf.py) | 1 | 7 | Term weighting and the ranking function that still powers search. |
| · word2vec | [`word2vec.py`](word2vec.py) | 1 | 6 | Skip-gram/CBOW with negative sampling and subsampling. |
| · Byte-pair encoding | [`bpe.py`](bpe.py) | 1 | 6 | How text becomes tokens, from merges to unigram models. |
| · Contrastive learning | [`contrastive.py`](contrastive.py) | 1 | 8 | InfoNCE, SimCLR, MoCo and BYOL objectives. |
| · GloVe | [`glove.py`](glove.py) | 2 | 3 | Global co-occurrence factorisation with weighted least squares. |
| · fastText | [`fasttext.py`](fasttext.py) | 2 | 3 | Subword n-grams for morphology and out-of-vocabulary words. |
| · Masked modelling | [`masked_modeling.py`](masked_modeling.py) | 2 | 5 | BERT-style MLM and MAE-style reconstruction. |
| · Sentence embeddings | [`sentence_embeddings.py`](sentence_embeddings.py) | 2 | 6 | Pooling strategies, SBERT training and whitening. |
| · MinHash & LSH | [`minhash.py`](minhash.py) | 3 | 4 | Jaccard estimation and banded locality-sensitive hashing. |
| · SimHash | [`simhash.py`](simhash.py) | 3 | 3 | Random-hyperplane hashing and Hamming retrieval. |
| · Product quantisation | [`product_quantization.py`](product_quantization.py) | 3 | 5 | Codebooks and asymmetric distance computation for ANN search. |
| · Contrastive predictive coding | [`cpc.py`](cpc.py) | 3 | 4 | Predict future latents; maximise a mutual-information bound. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **TF-IDF & BM25** ([`tfidf.py`](tfidf.py) · [tfidf.md](tfidf.md)) — `tf` · `idf` · `tfidf` · `sublinear_tf` · `bm25` · `bm25_plus` · `ltc`
- **word2vec** ([`word2vec.py`](word2vec.py) · [word2vec.md](word2vec.md)) — `skipgram` · `cbow` · `negative_sampling` · `hierarchical_softmax` · `subsampling` · `sg_negative`
- **Byte-pair encoding** ([`bpe.py`](bpe.py) · [bpe.md](bpe.md)) — `bpe` · `wordpiece` · `unigram` · `sentencepiece` · `byte_level` · `merges`
- **Contrastive learning** ([`contrastive.py`](contrastive.py) · [contrastive.md](contrastive.md)) — `infonce` · `nt_xent` · `simclr` · `moco` · `byol` · `simsiam` · `triplet` · `supcon`
- **GloVe** ([`glove.py`](glove.py) · [glove.md](glove.md)) — `glove` · `glove_tfidf` · `cooccurrence`
- **fastText** ([`fasttext.py`](fasttext.py) · [fasttext.md](fasttext.md)) — `skipgram_subword` · `cbow_subword` · `supervised_fasttext`
- **Masked modelling** ([`masked_modeling.py`](masked_modeling.py) · [masked_modeling.md](masked_modeling.md)) — `bert_mlm` · `roberta` · `electra` · `mae` · `denoising_ae`
- **Sentence embeddings** ([`sentence_embeddings.py`](sentence_embeddings.py) · [sentence_embeddings.md](sentence_embeddings.md)) — `mean_pool` · `cls_pool` · `max_pool` · `sbert` · `whitening` · `weighted_mean`
- **MinHash & LSH** ([`minhash.py`](minhash.py) · [minhash.md](minhash.md)) — `minhash` · `lsh_banding` · `one_permutation` · `b_bit`
- **SimHash** ([`simhash.py`](simhash.py) · [simhash.md](simhash.md)) — `simhash` · `random_hyperplane` · `hamming`
- **Product quantisation** ([`product_quantization.py`](product_quantization.py) · [product_quantization.md](product_quantization.md)) — `pq` · `opq` · `ivf_pq` · `adc` · `symmetric`
- **Contrastive predictive coding** ([`cpc.py`](cpc.py) · [cpc.md](cpc.md)) — `cpc` · `vqvae_cpc` · `infonce_cpc` · `autoregressive_cpc`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
