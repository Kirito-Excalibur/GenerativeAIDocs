# Embeddings and Vector Search

> **Summary** — An embedding maps text to a vector such that semantic similarity becomes geometric
> proximity. This page covers how embedding models are trained (contrastive learning), why cosine
> similarity thresholds are not transferable, the ANN index structures (HNSW, IVF-PQ) with their
> actual memory and recall trade-offs, and hybrid search — which beats pure dense retrieval more
> often than people expect.

**Prerequisites**: → [Math toolkit §4](../01-foundations/03-math-toolkit.md#4-high-dimensional-geometry-why-your-intuition-is-wrong) · **Next**: → [RAG](03-rag.md)

---

## 1. What an embedding is

$$f_\theta : \text{text} \to \mathbb{R}^d$$

such that semantically similar texts land close together.

```
                  ▲
                  │   • "king"
                  │       ╲
                  │        ╲ (the "royalty→commoner" direction)
                  │   • "queen"   ╲
                  │                 ▼
                  │   • "man"   • "woman"
                  │
                  │            • "banana"      ← unrelated: far away
                  └──────────────────────────►
```

> [!TIP]
> The classic `king − man + woman ≈ queen` arithmetic was a word2vec result. Modern sentence
> embeddings are less cleanly linear but far more useful, because they encode whole passages in
> context rather than context-free word identities.

**What embeddings are used for:**

| Use | How |
|---|---|
| **Semantic search / RAG** | nearest neighbours of the query | 
| Clustering | k-means over embeddings |
| Deduplication | near-duplicate = cosine above a threshold |
| Classification | logistic regression on embeddings (very strong baseline) |
| Recommendation | item and user embeddings in one space |
| Anomaly detection | far from all cluster centroids |

---

## 2. How embedding models are trained

**Contrastive learning**, the same InfoNCE objective as CLIP
(→ [Multimodal §2](../05-diffusion-and-vision/05-multimodal.md#2-clip-a-shared-embedding-space)):

$$\mathcal{L} = -\log\frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q,d^+)/\tau) + \sum_{j}\exp(\text{sim}(q, d_j^-)/\tau)}$$

Pull the query toward its positive; push it away from negatives.

```
   Before                          After training
        q                                q
      ╱ │ ╲                              │
    d⁺  d⁻ d⁻                            d⁺        d⁻   d⁻
   all equidistant                  positive pulled close,
                                    negatives pushed away
```

**The three kinds of negatives, and why it matters:**

| Type | Source | Effect |
|---|---|---|
| **In-batch** | other documents in the same batch | free, but usually too easy |
| **Random** | sampled from the corpus | easy; little learning signal |
| **Hard** | high BM25 or dense score but actually wrong | ⭐ the critical ingredient |

> [!TIP]
> **Hard negatives are what separate good embedding models from mediocre ones.** If every negative
> is obviously unrelated, the model only learns coarse topical similarity. Hard negatives — documents
> that *look* relevant but aren't — force it to learn fine distinctions. Standard practice: retrieve
> top-$k$ with a first-pass model, remove known positives, and use the rest as hard negatives.

> [!WARNING]
> **False negatives are the trap.** Mining hard negatives from an unlabelled corpus will pull in
> documents that are actually relevant but unlabelled, and training on them teaches the model
> exactly the wrong thing. Mitigations: filter with a cross-encoder, or use a margin so only
> *clearly* worse documents count as negatives.

**The modern pipeline** for a strong embedding model:
1. Pretrain a Transformer (often initialized from an LLM).
2. Weakly-supervised contrastive training on billions of naturally-paired texts (title/body,
   question/answer, citation pairs).
3. Supervised fine-tuning on labelled retrieval data with hard negatives.
4. **Instruction prefixes** — "Represent this query for retrieval:" vs "Represent this document:" —
   so one model handles asymmetric search, clustering and classification.

---

## 3. Similarity metrics

| Metric | Formula | Notes |
|---|---|---|
| **Cosine** | $\dfrac{a\cdot b}{\|a\|\|b\|}$ | ignores magnitude; the default |
| Dot product | $a\cdot b$ | magnitude matters; can encode document "importance" |
| Euclidean (L2) | $\|a-b\|_2$ | equivalent to cosine **if vectors are normalized** |

**The identity worth knowing.** For unit vectors,

$$\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a\cdot b = 2 - 2\cos\theta$$

So L2 ranking and cosine ranking are **identical** after normalization. This is why every vector
database says "normalize your vectors" — it lets metric-assuming index structures handle cosine
correctly.

> [!WARNING]
> **Cosine thresholds do not transfer between models.** From
> → [Math toolkit §4](../01-foundations/03-math-toolkit.md#4-high-dimensional-geometry-why-your-intuition-is-wrong):
> random vectors in $\mathbb{R}^d$ have cosine $\approx \pm 1/\sqrt{d}$. But trained embeddings are
> *anisotropic* — they occupy a narrow cone, so unrelated texts can score 0.7.

**Realistic values differ wildly by model:**

| Model family | Unrelated texts | Related | Near-duplicate |
|---|---|---|---|
| Model A | 0.10 | 0.45 | 0.90 |
| Model B | 0.70 | 0.82 | 0.97 |

**A threshold of 0.75 means "very similar" for model A and "probably unrelated" for model B.**
Always calibrate on your own data: embed 1000 known-unrelated pairs, look at the distribution, and
set your threshold from that — never from a number you read somewhere.

---

## 4. Approximate nearest neighbour search

Exact search over $N$ vectors costs $O(Nd)$ per query.

10 M vectors × 768 dims × 4 bytes = 30.7 GB, and one query is $7.7\times10^9$ FLOPs. At 100 QPS
that's 770 GFLOP/s just for search — feasible but wasteful. ANN trades a small amount of recall
for orders of magnitude of speed.

### HNSW: the quality default

**Hierarchical Navigable Small World**: a multi-layer graph where upper layers are sparse
"highways" and the bottom layer contains every point.

```
  Layer 2   ●───────────────●              long-range links: coarse navigation
            │               │
  Layer 1   ●───────●───────●───────●
            │       │       │       │
  Layer 0   ●─●─●─●─●─●─●─●─●─●─●─●─●      every vector; short local links

  Search: enter at the top, greedily walk toward the query,
          descend a layer, repeat. O(log N).
```

| Parameter | Meaning | Trade-off |
|---|---|---|
| `M` | links per node | higher = better recall, more memory |
| `efConstruction` | candidates during build | higher = better index, slower build |
| `efSearch` | candidates during query | **tune at query time**: higher = better recall, slower |

**Memory**: roughly $N \times (d\times4 + M\times2\times4)$ bytes. For 10 M vectors, $d=768$,
$M=16$: $10^7 \times (3072 + 128) = 32$ GB.

Typical: 95–99% recall at ~1 ms per query. **`efSearch` is the knob you actually turn in
production** — it lets you trade latency for recall without rebuilding.

### IVF-PQ: the memory default

Two ideas stacked:

**1. IVF (inverted file)**: k-means the vectors into $n_{\text{list}}$ clusters; at query time
search only the $n_{\text{probe}}$ nearest clusters.

**2. PQ (product quantization)**: split each vector into $m$ subvectors; quantize each against its
own 256-entry codebook. Store $m$ bytes instead of $4d$.

```
   768-dim float vector = 3072 bytes
        │
        ▼ split into m=96 subvectors of 8 dims each
   [8][8][8]...[8]
    │  │  │     │
    ▼  ▼  ▼     ▼   each → nearest of 256 centroids → 1 byte
   [b][b][b]...[b]  = 96 bytes

   32× compression
```

**10 M vectors**: 30.7 GB → **0.96 GB**. That's the difference between needing a machine with
64 GB of RAM and running on a laptop.

> [!WARNING]
> **PQ is lossy.** Recall drops to roughly 80–95% depending on $m$. The standard fix is
> **re-ranking**: retrieve 5–10× more candidates with PQ, then rescore the shortlist with the exact
> full-precision vectors.

### Comparison

| Index | Build | Query | Memory (10M×768) | Recall | Use when |
|---|---|---|---|---|---|
| Flat (exact) | instant | slow | 30.7 GB | 100% | < 100k vectors |
| **HNSW** | slow | **fast** | 32 GB | 95–99% | quality matters, RAM available |
| IVF-Flat | medium | medium | 30.7 GB | 90–98% | moderate scale |
| **IVF-PQ** | medium | fast | **1 GB** | 80–95% | huge scale, RAM-constrained |
| **HNSW-PQ** | slow | fast | ~2 GB | 90–97% | the usual production compromise |
| ScaNN | medium | very fast | ~2 GB | 90–97% | anisotropic quantization; strong |

> [!TIP]
> **Rule of thumb**: under 100k vectors, use brute force — it is simpler, exact, and fast enough.
> ANN complexity is only justified above ~1 M.

---

## 5. Matryoshka embeddings

A model trained with **Matryoshka Representation Learning** applies the contrastive loss at
*multiple* truncation lengths simultaneously, so the first $k$ dimensions of the vector are
themselves a valid (slightly weaker) embedding.

```
   full 1536-dim  ████████████████████████████████  100% quality
   first 768      ████████████████                   ~99%
   first 256      █████                              ~97%
   first 64       █                                  ~90%
```

> [!TIP]
> **The practical pattern this enables — adaptive retrieval:**
> 1. Store 64-dim truncations for everything (24× less memory).
> 2. Retrieve 1000 candidates fast on the short vectors.
> 3. Re-rank those 1000 with the full-length vectors.

Near-full accuracy at a fraction of the index cost, with no separate model.

---

## 6. Hybrid search

> [!WARNING]
> **Dense embeddings fail at exact matching.** Product codes, error numbers, function names, rare
> proper nouns, legal citations — these have weak or no semantic signal, and a dense model maps them
> somewhere arbitrary.

**BM25** — the classic sparse retriever — handles exactly those cases:

$$\text{BM25}(q,d) = \sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d) + k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}$$

with $k_1 \approx 1.2$ (term-frequency saturation) and $b \approx 0.75$ (length normalization).

| | BM25 (sparse) | Dense embeddings |
|---|---|---|
| Exact terms, IDs, codes | ✅ excellent | ❌ poor |
| Synonyms, paraphrase | ❌ none | ✅ excellent |
| Rare/out-of-vocab words | ✅ handled | ⚠️ depends on tokenizer |
| Typos | ❌ | ✅ somewhat robust |
| Cross-lingual | ❌ | ✅ with a multilingual model |
| Cost | very cheap | needs a model + index |
| Explainability | ✅ which terms matched | ❌ opaque |

**Reciprocal Rank Fusion** — the standard way to combine them, and notably it needs no score
calibration:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60$$



```python
def reciprocal_rank_fusion(*ranked_lists, k=60):
    """Each input is a list of doc ids, best first."""
    scores = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

results = reciprocal_rank_fusion(bm25_search(q, 100), dense_search(q, 100))
```

> [!TIP]
> **Why RRF uses ranks rather than scores**: BM25 scores are unbounded and corpus-dependent; cosine
> scores live in $[-1,1]$ and are model-dependent. There is no principled way to put them on a
> common scale. Ranks sidestep the problem entirely.

Hybrid retrieval consistently beats either component alone on realistic benchmarks (BEIR), often
by 5–15 points of nDCG. **If you are building retrieval and only have dense search, adding BM25 is
usually the highest-value next step.**

---

## 7. Rerankers (cross-encoders)

```
   BI-ENCODER (for retrieval)          CROSS-ENCODER (for reranking)

   query ──►[model]──► q_vec           ┌──────────────────────┐
                        ╲              │ [CLS] query [SEP] doc│
                         cos           └──────────┬───────────┘
                        ╱                    [model]
   doc ────►[model]──► d_vec                     ▼
                                              score

   ✅ docs embedded ONCE, offline      ❌ must run the model per (q,d) pair
   ✅ fast: index lookup               ❌ 100 docs = 100 forward passes
   ❌ query and doc never interact     ✅ full cross-attention between them
                                       ✅ much more accurate
```

**The standard two-stage pipeline:**

```
  10 M docs ──[ANN, ~1ms]──► top 100 ──[cross-encoder, ~50ms]──► top 10 ──► LLM
```

Reranking typically adds 5–15 points of nDCG@10 over retrieval alone. It is the second-highest-
value addition to a RAG system after hybrid search.

> [!TIP]
> **Why cross-encoders are so much better**: a bi-encoder must compress a document into a fixed
> vector *without knowing the query*. A cross-encoder sees both together and can attend from query
> terms directly to document terms. The cost is that you cannot precompute anything.

---

## 8. Implementation

A complete hybrid search + rerank pipeline:

```python
import numpy as np, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

class HybridSearch:
    def __init__(self, docs, embed_model="BAAI/bge-base-en-v1.5",
                 rerank_model="BAAI/bge-reranker-base"):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model)

        # --- sparse index ---
        self.bm25 = BM25Okapi([d.lower().split() for d in docs])

        # --- dense index ---
        emb = self.embedder.encode(docs, normalize_embeddings=True,   # normalize!
                                   batch_size=64, show_progress_bar=True)
        self.index = faiss.IndexHNSWFlat(emb.shape[1], 32)   # M=32
        self.index.hnsw.efConstruction = 200
        self.index.add(emb.astype('float32'))                # inner product == cosine
        self.index.hnsw.efSearch = 128                       # tune this for recall/latency

    def search(self, query, k=5, candidates=100):
        # 1. sparse
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:candidates].tolist()

        # 2. dense
        q = self.embedder.encode([query], normalize_embeddings=True).astype('float32')
        _, dense_top = self.index.search(q, candidates)
        dense_top = dense_top[0].tolist()

        # 3. fuse by rank
        fused = reciprocal_rank_fusion(bm25_top, dense_top)[:candidates]

        # 4. rerank the shortlist with a cross-encoder
        pairs = [(query, self.docs[i]) for i in fused]
        scores = self.reranker.predict(pairs)
        order = np.argsort(scores)[::-1][:k]
        return [(self.docs[fused[i]], float(scores[i])) for i in order]
```

> [!WARNING]
> **`normalize_embeddings=True` on both indexing and querying.** Mismatched normalization is one
> of the most common and most silent retrieval bugs — recall degrades without any error.

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Embeddings are trained contrastively; **hard negatives** are what make them good. |
| 2 | Normalize vectors, then cosine and L2 ranking are identical. |
| 3 | Cosine thresholds are model-specific — calibrate on your own data, never copy a number. |
| 4 | Under ~100k vectors, brute force is fine. ANN is for ≥1 M. |
| 5 | HNSW for quality, IVF-PQ for memory (32× compression), HNSW-PQ for the usual compromise. |
| 6 | PQ is lossy — always re-rank the shortlist with exact vectors. |
| 7 | Matryoshka embeddings let you truncate: retrieve on 64 dims, re-rank on 1536. |
| 8 | Dense fails on IDs, codes and rare terms. **Hybrid (BM25 + dense) with RRF beats either alone.** |
| 9 | RRF fuses on ranks, avoiding the impossible problem of calibrating incomparable scores. |
| 10 | A cross-encoder reranker over the top 100 adds 5–15 nDCG points. |

---

## Further reading

- Karpukhin et al., *Dense Passage Retrieval* (2020) — the foundational DPR paper.
- Malkov & Yashunin, *Efficient and Robust ANN Search using HNSW* (2016).
- Jégou et al., *Product Quantization for Nearest Neighbor Search* (2011).
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Information Retrieval* (2021).
- Kusupati et al., *Matryoshka Representation Learning* (2022).
- Cormack et al., *Reciprocal Rank Fusion* (2009).

**Next** → [Retrieval-augmented generation](03-rag.md)
