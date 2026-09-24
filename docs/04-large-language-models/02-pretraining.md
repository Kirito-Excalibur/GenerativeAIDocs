# Pretraining

> **Summary** — How a base model is actually made: where the data comes from and how it is
> filtered, what the objective is, how the model is split across thousands of GPUs (data, tensor,
> pipeline and sequence parallelism), what it costs, and what goes wrong. Pretraining is ~99% of
> the compute in an LLM's life and the stage where essentially all knowledge is acquired.

**Prerequisites**: → [LLM architecture](01-llm-architecture.md), → [Optimization](../01-foundations/05-optimization.md) · **Next**: → [Scaling laws](03-scaling-laws.md)

---

## 1. The objective

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$$

That's it. Next-token prediction, cross-entropy, teacher forcing.

> [!TIP]
> **Why such a simple objective produces such general capability.** To predict the next token well
> across a corpus of everything humans have written, a model must implicitly learn:

| To predict… | The model must learn… |
|---|---|
| the next word in a sentence | syntax, morphology |
| `"The capital of France is ___"` | facts |
| the closing brace in code | structure, scope, state tracking |
| `"1234 × 5678 = ___"` | arithmetic procedures |
| the next line of a dialogue | theory of mind, social conventions |
| the conclusion of an argument | inference, logic |
| the punchline | humour, expectation violation |

Next-token prediction is a *universal* task: any capability that reduces prediction loss on human
text will be acquired if the data contains it and the model has capacity. Ilya Sutskever's framing
— "to compress text well you must understand it" — is the compact version, and it is backed by the
formal equivalence between compression and prediction (→ [Probability & information theory](../01-foundations/02-probability-and-information-theory.md)).

> [!WARNING]
> **The limits of the framing.** The objective rewards *predicting what a human would write*, not
> *being correct*. If the training corpus contains confident-sounding falsehoods, predicting them
> accurately is rewarded. This is one structural root of hallucination and sycophancy.
> → [Safety](../08-safety-and-ethics/01-safety.md)

---

## 2. Data: the actual differentiator

A typical frontier pretraining mix (proportions vary; this is representative):

| Source | Share | Tokens | Notes |
|---|---|---|---|
| Filtered web (CommonCrawl-derived) | 50–70% | 5–20 T | the bulk; quality filtering is everything |
| Code (GitHub, permissive licences) | 10–20% | 1–3 T | improves reasoning even on non-code tasks |
| Books | 5–10% | 0.5–1 T | long-form coherence, narrative structure |
| Academic (arXiv, PubMed) | 2–5% | 0.2–0.5 T | technical depth, LaTeX, citations |
| Wikipedia & references | 1–3% | 0.05 T | high-quality factual grounding, often upsampled |
| Q&A, forums (StackExchange) | 2–5% | 0.2 T | instruction-shaped text |
| Multilingual | 5–15% | varies | |
| **Synthetic / model-generated** | **5–30%** ↑ | varies | textbooks, rewrites, distilled reasoning traces |

> [!TIP]
> **Code is the interesting entry.** Models trained with code do better on *non-code* reasoning
> tasks. The leading hypotheses: code is unusually structured (long-range dependencies, explicit
> state), it contains many worked procedures, and it is implicitly verified (it had to run). This is
> now a standard ingredient regardless of whether the model targets programming.

**Synthetic data is the fastest-growing entry.** The Phi model series demonstrated that
"textbook-quality" generated data can produce models far above their weight class. The tradeoff:
synthetic data inherits its generator's biases and errors, and heavy reliance risks **model
collapse** (Shumailov et al., 2024) — variance narrowing across generations as models train on
their own distribution. Current practice keeps a substantial human-written base.

### The filtering pipeline

```
  Common Crawl: ~100 T tokens raw
           │
           ▼  language ID (fastText)      ──► drop ~60%
  ┌─────────────────┐
  │  URL/domain     │  blocklists (adult, spam, known-bad)
  │  filtering      │
  └────────┬────────┘
           ▼  quality classifier          ──► drop ~50% of remainder
  ┌─────────────────┐   • perplexity under a reference LM
  │  quality        │   • classifier trained on "good" vs random web
  │  filtering      │   • heuristics: symbol ratio, mean word length,
  └────────┬────────┘     stopword presence, bullet fraction, line repetition
           ▼  deduplication               ──► drop ~30%
  ┌─────────────────┐   • exact: hash whole documents
  │  dedup          │   • fuzzy: MinHash + LSH on shingles
  │  (exact+fuzzy)  │   • substring: suffix-array for long repeated spans
  └────────┬────────┘
           ▼  decontamination
  ┌─────────────────┐   remove documents containing benchmark test items
  │  PII removal    │   strip emails, phone numbers, keys
  └────────┬────────┘
           ▼
     ~10-15 T high-quality tokens
```

**Deduplication is the highest-value single step.** Lee et al. (2022) showed that deduplicating
training data:
- reduces memorized regurgitation by an order of magnitude,
- *improves* perplexity on held-out data,
- reduces training compute needed for a given loss.

**Why duplicates hurt so much**: a document appearing 1000 times gets 1000× the gradient weight,
so the model memorizes it verbatim instead of generalizing. Web crawls are full of boilerplate,
mirrored sites, and syndicated articles.

> [!WARNING]
> **Decontamination is harder than it sounds** and routinely fails. N-gram overlap detection
> misses paraphrases, translations, and reformatted versions of benchmark questions. When you read a
> benchmark score, assume some contamination.
> → [Benchmarks](../07-evaluation/02-benchmarks.md)

### Data ordering

Not all data is equal at all times. Common practices:

- **Curriculum**: general web early, high-quality/technical data upweighted later.
- **Annealing phase**: for the last ~5% of tokens, train almost exclusively on the highest-quality
  data at a decaying learning rate. This has an outsized effect on final benchmark scores and is
  now standard.
- **Long-context staging**: train at 8k, then extend to 32k, then 128k — each stage on
  progressively longer documents. → [Long context](09-long-context.md)

---

## 3. Distributed training: the four parallelisms

A 70B model needs ~1.1 TB for weights + gradients + optimizer states. You must split it.

```
 DATA PARALLEL (DP)              TENSOR PARALLEL (TP)
 same model, different data      one layer split across GPUs

  GPU0    GPU1    GPU2            GPU0      GPU1
 ┌────┐  ┌────┐  ┌────┐         ┌──────┐  ┌──────┐
 │full│  │full│  │full│         │ W[:, │  │ W[:, │   split the weight
 │model│ │model│ │model│        │  :h] │  │  h:] │   matrix column-wise
 └────┘  └────┘  └────┘         └──────┘  └──────┘
 batch0  batch1  batch2          same input, partial outputs
     ↓ all-reduce gradients          ↓ all-reduce activations
  ✅ simple                       ✅ shrinks per-GPU memory
  ❌ full model on each GPU       ❌ communication EVERY layer
                                     ⇒ needs NVLink; keep within a node

 PIPELINE PARALLEL (PP)          SEQUENCE / CONTEXT PARALLEL (SP)
 different layers on different   split the sequence dimension
 GPUs

  GPU0     GPU1     GPU2          GPU0          GPU1
 ┌─────┐  ┌─────┐  ┌─────┐       tokens 0-4k   tokens 4k-8k
 │L1-8 │─►│L9-16│─►│L17-24│      ┌──────────┐ ┌──────────┐
 └─────┘  └─────┘  └─────┘       │ full model│ │full model│
  ✅ low communication            └──────────┘ └──────────┘
  ❌ pipeline BUBBLES              ✅ enables very long context
     (idle time)                   ❌ attention needs cross-GPU comm
                                      (ring attention)
```

### The pipeline bubble

> [!WARNING]
> With naive pipelining, GPU 0 computes layer 1 while GPUs 1–3 sit idle, then GPU 1 works while
> the rest idle. Utilization is $1/P$.

**Fix: micro-batching.** Split the batch into $m$ micro-batches and pipeline them:

$$\text{bubble fraction} = \frac{P-1}{m + P - 1}$$

$P = 4$ stages, $m = 32$ micro-batches: bubble $= 3/35 = 8.6\%$. Acceptable.
With $m = 4$: bubble $= 3/7 = 43\%$. Unacceptable. **Always use $m \gg P$.**

### ZeRO: sharding the optimizer state

> [!TIP]
> In plain data parallelism, every GPU stores an identical copy of the 16 bytes/param of optimizer
> state. That is pure redundancy. **ZeRO** (Rajbhandari et al., 2020) shards it:

| Stage | Shards | Memory per GPU ($N$ params, $D$ GPUs) | Extra communication |
|---|---|---|---|
| ZeRO-0 (plain DP) | nothing | $16N$ | baseline |
| **ZeRO-1** | optimizer states | $4N + 12N/D$ | none |
| **ZeRO-2** | + gradients | $2N + 14N/D$ | none |
| **ZeRO-3 (FSDP)** | + parameters | $16N/D$ | +50% |

**7B model on 64 GPUs**: ZeRO-0 needs 112 GB/GPU (impossible on 80 GB). ZeRO-3 needs
$112/64 = 1.75$ GB/GPU. That is the difference between infeasible and comfortable.

> [!WARNING]
> ZeRO-3 gathers each layer's parameters just before use and frees them after, so it adds
> communication. On a well-connected cluster this overlaps with compute and is nearly free; on a
> poorly-connected one it dominates.

### Putting it together: 3-D parallelism

A real frontier configuration, 70B model on 1024 GPUs:

| Axis | Degree | Scope |
|---|---|---|
| Tensor parallel | 8 | within a node (NVLink, ~900 GB/s) |
| Pipeline parallel | 8 | across nodes (InfiniBand) |
| Data parallel | 16 | across pipeline replicas |
| **Total** | $8\times8\times16 = 1024$ | |

> [!TIP]
> **The placement rule**: put the chattiest parallelism on the fastest interconnect. Tensor
> parallelism communicates twice per layer, so it must stay inside a node. Data parallelism
> communicates once per step, so it can span the whole cluster.

---

## 4. Cost

**The formula**: $C = 6ND$ FLOPs.

| Model | $N$ | $D$ (tokens) | FLOPs | GPU-hours @400 TF/s | Cost @\$2/h |
|---|---|---|---|---|---|
| 1 B | $10^9$ | 20 B | $1.2\times10^{20}$ | 83 | \$170 |
| 7 B | $7\times10^9$ | 2 T | $8.4\times10^{22}$ | 58,000 | \$117 K |
| 8 B | $8\times10^9$ | 15 T | $7.2\times10^{23}$ | 500,000 | \$1.0 M |
| 70 B | $7\times10^{10}$ | 15 T | $6.3\times10^{24}$ | 4.4 M | \$8.8 M |
| 400 B | $4\times10^{11}$ | 15 T | $3.6\times10^{25}$ | 25 M | \$50 M |

> [!WARNING]
> **These are compute costs only.** Real budgets add: failed runs and ablations (often 2–5× the
> final run), data acquisition and processing, engineering salaries, and idle cluster time. Public
> estimates for frontier models are typically 3–10× the raw compute number.

**MFU** (model FLOPs utilization) is the efficiency metric:

$$\text{MFU} = \frac{6ND}{t \cdot F_{\text{peak}} \cdot n_{\text{GPU}}}$$

| MFU | Verdict |
|---|---|
| < 25% | something is badly wrong |
| 35–45% | typical for a well-tuned large run |
| 50–60% | excellent |
| > 65% | rare; usually small models or unusual hardware |

---

## 5. What goes wrong

Public training logs (OPT-175B, BLOOM) document this honestly — large runs are messy.

| Failure | Frequency | Response |
|---|---|---|
| **Hardware failure** | 1 GPU/node failure every few hours at 1000+ GPUs | frequent checkpointing (every 15–60 min), automatic restart |
| **Loss spike** | several per run | roll back ~500 steps, skip the data shard, resume |
| **Silent data corruption** | rare but devastating | checksums; monitor loss on a held-out canary set |
| **NaN / inf** | occasional | BF16 instead of FP16; check for bad data |
| **Slow node ("straggler")** | common | every GPU waits for the slowest; detect and eject |
| **Divergence** | run-ending if unfixed | lower LR, longer warmup, QK-norm, z-loss |

> [!TIP]
> **The checkpointing arithmetic**: with 1000 GPUs and a mean time between failures of 4 hours,
> you will lose all unsaved progress that often. Checkpointing a 70B model with optimizer state
> means writing ~1.1 TB; at 10 GB/s that's ~110 seconds. Checkpoint every 30 minutes and you lose
> 6% to checkpoint writes plus ~15 minutes of work per failure. That is the equilibrium most runs
> settle on.

**Metrics to watch, in priority order:**

1. **Training loss** — should decrease smoothly on a log-log plot as a near-straight line.
2. **Gradient norm** — slow decay with occasional 2–3× spikes; sustained growth is a warning.
3. **MFU** — a drop means a systems problem (straggler, network, data loader).
4. **Held-out loss on a fixed canary set** — catches data corruption that training loss hides.
5. **Downstream evals** every few thousand steps — loss can improve while capabilities don't.
6. **Activation/weight norms per layer** — outlier growth predicts instability.

---

## 6. Long-context and other staged phases

Modern pretraining is not one uniform phase. A representative LLaMA-3-style schedule:

| Phase | Tokens | Context | LR | Data |
|---|---|---|---|---|
| Main pretraining | 14 T | 8 K | peak → decayed | full mix |
| Long-context extension | 0.8 T | 8 K → 128 K staged | low | long documents, books, repos |
| **Annealing** | 40 B | 8 K | → 0 | highest-quality subset only |

> [!TIP]
> **Why long context comes last**: attention is $O(T^2)$, so training at 128k from the start would
> be ruinously expensive. Train cheaply at short context to acquire knowledge, then spend a small
> fraction of the budget teaching the model to *use* long contexts. The RoPE base is raised at the
> same time (→ [Positional encoding §7](../03-sequence-models/05-positional-encoding.md#7-extending-the-context-window)).

> [!TIP]
> **Why annealing works**: at the end of training the learning rate is small, so updates are
> precise and mostly preserved. Feeding the highest-quality data in this window lets it
> disproportionately shape the final weights. Public reports describe several-point benchmark gains
> from the annealing phase alone.

---

## 7. A small-scale recipe you can actually run

Reproducible settings for a ~124M-parameter GPT on a single 24 GB GPU:

```python
# Model (GPT-2 small scale)
config = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
              vocab_size=50304)            # padded to a multiple of 64 for kernel efficiency

# Data: ~10B tokens (e.g. FineWeb-Edu sample), pre-tokenized into a uint16 memmap
# Chinchilla-optimal for 124M is ~2.5B tokens; 10B is a reasonable over-train.

# Optimization
optimizer      = "AdamW"
betas          = (0.9, 0.95)
weight_decay   = 0.1          # matrices only
grad_clip      = 1.0
learning_rate  = 6e-4
warmup_iters   = 700
lr_decay_iters = 600_000
min_lr         = 6e-5         # 10% of peak

# Batching: 0.5M tokens/step via gradient accumulation
batch_size            = 12     # per-device micro-batch
gradient_accumulation = 40     # 12 * 1024 * 40 = 491,520 tokens/step
dtype                 = "bfloat16"
compile               = True   # torch.compile: ~1.5-2x speedup
```

**Expected results**: ~4 days on one RTX 4090, final validation loss ≈ 3.0 (perplexity ≈ 20),
comparable to the original GPT-2 124M. Total electricity cost: a few dollars.

> [!TIP]
> **This is the single highest-value exercise in this wiki.** Actually running a pretraining job
> end to end — watching the loss curve, hitting an instability, fixing a data-loader bug — teaches
> more than any amount of reading. `nanoGPT` and `modded-nanogpt` are the standard starting points.

---

## 8. Key takeaways

| # | Takeaway |
|---|---|
| 1 | The objective is next-token cross-entropy. Its generality comes from the generality of text. |
| 2 | Data quality and deduplication matter more than architecture. Dedup improves loss *and* reduces memorization. |
| 3 | Code in the mix improves general reasoning, not just coding. |
| 4 | Four parallelisms: data (simple), tensor (intra-node only), pipeline (needs $m \gg P$), sequence (for long context). |
| 5 | ZeRO-3/FSDP shards optimizer state, gradients and parameters → memory scales as $16N/D$. |
| 6 | $C = 6ND$; a 70B model on 15T tokens is ~$6\times10^{24}$ FLOPs ≈ \$10 M of compute alone. |
| 7 | MFU 40–50% is good. Below 30% means a systems bug, not a model problem. |
| 8 | Failures are routine: checkpoint every 15–60 min and expect to roll back through loss spikes. |
| 9 | Training is staged: main → long-context extension → high-quality annealing. |

---

## Further reading

- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020).
- Grattafiori et al., *The Llama 3 Herd of Models* (2024) — the most detailed public account of a real run.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020).
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters* (Megatron-LM, 2021).
- Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022).
- Penedo et al., *The FineWeb Datasets* (2024) — an open, well-documented filtering pipeline.
- Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022) — read the appendix logbook.

**Next** → [Scaling laws](03-scaling-laws.md)
