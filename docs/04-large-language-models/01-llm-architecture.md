# LLM Architecture

> **Summary** — What a modern large language model actually is, component by component: the
> decoder-only stack, the specific choices (RMSNorm, SwiGLU, RoPE, GQA, no biases) and why each
> one is there, how a request flows through the system, and the memory/compute budget at every
> stage. This page is the practical companion to → [The Transformer](../03-sequence-models/04-transformer.md).

**Prerequisites**: → [The Transformer](../03-sequence-models/04-transformer.md), → [Positional encoding](../03-sequence-models/05-positional-encoding.md) · **Next**: → [Pretraining](02-pretraining.md)

---

## 1. The whole model on one diagram

```
                     token ids  (B, T)
                         │
              ┌──────────▼──────────┐
              │  Token Embedding    │   V × d      e.g. 128256 × 4096
              └──────────┬──────────┘
                         │  (B, T, d)
   ┌─────────────────────▼─────────────────────┐
   │                                            │  ×L  (e.g. 32 layers)
   │   ┌────────────────────────────────────┐  │
   │   │  RMSNorm                           │  │
   │   │  Grouped-Query Attention + RoPE    │  │   ← positions mix here
   │   │  residual add                      │  │
   │   └────────────────────────────────────┘  │
   │   ┌────────────────────────────────────┐  │
   │   │  RMSNorm                           │  │
   │   │  SwiGLU MLP  (d → 8d/3 → d)        │  │   ← per-token compute
   │   │  residual add                      │  │
   │   └────────────────────────────────────┘  │
   └─────────────────────┬─────────────────────┘
                         │  (B, T, d)
              ┌──────────▼──────────┐
              │   Final RMSNorm     │
              └──────────┬──────────┘
              ┌──────────▼──────────┐
              │  LM head (d × V)    │   often tied to the embedding
              └──────────┬──────────┘
                         │  (B, T, V)  logits
                         ▼
                  softmax → next-token distribution
```

**That is the entire model.** Everything that distinguishes a 1B model from a 500B one is $L$, $d$,
the amount of data, and the post-training.

---

## 2. Component-by-component rationale

| Component | Choice | Why this one |
|---|---|---|
| Norm placement | **pre-norm** | clean residual highway → stable at any depth |
| Norm type | **RMSNorm** | drops mean subtraction; ~10% faster, no quality cost |
| Activation | **SwiGLU** | multiplicative gating; best perplexity per parameter |
| $d_{\text{ff}}$ | $\frac83 d$ rounded to ×256 | keeps SwiGLU's 3 matrices parameter-neutral vs 2×4d |
| Position | **RoPE** in attention | relative, parameter-free, extensible after training |
| Attention | **GQA**, $H_{kv}\in\{4,8\}$ | 8× smaller KV cache at ~no quality cost |
| Biases | **none** | redundant after a norm; improves stability |
| Dropout | **0.0** in pretraining | models under-fit at <1 epoch |
| Embedding tying | tied at small $V$, untied at large | untied helps when $Vd$ is a large parameter share |
| Attention stability | **QK-norm** | bounds pre-softmax logits; prevents late-run spikes |
| Precision | **BF16** + FP32 optimizer states | full FP32 range, half the memory |

🧠 **The pattern across this table**: the modern architecture is the 2017 Transformer with things
*removed* (biases, dropout, mean-centering, separate encoder) and two things *replaced* (activation,
position encoding). Almost nothing was added. That convergence is a sign the architecture is near a
local optimum — the action has moved to data, scale, and post-training.

---

## 3. Where the parameters and the memory go

🔢 **LLaMA-3 8B**, fully worked ($L{=}32$, $d{=}4096$, $H{=}32$, $H_{kv}{=}8$, $d_{ff}{=}14336$, $V{=}128256$):

| Component | Formula | Parameters |
|---|---|---|
| Token embedding | $Vd$ | 525.3 M |
| Attention per layer | $d^2 + 2\cdot d\cdot\frac{d}{4} + d^2 = 2.5d^2$ | 41.9 M |
| MLP per layer | $3\,d\,d_{ff}$ | 176.2 M |
| Norms per layer | $2d$ | 8 K |
| **Per layer total** | | **218.1 M** |
| × 32 layers | | 6.98 B |
| Final norm | $d$ | 4 K |
| LM head (untied) | $Vd$ | 525.3 M |
| **Total** | | **8.03 B** ✓ |

Note GQA's effect: attention is $2.5d^2$ instead of $4d^2$ because $W_K$ and $W_V$ produce only
$d/4$ outputs each.

🔢 **Inference memory**, BF16, batch 1, 8192 tokens:

| Item | Size |
|---|---|
| Weights | $8.03\text{B}\times2 = 16.1$ GB |
| KV cache | $2\times32\times8\times128\times8192\times2 = 1.07$ GB |
| Activations (transient) | ~0.5 GB |
| **Total** | **~17.7 GB** |

Fits on a 24 GB consumer GPU. In **INT4** the weights drop to ~4 GB and the whole thing fits in
8 GB — which is why 4-bit quantization made local LLMs a real thing.
→ [Efficiency](07-efficiency.md)

🔢 **Training memory** for the same model (AdamW, BF16, FP32 master):

$$8.03\times10^9 \times 16\text{ bytes} = 128\text{ GB} \quad\text{+ activations}$$

Needs multiple GPUs with sharding. → [Pretraining](02-pretraining.md)

---

## 4. Request lifecycle: prefill and decode

Serving an LLM has **two phases with completely different performance characteristics**. Confusing
them is the most common source of bad inference engineering.

```
  PREFILL                                DECODE
  process the whole prompt at once       generate one token at a time

  prompt: 2000 tokens                    ┌─► forward pass on 1 token
      │                                  │        │
      ▼  ONE forward pass                │        ▼
  ┌────────────────┐                     │   sample next token
  │ 2000×2000 attn │                     │        │
  │ big matmuls    │                     └────────┘  repeat N times
  └────────────────┘
      │                                  each step: tiny matmul,
      ▼                                  but must READ ALL WEIGHTS
  KV cache filled                        from HBM

  ⇒ COMPUTE-bound                        ⇒ MEMORY-BANDWIDTH-bound
  ⇒ high GPU utilization                 ⇒ ~5% GPU utilization
  ⇒ scales with prompt length            ⇒ scales with output length
```

📐 **The arithmetic intensity argument, which explains everything about LLM serving.**

Decoding one token for a 7B model in BF16:
- FLOPs: $2N = 1.4\times10^{10}$
- Bytes read from HBM: $14\times10^9$ (all the weights, once)
- Arithmetic intensity: $\mathbf{1}$ FLOP per byte

An H100 has ~1000 TFLOP/s BF16 and ~3.35 TB/s HBM bandwidth → a ratio of ~300 FLOPs per byte. At
intensity 1, you are using roughly $1/300$ of the available compute. **The GPU is idle, waiting on
memory.**

🧠 **The one fix that matters: batching.** Process 64 sequences at once and you read the weights
*once* for 64 tokens of output. Arithmetic intensity rises to 64, throughput rises nearly 64×, and
per-request latency barely changes.

📊 **Practical consequences:**

| Observation | Explanation |
|---|---|
| Throughput scales ~linearly with batch size until the KV cache fills memory | decode is bandwidth-bound |
| Time-to-first-token scales with prompt length | prefill is compute-bound |
| Time-per-output-token is nearly constant | decode cost is dominated by weight reads |
| Quantization speeds up decode a lot, prefill little | fewer bytes to read; same FLOPs |
| A 2× smaller model is ~2× faster at decode | half the weights to read |

**Continuous batching** (vLLM, TGI, SGLang) exploits this: instead of waiting for a whole batch to
finish, evict finished sequences and admit new ones every step. Reported throughput gains over
naive static batching are large (often 5–20×).
→ [Inference & decoding](06-inference-and-decoding.md)

---

## 5. Model size tiers, and what each is for

📊 A practical map (parameter counts, September 2026):

| Tier | Params | Runs on | Typical use |
|---|---|---|---|
| **Tiny** | 0.1–1 B | phone, browser (WASM) | classification, autocomplete, on-device assist |
| **Small** | 1–4 B | laptop CPU, 8 GB GPU | drafting, simple extraction, speculative-decoding drafts |
| **Mid** | 7–15 B | one consumer/prosumer GPU | general assistant, RAG, fine-tuning target |
| **Large** | 30–80 B | 1–4 datacenter GPUs | strong reasoning, code, agents |
| **Frontier** | 200 B+ (often MoE) | a cluster | best-in-class capability |

🧠 **The most useful trend to internalize**: capability at a given parameter count keeps improving.
A 2026-vintage 8B model outperforms a 2022 70B model on most benchmarks. The gains come from
better data (more, cleaner, more synthetic/curated), longer training well past Chinchilla-optimal,
and much better post-training — **not** from architecture.

📊 **Why models are trained far past compute-optimal**: Chinchilla optimizes *training* cost. If you
will serve billions of tokens, inference cost dominates the total, and a smaller model trained on
more data is cheaper forever. LLaMA-3 8B saw ~15T tokens — about **1875 tokens per parameter**,
roughly 94× the Chinchilla ratio of 20. → [Scaling laws](03-scaling-laws.md)

---

## 6. Dense vs Mixture-of-Experts

```
  DENSE                             MIXTURE OF EXPERTS

  every token → every parameter     every token → router → top-2 of 8 experts

  ┌──────────────┐                  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
  │   MLP (big)  │                  │E1 ││E2 ││E3 ││E4 ││E5 ││E6 ││E7 ││E8 │
  └──────────────┘                  └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘
                                       ▲          ▲
  N params, N active                   └── router picks 2 ──┘

                                    8N params total, 2N active
                                    ⇒ capacity of 8N, cost of 2N
```

| | Dense 70B | MoE 8×22B (≈141B total, ≈39B active) |
|---|---|---|
| Total parameters | 70 B | 141 B |
| Active per token | 70 B | 39 B |
| Training FLOPs/token | $6\times70$B | $6\times39$B |
| **Memory to serve** | 140 GB (BF16) | **282 GB** ⚠️ |
| Quality | baseline | better per FLOP |

⚠️ **The MoE catch**: you save *compute*, not *memory*. All experts must be resident, because any
token might route to any of them. MoE is a win when you are compute-bound (large batches) and lose
when you are memory-bound (single-user local inference). → [Mixture of Experts](08-mixture-of-experts.md)

---

## 7. The pipeline from base model to product

```mermaid
graph LR
    A["Raw web data<br/>~100T tokens"] --> B["Filter & dedup<br/>~15T tokens"]
    B --> C["PRETRAIN<br/>next-token prediction<br/>months, $10M+"]
    C --> D["Base model<br/>completes text,<br/>doesn't follow instructions"]
    D --> E["SFT<br/>10k-1M curated<br/>instruction pairs"]
    E --> F["Preference tuning<br/>RLHF / DPO"]
    F --> G["Chat model"]
    G --> H["+ tools, RAG,<br/>system prompt,<br/>safety filters"]
    H --> I["Product"]

    style C fill:#2b6cb0,stroke:#2c5282,color:#fff
    style F fill:#276749,stroke:#22543d,color:#fff
```

📊 **The cost asymmetry is striking:**

| Stage | Compute | Data | Wall-clock |
|---|---|---|---|
| Pretraining | ~99% | 10–30 T tokens | weeks–months |
| SFT | ~0.5% | 10 K–1 M examples | hours–days |
| Preference tuning | ~0.5% | 10 K–1 M comparisons | hours–days |

🧠 **The "Superficial Alignment Hypothesis"** (Zhou et al., LIMA, 2023): essentially all knowledge
and capability is acquired during pretraining; post-training only teaches the model *which
distribution of responses to produce*. Evidence: LIMA fine-tuned LLaMA-65B on **1,000** carefully
curated examples and was competitive with RLHF'd models on many prompts.

⚠️ **The hypothesis is only partly right, and the exception matters.** RL on verifiable tasks
(math, code) at scale does appear to teach genuinely new capability, not just style — reasoning
models trained this way solve problems their base models could not solve at any sampling budget.
The current best understanding: **style is superficial, reasoning is not.**
→ [Reasoning](10-reasoning.md)

---

## 8. Reading a model config

💻 A real `config.json` and what every field means:

```jsonc
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,               // d — the residual stream width
  "intermediate_size": 14336,        // d_ff — SwiGLU hidden (note: 3.5d, not 8d/3)
  "num_hidden_layers": 32,           // L
  "num_attention_heads": 32,         // H  (query heads)
  "num_key_value_heads": 8,          // H_kv — GQA with 4 queries per kv head
  "vocab_size": 128256,              // V
  "max_position_embeddings": 8192,   // trained context length
  "rope_theta": 500000.0,            // RoPE base — raised from 10000 for long context
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": false,      // separate LM head (adds V·d params)
  "torch_dtype": "bfloat16",
  "attention_bias": false,           // no biases
  "hidden_act": "silu"               // SwiGLU (silu = swish)
}
```

🔢 **Estimate the parameter count from this file in your head:**

$$N \approx L(2.5d^2 + 3d\,d_{ff}) + 2Vd = 32(4.19\times10^7 + 1.76\times10^8) + 1.05\times10^9 \approx 8.0\text{ B}$$ ✓

🔢 **And the FLOPs to train it on 15T tokens:**

$$6 \times 8\times10^9 \times 1.5\times10^{13} = 7.2\times10^{23}\text{ FLOPs}$$

At 400 TFLOP/s effective per GPU on 16,000 GPUs: $\approx 1.25\times10^5$ s $\approx$ **1.5 days**
of pure compute (real runs take longer due to restarts, evals and data loading).

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Modern LLM = pre-norm decoder stack with RMSNorm, SwiGLU, RoPE, GQA, no biases, no dropout. |
| 2 | The architecture converged; progress now comes from data, scale, and post-training. |
| 3 | Serving has two phases: prefill is compute-bound, decode is memory-bandwidth-bound. |
| 4 | Decode has arithmetic intensity ≈ 1 FLOP/byte on a ~300 FLOP/byte machine — batching is the fix. |
| 5 | Models are trained far past Chinchilla-optimal because inference cost dominates over a model's lifetime. |
| 6 | MoE saves compute, not memory — all experts stay resident. |
| 7 | Pretraining is ~99% of the compute; post-training is ~99% of the perceived quality. |
| 8 | You can estimate any model's size and training cost from its config in under a minute. |

---

## Further reading

- Touvron et al., *LLaMA 2* (2023); Grattafiori et al., *The Llama 3 Herd of Models* (2024) — unusually detailed.
- Chowdhery et al., *PaLM* (2022) — thorough on architecture and training infrastructure.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023).
- Zhou et al., *LIMA: Less Is More for Alignment* (2023).
- Pope et al., *Efficiently Scaling Transformer Inference* (2022) — the arithmetic-intensity analysis.

**Next** → [Pretraining](02-pretraining.md)
