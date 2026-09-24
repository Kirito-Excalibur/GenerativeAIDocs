# The Transformer

> **Summary** — The architecture that everything else is built on. A stack of identical blocks,
> each containing multi-head attention and an MLP, wired with residual connections and
> normalization. This page goes through it block by block, gives exact parameter-count and FLOP
> formulas you can compute in your head, walks a tensor through the whole stack with shapes at
> every step, and catalogues the design variants and which ones actually won.

**Prerequisites**: → [Attention](03-attention.md) · **Next**: → [Positional encoding](05-positional-encoding.md)

---

## 1. The three families

```
   ENCODER-ONLY            DECODER-ONLY           ENCODER-DECODER
   (BERT)                  (GPT, LLaMA)           (T5, original)

   ┌──────────┐            ┌──────────┐        ┌────────┐  ┌──────────┐
   │ bidirec- │            │  causal  │        │ bidir. │─►│ causal + │
   │  tional  │            │   attn   │        │  enc   │  │  cross   │
   │   attn   │            └──────────┘        └────────┘  └──────────┘
   └──────────┘
   sees left+right         sees left only       enc: full input
   ✅ understanding        ✅ generation         dec: generates
   ❌ can't generate       ✅ scales best        ✅ clean separation
                                                 ❌ 2× params, less used

   classification,         ALL modern LLMs       translation, T5-style
   embeddings, NER                               multitask, some VLMs
```

> [!TIP]
> **Why decoder-only won.** Three reasons, in order of importance:

1. **Every token is a training signal.** A decoder-only model predicts *all* $T$ positions from one
   forward pass. BERT's masked LM only supervises the ~15% of tokens that were masked — roughly
   6× less learning per FLOP.
2. **Generation and understanding unify.** Any task can be phrased as text continuation, so one
   model and one objective cover everything.
3. **Simplicity scales.** One homogeneous stack is easier to shard, pipeline, and reason about
   than an asymmetric encoder–decoder.

Raffel et al. (T5, 2020) compared all three systematically. Encoder–decoder was slightly better
at matched parameter count — but decoder-only was better at matched *compute*, and compute is what
you actually spend.

---

## 2. The block, in detail

**Pre-norm** (every modern model):

$$
\begin{aligned}
x &\leftarrow x + \text{MHA}(\text{Norm}(x))\\
x &\leftarrow x + \text{MLP}(\text{Norm}(x))
\end{aligned}
$$

```
                  x  (B, T, d)
                  │
          ┌───────┴───────┐
          │               │  residual
     ┌────▼─────┐         │
     │  RMSNorm │         │
     └────┬─────┘         │
     ┌────▼──────────┐    │
     │  Multi-Head   │    │      ← the ONLY place tokens exchange
     │  Attention    │    │        information
     │  (+ RoPE)     │    │
     └────┬──────────┘    │
          │               │
         (+)◄─────────────┘
          │
          ├───────────────┐
     ┌────▼─────┐         │  residual
     │  RMSNorm │         │
     └────┬─────┘         │
     ┌────▼──────────┐    │
     │  SwiGLU MLP   │    │      ← per-token processing; no mixing
     │  d → 8d/3 → d │    │        (2/3 of all parameters live here)
     └────┬──────────┘    │
          │               │
         (+)◄─────────────┘
          │
          ▼  (B, T, d)
```

> [!TIP]
> **The essential division of labour**: attention **mixes across positions**, the MLP **transforms
> each position independently**. Every Transformer is an alternation of "gather information from
> elsewhere" and "think about what you've gathered." Neither alone suffices — an attention-only stack
> is nearly linear; an MLP-only stack cannot see context.

> [!TIP]
> **The residual stream view** (from interpretability): think of the $(B,T,d)$ tensor flowing
> down the stack as a **shared communication bus**. Each sublayer *reads* a projection of it,
> computes, and *writes an additive update* back. Nothing is ever overwritten. This is why you can
> meaningfully talk about "the direction in the residual stream that represents X," and why
> techniques like activation steering and logit lens work at all.

---

## 3. Parameter counting, exactly

Per layer, with $d$ = model dimension, $d_{\text{ff}}$ = MLP hidden dimension:

| Component | Parameters | With $d_{\text{ff}} = 4d$ (GELU) | With SwiGLU, $d_{\text{ff}}=\frac83 d$ |
|---|---|---|---|
| $W_Q, W_K, W_V, W_O$ | $4d^2$ | $4d^2$ | $4d^2$ |
| MLP (2 matrices) | $2 d\,d_{\text{ff}}$ | $8d^2$ | — |
| MLP (3 matrices, SwiGLU) | $3 d\,d_{\text{ff}}$ | — | $8d^2$ |
| Norms | $2d$ | negligible | negligible |
| **Per layer** | | $\mathbf{12d^2}$ | $\mathbf{12d^2}$ |

**Total model:**

$$\boxed{\;N \approx 12\,L\,d^2 + V d \;(\times 2 \text{ if embeddings untied})\;}$$

**Verify against real models:**

| Model | $L$ | $d$ | $V$ | $12Ld^2$ | + embed | Actual |
|---|---|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 50257 | 85.0 M | +38.6 M | **124 M** ✓ |
| GPT-2 XL | 48 | 1600 | 50257 | 1.475 B | +80 M | **1.56 B** ✓ |
| GPT-3 | 96 | 12288 | 50257 | 173.9 B | +0.6 B | **175 B** ✓ |
| LLaMA-2 7B | 32 | 4096 | 32000 | 6.44 B | +0.26 B | **6.7 B** ✓ |
| LLaMA-2 70B | 80 | 8192 | 32000 | 64.4 B | +0.26 B | 69 B ⚠️ −6% |

The formula is accurate to a few percent for models that follow the $d_{\text{ff}} = \frac83 d$
convention. **Memorize $N \approx 12Ld^2$** — it lets you estimate any model's size from its config
file in five seconds.

> [!WARNING]
> **When it under-counts**: LLaMA-2 70B uses $d_{\text{ff}} = 28672 = 3.5d$, so its SwiGLU MLP is
> $3\cdot d\cdot 3.5d = 10.5d^2$ rather than $8d^2$. Exactly: attention $151$ M + MLP $705$ M
> $= 856$ M per layer, $\times 80$ layers $+ 2Vd = \mathbf{69.0}$ B. Always check $d_{\text{ff}}$ in
> the config before trusting the shortcut.

**The parameter split** (7B model): attention $4d^2/12d^2 = 33\%$, MLP $67\%$. **Two-thirds of an
LLM's parameters are in the feed-forward layers.** This is why MoE replaces the MLP, not attention.
→ [Mixture of Experts](../04-large-language-models/08-mixture-of-experts.md)

### Aspect ratio

Real models keep $d/L \approx 100$–$130$:

| Model | $d$ | $L$ | $d/L$ |
|---|---|---|---|
| GPT-2 small | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |
| LLaMA-2 7B | 4096 | 32 | 128 |
| LLaMA-2 70B | 8192 | 80 | 102 |

> [!TIP]
> Kaplan et al. found model quality is remarkably insensitive to aspect ratio over a wide range —
> what matters is total $N$. But there are practical constraints: deeper models are harder to
> parallelize (pipeline bubbles) and train less stably; wider models waste compute on small batches.
> $d/L \approx 128$ is the empirical compromise.

---

## 4. FLOP counting

A matrix multiply $(m\times k)\times(k\times n)$ costs $2mkn$ FLOPs (one multiply + one add per
element pair).

**Forward pass per token:**

| Component | FLOPs per token |
|---|---|
| QKVO projections | $8d^2$ |
| Attention scores + weighted sum | $4Td$ |
| MLP | $16d^2$ |
| **Per layer** | $24d^2 + 4Td$ |
| **Full model** | $L(24d^2 + 4Td) \approx 2N$ when $T \ll d$ |

**The rule everyone uses:**

$$\boxed{\;C_{\text{forward}} \approx 2N \text{ FLOPs/token}, \qquad C_{\text{train}} \approx 6N \text{ FLOPs/token}\;}$$

(backward pass ≈ 2× forward: one gradient w.r.t. inputs, one w.r.t. weights)

**Worked example — what does training a 7B model on 2T tokens cost?**

$$C = 6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}\text{ FLOPs}$$

On 1024 H100s at 400 TFLOP/s effective (≈40% MFU of BF16 peak):

$$\frac{8.4\times10^{22}}{1024 \times 4\times10^{14}} = 2.05\times10^5 \text{ seconds} = \textbf{2.4 days}$$

At \$2/GPU-hour: $1024 \times 57\text{ h} \times \$2 \approx \$117{,}000$. 📊

**When does attention start to matter?** The $4Td$ term vs the $24d^2$ term:

| $d$ | $T$ | $24d^2$ | $4Td$ | Attention share |
|---|---|---|---|---|
| 4096 | 2048 | $4.0\times10^8$ | $3.4\times10^7$ | 7.7% |
| 4096 | 8192 | $4.0\times10^8$ | $1.3\times10^8$ | 25% |
| 4096 | 32768 | $4.0\times10^8$ | $5.4\times10^8$ | **57.1%** |
| 4096 | 131072 | $4.0\times10^8$ | $2.1\times10^9$ | **84.2%** |

> [!WARNING]
> **Common misconception corrected**: at typical training lengths (2–8k), attention is only
> 10–25% of compute. It only dominates past ~32k tokens. Optimizing attention for a 2k-context model
> is optimizing the wrong thing.

**MFU (model FLOPs utilization)** — the metric to track:

$$\text{MFU} = \frac{6ND/t}{\text{peak FLOP/s}\times\text{n\_gpus}}$$

Good large-scale runs achieve 40–55% MFU. Below 30%, you have a systems problem (bad sharding,
communication bottleneck, small batch, poor kernel coverage).

---

## 5. A tensor's journey, with shapes

$B=2$, $T=1024$, $d=4096$, $L=32$, $H=32$, $H_{kv}=8$, $V=32000$:

| Step | Operation | Output shape | Params used |
|---|---|---|---|
| 0 | input token IDs | $(2, 1024)$ | — |
| 1 | embedding lookup | $(2,1024,4096)$ | 131 M |
| 2 | RMSNorm | $(2,1024,4096)$ | 4 K |
| 3 | $W_Q$ | $(2,1024,4096)$ | 16.8 M |
| 4 | $W_K$, $W_V$ (GQA) | $(2,1024,1024)$ each | 4.2 M each |
| 5 | reshape to heads | $q:(2,32,1024,128)$, $kv:(2,8,1024,128)$ | — |
| 6 | RoPE on $q,k$ | same | 0 (no params) |
| 7 | repeat kv 4× | $(2,32,1024,128)$ | — |
| 8 | attention (flash) | $(2,32,1024,128)$ | — |
| 9 | merge heads | $(2,1024,4096)$ | — |
| 10 | $W_O$ | $(2,1024,4096)$ | 16.8 M |
| 11 | **residual add** | $(2,1024,4096)$ | — |
| 12 | RMSNorm | $(2,1024,4096)$ | 4 K |
| 13 | gate & up proj | $(2,1024,11008)$ each | 45.1 M each |
| 14 | SiLU × gate | $(2,1024,11008)$ | — |
| 15 | down proj | $(2,1024,4096)$ | 45.1 M |
| 16 | **residual add** | $(2,1024,4096)$ | — |
| | *repeat 2–16 × 32 layers* | | **6.5 B total** |
| 17 | final RMSNorm | $(2,1024,4096)$ | 4 K |
| 18 | LM head | $(2,1024,32000)$ | 131 M |

> [!WARNING]
> **Step 18 is a memory trap.** The logits tensor is $2\times1024\times32000 = 6.6\times10^7$
> floats = 262 MB in FP32 — often larger than any activation in the model. With a 128k vocabulary and
> a bigger batch it can exceed 10 GB. Fused cross-entropy kernels (which compute the loss in chunks
> without materializing all logits) are standard in modern training code for exactly this reason.

---

## 6. Design decisions: 2017 vs today

| Component | Original (2017) | Modern (2024–26) | Why it changed |
|---|---|---|---|
| Norm placement | post-norm | **pre-norm** | trains at depth without fragile warmup |
| Norm type | LayerNorm | **RMSNorm** | ~10% faster, no quality loss |
| Activation | ReLU | **SwiGLU** | multiplicative gating, better perplexity |
| $d_{\text{ff}}$ | $4d$ | $\frac83 d$ (3 matrices) | keeps SwiGLU parameter-neutral |
| Position | sinusoidal, added | **RoPE**, applied in attention | relative, extrapolates, no params |
| Attention | MHA | **GQA** | 8× smaller KV cache |
| Biases | everywhere | **none** | free parameters, no quality cost, better stability |
| Dropout | 0.1 | **0.0** in pretraining | under-fitting, not over-fitting |
| Embeddings | tied | tied or untied | untied helps at large $V$ |
| Attention stability | — | **QK-norm** | prevents logit blowup in long runs |
| Precision | FP32 | **BF16** + FP32 master | 2× speed, same range |

> [!TIP]
> **The most interesting removal is biases.** LLaMA and PaLM dropped every bias term. The reasoning:
> with a normalization layer immediately before each linear, the bias is largely redundant, and
> removing it improves stability (biases are the parameters most prone to drift) and saves a little
> memory. No measurable quality cost. A good example of the field converging on *less* machinery.

---

## 7. The full modern block

A complete LLaMA-style Transformer block:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d)
        self.attn      = MultiHeadAttention(d, n_heads, n_kv_heads)   # from the attention page
        self.mlp_norm  = nn.RMSNorm(d)
        self.mlp       = SwiGLU(d, d_ff)

    def forward(self, x, freqs_cis, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab=32000, d=4096, n_layers=32, n_heads=32,
                 n_kv_heads=8, max_seq=4096):
        super().__init__()
        # SwiGLU: shrink d_ff to 8d/3, rounded to a multiple of 256 for kernel efficiency
        d_ff = int(2 * (4 * d) / 3)
        d_ff = 256 * ((d_ff + 255) // 256)                   # 4096 -> 11008
        self.tok_emb = nn.Embedding(vocab, d)
        self.layers  = nn.ModuleList(Block(d, n_heads, n_kv_heads, d_ff)
                                     for _ in range(n_layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer('freqs_cis', precompute_rope(d // n_heads, max_seq))

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        freqs = self.freqs_cis[:idx.size(1)]
        for layer in self.layers:
            x = layer(x, freqs)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

Note `d_ff = 11008` for $d = 4096$: that is $\frac83 \times 4096 = 10922.7$, rounded up to the
nearest multiple of 256. LLaMA-2 7B uses exactly this. The rounding matters because GPU tensor
cores want dimensions that are multiples of 128 or 256.

---

## 8. Model configurations reference

Read this table alongside $N \approx 12Ld^2$:

| Model | $N$ | $L$ | $d$ | $H$ | $H_{kv}$ | $d_{ff}$ | Context | Vocab |
|---|---|---|---|---|---|---|---|---|
| GPT-2 small | 124 M | 12 | 768 | 12 | 12 | 3072 | 1024 | 50257 |
| GPT-2 XL | 1.5 B | 48 | 1600 | 25 | 25 | 6400 | 1024 | 50257 |
| GPT-3 | 175 B | 96 | 12288 | 96 | 96 | 49152 | 2048 | 50257 |
| LLaMA-2 7B | 6.7 B | 32 | 4096 | 32 | 32 | 11008 | 4096 | 32000 |
| LLaMA-2 70B | 69 B | 80 | 8192 | 64 | 8 | 28672 | 4096 | 32000 |
| LLaMA-3 8B | 8.0 B | 32 | 4096 | 32 | 8 | 14336 | 8192 | 128256 |
| LLaMA-3 70B | 70 B | 80 | 8192 | 64 | 8 | 28672 | 8192 | 128256 |
| Mistral 7B | 7.2 B | 32 | 4096 | 32 | 8 | 14336 | 8192 (SWA 4096) | 32000 |
| Mixtral 8×7B | 46.7 B total / 12.9 B active | 32 | 4096 | 32 | 8 | 14336 ×8 experts | 32768 | 32000 |

> [!TIP]
> Note LLaMA-3's changes from LLaMA-2: vocabulary 32k → 128k (better compression, especially
> multilingual), GQA at every size (not just 70B), and a larger $d_{ff}$. All three are
> compression/efficiency wins rather than architectural novelty — which is characteristic of where
> the field is.

---

## 9. What the Transformer cannot do

> [!WARNING]
> Worth being precise about, because it clarifies what architectures come next.

| Limitation | Why | Mitigation |
|---|---|---|
| $O(T^2)$ attention | all-pairs by construction | sparse/linear attention, SSMs, retrieval |
| Fixed context window | positional encodings and training distribution | RoPE scaling, YaRN, long-context training |
| No persistent memory across calls | stateless by design | RAG, external memory, caching |
| Fixed compute per token | the same $2N$ FLOPs for "the" and for a hard inference step | chain-of-thought, adaptive depth, MoE |
| Cannot revise emitted tokens | autoregressive commitment | CoT scratchpads, best-of-$n$, diffusion LMs |
| Serial generation | $T$ sequential forward passes | speculative decoding, multi-token prediction |

> [!TIP]
> **"Fixed compute per token" is the deepest one.** A Transformer spends identical FLOPs predicting
> the next token whether the context is "the cat sat on the ___" or "the 47th prime number is ___".
> There is no mechanism for thinking longer about a harder problem. **Chain-of-thought is the
> workaround the field converged on**: use the token stream itself as a variable-length scratchpad,
> buying serial compute at the cost of sequence length. That reframing is what makes reasoning
> models work. → [Reasoning](../04-large-language-models/10-reasoning.md)

---

## 10. Exercises

**Problem 1 — parameter count, a new config.** Estimate $N$ for $L=24$, $d=1536$, $V=32000$
using the $12Ld^2+Vd$ formula from §3. Which size tier (per → [LLM architecture §5](../04-large-language-models/01-llm-architecture.md#5-model-size-tiers-and-what-each-is-for))
does this land in?

<details><summary>Solution</summary>

$$N = 12\times24\times1536^2 + 32000\times1536 = 679{,}477{,}248 + 49{,}152{,}000 = 728{,}629{,}248 \approx 0.73\text{ B}$$

This falls in the "small" tier (1–4 B is small per the linked table — actually 0.73B is just
*below* that range, closer to "tiny/small" boundary), roughly GPT-2-XL scale (1.5B was §3's own
GPT-2 XL example) but somewhat smaller — the kind of size that runs comfortably on a laptop CPU
or a modest GPU, useful for on-device assistance or as a speculative-decoding draft model.

</details>

**Problem 2 — attention's FLOP share, a long-context case.** Using §4's formula
(attention share $= 4Td/(24d^2+4Td)$), compute the attention share of total FLOPs for $d=4096$
at $T=16384$ (an intermediate length not in §4's table). Interpolating between the table's $T=8192$
(25%) and $T=32768$ (57.1%) rows, does your computed value fall roughly where linear
interpolation would suggest, or does the curve bend?

<details><summary>Solution</summary>

$$\text{share} = \frac{4\times16384\times4096}{24\times4096^2+4\times16384\times4096}
= \frac{2.684\times10^8}{4.027\times10^8+2.684\times10^8} = \frac{2.684}{6.711}=0.400=\mathbf{40.0\%}$$

Linear interpolation between 25% ($T{=}8192$) and 57.1% ($T{=}32768$) at the midpoint $T{=}16384$
(which is *not* the linear midpoint of $8192$ and $32768$, but is exactly $2\times$ the smaller
and $\frac12\times$ the larger) would naively suggest something well below the true 40% if you
interpolated on $T$ linearly. The actual relationship is **not linear in $T$** — the formula is a
ratio $\frac{c_1T}{c_2+c_1T}$, which is concave (grows quickly at first, then flattens toward
100%) — so simple linear interpolation between two table rows systematically *underestimates*
the true share at intermediate $T$. Always compute the formula directly rather than
interpolating a nonlinear curve.

</details>

**Problem 3 — the residual-stream reading, applied.** A colleague claims: "since pre-norm
Transformers add every layer's output to the residual stream, and nothing is ever overwritten,
you could in principle train a 1000-layer Transformer and it would work exactly as well as a
32-layer one, just slower." Using §2 and §6, what's right and what's incomplete about this claim?

<details><summary>Solution</summary>

**Right**: the identity-preserving gradient path from pre-norm (§6: "trains at any depth") means
depth alone doesn't cause the training *instability* that killed post-norm at depth — this part
of the claim is well-supported.

**Incomplete**: §2's "residual stream as shared bus" framing implies each layer's contribution is
an *additive update of bounded relative size* — but nothing guarantees a 1000-layer model's
*capacity* is well-used just because it trains stably. In practice: (a) the residual stream's
*magnitude* grows with depth (noted as a known issue in §6's "design decisions" discussion of
quantization outliers), which can create its own numerical problems at extreme depth even without
instability in the classic sense; (b) far more layers means far more parameters and FLOPs for the
same width, and scaling laws (→ [Scaling laws](../04-large-language-models/03-scaling-laws.md))
say the *compute-optimal* shape keeps $d/L\approx100$–130 — a 1000-layer, narrow model is likely
far from compute-optimal compared to a wider, shallower one at the same parameter budget. So
"trains stably" and "is the best use of your parameter/compute budget" are different claims —
the first follows from pre-norm, the second does not.

</details>

## 11. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Decoder-only won because every token is a training signal, and one objective covers all tasks. |
| 2 | Attention mixes across positions; the MLP transforms each position. Alternating these is the whole design. |
| 3 | The residual stream is a shared bus that sublayers read from and add to — nothing overwrites. |
| 4 | $N \approx 12Ld^2 + Vd$. Two-thirds of parameters are in MLPs. |
| 5 | $C_{\text{train}} \approx 6N$ FLOPs/token, $C_{\text{inference}} \approx 2N$. Memorize both. |
| 6 | Attention is only 10–25% of FLOPs at 2–8k context; it dominates past ~32k. |
| 7 | Modern stack: pre-norm + RMSNorm + SwiGLU + RoPE + GQA + no biases + no dropout + BF16. |
| 8 | The final logits tensor ($B\times T\times V$) is often the largest allocation — use fused cross-entropy. |
| 9 | Fixed compute per token is the deepest limitation; chain-of-thought is the workaround. |

---

## Further reading

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019).
- Touvron et al., [*LLaMA*](https://arxiv.org/abs/2302.13971) (2023) and [*LLaMA 2*](https://arxiv.org/abs/2307.09288) (2023) — the modern reference architecture.
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (2020).
- Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021) — the residual stream view.
- Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://github.com/karpathy/nanoGPT) — the best video walkthrough.

**Next** → [Positional encoding](05-positional-encoding.md)
