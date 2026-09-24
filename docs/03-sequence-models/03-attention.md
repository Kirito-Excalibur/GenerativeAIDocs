# Attention

> **Summary** — Attention is a differentiable, content-addressed lookup table. Each position emits
> a *query*, every position offers a *key* and a *value*, and the output is a weighted average of
> values with weights given by query–key similarity. This page derives it from the database
> analogy, traces a full numeric example end-to-end with real numbers, explains the $\sqrt{d_k}$
> scaling factor with its variance derivation, and covers multi-head, MQA/GQA, and FlashAttention.

**Prerequisites**: → [RNNs, LSTMs & GRUs](02-rnn-lstm-gru.md) · **Next**: → [The Transformer](04-transformer.md)

---

## 1. From a dictionary to attention

Start with a Python dictionary:

```python
d = {"cat": [1, 0], "dog": [0, 1]}
d["cat"]     # -> [1, 0]       exact match, hard lookup
d["kitten"]  # -> KeyError     no partial credit
```

Now make it **soft**: instead of requiring an exact key match, compute a *similarity* between the
query and every key, and return a weighted blend of all values.

```
   HARD LOOKUP                          SOFT LOOKUP (attention)

   query "cat"                          query q
      │                                    │
      ├─ key "cat"  ✓ ──► value₁           ├─ key₁ ── sim 0.7 ──┐
      ├─ key "dog"  ✗                      ├─ key₂ ── sim 0.2 ──┼──► Σ wᵢ·valueᵢ
      └─ key "cow"  ✗                      └─ key₃ ── sim 0.1 ──┘

   returns one value                    returns a weighted average
   not differentiable                   DIFFERENTIABLE ⇒ learnable
```

$$\text{Attention}(q, K, V) = \sum_{i} \underbrace{\frac{\exp(q\cdot k_i/\sqrt{d_k})}{\sum_j \exp(q\cdot k_j/\sqrt{d_k})}}_{\text{attention weight } \alpha_i} v_i$$

> [!TIP]
> **The essential move**: `softmax` turns similarities into a probability distribution, and the
> output is an expectation. Everything is differentiable, so the model *learns what to look up*.

**In matrix form**, for all queries at once:

$$\boxed{\;\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\;}$$

| Symbol | Shape | Meaning |
|---|---|---|
| $Q = XW_Q$ | $(T, d_k)$ | "what am I looking for?" |
| $K = XW_K$ | $(T, d_k)$ | "what do I contain?" |
| $V = XW_V$ | $(T, d_v)$ | "what do I contribute if selected?" |
| $QK^\top$ | $(T, T)$ | all pairwise similarities |
| output | $(T, d_v)$ | one blended vector per position |

> [!TIP]
> **Why three separate projections?** Because the role a token plays when *searching* differs from
> the role it plays when *being found*, which differs again from what it *contributes*. The word
> "bank" asking about its own meaning (query) needs to match against "river" advertising itself as a
> disambiguator (key), and then receive "water-related context" (value). Using one shared vector for
> all three collapses these roles and measurably hurts.

---

## 2. A complete numeric trace

Those three projections and their roles are easiest to trust once you've seen them act on actual numbers — Q, K and V computed, dot products taken, softmax applied, output produced, start to finish.

Let's compute self-attention on a 3-token sequence with $d_{\text{model}} = 4$, $d_k = 2$.

**Input embeddings** (rows = tokens: "the", "cat", "sat"):

$$X = \begin{pmatrix} 1 & 0 & 1 & 0\\ 0 & 1 & 0 & 1\\ 1 & 1 & 0 & 0\end{pmatrix}$$

**Weight matrices** (each $4\times2$):

$$W_Q = \begin{pmatrix}1&0\\0&1\\1&0\\0&1\end{pmatrix},\quad
W_K = \begin{pmatrix}0&1\\1&0\\0&1\\1&0\end{pmatrix},\quad
W_V = \begin{pmatrix}1&1\\0&1\\1&0\\1&1\end{pmatrix}$$

**Step 1 — project.**

$$Q = XW_Q = \begin{pmatrix}2&0\\0&2\\1&1\end{pmatrix},\qquad
K = XW_K = \begin{pmatrix}0&2\\2&0\\1&1\end{pmatrix},\qquad
V = XW_V = \begin{pmatrix}2&1\\1&2\\1&2\end{pmatrix}$$

*(Check row 1 of $Q$: $[1,0,1,0]\cdot$ col 1 of $W_Q$ $= 1\cdot1+0\cdot0+1\cdot1+0\cdot0 = 2$ ✓)*

**Step 2 — scores $S = QK^\top$.**

$$S = \begin{pmatrix}2&0\\0&2\\1&1\end{pmatrix}\begin{pmatrix}0&2&1\\2&0&1\end{pmatrix}
= \begin{pmatrix}0&4&2\\4&0&2\\2&2&2\end{pmatrix}$$

**Step 3 — scale by $\sqrt{d_k} = \sqrt{2} = 1.414$.**

$$S/\sqrt{2} = \begin{pmatrix}0&2.828&1.414\\2.828&0&1.414\\1.414&1.414&1.414\end{pmatrix}$$

**Step 4 — row-wise softmax.**

Row 1: $\exp(0)=1.000$, $\exp(2.828)=16.91$, $\exp(1.414)=4.113$. Sum $= 22.03$.

$$\alpha_1 = (0.045,\ 0.768,\ 0.187)$$

Row 2 (symmetric): $\alpha_2 = (0.768,\ 0.045,\ 0.187)$

Row 3: all scores equal → $\alpha_3 = (0.333,\ 0.333,\ 0.333)$

$$A = \begin{pmatrix}0.045&0.768&0.187\\0.768&0.045&0.187\\0.333&0.333&0.333\end{pmatrix}$$

*(Every row sums to 1.0 ✓)*

**Step 5 — output $= AV$.**

Row 1: $0.045(2,1) + 0.768(1,2) + 0.187(1,2) = (0.090+0.768+0.187,\ 0.045+1.536+0.374)$
$= (1.045, 1.955)$

Row 2: $0.768(2,1)+0.045(1,2)+0.187(1,2) = (1.536+0.045+0.187,\ 0.768+0.090+0.374) = (1.768, 1.232)$

Row 3: $\frac13[(2,1)+(1,2)+(1,2)] = (1.333, 1.667)$

$$\text{Output} = \begin{pmatrix}1.045 & 1.955\\ 1.768 & 1.232\\ 1.333 & 1.667\end{pmatrix}$$

> [!TIP]
> **Read the attention matrix $A$.** Token 1 put 77% of its weight on token 2; token 2 put 77% on
> token 1. They have "found" each other because their query/key projections align. Token 3's query
> is equidistant from all keys, so it averages everything — the *uninformative* case. In a trained
> model, most rows look like row 1 (peaked) and few look like row 3.

---

## 3. Why scale the scores? (the √d factor)

That trace used small, well-behaved numbers on purpose. At the dimensions Transformers actually use, the raw dot products $QK^\top$ grow large enough to break the softmax — which is exactly why a scaling factor sits in the attention formula.

**The variance argument.** Let $q, k \in \mathbb{R}^{d_k}$ have independent components with mean
0 and variance 1. Then

$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i$$

$$\mathbb{E}[q\cdot k] = 0, \qquad \operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i) = d_k$$

So the dot product has standard deviation $\sqrt{d_k}$. **Dividing by $\sqrt{d_k}$ restores unit
variance**, independent of head dimension.

> [!WARNING]
> **Why it matters so much.** Softmax saturates. With $d_k = 64$, unscaled scores have
> $\sigma = 8$, so typical score *gaps* between the largest and second-largest are ~10–15. Then:

$\operatorname{softmax}(15, 0, 0) = (0.99999969,\ 1.5\times10^{-7},\ 1.5\times10^{-7})$

The distribution is effectively one-hot. And the gradient of softmax is
$\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$, so with $p_i \approx 1$ or $\approx 0$ the
gradient is $\approx 0$ **everywhere**. Training stalls at initialization.

**Side-by-side, $d_k = 64$, scores drawn from $\mathcal{N}(0, d_k)$:**

| | max prob | entropy (nats) | max gradient |
|---|---|---|---|
| Unscaled | 0.9997 | 0.003 | $3\times10^{-4}$ |
| Scaled by $\sqrt{64}=8$ | 0.31 | 2.1 | 0.21 |

Three orders of magnitude difference in gradient signal. The $\sqrt{d_k}$ is not a cosmetic
detail — without it, deep Transformers do not train.

**Modern addendum**: even with scaling, attention logits can grow during long training runs as
$\|q\|$ and $\|k\|$ drift upward, causing instability. The fix now standard in large models is
**QK-norm** — apply RMSNorm to $Q$ and $K$ before the dot product, bounding the logits by
construction. → [Optimization §7](../01-foundations/05-optimization.md#7-loss-spikes-the-debugging-playbook)

---

## 4. Multi-head attention

Scaling fixes the softmax for one query-key comparison at a time. A single set of Q/K/V projections still only lets each position look for *one* kind of relationship at once — multiple heads is how attention looks for several at the same time, in parallel.

One attention operation produces one weighted average — one "relationship" per position. Language
needs many simultaneously: syntactic dependency, coreference, and topical relevance are different
lookups.

$$\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O, \qquad
\text{head}_h = \text{Attention}(XW_Q^h, XW_K^h, XW_V^h)$$

with $d_h = d_{\text{model}}/H$ so total compute matches single-head attention at full width.

```
              X  (T, d=512)
              │
   ┌──────────┼──────────┬──────────┐
   ▼          ▼          ▼          ▼
 head 1     head 2     head 3  …  head 8      each: d_h = 64
 (T,64)     (T,64)     (T,64)     (T,64)
   │          │          │          │
   └──────────┴────┬─────┴──────────┘
                   ▼
            concat → (T, 512)
                   ▼
              W_O (512,512)
                   ▼
              output (T, 512)
```

> [!TIP]
> **Why splitting helps rather than hurts.** A single 512-dim attention computes *one* softmax
> distribution per query. Eight 64-dim heads compute *eight independent* distributions. The model
> gets 8 different "views" of the sequence for the same FLOPs. Lower per-head dimension is a real
> cost (each head's similarity is lower-rank), but the diversity wins — up to a point. Empirically
> $d_h = 64$ or $128$ is the sweet spot; heads much narrower than 64 degrade.

**Interpretability finding** — heads specialize in identifiable ways:

| Head type | What it does |
|---|---|
| Positional / previous-token | attends to $i-1$ |
| Syntactic | verb → subject, noun → its determiner |
| **Induction head** | finds a previous occurrence of the current token and attends to what followed it |
| Coreference | pronoun → antecedent |
| Delimiter / "attention sink" | dumps weight on token 0 or a period — a **no-op** |

> [!TIP]
> **Induction heads deserve special mention.** They implement the pattern "…[A][B]… [A] → [B]": if
> the current token appeared earlier, copy whatever followed it last time. Olsson et al. (2022)
> showed these heads form abruptly during training, and their formation coincides with a visible
> bump in the loss curve and with the emergence of **in-context learning**. This is the best-understood
> mechanistic story linking a circuit to a capability.

> [!WARNING]
> **Attention sinks** — trained models put large weight on the first token even when it is
> semantically irrelevant. The reason: softmax *must* sum to 1, so when a head has nothing to
> attend to, it needs somewhere to dump the probability mass. Token 0 is visible to every position
> (under causal masking) and becomes the default dump. Practical consequence: **if you evict token 0
> from the KV cache during streaming, quality collapses.** StreamingLLM's fix is to always keep the
> first few tokens. An alternative architectural fix is to add a learnable "no-op" slot to the
> softmax denominator.

---

## 5. Complexity, and the memory wall

Multiple heads multiply how much attention costs, not just how well it works. That cost is worth making precise, because it's what eventually breaks — and where.

| | Time | Memory |
|---|---|---|
| Compute $Q,K,V$ | $O(Td^2)$ | $O(Td)$ |
| $QK^\top$ | $O(T^2 d)$ | $O(T^2)$ ⚠️ |
| Softmax | $O(T^2)$ | $O(T^2)$ |
| $AV$ | $O(T^2 d)$ | $O(Td)$ |
| **Total** | $O(T^2 d + Td^2)$ | $O(T^2 + Td)$ |

**The crossover point.** $T^2d$ vs $Td^2$: attention dominates when $T > d$. For $d = 4096$,
that's sequences beyond 4096 tokens. Below that, the MLP and projections dominate — which
surprises people who assume attention is always the bottleneck.

**The memory wall**: $B=8$, $H=32$, $T=8192$, BF16:

$$8 \times 32 \times 8192^2 \times 2\text{ bytes} = 34.4\text{ GB}$$

**For one layer's attention matrix.** An 80 GB GPU cannot materialize it. This is the problem
FlashAttention solves.

---

## 6. FlashAttention: never materialize the matrix

34.4 GB for one layer's attention matrix isn't a fundamental limit of the *computation* — it's a limit of naively writing that matrix to memory. Avoiding the write, without changing what gets computed, is FlashAttention's entire trick.

> [!TIP]
> **The key realization**: attention is **memory-bandwidth bound**, not compute bound. The GPU
> spends its time moving the $T\times T$ matrix between fast SRAM (~20 MB, ~19 TB/s) and slow HBM
> (80 GB, ~2 TB/s) — not doing arithmetic.

**FlashAttention** (Dao et al., 2022) tiles the computation so the full matrix is never written to
HBM:

```
   STANDARD                              FLASHATTENTION
   
   1. S = QKᵀ        write T×T to HBM    for each block of Q rows:
   2. P = softmax(S) read+write T×T        for each block of K,V:
   3. O = PV         read T×T                load tiles into SRAM
                                             compute partial scores
   HBM traffic: O(T² + Td)                   update running softmax
                                             accumulate into O
                                        HBM traffic: O(T²d/M + Td)
                                        (M = SRAM size)
```

**Online softmax** — the trick that makes tiling possible. Softmax needs a global maximum and a
global sum, which seems to require seeing everything. But both can be updated incrementally:

$$m^{(new)} = \max(m^{(old)}, m^{(block)}), \qquad
\ell^{(new)} = e^{m^{(old)} - m^{(new)}}\ell^{(old)} + e^{m^{(block)}-m^{(new)}}\ell^{(block)}$$

Rescale the accumulated output by the same correction factor and you get the exact same answer as
full softmax, block by block.

**Results**: 2–4× wall-clock speedup, and memory drops from $O(T^2)$ to $O(T)$ — which is what
made 100k+ token contexts feasible at all. **FlashAttention is exact**, not an approximation. This
is a pure systems win with zero quality cost.

You get it for free:

```python
# PyTorch dispatches to FlashAttention automatically when shapes/dtypes allow
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 7. MQA and GQA: shrinking the KV cache

FlashAttention makes the attention *matrix* cheap to compute during training. At inference, a different piece of memory dominates — the cached keys and values from every previous token — and that one takes a different fix.

At inference, every generated token must attend to all previous keys and values, so they are
cached. The cache grows linearly with sequence length **and with the number of heads**.

$$\text{KV cache bytes} = 2 \times L \times H_{kv} \times d_h \times T \times B \times \text{bytes}$$

```
  MHA (H=32 query, 32 kv)     GQA (32 query, 8 kv)      MQA (32 query, 1 kv)

  q₁ q₂ q₃ … q₃₂              q₁q₂q₃q₄ q₅q₆q₇q₈ …       q₁ q₂ q₃ … q₃₂
  │  │  │     │                 ╲ │ │ ╱    ╲│││╱           ╲ ╲ │ ╱ ╱
  k₁ k₂ k₃ … k₃₂                  kv₁        kv₂             ╲╲│╱╱
                                                               kv₁
  full quality                8 kv heads                  1 kv head
  32× cache                   4× smaller cache            32× smaller cache
                              ~no quality loss            measurable quality loss
```

**70B model, 8192 tokens, BF16** ($L = 80$, $d_h = 128$):

| Scheme | $H_{kv}$ | KV cache / sequence |
|---|---|---|
| MHA | 64 | **21.5 GB** |
| GQA | 8 | **2.7 GB** |
| MQA | 1 | **0.34 GB** |

**Why GQA won**: MQA saves the most memory but degrades quality noticeably. GQA (Ainslie et al.,
2023) with 8 KV heads recovers essentially all MHA quality at 8× less cache. Every major model
since LLaMA-2-70B uses GQA.

**MLA (multi-head latent attention)**, introduced by DeepSeek, goes further: compress K and V
into a shared low-rank latent, cache only that, and reconstruct per-head keys and values on the
fly. Reported cache reduction is larger than GQA's with quality equal to or better than MHA.

---

## 8. Cross-attention vs self-attention

Every optimization so far — heads, FlashAttention, GQA — assumed queries, keys and values all come from the same sequence. That's a design choice, not a requirement, and dropping it is what lets attention condition on something else entirely.

| | Self-attention | Cross-attention |
|---|---|---|
| $Q$ from | the sequence itself | the *decoder* / target stream |
| $K, V$ from | the sequence itself | the *encoder* / conditioning stream |
| Purpose | contextualize within one sequence | inject external information |
| Where | every Transformer block | encoder–decoder models, text-to-image U-Nets, VLMs |

```
  SELF-ATTENTION                    CROSS-ATTENTION
  
   X ──┬──► Q ──┐                   image latents ──► Q ──┐
       ├──► K ──┼──► out                                  ├──► out
       └──► V ──┘                   text embeddings ─► K,V┘

  "how do my tokens relate          "which parts of the text
   to each other?"                   should this image region use?"
```

> [!TIP]
> **Cross-attention is *the* conditioning mechanism in generative AI.** In Stable Diffusion, the
> text prompt is encoded once by CLIP, and every U-Net block cross-attends to it — that is precisely
> how the prompt steers the image. In a VLM, text tokens cross-attend to image patch embeddings. In
> translation, target tokens cross-attend to source tokens.
> → [Latent diffusion](../05-diffusion-and-vision/03-latent-diffusion.md)

---

## 9. Implementation

That's every variant of attention this page covers, in words and diagrams. Putting the core mechanism — queries, keys, values, softmax — into a working few lines of code ties it back to the numeric trace in §2.

Multi-head attention written out, then the fast version:

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, causal=True):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads     # GQA: n_kv_heads < n_heads
        self.d_head     = d_model // n_heads
        self.causal     = causal
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        # QK-norm for training stability in deep models
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if kv_cache is not None:                      # incremental decoding
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'], kv_cache['v'] = k, v

        if self.n_kv_heads != self.n_heads:           # GQA: repeat kv heads
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # dispatches to FlashAttention; never materializes the T×T matrix
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and kv_cache is None)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)
```

> [!WARNING]
> Note `is_causal=self.causal and kv_cache is None`: during incremental decoding you feed exactly
> one query token that legitimately attends to *all* cached keys, so the causal mask must be off.
> Leaving it on is a classic bug that produces subtly wrong generations.

---

## 10. Exercises

**Problem 1 — small attention step, $d_k=4$.** Query $q=(1, 0.5, -0.5, 0)$, keys
$k_1=(1,0,0,0)$, $k_2=(0,1,0,0)$. Compute the scaled dot-product scores and the resulting
softmax weights (ignore any other keys). Which key gets more attention, and does that match
which key $q$ is more "aligned" with?

<details markdown="1"><summary>Solution</summary>

$q\cdot k_1 = 1$, $q\cdot k_2=0.5$. Scaled by $1/\sqrt{4}=0.5$: $s_1=0.5$, $s_2=0.25$.

Softmax: $p_1 = e^{0.5}/(e^{0.5}+e^{0.25}) = 0.562$, $p_2=0.438$.

Key 1 gets more weight (56.2% vs 43.8%), matching the raw dot products: $q$'s first component
(1.0) is larger than its second (0.5), and $k_1,k_2$ each pick out exactly one of those
components, so $q$ is "more aligned" with $k_1$'s direction — the softmax weighting reflects that
alignment, though notice it's far softer than the raw ratio (1.0 vs 0.5 raw scores becomes only
56/44 in probability, not 67/33) — this is the scaling and the softmax's inherent smoothing at
work.

</details>

**Problem 2 — why $\sqrt{d_k}$ and not $d_k$ or $\log d_k$.** §3 derives that
$\text{Var}(q\cdot k)=d_k$ under the stated assumptions, hence dividing by $\sqrt{d_k}$ restores
unit variance. Suppose someone instead divides by $d_k$ (not $\sqrt{d_k}$). What would happen to
the variance of the scaled scores as $d_k$ grows, and what symptom would you expect for large
$d_k$ (e.g. $d_k=128$)?

<details markdown="1"><summary>Solution</summary>

If scores are divided by $d_k$ instead of $\sqrt{d_k}$: variance of the scaled score becomes
$\text{Var}(q\cdot k)/d_k^2 = d_k/d_k^2 = 1/d_k$. As $d_k$ grows, this variance **shrinks toward
0** rather than staying at 1.

Symptom at $d_k{=}128$: variance $\approx1/128=0.0078$ — scores would be almost identical to each
other (tiny spread), pushing softmax toward the **opposite** failure mode from the unscaled case
in §3's table: instead of one-hot saturation, you'd get an almost perfectly *uniform*
distribution regardless of the actual query-key alignment, discarding essentially all of the
similarity signal the dot product carries. Both over-scaling and under-scaling break attention,
just in opposite directions — $\sqrt{d_k}$ is not an arbitrary choice but the unique scaling that
keeps variance at exactly 1 regardless of head dimension.

</details>

**Problem 3 — RoPE relative-position property, different offset.** Using §6's rotation
formula with $\theta=(1.0, 0.1)$, verify that $\langle R_3 q, R_7 k\rangle$ (positions 3 and 7,
offset 4) equals $\langle R_0 q, R_4 k\rangle$ (positions 0 and 4, offset 4) for
$q=k=(1,0,1,0)$, confirming the relative-position property holds for an offset other than the
one worked in §6 (which used $m{=}2,n{=}5$).

<details markdown="1"><summary>Solution</summary>

Rotating $q=(1,0,1,0)$ by $m=3$ and $k=(1,0,1,0)$ by $n=7$ (offset $n-m=4$), then computing the
dot product, gives $\langle R_3q, R_7k\rangle = 0.2674$.

Rotating $q$ by $m=0$ (i.e. unrotated) and $k$ by $n=4$ (offset $4-0=4$) gives
$\langle R_0q, R_4k\rangle = 0.2674$ — **identical** to 6 decimal places.

This confirms $\langle R_mq,R_nk\rangle$ depends only on $n-m$, independent of the absolute
positions $m,n$ themselves — the algebraic identity $R_m^\top R_n = R_{n-m}$ derived in §6 holds
for this new offset just as it did for the offset-3 example worked in the page itself.

</details>

## 11. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Attention is a differentiable soft dictionary: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. |
| 2 | Q, K, V are separate projections because searching, being found, and contributing are different roles. |
| 3 | $\sqrt{d_k}$ scaling keeps score variance at 1; without it softmax saturates and gradients vanish at init. |
| 4 | Multi-head = multiple independent lookups for the same FLOPs. $d_h \approx 64$–128 is the sweet spot. |
| 5 | Induction heads implement "[A][B]…[A]→[B]" and their formation coincides with in-context learning. |
| 6 | Attention sinks exist because softmax must sum to 1; never evict token 0 from a streaming cache. |
| 7 | Attention dominates cost only when $T > d$; below that, the MLP does. |
| 8 | FlashAttention is exact — it only changes memory traffic, giving 2–4× speed and $O(T)$ memory. |
| 9 | GQA with 8 KV heads cuts the cache 8× at negligible quality cost. This is why every modern model uses it. |
| 10 | Cross-attention is the universal conditioning mechanism: text→image, image→text, source→target. |

---

## Further reading

- Bahdanau et al., [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) (2015) — attention's origin.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017).
- Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) and [*FlashAttention-2*](https://arxiv.org/abs/2307.08691) (2023).
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245) (2023).
- Olsson et al., [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022).
- Xiao et al., [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453) (2023).
- Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021).

**Next** → [The Transformer](04-transformer.md)
