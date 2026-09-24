# Positional Encoding

> **Summary** — Attention is permutation-equivariant: shuffle the input tokens and the outputs
> shuffle identically. Position must be injected explicitly. This page covers sinusoidal, learned,
> relative, ALiBi and RoPE encodings, derives RoPE's rotation algebra, and explains how context
> windows get extended after training (position interpolation, NTK scaling, YaRN).

**Prerequisites**: → [Attention](03-attention.md) · **Next**: Part IV → [LLM architecture](../04-large-language-models/01-llm-architecture.md)

---

## 1. Why position must be added

**The proof, in two lines.** Let $P$ be a permutation matrix. Then

$$\text{Attention}(PX) = \operatorname{softmax}\!\left(\frac{PQ(PK)^\top}{\sqrt{d_k}}\right)PV
= \operatorname{softmax}\!\left(\frac{PQK^\top P^\top}{\sqrt{d_k}}\right)PV = P\cdot\text{Attention}(X)$$

Permute the inputs, and the outputs are permuted the same way — **no information about order
enters the computation**.

```
  "the cat sat"   ─┐
  "sat cat the"   ─┼──► identical attention outputs (up to permutation)
  "cat the sat"   ─┘
```

> [!TIP]
> A Transformer without positional information is a **bag-of-words model with fancy feature
> mixing**. That is fatal for language, where "dog bites man" and "man bites dog" differ entirely.

> [!WARNING]
> Note that an *MLP* is position-independent too, and the residual stream just carries values
> along. Attention is the only place where positions could interact — so position information must
> be present before or inside attention.

---

## 2. The design space

| Method | Injected | Type | Extrapolates? | Params | Used by |
|---|---|---|---|---|---|
| Sinusoidal | added to embeddings | absolute | ⚠️ poorly | 0 | original Transformer |
| Learned | added to embeddings | absolute | ❌ never | $T_{\max}\cdot d$ | BERT, GPT-2/3 |
| Relative (T5) | bias added to scores | relative | ⚠️ somewhat | $\sim$ buckets × $H$ | T5 |
| **ALiBi** | linear bias on scores | relative | ✅ well | 0 | BLOOM, MPT |
| **RoPE** | rotation applied to $Q,K$ | relative | ⚠️ with scaling | 0 | **LLaMA, GPT-NeoX, Qwen, most modern** |
| NoPE | nothing (causal mask only) | implicit | ⚠️ | 0 | research curiosity |

> [!TIP]
> **The direction of travel**: absolute → relative, added-to-embeddings → applied-in-attention,
> learned → parameter-free. Each shift improves length generalization.

> [!WARNING]
> **NoPE is a genuinely surprising result**: decoder-only Transformers with *no* positional
> encoding at all still learn positional behaviour, because the causal mask breaks symmetry (token
> $i$ sees $i$ tokens, token $j$ sees $j$). Performance is competitive at small scale. It is not used
> in practice but it shows how much structure the mask alone provides.

---

## 3. Sinusoidal encoding

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Added directly to the token embedding: $x_{\text{in}} = E[\text{token}] + PE[pos]$.

```
  dimension i (wavelength)
   0  ──╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲──   λ = 2π      (fastest: alternates every token)
   2  ──╱──╲──╱──╲──╱──╲──   λ ≈ 2π·10
   4  ──╱────╲────╱────╲──   λ ≈ 2π·100
   6  ──╱──────────╲──────   λ ≈ 2π·1000
   …                          
 d-2  ──────────╱────────    λ ≈ 2π·10000  (slowest: barely changes)
      └────────────────────► position
```

> [!TIP]
> **It is binary counting in continuous form.** Low dimensions oscillate fast (fine-grained "which
> of the last few tokens"), high dimensions oscillate slowly (coarse "roughly where in the
> document"). Together they give a unique fingerprint per position at multiple resolutions.

**The elegant property** — $PE_{pos+k}$ is a *linear function* of $PE_{pos}$:

$$\begin{pmatrix}\sin(\omega(pos+k))\\ \cos(\omega(pos+k))\end{pmatrix}
= \begin{pmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{pmatrix}
\begin{pmatrix}\sin(\omega\, pos)\\ \cos(\omega\, pos)\end{pmatrix}$$

A rotation by a fixed angle $\omega k$ depending only on the *offset* $k$. In principle the model
can learn relative attention from absolute encodings.

> [!WARNING]
> **But it extrapolates badly in practice.** The theory says positions beyond training length are
> well-defined; the reality is the model never learned to use those particular phase combinations.
> Quality degrades sharply past $T_{\text{train}}$.

---

## 4. Learned absolute encoding

$$x_{\text{in}} = E[\text{token}] + P[\text{pos}], \qquad P \in \mathbb{R}^{T_{\max}\times d}$$

Just an embedding table for positions. Used by BERT and GPT-2/3.

| ✅ | ❌ |
|---|---|
| simple; the model learns whatever it needs | **hard ceiling at $T_{\max}$** — position 1025 has no embedding |
| slightly better than sinusoidal in-distribution | zero extrapolation |
| | $T_{\max}\cdot d$ wasted parameters (GPT-3: 25 M) |

GPT-2's `wpe` is a $1024 \times 768$ table. Feed it 1025 tokens and you get an index error — not
degraded quality, a **crash**. This hard limit is why learned absolute encodings were abandoned.

---

## 5. ALiBi: a linear bias, no embeddings at all

$$\text{scores}_{ij} = \frac{q_i\cdot k_j}{\sqrt{d_k}} - m_h\cdot|i - j|$$

where $m_h$ is a **fixed, head-specific slope** (geometric: $2^{-8h/H}$ for head $h$).

```
  Bias added to attention scores, one head (m = 0.25):

        j=0    j=1    j=2    j=3    j=4
  i=4  -1.00  -0.75  -0.50  -0.25   0.00
  i=3  -0.75  -0.50  -0.25   0.00    ·
  i=2  -0.50  -0.25   0.00    ·      ·
  i=1  -0.25   0.00    ·      ·      ·
  i=0   0.00    ·      ·      ·      ·

  The further back, the bigger the penalty. A soft recency bias.
```

> [!TIP]
> **Different heads get different slopes**, so they have different effective "attention
> horizons": a head with $m = 2^{-1}$ looks only a few tokens back; a head with $m = 2^{-8}$ sees
> thousands. The model gets a built-in multi-scale view of the context.

✅ **Extrapolation is excellent** — the bias is defined for *any* distance, so a model trained at
1024 tokens works usably at 4096+ without modification. This was ALiBi's headline claim and it
holds.

> [!WARNING]
> But the penalty is monotonic and unbounded: information from very far back is always
> down-weighted, so ALiBi models effectively have a soft sliding window. For tasks requiring exact
> retrieval from deep in a long context, that is a real limitation. It is one reason RoPE + explicit
> context extension overtook ALiBi.

---

## 6. RoPE: rotary position embedding

**The modern default.** Instead of *adding* position information, **rotate** the query and key
vectors by an angle proportional to their position.

Treat each consecutive pair of dimensions as a point in 2-D and rotate it:

$$
\begin{pmatrix}q'_{2i}\\ q'_{2i+1}\end{pmatrix} =
\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
\begin{pmatrix}q_{2i}\\ q_{2i+1}\end{pmatrix},
\qquad \theta_i = 10000^{-2i/d}
$$

where $m$ is the position index.

```
   d=8 vector, split into 4 pairs, each rotated by a different rate:

   pair 0 (θ₀=1.00)      pair 1 (θ₁=0.10)     pair 2 (θ₂=0.01)   pair 3 (θ₃=0.001)
        ↑                      ↑                    ↑                   ↑
      ╱ │                    ╱ │                  · │                 · │
     ╱  │ rotates fast      ╱  │ medium          ─ │ slow            ─ │ barely
    └───┴──►               └───┴──►             └──┴──►             └──┴──►

   position m=0:  no rotation
   position m=1:  rotate pair i by θᵢ
   position m=10: rotate pair i by 10·θᵢ
```

**The key property — the inner product depends only on the relative distance:**

$$\langle R_m q,\; R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

because rotation matrices satisfy $R_m^\top R_n = R_{n-m}$. **Absolute rotations, relative
similarity.** This is the whole trick: you apply an absolute transform, and attention automatically
sees only the difference.

**Worked example.** $d = 4$ (2 pairs), $\theta_0 = 1.0$, $\theta_1 = 0.01$.
Take $q = [1, 0, 1, 0]$ at position $m = 2$, $k = [1, 0, 1, 0]$ at position $n = 5$.

*Rotate $q$ by $m\theta$:*
- pair 0, angle $2 \times 1.0 = 2.0$ rad: $[\cos 2, \sin 2] = [-0.416, 0.909]$
- pair 1, angle $2 \times 0.01 = 0.02$ rad: $[\cos 0.02, \sin 0.02] = [0.9998, 0.0200]$

$q' = [-0.416, 0.909, 0.9998, 0.0200]$

*Rotate $k$ by $n\theta$:*
- pair 0, angle $5.0$ rad: $[0.284, -0.959]$
- pair 1, angle $0.05$ rad: $[0.9988, 0.0500]$

$k' = [0.284, -0.959, 0.9988, 0.0500]$

*Inner product:*
$$q'\cdot k' = (-0.416)(0.284) + (0.909)(-0.959) + (0.9998)(0.9988) + (0.0200)(0.0500)$$
$$= -0.118 - 0.872 + 0.999 + 0.001 = 0.010$$

*Now verify with the relative formula*, $n - m = 3$: rotate $k$ by 3 relative to an unrotated $q$:
- pair 0, angle 3.0: $[-0.990, 0.141]$; pair 1, angle 0.03: $[0.9996, 0.0300]$

$$q \cdot R_3 k = (1)(-0.990) + (0)(0.141) + (1)(0.9996) + (0)(0.0300) = 0.010$$ ✓

**Identical.** Only the distance mattered.

The efficient implementation uses complex numbers:

```python
def precompute_rope(dim, max_seq, base=10000.0):
    # theta_i = base^(-2i/dim) for i in 0..dim/2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)                  # (max_seq, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)   # e^{i·m·theta}

def apply_rope(x, freqs_cis):
    # x: (B, H, T, d_head). Pair up the last dim and treat as complex.
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_r = torch.view_as_real(x_c * freqs_cis[None, None, :, :]).flatten(-2)
    return x_r.type_as(x)

# applied to q and k inside attention, NOT to v, and NOT to the embeddings
q = apply_rope(q, freqs_cis)
k = apply_rope(k, freqs_cis)
```

> [!WARNING]
> **RoPE is applied to $Q$ and $K$ only, never to $V$.** Values carry content, not position —
> rotating them would corrupt what gets copied. Also note it is applied **inside every attention
> layer**, not once at the input. That is a genuine structural difference from additive encodings.

**Why RoPE won:**

| Property | Benefit |
|---|---|
| Relative by construction | generalizes across positions |
| Zero parameters | no table, no ceiling |
| Applied per layer in attention | position information cannot be "forgotten" by the residual stream |
| Decaying inner product with distance | a soft, learnable recency prior |
| **Extensible after training** | the base $\theta$ can be rescaled (§7) — this is the killer feature |

---

## 7. Extending the context window

A model trained at 4k tokens fails badly at 32k — perplexity explodes. The cause: RoPE angles
$m\theta_i$ for $m > T_{\text{train}}$ land in regions of phase space the model never saw.

### Position interpolation (PI)

Squeeze positions into the trained range: replace $m$ with $m \cdot \frac{T_{\text{train}}}{T_{\text{new}}}$.

```
  Original (trained to 4096)    Interpolated (to 16384, scale 4)
  
  m: 0  1  2  3 … 4095          m: 0  0.25  0.5  0.75 … 4095
     │  │  │  │     │              │   │     │    │       │
     all angles in the             all angles STILL in the
     trained range ✓               trained range ✓
                                   but adjacent tokens are now
                                   4× closer in phase ⇒ loses
                                   fine-grained local resolution
```

Works, needs a short fine-tune (~1000 steps), and costs some local precision.

### NTK-aware scaling

> [!TIP]
> **The insight**: don't scale all frequencies equally. High-frequency dimensions encode *local*
> order (which matters and shouldn't be squeezed); low-frequency dimensions encode *global* position
> (which needs the extension). So increase the base $\theta$ instead:

$$\theta_{\text{new}} = \theta_{\text{old}} \cdot s^{d/(d-2)}$$

This stretches low frequencies a lot and high frequencies almost not at all. **Often works with no
fine-tuning at all** — a remarkable, cheap win.

### YaRN

Combines both, with per-dimension treatment based on the wavelength relative to the training
context, plus a temperature correction on the attention softmax to compensate for the increased
number of tokens being averaged.

**Comparison** (extending a 4k model to 32k):

| Method | Fine-tuning needed | Quality retention |
|---|---|---|
| Naive extrapolation | — | ❌ catastrophic |
| Position interpolation | ~1000 steps | ✅ good |
| NTK-aware | 0 steps | ✅ good |
| **YaRN** | ~400 steps | ✅ **best** |

**LLaMA-3.1's approach** (a good concrete reference): base $\theta$ raised from 10,000 to
500,000, plus continued pretraining on long documents in stages — 8k → 32k → 128k. The staged
curriculum matters: jumping straight to the target length works much less well.

→ [Long context](../04-large-language-models/09-long-context.md) for the full treatment, including
what "128k context" actually delivers in practice.

---

## 8. Multi-dimensional positions

For images and video, position is not a scalar.

| Approach | Used by |
|---|---|
| Learned 2-D embeddings (or 1-D over flattened patches) | ViT |
| **Axial RoPE**: split $d$, apply RoPE on $x$ to one half and on $y$ to the other | modern vision/video models |
| 3-D RoPE (height, width, time) | video diffusion Transformers |
| Factorized: $PE(x,y) = PE_x(x) + PE_y(y)$ | many |

> [!TIP]
> Axial RoPE generalizes cleanly: allocate disjoint dimension groups to each axis, and the
> relative-position property holds independently per axis. A video model can then represent "two
> frames earlier and 30 pixels to the left" as a single relative offset.

---

## 9. Exercises

**Problem 1 — NTK-aware scaling, a 4× extension.** Using §7's formula
$\theta_{\text{new}} = \theta_{\text{old}}\cdot s^{d/(d-2)}$, compute the new RoPE base for
$d=128$, extending context by $s=4\times$, starting from $\theta_{\text{old}}=10000$. Compare
the exponent $d/(d-2)$ to a naive "$\theta_{\text{new}}=s\times\theta_{\text{old}}$" — is
NTK-aware scaling more or less aggressive than a plain linear scale-up?

<details markdown="1"><summary>Solution</summary>

$d/(d-2) = 128/126 = 1.0159$ — just barely above 1. So
$\theta_{\text{new}} = 10000\times4^{1.0159} = 10000\times4.089=40{,}890$.

A naive linear scale would give $\theta_{\text{new}}=4\times10000=40{,}000$. The NTK-aware value
(40,890) is only **slightly larger** than the naive linear one at this $d$ — the exponent
$d/(d-2)\to1$ as $d$ grows, so for large head dimensions the two approaches nearly coincide
numerically. The real difference NTK-aware scaling makes isn't primarily in this aggregate
number; it's in *which frequencies* get stretched (per §7: "stretches low frequencies a lot and
high frequencies almost not at all") — a property invisible from the single scalar $\theta$ and
only visible when you look at the per-dimension effect on $\theta_i=\theta^{-2i/d}$.

</details>

**Problem 2 — ALiBi slope, a different head.** A head has slope $m=0.125$ ($=2^{-3}$). Compute
its bias penalty at distances 1, 10, and 100. Roughly how many tokens back does this head's
"attention horizon" (per §5's framing of different heads having different effective ranges)
extend before the penalty exceeds, say, a typical attention-score range of $\pm10$?

<details markdown="1"><summary>Solution</summary>

Bias $=-m\times\text{dist}$: at dist 1, $-0.125$; at dist 10, $-1.25$; at dist 100, $-12.5$.

The penalty exceeds $10$ (in magnitude) somewhere between distance 80 ($-10.0$ exactly at
dist=80) — so this head's effective horizon is roughly **~80 tokens** before the linear penalty
overwhelms a typical-magnitude attention score and effectively zeroes out that position's
contribution after softmax. Per §5, different heads in the same layer use different slopes
($m_h=2^{-8h/H}$), so some heads (small $h$, larger $m$) have short horizons like this one, while
others (large $h$, tiny $m$) extend much further — giving the model, in aggregate, the
"built-in multi-scale view" §5 describes.

</details>

**Problem 3 — sinusoidal vs RoPE, the extrapolation gap explained.** §3 shows sinusoidal
encoding has the algebraic property that $PE_{pos+k}$ is a linear function of $PE_{pos}$ (so in
principle a model *could* learn relative attention from it), yet §3 also says it "extrapolates
poorly in practice," while §6 says RoPE, built on a similar rotation idea, extrapolates
better (with scaling). What's the actual mechanistic difference that explains this gap, given
both use rotation-like math?

<details markdown="1"><summary>Solution</summary>

The key difference is *where* the position information enters the computation, not the
underlying trigonometry. Sinusoidal encoding is **added once, to the input embeddings**, before
any Transformer layers — so the model must learn, indirectly through training, to *extract* the
relative-offset structure from an additive signal that's been mixed with content and then
transformed by every subsequent layer. Nothing forces the model to actually use the linear
relationship; it's merely *available* in principle.

RoPE instead **rotates $Q$ and $K$ directly inside every attention computation** (§6), so the
relative-position property $\langle R_mq,R_nk\rangle = q^\top R_{n-m}k$ is not something the
model has to discover — it's **structurally guaranteed by the attention operation itself**,
every layer, regardless of what the model has learned. A model trained at length 4096 has still
only ever seen $m\theta_i$ values up to a certain range for both encodings, so *some*
degradation is expected either way when extrapolating — but RoPE's guarantee that similarity
depends only on relative offset (not absolute position) is what makes the reweighting-based
extension methods in §7 (interpolation, NTK-aware scaling) even *meaningful* operations: you're
adjusting a quantity the model's forward pass structurally depends on, not hoping the model
generalizes an emergent pattern.

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Attention is permutation-equivariant — without positional information a Transformer is a bag of words. |
| 2 | The trend: absolute → relative, added-to-embeddings → applied-in-attention, learned → parameter-free. |
| 3 | Learned absolute encodings have a hard ceiling at $T_{\max}$ and zero extrapolation. |
| 4 | ALiBi adds a linear distance penalty per head — great extrapolation, but a soft window. |
| 5 | RoPE rotates $Q$ and $K$ by angle $\propto$ position; $\langle R_mq, R_nk\rangle$ depends only on $n-m$. |
| 6 | Apply RoPE to $Q$ and $K$ only, inside every attention layer — never to $V$, never to embeddings. |
| 7 | Context extension: PI squeezes uniformly, NTK-aware scales low frequencies more, YaRN combines both. |
| 8 | Raising RoPE's base $\theta$ (10k → 500k) plus staged long-context training is the standard recipe. |

---

## Further reading

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017), §3.5 — sinusoidal encoding.
- Shaw et al., [*Self-Attention with Relative Position Representations*](https://arxiv.org/abs/1803.02155) (2018).
- Press et al., [*Train Short, Test Long: Attention with Linear Biases*](https://arxiv.org/abs/2108.12409) (ALiBi, 2021).
- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021).
- Chen et al., [*Extending Context Window of LLMs via Position Interpolation*](https://arxiv.org/abs/2306.15595) (2023).
- Peng et al., [*YaRN: Efficient Context Window Extension of Large Language Models*](https://arxiv.org/abs/2309.00071) (2023).
- Kazemnejad et al., [*The Impact of Positional Encoding on Length Generalization*](https://arxiv.org/abs/2305.19466) (2023) — the NoPE result.

**Next** → Part IV: [LLM architecture](../04-large-language-models/01-llm-architecture.md)
