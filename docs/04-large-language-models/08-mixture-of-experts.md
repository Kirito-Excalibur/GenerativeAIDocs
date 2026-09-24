# Mixture of Experts

> **Summary** — Replace each MLP block with $E$ parallel "expert" MLPs and a router that sends each
> token to only the top-$k$ of them. Total parameters grow $E\times$ while compute per token grows
> only $k\times$ — you decouple *capacity* from *cost*. The price is memory (all experts must be
> resident), training instability (routing is discrete), and communication overhead in distributed
> settings. Most frontier models are now MoE.

**Prerequisites**: → [The Transformer](../03-sequence-models/04-transformer.md) · **Next**: → [Long context](09-long-context.md)

---

## 1. The idea

```
   DENSE BLOCK                          MoE BLOCK

   x ──► ┌────────────┐                 x ──► ┌────────┐
         │    MLP     │                       │ Router │──► gate scores
         │  d → 4d → d│                       └───┬────┘    over E experts
         └─────┬──────┘                           │
               ▼                       ┌──────────┼──────────┐
              out                      ▼          ▼          ▼
                                    ┌─────┐   ┌─────┐    ┌─────┐
   every token                      │ E₁  │   │ E₂  │ …  │ E₈  │
   uses every parameter             └──┬──┘   └──┬──┘    └─────┘
                                       │ g₁      │ g₂       (not selected:
                                       └────┬────┘           zero compute)
                                            ▼
                                     out = g₁E₁(x) + g₂E₂(x)
```

$$y = \sum_{i \in \text{TopK}(x)} g_i(x)\cdot E_i(x), \qquad
g(x) = \operatorname{softmax}\big(\text{TopK}(x W_r)\big)$$

> [!TIP]
> **The economics.** With $E = 8$ experts and $k = 2$ active:
> - Parameters: $8\times$ a dense block
> - FLOPs: $2\times$ a dense block
> - **You get 4× more capacity per unit of compute.**

Since loss follows a power law in parameters (→ [Scaling laws](03-scaling-laws.md)), more parameters
at the same compute means lower loss. That is the entire argument, and it holds empirically.

> [!WARNING]
> **Only the MLP is replaced**, not attention. MLPs hold ~2/3 of a Transformer's parameters
> (→ [Transformer §3](../03-sequence-models/04-transformer.md#3-parameter-counting-exactly)), so that
> is where the capacity is. Attention is shared across all tokens, which also keeps the routing
> problem tractable.

---

## 2. Active vs total parameters

That capacity-per-compute argument is about training FLOPs. It says nothing about the other resource a model needs — memory — and MoE's story there is the opposite of flattering.

The two numbers you must always distinguish:

| Model | Total params | Active/token | Experts | Top-$k$ |
|---|---|---|---|---|
| Switch-C | 1.6 T | — | 2048 | 1 |
| **Mixtral 8×7B** | **46.7 B** | **12.9 B** | 8 | 2 |
| Mixtral 8×22B | 141 B | 39 B | 8 | 2 |
| DeepSeek-V2 | 236 B | 21 B | 160 + 2 shared | 6 |
| DeepSeek-V3 | 671 B | 37 B | 256 + 1 shared | 8 |
| Qwen1.5-MoE-A2.7B | 14.3 B | 2.7 B | 60 + 4 shared | 4 |

**Why "8×7B" is 46.7 B, not 56 B.** Only the MLPs are replicated. For Mixtral:

| Component | Params |
|---|---|
| Attention + embeddings + norms (shared) | ~1.3 B |
| MLP per layer per expert | $3 \times 4096 \times 14336 = 176$ M |
| × 32 layers × 8 experts | 45.1 B |
| **Total** | **46.4 B** ≈ 46.7 B ✓ |

**And active parameters**: shared (1.3 B) + 2 experts × 32 layers × 176 M = 1.3 + 11.3 = **12.6 B**
≈ 12.9 B ✓

> [!WARNING]
> **The serving consequence.** Mixtral 8×7B needs **93 GB** in BF16 (all experts resident) but
> computes like a 13B model. It is *cheap to run at scale* and *expensive to fit on one GPU*. MoE is
> a datacenter architecture, not a laptop one.

```
     COMPUTE             MEMORY
   ┌──────────┐      ┌──────────────────────────────┐
   │  13 B    │      │           47 B               │
   │  worth   │      │  all experts must be loaded  │
   └──────────┘      └──────────────────────────────┘
   MoE wins here      MoE loses here
```

---

## 3. Load balancing: the core difficulty

That memory bill assumes every expert actually gets used. Getting the router to spread tokens evenly across experts — rather than collapsing onto a favorite few — turns out to be the hardest part of making MoE work at all.

> [!WARNING]
> **The failure mode**: routing is learned, and it has a self-reinforcing bias. An expert that is
> chosen slightly more often gets more gradient, becomes better, and is chosen even more often.
> Left alone, a handful of experts absorb everything and the rest are dead weight.

```
  Without balancing                 With an auxiliary loss
  
  E₁ ████████████████ 62%           E₁ ███ 13%
  E₂ ██████ 21%                     E₂ ███ 12%
  E₃ ███ 11%                        E₃ ███ 13%
  E₄ ▌ 3%                           E₄ ███ 12%
  E₅ ▏ 1%                           E₅ ███ 13%
  E₆ ▏ 1%                           E₆ ███ 12%
  E₇  0%   ← dead                   E₇ ███ 13%
  E₈  0%   ← dead                   E₈ ███ 12%
  ⇒ effectively a dense 2-expert     ⇒ full capacity used
    model with wasted parameters
```

### The auxiliary loss

$$\mathcal{L}_{\text{aux}} = \lambda \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ = fraction of tokens routed to expert $i$ (a count), and $P_i$ = mean router
probability for expert $i$ (differentiable).

> [!TIP]
> **Why the product $f_i P_i$?** $f_i$ comes from a hard top-$k$ selection and has no gradient.
> $P_i$ is differentiable but doesn't reflect actual assignment. Multiplying them gives a
> differentiable surrogate: the loss is minimized when both are uniform ($f_i = P_i = 1/E$), giving
> $\mathcal{L}_{\text{aux}} = \lambda$. Any imbalance raises it.

Typical $\lambda = 0.01$. Too high and routing becomes random (destroying specialization); too
low and experts collapse.

### Capacity factor and token dropping

Each expert gets a fixed buffer:

$$\text{capacity} = \text{CF}\times\frac{\text{tokens per batch}\times k}{E}$$

Tokens beyond capacity are **dropped** — they skip the MLP entirely and pass through on the
residual stream only.

With CF = 1.25, 8 experts, 4096 tokens, $k=2$: capacity $= 1.25\times\frac{4096\times2}{8} = 1280$
tokens per expert. A perfectly balanced batch sends 1024 per expert, so there's 25% headroom.

> [!WARNING]
> Token dropping is a real quality cost, and it makes the model's output depend on what *else* is
> in the batch — which breaks reproducibility. Trade-off: higher CF = less dropping, more wasted
> compute and memory.

### Loss-free balancing

DeepSeek-V3 introduced an alternative: add a **learnable per-expert bias** to the routing logits,
adjusted after each step (increase the bias of under-used experts, decrease the over-used ones).
The bias affects *selection* but not the *gating weights*, so it balances load without adding a
gradient term that fights the language-modelling objective.

> [!TIP]
> This is a nice example of solving a problem with a control loop instead of a loss term. The
> auxiliary loss always trades some quality for balance; the bias approach doesn't.

---

## 4. Routing variants

Balancing load is about making sure the router doesn't collapse. A separate design question is what the router and its experts actually look like architecturally — how many experts, how many chosen per token, and whether routing is even a hard, discrete choice at all.

| Scheme | How it works | Notes |
|---|---|---|
| **Top-$k$ token choice** | each token picks its top $k$ experts | standard; needs balancing |
| **Expert choice** | each *expert* picks its top-$c$ tokens | ✅ perfect balance by construction; ⚠️ some tokens get 0 experts, and it leaks future information in causal settings |
| Top-1 (Switch) | $k=1$ | cheapest; Switch showed it works |
| **Shared experts** | 1–2 experts always active, plus routed ones | ✅ captures common patterns; avoids every expert relearning basics |
| Fine-grained | many small experts (256) instead of few big ones | ✅ more combinations, better specialization |
| Soft MoE | weighted mixture of *all* experts, no hard routing | fully differentiable; used in vision |
| Hash routing | fixed hash of the token id | no learned router, surprisingly decent baseline |

> [!TIP]
> **DeepSeek's combination — fine-grained + shared experts — is the current best practice.**
> Splitting into many small experts gives $\binom{256}{8}$ possible combinations rather than
> $\binom{8}{2} = 28$, so routing can express far more specialization. Meanwhile 1 shared expert
> handles the generic transformations every token needs, freeing routed experts to specialize
> genuinely.

---

## 5. What do experts specialize in?

Fine-grained routing is motivated by the idea that more, smaller experts let the model specialize more precisely. Whether that specialization looks anything like what you'd intuitively expect — a "code expert," a "poetry expert" — turns out to have a surprising answer.

**Not what you'd expect.** Mixtral's authors looked for topic specialization (one expert for
biology, one for code) and **found almost none**. What they found instead:

- **Syntactic/positional patterns** — experts specialize on token type and local structure.
- **Strong consecutive-token locality** — the same expert handles adjacent tokens far more often
  than chance.
- **Some domain signal in code and math**, but weaker than expected.

> [!TIP]
> **Why the "topic expert" intuition is wrong.** Routing happens at *every layer* for *every
> token*. A token's path through the network is a sequence of $L$ expert choices, so specialization
> is distributed and compositional rather than assigned per-domain. Experts are more like reusable
> sub-operations than subject-matter specialists.

Fine-grained MoE models with more experts do show somewhat clearer specialization, which is one
argument for that design.

---

## 6. Systems: expert parallelism

Specialization is a modeling question. Getting $E$ experts, scattered across many GPUs, to actually receive the right tokens fast enough is a completely separate, and often harder, systems question.

At scale, experts live on different devices. Every MoE layer becomes an **all-to-all** communication.

```
   GPU 0        GPU 1        GPU 2        GPU 3
  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │ E₁E₂ │     │ E₃E₄ │     │ E₅E₆ │     │ E₇E₈ │
  └──────┘     └──────┘     └──────┘     └──────┘
      ▲▼           ▲▼           ▲▼           ▲▼
      └────────────┴─── ALL-TO-ALL ──┴───────┘
                        (twice per MoE layer:
                         dispatch tokens, combine results)
```

> [!WARNING]
> **This is the main systems cost of MoE**, and it can easily exceed the compute savings on a
> poorly connected cluster. Mitigations:

| Technique | Effect |
|---|---|
| Keep expert parallelism within a node (NVLink) | avoids slow inter-node all-to-all |
| Overlap communication with computation | hide the latency behind the attention block |
| Limit the number of devices a token can reach | DeepSeek caps tokens to $M$ nodes |
| Capacity-limited dispatch | bounds the message size, making it predictable |

**For inference**, MoE has a different problem: batching. In a dense model, a batch of 64 tokens
all use the same weights. In an MoE, they scatter across experts, so each expert gets a small
batch — worse GPU utilization. MoE inference therefore wants *very* large batches to keep the
per-expert matmuls efficient.

---

## 7. Implementation

Every mechanism above — routing, balancing, expert parallelism — is describable in words and diagrams, but a router is, at the end of the day, just one more small network making a discrete choice. Seeing it in code is the fastest way to see exactly what "top-$k$ of $E$ experts" means in practice.

A complete, correct MoE layer:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, top_k=2, n_shared=0, aux_loss_coef=0.01):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.aux_loss_coef = aux_loss_coef
        self.router  = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, d_ff) for _ in range(n_experts))
        # shared experts (DeepSeek-style) always run, for every token
        self.shared = nn.ModuleList(SwiGLU(d, d_ff) for _ in range(n_shared))

    def forward(self, x):
        B, T, d = x.shape
        x_flat = x.view(-1, d)                          # (N, d) with N = B*T

        logits = self.router(x_flat)                    # (N, E)
        probs  = logits.softmax(dim=-1)
        topk_p, topk_i = probs.topk(self.top_k, dim=-1) # (N, k)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)   # renormalize over chosen

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            # which (token, slot) pairs chose this expert?
            tok_idx, slot_idx = (topk_i == e).nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            # run the expert ONCE on its assigned tokens, not once per token
            expert_out = self.experts[e](x_flat[tok_idx])
            out.index_add_(0, tok_idx, expert_out * topk_p[tok_idx, slot_idx, None])

        for shared in self.shared:
            out = out + shared(x_flat)

        # ---- auxiliary load-balancing loss ----
        # f_i: fraction of token-slots routed to expert i (non-differentiable count)
        # P_i: mean router probability for expert i (differentiable)
        with torch.no_grad():
            one_hot = F.one_hot(topk_i, self.n_experts).float().sum(dim=1)   # (N, E)
            f = one_hot.mean(dim=0)
        P = probs.mean(dim=0)
        self.aux_loss = self.aux_loss_coef * self.n_experts * (f * P).sum()

        return out.view(B, T, d)
```

> [!WARNING]
> **The critical detail is the loop over experts, not tokens.** Gathering each expert's tokens and
> running one batched matmul is what makes MoE fast. A naive per-token loop is catastrophically slow.
> Production kernels (MegaBlocks, grouped GEMM) go further, expressing the whole layer as a single
> block-sparse matmul.

> [!WARNING]
> **Don't forget to add `aux_loss` to your training loss.** Summing it across all MoE layers:

```python
total_loss = ce_loss + sum(m.aux_loss for m in model.modules() if isinstance(m, MoELayer))
```

---

## 8. Trade-offs, honestly

| ✅ | ❌ |
|---|---|
| Better loss per training FLOP | All experts must be in memory — poor fit for single-GPU serving |
| Faster training to a given loss | All-to-all communication overhead |
| Cheaper inference per token at scale | Training instability (routing is discrete, gradients are noisy) |
| Natural fit for multi-node parallelism | Extra hyperparameters: $E$, $k$, CF, $\lambda$ |
| Scales total capacity without scaling FLOPs | Token dropping makes outputs batch-dependent |
| | Fine-tuning MoE is harder (experts overfit unevenly) |
| | Poor small-batch inference efficiency |

> [!TIP]
> **When MoE is right**: you are compute-constrained in training, serve at high throughput, and
> have a well-connected multi-GPU cluster. **When it isn't**: single-GPU deployment, latency-critical
> small-batch serving, or memory-constrained environments. For local inference, a dense model of the
> same *active* size is usually the better choice.

---

## 9. Exercises

**Problem 1 — total vs active, a 16-expert model.** Using §2's Mixtral-style method
($d{=}4096$, $d_{ff}{=}14336$, $L{=}32$, shared params $\approx1.3$B), compute total and active
parameters for $E{=}16$ experts, top-$k{=}4$. What's the total/active ratio, and how does it
compare with $E/k$ (the naive ratio you'd expect if the shared components were negligible)?

<details markdown="1"><summary>Solution</summary>

Per-expert-per-layer MLP: $3\times4096\times14336=176.2$M (identical to §2's Mixtral figure,
since $d,d_{ff}$ are unchanged — only $E,k$ differ here).

Total: $1.3\text{B} + 32\times16\times176.2\text{M} = 1.3\text{B}+90.2\text{B}=91.5$B.

Active: $1.3\text{B}+32\times4\times176.2\text{M}=1.3\text{B}+22.6\text{B}=23.8$B.

Ratio: $91.5/23.8=3.84\times$. Naive $E/k=16/4=4.0\times$. The actual ratio ($3.84$) is *slightly
below* the naive $E/k$ ratio because the shared components (1.3B, identical in both total and
active) don't get multiplied by $E$ or $k$ — they dilute the ratio slightly toward 1. This
matches §2's own Mixtral example: real ratio $46.7/12.9=3.62$ vs naive $E/k=8/2=4.0$ — same
direction and similar-sized gap.

</details>

**Problem 2 — load balancing, a different capacity factor.** A batch has 8192 tokens routed
through $E{=}8$ experts at top-$k{=}2$. Using §3's capacity formula, compute the per-expert
capacity at CF$=1.5$, and say how many tokens per expert a *perfectly balanced* batch would send.
How much headroom (in tokens) does CF$=1.5$ give over perfect balance?

<details markdown="1"><summary>Solution</summary>

Capacity $= \text{CF}\times\frac{\text{tokens}\times k}{E} = 1.5\times\frac{8192\times2}{8}
= 1.5\times2048=3072$ tokens per expert.

Perfectly balanced: $8192\times2/8=2048$ tokens per expert (exactly the un-scaled term).

Headroom: $3072-2048=1024$ tokens, i.e. **50% above perfect balance** — matching CF's definition
directly (CF$=1.5$ means "50% more capacity than the balanced case"). This is more generous than
§3's own worked example (CF$=1.25$, giving 25% headroom), trading more wasted compute/memory
(unused capacity slots) for a lower chance of token dropping under imbalanced routing.

</details>

**Problem 3 — MoE inference batching, reasoned.** Using §6's discussion of poor small-batch MoE
efficiency, explain why a chatbot serving *one user at a time* would get worse GPU utilization
from an MoE model than from a dense model of the same *active* parameter count, even though both
require exactly the same FLOPs per token.

<details markdown="1"><summary>Solution</summary>

For a dense model, every token in a batch uses the *same* weights, so a batch of (say) 1 token
still runs one clean, reasonably-shaped matmul against the full weight matrices — GPU utilization
is limited mainly by the usual memory-bandwidth-bound decode problem (→ [LLM architecture
§4](01-llm-architecture.md#4-request-lifecycle-prefill-and-decode)), not by anything MoE-specific.

For an MoE model serving one user, each of the few tokens being decoded independently selects its
own top-$k$ experts via the router — with only 1 (or a handful of) tokens in flight, the tokens
routed to any *particular* expert form a batch of size 0 or 1 for that expert's matmul. Per §6,
"each expert gets a small batch — worse GPU utilization" — you've lost the ability to batch many
tokens against the same expert's weights, which is exactly the mechanism that makes matmuls
efficient on GPU hardware. The FLOPs-per-token being identical to a dense model doesn't help,
because the bottleneck here isn't FLOPs — it's how efficiently those FLOPs can be scheduled onto
the hardware, and MoE's fragmentation across experts hurts that efficiency precisely when batch
size is small, which is exactly the single-user chatbot case.

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | MoE replaces the MLP with $E$ experts + a router; top-$k$ active gives $E\times$ capacity at $k\times$ compute. |
| 2 | Always distinguish **total** from **active** parameters. Mixtral 8×7B = 46.7 B total, 12.9 B active. |
| 3 | You save compute, not memory. MoE is a datacenter architecture. |
| 4 | Routing collapses without load balancing — auxiliary loss ($\lambda \approx 0.01$) or a learned routing bias. |
| 5 | Capacity factor bounds per-expert buffers; overflow tokens are dropped, making output batch-dependent. |
| 6 | Experts specialize by syntax and position, **not** by topic. The "biology expert" intuition is wrong. |
| 7 | Fine-grained experts + a shared always-on expert is current best practice. |
| 8 | All-to-all communication is the main systems cost; keep expert parallelism within a node. |
| 9 | Implement by looping over experts (batched), never over tokens. |

---

## Further reading

- Shazeer et al., [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538) (2017).
- Fedus et al., [*Switch Transformers*](https://arxiv.org/abs/2101.03961) (2021) — top-1 routing at scale.
- Jiang et al., [*Mixtral of Experts*](https://arxiv.org/abs/2401.04088) (2024) — including the expert-specialization analysis.
- DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) (2024) — fine-grained experts, shared experts, loss-free balancing.
- Zhou et al., [*Mixture-of-Experts with Expert Choice Routing*](https://arxiv.org/abs/2202.09368) (2022).
- Gale et al., [*MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*](https://arxiv.org/abs/2211.15841) (2022).

**Next** → [Long context](09-long-context.md)
