# Autoregressive Models

> **Summary** — The chain rule of probability, turned into an architecture. Autoregressive models
> factorize $p(x) = \prod_i p(x_i \mid x_{<i})$ and learn each conditional with a shared network.
> Exact likelihood, stable training, parallel training via teacher forcing, and inherently serial
> sampling. Every LLM is an autoregressive model.

**Prerequisites**: → [Taxonomy](../01-foundations/06-taxonomy.md) · **Next**: → [VAE](02-vae.md)

---

## 1. The idea, in one equation

$$p_\theta(x_1,\dots,x_n) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1,\dots,x_{i-1})$$

This is **exact** — the chain rule is an identity, not an approximation. The modelling choice is
only in how you parameterize each conditional.

> [!TIP]
> **Intuition** — you have replaced one impossible problem (a distribution over $|\mathcal{V}|^n$
> sequences) with $n$ tractable problems (a distribution over $|\mathcal{V}|$ next tokens). It is the
> same move as writing a long number in decimal: rather than naming one of $10^{20}$ integers, you
> name 20 digits in order.

The cost is that you must pick an **order**. For text the order is obvious. For images it is not —
raster scan is arbitrary, and the arbitrariness hurts.

**Concretely** — vocabulary 50,000, sequence length 1024:
- Joint distribution table: $50000^{1024} \approx 10^{4800}$ entries. Impossible.
- AR factorization: 1024 softmaxes over 50,000 options each. A single forward pass of a network.

---

## 2. Training: teacher forcing

That factorization tells you what to compute at inference. It doesn't yet tell you how to train $n$ conditionals at once without waiting for the model to generate each token in sequence — which is what teacher forcing solves.

The crucial engineering fact: **you can compute all $n$ conditionals in one parallel forward pass**
if you (a) feed the *ground truth* prefix rather than the model's own outputs, and (b) mask the
attention/convolution so position $i$ cannot see positions $\ge i$.

$$\mathcal{L} = -\sum_{i=1}^{n}\log p_\theta(x_i \mid x_{<i}) \quad\text{(cross-entropy, summed over positions)}$$

```
 TRAINING (parallel, teacher forcing)         INFERENCE (serial)

 input:   <s>  The  cat  sat  on             step 1: <s>            → "The"
           │    │    │    │    │             step 2: <s> The        → "cat"
        ┌──▼────▼────▼────▼────▼──┐          step 3: <s> The cat    → "sat"
        │   causal-masked model   │          step 4: <s> The cat sat→ "on"
        └──┬────┬────┬────┬────┬──┘                  ▲          │
           │    │    │    │    │                     └──────────┘
 predict: The  cat  sat  on   .                    feed own output back

 ALL positions computed in ONE pass          n passes, cannot parallelize
 loss = mean of n cross-entropies            (this is the latency problem)
```

> [!WARNING]
> **Exposure bias** — at training time the model always sees a *correct* prefix; at inference it
> sees its own, possibly flawed, output. Errors compound: one bad token shifts the distribution into
> a region the model never trained on. This is the theoretical explanation for degeneration
> (repetition loops, drift in long generations).

In practice the effect is smaller than the theory suggests for well-trained large models, and the
main mitigations are decoding-side (nucleus sampling, repetition penalties) rather than
training-side. Scheduled sampling — feeding the model's own predictions during training — was
proposed to fix it but introduces a biased gradient and is rarely used for LLMs.
→ [Inference & decoding](../04-large-language-models/06-inference-and-decoding.md)

---

## 3. A complete, worked micro-example

That's the training procedure in the abstract. Running it by hand on a tiny vocabulary, end to end from probabilities to a loss number, makes every piece of it concrete.

Let's actually build a bigram model on a tiny corpus, by hand.

**Corpus**: `"the cat sat on the mat the cat ate"`
**Vocabulary**: `{the, cat, sat, on, mat, ate}` → 6 tokens.

**Count the bigrams:**

| context | next | count |
|---|---|---|
| the | cat | 2 |
| the | mat | 1 |
| cat | sat | 1 |
| cat | ate | 1 |
| sat | on | 1 |
| on | the | 1 |
| mat | the | 1 |

**Maximum-likelihood conditionals** — $p(w' \mid w) = \frac{C(w,w')}{C(w)}$:

$$p(\text{cat}\mid\text{the}) = \tfrac{2}{3} = 0.667, \quad p(\text{mat}\mid\text{the}) = \tfrac13 = 0.333$$
$$p(\text{sat}\mid\text{cat}) = p(\text{ate}\mid\text{cat}) = 0.5$$

**Score the sequence** `"the cat sat"`:

$$p(\text{the cat sat}) = p(\text{the}) \cdot p(\text{cat}\mid\text{the})\cdot p(\text{sat}\mid\text{cat})
= 0.333 \times 0.667 \times 0.5 = 0.111$$

Per-token cross-entropy: $-\frac13(\ln 0.333 + \ln 0.667 + \ln 0.5) = \frac13(1.100+0.405+0.693) = 0.733$ nats.
Perplexity $= e^{0.733} = 2.08$ — the model is about as confused as a fair coin flip per token.

> [!WARNING]
> **Why n-grams fail, quantitatively** — a 5-gram model over a 50k vocabulary has
> $50000^5 = 3\times10^{23}$ parameters. Essentially every context is unseen; you need smoothing
> (Kneser-Ney) just to avoid assigning zero probability. **Neural models fix this by sharing
> parameters across contexts**: a learned embedding means "cat" and "dog" produce similar
> predictions even if "the dog sat" never appeared in training. That generalization-by-sharing is
> the entire contribution of neural language modelling.

---

## 4. The architecture progression

Sharing parameters across contexts is *why* neural models work at all. Which network computes $p(x_i \mid x_{<i})$ — and how that architecture evolved from n-grams to RNNs to Transformers — is a separate question of *how well* and *how efficiently* it works.

| Year | Model | Context handling | Sampling cost | Key limitation |
|---|---|---|---|---|
| 1990s | n-gram | fixed window, counts | $O(n)$ lookup | no generalization, exponential table |
| 2003 | Bengio NNLM | fixed window, embeddings | $O(n)$ | fixed context |
| 2010 | RNN-LM | unbounded, recurrent | $O(n)$ | vanishing gradients, serial training |
| 2014 | LSTM-LM | gated recurrence | $O(n)$ | still serial training |
| 2016 | WaveNet | dilated causal conv | $O(n)$ | receptive field fixed by depth |
| 2016 | PixelCNN | masked 2-D conv | $O(HW)$ | blind spot; slow on big images |
| 2018+ | **Transformer LM** | full attention | $O(n)$ w/ KV cache | $O(n^2)$ attention |

### Causal masking: the three ways

```
 MASKED CONVOLUTION            DILATED CAUSAL CONV          MASKED SELF-ATTENTION
 (PixelCNN)                    (WaveNet)                    (Transformer)

   ┌───┬───┬───┐                 out ●                          j→ 1  2  3  4
   │ 1 │ 1 │ 1 │                    ╱│╲                      i  ┌──┬──┬──┬──┐
   ├───┼───┼───┤                   ╱ │ ╲   dilation 4        ↓ 1│ ✓│ ✗│ ✗│ ✗│
   │ 1 │ ✗ │ 0 │ ← current         ●  ●  ●                     2│ ✓│ ✓│ ✗│ ✗│
   ├───┼───┼───┤   pixel          ╱│╲ │ ╱│╲ dilation 2         3│ ✓│ ✓│ ✓│ ✗│
   │ 0 │ 0 │ 0 │                 ● ● ● ● ● ●                   4│ ✓│ ✓│ ✓│ ✓│
   └───┴───┴───┘                 │╲│╱│╲│╱│╲│  dilation 1       └──┴──┴──┴──┘
                                 ●●●●●●●●●●●
 zero out "future"               receptive field 2^L        set scores to −∞
 kernel weights                   with L layers              before softmax
```

**Causal attention mask** — the three lines that make a Transformer autoregressive:

```python
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))      # lower triangular
scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)       # (B, H, T, T)
scores = scores.masked_fill(~mask, float('-inf'))          # future → -inf
attn = scores.softmax(dim=-1)                              # -inf → 0 weight
```

> [!WARNING]
> **The off-by-one that breaks everything** — the model must predict token $i{+}1$ from tokens
> $\le i$. Get the shift wrong and the model can see the answer; your loss drops to near-zero and
> your generations are garbage. The canonical form:

```python
logits = model(tokens[:, :-1])          # inputs:  positions 0 .. T-2
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1)           # targets: positions 1 .. T-1
)
```

**Sanity check for any new AR implementation**: train on 100 copies of one sequence. Loss should
approach 0 and generation should reproduce it exactly. If loss hits exactly 0 within a few steps,
you have information leakage.

---

## 5. PixelCNN and WaveNet: AR beyond text

Text has an obvious left-to-right order, which is most of why Transformers dominate it. Other modalities don't have a natural order handed to them, and forcing autoregression onto them anyway is where the framework gets more interesting.

### PixelCNN: images as a raster scan

$$p(x) = \prod_{i=1}^{H\cdot W} p(x_i \mid x_{<i})$$

with the pixels ordered row by row, left to right, and each pixel's three colour channels ordered
R→G→B.

**Why it lost to diffusion:**

| Issue | Detail |
|---|---|
| Arbitrary order | Raster scan says the pixel above-left matters more than the one below-right. False. |
| Sampling cost | A $256\times256$ RGB image is 196,608 sequential forward passes. Minutes per image. |
| Long-range structure | Global coherence must travel through thousands of local steps. |
| The "blind spot" | Naive masked convolutions leave a wedge of context unreachable; Gated PixelCNN needed two separate stacks (vertical + horizontal) to fix it. |

> [!TIP]
> But the *idea* survived in a better form: **tokenize the image first**. VQ-VAE compresses a
> $256\times256$ image to a $32\times32$ grid of discrete codes — 1024 tokens instead of 196,608.
> Then run a Transformer over those. This is VQGAN/DALL·E-1, and it is still how most video and
> audio generation handles discreteness.
> → [VAE § VQ-VAE](02-vae.md#7-vq-vae-discrete-latents)

### WaveNet: raw audio

Dilated causal convolutions: layer $l$ has dilation $2^l$, so $L$ layers give a receptive field of
$2^L$ samples with only $O(L)$ depth.

30 layers → receptive field of $2^{30}$? No — stacked in blocks of 10 (dilations 1,2,…,512),
repeated 3×, giving a receptive field of $3 \times 1023 \approx 3000$ samples ≈ 190 ms at 16 kHz.

The killer statistic: generating 1 second of 16 kHz audio requires **16,000 sequential forward
passes**. Original WaveNet took minutes to generate seconds of speech. Parallel WaveNet solved this
by distilling into an inverse-autoregressive flow — an early, important example of
*distillation to break the sequential bottleneck*, the same idea that now powers few-step
diffusion.

---

## 6. Complexity analysis

Whether it's text, pixels or audio samples, autoregressive generation pays the same structural cost: one forward pass per output, in sequence. Making that cost precise — and finding the trick that tames it — is the point of the next two sections.

| Quantity | Transformer AR | Notes |
|---|---|---|
| Training time | $O(n^2 d + nd^2)$ per sequence | one parallel pass |
| Training memory | $O(n^2 H + ndL)$ | attention matrix dominates for long $n$ |
| Sampling, naive | $O(n^3 d)$ | recompute everything each step — never do this |
| **Sampling, KV cache** | $O(n^2 d + nd^2)$ | amortized $O(nd + d^2)$ per token |
| KV cache memory | $2 L H_{kv} d_h n \cdot$ bytes | the real inference constraint |

**KV cache for a 70B model** ($L=80$, $H_{kv}=8$ with GQA, $d_h=128$, BF16), 8192 tokens:

$$2 \times 80 \times 8 \times 128 \times 8192 \times 2\text{ bytes} = 2.68 \text{ GB per sequence}$$

Without GQA ($H_{kv} = 64$): **21.5 GB per sequence.** This single calculation is why grouped-query
attention was adopted universally.
→ [Inference & decoding](../04-large-language-models/06-inference-and-decoding.md)

---

## 7. Minimal implementation

That 21.5 GB number is an argument on paper. Seeing the same mechanism — one token in, one token out, feeding back into itself — in fewer than 20 lines of working code makes it concrete.

A complete character-level autoregressive Transformer. This is the whole idea in ~60 lines.

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head, self.d_head = n_head, d // n_head
        self.qkv  = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        # (B, T, d) -> (B, n_head, T, d_head)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))
        # is_causal=True applies the triangular mask inside the fused kernel
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.mlp  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):                 # pre-norm: norm inside the residual branch
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, d=256, n_layer=6, n_head=8, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight        # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature        # only the last position matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
```

Trained on ~1 MB of Shakespeare (~40 min on a laptop GPU), this produces syntactically
plausible pseudo-Shakespeare. The architecture is the same one used at 10,000× the scale; only
the numbers change.

> [!WARNING]
> Note `generate` above recomputes the full forward pass every step — $O(n^2)$ total work. A real
> implementation caches K and V. → [Inference & decoding](../04-large-language-models/06-inference-and-decoding.md)

---

## 8. Strengths and weaknesses

That toy model already exhibits every structural property of a frontier LLM: exact likelihood, stable training, and strictly sequential sampling. Whether those properties are a good trade depends on what you're building, which is the honest summary this page closes with.

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Exact likelihood — clean, unambiguous training signal | Sampling is inherently serial: $n$ forward passes |
| Training is stable (plain cross-entropy, no min-max, no bound) | Requires a canonical ordering |
| Parallel training via teacher forcing | Exposure bias between train and inference |
| No architectural constraints beyond causality | Quadratic attention cost limits context |
| Scales predictably — → [Scaling laws](../04-large-language-models/03-scaling-laws.md) | No learned latent space for editing or interpolation |
| Trivially handles variable-length data | Cannot revise an earlier token once emitted |

> [!TIP]
> **On "cannot revise"** — an AR model commits to token $i$ before seeing token $i{+}1$. Humans
> draft and revise. This is the structural argument for diffusion language models and for
> **chain-of-thought**: CoT lets the model use extra tokens as a scratchpad, effectively buying
> revision at the cost of sequence length.
> → [Reasoning](../04-large-language-models/10-reasoning.md)

---

## 9. Exercises

**Problem 1 — score a different sentence.** Using the bigram counts from §3's worked example
($p(\text{cat}|\text{the})=2/3$, $p(\text{mat}|\text{the})=1/3$, $p(\text{sat}|\text{cat})=0.5$,
$p(\text{ate}|\text{cat})=0.5$), score the sentence "the cat ate" the same way §3 scored "the cat
sat": compute $p(\text{the cat ate})$, the per-token cross-entropy, and the perplexity.

<details markdown="1"><summary>Solution</summary>

$$p(\text{the cat ate}) = p(\text{the})\cdot p(\text{cat}|\text{the})\cdot p(\text{ate}|\text{cat})
= \tfrac13\times\tfrac23\times\tfrac12 = 0.111$$

(Identical to $p(\text{the cat sat})$ in §3, since $p(\text{sat}|\text{cat})=p(\text{ate}|\text{cat})=0.5$
— the model is exactly as confident about either continuation.)

Cross-entropy $=-\frac13(\ln\frac13+\ln\frac23+\ln\frac12)=0.732$ nats; perplexity
$=e^{0.732}=2.08$ — matching §3's numbers exactly, which makes sense since the two sentences have
identical per-step probabilities.

</details>

**Problem 2 — off-by-one, spot the bug.** A colleague writes:

```python
logits = model(tokens)                    # full sequence, no shift
loss = F.cross_entropy(logits.view(-1, V), tokens.view(-1))
```

Using §4's discussion, what's wrong, and what will you observe if you train with this code?

<details markdown="1"><summary>Solution</summary>

This computes the loss for predicting token $i$ from a forward pass that **already includes**
token $i$ in the input (since `tokens` is fed whole, and position $i$'s output is compared
against `tokens[i]` itself, not `tokens[i+1]`). Under causal masking, position $i$'s hidden state
already attends to tokens $\le i$ — including $i$ — so the model is being asked to predict the
token it can already see.

Per §4's warning, this is "information leakage": the loss will collapse to near-zero within a
few steps (the model trivially learns "copy the current input"), and any text the model generates
will be garbage, because at real inference time (autoregressive generation) it has no future
token to copy. Correct form:
`logits = model(tokens[:, :-1]); loss = F.cross_entropy(logits.view(-1,V), tokens[:, 1:].reshape(-1))`.

</details>

**Problem 3 — KV cache sizing.** Using the formula from §6, compute the KV-cache memory (in GB)
for a 13B-parameter model with $L=40$, $H_{kv}=40$ (no GQA — full multi-head), $d_h=128$, at
context length $T=4096$, batch size 1, BF16 (2 bytes/element). Then recompute with GQA at
$H_{kv}=8$. What's the memory reduction factor, and does it match the ratio $H_{kv,\text{full}}/H_{kv,\text{GQA}}$?

<details markdown="1"><summary>Solution</summary>

Formula: $2\times L\times H_{kv}\times d_h\times T\times B\times\text{bytes}$.

No GQA: $2\times40\times40\times128\times4096\times1\times2 = 3{,}355{,}443{,}200$ bytes
$\approx 3.36$ GB.

With GQA ($H_{kv}{=}8$): $2\times40\times8\times128\times4096\times1\times2 = 671{,}088{,}640$
bytes $\approx 0.671$ GB.

Reduction factor: $3.36/0.671 = 5.0\times$ — and indeed $H_{kv,\text{full}}/H_{kv,\text{GQA}} =
40/8 = 5$. The formula is linear in $H_{kv}$, so the memory reduction from GQA is *exactly* the
ratio of query heads to KV heads, with everything else held fixed — confirming the general claim
in → [Attention §7](../03-sequence-models/03-attention.md#7-mqa-and-gqa-shrinking-the-kv-cache).

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | $p(x)=\prod_i p(x_i\mid x_{<i})$ is exact; the only modelling choice is the parameterization. |
| 2 | Teacher forcing + causal masking makes training fully parallel; sampling stays serial. |
| 3 | n-grams fail because contexts are unseen; neural models fix it by sharing parameters via embeddings. |
| 4 | Ordering is a real cost — it is why AR lost images to diffusion. |
| 5 | The AR pattern survives for continuous data by tokenizing first (VQ-VAE → Transformer). |
| 6 | The KV cache converts $O(n^3)$ naive sampling into $O(n^2)$; its *memory* is the inference bottleneck. |
| 7 | Watch the off-by-one: inputs `[:-1]`, targets `[1:]`. Sanity-check by memorizing one sequence. |

---

## Further reading

- Bengio et al., *A Neural Probabilistic Language Model* (2003) — where embeddings for LM began.
- van den Oord et al., [*Pixel Recurrent Neural Networks*](https://arxiv.org/abs/1601.06759) (2016) and [*WaveNet*](https://arxiv.org/abs/1609.03499) (2016).
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018).
- Karpathy, [*nanoGPT*](https://github.com/karpathy/nanoGPT) — the clearest readable implementation in existence.

**Next** → [Variational autoencoders](02-vae.md)
