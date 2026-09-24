# Inference & Decoding

> **Summary** — Having a probability distribution over the next token is not the same as having
> text. Decoding strategy — greedy, beam, temperature, top-$k$, top-$p$, min-$p$ — determines
> output quality as much as the model does. This page explains why maximizing likelihood produces
> *worse* text than sampling, covers the KV cache and PagedAttention, and works through
> speculative decoding, which gives 2–3× speedup with mathematically identical output.

**Prerequisites**: → [LLM architecture](01-llm-architecture.md) · **Next**: → [Efficiency](07-efficiency.md)

---

## 1. The decoding problem

At each step the model gives you $p_\theta(x_t \mid x_{<t})$ over ~100,000 tokens. You must pick
one. The obvious choice — pick the most likely — is **wrong**, and understanding why is the key
insight of this page.

```
   logits ──► softmax ──► distribution ──► [DECODING STRATEGY] ──► token
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   DETERMINISTIC         STOCHASTIC          SEARCH-BASED
   greedy                temperature         beam search
                         top-k               best-of-n
                         top-p (nucleus)     MBR decoding
                         min-p
```

---

## 2. The likelihood trap

⚠️ **Greedy and beam search produce degenerate text.** This is not a subtle effect:

```
  PROMPT: "In a shocking finding, scientists discovered a herd of unicorns"

  GREEDY:
    "in the Andes Mountains. The researchers found that the unicorns were
     able to communicate with each other. The researchers found that the
     unicorns were able to communicate with each other. The researchers
     found that the unicorns were able to..."   ← infinite loop

  NUCLEUS (p=0.95):
    "living in a remote valley in the Peruvian Andes. Dr. Elena Vasquez,
     a zoologist at the University of La Paz, described the discovery as
     'utterly without precedent'..."            ← varied, coherent
```

📊 **The measurement** (Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2019):

| | Human text | Beam search ($b{=}16$) | Pure sampling | Nucleus ($p{=}0.95$) |
|---|---|---|---|---|
| Perplexity of the produced text | 12.38 | **1.48** | 22.73 | 13.13 |
| Repetition | 0.28% | **28.94%** | 0.22% | 0.36% |
| Self-BLEU (lower = more diverse) | 0.31 | 0.44 | 0.28 | 0.32 |

*Source: [Holtzman et al. 2019](https://arxiv.org/abs/1904.09751), Table 1 (GPT-2 Large). Note how
nucleus sampling lands close to human text on every column, while beam search is far too
predictable and repeats itself 100× more often than people do.*

🧠 **Why maximizing likelihood fails.** Human language is *not* a sequence of locally most-probable
words. Real text has bursts of surprise: unexpected word choices are what carry information.
A maximum-likelihood decode produces the *most typical* sequence, which is bland, repetitive and
unlike anything a person would write.

```
   per-token probability along a sequence

   1.0 │  ████████████████████████████   beam search: uniformly high,
       │                                  no surprise, degenerate
   0.5 │  ╱╲  ╱╲    ╱╲  ╱╲╱╲   ╱╲
       │ ╱  ╲╱  ╲  ╱  ╲╱    ╲ ╱  ╲       human text: varied,
   0.0 │╱        ╲╱           ╲    ╲      bursts of low-probability
       └────────────────────────────────►  choices
```

🧠 **The information-theoretic framing**: a well-calibrated model has an entropy rate matching
language itself (~1 bit/char). Text generated with near-zero entropy carries almost no
information — it is, literally, uninformative. **Good generation should match the model's own
entropy, not minimize it.**

⚠️ **The important exception**: for tasks with a single correct answer — translation, extraction,
classification, math with a checkable answer — greedy or beam search is *better*. The likelihood
trap applies to open-ended generation, not to constrained tasks. Use temperature 0 for a JSON
extraction task; use temperature 0.8 for creative writing.

---

## 3. The sampling methods

### Temperature

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}$$

| $\tau$ | Effect |
|---|---|
| 0 | argmax (greedy) |
| 0.1–0.5 | focused, near-deterministic — factual QA, code |
| 0.7–1.0 | balanced — general chat |
| 1.0 | the model's calibrated distribution |
| 1.2–1.5 | creative, higher risk of incoherence |
| >2.0 | usually incoherent |

⚠️ Temperature alone is a bad knob: raising it inflates the probability of the whole long tail,
including thousands of tokens that are outright wrong. With a 100k vocabulary, the tail holds a
lot of aggregate mass.

### Top-$k$

Keep only the $k$ highest-probability tokens; renormalize.

⚠️ **The flaw is that $k$ is fixed while the distribution isn't.** After "The capital of France is",
the distribution is a spike — $k=50$ admits 49 wrong answers. After "Once upon a", it is broad —
$k=50$ cuts off many valid continuations.

### Top-$p$ (nucleus) — the standard

Keep the smallest set of tokens whose cumulative probability exceeds $p$.

🔢 **Worked example**, $p = 0.9$:

| Token | prob | cumulative | in nucleus? |
|---|---|---|---|
| `the` | 0.40 | 0.40 | ✅ |
| `a` | 0.25 | 0.65 | ✅ |
| `this` | 0.15 | 0.80 | ✅ |
| `my` | 0.12 | 0.92 | ✅ ← crosses 0.9, included |
| `some` | 0.05 | 0.97 | ❌ |
| … 99,995 more | 0.03 | 1.00 | ❌ |

Nucleus size = 4. In the "capital of France" case the nucleus would be 1; after "Once upon a" it
might be 200. **The cutoff adapts to the model's confidence** — which is exactly the property
top-$k$ lacks.

📊 Typical: $p = 0.9$–$0.95$ with $\tau = 0.7$–$1.0$.

### Min-$p$

Keep tokens with $p_i \ge p_{\text{min}}\cdot\max_j p_j$ — a threshold *relative to the top token*.

🔢 With $p_{\text{min}} = 0.1$ and a top token at 0.4, the cutoff is 0.04. With a top token at 0.05
(a genuinely uncertain step), the cutoff is 0.005 — automatically much more permissive.

📊 Min-$p$ is more robust than top-$p$ at high temperatures, because the threshold scales with
confidence rather than accumulating tail mass. It has become popular in local-model communities and
is now widely supported.

### Repetition control

| Method | Mechanism | Caution |
|---|---|---|
| Repetition penalty | divide the logits of already-seen tokens by $\rho$ (~1.1) | ⚠️ punishes necessary words ("the", variable names in code) |
| Frequency penalty | subtract $\alpha\times$count | linear in occurrences, gentler |
| Presence penalty | subtract $\alpha$ if seen at all | encourages new topics |
| **No-repeat n-gram** | hard-ban any repeated $n$-gram | ⚠️ breaks legitimate repetition (code, lists, names) |

⚠️ Repetition penalties treat a *symptom*. Persistent looping usually indicates temperature too low,
a bad prompt, or a poorly-aligned model. Fix those first.

### Comparison table

| Method | Adapts to confidence | Typical setting | Best for |
|---|---|---|---|
| Greedy | — | $\tau = 0$ | extraction, classification, math |
| Beam search | — | beams 4–8 | translation, summarization |
| Temperature only | ❌ | 0.7–1.0 | never use alone |
| Top-$k$ | ❌ | $k=40$ | legacy |
| **Top-$p$** | ✅ | $p=0.9$–$0.95$ | **general default** |
| **Min-$p$** | ✅ | $0.05$–$0.1$ | high-temperature creative work |

---

## 4. Beam search, and when it's right

Keep the $B$ highest-scoring *partial* sequences at each step.

```
  step 1        step 2            step 3
   ┌─ "The" ──┬─ "The cat" ──┬─ "The cat sat"     ← keep top B=2 paths
   │          └─ "The dog" ──┴─ "The cat ran"       by cumulative log-prob
   └─ "A"   ──── (pruned)
```

$$\text{score}(y) = \frac{1}{|y|^\lambda}\sum_t \log p(y_t\mid y_{<t})$$

⚠️ **Length normalization is required.** Without dividing by $|y|^\lambda$, every extra token adds a
negative log-probability, so the highest-scoring sequence is always the shortest one. $\lambda
\approx 0.6$–$1.0$.

📊 **Where beam search still wins**: machine translation (+1–2 BLEU over greedy), summarization,
constrained generation. **Where it fails**: open-ended text, where larger beams produce measurably
*worse* output — the "beam search curse."

---

## 5. The KV cache

At step $t$ you need attention over all previous keys and values. Recomputing them is $O(t)$ work
per step, $O(T^2)$ overall. **Cache them.**

```
  WITHOUT CACHE (wrong)                WITH CACHE (correct)

  step 1: forward([t1])                step 1: forward([t1])      → cache K,V for t1
  step 2: forward([t1,t2])             step 2: forward([t2])      → append K,V for t2
  step 3: forward([t1,t2,t3])          step 3: forward([t3])      → append K,V for t3
  ...                                  ...
  O(T²) total compute                  O(T) total compute
                                       but O(T) MEMORY
```

$$\text{cache bytes} = 2 \times L \times H_{kv}\times d_h \times T \times B\times \text{bytes/elem}$$

🔢 **LLaMA-3 70B**, BF16 ($L=80$, $H_{kv}=8$, $d_h=128$):

| Context | Batch 1 | Batch 32 |
|---|---|---|
| 4 K | 1.3 GB | 42 GB |
| 32 K | 10.7 GB | **343 GB** ⚠️ |
| 128 K | 42.9 GB | — |

⚠️ **The KV cache, not the weights, is what limits your batch size.** Weights are 140 GB and fixed;
the cache grows with every concurrent request. This is *the* constraint in LLM serving.

### PagedAttention

🧠 The problem: naive implementations allocate a contiguous buffer sized for `max_length` per
request. A request that generates 100 tokens in a 4096-token buffer wastes 97.5% of it. Measured
waste in pre-vLLM systems was 60–80%.

**The fix, borrowed straight from operating systems**: split the cache into fixed-size **blocks**
(e.g. 16 tokens) and keep a per-sequence **block table** mapping logical to physical blocks.
Blocks need not be contiguous.

```
  CONTIGUOUS (wasteful)              PAGED (vLLM)

  req A: [████░░░░░░░░░░░░]          physical blocks: [A0][B0][A1][C0][B1][A2]
  req B: [██████░░░░░░░░░░]
  req C: [██░░░░░░░░░░░░░░]          block tables:
         ↑ reserved for max len        A → [0, 2, 5]
         ~70% wasted                   B → [1, 4]
                                       C → [3]
                                     <4% waste; blocks freed on completion
```

📊 Reported gains: waste under 4%, 2–4× higher throughput from the larger batches this enables, and
**copy-on-write block sharing** so parallel samples from one prompt share the prompt's blocks.

---

## 6. Speculative decoding

🧠 **The insight**: decoding is memory-bandwidth-bound, so verifying $k$ tokens costs nearly the
same as generating 1. Use a small fast model to *guess* several tokens, then have the large model
check them all in one forward pass.

```
  DRAFT (small model, 1B):  generates 5 candidate tokens autoregressively
        "The capital of France is Paris and it is"
                              ▼
  VERIFY (large model, 70B): ONE forward pass scores all 5 positions in parallel
                              ▼
  ACCEPT: compare each draft token against the target distribution
        "The capital of France is Paris and" ✓✓✓✓✗
        → accept 4, reject the 5th, resample it from the corrected distribution
        → 5 tokens produced for the cost of ~1 large forward pass
```

📐 **The acceptance rule**, which is what makes this *exact*. For draft distribution $q$ and target
distribution $p$, accept draft token $x$ with probability $\min\left(1, \frac{p(x)}{q(x)}\right)$.
If rejected, sample from the residual distribution

$$p'(x) = \frac{\max(0,\; p(x) - q(x))}{\sum_{x'}\max(0,\; p(x')-q(x'))}$$

**Theorem (Leviathan et al., 2023)**: the resulting token sequence is distributed **exactly** as if
sampled from $p$ alone.

🧠 **This is the remarkable part.** Speculative decoding is not an approximation, not a quality/speed
tradeoff, not a heuristic. The output distribution is provably identical to standard sampling. You
get speed for free.

🔢 **Expected speedup.** With acceptance rate $\alpha$ and $k$ draft tokens per round, the expected
number of tokens accepted per verification is

$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{k+1}}{1-\alpha}$$

| $\alpha$ | $k=4$ | $k=8$ |
|---|---|---|
| 0.6 | 2.31 | 2.47 |
| 0.7 | 2.77 | 3.20 |
| 0.8 | 3.36 | 4.33 |
| 0.9 | 4.10 | 6.13 |

📊 Real-world speedups are **2–3×** after accounting for draft cost. Acceptance rates are highest
when the draft model is a distilled or smaller version of the same family (shared tokenizer and
training data matter a lot).

📊 **Variants**:

| Method | Draft source |
|---|---|
| Classic | a separate small model |
| **Medusa** | extra prediction heads on the *same* model — no second model needed |
| **EAGLE** | predicts at the feature level instead of the token level; higher acceptance |
| **Lookahead / prompt lookup** | n-gram matches from the prompt itself — free, great for summarization and editing |
| Self-speculative | skip layers of the same model to form the draft |

---

## 7. Structured generation & constrained decoding

Force valid JSON, a regex match, or a grammar by **masking invalid tokens** before sampling.

```
  Schema: {"name": string, "age": integer}

  generated so far: {"name": "Ada", "age":
  valid next tokens: only digits and whitespace
  → set the logits of everything else to −∞
```

📊 Implementations (Outlines, XGrammar, llama.cpp GBNF) compile a grammar into a finite-state
machine and precompute, for each FSM state, the allowed-token bitmask. The runtime overhead is then
close to zero.

✅ **100% syntactic validity, guaranteed** — this is a correctness guarantee, not a probabilistic
improvement.
⚠️ Constraints can *hurt* semantic quality: forcing a schema the model finds unnatural pushes it
into a low-probability region. Mitigate by making the schema match how the model would naturally
answer, and by allowing a reasoning field before the structured output.

→ [Structured output](../06-applications/05-structured-output.md)

---

## 8. Implementation

💻 A complete sampler with the standard knobs:

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=256, temperature=0.8,
             top_p=0.95, min_p=None, repetition_penalty=1.0, eos_id=None):
    past = None
    generated = input_ids
    for _ in range(max_new_tokens):
        # after the first step, only feed the newest token; the cache holds the rest
        inp = generated if past is None else generated[:, -1:]
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()          # (B, V); float for stable softmax

        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                seen = generated[b].unique()
                # divide positive logits, multiply negative ones (the standard convention)
                l = logits[b, seen]
                logits[b, seen] = torch.where(l > 0, l / repetition_penalty,
                                                     l * repetition_penalty)

        if temperature == 0.0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = logits.softmax(dim=-1)

            if min_p is not None:
                threshold = min_p * probs.max(dim=-1, keepdim=True).values
                probs = probs.masked_fill(probs < threshold, 0.0)

            if top_p is not None and top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(descending=True, dim=-1)
                cum = sorted_p.cumsum(dim=-1)
                # remove tokens once cumulative prob exceeds top_p, but always keep the first
                remove = cum - sorted_p > top_p
                sorted_p[remove] = 0.0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_p)

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_tok = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_tok], dim=-1)
        if eos_id is not None and (next_tok == eos_id).all():
            break
    return generated
```

⚠️ Note `remove = cum - sorted_p > top_p` rather than `cum > top_p`: this keeps the token that
*crosses* the threshold, matching the standard definition. Using `cum > top_p` can leave an empty
nucleus when the top token already exceeds $p$.

---

## 9. Settings cheat sheet

📊 Good starting points by task:

| Task | $\tau$ | top-$p$ | Notes |
|---|---|---|---|
| Factual QA | 0.0–0.3 | 1.0 | determinism is a feature |
| Code generation | 0.0–0.2 | 0.95 | low temp; use tests, not sampling, for diversity |
| Math / reasoning | 0.6–0.7 | 0.95 | some diversity helps self-consistency |
| General chat | 0.7–1.0 | 0.9–0.95 | the default |
| Creative writing | 1.0–1.3 | 0.95–1.0 | add min-$p$ 0.05 to protect coherence |
| Brainstorming | 1.0–1.2 | 0.98 | you want the tail |
| Structured extraction | 0.0 | 1.0 | + constrained decoding |
| Self-consistency voting | 0.7 | 0.95 | sample $n$ times, take the majority |

⚠️ **Don't stack aggressive settings.** Temperature 1.5 *and* top-$p$ 0.99 *and* no repetition
penalty produces noise. Change one knob at a time and read the outputs.

---

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Maximizing likelihood produces degenerate text — human language has variable surprise by design. |
| 2 | The exception: for single-correct-answer tasks, greedy/beam is right. Match the strategy to the task. |
| 3 | Top-$p$ beats top-$k$ because the cutoff adapts to the model's confidence. |
| 4 | Min-$p$ scales its threshold with the top token's probability — more robust at high temperature. |
| 5 | The KV cache, not the weights, limits serving batch size. |
| 6 | PagedAttention cuts cache waste from 60–80% to <4% by treating it like OS virtual memory. |
| 7 | Speculative decoding gives 2–3× speedup with a **provably identical** output distribution. |
| 8 | Constrained decoding guarantees syntactic validity but can hurt semantic quality. |
| 9 | Change one sampling knob at a time and actually read the outputs. |

---

## Further reading

- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) — introduces nucleus sampling.
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023).
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023).
- Cai et al., *Medusa* (2024); Li et al., *EAGLE* (2024).
- Willard & Louf, *Efficient Guided Generation for LLMs* (Outlines, 2023).
- Meister et al., *Locally Typical Sampling* (2022) — the entropy-matching view of decoding.

**Next** → [Efficiency: quantization, distillation, sparsity](07-efficiency.md)
