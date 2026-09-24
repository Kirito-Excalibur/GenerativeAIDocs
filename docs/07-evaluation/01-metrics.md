# Evaluation Metrics

> **Summary** — How to measure whether a generative model is any good, and — more importantly —
> what each metric fails to capture. Covers perplexity and bits-per-character for language models,
> BLEU/ROUGE/BERTScore for text comparison, FID/IS/CLIPScore for images, and LLM-as-judge with its
> documented biases. The recurring theme: every automatic metric is a proxy, and every proxy can be
> gamed.

**Prerequisites**: → [Probability & information theory](../01-foundations/02-probability-and-information-theory.md) · **Next**: → [Benchmarks](02-benchmarks.md)

---

## 1. Perplexity

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i \mid x_{<i})\right)$$

> [!TIP]
> "The model is as uncertain as if choosing uniformly among PPL options at each step."

**Reference points** (vocabulary 50k):

| Loss (nats) | PPL | Interpretation |
|---|---|---|
| 10.82 | 50,257 | untrained (uniform) |
| 4.00 | 54.6 | weak |
| 3.00 | 20.1 | small neural LM |
| 2.00 | 7.39 | decent |
| 1.50 | 4.48 | strong |

> [!WARNING]
> **Perplexity's four failure modes:**

| Problem | Why |
|---|---|
| **Not comparable across tokenizers** | a 128k-vocab model packs more characters per token, inflating per-token PPL |
| **Not comparable across datasets** | PPL on Wikipedia ≠ PPL on Reddit |
| Weakly correlated with usefulness | a model can lower PPL by memorizing boilerplate |
| Meaningless after RLHF | alignment *raises* PPL on web text while improving the assistant |

**The fix for tokenizer comparison — bits per character:**

$$\text{BPC} = \frac{\text{total nats}}{\ln 2 \times \text{number of characters}}$$

A model at 2.0 nats/token with 4.1 chars/token: $\frac{2.0}{0.693\times4.1} = 0.704$ bits/char.
**Always report BPC (or bits per byte) when comparing across tokenizers.**

**The honest use of perplexity**: it is an excellent *training* diagnostic — smooth, cheap,
sensitive, and the thing scaling laws predict. It is a poor *product* metric.

---

## 2. Text-comparison metrics

Perplexity works because a language model outputs a probability, so you can score it against ground truth directly. Most generative tasks — translation, summarization — don't have that luxury: you only get the generated text, and have to compare it to a reference some other way.

### BLEU: n-gram precision

$$\text{BLEU} = \underbrace{\text{BP}}_{\text{brevity penalty}}\cdot\exp\!\left(\sum_{n=1}^{4} w_n\log p_n\right)$$

where $p_n$ is the (clipped) $n$-gram precision and
$\text{BP} = \min\left(1, e^{1 - r/c}\right)$ penalizes short outputs.

**Worked example.**
Reference: *"the cat sat on the mat"*
Candidate: *"the cat is on the mat"*

| $n$ | Candidate n-grams | Matching | $p_n$ |
|---|---|---|---|
| 1 | the, cat, is, on, the, mat | the, cat, on, the, mat = 5 | 5/6 = 0.833 |
| 2 | the cat, cat is, is on, on the, the mat | the cat, on the, the mat = 3 | 3/5 = 0.600 |
| 3 | the cat is, cat is on, is on the, on the mat | on the mat = 1 | 1/4 = 0.250 |
| 4 | 3 grams | 0 | 0/3 = 0 ⚠️ |

$p_4 = 0$ makes the geometric mean zero → **BLEU = 0**, despite an output that is clearly mostly
correct. (In practice smoothing is applied, but the brittleness is real.)

> [!WARNING]
> **BLEU cannot see meaning.** *"The film was terrible"* and *"The movie was awful"* share almost
> no n-grams and score near zero against each other.

### ROUGE: n-gram recall (for summarization)

| Variant | Measures |
|---|---|
| ROUGE-N | n-gram recall |
| **ROUGE-L** | longest common subsequence — allows gaps |
| ROUGE-W | weighted LCS, favouring contiguity |

> [!WARNING]
> ROUGE rewards **copying**. An extractive summary that lifts sentences verbatim scores higher
> than a better abstractive one. This is a known and serious bias in the summarization literature.

### BERTScore: embedding-based

Match candidate and reference tokens by **cosine similarity of contextual embeddings**, then take
the F1 of greedily-matched similarities.

- ✅ Captures paraphrase and synonymy. ✅ Correlates better with human judgement than BLEU/ROUGE.
- ⚠️ Depends on the embedding model; not interpretable; still assumes one reference is the truth.

### Comparison

| Metric | Measures | Sees meaning? | Use for |
|---|---|---|---|
| BLEU | n-gram precision | ❌ | translation (legacy) |
| ROUGE | n-gram recall | ❌ | summarization (legacy) |
| METEOR | + stems, synonyms | ⚠️ partly | translation |
| **BERTScore** | embedding similarity | ✅ | general text similarity |
| **COMET** | learned, trained on human ratings | ✅ | ⭐ translation (current standard) |
| **LLM-as-judge** | model judgement | ✅ | ⭐ open-ended tasks |
| Exact match | string equality | ❌ | QA with short canonical answers |

> [!TIP]
> **The structural problem with all reference-based metrics**: they assume a *single* correct
> output. For translation there are many valid renderings; for summarization, many valid summaries;
> for open generation, effectively infinite. Reference-based metrics fundamentally cannot handle
> this, which is why the field moved to learned metrics (COMET) and model judges.

---

## 3. Image metrics

That same single-correct-output problem is even sharper for images, where there's no text sequence to compare token by token at all — evaluation has to work in a completely different space.

### FID: Frechet Inception Distance

Embed real and generated images with InceptionV3, fit a Gaussian to each set, and measure the
distance between them:

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$$

**Reference values** (ImageNet 256×256, lower is better):

| Model | FID |
|---|---|
| BigGAN-deep | 6.95 |
| ADM (diffusion) | 4.59 |
| DiT-XL/2 | 2.27 |
| Modern flow/diffusion | ~2 |

> [!WARNING]
> **FID's problems are serious and widely underappreciated:**

| Problem | Detail |
|---|---|
| **Sample-size biased** | FID computed on 10k samples is systematically higher than on 50k. **Always report $N$.** |
| Assumes Gaussianity | Inception features are not Gaussian |
| Inception-specific | inherits ImageNet's biases; poor for faces, medical images, art |
| Insensitive to some distortions | can miss artifacts humans find glaring |
| Not comparable across implementations | resize interpolation and preprocessing differ between libraries |

> [!TIP]
> **The practical rule**: FID is useful for comparing *your own* models under an *identical*
> pipeline. Comparing FIDs across papers is unreliable unless they used the same code and sample
> count.

### Inception Score

$$\text{IS} = \exp\!\left(\mathbb{E}_x\big[D_{\mathrm{KL}}(p(y\mid x)\,\|\,p(y))\big]\right)$$

Rewards confident per-image classification (quality) and diverse marginal classes (diversity).

> [!WARNING]
> **Largely deprecated.** It never looks at real images, so it cannot detect a model that produces
> perfect ImageNet-class images unlike the training distribution. It is also trivially gamed by
> generating one crisp image per class.

### Precision and Recall for generative models

FID conflates two distinct failures. Split them:

```
   real manifold        generated
       ╭────╮            ╭────╮
       │ ▓▓ │            │ ░░ │
       ╰────╯            ╰────╯

   PRECISION: what fraction of generated samples fall on the real manifold?
              (low ⇒ producing unrealistic images)
   RECALL:    what fraction of the real manifold is covered by generated samples?
              (low ⇒ MODE COLLAPSE)
```

> [!TIP]
> This is much more diagnostic than FID. A GAN with mode collapse can have decent FID but terrible
> recall. Always report both when comparing generative models.

### CLIPScore: text–image alignment

$$\text{CLIPScore} = \max\big(0,\ w\cdot\cos(\text{CLIP}_{\text{img}}(I),\ \text{CLIP}_{\text{txt}}(T))\big)$$

> [!WARNING]
> Inherits every CLIP weakness — compositionality, counting, spatial relations
> (→ [Multimodal §2](../05-diffusion-and-vision/05-multimodal.md#2-clip-a-shared-embedding-space)).
> A model that generates "a blue cube on a red sphere" for "a red cube on a blue sphere" scores
> almost as well as a correct one.

Better alternatives for prompt adherence: **VQAScore** (ask a VLM yes/no questions about the
image) and structured benchmarks like T2I-CompBench.

---

## 4. LLM-as-judge

CLIPScore and its relatives are all narrow, single-number metrics built for one specific question. For open-ended quality — is this response actually good — the field has largely converged on a more flexible, more expensive judge: another model.

Use a strong model to score outputs. Now the dominant method for open-ended evaluation.

```
  ┌─────────────────────────────────────────────────┐
  │ Rate this response for helpfulness, 1-10.       │
  │                                                 │
  │ Question: {question}                            │
  │ Response: {response}                            │
  │                                                 │
  │ First explain your reasoning, then give a score │
  │ as: SCORE: X                                    │
  └─────────────────────────────────────────────────┘
```

**Agreement with human raters is typically 80–85%** — comparable to the agreement *between* two
humans. That's the reason it works at all.

> [!WARNING]
> **The documented biases, and their fixes:**

| Bias | Effect | Fix |
|---|---|---|
| **Position** | prefers the first (or second) option presented | **swap the order and average** ⭐ |
| **Verbosity** | prefers longer answers regardless of content | length-control, or penalize explicitly |
| **Self-preference** | models rate their own outputs higher | use a different model family as judge |
| Style over substance | prefers confident, well-formatted answers | ask about specific criteria, not overall quality |
| Sycophancy | agrees with a stated expected answer | never reveal the expected answer |
| Poor calibration | clusters on 7–8 out of 10 | prefer **pairwise comparison** to absolute scoring |

**Position bias is large.** In MT-Bench's test, a judge was "consistent" if it picked the same
winner after the two answers were swapped. GPT-4 was consistent only **65%** of the time;
GPT-3.5 **46%**; Claude-v1 **24%**, favouring whichever answer came first in 75% of cases
([Zheng et al. 2023](https://arxiv.org/abs/2306.05685), Table 2). A pairwise evaluation without
order-swapping is close to meaningless.

**A judge implementation with the fixes applied:**

```python
def judge_pairwise(judge_model, question, response_a, response_b, criteria):
    """Pairwise comparison with position-bias control."""
    template = """Compare two responses on: {criteria}

Question: {q}

Response 1:
{r1}

Response 2:
{r2}

Think step by step about how each meets the criteria, then output exactly one of:
VERDICT: 1
VERDICT: 2
VERDICT: TIE"""

    def ask(r1, r2):
        out = judge_model(template.format(criteria=criteria, q=question, r1=r1, r2=r2))
        v = out.rsplit("VERDICT:", 1)[-1].strip().split()[0]
        return v

    v1 = ask(response_a, response_b)          # A first
    v2 = ask(response_b, response_a)          # B first — swapped

    # translate the swapped verdict back to A/B terms
    flip = {"1": "2", "2": "1", "TIE": "TIE"}
    v2 = flip.get(v2, "TIE")

    if v1 == v2:
        return {"1": "A", "2": "B", "TIE": "TIE"}[v1]
    return "TIE"                              # disagreement under swap => genuinely close
```

> [!TIP]
> **Treating swap-disagreement as a tie is the key line.** If the judge changes its mind when you
> reorder the candidates, its preference is not real. This single change substantially improves
> agreement with human raters.

**Judge best practices:**
- **Pairwise > absolute scoring** (humans and models both compare better than they calibrate).
- **Always swap positions** and treat disagreement as a tie.
- **Give a rubric**, not "rate the quality."
- **Reasoning before the verdict** (autoregressive ordering).
- **Validate against human labels** on ~100 examples before trusting the judge at scale.
- Use a **different model family** than the one being evaluated.

---

## 5. Task-specific metrics

An LLM judge is a general-purpose tool for open-ended comparison. Plenty of tasks have a sharper, cheaper notion of correctness available — one that doesn't need a judge at all.

| Task | Metric | Note |
|---|---|---|
| Classification | accuracy, macro-F1 | use macro-F1 for imbalanced classes |
| Extraction | field-level precision/recall | measure per field, not per document |
| **Code** | **pass@k** (execution) | the gold standard: it either runs or it doesn't |
| Math | exact match on the final answer | normalize formatting first |
| RAG | faithfulness, recall@k | evaluate stages separately → [RAG §6](../06-applications/03-rag.md#6-evaluation) |
| Agents | task success rate over $n$ runs | → [Agents §7](../06-applications/04-agents-and-tool-use.md#7-evaluating-agents) |
| Safety | attack success rate, over-refusal rate | ⚠️ measure **both** |
| Calibration | ECE, Brier score | does stated confidence match accuracy? |

**pass@k, computed correctly.** The naive estimator (generate $k$, check if any passes) has high
variance. The unbiased estimator generates $n \ge k$ samples, counts $c$ correct, and computes:

$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$



```python
import numpy as np

def pass_at_k(n, c, k):
    """Unbiased estimator. n = samples generated, c = correct, k = the k in pass@k."""
    if n - c < k:
        return 1.0
    # 1 - P(all k chosen samples are from the n-c incorrect ones)
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

With $n=20$, $c=5$, $k=1$: pass@1 $= 5/20 = 0.25$. With $k=10$: pass@10 $= 0.984$. **The gap
between pass@1 and pass@10 tells you how much a verifier would buy you** — see
→ [Reasoning §3](../04-large-language-models/10-reasoning.md#3-scaling-test-time-compute).

**Expected Calibration Error** — bin predictions by confidence and compare to accuracy:

$$\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{N}\big|\text{acc}(B_m) - \text{conf}(B_m)\big|$$

Base models are reasonably calibrated; **RLHF makes calibration much worse** — aligned models
express high confidence almost uniformly. This is a direct consequence of preference training
rewarding confident-sounding answers.

---

## 6. Statistical significance

Every metric above — perplexity, BLEU, FID, judge win rate, calibration — is a number computed on a sample. Before trusting that one model beats another because its number is higher, it's worth asking whether that difference is even real.

> [!WARNING]
> **Most reported model comparisons are not statistically significant and do not say so.**

With $n=200$ test examples and an observed accuracy difference of 2 percentage points, the
standard error of a paired difference is roughly

$$\text{SE} \approx \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{0.25}{200}} = 0.035 = 3.5\%$$

**A 2-point difference on 200 examples is noise.** You need ~2,400 examples to resolve a 2-point
difference at 95% confidence, and more for smaller differences.

**Bootstrap, the simplest correct approach:**

```python
import numpy as np

def bootstrap_diff(scores_a, scores_b, n_boot=10000, seed=0):
    """Paired bootstrap CI for the difference in mean score."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    assert len(a) == len(b), "use paired scores on the same examples"
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(b.mean() - a.mean()),
            "ci95": (float(lo), float(hi)),
            "significant": bool(lo > 0 or hi < 0)}
```

> [!TIP]
> **Use paired comparisons.** Evaluating both models on the *same* examples removes example
> difficulty as a variance source and gives much tighter intervals than independent samples.

---

## 7. Building an evaluation suite

All of these metrics and statistical tools are ingredients. Actually running them on a schedule, on the right data, as part of a real workflow — rather than as one-off scripts — is what turns them into an evaluation suite.

A practical structure:

| Layer | Contents | Run when |
|---|---|---|
| **Smoke tests** | 10–20 cases covering basic function | every change |
| **Regression set** | 100–500 cases including every past bug | every change |
| **Capability set** | 500+ cases across task types | weekly / pre-release |
| **Adversarial set** | jailbreaks, edge cases, injections | pre-release |
| **Human evaluation** | 50–100 cases, real raters | major releases |
| **Production monitoring** | sampled live traffic | continuously |

> [!TIP]
> **The regression set is the most valuable and most neglected.** Every time you find a failure,
> add it to the set. Over months this becomes a precise map of your system's real failure modes —
> far more informative than any public benchmark, because it reflects *your* distribution.

> [!WARNING]
> **Hold out a genuine test set.** Iterating on your evaluation set overfits to it exactly like
> training. Keep a portion sealed and look at it rarely.

---

## 8. Exercises

**Problem 1 — BLEU precision, a new pair.** Reference: "the quick brown fox jumps". Candidate:
"the fast brown fox leaps". Compute unigram and bigram precision using §2's method. Does the
low bigram overlap despite a reasonable *semantic* match (the sentences mean nearly the same
thing) illustrate §2's core criticism of BLEU?

<details markdown="1"><summary>Solution</summary>

Unigram: candidate words $\{$the, fast, brown, fox, leaps$\}$; matches against the reference
$\{$the, quick, brown, fox, jumps$\}$: "the," "brown," "fox" match — $3/5=0.6$ precision.

Bigram: candidate bigrams (the,fast), (fast,brown), (brown,fox), (fox,leaps); only (brown,fox)
appears in the reference's bigrams — $1/4=0.25$ precision.

Yes — this is a direct illustration of §2's criticism. "the quick brown fox jumps" and "the fast
brown fox leaps" describe essentially the same scene with synonym substitutions (quick↔fast,
jumps↔leaps), a translation or paraphrase a human would rate as excellent — yet bigram precision
is only 25%, and a full 4-gram BLEU score here would very likely be **zero** (no 4-gram overlaps
at all, the same "BLEU = 0 despite a mostly-correct output" failure §2's worked example shows for
a different sentence pair), because BLEU has no mechanism for recognizing that "fast" and "quick"
carry the same meaning.

</details>

**Problem 2 — Elo, a new gap.** Using §4's logistic formula, compute the win probability at a
150-point gap and a 300-point gap. Interpolating between §4's table rows (100→64%, 200→76%,
400→91%), do your computed values fall roughly where you'd expect, and does the curve's shape
(concave, convex, or roughly linear in this range) matter for that intuition?

<details markdown="1"><summary>Solution</summary>

150-point gap: $P=1/(1+10^{-150/400})=0.703$ — **70.3%**, sitting between the 64% (100-gap) and
76% (200-gap) table rows, roughly where linear interpolation would suggest (linear interpolation
of 64% and 76% at the midpoint gives 70%, very close to the actual 70.3%).

300-point gap: $P=1/(1+10^{-300/400})=0.849$ — **84.9%**, between 76% (200-gap) and 91%
(400-gap); linear interpolation of those two at the midpoint would suggest 83.5%, close to but
slightly under the actual 84.9%.

The logistic curve is very close to linear over this particular range (100–400 point gaps,
50–91% probability), so simple interpolation works reasonably well here — but this is a
local property, not a general one: near the extremes (very large or very small gaps), the
logistic curve flattens out (approaching 0% or 100% asymptotically) and linear interpolation
would fail badly, e.g. interpolating from a 400-gap (91%) toward a hypothetical 800-gap would
wrongly suggest probabilities above 100%, when the true logistic curve instead approaches 99%
gradually.

</details>

**Problem 3 — confidence intervals, a bigger benchmark.** A 500-example benchmark shows model A
at 75% accuracy. Using §6's standard-error formula, compute the approximate 95% confidence
interval. How does this compare with §6's own HumanEval example (164 items, where a 1-point gap
was shown to be noise)? At what benchmark size would a 1-point gap become meaningful at this
same accuracy level?

<details markdown="1"><summary>Solution</summary>

$SE=\sqrt{0.75(0.25)/500}=0.0194$; 95% CI $\approx\pm1.96\times0.0194=\pm3.8$ points, i.e.
roughly $71.2\%$ to $78.8\%$.

A 500-item benchmark (this problem) gives a *tighter* interval ($\pm3.8$ points) than a 164-item
one, but $\pm3.8$ is still far larger than "1 point" — so even at $n{=}500$, a 1-point gap
between two models remains within noise, consistent with §6's general point that you need
*substantially* more examples than 500 to resolve single-point differences reliably.

To make a 1-point gap meaningful, use the same "gap $\approx 2\times SE$" rule implicit in
§6: a 1-point gap (0.01 in proportion terms) needs $SE\approx0.005$. Solving
$0.005=\sqrt{0.1875/n}$ gives $n\approx0.1875/0.005^2=7{,}500$ examples.

Applying the identical rule to §6's own 2-point claim gives $n\approx0.1875/0.01^2=1{,}875$ —
same order of magnitude as §6's stated "~2,400," with the small gap attributable to rounding and
exactly which confidence multiplier (1.96 vs a looser "2×") is used. The scaling relationship is
the robust part: since required $n\propto1/(\text{gap})^2$, halving the gap from 2 points to 1
point roughly **quadruples** the examples needed — 7,500 versus 1,875 here is almost exactly
that 4× relationship, confirming the scaling law even where the exact constant is only
approximate.

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Every automatic metric is a proxy; every proxy can be gamed (Goodhart). |
| 2 | Perplexity is a great training diagnostic and a poor product metric. Use BPC across tokenizers. |
| 3 | BLEU and ROUGE cannot see meaning; ROUGE actively rewards copying. |
| 4 | FID is sample-size biased and implementation-sensitive — only compare within one pipeline, and report $N$. |
| 5 | Report precision **and** recall for generative models; FID conflates realism with coverage. |
| 6 | LLM judges agree with humans ~80–85% — but only if you control position bias by swapping. |
| 7 | Prefer pairwise comparison to absolute scoring, for both humans and model judges. |
| 8 | Use the unbiased pass@k estimator; the pass@1 → pass@k gap measures verifier headroom. |
| 9 | RLHF degrades calibration — aligned models are confidently wrong more often. |
| 10 | A 2-point difference on 200 examples is noise. Bootstrap, and use paired evaluation. |
| 11 | Build a regression set from your own bugs — it beats any public benchmark for your use case. |

---

## Further reading

- Papineni et al., [*BLEU*](https://aclanthology.org/P02-1040/) (2002); Lin, *ROUGE* (2004) — read them to see what they actually claim.
- Zhang et al., [*BERTScore*](https://arxiv.org/abs/1904.09675) (2019); Rei et al., [*COMET*](https://arxiv.org/abs/2009.09025) (2020).
- Heusel et al., [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*](https://arxiv.org/abs/1706.08500) (FID, 2017).
- Kynkäänniemi et al., [*Improved Precision and Recall Metric for Assessing Generative Models*](https://arxiv.org/abs/1904.06991) (2019).
- Zheng et al., [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685) (2023).
- Chen et al., [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) (2021) — the pass@k estimator.
- Guo et al., [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) (2017).
- Chaganty et al., [*The Price of Debiasing Automatic Metrics*](https://arxiv.org/abs/1807.02202) (2018).

**Next** → [Benchmarks](02-benchmarks.md)
