# Prompt Engineering

> **Summary** — Prompting is conditioning: you are selecting a region of the model's learned
> distribution. This page explains the *mechanism* behind each technique rather than listing
> folklore, covers what the evidence actually supports, and is explicit about which popular tricks
> stopped working once models were trained to reason.

**Prerequisites**: → [Alignment](../04-large-language-models/05-alignment.md), → [Reasoning](../04-large-language-models/10-reasoning.md) · **Next**: → [Embeddings & vector search](02-embeddings-and-vector-search.md)

---

## 1. The mechanism

A language model computes $p(\text{output} \mid \text{prompt})$. Your prompt is the conditioning
variable. Every prompting technique works by one of exactly three mechanisms:

| Mechanism | What it does | Examples |
|---|---|---|
| **1. Distribution selection** | move to a region where good answers are likely | role framing, style cues, few-shot examples |
| **2. Computation allocation** | give the model more serial compute | chain-of-thought, scratchpads, "think first" |
| **3. Information provision** | supply facts the model lacks | RAG, context, tool outputs, definitions |

> [!TIP]
> **The diagnostic question**: when a prompt improves output, ask *which of the three* it used.
> This tells you whether the improvement will generalize, and what to try when it stops working.

```
   The model's full output distribution

   ┌─────────────────────────────────────────────┐
   │  ░░░░░░  casual  ░░░░░░░░░░░░░░░░░░░░░░░░   │
   │  ░░░░░░░░░░░  wrong ░░░░░░░░░░░░░░░░░░░░░   │
   │  ░░░░ ████████████ ░░░░░░░░░░░░░░░░░░░░░░   │  ← your prompt selects
   │  ░░░░ █ careful, █ ░░░░░░░ rambling ░░░░░   │    this region
   │  ░░░░ █ technical█ ░░░░░░░░░░░░░░░░░░░░░░   │
   │  ░░░░ ████████████ ░░░░░░░░░░░░░░░░░░░░░░   │
   └─────────────────────────────────────────────┘

   You cannot create capability that isn't there.
   You can only select where in the distribution to sample from.
```

> [!WARNING]
> **The hard limit**: prompting cannot add knowledge the model lacks, and it cannot exceed the
> model's ceiling on a task. If the model genuinely cannot do something, no prompt fixes it — you
> need a better model, retrieval, tools, or fine-tuning.

---

## 2. What reliably works

Within that hard limit, some techniques reliably move the model to a better region of its distribution. Worth going through what actually has evidence behind it, rather than folklore.

### Be specific about the output you want

```
  ❌ "Summarize this."
  ✅ "Summarize this in 3 bullet points, each under 20 words,
      focusing on financial implications. Use plain language."
```

> [!TIP]
> **Mechanism 1.** "Summarize this" is consistent with thousands of valid response styles, so you
> get the distributional average. Constraints collapse the space to what you actually want.

### Provide examples (few-shot)

```
  Classify sentiment.

  Review: "Broke after two days."        Sentiment: negative
  Review: "Exactly as described."         Sentiment: positive
  Review: "It's fine I guess."            Sentiment: neutral

  Review: "Shipping was slow but the product is great."  Sentiment:
```

> [!TIP]
> **Mechanism 1, powerfully.** Examples convey the label set, the output format, the decision
> boundary for edge cases, and the tone — all at once, and more precisely than any description.

**Findings that shape how you should write them:**
- **Format matters more than correctness.** Min et al. (2022) showed that randomizing the *labels*
  in few-shot examples barely hurts performance, while changing the *format* hurts a lot. The
  examples are mostly teaching the model "what shape of answer goes here."
- **Order matters** — accuracy can swing by tens of points with example ordering, especially on
  small models. Randomize, or calibrate.
- **Diminishing returns** past 5–10 examples for most tasks.
- **Put the hardest examples last** — recency bias makes them more influential.

### Ask for reasoning before the answer

```
  ❌ "What's the answer? Then explain."
  ✅ "Think through this step by step, then give your answer."
```

> [!TIP]
> **Mechanism 2**, and the ordering is not stylistic. Because generation is autoregressive, the
> answer token is conditioned on everything before it. If the answer comes first, the "explanation"
> is a post-hoc rationalization that cannot influence it. Reasoning must come **before** the answer
> to do any computational work.

> [!WARNING]
> **For reasoning-trained models this is often unnecessary or counterproductive** — they already
> produce internal reasoning, and explicit instructions can interfere with their trained format.
> Check your model class before adding it.

### Give the model an out

```
  "If the context doesn't contain the answer, say 'I don't know.'
   Do not guess."
```

> [!TIP]
> **Mechanism 1**, addressing a specific failure. RLHF trains models to be helpful; refusing to
> answer scores as unhelpful. Explicitly authorizing "I don't know" makes it a *correct* response
> rather than a failure, which measurably reduces hallucination.

### Structure long prompts with delimiters

```
  <document>
  {long text}
  </document>

  <question>
  {question}
  </question>
```

> [!TIP]
> **Mechanism 1.** Delimiters mark boundaries unambiguously, so instructions can't be confused
> with content — and they make prompt injection harder (though not impossible).
> → [Security](../08-safety-and-ethics/02-security.md)

XML-style tags work particularly well for models trained with them. Markdown headings work too.
The key is consistency.

### Put critical content at the start or end

Directly from the **lost-in-the-middle** result
(→ [Long context §6](../04-large-language-models/09-long-context.md#6-what-long-context-actually-delivers)).
With 20 retrieved documents, the ones in positions 8–14 contribute little. Put the most relevant
material first, and repeat the question at the end.

### Prefill the response

```
  User: List three benefits as JSON.
  Assistant: {"benefits": ["
```

> [!TIP]
> **Mechanism 1, at its most direct.** You've conditioned on tokens that make any non-JSON
> continuation extremely unlikely. This is the cheapest possible format enforcement, and it works on
> any API that lets you prefill the assistant turn.

---

## 3. What doesn't work (or stopped working)

Everything in §2 still earns its place. Just as important is knowing what used to be standard advice and has quietly stopped mattering, because the models it was designed for don't exist anymore.

> [!WARNING]
> Honest assessment of popular advice:

| Technique | Verdict |
|---|---|
| "You are an expert X" | ⚠️ Small effect on modern models. Was more useful pre-2023. Harmless. |
| Offering a tip / emotional appeals | ⚠️ Early results were noisy and haven't replicated robustly. |
| Threats, urgency, ALL CAPS | ❌ No reliable evidence. |
| "Take a deep breath" | ⚠️ Found by automated prompt search on one model; does not transfer. |
| "Let's think step by step" | ✅ Still works on non-reasoning models; unnecessary on reasoning models. |
| Saying "please" | ❌ No measurable effect. (Do it anyway if you like.) |
| Very long system prompts | ⚠️ Diminishing returns; instructions get lost. Under ~500 words is usually better. |
| Repeating instructions | ✅ Genuinely helps for long contexts — restate the task at the end. |
| Negative instructions ("don't be verbose") | ⚠️ Weaker than positive ones ("respond in under 50 words"). |

> [!TIP]
> **Why persona prompts faded.** Pre-RLHF base models genuinely needed distribution selection —
> "you are an expert" moved them from Reddit-comment-space to textbook-space. Post-RLHF models are
> *already* conditioned to respond as a knowledgeable assistant, so the persona adds little. The
> technique didn't stop working; the models changed such that it was already applied.

> [!TIP]
> **Why negative instructions are weak.** "Don't mention X" requires the model to represent X in
> order to avoid it, and the attention mechanism has no clean "suppress" operation. State the
> positive target instead.

---

## 4. Techniques by task type

Knowing what generally works and what doesn't is only half the picture — which specific technique to reach for depends heavily on what kind of task you're actually prompting for.

| Task | Approach |
|---|---|
| **Classification** | few-shot with all labels represented; constrain the output to the label set |
| **Extraction** | JSON schema + constrained decoding; temperature 0; give a "not found" value |
| **Summarization** | specify length, audience, focus; give an example if the style matters |
| **Reasoning / math** | CoT (or a reasoning model); self-consistency with $n\ge5$; verify if possible |
| **Code** | provide signatures, types, tests, and surrounding context; ask for tests too |
| **Creative writing** | higher temperature; specify voice and constraints, not content |
| **RAG QA** | context first, question last; cite-your-sources instruction; explicit "I don't know" |
| **Agents** | clear tool descriptions; ReAct format; explicit stop conditions |

---

## 5. A prompt template that generalizes

Those task-specific techniques all have to live inside one actual prompt, in some order. That ordering isn't arbitrary — where you place the task, the input and the constraints changes how well the model attends to each.

```
  ┌─ ROLE / CONTEXT (optional, brief) ────────────────────────┐
  │ You are reviewing legal contracts for a procurement team. │
  └───────────────────────────────────────────────────────────┘
  ┌─ TASK (specific, imperative) ─────────────────────────────┐
  │ Identify any clause that permits unilateral price changes.│
  └───────────────────────────────────────────────────────────┘
  ┌─ INPUT (delimited) ───────────────────────────────────────┐
  │ <contract>                                                │
  │ {text}                                                    │
  │ </contract>                                               │
  └───────────────────────────────────────────────────────────┘
  ┌─ CONSTRAINTS ─────────────────────────────────────────────┐
  │ - Quote the exact clause text.                            │
  │ - If no such clause exists, output: {"clauses": []}       │
  │ - Do not infer intent; report only explicit language.     │
  └───────────────────────────────────────────────────────────┘
  ┌─ OUTPUT FORMAT ───────────────────────────────────────────┐
  │ {"clauses": [{"section": "...", "quote": "...",           │
  │               "risk": "high|medium|low"}]}                │
  └───────────────────────────────────────────────────────────┘
  ┌─ EXAMPLES (1-3, optional but valuable) ───────────────────┐
  │ ...                                                       │
  └───────────────────────────────────────────────────────────┘
```

**The ordering is deliberate**: task before input (so the model reads the document knowing what
to look for), constraints and format last (so they're closest to the generation point and least
likely to be lost).

---

## 6. Evaluating prompts

That template is a strong starting point, not a guarantee. Whether one prompt is actually better than another for your task is an empirical question, and it needs the same rigor as any other empirical comparison.

> [!WARNING]
> **The most common mistake in prompt engineering is testing on three examples and declaring
> victory.** Prompt changes have high variance; you cannot distinguish improvement from noise
> without a proper set.

A minimal but real prompt evaluation harness:

```python
import json, statistics
from concurrent.futures import ThreadPoolExecutor

def evaluate_prompt(prompt_template, test_cases, model, scorer, n_repeats=3):
    """test_cases: [{"input": ..., "expected": ...}, ...]
       scorer: (output, expected) -> float in [0, 1]"""
    def run(case):
        scores = []
        for _ in range(n_repeats):                  # repeat: LLM outputs are stochastic
            out = model(prompt_template.format(**case["input"]))
            scores.append(scorer(out, case["expected"]))
        return statistics.mean(scores)

    with ThreadPoolExecutor(max_workers=8) as ex:
        per_case = list(ex.map(run, test_cases))

    mean = statistics.mean(per_case)
    # standard error: how confident are we that a difference is real?
    stderr = statistics.stdev(per_case) / len(per_case) ** 0.5 if len(per_case) > 1 else 0.0
    return {"mean": mean, "stderr": stderr,
            "worst": sorted(zip(per_case, test_cases))[:5]}   # inspect the failures

# A/B two prompts; only believe a difference larger than ~2 combined standard errors.
a, b = evaluate_prompt(P1, cases, model, score), evaluate_prompt(P2, cases, model, score)
delta = b["mean"] - a["mean"]
margin = 2 * (a["stderr"]**2 + b["stderr"]**2) ** 0.5
print(f"delta = {delta:+.3f} ± {margin:.3f} -> "
      f"{'significant' if abs(delta) > margin else 'NOT significant'}")
```

**Rules of thumb:**
- **50+ test cases minimum**; 200+ if differences are small.
- **Repeat each case** — the same prompt gives different outputs.
- **Always look at the worst failures.** Aggregate scores hide the failure modes that matter.
- **Hold out a test set.** Iterating on your evaluation set overfits to it, exactly like training.

---

## 7. Automatic prompt optimization

Manually iterating against that evaluation loop works, but it's slow and it's exactly the kind of search a model can do faster than a human — which is what automatic prompt optimization tools exist for.

Since prompts are just text, they can be optimized:

| Method | Idea |
|---|---|
| **APE** | an LLM generates candidate prompts; score them; keep the best |
| **OPRO** | an LLM sees the score history and proposes improvements |
| **DSPy** | compile a declarative program into optimized prompts + few-shot examples |
| **TextGrad** | "backpropagate" natural-language critiques through a pipeline |
| Evolutionary | mutate and recombine prompts |

> [!TIP]
> **DSPy is the most useful framing for practitioners**: separate *what* you want (a signature like
> `question -> answer`) from *how* to prompt for it. The optimizer then selects demonstrations and
> instruction wording against your metric. It converts prompt engineering from craft into a
> compilation step — which is the right direction, since hand-tuned prompts don't transfer across
> models and break on every model update.

> [!WARNING]
> Automatically found prompts are often strange and model-specific ("take a deep breath" came from
> exactly this process). They do not transfer. Re-optimize per model.

---

## 8. Exercises

**Problem 1 — the three mechanisms, diagnosed.** For each of the following prompt changes, say
which of §1's three mechanisms (distribution selection, computation allocation, information
provision) it primarily uses: (a) adding "Cite the exact paragraph you're quoting from" to a RAG
prompt; (b) adding "First list every constraint, then check your answer against each one" to a
scheduling prompt; (c) adding three worked examples of the desired output format.

<details markdown="1"><summary>Solution</summary>

(a) **Distribution selection.** It doesn't add new facts to the context (the source text is
already there); it constrains *how* the model must use what's already available, shaping which
region of the output distribution (citation-grounded vs free-form) gets selected.

(b) **Computation allocation** — per §1, this asks the model to do more serial work (enumerate
constraints, then verify) before producing the final answer, exactly the mechanism behind
chain-of-thought's effectiveness (§2's "reasoning must come before the answer" point).

(c) **Distribution selection** — per §2, few-shot examples convey "the label set, the output
format, the decision boundary... more precisely than any description," steering toward the
region of output-space matching the demonstrated format, without adding any new facts about the
task's subject matter.

</details>

**Problem 2 — few-shot ordering, applied.** Using §2's finding that format matters more than
label correctness and that order can swing accuracy substantially on small models, a colleague
proposes always putting the *easiest* example first in a few-shot prompt "so the model warms up
gradually." Using §2's "recency bias" note, is this the right call?

<details markdown="1"><summary>Solution</summary>

§2 states explicitly: "put the hardest examples last — recency bias makes them more
influential." Putting the *easiest* example first (and, implicitly, harder ones later or last)
actually aligns with this — if "warms up gradually" means easy-to-hard ordering, that's the
*correct* strategy per §2, since it puts the hardest, most informative example in the
highest-influence (last) position. The colleague's instinct happens to be right, though for a
different reason than "warming up": it's not that the model needs gradual exposure, it's that
recency bias means whatever appears last gets weighted more heavily in shaping the immediate
continuation — so you want your *most representative/hardest* example there, which for a
graduated easy→hard ordering happens to be the last one anyway.

</details>

**Problem 3 — evaluating a prompt change, is it real?** Using §6's bootstrap-style statistical
reasoning, two prompts are compared on 80 test cases: Prompt A scores 71%, Prompt B scores 76%.
Using the rule of thumb from → [Evaluation metrics §6](../07-evaluation/01-metrics.md#6-statistical-significance)
(standard error $\approx\sqrt{p(1-p)/n}$), is a 5-point gap on 80 examples likely to be
statistically meaningful, or likely noise?

<details markdown="1"><summary>Solution</summary>

Using $p\approx0.735$ (roughly the average of 0.71 and 0.76) and $n{=}80$:

$$SE \approx \sqrt{0.735(1-0.735)/80} = \sqrt{0.00244}=0.0494 \approx 4.9\text{ points}$$

A 5-point observed gap is only about **1 standard error** — well short of the conventional
~2-SE threshold for statistical significance. Per §6's warning ("a 2-point difference on 200
examples is noise"), this is an even smaller sample (80 vs 200) with a bigger nominal gap (5 vs
2 points), but the math still says this is **not distinguishable from noise** at typical
confidence levels — the honest conclusion is "run more test cases before trusting this
5-point improvement," not "Prompt B is better."

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Prompting does exactly three things: selects a distribution, allocates computation, or supplies information. |
| 2 | Prompting cannot create capability — only locate it. |
| 3 | Specificity about the output is the highest-value single change. |
| 4 | In few-shot examples, format matters more than label correctness; order matters too. |
| 5 | Reasoning must come **before** the answer or it does no computational work. |
| 6 | Explicitly authorize "I don't know" — RLHF otherwise penalizes refusal. |
| 7 | Put critical content at the start and end; the middle gets ignored. |
| 8 | Prefilling the assistant response is the cheapest possible format enforcement. |
| 9 | Persona prompts and emotional appeals have little reliable effect on modern models. |
| 10 | Evaluate on 50+ cases with repeats and a confidence margin, or you're measuring noise. |

---

## Further reading

- Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (2020) — in-context learning.
- Min et al., [*Rethinking the Role of Demonstrations*](https://arxiv.org/abs/2202.12837) (2022) — the label-randomization result.
- Wei et al., [*Chain-of-Thought Prompting*](https://arxiv.org/abs/2201.11903) (2022); Kojima et al., [*Zero-shot CoT*](https://arxiv.org/abs/2205.11916) (2022).
- Zhou et al., [*Large Language Models Are Human-Level Prompt Engineers*](https://arxiv.org/abs/2211.01910) (APE, 2022).
- Khattab et al., [*DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*](https://arxiv.org/abs/2310.03714) (2023).
- Yang et al., [*Large Language Models as Optimizers*](https://arxiv.org/abs/2309.03409) (OPRO, 2023).
- Sclar et al., [*Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*](https://arxiv.org/abs/2310.11324) (2023).

**Next** → [Embeddings & vector search](02-embeddings-and-vector-search.md)
