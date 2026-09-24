# Reasoning and Test-Time Compute

> **Summary** — A Transformer spends identical compute on every token, whether the next word is
> obvious or requires ten steps of deduction. Chain-of-thought works around this by using the token
> stream as a variable-length scratchpad, converting extra tokens into extra serial computation.
> Scaling that at inference — more samples, longer reasoning, search — is a **second scaling axis**
> that needs no retraining, and training models to use it well via RL on verifiable rewards is the
> basis of reasoning models.

**Prerequisites**: → [Alignment](05-alignment.md), → [RL basics](../01-foundations/07-reinforcement-learning.md), → [Scaling laws](03-scaling-laws.md) · **Next**: → [Diffusion language models](11-diffusion-language-models.md)

---

## 1. The computational argument

A Transformer with $L$ layers applies exactly $L$ sequential transformations per token. That is a
**fixed circuit depth**.

```
  "The capital of France is ___"         "What is 17 × 24 + 391 / 23 ? ___"
       │                                        │
       ▼                                        ▼
  ┌──────────┐                            ┌──────────┐
  │ L layers │  ← plenty                  │ L layers │  ← not enough serial depth
  └──────────┘                            └──────────┘
       ▼                                        ▼
    "Paris" ✓                              a guess ✗
```

**The formal version.** A fixed-depth Transformer with $O(\log T)$-bit precision lies in the
complexity class $\mathsf{TC}^0$ (constant-depth threshold circuits). Problems believed to lie
outside $\mathsf{TC}^0$ — such as evaluating arbitrary boolean formulas, or simulating $T$ steps of
a finite automaton — **cannot be solved in one forward pass regardless of width**.

**What chain-of-thought buys.** Merrill & Sabharwal (2024) showed that a Transformer generating
$k$ intermediate tokens can simulate a computation of depth $O(k)$. Each generated token is an
additional pass through the whole network, with the previous output available as input.

$$\text{Transformer} + \text{CoT of length } k \;\approx\; \text{a circuit of depth } O(L\cdot k)$$

> [!TIP]
> **This is the deep justification for chain-of-thought.** It is not a prompting trick that makes
> the model "try harder." It is a mechanism for **converting sequence length into computational
> depth** — the only way a fixed-depth architecture can perform variable-depth computation.

```
  Direct answer:           [prompt] ──L layers──► answer
                                                  ↑ one pass of depth L

  Chain of thought:        [prompt] ──L──► t₁ ──L──► t₂ ──L──► ... ──L──► answer
                                                  ↑ effective depth L·k
```

---

## 2. Chain-of-thought prompting

```
  STANDARD                                 CHAIN OF THOUGHT
  
  Q: Roger has 5 tennis balls. He buys     Q: (same)
     2 cans of 3 balls each. How many
     does he have now?                     A: Roger started with 5 balls.
  A: 11                                       2 cans × 3 balls = 6 balls.
                                              5 + 6 = 11.
  Q: The cafeteria had 23 apples...           The answer is 11.
  A: ???  ← one shot at the answer
                                           Q: The cafeteria had 23 apples...
                                           A: (model now generates its own steps)
```

**The original finding** (Wei et al., 2022), PaLM 540B on GSM8K:

| Method | Accuracy |
|---|---|
| Standard few-shot prompting | 17.9% |
| **Chain-of-thought prompting** | **56.9%** |

A 3× improvement from changing nothing but the prompt format.

**Zero-shot CoT** (Kojima et al., 2022) — appending "Let's think step by step" raised GSM8K
accuracy on InstructGPT (text-davinci-002) from 10.4% to 40.7%. No examples required.

> [!WARNING]
> **CoT only helps above a scale threshold.** Below roughly 10 B parameters, CoT prompting often
> *hurts* — smaller models generate plausible-looking but invalid reasoning chains and then follow
> them to a wrong answer. The model must be good enough that its intermediate steps are more often
> right than wrong.

> [!WARNING]
> **Modern caveat**: for models that were *trained* to reason (§5), explicit CoT prompting is
> unnecessary and can interfere. They produce reasoning by default, and adding "think step by step"
> may disrupt their trained format.

---

## 3. Scaling test-time compute

**The empirical law**: accuracy improves log-linearly in the amount of inference compute, across
several distinct mechanisms.

![pass@n versus number of samples on a log2 axis for per-sample success rates 0.01, 0.05 and 0.2](../assets/figures/best-of-n.svg)

*1 − (1 − p)ⁿ for independent samples and a perfect verifier. Each curve crosses 50% at n ≈ 0.69/p. Real best-of-n rises more slowly: errors are correlated, and verifiers are imperfect.*

| Method | Mechanism | Needs |
|---|---|---|
| **Longer CoT** | more serial steps | a model trained for it |
| **Self-consistency** | sample $n$ chains, take the majority answer | a task with a canonical answer |
| **Best-of-$n$** | sample $n$, pick the highest-scoring | a verifier or reward model |
| **Beam search over steps** | search the reasoning tree, pruning with a PRM | a process reward model |
| **MCTS / tree search** | explore and backtrack | a value estimate |
| **Self-refine** | generate, critique, revise | reliable self-critique |

### Self-consistency

Sample $n$ reasoning chains at temperature ~0.7, extract the final answers, take the mode.

> [!TIP]
> **Why it works**: there are many wrong paths to many *different* wrong answers, but the right
> reasoning tends to converge on the *same* right answer. Errors are diverse; correctness is
> concentrated.

GSM8K with PaLM 540B: 56.5% (greedy CoT) → **74.4%** ($n = 40$). An 18-point gain from sampling
alone.

> [!WARNING]
> Requires a well-defined final answer to vote on. It does not apply to open-ended generation,
> essays or code (though for code, "does it pass the tests" replaces voting entirely).

### Best-of-n with a verifier

If the verifier were perfect, accuracy would be $1 - (1-p)^n$ where $p$ is the per-sample
success rate:

| $p$ | $n=1$ | $n=4$ | $n=16$ | $n=64$ |
|---|---|---|---|---|
| 0.2 | 20% | 59% | 97% | 100% |
| 0.5 | 50% | 94% | 100% | 100% |

Real gains are much smaller because verifiers are imperfect, and because errors are correlated
(the model makes *the same* mistake repeatedly). Actual best-of-$n$ curves flatten well before
these numbers.

> [!WARNING]
> **The gap between pass@$n$ and best-of-$n$ is the verifier's quality.** pass@$n$ (is the right
> answer *anywhere* in $n$ samples?) rises quickly; best-of-$n$ (can we *find* it?) rises slowly.
> Closing that gap — building better verifiers — is where much of the current research effort sits.

### Process vs outcome reward models

| | **ORM** (outcome) | **PRM** (process) |
|---|---|---|
| Labels | was the final answer right? | is each *step* correct? |
| Label cost | cheap (automatic) | expensive (human or model annotation per step) |
| Signal | sparse, end-of-sequence | dense, per-step |
| Enables | best-of-$n$ | **step-level search and pruning** |
| Catches "right answer, wrong reasoning" | ❌ | ✅ |

Lightman et al. (2023) found PRMs substantially outperform ORMs for selecting solutions, and
released PRM800K (800k step-level labels) — the dataset that made process supervision practical.
PRMs also enable *search*: prune a branch as soon as a step is judged wrong, rather than waiting
for the end.

---

## 4. Prompting patterns

| Pattern | Idea | Use when |
|---|---|---|
| **Zero-shot CoT** | "Let's think step by step" | any reasoning task, no examples available |
| **Few-shot CoT** | demonstrate the reasoning format | you need a specific style or structure |
| **Self-consistency** | sample $n$, majority vote | the task has one canonical answer |
| **Least-to-most** | decompose into subproblems, solve in order | compositional problems |
| **Tree of Thoughts** | explore branches, evaluate, backtrack | puzzles, planning, search problems |
| **Self-refine** | generate → critique → revise | writing, code with a clear quality bar |
| **Program-of-Thought** | emit code, *execute* it, use the result | arithmetic, data manipulation |
| **ReAct** | interleave reasoning and tool calls | anything needing external information |

> [!TIP]
> **Program-of-Thought deserves emphasis.** Instead of reasoning about `17 × 24 + 391 / 23` in
> natural language — where the model must simulate arithmetic — emit Python and run it. The model
> handles what it's good at (translating a problem into a formal expression) and delegates what it's
> bad at (exact computation) to a machine that is perfect at it.

This reliably beats natural-language CoT on arithmetic-heavy problems, and it is why tool use
is central to modern agents. → [Agents & tool use](../06-applications/04-agents-and-tool-use.md)

---

## 5. Training models to reason: RLVR

The major shift since 2024: rather than *prompting* for reasoning, **train** for it with
reinforcement learning on verifiable rewards.

```
  1. Take a base or lightly-SFT'd model
  2. Sample many solutions to problems with checkable answers
     (math with known answers, code with unit tests)
  3. Reward = 1 if correct, 0 otherwise. No learned reward model.
  4. Optimize with GRPO/PPO
  5. The model discovers, on its own:
        - generating longer reasoning chains
        - checking its own work
        - backtracking when a path fails
        - trying alternative approaches
```

**The DeepSeek-R1 result** is the landmark. Applying pure RL to a base model — **no supervised
reasoning traces at all** — produced:

- Response length growing *spontaneously* from hundreds to thousands of tokens over training, as
  the model learned that thinking longer earns reward.
- Emergent self-verification ("wait, let me check that") and backtracking.
- Large gains on competition math and code benchmarks.

> [!TIP]
> **Why this is a big deal conceptually.** It contradicts the strong form of the "superficial
> alignment hypothesis" (→ [LLM architecture §7](01-llm-architecture.md#7-the-pipeline-from-base-model-to-product)).
> RL here does not merely select a style from the base model's repertoire — it produces problem-
> solving behaviour the base model would not exhibit at any sampling budget. **Style is superficial;
> reasoning trained this way is not.**

> [!WARNING]
> **The scope limitation is real.** RLVR needs a verifier, so it works for math, code, formal
> logic and anything with a checkable answer. It does not directly apply to essay quality, medical
> advice, or strategy. Whether reasoning learned on verifiable domains *transfers* to unverifiable
> ones is an active and genuinely open question — evidence so far suggests partial transfer.

### Distilling reasoning

Once you have a strong reasoning model, its traces are training data. SFT a smaller model on
them and much of the capability transfers — often more effectively than running RL on the small
model directly, because RL needs the model to *find* correct solutions before it can be rewarded
for them, and weak models rarely do.

> [!TIP]
> This creates a useful asymmetry: **RL is expensive and needs a capable base; distillation is
> cheap and spreads the result.** It is a large part of why strong small reasoning models appeared
> so quickly after the first large ones.

---

## 6. The inference-compute scaling law

Snell et al. (2024): for some problem distributions, **spending compute at inference beats
spending it on a bigger model.**

$$\text{Accuracy} \approx a + b\log(C_{\text{inference}})$$

| Setting | Better to… |
|---|---|
| Easy problems, plentiful | use a bigger model (one-shot is enough) |
| Hard problems, few of them | use a smaller model + heavy test-time search |
| Latency-critical | bigger model (test-time compute costs wall-clock) |
| Cost-critical, batch workload | smaller model + test-time compute |

> [!TIP]
> **The economic reframing.** Pretraining compute is a fixed, up-front cost amortized over all
> future queries. Inference compute is a marginal, per-query cost. This means you can now **choose
> how much to spend per question** — spend little on "what's 2+2", spend a lot on a research-grade
> problem. Model quality is no longer a single fixed point.

That flexibility is why "thinking budgets" (low/medium/high effort settings) have become a
standard product feature.

> [!WARNING]
> **Diminishing returns are steep.** The curve is logarithmic: going 1× → 10× compute gives a
> solid gain; 10× → 100× gives roughly the same absolute gain again, at 10× the cost. And there is
> a **ceiling** — if the model cannot solve a problem at all, no amount of sampling helps (pass@$n$
> saturates below 100%).

---

## 7. How much of this is "real" reasoning?

> [!WARNING]
> Worth engaging with honestly rather than dismissing either way.

**The case for scepticism:**

| Finding | Implication |
|---|---|
| Performance drops sharply on perturbed problems (changed names, added irrelevant clauses) — GSM-Symbolic | brittle, partly pattern-matched |
| CoT is often **unfaithful**: models reach an answer, then generate a rationalization that doesn't reflect the actual computation | the visible chain may not be the real mechanism |
| Benchmark contamination is pervasive | scores overstate genuine capability |
| Performance collapses past a complexity threshold on some synthetic tasks | not general algorithmic ability |

**The case against dismissal:**

| Finding | Implication |
|---|---|
| Models solve genuinely novel competition problems written after their cutoff | not pure memorization |
| Performance scales predictably with compute, as a real capability would | not a lookup table |
| RLVR discovers strategies absent from training data | genuine search over solution space |
| Transfer occurs: math RL improves code and vice versa | some shared underlying mechanism |

> [!TIP]
> **The most defensible position**: these models perform *a form of* reasoning that is genuinely
> useful, statistically grounded, and unlike human reasoning in important ways. They are
> simultaneously more capable than "stochastic parrot" implies and less robust than benchmark
> numbers suggest. **Both the hype and the dismissal are overclaims.**

> [!WARNING]
> **The faithfulness result is the one with practical consequences.** If a model's stated
> reasoning doesn't reflect its actual computation, you cannot audit its decisions by reading its
> chain of thought — which matters enormously for any high-stakes deployment, and for
> interpretability-based safety approaches.
> → [Safety](../08-safety-and-ethics/01-safety.md)

---

## 8. Implementation

Self-consistency, the highest value-per-line technique here:

```python
from collections import Counter
import re

def extract_answer(text: str):
    # adapt to your task's answer format
    m = re.findall(r"answer is\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    return m[-1].replace(",", "") if m else None

def self_consistency(model, question, n=16, temperature=0.7):
    prompt = f"{question}\n\nThink step by step, then state 'The answer is X'."
    # sample n independent chains; batch them if your backend supports it
    outputs = [model.generate(prompt, temperature=temperature, top_p=0.95)
               for _ in range(n)]
    answers = [a for a in map(extract_answer, outputs) if a is not None]
    if not answers:
        return None, 0.0, outputs
    answer, count = Counter(answers).most_common(1)[0]
    return answer, count / len(answers), outputs     # the ratio is a confidence proxy
```

> [!TIP]
> **The agreement ratio is a genuinely useful confidence signal** — better calibrated than the
> model's stated confidence, which is typically overconfident. If 15/16 chains agree, trust it. If
> 6/16 agree, escalate to a human or a stronger model.

Best-of-$n$ with a verifier:

```python
def best_of_n(model, problem, verifier, n=16):
    candidates = [model.generate(problem, temperature=0.8) for _ in range(n)]
    scored = [(verifier.score(problem, c), c) for c in candidates]
    return max(scored)[1]

# For code, the "verifier" is just running the tests — a perfect verifier.
def code_best_of_n(model, spec, tests, n=16):
    for cand in (model.generate(spec, temperature=0.8) for _ in range(n)):
        if run_tests(cand, tests):     # sandboxed!
            return cand
    return None
```

> [!WARNING]
> Execute model-generated code only in a sandbox with no network access, a filesystem jail, and a
> timeout. This is not optional.

---

## 9. Exercises

**Problem 1 — pass@$k$, a different success rate.** Using the unbiased estimator from
→ [Evaluation metrics §5](../07-evaluation/01-metrics.md#5-task-specific-metrics), compute
pass@1 and pass@8 for $n{=}24$ samples with $c{=}6$ correct. Compare with §3's
best-of-$n$ table (which used a *simplified* perfect-verifier formula
$1-(1-p)^n$ at $p{=}0.25$) — do the two approaches roughly agree at $n{=}8$?

<details><summary>Solution</summary>

pass@1 $= c/n = 6/24 = 0.25$.

pass@8 (unbiased estimator): $1-\binom{18}{8}/\binom{24}{8} = 0.9405$ — **94.1%**.

§3's simplified formula at $p{=}0.2$, $n{=}4$ gave 59%; extrapolating its $p{=}0.2$ row isn't a
direct match, but comparing the *mechanism*: at $p{=}0.25$ (matching this problem's empirical
pass@1), the simplified independent-samples formula gives $1-(1-0.25)^8=1-0.75^8=1-0.100=0.900$
— **90.0%**, close to but slightly below the unbiased estimator's 94.1%. The gap exists because
the two formulas answer slightly different questions: $1-(1-p)^n$ assumes $p$ is a known,
fixed, *continuous* success probability, while the unbiased pass@$k$ estimator works from
actual discrete counts ($c$ successes out of $n$ *observed* samples) and correctly accounts for
sampling without replacement when choosing which $k$ of the $n$ to imagine using — a subtlety
that matters more at small $n$.

</details>

**Problem 2 — GRPO advantage, an unbalanced group.** A group of $G{=}8$ attempts at a coding
problem gets rewards $(1,1,1,1,1,1,1,0)$ — 7 correct, 1 wrong (an easy problem). Using §5's GRPO
formula, compute the advantage for a correct attempt and for the wrong attempt. Compare the
*magnitude* of these advantages with §7's worked example (3 correct, 5 wrong out of 8) — which
group gives a stronger training signal, and why does that make sense given how "surprising" each
outcome is?

<details><summary>Solution</summary>

Mean $=7/8=0.875$; variance $=\frac18[7(1-0.875)^2+1(0-0.875)^2]=\frac18[7(0.0156)+0.7656]
=\frac18[0.1094+0.7656]=0.1094$; std $=\sqrt{0.1094}=0.3307$.

Correct: $A=(1-0.875)/0.3307=0.378$. Wrong: $A=(0-0.875)/0.3307=-2.646$.

§4's own worked example (3/8 correct) gave $A_{\text{correct}}=+1.29$, $A_{\text{wrong}}=-0.77$
— **much larger** advantage magnitude for correct answers there, and much *smaller* magnitude for
wrong answers, than in this easy-problem case (0.378 vs 1.29 for correct; $-2.646$ vs $-0.77$ for
wrong).

This makes sense: on the easy problem (7/8 correct), being correct is the *unsurprising* outcome
— it barely moves the mean, so it gets a small advantage — while the one wrong attempt is the
*surprising*, informative outlier and gets amplified into a large negative advantage. On the hard
problem (3/8 correct), the situation reverses: being correct is the surprising, valuable event
(large positive advantage) while being wrong is unsurprising (only mildly negative). GRPO's
normalization automatically concentrates the training signal on whichever outcome is rarer within
the group, without needing to know in advance which problems are "easy" or "hard."

</details>

**Problem 3 — chain-of-thought as computational depth.** Using §1's formalization
($\text{Transformer}+\text{CoT of length }k \approx \text{a circuit of depth }O(L\cdot k)$),
suppose a 24-layer model needs an estimated depth of 600 to reliably solve some hard combinatorial
puzzle in one shot. Roughly how many CoT tokens would be needed to reach that effective depth,
and does this help explain why very long reasoning traces (thousands of tokens) are sometimes
necessary for hard problems, rather than a slight prompting tweak?

<details><summary>Solution</summary>

$O(L\times k)\ge600$ with $L{=}24$ gives $k\ge600/24=25$ CoT tokens **at minimum**, by this
rough asymptotic argument — though in practice the constant hidden in the $O(\cdot)$, and the
fact that not every generated token performs "useful" additional computation (many are spent on
formatting, restating, or exploring dead ends), typically pushes the *actual* number of tokens
needed far higher than this idealized lower bound.

This does help explain the qualitative phenomenon: per §1, a fixed-depth Transformer's *one-shot*
computational depth is capped at $L$ (here 24) — nowhere near 600 — so no amount of clever
one-shot prompting can substitute for the missing depth; only *additional serial passes* (more
generated tokens, each a fresh forward pass with the previous output as new input) can supply it.
This is precisely why reasoning models trained via RLVR (§5) discover, on their own, that
generating substantially longer traces is necessary and rewarding for hard problems — they're
converting the one dimension they *can* scale at inference (sequence length) into the one they
structurally lack (circuit depth), exactly as §1's theorem predicts.

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | A Transformer has fixed circuit depth per token; some problems provably need more. |
| 2 | Chain-of-thought converts sequence length into computational depth — the mechanism, not a trick. |
| 3 | CoT requires scale: below ~10B params it often hurts, because intermediate steps are unreliable. |
| 4 | Self-consistency (sample $n$, majority vote) is the cheapest large gain, and gives a calibrated confidence signal. |
| 5 | Best-of-$n$ is bounded by verifier quality; the pass@$n$ / best-of-$n$ gap *is* that quality. |
| 6 | Process reward models beat outcome reward models and enable step-level search. |
| 7 | Program-of-Thought — emit code and execute it — beats natural-language arithmetic reliably. |
| 8 | RLVR (verifiable rewards) produces emergent long reasoning, self-checking and backtracking. |
| 9 | Inference compute is a second scaling axis: choose per-query how much to spend. |
| 10 | CoT is often unfaithful — do not treat a stated chain as an audit trail. |

---

## Further reading

- Wei et al., [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903) (2022).
- Wang et al., [*Self-Consistency Improves Chain of Thought Reasoning*](https://arxiv.org/abs/2203.11171) (2022).
- Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050) (2023) — process reward models.
- Yao et al., [*Tree of Thoughts*](https://arxiv.org/abs/2305.10601) (2023); Chen et al., [*Program of Thoughts*](https://arxiv.org/abs/2211.12588) (2022).
- Snell et al., [*Scaling LLM Test-Time Compute Optimally*](https://arxiv.org/abs/2408.03314) (2024).
- DeepSeek-AI, [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*](https://arxiv.org/abs/2501.12948) (2025).
- Merrill & Sabharwal, [*The Expressive Power of Transformers with Chain of Thought*](https://arxiv.org/abs/2310.07923) (2024).
- Turpin et al., *Language Models Don't Always Say What They Think* (2023) — CoT unfaithfulness.
- Mirzadeh et al., [*GSM-Symbolic*](https://arxiv.org/abs/2410.05229) (2024) — the brittleness evidence.

**Next** → [Diffusion language models](11-diffusion-language-models.md)
