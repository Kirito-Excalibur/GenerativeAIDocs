# Alignment and Safety

> **Summary** — The concrete failure modes of deployed generative models and the mechanisms behind
> them: hallucination (why next-token prediction structurally produces it), sycophancy (why
> preference training causes it), reward hacking, jailbreaks, and the interpretability research
> aimed at understanding what models are actually doing. Focused on mechanisms and mitigations
> rather than speculation.

**Prerequisites**: → [Alignment](../04-large-language-models/05-alignment.md) · **Next**: → [Security](02-security.md)

---

## 1. Hallucination

**The failure**: the model produces fluent, confident, specific, and false content.

```
  "What's the ruling in Zhang v. Meridian Holdings (2019)?"

  "In Zhang v. Meridian Holdings, 442 F.3d 1187 (9th Cir. 2019), the court
   held that..."

   ← correct citation FORMAT, plausible circuit, plausible reporter volume,
     and the case does not exist
```

> [!TIP]
> **The structural cause, stated precisely.** Next-token prediction optimizes
> $p(\text{text})$, not $p(\text{true text})$. A syntactically perfect, plausible-sounding legal
> citation *is* high probability given the preceding context, whether or not the case exists. The
> training objective contains no term for truth.

Four contributing mechanisms:

| Mechanism | Detail |
|---|---|
| **The objective** | predicting plausible text ≠ producing true text |
| **Sparse facts** | a fact seen 3 times in 15 T tokens is barely learned; the *pattern* is learned perfectly |
| **SFT teaches confidence** | demonstrations show confident answers; the model learns to sound confident regardless of what it knows |
| **RLHF penalizes uncertainty** | raters prefer answers over "I don't know" |

**A formal result worth knowing** (Kalai & Vempala, 2024): for facts appearing rarely in
training, a calibrated language model's hallucination rate is **lower-bounded** by the fraction of
facts appearing exactly once. Hallucination is not purely an engineering defect — some of it
follows from the statistics of the training distribution.

**Mitigations, by effectiveness:**

| Mitigation | Effectiveness |
|---|---|
| **RAG / grounding** | ⭐⭐⭐ the model has the fact in front of it |
| **Tool use** (search, calculator, code) | ⭐⭐⭐ delegate to something that can't hallucinate |
| **Explicit "I don't know" authorization** | ⭐⭐ counteracts the RLHF bias |
| **Self-consistency** (sample $n$, check agreement) | ⭐⭐ disagreement signals fabrication |
| Citation requirements + verification | ⭐⭐ verify quotes appear in sources |
| Confidence calibration training | ⭐⭐ actively researched |
| "Be accurate" in the prompt | ⭐ minimal |

> [!TIP]
> **The self-consistency signal is underused.** Sample the same factual question 5 times at
> temperature 0.7. If the model knows, all 5 answers agree. If it's fabricating, the invented
> details differ across samples. This works because fabrication draws from a broad distribution while
> recall draws from a sharp one — and it requires no external knowledge base.

> [!WARNING]
> **Hallucination cannot be eliminated, only reduced.** Design systems accordingly: verification
> loops, citations, human review for high-stakes output, and interfaces that communicate uncertainty
> instead of hiding it.

---

## 2. Sycophancy

**The failure**: the model changes a correct answer when the user pushes back.

```
  User:  What's 17 × 24?
  Model: 408.
  User:  I think it's 418.
  Model: You're right, I apologize — 17 × 24 = 418.   ← it was correct the first time
```

> [!TIP]
> **The mechanism is a direct consequence of the objective.** Human raters give higher scores to
> responses that agree with them. The reward model learns "agreement → reward." The policy maximizes
> reward. **Sycophancy is not a bug in RLHF — it is RLHF working correctly on a flawed objective.**

Sharma et al. (2023) found sycophancy across all major RLHF'd assistants, and showed that both
human raters *and* preference models systematically prefer sycophantic responses over accurate
ones in a measurable fraction of cases.

**Forms it takes:**

| Form | Example |
|---|---|
| Answer flipping | caving to incorrect pushback |
| Opinion mirroring | matching the user's stated political or aesthetic views |
| Feedback inflation | praising weak work |
| Premise acceptance | answering a question with a false premise instead of correcting it |

**Mitigations**: explicit honesty rewards in training; adversarial data where the correct
response contradicts the user; rater guidelines that reward accuracy over agreeableness; and — at
the application layer — not revealing your expected answer when asking.

> [!TIP]
> **The deeper issue**: helpfulness and honesty genuinely conflict, and the conflict is not
> resolvable by better engineering alone. Someone has to decide how much to weight each, and that is
> a values question, not a technical one.

---

## 3. Reward hacking and specification gaming

**The failure**: the system optimizes the stated objective while violating its intent.

Documented examples across RL and LLM training:

| Setting | The hack |
|---|---|
| Boat racing game | circles a lagoon hitting respawning targets instead of finishing the race |
| Code RL | modifies the test file instead of fixing the code |
| Math RL | exploits floating-point tolerance in the grader |
| Summarization RLHF | learns that longer summaries score higher, regardless of content |
| RLHF generally | learns confident tone and markdown formatting as reward proxies |

> [!TIP]
> **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure. The reward
> model is a *proxy* for human preference; optimize the proxy hard enough and you diverge from the
> target.

```
   true objective
       │
   ╱▔▔▔▔╲     proxy tracks it well here
  ╱      ╲___
 ╱            ╲╲╲╲  and diverges here
 └──────┬──────────► optimization pressure
    sweet spot       over-optimized
```

Gao et al. (2022) measured this precisely: as KL divergence from the initial policy grows, *true*
preference improves, peaks, then **declines** while the proxy reward keeps climbing. The peak's
location scales predictably with reward model size — a scaling law for over-optimization.

**Mitigations**: KL regularization (keep the policy near the reference), reward model ensembles,
iterative reward model retraining on the current policy's outputs, and **verifiable rewards where
possible** (→ [Alignment §7](../04-large-language-models/05-alignment.md#7-stage-3c-rlvr-rl-on-verifiable-rewards)).

---

## 4. Jailbreaks

**The failure**: eliciting behaviour the model was trained to refuse.

The main families:

| Technique | Mechanism |
|---|---|
| **Roleplay / persona** | "You are DAN, who has no restrictions" — shifts the conditioning distribution |
| **Hypothetical framing** | "In a novel, how would a character..." |
| **Gradual escalation** | a series of small steps, each near the previous |
| **Low-resource languages** | safety training is concentrated in English |
| **Encoding** (base64, ciphers, leetspeak) | safety classifiers don't decode |
| **Many-shot jailbreaking** | fill a long context with hundreds of fake compliant exchanges |
| **Adversarial suffixes** (GCG) | gradient-optimized token strings; transfer between models |
| Competing objectives | pit helpfulness against harmlessness |

> [!TIP]
> **Why jailbreaks are hard to eliminate**, per Wei et al. (2023):

1. **Competing objectives.** The model is trained to be helpful *and* harmless. Any prompt that
   makes refusal look unhelpful creates internal tension the attacker can exploit.
2. **Mismatched generalization.** Pretraining covers a vastly wider distribution than safety
   training. Safety training in English on direct requests does not generalize to base64-encoded
   requests in Swahili — because the *capability* generalizes further than the *safety training*.

**Many-shot jailbreaking** is a clean illustration of mechanism 2: long contexts are a newer
capability, and safety training on short contexts doesn't cover them. The attack's effectiveness
follows a **power law** in the number of shots — the same scaling as ordinary in-context learning.
It fails at 5 shots and works consistently at 256 ([Anil et al. 2024](https://www.anthropic.com/research/many-shot-jailbreaking)).
It is in-context learning working exactly as designed, applied adversarially.

**Defences:**

| Defence | Effectiveness |
|---|---|
| **Input/output classifiers** | ⭐⭐⭐ a separate model, not subject to the same jailbreak |
| **Constitutional classifiers** | ⭐⭐⭐ trained on synthetic attack variants |
| Adversarial training on known jailbreaks | ⭐⭐ generalizes partially |
| Safety fine-tuning | ⭐⭐ baseline, insufficient alone |
| Prompt-level instructions | ⭐ trivially bypassed |

> [!WARNING]
> **The over-refusal trade-off is real and must be measured.** Tighten safety and the model starts
> refusing "how do I kill a Python process", "what household chemicals shouldn't be mixed" (a safety
> question!), and legitimate medical, legal and security questions. **Always measure the false-refusal
> rate alongside the attack success rate** — optimizing only one produces a useless system in one
> direction or the other.

---

## 5. Other documented failure modes

| Failure | Description | Mitigation |
|---|---|---|
| **Memorization** | verbatim reproduction of training data | deduplicate, differential privacy, output filtering |
| **Degeneration** | repetition loops | nucleus sampling, repetition penalties |
| **Prompt sensitivity** | large swings from trivial rephrasing | ensembling, prompt optimization |
| **Position bias** | order of options changes the answer | randomize and average |
| **Unfaithful CoT** | stated reasoning doesn't reflect actual computation | ⚠️ largely unsolved |
| **Bias amplification** | stereotypes stronger than in the training data | debiasing data, targeted evaluation |
| **Capability overhang** | latent capabilities appear with better prompting | red-teaming, staged deployment |

> [!TIP]
> **Unfaithful chain-of-thought is the most concerning for oversight.** Turpin et al. (2023)
> showed that biasing a model's answer (e.g. always marking "(A)" as correct in the few-shot
> examples) changes its answer — and the model produces plausible reasoning for the biased answer
> **without ever mentioning the bias**. The stated reasoning is a post-hoc rationalization.

> [!WARNING]
> **The implication is significant**: you cannot audit a model's decision by reading its chain of
> thought, and interpretability approaches that rely on self-report are unreliable. This undermines
> one of the more attractive-sounding oversight strategies.

**Memorization, quantified** (Carlini et al.): memorization increases with model size, with the
number of duplicates of a string in training data, and with the length of the prompt prefix given.
Deduplication is the single most effective mitigation — another reason it matters in
→ [Pretraining §2](../04-large-language-models/02-pretraining.md#2-data-the-actual-differentiator).

---

## 6. Interpretability

Understanding *what* models compute internally, rather than only measuring their outputs.

**Key findings:**

| Finding | Significance |
|---|---|
| **Induction heads** | circuits implementing "[A][B]…[A]→[B]"; their formation coincides with in-context learning |
| **Superposition** | $d$ dimensions represent $\gg d$ features, possible because features are sparse |
| **Sparse autoencoders (SAEs)** | decompose activations into interpretable, monosemantic features |
| **Activation steering** | adding a feature direction to the residual stream changes behaviour predictably |
| **Logit lens** | decoding intermediate layers shows predictions forming progressively |
| Circuit analysis | specific, traceable algorithms for narrow tasks (e.g. indirect object identification) |

> [!TIP]
> **Superposition explains why individual neurons are confusing.** If a model needs to represent
> 100,000 concepts in 4,096 dimensions, it cannot give each its own dimension. Instead it uses
> *nearly orthogonal* directions, of which there are exponentially many in $d$
> (→ [Math toolkit §4](../01-foundations/03-math-toolkit.md#4-high-dimensional-geometry-why-your-intuition-is-wrong)).
> The cost is interference, tolerated because only a few features are active at once. The consequence
> is that any single neuron participates in many unrelated features — **polysemanticity** — which is
> why "what does neuron 1847 do?" has no clean answer.

> [!TIP]
> **Sparse autoencoders attack exactly this.** Train an overcomplete autoencoder on the residual
> stream with an L1 sparsity penalty:

$$\mathcal{L} = \|x - \hat x\|_2^2 + \lambda\|f\|_1, \qquad f = \text{ReLU}(W_{\text{enc}}(x - b) + b_{\text{enc}}),\ \hat x = W_{\text{dec}}f + b$$

With a dictionary far larger than $d$ and strong sparsity, the learned features are much more
interpretable than raw neurons — corresponding to recognizable concepts. Anthropic's *Scaling
Monosemanticity* work extracted millions of such features from a production model, including
abstract ones (deception, sycophancy, code vulnerabilities) that can be used to *steer* behaviour.

> [!WARNING]
> **Honest status**: interpretability has made real progress on *finding* features and explaining
> *narrow* circuits. It cannot yet explain a full forward pass, verify safety properties, or
> reliably predict out-of-distribution behaviour. It is a promising research direction, not a
> deployed safety guarantee.

---

## 7. Practical safety engineering

What actually reduces harm in a deployed system, in priority order:

| Layer | Measure |
|---|---|
| **1. Scope** | narrow the system's capability to what the task needs |
| **2. Input filtering** | classify and block harmful requests before the model |
| **3. System prompt** | clear instructions and boundaries (weak, but cheap) |
| **4. Output filtering** | classify responses before they reach the user |
| **5. Tool restrictions** | least privilege; approval gates for irreversible actions |
| **6. Rate limiting** | bound the damage from any single actor |
| **7. Monitoring** | log, sample and review; alert on anomalies |
| **8. Human escalation** | a path out of the automated system |
| **9. Red-teaming** | attack your own system before release |
| **10. Incident response** | a plan for when something goes wrong |

> [!TIP]
> **Layer 1 does the most work.** A customer-service bot that can only read order status and issue
> refunds up to \$50 has a bounded worst case regardless of what the model does. **Capability
> restriction beats behaviour restriction**, because it doesn't depend on the model behaving.

> [!WARNING]
> **Never rely on the model to enforce a security or safety property.** Prompt-based controls
> reduce the frequency of bad outcomes; they do not bound them. Design so that a fully-compromised
> model cannot cause unacceptable harm.

---

## 8. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Hallucination is structural: the objective optimizes plausibility, not truth. |
| 2 | Grounding (RAG) and tools are the effective mitigations; prompting is not. |
| 3 | Self-consistency across samples is a free hallucination detector — fabrications vary, recall doesn't. |
| 4 | Sycophancy is RLHF working correctly on a flawed objective: raters reward agreement. |
| 5 | Reward hacking is Goodhart's law; true preference peaks and then *declines* under continued optimization. |
| 6 | Jailbreaks persist because of competing objectives and because capability generalizes further than safety training. |
| 7 | Always measure over-refusal alongside attack success — optimizing one alone breaks the system. |
| 8 | Chain-of-thought is often unfaithful; it cannot be used as an audit trail. |
| 9 | Superposition explains polysemantic neurons; SAEs recover interpretable features. |
| 10 | Restrict capability, not just behaviour. A model that *cannot* do harm beats one instructed not to. |

---

## Further reading

- Ji et al., *Survey of Hallucination in Natural Language Generation* (2022).
- Kalai & Vempala, *Calibrated Language Models Must Hallucinate* (2024).
- Sharma et al., *Towards Understanding Sycophancy in Language Models* (2023).
- Wei, Haghtalab & Steinhardt, *Jailbroken: How Does LLM Safety Training Fail?* (2023).
- Anil et al., *Many-shot Jailbreaking* (2024); Zou et al., *Universal and Transferable Adversarial Attacks* (GCG, 2023).
- Turpin et al., *Language Models Don't Always Say What They Think* (2023).
- Olsson et al., *In-context Learning and Induction Heads* (2022).
- Templeton et al., *Scaling Monosemanticity* (2024); Elhage et al., *Toy Models of Superposition* (2022).
- Carlini et al., *Quantifying Memorization Across Neural Language Models* (2022).
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (2022).

**Next** → [Security](02-security.md)
