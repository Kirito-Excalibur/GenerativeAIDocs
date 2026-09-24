# Societal Impact

> **Summary** — The consequences of generative AI outside the model: bias and representational
> harm, the unresolved copyright questions, labour effects, environmental cost, synthetic media and
> provenance, and the regulatory landscape. This page aims to state what is empirically known,
> what is genuinely contested, and where the honest answer is "we don't know yet" — rather than
> advocating a position.

**Prerequisites**: → [Safety](01-safety.md) · **Next**: Part IX → [Glossary](../09-reference/01-glossary.md)

---

## 1. Bias

Models learn the statistical patterns of their training data, including the ones we'd rather they
didn't.

📊 **Where bias enters:**

| Stage | Mechanism |
|---|---|
| **Data** | the internet over-represents some languages, regions, demographics and viewpoints |
| **Filtering** | quality classifiers trained on "good" text can systematically exclude dialects (e.g. AAVE) |
| **Annotation** | raters have demographics and views; preference data encodes them |
| **Alignment** | RLHF encodes the values of whoever wrote the rater guidelines |
| **Evaluation** | benchmarks built by a narrow group measure a narrow notion of quality |

📊 **Documented, measured effects:**

| Finding | Detail |
|---|---|
| Occupational stereotyping | image models over-produce men for "CEO", women for "nurse" |
| **Skin-tone skew** | text-to-image outputs skew lighter than population baselines |
| Name-based discrimination | identical CVs score differently by name |
| **Dialect prejudice** | Hofmann et al. (2024): models assign more negative traits to AAVE speakers — and this is *stronger* in RLHF'd models, where overt bias was reduced but covert bias was not |
| Language quality gap | performance and tokenization efficiency are much worse for low-resource languages (→ [Tokenization §6](../03-sequence-models/01-tokenization.md#6-the-multilingual-tax)) |

🧠 **The dialect-prejudice finding is the most important one here**, because it complicates the
standard mitigation story. Alignment training removed *stated* stereotypes while leaving *implicit*
associations intact — and possibly made them harder to detect. **Measuring the surface behaviour is
not sufficient.**

⚠️ **"Debiasing" is harder than it sounds**, for three structural reasons:

1. **Fairness definitions conflict.** Demographic parity, equalized odds and calibration are
   mathematically incompatible except in degenerate cases (Kleinberg et al., 2016). You must choose
   which to satisfy — a values decision, not a technical one.
2. **Bias vs accuracy.** If a real-world distribution is skewed, should the model reflect it or
   correct it? "Show me a nurse" — match current demographics, or show a balanced set? Both answers
   are defensible and neither is neutral.
3. **Whack-a-mole.** Fixing measured biases doesn't fix unmeasured ones, and can push them into
   less visible forms.

📊 What helps in practice: diverse and documented training data; **measuring** across demographic
slices rather than only in aggregate; involving affected groups in evaluation design; and being
explicit about which fairness definition you chose and why.

---

## 2. Copyright and training data

⚠️ **This is genuinely unsettled law, varying by jurisdiction, with major cases in progress.** What
follows describes the arguments, not a legal conclusion.

**The questions:**

| Question | Status |
|---|---|
| Is training on copyrighted work infringement? | 🔴 contested; litigation ongoing in multiple jurisdictions |
| Are model outputs derivative works? | 🔴 contested |
| Can AI-generated work be copyrighted? | 🟡 US Copyright Office: **no** without substantial human authorship |
| Does verbatim reproduction infringe? | 🟢 generally yes — this part is not controversial |
| Do artistic *styles* get protection? | 🟡 style is generally not copyrightable, but trade dress and right-of-publicity claims exist |

**The main arguments:**

*For training being permissible (fair use / TDM exception):*
- Training is transformative — the model learns statistical patterns, not stored copies.
- Analogous to a human learning from reading.
- The output typically does not substitute for the original work.
- Several jurisdictions have explicit text-and-data-mining exceptions (EU, Japan, Singapore).

*Against:*
- Commercial use at scale, without licence or compensation.
- Models can and do reproduce training data verbatim (→ [Security §4](02-security.md#4-training-data-extraction)).
- Outputs can compete directly with the source works in the same market.
- Copies are made during the training process itself.

📊 **What is happening regardless of the legal outcome**: licensing deals between model developers
and publishers, opt-out mechanisms (`robots.txt`, `ai.txt`, Do-Not-Train headers), compensation
funds, and models trained exclusively on licensed or public-domain data. The market is settling
some of this ahead of the courts.

🧠 **For practitioners, the actionable parts are clear even while the law isn't:** know your
training data's provenance; do not reproduce substantial verbatim portions of copyrighted work;
check the licence of any model you deploy commercially (many "open" model licences have use
restrictions); and understand that "it was on the internet" is not a licence.

---

## 3. Labour

📊 **What the evidence shows so far:**

| Finding | Source pattern |
|---|---|
| **Productivity gains are real and large in some tasks** | 20–55% faster on constrained coding tasks; 30–40% on writing tasks in controlled studies |
| **Gains are largest for *less* experienced workers** | several studies find the biggest uplift at the bottom of the skill distribution, compressing performance spread |
| Exposure is concentrated in white-collar work | the reverse of previous automation waves |
| Early effects visible in some freelance markets | measurable declines in demand for simple writing and basic graphic design |
| Net employment effect | ⚠️ **unknown** — too early, and confounded by macroeconomic conditions |

🧠 **The "less experienced workers gain most" result is the most interesting and the most replicated
so far.** The proposed mechanism: the model provides a competent baseline, which raises a novice's
floor substantially and raises an expert's ceiling only a little. If it holds, the labour-market
effect is compression rather than uniform displacement — which has very different policy
implications.

⚠️ **Be sceptical of all productivity numbers, in both directions.** Most studies use constrained
tasks with clear success criteria over short horizons. Real work involves ambiguity, coordination,
maintenance and accountability — precisely where measurement is hardest. Studies of longer-horizon
real work have found smaller and occasionally *negative* effects, including cases where
experienced developers were slower while believing they were faster.

📊 **The pattern from previous automation waves**: tasks are automated, not whole jobs; new
categories appear that are hard to predict; the transition period is where the real hardship is,
and its costs fall unevenly. Whether AI follows this pattern or breaks it is the open question, and
confident predictions in either direction are not supported by current evidence.

⚠️ **Data-labelling labour deserves mention.** RLHF and content moderation depend on large numbers
of human annotators, often in lower-income countries, sometimes reviewing disturbing content for
low pay. This is a documented and ongoing labour issue, not a hypothetical one.

---

## 4. Environment

🔢 **Training costs** (approximate, published estimates):

| Model class | Energy | CO₂e |
|---|---|---|
| BERT-base (2019) | 1.5 MWh | ~0.65 t |
| GPT-3 (2020) | ~1,287 MWh | ~552 t |
| A modern frontier run | 10,000–50,000+ MWh | thousands of tonnes |

🔢 **For scale**: 552 t CO₂e ≈ 120 cars driven for a year, or ~300 round-trip transatlantic
flights. Significant, but small relative to a data-centre industry that consumes 1–2% of global
electricity.

🧠 **Inference now dominates.** A model is trained once and serves billions of requests. At
sufficient scale, cumulative inference energy exceeds training energy by a wide margin — which is
why efficiency work (quantization, distillation, smaller compute-optimal-for-inference models)
matters environmentally as well as economically.

🔢 **Per-query energy**: roughly 0.3–3 Wh depending on model size and query length — order-of-
magnitude comparable to a few web searches, and far less than streaming video for a minute. The
aggregate matters more than the individual query.

📊 **What reduces impact:**
- Efficient architectures (MoE: capacity without proportional compute)
- Quantization and distillation (→ [Efficiency](../04-large-language-models/07-efficiency.md))
- Carbon-aware scheduling (train where and when the grid is clean)
- Better hardware utilization (MFU)
- **Not training a model you don't need** — fine-tuning or prompting an existing one is orders of
  magnitude cheaper

⚠️ **The honest framing**: AI's energy use is growing fast and is a legitimate concern, especially
its local impact on grids and water use for cooling. It is also currently a small fraction of
global emissions, and some applications (materials discovery, grid optimization, climate
modelling) may offset their own cost. Both the alarmist and the dismissive framings overstate
their case.

---

## 5. Synthetic media and provenance

📊 **Current capabilities:**

| Medium | State |
|---|---|
| Text | indistinguishable from human writing in most contexts |
| Images | photorealistic; residual artifacts in hands, text, reflections are shrinking |
| **Voice** | convincing cloning from seconds of reference audio |
| Video | rapidly improving; short clips can be convincing |
| Real-time video | emerging |

⚠️ **The documented harms are concrete, not speculative:**

| Harm | Status |
|---|---|
| **Non-consensual intimate imagery** | the largest measured category of deepfake harm by volume, overwhelmingly targeting women |
| **Voice-cloning fraud** | active; used in "relative in trouble" and CEO-fraud scams |
| Political disinformation | documented in multiple elections |
| **The liar's dividend** | real evidence dismissed as "probably AI" — arguably the larger systemic risk |
| Fraudulent documents and identity | growing |

🧠 **The liar's dividend is the underappreciated one.** The existence of convincing fakes lets
genuine evidence be dismissed. The harm isn't only that people believe false things — it's that
verification itself becomes harder, which erodes the evidentiary basis for accountability.

📊 **Detection does not work reliably, and this is unlikely to change.**

| Approach | Status |
|---|---|
| Classifier-based detection | ⚠️ poor generalization; fails on new generators; high false-positive rates |
| AI-text detectors | ❌ unreliable; **documented bias against non-native English writers** — do not use for academic penalties |
| **Watermarking** (statistical, e.g. SynthID) | ⚠️ helps, but removable by paraphrase or re-encoding |
| **C2PA / content credentials** | ⭐ signed provenance metadata; the most promising direction |

🧠 **Why provenance beats detection.** Detection asks "is this fake?" — an adversarial problem you
lose as generators improve. Provenance asks "can this be traced to a source?" — a cryptographic
problem with a real answer. C2PA attaches a signed chain of custody at capture and through edits.
⚠️ It only works if capture devices, editing software and platforms all support it, and metadata
can be stripped — so absence of provenance can't be treated as proof of fakery.

---

## 6. Access and concentration

⚠️ Frontier model training requires capital available to a small number of organizations.

| Concern | Detail |
|---|---|
| **Compute concentration** | training a frontier model costs $10^8$+ |
| Geographic concentration | development and benefit concentrated in a few countries |
| **Language inequity** | most languages are poorly served (→ [Tokenization §6](../03-sequence-models/01-tokenization.md#6-the-multilingual-tax)) |
| Data concentration | the web's content is unevenly distributed |
| API dependency | applications built on models that can change or be withdrawn |

📊 **Counterweights that are actually working**: capable open-weight models released within months
of frontier capability; efficiency gains that make small models increasingly good; distillation
spreading capability downward; and academic access programmes. The capability gap between frontier
and open models exists, but it has not widened as much as many predicted.

🧠 **The "open" terminology is imprecise and worth being careful about.** Most "open source" models
release *weights* but not training data, training code, or data-filtering pipelines — so they are
not reproducible and many carry use restrictions. **"Open weights" is the accurate term for most
of them.** Genuinely open models (OLMo, Pythia, some others) release everything and are far more
valuable for research precisely because they are reproducible.

---

## 7. Regulation

📊 A snapshot of the landscape (this section dates fastest — verify before relying on it):

| Jurisdiction | Approach |
|---|---|
| **EU (AI Act)** | risk-tiered: prohibited / high-risk / limited / minimal, plus obligations for general-purpose models above a compute threshold; phased application |
| **US** | sectoral and state-level rather than comprehensive federal; NIST AI RMF as voluntary guidance |
| **China** | registration and labelling requirements for generative services |
| **UK** | regulator-led, sector-specific |
| International | various summits, voluntary commitments, evaluation bodies |

**Recurring themes across regimes:**
- Transparency: disclose AI-generated content; document training data
- Risk assessment before deploying in high-stakes domains
- Human oversight for consequential decisions
- Model evaluation and red-teaming obligations
- Compute thresholds as a proxy for capability

⚠️ **The structural difficulties are real, not excuses**: technology outpaces legislation;
capability is hard to define legally; compute thresholds are a crude proxy that will age badly;
jurisdictional arbitrage is easy; and compliance burden falls hardest on small developers, which
can entrench incumbents.

---

## 8. What to actually do

📊 If you build with these systems, the concrete practices:

| Practice | Why |
|---|---|
| **Know your data's provenance** | licensing, bias, and contamination all trace back to it |
| **Measure across demographic slices**, not just in aggregate | aggregate scores hide disparate impact |
| **Disclose AI involvement** to users | increasingly a legal requirement, and always the right default |
| **Keep humans in consequential loops** | hiring, lending, medical, legal, criminal justice |
| **Provide appeal paths** | automated decisions need a route to a human |
| **Document limitations honestly** | model cards, system cards, known failure modes |
| **Use the smallest model that works** | cost, latency and energy all improve |
| **Respect opt-outs** | `robots.txt`, licence terms, expressed preferences |
| **Think about your worst case** | who is harmed if this fails badly, and how much? |

🧠 **The most useful single question**: *who bears the cost when this system is wrong?* If the
answer is "someone other than the people who deployed it", that asymmetry is where the ethical
weight sits, and it is where extra care is warranted.

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Bias enters at every stage — data, filtering, annotation, alignment, evaluation. |
| 2 | Alignment can reduce *overt* bias while leaving *covert* bias intact, or make it harder to see. |
| 3 | Fairness definitions are mathematically incompatible; choosing one is a values decision. |
| 4 | Copyright is genuinely unsettled. Verbatim reproduction is clearly infringing; training is contested. |
| 5 | Productivity gains are real and largest for *less* experienced workers — compression, not uniform displacement. |
| 6 | Be sceptical of productivity numbers in both directions; most studies use short, constrained tasks. |
| 7 | Inference energy exceeds training energy at scale — efficiency is an environmental lever too. |
| 8 | AI-text detectors are unreliable and biased against non-native speakers. Don't use them punitively. |
| 9 | **Provenance (C2PA) beats detection** — cryptography beats an adversarial classification game. |
| 10 | "Open weights" ≠ "open source". Most released models aren't reproducible. |
| 11 | Ask who bears the cost when the system is wrong. Asymmetry is where the ethical weight is. |

---

## Further reading

- Bender et al., *On the Dangers of Stochastic Parrots* (2021).
- Hofmann et al., *AI Generates Covertly Racist Decisions about People Based on Their Dialect* (2024).
- Kleinberg, Mullainathan & Raghavan, *Inherent Trade-Offs in the Fair Determination of Risk Scores* (2016).
- Strubell et al., *Energy and Policy Considerations for Deep Learning in NLP* (2019); Luccioni et al., *Estimating the Carbon Footprint of BLOOM* (2022).
- Noy & Zhang, *Experimental Evidence on the Productivity Effects of Generative AI* (2023); Peng et al., *The Impact of AI on Developer Productivity* (2023).
- Liang et al., *GPT Detectors Are Biased Against Non-Native English Writers* (2023).
- Chesney & Citron, *Deep Fakes: A Looming Challenge for Privacy, Democracy, and National Security* (2019) — the liar's dividend.
- Mitchell et al., *Model Cards for Model Reporting* (2019).
- The C2PA specification — `c2pa.org`.

**Next** → Part IX: [Glossary](../09-reference/01-glossary.md)
