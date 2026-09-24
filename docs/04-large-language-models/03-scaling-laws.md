# Scaling Laws

> **Summary** — Language model loss follows a smooth power law in model size, dataset size and
> compute over many orders of magnitude. This predictability is why frontier labs can commit
> \$100M to a training run before it starts: you fit a curve on small models and extrapolate.
> This page derives the compute-optimal allocation (Chinchilla), works through real budget
> calculations, and explains why everyone now deliberately trains past the "optimal" point.

**Prerequisites**: → [Pretraining](02-pretraining.md) · **Next**: → [Fine-tuning & PEFT](04-finetuning-peft.md)

---

## 1. The empirical finding

Loss decreases as a **power law** in each of the three resources, when the others are not limiting:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

with (Kaplan et al., 2020): $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$.

![Five U-shaped curves of predicted loss versus model size, one per compute budget from 1e19 to 1e23 FLOPs, each with its minimum marked](../assets/figures/scaling-isoflops.svg)

*IsoFLOP curves from the Chinchilla parametric fit L = E + A/N^α + B/D^β with the paper's published constants, where each curve holds C = 6ND fixed. For any budget there is a best model size: too small and it can't learn; too big and it sees too little data. Joining the minima traces out the power law. (See §2 for why the published constants disagree with the paper's 20-tokens-per-parameter rule.)*

> [!TIP]
> **Why this is remarkable.** Nothing in deep learning theory predicts a clean power law. It holds
> across seven orders of magnitude of compute, across architectures, across modalities (text, image,
> video, math), and across languages. It is one of the most robust empirical regularities in machine
> learning — and nobody fully understands why.

> [!WARNING]
> **Power laws are ruthless, though.** $\alpha_N = 0.076$ means a **10× bigger model** gives
> $10^{0.076} = 1.19\times$ lower loss — a **19% reduction**. To halve the loss you need
> $2^{1/0.076} \approx 10^{2.7} \approx 500\times$ the parameters. Progress is real but expensive,
> and diminishing returns are built in.

---

## 2. Kaplan vs Chinchilla: the correction that changed the field

### Kaplan et al. (2020)

Conclusion: **model size matters much more than data.** Given 10× more compute, make the model
~5.5× bigger and the dataset only ~1.8× bigger.

Consequence: GPT-3 (175 B parameters) was trained on only 300 B tokens — a ratio of **1.7 tokens
per parameter**.

### Hoffmann et al. (2022): "Chinchilla"

> [!WARNING]
> Kaplan's experiments held the learning-rate schedule fixed regardless of the training length, so
> short runs were systematically under-trained. Correcting this changes the conclusion **completely**.

The Chinchilla parametric fit:

$$\boxed{\;L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}\;}$$

with fitted values:

| Parameter | Value | Meaning |
|---|---|---|
| $E$ | 1.69 | **irreducible loss** — the entropy of natural language itself |
| $A$ | 406.4 | model-size coefficient |
| $\alpha$ | 0.34 | model-size exponent |
| $B$ | 410.7 | data coefficient |
| $\beta$ | 0.28 | data exponent |

> [!TIP]
> **The $E$ term matters conceptually**: no matter how much compute you spend, loss cannot go below
> ~1.69 nats/token. Language is genuinely stochastic — many continuations are valid. Any claim of
> unbounded improvement from scale alone contradicts this fit.

**Deriving the compute-optimal allocation.** Minimize $L(N,D)$ subject to $C = 6ND$.

Substitute $D = C/(6N)$:

$$L(N) = E + AN^{-\alpha} + B\left(\frac{C}{6N}\right)^{-\beta} = E + AN^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta}$$

$$\frac{dL}{dN} = -\alpha A N^{-\alpha-1} + \beta B\,6^\beta C^{-\beta} N^{\beta-1} = 0$$

$$\alpha A N^{-\alpha-1} = \beta B\,6^\beta C^{-\beta}N^{\beta-1}$$

$$N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\cdot\frac{C^\beta}{6^\beta}
\quad\Longrightarrow\quad N_{\text{opt}} \propto C^{\frac{\beta}{\alpha+\beta}}$$

With $\alpha = 0.34$, $\beta = 0.28$: $\frac{\beta}{\alpha+\beta} = \frac{0.28}{0.62} = 0.452$,
and symmetrically $D_{\text{opt}} \propto C^{0.548}$.

$$\boxed{\;N_{\text{opt}} \propto C^{0.45}, \qquad D_{\text{opt}} \propto C^{0.55}\;}$$

These are close enough to $0.5$ that the practical rule is to **scale $N$ and $D$ in roughly
equal proportion**. The paper's other two estimation methods (fitting the minima of many
training runs directly) put the constant at about **20 tokens per parameter**:

$$\boxed{\;\textbf{scale } N \textbf{ and } D \textbf{ equally: } D \approx 20N\;}$$

> [!WARNING]
> **A known inconsistency worth knowing about.** The exponents above follow from the paper's
> published fit, but its *constants* do not reproduce the 20:1 ratio: plugging $A$, $B$, $E$ into
> the optimization gives 30–80 tokens per parameter, rising with budget (see the IsoFLOP figure
> above). [Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102) re-fit the paper's data and
> found that the published parametric estimates are inconsistent with the other two methods and
> have implausibly narrow confidence intervals; their corrected fit agrees with the ~20:1 rule. Use
> 20 tokens/param as the rule of thumb, and treat the published $A$, $B$, $E$ as illustrative.

### The Chinchilla demonstration

Gopher (280 B params, 300 B tokens) vs Chinchilla (70 B params, 1.4 T tokens) — **identical
compute budget**:

| | Gopher | Chinchilla |
|---|---|---|
| Parameters | 280 B | **70 B** (4× smaller) |
| Tokens | 300 B | **1.4 T** (4.7× more) |
| Tokens/param | 1.1 | **20** |
| MMLU | 60.0% | **67.6%** |
| Inference cost | 4× | **1×** |

**A 4× smaller model that is better AND 4× cheaper to serve.** This result immediately
reorganized the field.

**How badly was GPT-3 mis-allocated?** At $C = 3.14\times10^{23}$ FLOPs, the $D = 20N$ rule gives
$N = \sqrt{C/120} \approx 51$ B parameters and $D \approx 1.0$ T tokens. GPT-3 used 175 B parameters
on 300 B tokens: **~3.4× too many parameters and ~3.4× too few tokens** for its budget. (Check:
$6 \times 51\text{B} \times 1.02\text{T} \approx 3.1\times10^{23}$ ✓.)

---

## 3. Working the numbers

**Problem 1: I have \$1M of H100 time. What should I train?**

*Step 1 — convert money to FLOPs.* \$1M at \$2/GPU-hour = 500,000 GPU-hours. At 400 TFLOP/s
effective:

$$C = 5\times10^5 \times 3600 \times 4\times10^{14} = 7.2\times10^{23}\text{ FLOPs}$$

*Step 2 — apply Chinchilla.* Substitute $D = 20N$ into $C = 6ND$:

$$C = 6N(20N) = 120N^2 \quad\Longrightarrow\quad N = \sqrt{\frac{C}{120}} = \sqrt{\frac{7.2\times10^{23}}{120}} = \sqrt{6\times10^{21}} = 7.75\times10^{10}$$

$$\boxed{\;N \approx 77\text{ B parameters}, \qquad D = 20N \approx 1.55\text{ T tokens}\;}$$

*Check*: $6 \times 7.75\times10^{10}\times1.55\times10^{12} = 7.2\times10^{23}$ ✓

**Problem 2: but I will serve 10 trillion tokens. Now what?**

Total lifetime cost = training + inference:

$$C_{\text{total}} = 6ND_{\text{train}} + 2ND_{\text{inference}}$$

With $D_{\text{inference}} = 10^{13}$, inference costs $2N\times10^{13} = 2\times10^{13}N$, while
training costs $6N\times1.55\times10^{12} = 9.3\times10^{12}N$. **Inference is already 2× training.**

Minimizing total cost pushes $N$ down and $D_{\text{train}}$ up. Sardana et al. (2023) work this out
formally; the qualitative answer is dramatic:

| Expected inference volume | Optimal $D/N$ ratio |
|---|---|
| ~0 (research model) | 20 (Chinchilla) |
| 100 B tokens | ~50 |
| 1 T tokens | ~150 |
| 10 T+ tokens | **500–2000** |

**This is exactly what the industry does.** LLaMA-3 8B: 15 T tokens = **1875 tokens/parameter**,
94× Chinchilla. It is "compute-inefficient" for training and dramatically cheaper for everyone who
uses it.

> [!TIP]
> **The generalizable lesson**: Chinchilla answers "how do I get the lowest loss for a fixed
> training budget?" That is almost never the actual business question. The real question is "how do I
> get the best capability per dollar over the model's lifetime?", and its answer is a much smaller,
> much longer-trained model.

---

## 4. The scaling law zoo

Power laws show up nearly everywhere someone has looked:

| Domain | Scaling behaviour | Source |
|---|---|---|
| Language modelling | $L \propto C^{-0.05}$ | Kaplan 2020, Hoffmann 2022 |
| Image generation | similar exponents | Henighan et al. 2020 |
| Video, math, code | similar exponents | Henighan et al. 2020 |
| Multimodal contrastive (CLIP) | power law in data and params | Cherti et al. 2022 |
| **Transfer / fine-tuning** | power law in fine-tune data | Hernandez et al. 2021 |
| Reward model accuracy | log-linear in params | Bai et al. 2022 |
| **Data-constrained training** | repeating data ≈ new data for ~4 epochs | Muennighoff et al. 2023 |
| Sparse/MoE models | power law in *active* params, with a capacity bonus | Clark et al. 2022 |
| **Inference-time compute** | log-linear in samples/thinking tokens | Snell 2024, Brown 2024 |
| Vocabulary size | compute-optimal $V$ grows sub-linearly with $N$ | Tao et al. 2024 |

> [!TIP]
> **The data-constrained result deserves emphasis** (Muennighoff et al., 2023). If you are out of
> unique tokens, repeating data is *nearly* as good as new data for up to **~4 epochs**; value decays
> after that and becomes negligible past ~16 epochs. This matters because frontier runs are
> approaching the limit of high-quality public text.

> [!TIP]
> **The inference-compute law is the most important recent addition.** Accuracy improves
> predictably with the number of samples drawn or reasoning tokens generated — a *second* axis of
> scaling that doesn't require retraining. It is the theoretical basis for reasoning models.
> → [Reasoning](10-reasoning.md)

---

## 5. Where scaling laws break

> [!WARNING]
> Do not over-extrapolate. Known limits and caveats:

| Limit | Detail |
|---|---|
| **Irreducible loss $E$** | the fit itself says you asymptote at ~1.69 nats. You cannot scale to zero loss. |
| **Data exhaustion** | high-quality public text is ~$10^{13}$–$10^{14}$ tokens; frontier runs are within ~1 order of magnitude. |
| **Loss ≠ capability** | a 10% loss reduction may produce a huge or a negligible change in downstream accuracy. The mapping is not predictable. |
| **Coefficients are setup-specific** | $A, B, E$ depend on tokenizer, data mix, and architecture. Refit for your own setting. |
| **Optimizer/hyperparameter assumptions** | the laws assume near-optimal LR, batch size and schedule. A badly-tuned run does not follow them. |
| **Downstream-task laws are noisier** | benchmark accuracy vs compute is much less clean than loss vs compute. |

> [!TIP]
> **"Loss ≠ capability" is the one that matters most in practice.** Scaling laws predict the *loss*
> beautifully and say essentially nothing about whether your model will be able to do multi-step
> arithmetic. Labs run downstream evals during training precisely because loss alone is not a
> sufficient statistic.

---

## 6. Fitting your own scaling law

The practical procedure, used before every large run:

```python
import numpy as np
from scipy.optimize import curve_fit

# Train a ladder of small models. Vary N and D; measure final loss.
# e.g. N in {10M, 30M, 100M, 300M, 1B}, D in {0.5, 1, 2, 4} x Chinchilla
runs = [
    # (N params, D tokens, final loss)
    (1.0e7, 2.0e8, 3.85), (1.0e7, 8.0e8, 3.61),
    (3.0e7, 6.0e8, 3.42), (3.0e7, 2.4e9, 3.21),
    (1.0e8, 2.0e9, 3.05), (1.0e8, 8.0e9, 2.88),
    (3.0e8, 6.0e9, 2.74), (3.0e8, 2.4e10, 2.60),
    (1.0e9, 2.0e10, 2.48), (1.0e9, 8.0e10, 2.36),
]
N, D, L = map(np.array, zip(*runs))

def chinchilla(X, E, A, B, alpha, beta):
    n, d = X
    return E + A * n ** (-alpha) + B * d ** (-beta)

p0 = [1.7, 400.0, 400.0, 0.34, 0.28]
params, _ = curve_fit(chinchilla, (N, D), L, p0=p0, maxfev=100000)
E, A, B, alpha, beta = params
print(f"E={E:.3f} A={A:.1f} B={B:.1f} alpha={alpha:.3f} beta={beta:.3f}")

# Now predict the big run, and derive the optimal split for a target budget
def optimal_split(C, alpha, beta, A, B):
    # N_opt from the stationarity condition derived in section 2
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    a = beta / (alpha + beta)
    N_opt = G * (C / 6) ** a
    return N_opt, C / (6 * N_opt)

C_target = 1e24
N_opt, D_opt = optimal_split(C_target, alpha, beta, A, B)
print(f"For C={C_target:.0e}: N={N_opt/1e9:.1f}B params, D={D_opt/1e12:.2f}T tokens")
print(f"Predicted loss: {chinchilla((N_opt, D_opt), *params):.3f}")
```

> [!WARNING]
> **Fit in log space** and weight runs equally in log-loss, not linear loss — otherwise the
> large-loss small runs dominate the fit. Also: **use runs that are each individually well-tuned**
> (correct LR schedule for that run's length). The original Kaplan/Chinchilla discrepancy came
> entirely from getting this wrong.

Frontier labs run "scaling ladders" of 10–30 small models (1M–1B parameters) before committing
to a large run, and predict the final loss to within a few percent. The prediction is what makes a
$100M commitment defensible.

---

## 7. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Loss follows a power law in $N$, $D$ and $C$ over 7+ orders of magnitude. Nobody fully knows why. |
| 2 | $L(N,D) = E + A/N^\alpha + B/D^\beta$ with $E \approx 1.69$ — an irreducible floor. |
| 3 | Compute-optimal: $N \propto C^{0.45}$, $D \propto C^{0.55}$ → the rule of thumb **$D \approx 20N$**. |
| 4 | Kaplan's "bigger models" conclusion was an artifact of a fixed LR schedule. Chinchilla corrected it. |
| 5 | Chinchilla (70B/1.4T) beat Gopher (280B/300B) at equal compute, and was 4× cheaper to serve. |
| 6 | If you will serve a lot of tokens, train far past Chinchilla: 500–2000 tokens/param is rational. |
| 7 | Repeating data is ~as good as new data for up to 4 epochs — relevant now that data is scarce. |
| 8 | Inference-time compute is a second scaling axis that requires no retraining. |
| 9 | Loss is predictable; downstream capability is not. Run real evals during training. |
| 10 | Fit your own coefficients on a small ladder before committing to a large run. |

---

## Further reading

- Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020).
- Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022).
- Henighan et al., [*Scaling Laws for Autoregressive Generative Modeling*](https://arxiv.org/abs/2010.14701) (2020) — multimodal.
- Muennighoff et al., [*Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264) (2023).
- Sardana et al., [*Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*](https://arxiv.org/abs/2401.00448) (2023).
- Snell et al., [*Scaling LLM Test-Time Compute Optimally*](https://arxiv.org/abs/2408.03314) (2024).
- Besiroglu et al., [*Chinchilla Scaling: A Replication Attempt*](https://arxiv.org/abs/2404.10102) (2024) — a careful re-analysis of the fit.

**Next** → [Fine-tuning & PEFT](04-finetuning-peft.md)
