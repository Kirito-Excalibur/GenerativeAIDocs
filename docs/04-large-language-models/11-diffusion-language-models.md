# Diffusion Language Models

> **Summary**: Every mainstream LLM generates text one token at a time, left to right. Diffusion
> language models generate differently: start from a sequence of blank (masked) tokens and fill
> them in over several rounds, many positions at once. This page explains why Gaussian diffusion
> doesn't carry over to text, how masked (absorbing-state) diffusion works instead, how its training
> loss turns out to be a weighted version of BERT's, and where these models stand against
> autoregressive ones.

**Prerequisites**: → [Diffusion models](../05-diffusion-and-vision/01-diffusion-models.md), → [Autoregressive models](../02-classical-models/01-autoregressive-models.md) · **Next**: Part V → [Diffusion models](../05-diffusion-and-vision/01-diffusion-models.md)

---

## 1. Why try something other than left to right?

Autoregressive generation has three structural limits
(→ [Autoregressive models §8](../02-classical-models/01-autoregressive-models.md#8-strengths-and-weaknesses)):

| Limit | Consequence |
|---|---|
| **One token per forward pass** | latency grows linearly with output length |
| **No revision** | once a token is emitted it is permanent, even if later context shows it was wrong |
| **A fixed order** | the model is trained only to continue left to right, so it struggles with tasks that depend on later text, like filling a gap |

Diffusion addresses all three at once: many positions per step, every position can be
reconsidered, and nothing in the model assumes a direction.

```
  AUTOREGRESSIVE                        MASKED DIFFUSION

  step 1:  The                          step 0:  [M] [M] [M] [M] [M] [M]
  step 2:  The cat                      step 1:  The [M] [M] [M] [M] mat
  step 3:  The cat sat                  step 2:  The cat [M] on  the mat
  step 4:  The cat sat on               step 3:  The cat sat on  the mat
  step 5:  The cat sat on the
  step 6:  The cat sat on the mat

  6 passes for 6 tokens                 3 passes for 6 tokens, filled
  strictly in order                     in whatever order is easiest
```

---

## 2. Why Gaussian noise doesn't work for text

Image diffusion adds a small amount of Gaussian noise to continuous pixel values. Text is
**discrete**: a token is one of about 100,000 symbols, and "the cat" plus a little noise is not a
meaningful point. There are two ways around this.

**(a) Diffuse in embedding space.** Map tokens to continuous vectors, run ordinary Gaussian
diffusion, and snap each final vector back to the nearest token (Diffusion-LM, Li et al. 2022).
This works but is awkward: the rounding step introduces errors, and the model spends effort on
regions of embedding space that correspond to no token at all.

**(b) Use a discrete corruption process.** Define "noise" directly on tokens: at each step, some
tokens are replaced. D3PM (Austin et al., 2021) studied several choices:

| Corruption | What happens to a token | Verdict |
|---|---|---|
| Uniform | replaced by a random token | works, but the model must detect *which* tokens are corrupted |
| **Absorbing (masking)** | replaced by a special `[MASK]` token, and stays masked | **the approach that won** |
| Nearest-neighbour | replaced by a token with a similar embedding | little benefit |

> [!TIP]
> **Why masking won.** With masking, the model always knows which positions are corrupted, since
> they say `[MASK]`, and it only has to predict what belongs there. With uniform replacement, a
> plausible-looking wrong token is hidden among correct ones, so the model must first work out
> what's wrong before fixing it. Masking splits a hard problem into an easy one.

---

## 3. Masked diffusion: the forward process

Pick a noise level $t \in [0, 1]$. Independently for every position, keep the token with
probability $\alpha_t$ and replace it with `[MASK]` otherwise:

$$q(x_t^i \mid x_0^i) = \begin{cases} x_0^i & \text{with probability } \alpha_t\\ \texttt{[MASK]} & \text{with probability } 1-\alpha_t\end{cases}$$

where $\alpha_t$ falls from $\alpha_0 = 1$ (clean text) to $\alpha_1 = 0$ (all masks). A common
choice is the linear schedule $\alpha_t = 1 - t$, under which $t$ is simply the fraction of
tokens masked.

**Worked example.** "the cat sat on the mat" (6 tokens) at $t = 0.5$ with $\alpha_t = 0.5$: each
token is masked by an independent coin flip, so one sample might be
`the [M] sat [M] [M] mat`. On average 3 of 6 tokens are masked. At $t = 0.9$, about 5.4 of 6.

Compared with Gaussian diffusion, this is simpler in one important way: a token is either intact
or gone. There is no partially corrupted state.

---

## 4. The training objective, and its link to BERT

The reverse model $p_\theta(x_0 \mid x_t)$ is a Transformer that sees the partly masked sequence
and predicts the original token at every masked position. Working through the variational bound
(Sahoo et al. 2024; Shi et al. 2024) gives a strikingly simple loss:

$$\boxed{\;\mathcal{L} = \mathbb{E}_{t\sim\mathcal{U}(0,1)}\;\mathbb{E}_{x_t\sim q(\cdot\mid x_0)}\left[\frac{-\alpha_t'}{1-\alpha_t}\sum_{i:\,x_t^i=\texttt{[MASK]}} -\log p_\theta\big(x_0^i \mid x_t\big)\right]\;}$$

With the linear schedule, $\alpha_t' = -1$ and $1-\alpha_t = t$, so the weight is $1/t$:

$$\mathcal{L} = \mathbb{E}_{t}\left[\frac{1}{t}\sum_{\text{masked } i} -\log p_\theta(x_0^i\mid x_t)\right]$$

**Read it.** Mask a random fraction $t$ of tokens, predict them with cross-entropy, and weight the
result by $1/t$. That is **BERT's masked-language-modelling loss**, run at every masking rate from
0 to 100% instead of BERT's fixed 15%, with a weight that makes the sum a proper likelihood bound.

> [!TIP]
> **Why the $1/t$ weight.** At a low masking rate, only a few tokens are masked, so the sum has few
> terms. Dividing by $t$ normalizes that out: each noise level contributes as if it were a full
> sequence's worth of predictions. Without it, the loss would mostly train the high-noise regime.

The bound matters for a practical reason: it gives masked diffusion models a **perplexity upper
bound**, so they can be compared with autoregressive models on the same scale. BERT, trained at one
masking rate, is not a generative model and has no such bound.

```python
def masked_diffusion_loss(model, x0, mask_id, eps=1e-3):
    """x0: (B, L) clean token ids. Linear schedule: t = fraction masked."""
    B, L = x0.shape
    t = torch.rand(B, 1, device=x0.device).clamp(min=eps)       # noise level per sequence
    masked = torch.rand(B, L, device=x0.device) < t              # mask each token w.p. t
    xt = torch.where(masked, torch.full_like(x0, mask_id), x0)
    logits = model(xt)                                           # (B, L, V); no causal mask
    nll = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")   # (B, L)
    return ((nll * masked).sum(1) / t.squeeze(1)).mean() / L     # 1/t weight, per-token scale
```

> [!WARNING]
> **The model uses bidirectional attention.** Unlike a GPT, nothing here is causal: every position
> sees the whole sequence, masks included. Reusing a causal decoder without removing its mask
> quietly throws away most of the context and trains a much weaker model.

---

## 5. Sampling

Start from a fully masked sequence and step the noise level down from 1 to 0. At each step:

1. Run the model once on the current sequence to get a distribution at every masked position.
2. Decide which masked positions to fill this step.
3. Sample tokens there; leave the rest masked.

```
  t = 1.00  [M] [M]  [M]  [M]  [M]  [M]  [M]  [M]
  t = 0.75  [M] cat  [M]  [M]  [M]  [M]  [M]  mat
  t = 0.50  The cat  [M]  [M]  [M]  the  [M]  mat
  t = 0.25  The cat  sat  [M]  [M]  the  soft mat
  t = 0.00  The cat  sat  down on   the  soft mat
```

**Which positions to fill?** The simplest rule follows the schedule: unmask each remaining position
with the probability implied by the step size. Better in practice is **confidence-based
unmasking**: fill the positions where the model is most certain first, and leave the uncertain
ones for later rounds, when more context is available. This is the text version of solving the
easy parts of a crossword first.

**The speed trade-off.** With $K$ sampling steps for a sequence of length $L$:

| $K$ | Tokens per forward pass | Quality |
|---|---|---|
| $K = L$ | 1 | best, but no faster than autoregressive |
| $K = L/4$ | 4 | usually close |
| $K = L/16$ | 16 | noticeable degradation |

**Why quality drops with fewer steps.** Tokens filled in the same step are sampled *independently*
given the current context. If two masked positions depend on each other, like the two halves of
"New York" or "San Francisco", filling both at once can produce "New Francisco". Fewer steps means
more tokens filled together, so more of these conflicts.

> [!WARNING]
> **No KV cache, by default.** Autoregressive decoding reuses cached keys and values because the
> prefix never changes (→ [Inference §5](06-inference-and-decoding.md#5-the-kv-cache)). In
> diffusion, every position can change at every step, so each step recomputes attention over the
> whole sequence. A diffusion step therefore costs more than an autoregressive step, and the
> speed-up from filling many tokens per step has to outweigh that. Caching schemes for diffusion
> models, such as generating in blocks, are an active research area.

---

## 6. Where they stand

**Evidence from the main papers:**

| Paper | Claim (from the abstract) |
|---|---|
| D3PM, Austin et al. 2021 | framework for discrete diffusion; absorbing-state corruption works best for text |
| SEDD, Lou et al. 2024 | perplexity 25–75% lower than earlier language diffusion approaches; outperforms GPT-2; similar quality with **32× fewer network evaluations** than autoregressive sampling |
| MDLM, Sahoo et al. 2024 | simple masked diffusion is "more performant than previously thought"; its objective is a mixture of masked-LM losses; approaches autoregressive perplexity |
| **LLaDA**, Nie et al. 2025 | an **8B** masked diffusion model trained from scratch is competitive with **LLaMA3 8B** at in-context learning, and addresses the "reversal curse" |
| Mercury, Inception Labs 2025 | commercial diffusion code models reporting **1109** (Mini) and **737** (Small) tokens/s on an H100 |

**The reversal curse** is worth understanding because it shows the order constraint at work. An
autoregressive model trained on "A is B" often cannot answer "what is B?" with A. It learned only
the left-to-right association. A model trained to fill masks in every direction sees both
directions during training.

**Honest assessment.**

| Strength | Weakness |
|---|---|
| parallel generation: many tokens per step | each step costs more (no standard KV cache) |
| can revise and infill naturally | quality drops as steps get fewer |
| no left-to-right bias; helps with reversal and infilling | ecosystem, tooling and scale all lag autoregressive models |
| training loss is a proper likelihood bound | variable-length generation is awkward (the output length is set up front) |

> [!TIP]
> **The takeaway.** Diffusion LMs are no longer a curiosity: they reach 8B scale and ship
> commercially for latency-sensitive uses such as code completion. Whether they displace
> autoregressive models or end up as one specialized option, for example in hybrids that generate
> blocks autoregressively and fill each block by diffusion, is still open.

---

## 7. Exercises

**Problem 1 — the forward process, by hand.** A 10-token sequence is masked at $t=0.4$ (linear
schedule, so $\alpha_t=1-t=0.6$). What's the expected number of masked tokens? What's the
probability that *all 10* tokens happen to be masked in one particular sample (an unlucky but
possible draw)?

<details><summary>Solution</summary>

Expected masked count $= t\times L = 0.4\times10=4.0$ tokens (each of the 10 positions is masked
independently with probability $t$, per §3).

$P(\text{all 10 masked}) = t^{10} = 0.4^{10} = 0.000105$ — about 1 in 9,540. Rare but nonzero:
since masking is an independent coin flip per position (§3), there's no mechanism preventing an
unusually heavy or light draw at any given $t$; the loss function (§4) handles this correctly
because it's an *expectation* over $t$ and over the random masking pattern, not a per-sample
guarantee.

</details>

**Problem 2 — the $1/t$ weight, a small-$t$ case.** Using §4's linear-schedule loss weight
$1/t$, what does the weight become at $t=0.05$ (very light masking, few tokens to predict)? Given
that few terms are being summed at small $t$ (only ~5% of tokens are masked), does a large weight
here make intuitive sense, or does it seem like it's over-correcting?

<details><summary>Solution</summary>

Weight $=1/0.05=20$ — a large multiplier.

This makes sense once you recall *why* the weight exists (§4's explanation): at small $t$, the
inner sum has few terms (only ~5% of $L$ tokens are masked), so without reweighting, low-$t$
noise levels would contribute far less to the total expected loss than high-$t$ levels simply by
having fewer terms to sum — not because predicting them is easier or less important. The $1/t$
weight is specifically designed to counteract this: it inflates each rare masked-token's
contribution so that, in expectation over the random masking, every noise level $t$ contributes
*as if it were a full sequence's worth of predictions* (exactly as §4 states). So a weight of 20
at $t{=}0.05$ isn't over-correcting — it's precisely restoring the "one full sequence of signal
per noise level" balance the bound requires; without it, the model would effectively never learn
to predict well at low masking rates, exactly the regime closest to final, nearly-clean-text
generation.

</details>

**Problem 3 — bidirectional attention, spot the bug.** A team fine-tunes an existing GPT-style
(causal) checkpoint to do masked diffusion, reusing the model's existing attention layers
unchanged, and are confused why training loss plateaus much higher than a diffusion model trained
from scratch. Using §4's warning, diagnose the likely cause.

<details><summary>Solution</summary>

Per §4's explicit warning, masked diffusion needs **bidirectional** attention — every position,
including masked ones, must be able to attend to *every other* position (past and future) to
correctly predict what belongs at a masked slot using both left and right context. A GPT
checkpoint's attention layers have a **causal mask baked in**: position $i$ can only attend to
positions $\le i$. If the team reused the causal mask unchanged, every masked position can only
be filled in using *earlier* context, silently discarding all information from later in the
sequence — exactly the failure §4 warns about ("quietly throws away most of the context and
trains a much weaker model"). The fix is to remove the causal mask entirely for this training
regime (full bidirectional attention, as sketched in §4's code block), which also means the
underlying pretrained weights, having been trained under a causal mask, may not transfer as
cleanly as hoped — a mismatch worth checking for separately once the masking bug itself is fixed.

</details>

## 8. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Autoregressive LMs are one token per pass, can't revise, and are tied to left to right. Diffusion LMs relax all three. |
| 2 | Gaussian noise doesn't suit discrete tokens; masking (absorbing-state diffusion) is the corruption that works. |
| 3 | Masking tells the model exactly which positions to predict, which makes the task much easier. |
| 4 | The loss is BERT's masked-LM loss at every masking rate, weighted by $1/t$, which makes it a proper likelihood bound. |
| 5 | Diffusion LMs use bidirectional attention: never reuse a causal mask. |
| 6 | Sampling fills masks over several rounds; filling the most confident positions first works best. |
| 7 | Fewer steps means faster generation but more conflicts between tokens filled at the same time. |
| 8 | LLaDA reached 8B parameters, competitive with LLaMA3 8B at in-context learning; Mercury reports over 1,000 tokens/s. |

---

## Further reading

- Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM, 2021).
- Li et al., *Diffusion-LM Improves Controllable Text Generation* (2022): diffusion in embedding space.
- Lou et al., *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD, 2024).
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (MDLM, 2024).
- Shi et al., *Simplified and Generalized Masked Diffusion for Discrete Data* (2024).
- Nie et al., *Large Language Diffusion Models* (LLaDA, 2025).
- Inception Labs, *Mercury: Ultra-Fast Language Models Based on Diffusion* (2025).
- Berglund et al., *The Reversal Curse: LLMs Trained on "A is B" Fail to Learn "B is A"* (2023).

**Next** → Part V: [Diffusion models](../05-diffusion-and-vision/01-diffusion-models.md)
