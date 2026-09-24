# Efficiency: Quantization, Distillation and Sparsity

> **Summary** — Making models smaller and faster without making them worse. Quantization (the
> biggest win: 4× memory reduction for ~1% quality loss), distillation (transfer capability into a
> smaller model), and pruning/sparsity (the least effective of the three, and worth knowing why).
> This page includes the quantization arithmetic, the outlier problem that makes naive INT8 fail,
> and a decision guide.

**Prerequisites**: → [Inference & decoding](06-inference-and-decoding.md) · **Next**: → [Mixture of Experts](08-mixture-of-experts.md)

---

## 1. Why this matters: the memory-bandwidth argument

From → [LLM architecture §4](01-llm-architecture.md#4-request-lifecycle-prefill-and-decode): decoding
is memory-bandwidth-bound. Every generated token requires reading **all** model weights from HBM.

$$t_{\text{per token}} \approx \frac{\text{bytes of weights}}{\text{memory bandwidth}}$$

**70B model on an H100** (3.35 TB/s):

| Precision | Weight bytes | Theoretical tokens/s | Fits on |
|---|---|---|---|
| FP32 | 280 GB | 12 | 4× H100 |
| BF16 | 140 GB | 24 | 2× H100 |
| INT8 | 70 GB | 48 | 1× H100 |
| **INT4** | **35 GB** | **96** | **1× A100 40GB** |

**Quantization gives a near-linear speedup on decode** because it directly reduces the bytes
moved. This is a rare case where the same change improves memory, cost *and* latency together.

> [!WARNING]
> It does **not** speed up prefill much, which is compute-bound. And it does not reduce the KV
> cache unless you quantize that separately (which you can — KV cache quantization to INT8/FP8 is
> standard).

---

## 2. Quantization: the arithmetic

Map floats to a small integer grid.

**Symmetric (used for weights):**

$$s = \frac{\max|W|}{2^{b-1}-1}, \qquad W_q = \operatorname{round}\!\left(\frac{W}{s}\right), \qquad \hat W = s\cdot W_q$$

**Asymmetric (used for activations, which aren't centred):**

$$s = \frac{\max(W) - \min(W)}{2^b - 1}, \qquad z = -\operatorname{round}\!\left(\frac{\min(W)}{s}\right), \qquad W_q = \operatorname{round}\!\left(\frac{W}{s}\right) + z$$

**Worked example — INT8 symmetric.** Weights $W = [0.12, -0.45, 0.83, -0.21, 0.05]$.

$$s = \frac{0.83}{127} = 0.006535$$

| $w$ | $w/s$ | $W_q$ | $\hat w = sW_q$ | error |
|---|---|---|---|---|
| 0.12 | 18.36 | 18 | 0.11763 | 0.00237 |
| −0.45 | −68.86 | −69 | −0.45092 | 0.00092 |
| 0.83 | 127.0 | 127 | 0.83000 | 0.00000 |
| −0.21 | −32.13 | −32 | −0.20913 | 0.00087 |
| 0.05 | 7.65 | 8 | 0.05228 | 0.00228 |

Max error $= s/2 = 0.0033$. Relative to the weight range, that is ~0.4%. **8 bits is plenty for
weights** — as long as the range is well-behaved, which brings us to the problem.

---

## 3. The outlier problem (why naive INT8 fails)

> [!WARNING]
> Transformer activations contain **massive outliers** — a few feature dimensions with magnitudes
> 10–100× everything else. They appear in specific channels, consistently across tokens, and they
> emerge as models scale past ~6.7B parameters.

```
   activation magnitudes across the hidden dimension

   100 │        █                    █              ← outlier channels
       │        █                    █                 (systematic, not noise)
    10 │        █                    █
       │ ▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁    ← everything else
     1 │ ████████████████████████████████████████
       └──────────────────────────────────────────►  channel index
```

**Why this destroys quantization.** If one value is 100 and the rest are ~1, then
$s = 100/127 = 0.787$. Every normal value in $[-1, 1]$ now rounds to one of $\{-1, 0, 1\}$ —
**you've reduced them to 1.5 bits.** The outlier consumed the entire dynamic range.

The fixes, in order of how much they changed practice:

### LLM.int8(): mixed-precision decomposition

Split the matmul: outlier dimensions (~0.1% of channels) stay in FP16, everything else goes INT8.

$$XW \approx \underbrace{X_{\text{outlier}}W_{\text{outlier}}}_{\text{FP16}} + \underbrace{X_{\text{normal}}W_{\text{normal}}}_{\text{INT8}}$$

✅ Essentially lossless. ⚠️ Slow — the mixed kernel is awkward and often no faster than FP16.

### SmoothQuant: move the difficulty

> [!TIP]
> Activations have outliers; weights don't. So **migrate** the difficulty from activations to
> weights with a per-channel scaling that cancels out mathematically:

$$Y = (X\operatorname{diag}(s)^{-1})\cdot(\operatorname{diag}(s)W)$$

Choose $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$ with $\alpha \approx 0.5$. The product is
unchanged, but now *both* factors are quantization-friendly. ✅ Fast, ✅ INT8 throughout.

### GPTQ: error-compensating weight quantization

Quantize weights one column at a time; after each, **update the remaining un-quantized weights** to
compensate for the error introduced, using second-order (Hessian) information from a small
calibration set.

Enables good 4-bit and even 3-bit weights. Takes minutes to hours to run once, offline.

### AWQ: activation-aware weight quantization

> [!TIP]
> Not all weights matter equally. Weights multiplying **large-activation** channels are far more
> important. AWQ identifies the ~1% salient channels (by activation magnitude, not weight magnitude)
> and scales them up before quantizing, preserving their precision.

Faster to compute than GPTQ, comparable or better quality, and no backpropagation needed.

### NF4: for fine-tuning

Quantile-based 4-bit levels matched to the normal distribution of weights.
→ [Fine-tuning & PEFT §4](04-finetuning-peft.md#4-qlora-fine-tune-a-65b-model-on-one-gpu)

---

## 4. The quantization method table

Quality impact on standard benchmarks (approximate; varies by model):

| Method | Bits (W/A) | Perplexity increase | Speedup | Use when |
|---|---|---|---|---|
| BF16 | 16/16 | baseline | 1× | training, reference |
| FP8 | 8/8 | ~0% | 1.5–2× | H100+ hardware |
| **INT8 (SmoothQuant)** | 8/8 | **<1%** | ~1.8× | server inference |
| **GPTQ / AWQ INT4** | 4/16 | **1–3%** | ~3× | **the standard local setup** |
| GGUF Q4_K_M | ~4.5/16 | 1–2% | ~3× | llama.cpp / CPU / Mac |
| 3-bit | 3/16 | 5–15% | 3.5× | only if desperate |
| 2-bit | 2/16 | 20%+ | 4× | generally unusable |
| **BitNet b1.58** | 1.58/8 | — (trained this way) | large | requires training from scratch |

> [!TIP]
> **The 4-bit sweet spot is real and worth internalizing.** Going 16→8 bits is nearly free.
> 8→4 costs ~1–3%. 4→3 falls off a cliff. There appears to be a genuine information threshold around
> 4 bits per weight for post-training quantization of standard models.

> [!TIP]
> **BitNet is the interesting outlier.** If you *train* with ternary weights $\{-1, 0, +1\}$ from
> scratch, the model adapts and reaches quality competitive with FP16 at the same parameter count.
> The catch: you can't convert an existing model — you must train it that way. Matrix multiplication
> becomes addition, which could eventually change inference hardware.

Using quantized models in practice:

```python
# AWQ / GPTQ via transformers
from transformers import AutoModelForCausalLM, AwqConfig
model = AutoModelForCausalLM.from_pretrained(
    "model-AWQ", device_map="auto")          # pre-quantized checkpoint

# bitsandbytes on the fly (convenient, slower than AWQ/GPTQ kernels)
from transformers import BitsAndBytesConfig
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)

# KV cache quantization — separate from weight quantization, and often forgotten
# vLLM: --kv-cache-dtype fp8   (halves cache memory, ~no quality loss)
```

> [!WARNING]
> **Always evaluate after quantizing.** Perplexity is a weak proxy — a model can hold its
> perplexity while losing instruction-following or long-context retrieval. Run your actual task
> evaluation.

---

## 5. Distillation

Train a small **student** to match a large **teacher**.

$$\mathcal{L} = \alpha\underbrace{\mathcal{L}_{\text{CE}}(y_{\text{true}}, p_S)}_{\text{hard labels}}
+ (1-\alpha)\,\tau^2\underbrace{D_{\mathrm{KL}}\big(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\big)}_{\text{soft labels}}$$

> [!TIP]
> **Why soft labels carry more information — "dark knowledge".** A hard label says "this is a 7".
> The teacher's full distribution says "this is a 7 (0.9), but it looks somewhat like a 1 (0.07) and
> a bit like a 9 (0.02)". That *relative structure over wrong answers* encodes the teacher's learned
> similarity metric, and it is far richer supervision than a one-hot vector.

**Why $\tau^2$?** Softening by $\tau$ scales the gradients of the KL term by $1/\tau^2$.
Multiplying by $\tau^2$ keeps the two loss terms balanced as you change $\tau$. Typical
$\tau = 2$–$5$.

**The variants:**

| Type | Signal | Notes |
|---|---|---|
| **Response (black-box)** | teacher's generated *text* | works via API; this is how most open models are built |
| **Logit** | teacher's output distribution | needs logit access; strongest signal |
| Feature | intermediate hidden states | needs matching architectures |
| Attention transfer | attention maps | vision and older NLP work |
| **Sequence-level** | teacher's full sampled sequences | standard for generative tasks |
| **On-policy / GKD** | student samples, teacher scores them | fixes the train/inference mismatch |

> [!TIP]
> **On-policy distillation is the important refinement.** Standard sequence distillation trains
> the student on the *teacher's* outputs, but at inference the student sees its own. Generalized
> knowledge distillation (GKD) has the student generate and the teacher provide per-token feedback —
> the same exposure-bias fix as in autoregressive modelling generally.

**Notable results:**

| Model | Approach | Outcome |
|---|---|---|
| DistilBERT | layer-wise + logit distillation | 40% smaller, 60% faster, 97% of BERT's GLUE |
| Alpaca / Vicuna | response distillation from a strong API model | usable assistants for ~\$600 of API calls |
| Phi series | training on synthetic "textbook" data from a strong model | far above their weight class |
| Distilled reasoning models | SFT on reasoning traces from a large RL-trained model | small models inherit much of the reasoning |

> [!WARNING]
> **Licensing**: most commercial API terms prohibit using outputs to train competing models. The
> technique works; check the terms.

> [!TIP]
> **The generalizable finding**: distillation transfers *behaviour* very effectively and *knowledge*
> less so. A distilled 7B model can imitate a frontier model's style and reasoning format nearly
> perfectly while still lacking the underlying factual coverage. This mirrors the fine-tuning
> lesson in → [PEFT §1](04-finetuning-peft.md#1-should-you-fine-tune-read-this-first).

---

## 6. Pruning and sparsity

Remove weights. Two kinds:

```
  UNSTRUCTURED                        STRUCTURED (2:4 semi-structured)

  ┌──┬──┬──┬──┬──┬──┬──┬──┐          ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │▓▓│  │▓▓│  │  │▓▓│  │▓▓│          │▓▓│▓▓│  │  │▓▓│  │▓▓│  │
  └──┴──┴──┴──┴──┴──┴──┴──┘          └──┴──┴──┴──┴──┴──┴──┴──┘
   arbitrary positions                exactly 2 of every 4 kept
   ✅ best quality at a given sparsity ✅ hardware-accelerated (Ampere+)
   ❌ NO speedup on real hardware      ⚠️ ~50% sparsity ceiling
```

> [!WARNING]
> **The blunt truth about unstructured pruning**: 90% of weights zeroed gives **zero speedup** on a
> GPU, because dense matmul kernels don't skip zeros and sparse kernels have too much overhead at
> these sparsity levels. You save disk space and nothing else. This is why pruning has largely lost
> to quantization in practice.

**Modern methods that do work:**

| Method | Idea | Result |
|---|---|---|
| **SparseGPT** | one-shot pruning with Hessian-based weight updates | 50% sparsity, small quality loss, no retraining |
| **Wanda** | score = $\|W_{ij}\|\cdot\|X_j\|_2$ (weight × input activation norm) | matches SparseGPT, trivially simple, no gradients |
| 2:4 structured | exactly 2 nonzero per 4 | real ~1.5–2× speedup on Ampere+ tensor cores |
| **Layer/depth pruning** | drop whole transformer blocks | real speedup; later layers are surprisingly redundant |

> [!TIP]
> **Wanda is worth knowing because it is so simple**: importance = weight magnitude times input
> activation magnitude. No Hessian, no gradients, one calibration pass. That it matches much more
> sophisticated methods suggests importance really is mostly about "does this weight see large
> inputs."

> [!TIP]
> **The lottery ticket hypothesis** (Frankle & Carbin, 2018) — a randomly initialized dense network
> contains a sparse subnetwork that, trained *from the same initialization*, matches the full
> network. Theoretically fascinating; practically limited, because finding the ticket requires
> training the dense network first.

---

## 7. Decision guide

```
  I need the model to be smaller/faster.
  │
  ├─ Can't change training at all?
  │   ├─ need 2× ────► INT8 (SmoothQuant) or FP8 — nearly free
  │   ├─ need 4× ────► INT4 (AWQ or GPTQ) — the standard answer
  │   └─ need more ──► distill into a smaller model, then quantize that
  │
  ├─ Latency-bound at batch size 1?
  │   └────► quantization (bandwidth) + speculative decoding (parallelism)
  │
  ├─ Throughput-bound at large batch?
  │   └────► continuous batching + PagedAttention first;
  │          you are compute-bound, so quantization helps less
  │
  ├─ Memory-bound by the KV cache?
  │   └────► GQA/MLA, KV cache quantization (FP8), sliding-window attention
  │
  └─ Can afford to train?
      └────► distill to a smaller dense model (best quality/size)
             or train a native low-bit model (BitNet-style)
```

**Stacking works, with caveats:**

| Combination | Combined effect |
|---|---|
| INT4 + speculative decoding | ~5–6× faster decode |
| Distill to 1/3 size + INT4 | ~12× smaller |
| INT4 + FP8 KV cache | fits a 70B model with long context on one GPU |
| ⚠️ Distill + prune + quantize aggressively | errors compound — evaluate at every step |

---

## 8. Exercises

**Problem 1 — INT8 quantization, new weights.** Using §2's method, quantize
$W=(0.05, -0.62, 0.91, -0.08, 0.33)$ to INT8, compute the dequantized values, and the maximum
possible error. Which weight has the largest *relative* error, and why (hint: think about which
weight is smallest in magnitude)?

<details><summary>Solution</summary>

$s = \max|W|/127 = 0.91/127 = 0.007165$.

$W_q = \text{round}(W/s) = (7, -87, 127, -11, 46)$.

Dequantized: $(0.0502, -0.6234, 0.91, -0.0788, 0.3296)$.

Max error $=s/2=0.00358$ (absolute, same for every weight — this is §2's key point: the
quantization grid is uniform, so absolute error is bounded the same way everywhere).

**Relative** error is largest for the *smallest*-magnitude weight: $0.05\to0.0502$ has absolute
error $0.0016$ but relative error $0.0016/0.05=3.2\%$; the max-magnitude weight $0.91\to0.91$
(rounds exactly, since $127\times s=0.91$ by construction) has $0\%$ relative error. This is the
mechanistic reason activation *outliers* are so damaging (§3): a single large-magnitude value
sets the scale for everyone, and every smaller value — including ones that matter — inherits a
worse relative error because the grid resolution is fixed by the outlier, not by the typical
value.

</details>

**Problem 2 — distillation temperature, applied.** Teacher logits over 4 classes:
$(4.0, 1.0, 0.5, -2.0)$. Compute the softened teacher distribution at $T{=}1$ (no softening) and
$T{=}4$, per §5. How much "dark knowledge" (relative probability on the non-top classes) does
$T{=}4$ reveal that $T{=}1$ essentially hides?

<details><summary>Solution</summary>

$T{=}1$: $(0.924, 0.046, 0.028, 0.002)$ — the top class dominates almost completely; classes 2
and 3 (probabilities 0.046 and 0.028) are barely distinguishable from each other in relative
terms once rounded, and class 4 is nearly invisible.

$T{=}4$: $(0.473, 0.224, 0.197, 0.106)$ — all four classes now carry substantial, clearly
*differentiated* probability mass. Notice classes 2 and 3, which were both "small and similar"
under $T{=}1$ (0.046 vs 0.028, a ratio of 1.64), remain in a similar ratio under $T{=}4$ (0.224 vs
0.197, ratio 1.14) but are now large enough in absolute terms for a student's cross-entropy
gradient to actually pick up on the *relative* ordering between them — this is exactly the "dark
knowledge" §5 describes: the teacher's belief that class 3 is more plausible than class 4 is
present in both temperatures, but only *usable as a training signal* once softening makes it
numerically significant.

</details>

**Problem 3 — the 4-bit cliff, reasoned.** §4's table shows perplexity increase roughly
$<1\%$ at INT8, $1$–$3\%$ at INT4, and $5$–$15\%$ at 3-bit. Using §3's outlier discussion and
§2's "8 bits is plenty" result, explain in mechanistic terms (not just "it's an empirical fact")
why the degradation from 8→4 bits is so much milder than from 4→3 bits, referencing the number of
representable levels at each bit-width.

<details><summary>Solution</summary>

Bit-width sets the number of representable levels: 8-bit → 256 levels, 4-bit → 16 levels, 3-bit →
8 levels. Going 8→4 bits is a $16\times$ reduction in levels (256→16); going 4→3 bits is only a
further $2\times$ reduction (16→8) — a much *smaller* relative step in levels, yet it produces a
*larger* jump in perplexity degradation. The reason isn't the ratio of levels but where you land
relative to the data's actual information content: §2 showed 8-bit error is already small
relative to typical weight magnitudes ($s/2$ with $s=\max|W|/127$, giving sub-percent relative
error for most weights). At 4 bits (16 levels), you're still resolving the *bulk* of the weight
distribution reasonably, since most weights cluster near the mode and 16 levels can still capture
that shape roughly. At 3 bits (8 levels), you cross a threshold where there simply aren't enough
distinct values left to represent the weight distribution's shape at all — many *meaningfully
different* weights collapse onto the *same* quantized value, which is a qualitatively different
kind of error (information loss, not just rounding noise) — matching §4's framing of "a genuine
information threshold around 4 bits" rather than a smooth continuation of the same rounding-error
trend.

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Decode is bandwidth-bound, so halving the bytes nearly halves the latency. |
| 2 | Activation outliers (systematic, from ~6.7B params up) break naive INT8; SmoothQuant migrates the difficulty to the weights. |
| 3 | 16→8 bits is nearly free; 8→4 costs 1–3%; below 4 bits quality collapses. |
| 4 | AWQ/GPTQ INT4 is the standard local-inference setup. |
| 5 | Quantize the KV cache separately — it's often the actual memory constraint. |
| 6 | Distillation's value is the teacher's *full distribution* over wrong answers ("dark knowledge"). |
| 7 | Distillation transfers behaviour well and knowledge poorly. |
| 8 | Unstructured pruning gives no real speedup on GPUs. Use 2:4 structured or layer pruning if you must. |
| 9 | Wanda ($\|W\|\cdot\|X\|$) matches far more complex pruning methods — importance is mostly about input scale. |
| 10 | Always re-evaluate on your actual task after any compression; perplexity hides real regressions. |

---

## Further reading

- Dettmers et al., *LLM.int8()* (2022) — the outlier discovery.
- Xiao et al., [*SmoothQuant*](https://arxiv.org/abs/2211.10438) (2022); Lin et al., *AWQ* (2023); Frantar et al., [*GPTQ*](https://arxiv.org/abs/2210.17323) (2022).
- Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015).
- Agarwal et al., [*GKD: Generalized Knowledge Distillation*](https://arxiv.org/abs/2306.13649) (2023) — on-policy distillation.
- Frantar & Alistarh, [*SparseGPT*](https://arxiv.org/abs/2301.00774) (2023); Sun et al., [*Wanda*](https://arxiv.org/abs/2306.11695) (2023).
- Ma et al., [*The Era of 1-bit LLMs: BitNet b1.58*](https://arxiv.org/abs/2402.17764) (2024).

**Next** → [Mixture of Experts](08-mixture-of-experts.md)
