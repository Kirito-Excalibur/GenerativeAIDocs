# Fine-tuning & Parameter-Efficient Fine-Tuning

> **Summary** — Adapting a pretrained model to your task. Full fine-tuning updates every parameter
> and needs ~16 bytes/parameter of optimizer memory; PEFT methods update <1% of parameters and fit
> on a single consumer GPU. This page derives LoRA's low-rank math, gives exact memory tables for
> every method, covers QLoRA's three tricks, and answers the question people actually have:
> *should I fine-tune at all, or use prompting/RAG?*

**Prerequisites**: → [Optimization](../01-foundations/05-optimization.md), → [Math toolkit §5](../01-foundations/03-math-toolkit.md#5-svd-and-low-rank-structure-the-math-behind-lora) · **Next**: → [Alignment](05-alignment.md)

---

## 1. Should you fine-tune? (read this first)

```
  Is the model failing?
   │
   ├─ It lacks KNOWLEDGE (facts, docs, recent events)
   │     └──► ❌ Don't fine-tune. Use RAG.
   │          Fine-tuning teaches style and format far more reliably
   │          than it teaches facts, and it makes updating them hard.
   │
   ├─ It doesn't follow the FORMAT / STYLE / TONE you want
   │     ├─ few examples fix it in the prompt? ──► ✅ Few-shot prompting
   │     └─ needs consistency at scale? ────────► ✅ Fine-tune (this is the sweet spot)
   │
   ├─ It lacks a SKILL (a domain task, a specialized classification)
   │     └──► ✅ Fine-tune, if you have ≥1000 good examples
   │
   ├─ It's too SLOW or EXPENSIVE
   │     └──► ✅ Distill a big model into a small fine-tuned one
   │
   └─ It lacks REASONING ability
         └──► ⚠️ Fine-tuning rarely fixes this. Use a stronger base model,
              or RL on verifiable rewards (hard, expensive).
```

🧠 **The single most useful heuristic**: *fine-tuning changes behaviour; retrieval changes
knowledge.* Trying to inject facts by fine-tuning gives you a model that has learned the *style* of
your documents and hallucinates content in that style — the worst of both worlds.

📊 **Order of things to try, cheapest first**:
1. Better prompt (minutes, free)
2. Few-shot examples (minutes, costs context)
3. RAG (days, needs infrastructure)
4. LoRA fine-tune (days, needs ~1k examples + a GPU)
5. Full fine-tune (weeks, needs data + a cluster)
6. Continued pretraining (months, needs billions of tokens)

---

## 2. Full fine-tuning and its memory bill

Same as pretraining, just fewer steps on different data.

🔢 **Memory for a 7B model with AdamW**:

| Component | Bytes/param | 7B total |
|---|---|---|
| Weights (BF16) | 2 | 14 GB |
| Gradients (BF16) | 2 | 14 GB |
| Adam $m$ (FP32) | 4 | 28 GB |
| Adam $v$ (FP32) | 4 | 28 GB |
| FP32 master weights | 4 | 28 GB |
| **Subtotal** | **16** | **112 GB** |
| Activations (batch 8, 2048 ctx, checkpointed) | — | ~8 GB |
| **Total** | | **~120 GB** |

**Does not fit on an 80 GB GPU.** That single fact is the entire motivation for PEFT.

⚠️ **Catastrophic forgetting** — full fine-tuning on a narrow dataset degrades general capability.
A model fine-tuned hard on medical QA gets worse at arithmetic, coding and instruction-following.

| Mitigation | How |
|---|---|
| Low learning rate | $1$–$2\times10^{-5}$, 10–30× lower than pretraining |
| Few epochs | 1–3; more overfits and forgets |
| **Replay** | mix in 5–20% general instruction data |
| **PEFT** | frozen base weights cannot be forgotten — the strongest structural fix |
| Weight averaging | interpolate fine-tuned and base weights (model soups / WiSE-FT) |

---

## 3. LoRA

📐 **The hypothesis.** The *update* learned during fine-tuning has low intrinsic rank. So
parameterize it directly as a low-rank product:

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA, \qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll \min(d,k)$$

```
     x
     │
     ├────────────────────┐
     ▼                    ▼
  ┌─────────┐        ┌─────────┐
  │   W₀    │        │    A    │  (r × k)   ← init: N(0, σ²)
  │ FROZEN  │        └────┬────┘
  │ (d × k) │             ▼
  └────┬────┘        ┌─────────┐
       │             │    B    │  (d × r)   ← init: ZEROS
       │             └────┬────┘
       │                  │ × α/r
       ▼                  ▼
      (+)◄────────────────┘
       │
       ▼  h = W₀x + (α/r)·BAx
```

🧠 **Why $B$ is initialized to zero.** At step 0, $BA = 0$, so $W' = W_0$ exactly — **the fine-tuned
model starts identical to the base model.** No random perturbation, no initial quality drop. $A$ is
random (Kaiming) so gradients can flow; if both were zero the product would have zero gradient
forever.

🔢 **Parameter count.** For a $4096\times4096$ projection with $r = 8$:

| | Parameters |
|---|---|
| Full $W$ | 16,777,216 |
| LoRA ($A$ + $B$) | $2\times4096\times8 = 65{,}536$ |
| **Fraction** | **0.39%** |

🔢 **Full model** — LLaMA-2 7B, LoRA on Q and V projections only, $r=8$:
$32\text{ layers}\times2\text{ modules}\times65{,}536 = 4.2$ M trainable parameters = **0.06%** of
6.7 B.

### The $\alpha/r$ scaling factor

$$\Delta W = \frac{\alpha}{r}BA$$

🧠 **Purpose**: decouple the *learning rate* from the *rank*. Without it, doubling $r$ doubles the
magnitude of the update at initialization-scale, forcing you to retune the LR every time you change
rank. With the scaling, $\alpha$ controls the effective update strength independently.

📊 **Common settings**: $\alpha = 2r$ (so $\alpha/r = 2$) or $\alpha = r$ (so $\alpha/r = 1$).
A widely-used default is $r = 16, \alpha = 32$.

⚠️ **rsLoRA** — the theoretically better scaling is $\alpha/\sqrt{r}$, not $\alpha/r$. With the
standard $\alpha/r$, high ranks are effectively down-scaled, which is why people often observe "high
rank doesn't help." With $\sqrt{r}$ scaling, higher ranks do continue to help. Worth knowing if your
rank ablation looks flat.

### Which modules to target

📊 Empirical findings:

| Target | Trainable % | Quality |
|---|---|---|
| Q, V only (original paper) | 0.06% | good |
| Q, K, V, O | 0.12% | slightly better |
| **All linear layers** (+ gate/up/down) | 0.3% | **best** |
| MLP only | 0.2% | surprisingly competitive |

🧠 The QLoRA paper's finding — **apply LoRA to *all* linear layers** — matters because it closes
most of the gap with full fine-tuning. Restricting to attention was an early convention, not a
principled choice.

### Rank selection

📊 A rough guide:

| $r$ | Use for |
|---|---|
| 4–8 | style/format adaptation, single narrow task |
| 16–32 | most tasks — **start here** |
| 64–128 | complex tasks, large datasets, domain shift |
| 256+ | approaching full fine-tuning; consider whether LoRA is still the right tool |

⚠️ Higher rank is not free: it increases memory, training time, and **overfitting risk on small
datasets**. Sweep $r \in \{8, 16, 32, 64\}$ on a validation set rather than guessing.

### Merging

At inference, fold the adapter into the weights:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$$

✅ **Zero inference overhead** — the merged model is architecturally identical to the base. This is
LoRA's decisive practical advantage over adapter layers, which add depth and latency.

✅ **Adapter swapping** — keep $W_0$ in memory once and hot-swap 20 MB adapters per request. Serving
100 fine-tuned variants costs barely more than serving one. (S-LoRA and similar systems do exactly
this.)

---

## 4. QLoRA: fine-tune a 65B model on one GPU

Three techniques, each independently useful:

### (a) 4-bit NormalFloat (NF4)

🧠 Neural network weights are approximately normally distributed. Standard INT4 quantization uses
*uniformly* spaced levels, which wastes resolution in the tails where few weights live. **NF4**
places its 16 levels at the quantiles of a standard normal — equal probability mass per bucket,
which is information-theoretically optimal for normally distributed data.

```
  INT4 (uniform levels)              NF4 (quantile levels)
  
  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │    ││││ │ │  │   │    │      │
  ─────────────────────────────     ─────────────────────────────
       ╱▔▔▔╲                             ╱▔▔▔╲
      ╱     ╲   weight distribution     ╱     ╲
  ___╱       ╲___                   ___╱       ╲___
  many levels wasted in the tails   levels concentrated where mass is
```

📊 NF4 gives measurably lower quantization error than INT4 or FP4 at the same bit width.

### (b) Double quantization

Quantization needs a scale factor per block (typically 64 weights). Those scale factors are FP32 —
$32/64 = 0.5$ bits per weight of overhead. Quantize *them* too (to 8-bit, in blocks of 256),
reducing overhead to ~0.127 bits/weight. Saves ~0.37 bits/param — about 3 GB on a 65B model.

### (c) Paged optimizers

Use NVIDIA unified memory so optimizer states spill to CPU RAM during transient memory spikes
(e.g. a long sequence in the batch) instead of OOM-ing.

🔢 **The combined result**:

| Method | 65B model memory | Fits on |
|---|---|---|
| Full fine-tuning | ~1040 GB | 16× A100 |
| LoRA (BF16 base) | ~140 GB | 2× A100 |
| **QLoRA (NF4 base)** | **~48 GB** | **1× A100 80GB** |
| QLoRA, 7B model | ~6 GB | a laptop GPU |

📊 QLoRA reported matching 16-bit full fine-tuning performance on their benchmarks. The forward
pass dequantizes NF4 → BF16 on the fly, so compute is unchanged; only storage shrinks.

⚠️ **The tradeoff to know**: QLoRA is *slower* than LoRA (~30–40%) because of dequantization
overhead. It trades speed for memory. If the model fits in BF16, use plain LoRA.

---

## 5. The PEFT family

| Method | Trainable | Inference overhead | Key idea |
|---|---|---|---|
| **LoRA** | 0.1–1% | **none** (merged) | low-rank update to weight matrices |
| **QLoRA** | 0.1–1% | none | LoRA on a 4-bit frozen base |
| **DoRA** | ~0.1% | none | decompose $W$ into magnitude + direction; LoRA the direction |
| Adapters (Houlsby) | 0.5–5% | ⚠️ +latency | small bottleneck MLPs inserted in each block |
| Prefix tuning | 0.1% | ⚠️ consumes context | learned key/value vectors prepended per layer |
| Prompt tuning | <0.1% | ⚠️ consumes context | learned soft tokens at the input only |
| $(IA)^3$ | ~0.01% | none | learned per-channel rescaling of K, V, MLP |
| BitFit | ~0.1% | none | train only the bias terms |

🧠 **DoRA is the notable recent improvement.** Decompose $W = m\frac{V}{\|V\|}$ into a magnitude
vector $m$ and a direction $V$; train $m$ fully and apply LoRA to $V$. The observation motivating it:
full fine-tuning changes magnitude and direction in a pattern LoRA cannot reproduce. DoRA closes
much of the remaining LoRA–full-FT gap, especially at low rank, for ~no extra inference cost.

🧠 **Prompt/prefix tuning lost** for a simple practical reason: they consume context window, are
harder to tune, and don't merge. LoRA's mergeability is worth more than the extra parameter
savings.

---

## 6. The data is what actually matters

📊 **Quality beats quantity, decisively.**

| Dataset | Size | Result |
|---|---|---|
| LIMA | **1,000** hand-curated examples | competitive with RLHF'd models |
| Alpaca | 52,000 GPT-generated | decent, noisy |
| AlpaGasus | **9,000** (filtered from Alpaca by an LLM judge) | **better** than the full 52k |

🧠 **The AlpaGasus result is the one to remember**: they removed 83% of Alpaca's data and the model
got *better*. Bad examples don't just fail to help — they actively teach bad behaviour.

📊 **Practical dataset guidance:**

| Goal | Examples needed |
|---|---|
| Output format/style | 100–1,000 |
| A specific narrow task | 1,000–10,000 |
| General instruction following | 10,000–100,000 |
| Domain expertise | 100,000+ (or continued pretraining) |

**Checklist for building a fine-tuning set:**
- [ ] Every example is one you'd be happy to see as output
- [ ] Deduplicated (near-duplicates included)
- [ ] Diverse in phrasing, length and difficulty
- [ ] Consistent format (same template, same system prompt)
- [ ] Held-out split that is genuinely held out
- [ ] Includes refusals/edge cases if you want the model to handle them
- [ ] Loss masked on the prompt — **train only on the completion tokens**

⚠️ **That last item is the most common bug in fine-tuning code.** If you compute loss over the
prompt too, the model spends capacity learning to generate *questions* instead of *answers*. Set
prompt token labels to `-100` (PyTorch's cross-entropy ignore index).

---

## 7. Implementation

💻 A complete, realistic QLoRA setup:

```python
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---- 4-bit base model ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4, not plain int4
    bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to BF16 for the matmul
    bnb_4bit_use_double_quant=True,         # quantize the quantization constants
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ---- LoRA ----
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",     # attention
                    "gate_proj", "up_proj", "down_proj"],       # MLP - include these!
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196

# ---- Training ----
args = TrainingArguments(
    output_dir="out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,          # effective batch 32
    learning_rate=2e-4,                     # 10x higher than full FT - LoRA wants this
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=200,
    optim="paged_adamw_8bit",               # paged + 8-bit optimizer states
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    max_seq_length=2048,
    # completion-only loss: mask everything up to and including the response marker
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template="<|start_header_id|>assistant<|end_header_id|>",
        tokenizer=tokenizer),
)
trainer.train()

# ---- Merge for deployment (optional; can't merge into a 4-bit base directly) ----
# reload the base in BF16, then merge:
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",
                                            torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "out").merge_and_unload()
merged.save_pretrained("merged-model")
```

📊 **Hyperparameters that matter, ranked:**

| Rank | Hyperparameter | Typical | Note |
|---|---|---|---|
| 1 | **Data quality** | — | dominates everything below |
| 2 | Learning rate | $1$–$3\times10^{-4}$ | LoRA wants ~10× higher LR than full FT |
| 3 | Epochs | 1–3 | more overfits fast on small sets |
| 4 | Rank $r$ | 16–32 | sweep it |
| 5 | Target modules | all linear | not just attention |
| 6 | $\alpha$ | $2r$ | rarely worth tuning separately |
| 7 | LoRA dropout | 0.05 | 0.1 for small datasets |

---

## 8. Evaluating a fine-tune

⚠️ Training loss going down tells you almost nothing. Check all of:

1. **Held-out loss** on data from the same distribution — is it overfitting?
2. **Task metric** — accuracy, exact match, or an LLM judge on your actual task.
3. **Regression check** — run a general benchmark (MMLU, a small instruction set) against the
   *base* model. Did you break anything?
4. **Qualitative sampling** — read 50 outputs. Nothing substitutes for this.
5. **Format compliance rate** — if you fine-tuned for JSON output, what % parses?

📊 **The overfitting signature in LoRA**: training loss keeps dropping while held-out loss turns up
after 2–3 epochs, and outputs become repetitive or start reproducing training examples verbatim.
Lower $r$, add dropout, get more data, or stop earlier.

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Fine-tuning teaches behaviour; RAG supplies knowledge. Don't confuse the two. |
| 2 | Full fine-tuning needs ~16 bytes/param — 112 GB for a 7B model. PEFT is not optional at scale. |
| 3 | LoRA: $W' = W_0 + \frac{\alpha}{r}BA$, $B$ init to **zero** so training starts at the base model. |
| 4 | Apply LoRA to **all** linear layers, not just Q/V. This closes most of the gap with full FT. |
| 5 | LoRA merges into the base weights → **zero inference overhead** and cheap multi-adapter serving. |
| 6 | QLoRA = NF4 + double quantization + paged optimizers → a 65B fine-tune on one 80 GB GPU. |
| 7 | Data quality dominates: 9k filtered examples beat 52k unfiltered (AlpaGasus). |
| 8 | Mask the loss on prompt tokens. This is the most common fine-tuning bug. |
| 9 | LoRA wants a ~10× higher learning rate than full fine-tuning ($2\times10^{-4}$ vs $2\times10^{-5}$). |
| 10 | Always run a regression check against the base model — fine-tuning breaks things silently. |

---

## Further reading

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024).
- Zhou et al., *LIMA: Less Is More for Alignment* (2023); Chen et al., *AlpaGasus* (2023).
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024) — an honest comparison with full FT.
- Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023).

**Next** → [Alignment: RLHF, DPO & friends](05-alignment.md)
