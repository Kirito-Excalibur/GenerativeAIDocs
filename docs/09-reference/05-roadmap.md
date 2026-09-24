# Study Roadmap

> A 12-week plan, assuming ~10–15 hours per week and a background in undergraduate calculus,
> linear algebra, probability, and Python. Each week has reading, a build project, and a
> **checkpoint** — a concrete question you should be able to answer before moving on.
>
> 🧠 **The single most important instruction on this page: build things.** Reading about
> backpropagation and implementing backpropagation produce different kinds of understanding, and
> only one of them survives contact with a real problem.

---

## Prerequisites check

Before starting, you should be able to:

- [ ] Take the derivative of a composite function by hand
- [ ] Multiply matrices and know why shapes must match
- [ ] Explain what a probability density is, and what conditioning means
- [ ] Write a Python class and use NumPy
- [ ] Read a loss curve

If any of these are shaky, spend a week on them first. 3Blue1Brown's linear algebra and calculus
series, plus Khan Academy's probability, are the standard fast path.

---

## Weeks 1–2: Foundations

**Read**: → [What is generative AI?](../01-foundations/01-what-is-generative-ai.md) through
→ [Taxonomy](../01-foundations/06-taxonomy.md) (all of Part I)

**Build**: a character-level n-gram language model, from scratch, in NumPy.
- Count bigrams and trigrams on a text file
- Add add-$k$ smoothing
- Sample from it
- Compute perplexity on held-out text

**Expected result**: a trigram model on Shakespeare reaches perplexity ~9 per character and
generates pronounceable nonsense.

**Checkpoint** — you should be able to answer, without looking:
1. Why is maximizing likelihood the same as minimizing KL divergence?
2. Why does that make VAEs blurry?
3. What is the generative trilemma, and where does each model family sit on it?
4. Why can't you model $p(x)$ for images with a lookup table or kernel density estimation?

---

## Week 3: Autoregressive models

**Read**: → [Autoregressive models](../02-classical-models/01-autoregressive-models.md),
→ [Tokenization](../03-sequence-models/01-tokenization.md)

**Build**:
1. A BPE tokenizer from scratch (train merges, encode, decode). Verify round-tripping.
2. A character-level MLP language model (Bengio-style: embed a fixed window, MLP, softmax).

**Checkpoint**:
1. Tokenize "The unbelievable 1234 résumé" — explain every token boundary.
2. Why does `"the"` differ from `" the"`, and why does that matter for prompts?
3. What exactly does teacher forcing do, and what problem does it create?

---

## Week 4: VAE and GAN

**Read**: → [VAE](../02-classical-models/02-vae.md), → [GAN](../02-classical-models/03-gan.md)

**Build**: a VAE on MNIST (PyTorch, ~80 lines).
- Get reconstruction ≈ 70 nats, KL ≈ 25 nats
- Sample from the prior; look at the outputs
- **Interpolate between two digits — with lerp and with slerp. Compare.**
- Deliberately cause posterior collapse by setting $\beta = 10$, then fix it with free bits

*(Optional)*: a DCGAN on MNIST. Watch it mode-collapse at least once — it's instructive.

**Checkpoint**:
1. Why does the reparameterization trick work, and what breaks without it?
2. Derive the Gaussian KL term. Check that $\mu=0,\sigma=1$ gives 0.
3. Why is the GAN's optimal discriminator a density ratio?
4. Your VAE samples are blurry. Give three distinct fixes and say why each works.

---

## Weeks 5–6: Attention and the Transformer

**Read**: → [RNNs](../03-sequence-models/02-rnn-lstm-gru.md) through
→ [Positional encoding](../03-sequence-models/05-positional-encoding.md)

**Build**: ⭐ **a GPT from scratch.** This is the single highest-value project in the roadmap.
- Multi-head causal self-attention (write the einsum yourself before using the fused kernel)
- Pre-norm blocks, RMSNorm, SwiGLU
- RoPE
- Train on TinyShakespeare (~1 MB)

**Expected result**: ~10M parameters, 30–60 min on a consumer GPU, validation loss ~1.5
bits/char, output that looks like Shakespeare's punctuation and structure with nonsense content.

> [!WARNING]
> **Do this without copying nanoGPT.** Read it *after* you've struggled. The struggle is the
> learning.

**Checkpoint**:
1. Why divide by $\sqrt{d_k}$? Compute the softmax entropy with and without, for $d_k=64$.
2. Derive $N \approx 12Ld^2$. Verify it against GPT-2 small's config.
3. Why is pre-norm necessary at depth? What exactly goes wrong with post-norm?
4. Prove that RoPE's inner product depends only on relative position.
5. Your model generates the same phrase in a loop. Name three causes and three fixes.

---

## Week 7: Pretraining at small scale

**Read**: → [LLM architecture](../04-large-language-models/01-llm-architecture.md),
→ [Pretraining](../04-large-language-models/02-pretraining.md)

**Build**: train a ~10–50 M parameter model on a real dataset (TinyStories or a FineWeb-Edu sample).
- Pre-tokenize into a memory-mapped uint16 array
- Implement warmup + cosine decay
- Implement gradient accumulation and clipping
- **Log grad_norm, MFU, and loss. Watch them.**
- Deliberately set the learning rate 10× too high and observe the divergence

**Checkpoint**:
1. Compute the FLOPs and estimated cost of your run. Compare to your wall-clock time. What's your MFU?
2. Why does warmup exist? What specifically fails without it?
3. Your loss spiked at step 8,000. List five possible causes and how you'd distinguish them.

---

## Week 8: Scaling and fine-tuning

**Read**: → [Scaling laws](../04-large-language-models/03-scaling-laws.md),
→ [Fine-tuning & PEFT](../04-large-language-models/04-finetuning-peft.md)

**Build**:
1. **Fit your own scaling law.** Train 5 models (1M → 30M params) at 2 data sizes each. Fit
   $L = E + A/N^\alpha + B/D^\beta$. Predict a 6th model's loss, then train it and check.
2. LoRA fine-tune an open 7–8B model on a small task (QLoRA on one consumer GPU).

**Checkpoint**:
1. You have $500k of compute. What model size and dataset size, and why?
2. Same budget, but you'll serve 5 trillion tokens. Now what, and why does the answer change?
3. Why is $B$ in LoRA initialized to zero?
4. Your fine-tuned model is worse at general tasks. Diagnose and fix.

---

## Week 9: Alignment and inference

**Read**: → [Alignment](../04-large-language-models/05-alignment.md),
→ [Inference & decoding](../04-large-language-models/06-inference-and-decoding.md)

**Build**:
1. Implement DPO from scratch (the loss is 4 lines) and run it on a small preference set.
2. Implement a sampler with temperature, top-$k$, top-$p$ and min-$p$. **Compare outputs at
   temperature 0 vs 0.8 vs 1.5 on the same prompt.**
3. Implement a KV cache. Measure the speedup.

**Checkpoint**:
1. Derive the DPO loss from the KL-constrained optimum. Where does $\log Z(x)$ go?
2. Why does greedy decoding produce repetition loops? Why doesn't nucleus sampling?
3. Compute the KV cache size for a 70B model at 32k context with and without GQA.
4. Why is sycophancy a predictable consequence of RLHF rather than a bug?

---

## Week 10: Diffusion

**Read**: → [Diffusion models](../05-diffusion-and-vision/01-diffusion-models.md) through
→ [Flow matching](../05-diffusion-and-vision/04-flow-matching.md)

**Build**: ⭐ **a DDPM on MNIST or CIFAR-10.**
- A small U-Net with timestep embeddings
- The linear schedule, then the cosine schedule — compare
- DDPM ancestral sampling (1000 steps), then DDIM (50 steps) — compare quality and time
- Then implement **flow matching** on the same data. Note that the training loop is shorter.

**Expected result**: recognizable MNIST digits after ~30 min on a consumer GPU.

**Checkpoint**:
1. Derive $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ from the single-step form.
2. Why does the network predict noise rather than the clean image?
3. Why does dropping the theoretical ELBO weighting *improve* sample quality?
4. Explain classifier-free guidance. Why does $w=1$ give weak prompt adherence?
5. Why does flow matching need fewer sampling steps?

---

## Week 11: RAG and agents

**Read**: → [Embeddings](../06-applications/02-embeddings-and-vector-search.md),
→ [RAG](../06-applications/03-rag.md), → [Agents](../06-applications/04-agents-and-tool-use.md)

**Build**: ⭐ **a RAG system over your own notes or a document collection you care about.**
- Chunk with structure awareness
- Hybrid search: BM25 + dense, fused with RRF
- Add a cross-encoder reranker
- Add an abstention path with a relevance floor
- **Build a 50-question evaluation set and measure recall@5 before and after each addition**

Then: a small agent with 3 tools (search your RAG index, a calculator, a file reader) and a step
limit.

**Checkpoint**:
1. Your RAG gives a wrong answer. Walk through the 8-point failure taxonomy to localize it.
2. Why does hybrid search beat dense alone? Give a query where dense fails.
3. Your agent succeeds 90% of the time per step. What's the success rate on a 15-step task?
4. Describe an indirect prompt injection against your agent and the architectural fix.

---

## Week 12: Evaluation and safety

**Read**: Parts VII and VIII

**Build**: an evaluation harness for something you built earlier.
- 100+ test cases, stratified by difficulty
- Repeats per case, with bootstrap confidence intervals
- An LLM judge with **position-swap debiasing**, validated against ~30 of your own labels
- A regression set seeded with every bug you hit during the previous 11 weeks

**Checkpoint**:
1. Model A scores 84% and model B scores 86% on 200 examples. Is B better? Show the arithmetic.
2. Why is perplexity not comparable across tokenizers? What do you use instead?
3. Compute pass@5 given $n=20$, $c=4$.
4. Name three things your benchmark does not measure that would matter in production.

---

## After week 12: pick a direction

| Direction | Next steps |
|---|---|
| **Research** | pick a subfield, reproduce a recent paper end to end, then find what it doesn't explain |
| **LLM engineering** | serving (vLLM/SGLang internals), quantization kernels, distributed training |
| **Applications** | agents, evaluation infrastructure, domain-specific systems, product integration |
| **Image/video** | train a small latent diffusion model end to end, including the VAE |
| **Interpretability** | train sparse autoencoders on a small model; find and steer features |
| **Safety** | red-teaming, evaluation design, robustness research |

---

## Resources

**Courses**
- Karpathy, *Neural Networks: Zero to Hero* — ⭐ the best starting point that exists
- Stanford CS336, *Language Modeling from Scratch*
- Stanford CS231n (vision), CS224n (NLP)
- fast.ai, *Practical Deep Learning*

**Books**
- Goodfellow, Bengio & Courville, *Deep Learning* — the classic reference
- Murphy, *Probabilistic Machine Learning: Advanced Topics* (2023) — the most current
- Prince, *Understanding Deep Learning* (2023) — free, excellently illustrated
- Tomczak, *Deep Generative Modeling* (2024)
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft) — free

**Code to read** (in this order)
- `karpathy/micrograd` — autograd in 100 lines
- `karpathy/nanoGPT` — ⭐ the clearest GPT implementation
- `karpathy/minbpe` — tokenization
- `lucidrains/denoising-diffusion-pytorch` — diffusion
- `huggingface/transformers` — production reality (messier, but it's what you'll use)
- `vllm-project/vllm` — serving internals

**Staying current**
- arXiv `cs.CL`, `cs.LG`, `cs.CV` — skim titles, read abstracts of ~5/day
- Papers with Code, Hugging Face Daily Papers
- Anthropic, DeepMind, OpenAI, Meta AI, DeepSeek research blogs
- Lilian Weng's blog, Sebastian Raschka's newsletter, Simon Willison's blog

---

## Six pieces of advice

1. **Build before you read the solution.** The struggle is what encodes the knowledge.
2. **Implement the math.** If you can't code it, you don't understand it.
3. **Start smaller than feels reasonable.** A model that trains in 10 minutes teaches you more per
   day than one that takes 10 hours.
4. **Read the loss curve.** Most debugging information is in it, and most people ignore it.
5. **Reproduce one paper completely.** It teaches you how much papers leave out.
6. **Keep a failure log.** Every bug you hit is a test case for your future evaluation suite, and
   re-reading it after six months is genuinely instructive.

> [!TIP]
> **The field moves fast, but the fundamentals don't.** Attention, the ELBO, score matching, KL
> divergence and scaling laws will still be correct in ten years. Model names, benchmark scores and
> API details will not. Spend your learning budget accordingly.

---

**Next** → [Notation](06-notation.md)
