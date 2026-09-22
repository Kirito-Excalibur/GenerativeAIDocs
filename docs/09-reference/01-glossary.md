# Glossary

> ~220 terms, one crisp line each, with links to the page that explains them properly.
> Organized alphabetically within thematic groups for browsability; use Ctrl-F for lookup.

**Jump to**: [A–C](#ac) · [D–F](#df) · [G–L](#gl) · [M–P](#mp) · [Q–S](#qs) · [T–Z](#tz)

---

## A–C

| Term | Definition |
|---|---|
| **Ablation** | removing a component to measure its contribution. |
| **Activation checkpointing** | recompute activations in the backward pass instead of storing them; ~33% more compute for $O(\sqrt L)$ memory. → [NN refresher](../01-foundations/04-neural-network-refresher.md) |
| **Active parameters** | parameters used per token in an MoE — distinct from *total* parameters. → [MoE](../04-large-language-models/08-mixture-of-experts.md) |
| **AdaLN-Zero** | conditioning via LayerNorm scale/shift, residual branch zero-initialized; used in DiT. |
| **AdamW** | Adam with *decoupled* weight decay. The standard optimizer. → [Optimization](../01-foundations/05-optimization.md) |
| **ALiBi** | positional method adding a linear distance penalty to attention scores. → [Positional encoding](../03-sequence-models/05-positional-encoding.md) |
| **Alignment** | making a model behave as intended: helpful, honest, harmless. → [Alignment](../04-large-language-models/05-alignment.md) |
| **Alignment tax** | capability loss on benchmarks caused by alignment training. |
| **ANN** | approximate nearest neighbour search. → [Vector search](../06-applications/02-embeddings-and-vector-search.md) |
| **Attention** | weighted average of values, weights from query–key similarity. → [Attention](../03-sequence-models/03-attention.md) |
| **Attention sink** | a token (usually the first) that absorbs excess attention mass because softmax must sum to 1. |
| **Autoregressive** | $p(x)=\prod_i p(x_i\mid x_{<i})$. → [AR models](../02-classical-models/01-autoregressive-models.md) |
| **AWQ** | activation-aware 4-bit weight quantization. → [Efficiency](../04-large-language-models/07-efficiency.md) |
| **Backpropagation** | chain rule with memoization; backward ≈ 2× forward cost. |
| **Base model** | a pretrained model before instruction tuning or alignment. |
| **Batch size (critical)** | the point past which larger batches stop reducing step count proportionally. |
| **Beam search** | keep the $B$ best partial sequences; good for constrained tasks, bad for open generation. |
| **BERTScore** | text similarity via contextual embedding matching. → [Metrics](../07-evaluation/01-metrics.md) |
| **BF16** | bfloat16: FP32's exponent range with 7 mantissa bits. The training default. |
| **Bias (statistical)** | systematic error; in generative AI, also representational harm. → [Societal impact](../08-safety-and-ethics/03-societal-impact.md) |
| **BitNet** | models trained natively with ternary $\{-1,0,1\}$ weights. |
| **BLEU** | n-gram precision metric for translation. Cannot see meaning. |
| **BM25** | classic sparse lexical retrieval scoring function. → [Vector search](../06-applications/02-embeddings-and-vector-search.md) |
| **BPC / BPB** | bits per character / per byte; the tokenizer-independent way to compare LMs. |
| **BPE** | Byte-Pair Encoding: greedily merge the most frequent adjacent pair. → [Tokenization](../03-sequence-models/01-tokenization.md) |
| **Bradley–Terry** | model turning pairwise preferences into a scalar score; the basis of reward models. |
| **C2PA** | signed content-provenance standard. |
| **Calibration** | whether stated confidence matches actual accuracy. RLHF degrades it. |
| **Capacity factor** | per-expert token buffer size in MoE; overflow tokens are dropped. |
| **Catastrophic forgetting** | losing prior capability during fine-tuning. |
| **Causal mask** | prevents attending to future positions. |
| **CFG** | classifier-free guidance: $\epsilon(\varnothing) + w(\epsilon(c)-\epsilon(\varnothing))$. → [Latent diffusion](../05-diffusion-and-vision/03-latent-diffusion.md) |
| **Chain-of-thought (CoT)** | intermediate reasoning tokens; converts sequence length into computational depth. → [Reasoning](../04-large-language-models/10-reasoning.md) |
| **Chinchilla** | the compute-optimal scaling result: ~20 tokens per parameter. → [Scaling laws](../04-large-language-models/03-scaling-laws.md) |
| **CLIP** | contrastively trained joint image–text embedding model. → [Multimodal](../05-diffusion-and-vision/05-multimodal.md) |
| **Consistency model** | maps any point on a diffusion trajectory directly to its endpoint → 1-step generation. |
| **Constitutional AI** | alignment using AI feedback guided by explicit written principles. |
| **Constrained decoding** | mask invalid tokens to guarantee grammar conformance. → [Structured output](../06-applications/05-structured-output.md) |
| **Contamination** | benchmark test data leaking into training data. → [Benchmarks](../07-evaluation/02-benchmarks.md) |
| **Context window** | maximum sequence length the model can process. |
| **Contextual retrieval** | prepending an LLM-written situating sentence to each chunk before embedding. |
| **Contrastive learning** | pull positives together, push negatives apart (InfoNCE). |
| **ControlNet** | spatial conditioning via a trainable copy of the encoder with zero-init connections. |
| **Cross-attention** | Q from one stream, K/V from another; the universal conditioning mechanism. |
| **Cross-encoder** | scores a (query, document) pair jointly; accurate, not precomputable. |
| **Cross-entropy** | $-\sum p\log q$; the standard training loss. → [Info theory](../01-foundations/02-probability-and-information-theory.md) |
| **Curse of dimensionality** | data requirements grow exponentially with dimension. |

## D–F

| Term | Definition |
|---|---|
| **DDIM** | deterministic diffusion sampler; a discretization of the probability-flow ODE. |
| **DDPM** | denoising diffusion probabilistic model. → [Diffusion](../05-diffusion-and-vision/01-diffusion-models.md) |
| **Decoder-only** | the modern LLM architecture: causal attention, one homogeneous stack. |
| **Decoding** | turning a probability distribution into tokens. → [Inference](../04-large-language-models/06-inference-and-decoding.md) |
| **Deduplication** | removing duplicate training documents; improves loss *and* reduces memorization. |
| **Dequantization** | adding noise to discrete data so a continuous density model is well-defined. |
| **Diffusion model** | learns to reverse a gradual noising process. |
| **Distillation** | training a small student to match a large teacher. |
| **DiT** | Diffusion Transformer — a Transformer backbone replacing the U-Net. |
| **DoRA** | LoRA variant decomposing weights into magnitude and direction. |
| **DPO** | Direct Preference Optimization: alignment without a reward model. → [Alignment](../04-large-language-models/05-alignment.md) |
| **Dropout** | random unit zeroing; mostly absent from modern pretraining. |
| **EBM** | energy-based model: $p(x)\propto e^{-E(x)}$. → [EBM](../02-classical-models/05-energy-based-models.md) |
| **ECE** | expected calibration error. |
| **Elo** | rating system used by preference arenas; 100 points ≈ 64% win rate. |
| **ELBO** | evidence lower bound: $\log p(x) = \text{ELBO} + \text{KL}(q\|p(z\mid x))$. |
| **Embedding** | a vector representation where semantic similarity is geometric proximity. |
| **Emergence** | capabilities appearing abruptly with scale; partly a metric artifact. |
| **Encoder-only** | bidirectional architecture (BERT); good at understanding, can't generate. |
| **Entropy** | $-\sum p\log p$; the irreducible number of bits needed to describe a distribution. |
| **Exposure bias** | train/inference mismatch: teacher forcing vs the model's own outputs. |
| **Expert parallelism** | distributing MoE experts across devices; requires all-to-all communication. |
| **Faithfulness** | (a) whether generated claims are supported by context; (b) whether stated reasoning reflects actual computation. |
| **Few-shot** | including examples in the prompt. |
| **FID** | Fréchet Inception Distance; the standard image-generation metric, with real caveats. |
| **FlashAttention** | exact attention with $O(T)$ memory via tiling and online softmax. |
| **Flow matching** | regress onto a straight-line velocity field; the modern diffusion alternative. → [Flow matching](../05-diffusion-and-vision/04-flow-matching.md) |
| **FP8** | 8-bit float; used on recent hardware for training and inference. |
| **Free bits** | a KL floor per latent dimension; prevents posterior collapse. |
| **FSDP** | Fully Sharded Data Parallel — PyTorch's ZeRO-3. |

## G–L

| Term | Definition |
|---|---|
| **GAN** | generator vs discriminator minimax game. → [GAN](../02-classical-models/03-gan.md) |
| **GELU** | $x\Phi(x)$; smooth activation used in BERT and GPT-2/3. |
| **Goodhart's law** | when a measure becomes a target, it ceases to be a good measure. |
| **GPTQ** | 4-bit weight quantization with Hessian-based error compensation. |
| **GQA** | grouped-query attention; fewer K/V heads than Q heads → 8× smaller KV cache. |
| **Gradient clipping** | cap the global gradient norm (typically 1.0). Essential. |
| **GraphRAG** | RAG over an entity/relation graph; handles corpus-wide questions. |
| **GRPO** | Group Relative Policy Optimization; PPO without a value model. |
| **Guidance scale** | the $w$ in CFG; trades diversity for prompt fidelity. |
| **Hallucination** | fluent, confident, false output. Structural, not a mere bug. → [Safety](../08-safety-and-ethics/01-safety.md) |
| **Hard negative** | a plausible-looking but incorrect training negative; what makes embeddings good. |
| **HNSW** | hierarchical navigable small-world graph; the quality-default ANN index. |
| **HyDE** | embed a *hypothetical answer* rather than the query, to search in document-space. |
| **In-context learning** | learning from examples in the prompt, without weight updates. |
| **Induction head** | attention circuit implementing "[A][B]…[A]→[B]"; linked to in-context learning. |
| **InfoNCE** | the contrastive loss; bounds mutual information by $\log N$ (batch size). |
| **Instruction tuning** | supervised fine-tuning on (instruction, response) pairs. |
| **IVF-PQ** | inverted file + product quantization; ~32× index compression. |
| **Jailbreak** | eliciting behaviour the model was trained to refuse. |
| **JSD** | Jensen–Shannon divergence; the symmetric, bounded relative of KL. |
| **KL divergence** | $\sum p\log(p/q)$; excess bits from using the wrong model. Asymmetric. |
| **KV cache** | stored keys and values for generated tokens; the real inference memory constraint. |
| **Langevin dynamics** | noisy gradient ascent on log-density; samples using only the score. |
| **Latent diffusion** | diffusion in a VAE's compressed latent space. → [Latent diffusion](../05-diffusion-and-vision/03-latent-diffusion.md) |
| **LayerNorm** | normalize across features per token. |
| **Lethal trifecta** | private data + untrusted content + external communication ⇒ exfiltration risk. |
| **LLM-as-judge** | using a strong model to evaluate outputs; ~80–85% human agreement if debiased. |
| **LoRA** | $W' = W_0 + \frac{\alpha}{r}BA$; low-rank fine-tuning that merges at inference. → [PEFT](../04-large-language-models/04-finetuning-peft.md) |
| **Lost in the middle** | information in the middle of a long context is used poorly. |

## M–P

| Term | Definition |
|---|---|
| **Mamba** | selective state-space model; linear-time training, $O(1)$ inference. |
| **Manifold hypothesis** | real data occupies a low-dimensional manifold in its ambient space. |
| **Matryoshka embedding** | truncatable embeddings — the first $k$ dimensions are still valid. |
| **Memorization** | verbatim reproduction of training data; scales with duplicates and model size. |
| **MFU** | model FLOPs utilization; 40–50% is good for large runs. |
| **MLA** | multi-head latent attention; compresses the KV cache into a shared low-rank latent. |
| **Mode collapse** | a generator producing limited variety; the classic GAN failure. |
| **MoE** | mixture of experts: route each token to top-$k$ of $E$ expert MLPs. → [MoE](../04-large-language-models/08-mixture-of-experts.md) |
| **Monte Carlo** | estimating expectations by sampling. |
| **MQA** | multi-query attention: one shared K/V head. Max cache savings, some quality loss. |
| **Mutual information** | $I(X;Y)$: bits that knowing $Y$ saves in describing $X$. |
| **nDCG** | position-weighted, graded retrieval quality metric. |
| **NF4** | 4-bit NormalFloat; quantile-spaced levels matched to a normal distribution. |
| **Normalizing flow** | invertible map with a tractable Jacobian → exact likelihood. → [Flows](../02-classical-models/04-normalizing-flows.md) |
| **Nucleus sampling** | top-$p$: keep the smallest token set with cumulative probability $\ge p$. |
| **ODE (probability flow)** | the deterministic ODE with the same marginals as a diffusion SDE. |
| **ORM / PRM** | outcome / process reward model — final-answer vs per-step supervision. |
| **Over-refusal** | refusing benign requests; must be measured alongside attack success. |
| **PagedAttention** | OS-style paging for the KV cache; cuts waste from ~70% to <4%. |
| **pass@k** | probability that at least one of $k$ samples passes; use the unbiased estimator. |
| **PEFT** | parameter-efficient fine-tuning. |
| **Perplexity** | $\exp(\text{cross-entropy})$; not comparable across tokenizers. |
| **PII** | personally identifiable information. |
| **Positional encoding** | injecting order information; attention alone is permutation-equivariant. |
| **Posterior collapse** | the VAE latent carrying no information ($I(x;z)\to 0$). |
| **PPO** | Proximal Policy Optimization; the RL algorithm in classic RLHF. |
| **Pre-norm** | normalization inside the residual branch; enables training at depth. |
| **Prefill** | processing the prompt; compute-bound, unlike decode. |
| **Prompt caching** | caching KV states for a reused prefix; large cost and latency win. |
| **Prompt injection** | untrusted content acting as instructions. **No complete solution exists.** → [Security](../08-safety-and-ethics/02-security.md) |

## Q–S

| Term | Definition |
|---|---|
| **QK-norm** | normalizing Q and K before the dot product; prevents attention-logit blowup. |
| **QLoRA** | LoRA on a 4-bit (NF4) frozen base; a 65B fine-tune on one GPU. |
| **Quantization** | representing weights/activations in fewer bits. 4-bit is the sweet spot. |
| **RAG** | retrieval-augmented generation. → [RAG](../06-applications/03-rag.md) |
| **ReAct** | interleaved reasoning and tool-use loop. |
| **Rectified flow** | flow matching with straight-line interpolation; SD3 and Flux use it. |
| **Reflow** | re-couple $(x_0,x_1)$ pairs to straighten flow trajectories → 1–2 step generation. |
| **Reparameterization trick** | $z=\mu+\sigma\epsilon$: moves randomness off the gradient path. |
| **Reranker** | a cross-encoder rescoring retrieved candidates; +5–15 nDCG. |
| **Residual connection** | $y = x + F(x)$; makes identity the default and preserves gradients. |
| **Residual stream** | the additive $(B,T,d)$ bus that sublayers read from and write to. |
| **Reward hacking** | optimizing the proxy objective while violating its intent. |
| **Reward model** | learns human preference from pairwise comparisons; 65–75% accurate. |
| **RLHF** | reinforcement learning from human feedback. |
| **RLVR** | RL from verifiable rewards (math, code); how reasoning models are trained. |
| **RMSNorm** | LayerNorm without mean subtraction; ~10% faster, no quality loss. |
| **RoPE** | rotary position embedding; rotate Q and K so similarity depends only on distance. |
| **ROUGE** | n-gram recall metric for summarization; rewards copying. |
| **RRF** | reciprocal rank fusion; combines rankers without score calibration. |
| **SAE** | sparse autoencoder; decomposes activations into interpretable features. |
| **Scaling law** | loss as a power law in parameters, data and compute. |
| **Score** | $\nabla_x\log p(x)$; independent of the normalizing constant. |
| **Score matching** | learning the score without knowing the density. |
| **SDE** | stochastic differential equation; the continuous-time view of diffusion. |
| **Self-consistency** | sample $n$ chains, take the majority answer; also a confidence signal. |
| **SFT** | supervised fine-tuning. |
| **Slerp** | spherical interpolation; the correct way to interpolate high-dimensional latents. |
| **Sliding window attention** | attend only to the last $w$ tokens; $L\times w$ indirect reach. |
| **SmoothQuant** | migrate activation outliers into the weights so INT8 works. |
| **Softmax** | $e^{z_i}/\sum_j e^{z_j}$; shift-invariant, temperature-controlled. |
| **Speculative decoding** | draft with a small model, verify with a large one; **exact** output distribution. |
| **SSM** | state-space model; linear recurrence, parallelizable by prefix scan. |
| **Superposition** | representing more features than dimensions using near-orthogonal directions. |
| **SwiGLU** | gated activation $\text{Swish}(xW_1)\odot(xW_2)$; the modern LLM default. |
| **Sycophancy** | agreeing with the user against the evidence; a direct consequence of RLHF. |

## T–Z

| Term | Definition |
|---|---|
| **Teacher forcing** | feeding ground-truth prefixes during training; enables parallel training. |
| **Temperature** | softmax scaling; 0 = greedy, ∞ = uniform. |
| **Tensor parallelism** | splitting a layer's weights across devices; must stay intra-node. |
| **Test-time compute** | inference-time compute as a second scaling axis. |
| **Token** | the discrete unit a model operates on; ~4 characters in English. |
| **Tokenizer** | text ↔ integers. Cause of arithmetic and character-counting failures. |
| **Top-k / top-p** | truncated sampling; top-$p$ adapts to model confidence, top-$k$ doesn't. |
| **Transformer** | attention + MLP blocks with residuals. → [Transformer](../03-sequence-models/04-transformer.md) |
| **Trilemma (generative)** | quality, speed, coverage — pick two. |
| **U-Net** | encoder–decoder with skip connections; the classic diffusion backbone. |
| **Unfaithful CoT** | stated reasoning that doesn't reflect the actual computation. |
| **VAE** | variational autoencoder; ELBO = reconstruction − KL. → [VAE](../02-classical-models/02-vae.md) |
| **Vanishing gradient** | exponential gradient decay through depth/time; what killed RNNs. |
| **v-prediction** | predicting $v=\sqrt{\bar\alpha}\epsilon - \sqrt{1-\bar\alpha}x_0$; better-behaved at extremes. |
| **Vector database** | store + ANN index for embeddings. |
| **ViT** | Vision Transformer; treats 16×16 patches as tokens. |
| **VQ-VAE** | VAE with a discrete codebook latent; enables Transformers over images/audio/video. |
| **Warmup** | ramping the learning rate from 0; needed because early Adam variance estimates are unreliable. |
| **Wasserstein distance** | earth-mover's distance; smooth even when supports don't overlap. |
| **Watermarking** | statistical signal embedded in generated content; removable by paraphrase. |
| **Weight decay** | L2 penalty on weights; use AdamW's decoupled form, matrices only. |
| **Weight tying** | sharing the embedding and output projection matrices. |
| **WSD schedule** | warmup–stable–decay; resumable alternative to cosine. |
| **YaRN** | RoPE context-extension method combining interpolation with NTK-aware scaling. |
| **Zero-shot** | performing a task with no examples. |
| **ZeRO** | sharding optimizer states, gradients and parameters across data-parallel ranks. |
| **z-loss** | auxiliary loss on $\log Z$; stabilizes training and helps quantization. |

---

**Next** → [Math cheatsheet](02-math-cheatsheet.md)
