# Timeline

> The history of generative AI, with **why each step mattered** rather than just what happened.
> Entries marked ⭐ are the ones whose removal would most change the field.

---

## Prehistory (1943–2005)

| Year | Event | Why it mattered |
|---|---|---|
| 1943 | McCulloch & Pitts neuron | the first mathematical model of a neuron |
| 1948 | **Shannon**, *A Mathematical Theory of Communication* ⭐ | entropy, cross-entropy, and the compression↔prediction equivalence the whole field rests on |
| 1950 | Turing, *Computing Machinery and Intelligence* | framed the question |
| 1958 | Rosenblatt's perceptron | learning from data |
| 1969 | Minsky & Papert, *Perceptrons* | showed the XOR limitation; contributed to the first AI winter |
| 1982 | Hopfield network | energy-based associative memory (2024 Nobel Prize in Physics) |
| 1985 | **Boltzmann machine** | the first explicitly probabilistic generative neural model |
| 1986 | Backpropagation popularized (Rumelhart, Hinton, Williams) | made multilayer training practical |
| 1989 | LeNet / CNNs | convolution, weight sharing, translation equivariance |
| 1997 | **LSTM** (Hochreiter & Schmidhuber) ⭐ | the additive memory path solved vanishing gradients |
| 2002 | Contrastive divergence (Hinton) | made RBMs trainable |
| 2003 | **Bengio et al., neural LM** ⭐ | learned word embeddings; generalization by parameter sharing |

---

## The deep learning revival (2006–2016)

| Year | Event | Why it mattered |
|---|---|---|
| 2006 | Deep belief networks | layerwise pretraining; started the "deep learning" label |
| 2009 | **ImageNet** ⭐ | the benchmark that made scale measurable |
| 2012 | **AlexNet** ⭐ | GPU training; proved deep nets scale with compute. The starting gun. |
| 2013 | **word2vec** | embeddings as a practical tool; `king − man + woman ≈ queen` |
| 2013 | **VAE** (Kingma & Welling) ⭐ | the reparameterization trick made deep latent-variable models trainable |
| 2014 | **GAN** (Goodfellow et al.) ⭐ | adversarial training; the first sharp deep generative images |
| 2014 | Seq2seq (Sutskever et al.) | encoder–decoder for translation |
| 2014 | Adam optimizer | the default optimizer for a decade |
| 2015 | **Attention** (Bahdanau et al.) ⭐ | removed the fixed-size bottleneck; the direct ancestor of the Transformer |
| 2015 | **ResNet** ⭐ | residual connections made very deep networks trainable |
| 2015 | BatchNorm | training stability |
| 2015 | **Diffusion** (Sohl-Dickstein et al.) | the correct idea, five years before the field noticed |
| 2016 | PixelCNN, WaveNet | deep autoregressive models for images and audio |
| 2016 | RealNVP | normalizing flows with tractable Jacobians |
| 2016 | AlphaGo | RL + search beats humans at Go |

---

## The Transformer era (2017–2020)

| Year | Event | Why it mattered |
|---|---|---|
| 2017 | **Transformer** (Vaswani et al.) ⭐⭐ | replaced sequential recurrence with parallel attention. Traded asymptotics for parallelism — the single most consequential architecture decision in the field. |
| 2017 | MoE layer (Shazeer et al.) | decoupled capacity from compute |
| 2017 | WGAN, spectral norm | made GAN training tractable |
| 2017 | PPO | the RL algorithm later used for RLHF |
| 2018 | **BERT** ⭐ | pretrain-then-fine-tune became the NLP standard |
| 2018 | **GPT-1** | decoder-only generative pretraining |
| 2018 | Glow | flows at scale |
| 2019 | **GPT-2** | byte-level BPE; zero-shot task transfer; the staged-release debate |
| 2019 | **StyleGAN** | photorealistic faces; the mapping network and latent editing |
| 2019 | T5 | text-to-text unification; systematic architecture comparison |
| 2019 | **Score matching / NCSN** (Song & Ermon) ⭐ | annealed Langevin sampling; the other half of diffusion |
| 2020 | **Scaling laws** (Kaplan et al.) ⭐ | made capability *predictable*; turned scaling into engineering |
| 2020 | **GPT-3** ⭐⭐ | 175 B parameters; in-context learning; the moment the public noticed |
| 2020 | **DDPM** (Ho et al.) ⭐ | the simple noise-prediction loss; diffusion became practical |
| 2020 | ViT | images as patch sequences; one architecture for everything |
| 2020 | RAG (Lewis et al.) | retrieval + generation |
| 2020 | ZeRO | memory sharding; trillion-parameter training became possible |

---

## Alignment and diffusion (2021–2022)

| Year | Event | Why it mattered |
|---|---|---|
| 2021 | **CLIP** ⭐ | a shared image–text space; enabled text-conditioned generation |
| 2021 | DALL·E 1 | discrete VQ tokens + autoregressive Transformer for images |
| 2021 | **Score SDE** (Song et al.) ⭐ | unified diffusion and score matching; introduced the probability-flow ODE |
| 2021 | DDIM | deterministic sampling; 1000 → 20 steps |
| 2021 | **Diffusion beats GANs** (Dhariwal & Nichol) ⭐ | the changing of the guard in image generation |
| 2021 | Classifier-free guidance | the conditioning mechanism every text-to-image model uses |
| 2021 | **LoRA** ⭐ | fine-tuning became affordable |
| 2021 | Codex / GitHub Copilot | code generation as a product |
| 2022 | **Chinchilla** ⭐ | corrected the scaling law; 20 tokens/param; reorganized the field |
| 2022 | **InstructGPT / RLHF** ⭐⭐ | alignment turned a text predictor into an assistant |
| 2022 | **Chain-of-thought** ⭐ | reasoning as a prompting technique |
| 2022 | **Latent diffusion / Stable Diffusion** ⭐⭐ | 8× compression made image generation run on consumer hardware — and open weights changed the ecosystem |
| 2022 | Imagen | showed the text encoder matters more than the diffusion model |
| 2022 | **FlashAttention** ⭐ | exact attention with $O(T)$ memory; unlocked long context |
| 2022 | Constitutional AI | AI feedback guided by explicit written principles |
| 2022 | **ChatGPT** ⭐⭐ | the interface, not the model — generative AI became a mass-market product |

---

## Scale, openness and agents (2023–2024)

| Year | Event | Why it mattered |
|---|---|---|
| 2023 | GPT-4 | multimodal, large capability jump, closed details |
| 2023 | **LLaMA / LLaMA 2** ⭐⭐ | open weights at frontier-adjacent quality; created the open ecosystem |
| 2023 | **QLoRA** ⭐ | 65 B fine-tuning on a single GPU |
| 2023 | **DPO** ⭐ | alignment without a reward model or an RL loop |
| 2023 | Mistral 7B | small models punching far above their weight; sliding-window attention |
| 2023 | **vLLM / PagedAttention** ⭐ | LLM serving became efficient; 60–80% KV waste → <4% |
| 2023 | ControlNet | precise spatial control over image generation |
| 2023 | SDXL | higher resolution, better VAE |
| 2023 | Mamba | selective state-space models; recurrence returns |
| 2023 | GQA | 8× smaller KV cache; adopted universally |
| 2023 | Speculative decoding | 2–3× faster inference with identical output distribution |
| 2023 | **Flow matching** ⭐ | simulation-free continuous flows; straight paths |
| 2024 | Mixtral 8×7B | open MoE; active vs total parameters entered common vocabulary |
| 2024 | **Sora** and video models | video generation became credible |
| 2024 | LLaMA 3 / 3.1 | 15 T tokens; 405 B open weights; 128 k context |
| 2024 | **SD3 / Flux** | rectified flow + MMDiT; text rendering in images |
| 2024 | **Test-time compute scaling** ⭐ | reasoning models; a second scaling axis that needs no retraining |
| 2024 | DeepSeek-V3 | fine-grained MoE, MLA, loss-free load balancing; frontier quality at low cost |
| 2024 | Sparse autoencoders at scale | interpretability extracted millions of features from a production model |

---

## 2025–2026

| Year | Event | Why it mattered |
|---|---|---|
| 2025 | **DeepSeek-R1** ⭐ | pure RL on verifiable rewards produced emergent long reasoning, self-verification and backtracking — **and it was published openly** |
| 2025 | Reasoning models mainstream | thinking budgets as a product feature; inference compute as a dial |
| 2025 | Agentic coding | SWE-bench scores rose sharply; agents became practically useful |
| 2025–26 | Long context standard | 128 k–1 M windows common; RULER exposed the gap between advertised and effective context |
| 2025–26 | Small models catch up | 8 B models matching 2023's 70 B on many tasks — data and post-training, not architecture |
| 2025–26 | Multimodal by default | native image/audio/video input and output in frontier models |
| 2025–26 | Regulation phases in | EU AI Act obligations; provenance and disclosure requirements |

---

## The pattern

```
  1943 ──── 2012 ──────── 2017 ──── 2020 ──── 2022 ──── 2025 ────►
   │          │             │         │         │         │
  ideas    COMPUTE      ARCHITECTURE  SCALE  ALIGNMENT  TEST-TIME
  exist    arrives      converges    works   makes it   COMPUTE
  but              (Transformer)            usable      (a second axis)
  don't                                                    
  work

  Each era's bottleneck became the next era's solved problem.
```

🧠 **Three observations worth carrying away:**

1. **Most "new" ideas are old ideas that finally had enough compute.** Diffusion (1949 physics,
   2015 ML), attention (1990s), neural LMs (2003), MoE (1991), RL from human preferences (2017).
   The 2015 diffusion paper is nearly identical in concept to DDPM; it lacked the compute, the
   architecture, and the noise-prediction simplification.

2. **The compounding factors were compute, then data, then post-training, then inference compute.**
   Architecture converged around 2019 and has barely changed since — the modern Transformer is the
   2017 one with things removed.

3. **The interface mattered as much as the model.** ChatGPT's underlying model was not new. Making
   it a conversation was.

---

**Next** → [Paper list](04-papers.md)
