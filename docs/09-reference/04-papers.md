# Paper List

> ~95 papers that matter, annotated with what to take from each. ⭐ marks the ~25 that are worth
> reading in full; the rest are worth knowing the result of.
> Ordered by topic, with a suggested reading order at the end.

---

## Foundations

| Paper | Year | Take-away |
|---|---|---|
| Shannon, *A Mathematical Theory of Communication* ⭐ | 1948 | entropy, cross-entropy, compression ↔ prediction |
| Hochreiter & Schmidhuber, *Long Short-Term Memory* | 1997 | the additive memory path |
| Bengio et al., *A Neural Probabilistic Language Model* | 2003 | learned embeddings; generalization by parameter sharing |
| Hinton, *Training Products of Experts by Minimizing Contrastive Divergence* | 2002 | made EBMs trainable |
| Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching* | 2005 | score matching without $Z$ |
| Vincent, *A Connection Between Score Matching and Denoising Autoencoders* ⭐ | 2011 | **the bridge to diffusion** |
| Krizhevsky et al., *ImageNet Classification with Deep CNNs* (AlexNet) | 2012 | GPUs + depth + data |
| He et al., *Deep Residual Learning* ⭐ | 2015 | residual connections |
| Kingma & Ba, *Adam* | 2014 | the default optimizer |
| Loshchilov & Hutter, *Decoupled Weight Decay Regularization* | 2017 | AdamW; Adam + L2 is wrong |
| Zhang & Sennrich, *Root Mean Square Layer Normalization* | 2019 | RMSNorm |
| Shazeer, *GLU Variants Improve Transformer* | 2020 | SwiGLU |
| Xiong et al., *On Layer Normalization in the Transformer Architecture* | 2020 | pre-norm vs post-norm, with the analysis |

## Classical generative models

| Paper | Year | Take-away |
|---|---|---|
| Kingma & Welling, *Auto-Encoding Variational Bayes* ⭐ | 2013 | VAE; the reparameterization trick |
| Kingma & Welling, *An Introduction to Variational Autoencoders* | 2019 | the better tutorial version |
| Goodfellow et al., *Generative Adversarial Networks* ⭐ | 2014 | the adversarial game |
| Arjovsky et al., *Wasserstein GAN* | 2017 | why JSD fails; earth-mover distance |
| Gulrajani et al., *Improved Training of WGANs* | 2017 | gradient penalty |
| Miyato et al., *Spectral Normalization for GANs* | 2018 | the cheapest Lipschitz constraint |
| Karras et al., *A Style-Based Generator Architecture* (StyleGAN) ⭐ | 2018 | the mapping network; latent editing |
| Karras et al., *Analyzing and Improving StyleGAN* | 2019 | StyleGAN2 |
| Dinh et al., *Density Estimation using Real NVP* | 2016 | affine coupling; triangular Jacobians |
| Kingma & Dhariwal, *Glow* | 2018 | invertible 1×1 convolutions |
| Papamakarios et al., *Normalizing Flows for Probabilistic Modeling* | 2021 | the definitive survey |
| van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE) ⭐ | 2017 | discrete latents; straight-through |
| Esser et al., *Taming Transformers* (VQGAN) | 2020 | VQ + perceptual + adversarial |
| Du & Mordatch, *Implicit Generation and Modeling with EBMs* | 2019 | EBMs on images |
| Grathwohl et al., *Your Classifier is Secretly an EBM* | 2019 | JEM |
| Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?* | 2018 | high likelihood ≠ in-distribution |
| Bond-Taylor et al., *Deep Generative Modelling: A Comparative Review* | 2021 | the best cross-family survey |

## Sequence models & the Transformer

| Paper | Year | Take-away |
|---|---|---|
| Bahdanau et al., *Neural MT by Jointly Learning to Align and Translate* ⭐ | 2015 | attention's origin |
| **Vaswani et al., *Attention Is All You Need*** ⭐⭐ | 2017 | the architecture |
| Sennrich et al., *NMT of Rare Words with Subword Units* | 2016 | BPE |
| Kudo, *Subword Regularization* | 2018 | the Unigram LM tokenizer |
| Kudo & Richardson, *SentencePiece* | 2018 | language-agnostic tokenization |
| Su et al., *RoFormer* (RoPE) ⭐ | 2021 | rotary position embedding |
| Press et al., *Train Short, Test Long* (ALiBi) | 2021 | linear position bias |
| Ainslie et al., *GQA* | 2023 | 8× smaller KV cache |
| Dao et al., *FlashAttention* ⭐ | 2022 | tiling + online softmax; exact, $O(T)$ memory |
| Gu & Dao, *Mamba* ⭐ | 2023 | selective state-space models |
| Elhage et al., *A Mathematical Framework for Transformer Circuits* ⭐ | 2021 | the residual stream view |
| Olsson et al., *In-context Learning and Induction Heads* ⭐ | 2022 | a circuit tied to a capability |

## LLMs: pretraining, scaling, efficiency

| Paper | Year | Take-away |
|---|---|---|
| Radford et al., *Improving Language Understanding by Generative Pre-Training* | 2018 | GPT-1 |
| Devlin et al., *BERT* | 2018 | bidirectional pretraining |
| Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) | 2019 | byte-level BPE; zero-shot transfer |
| **Brown et al., *Language Models are Few-Shot Learners*** (GPT-3) ⭐ | 2020 | in-context learning |
| Raffel et al., *Exploring the Limits of Transfer Learning* (T5) | 2020 | systematic architecture comparison |
| **Kaplan et al., *Scaling Laws for Neural Language Models*** ⭐ | 2020 | power laws |
| **Hoffmann et al., *Training Compute-Optimal LLMs*** (Chinchilla) ⭐ | 2022 | 20 tokens/param |
| Sardana et al., *Beyond Chinchilla-Optimal* | 2023 | account for inference cost |
| Muennighoff et al., *Scaling Data-Constrained Language Models* | 2023 | repeating data is fine for ~4 epochs |
| Touvron et al., *LLaMA* / *LLaMA 2* ⭐ | 2023 | the open reference architecture |
| Grattafiori et al., *The Llama 3 Herd of Models* ⭐ | 2024 | the most detailed public account of a real training run |
| Chowdhery et al., *PaLM* | 2022 | honest on training instabilities |
| Rajbhandari et al., *ZeRO* | 2020 | memory sharding |
| Narayanan et al., *Efficient Large-Scale LM Training on GPU Clusters* (Megatron) | 2021 | 3-D parallelism |
| Lee et al., *Deduplicating Training Data Makes LMs Better* ⭐ | 2022 | dedup improves loss *and* cuts memorization |
| Penedo et al., *The FineWeb Datasets* | 2024 | an open, documented filtering pipeline |
| Yang et al., *Tensor Programs V* (μP) | 2022 | hyperparameter transfer across scale |
| Shazeer et al., *Outrageously Large Neural Networks* (MoE) | 2017 | sparse gating |
| Fedus et al., *Switch Transformers* | 2021 | top-1 routing at scale |
| Jiang et al., *Mixtral of Experts* | 2024 | open MoE; expert specialization analysis |
| DeepSeek-AI, *DeepSeek-V3 Technical Report* ⭐ | 2024 | fine-grained + shared experts; MLA; loss-free balancing |
| Dettmers et al., *LLM.int8()* | 2022 | the outlier discovery |
| Xiao et al., *SmoothQuant* | 2022 | migrate outliers into the weights |
| Frantar et al., *GPTQ* / Lin et al., *AWQ* | 2022/23 | 4-bit weight quantization |
| Ma et al., *The Era of 1-bit LLMs* (BitNet b1.58) | 2024 | ternary weights, trained natively |
| Hinton et al., *Distilling the Knowledge in a Neural Network* | 2015 | dark knowledge |
| Sun et al., *Wanda* | 2023 | pruning by $\|W\|\cdot\|X\|$; simple and effective |

## Fine-tuning, alignment & reasoning

| Paper | Year | Take-away |
|---|---|---|
| **Hu et al., *LoRA*** ⭐ | 2021 | low-rank adaptation |
| **Dettmers et al., *QLoRA*** ⭐ | 2023 | NF4 + double quant + paged optimizers |
| Liu et al., *DoRA* | 2024 | magnitude/direction decomposition |
| Biderman et al., *LoRA Learns Less and Forgets Less* | 2024 | the honest LoRA vs full-FT comparison |
| Zhou et al., *LIMA* | 2023 | 1,000 examples; the superficial alignment hypothesis |
| **Ouyang et al., *Training LMs to Follow Instructions with Human Feedback*** ⭐ | 2022 | InstructGPT / RLHF |
| **Rafailov et al., *Direct Preference Optimization*** ⭐ | 2023 | the derivation removing the reward model |
| Bai et al., *Constitutional AI* ⭐ | 2022 | AI feedback from written principles |
| Shao et al., *DeepSeekMath* | 2024 | GRPO |
| **DeepSeek-AI, *DeepSeek-R1*** ⭐ | 2025 | pure RL → emergent reasoning |
| Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* ⭐ | 2022 | true preference peaks then declines |
| Casper et al., *Open Problems and Fundamental Limitations of RLHF* | 2023 | the honest critique |
| **Wei et al., *Chain-of-Thought Prompting*** ⭐ | 2022 | reasoning via intermediate tokens |
| Kojima et al., *Large Language Models are Zero-Shot Reasoners* | 2022 | "let's think step by step" |
| Wang et al., *Self-Consistency* ⭐ | 2022 | sample $n$, majority vote |
| Lightman et al., *Let's Verify Step by Step* ⭐ | 2023 | process reward models; PRM800K |
| Yao et al., *Tree of Thoughts* | 2023 | search over reasoning |
| Snell et al., *Scaling LLM Test-Time Compute Optimally* ⭐ | 2024 | the second scaling axis |
| Merrill & Sabharwal, *The Expressive Power of Transformers with CoT* ⭐ | 2024 | the complexity-theoretic justification for CoT |
| Turpin et al., *Language Models Don't Always Say What They Think* ⭐ | 2023 | CoT is often unfaithful |

## Inference & serving

| Paper | Year | Take-away |
|---|---|---|
| **Holtzman et al., *The Curious Case of Neural Text Degeneration*** ⭐ | 2019 | nucleus sampling; why MLE decoding fails |
| Leviathan et al., *Fast Inference via Speculative Decoding* ⭐ | 2023 | 2–3× speedup, **exact** distribution |
| **Kwon et al., *Efficient Memory Management with PagedAttention*** (vLLM) ⭐ | 2023 | OS paging for the KV cache |
| Pope et al., *Efficiently Scaling Transformer Inference* | 2022 | the arithmetic-intensity analysis |
| Xiao et al., *Efficient Streaming LMs with Attention Sinks* | 2023 | why token 0 matters |
| Willard & Louf, *Efficient Guided Generation for LLMs* (Outlines) | 2023 | FSM-based constrained decoding |
| Chen et al., *Extending Context Window via Position Interpolation* | 2023 | PI |
| Peng et al., *YaRN* | 2023 | the best context-extension method |

## Diffusion & vision

| Paper | Year | Take-away |
|---|---|---|
| Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* | 2015 | the idea, five years early |
| Song & Ermon, *Generative Modeling by Estimating Gradients* (NCSN) ⭐ | 2019 | annealed Langevin |
| **Ho et al., *Denoising Diffusion Probabilistic Models*** ⭐ | 2020 | the simple loss |
| Song et al., *Denoising Diffusion Implicit Models* (DDIM) ⭐ | 2020 | deterministic sampling |
| **Song et al., *Score-Based Generative Modeling through SDEs*** ⭐ | 2021 | the unification; probability-flow ODE |
| Nichol & Dhariwal, *Improved DDPM* | 2021 | cosine schedule |
| **Dhariwal & Nichol, *Diffusion Models Beat GANs*** ⭐ | 2021 | classifier guidance; the changing of the guard |
| Ho & Salimans, *Classifier-Free Diffusion Guidance* ⭐ | 2022 | the CFG formula |
| **Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion***  ⭐⭐ | 2022 | Stable Diffusion |
| Saharia et al., *Photorealistic Text-to-Image Diffusion* (Imagen) | 2022 | the text encoder matters most |
| **Karras et al., *Elucidating the Design Space of Diffusion Models*** (EDM) ⭐ | 2022 | the best systematic study; the $\sigma$ schedule |
| Lu et al., *DPM-Solver++* | 2022 | exploiting the semi-linear ODE structure |
| Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT) ⭐ | 2023 | Transformers scale better than U-Nets |
| Zhang et al., *Adding Conditional Control* (ControlNet) ⭐ | 2023 | zero-init, frozen base |
| Song et al., *Consistency Models* | 2023 | 1-step generation |
| **Lipman et al., *Flow Matching for Generative Modeling*** ⭐ | 2023 | simulation-free CNFs |
| Liu et al., *Flow Straight and Fast* (Rectified Flow) ⭐ | 2022 | straight paths; reflow |
| Esser et al., *Scaling Rectified Flow Transformers* (SD3) ⭐ | 2024 | the practical details: logit-normal $t$, resolution shift, MMDiT |
| Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT) ⭐ | 2020 | patches as tokens |
| **Radford et al., *Learning Transferable Visual Models*** (CLIP) ⭐ | 2021 | the shared embedding space |
| Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP) | 2023 | no global batch normalization |
| Liu et al., *Visual Instruction Tuning* (LLaVA) ⭐ | 2023 | the cheap projection-based VLM recipe |
| Alayrac et al., *Flamingo* | 2022 | gated cross-attention VLMs |
| Défossez et al., *EnCodec* | 2022 | residual VQ audio tokens |

## Retrieval & agents

| Paper | Year | Take-away |
|---|---|---|
| Lewis et al., *Retrieval-Augmented Generation* ⭐ | 2020 | the original RAG |
| Karpukhin et al., *Dense Passage Retrieval* | 2020 | dense retrieval with hard negatives |
| Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* | 2009 | the sparse baseline you should still use |
| Malkov & Yashunin, *HNSW* | 2016 | the quality-default ANN index |
| Jégou et al., *Product Quantization* | 2011 | 32× index compression |
| Gao et al., *Precise Zero-Shot Dense Retrieval* (HyDE) | 2022 | embed a hypothetical answer |
| Kusupati et al., *Matryoshka Representation Learning* | 2022 | truncatable embeddings |
| **Liu et al., *Lost in the Middle*** ⭐ | 2023 | the U-shaped attention curve |
| Barnett et al., *Seven Failure Points When Engineering a RAG System* | 2024 | the failure taxonomy |
| Edge et al., *From Local to Global* (GraphRAG) | 2024 | corpus-wide questions |
| **Yao et al., *ReAct*** ⭐ | 2022 | reasoning + acting |
| Schick et al., *Toolformer* | 2023 | self-supervised tool learning |
| Shinn et al., *Reflexion* | 2023 | verbal self-critique |
| Jimenez et al., *SWE-bench* ⭐ | 2023 | realistic agentic evaluation |

## Evaluation, safety & security

| Paper | Year | Take-away |
|---|---|---|
| Hendrycks et al., *MMLU* | 2020 | the broad-knowledge standard |
| Chen et al., *Evaluating LLMs Trained on Code* (HumanEval) | 2021 | the pass@k estimator |
| Rein et al., *GPQA* | 2023 | genuinely hard, Google-proof |
| Zheng et al., *Judging LLM-as-a-Judge* ⭐ | 2023 | judge biases, measured |
| Chiang et al., *Chatbot Arena* | 2024 | preference Elo at scale |
| Liang et al., *Holistic Evaluation of Language Models* (HELM) | 2022 | the most thorough framework |
| Zhang et al., *A Careful Examination of LLM Performance on Grade School Arithmetic* (GSM1k) ⭐ | 2024 | contamination, quantified |
| Hsieh et al., *RULER* ⭐ | 2024 | effective vs advertised context |
| Schaeffer et al., *Are Emergent Abilities of LLMs a Mirage?* ⭐ | 2023 | metric discontinuity |
| **Wei et al., *Jailbroken: How Does LLM Safety Training Fail?*** ⭐ | 2023 | competing objectives; mismatched generalization |
| Zou et al., *Universal and Transferable Adversarial Attacks* (GCG) | 2023 | optimized adversarial suffixes |
| Anil et al., *Many-shot Jailbreaking* | 2024 | long context as an attack surface |
| Sharma et al., *Towards Understanding Sycophancy* ⭐ | 2023 | raters reward agreement |
| Kalai & Vempala, *Calibrated Language Models Must Hallucinate* | 2024 | a lower bound on hallucination |
| Carlini et al., *Quantifying Memorization* | 2022 | scales with duplicates and size |
| **Greshake et al., *Indirect Prompt Injection*** ⭐ | 2023 | the attack that has no complete fix |
| Nasr et al., *Scalable Extraction of Training Data* | 2023 | the divergence attack |
| Hubinger et al., *Sleeper Agents* ⭐ | 2024 | backdoors survive safety training |
| Templeton et al., *Scaling Monosemanticity* ⭐ | 2024 | SAE features in a production model |
| Elhage et al., *Toy Models of Superposition* ⭐ | 2022 | why neurons are polysemantic |
| Bender et al., *On the Dangers of Stochastic Parrots* | 2021 | the critical perspective, worth engaging with |
| Hofmann et al., *AI Generates Covertly Racist Decisions* | 2024 | alignment hides bias rather than removing it |

---

## Suggested reading order

If you read **ten papers**, read these, in this order:

1. Vaswani et al., *Attention Is All You Need* (2017)
2. Kingma & Welling, *Auto-Encoding Variational Bayes* (2013)
3. Ho et al., *DDPM* (2020)
4. Brown et al., *GPT-3* (2020)
5. Hoffmann et al., *Chinchilla* (2022)
6. Ouyang et al., *InstructGPT* (2022)
7. Rombach et al., *Latent Diffusion* (2022)
8. Rafailov et al., *DPO* (2023)
9. Song et al., *Score-Based Generative Modeling through SDEs* (2021)
10. DeepSeek-AI, *DeepSeek-R1* (2025)

**Then**, depending on direction:
- *Building systems* → RAG, ReAct, vLLM, Lost in the Middle, Indirect Prompt Injection
- *Training models* → LLaMA 3, ZeRO, μP, FineWeb, Deduplication
- *Image/video* → EDM, Flow Matching, SD3, DiT, ControlNet
- *Research* → Transformer Circuits, Superposition, Induction Heads, Scaling Monosemanticity

🧠 **How to read an ML paper efficiently**: abstract → figures → the main equation → the ablation
table → related work. Read the method section in full only if you intend to implement it. Most
papers have one idea; find it and move on.

---

**Next** → [Study roadmap](05-roadmap.md)
