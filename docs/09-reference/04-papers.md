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
| He et al., [*Deep Residual Learning*](https://arxiv.org/abs/1512.03385) ⭐ | 2015 | residual connections |
| Kingma & Ba, [*Adam*](https://arxiv.org/abs/1412.6980) | 2014 | the default optimizer |
| Loshchilov & Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) | 2017 | AdamW; Adam + L2 is wrong |
| Zhang & Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) | 2019 | RMSNorm |
| Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) | 2020 | SwiGLU |
| Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) | 2020 | pre-norm vs post-norm, with the analysis |

## Classical generative models

| Paper | Year | Take-away |
|---|---|---|
| Kingma & Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) ⭐ | 2013 | VAE; the reparameterization trick |
| Kingma & Welling, [*An Introduction to Variational Autoencoders*](https://arxiv.org/abs/1906.02691) | 2019 | the better tutorial version |
| Goodfellow et al., [*Generative Adversarial Networks*](https://arxiv.org/abs/1406.2661) ⭐ | 2014 | the adversarial game |
| Arjovsky et al., [*Wasserstein GAN*](https://arxiv.org/abs/1701.07875) | 2017 | why JSD fails; earth-mover distance |
| Gulrajani et al., [*Improved Training of WGANs*](https://arxiv.org/abs/1704.00028) | 2017 | gradient penalty |
| Miyato et al., [*Spectral Normalization for GANs*](https://arxiv.org/abs/1802.05957) | 2018 | the cheapest Lipschitz constraint |
| Karras et al., [*A Style-Based Generator Architecture*](https://arxiv.org/abs/1812.04948) (StyleGAN) ⭐ | 2018 | the mapping network; latent editing |
| Karras et al., [*Analyzing and Improving StyleGAN*](https://arxiv.org/abs/1912.04958) | 2019 | StyleGAN2 |
| Dinh et al., [*Density Estimation using Real NVP*](https://arxiv.org/abs/1605.08803) | 2016 | affine coupling; triangular Jacobians |
| Kingma & Dhariwal, [*Glow*](https://arxiv.org/abs/1807.03039) | 2018 | invertible 1×1 convolutions |
| Papamakarios et al., [*Normalizing Flows for Probabilistic Modeling*](https://arxiv.org/abs/1912.02762) | 2021 | the definitive survey |
| van den Oord et al., [*Neural Discrete Representation Learning*](https://arxiv.org/abs/1711.00937) (VQ-VAE) ⭐ | 2017 | discrete latents; straight-through |
| Esser et al., [*Taming Transformers*](https://arxiv.org/abs/2012.09841) (VQGAN) | 2020 | VQ + perceptual + adversarial |
| Du & Mordatch, *Implicit Generation and Generalization in EBMs* | 2019 | EBMs on images |
| Grathwohl et al., [*Your Classifier is Secretly an EBM*](https://arxiv.org/abs/1912.03263) | 2019 | JEM |
| Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?* | 2018 | high likelihood ≠ in-distribution |
| Bond-Taylor et al., *Deep Generative Modelling: A Comparative Review* | 2021 | the best cross-family survey |

## Sequence models and the Transformer

| Paper | Year | Take-away |
|---|---|---|
| Bahdanau et al., [*Neural MT by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) ⭐ | 2015 | attention's origin |
| **Vaswani et al., *Attention Is All You Need*** ⭐⭐ | 2017 | the architecture |
| Sennrich et al., [*NMT of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) | 2016 | BPE |
| Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959) | 2018 | the Unigram LM tokenizer |
| Kudo & Richardson, [*SentencePiece*](https://arxiv.org/abs/1808.06226) | 2018 | language-agnostic tokenization |
| Su et al., [*RoFormer*](https://arxiv.org/abs/2104.09864) (RoPE) ⭐ | 2021 | rotary position embedding |
| Press et al., [*Train Short, Test Long*](https://arxiv.org/abs/2108.12409) (ALiBi) | 2021 | linear position bias |
| Ainslie et al., *GQA* | 2023 | 8× smaller KV cache |
| Dao et al., [*FlashAttention*](https://arxiv.org/abs/2307.08691) ⭐ | 2022 | tiling + online softmax; exact, $O(T)$ memory |
| Gu & Dao, [*Mamba*](https://arxiv.org/abs/2312.00752) ⭐ | 2023 | selective state-space models |
| Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) ⭐ | 2021 | the residual stream view |
| Olsson et al., [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) ⭐ | 2022 | a circuit tied to a capability |

## LLMs: pretraining, scaling, efficiency

| Paper | Year | Take-away |
|---|---|---|
| Radford et al., *Improving Language Understanding by Generative Pre-Training* | 2018 | GPT-1 |
| Devlin et al., [*BERT*](https://arxiv.org/abs/1810.04805) | 2018 | bidirectional pretraining |
| Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) | 2019 | byte-level BPE; zero-shot transfer |
| **Brown et al., *Language Models are Few-Shot Learners*** (GPT-3) ⭐ | 2020 | in-context learning |
| Raffel et al., [*Exploring the Limits of Transfer Learning*](https://arxiv.org/abs/1910.10683) (T5) | 2020 | systematic architecture comparison |
| **Kaplan et al., *Scaling Laws for Neural Language Models*** ⭐ | 2020 | power laws |
| **Hoffmann et al., *Training Compute-Optimal LLMs*** (Chinchilla) ⭐ | 2022 | 20 tokens/param |
| Sardana et al., [*Beyond Chinchilla-Optimal*](https://arxiv.org/abs/2401.00448) | 2023 | account for inference cost |
| Muennighoff et al., [*Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264) | 2023 | repeating data is fine for ~4 epochs |
| Touvron et al., [*LLaMA*](https://arxiv.org/abs/2302.13971) / [*LLaMA 2*](https://arxiv.org/abs/2307.09288) ⭐ | 2023 | the open reference architecture |
| Grattafiori et al., [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) ⭐ | 2024 | the most detailed public account of a real training run |
| Chowdhery et al., [*PaLM*](https://arxiv.org/abs/2204.02311) | 2022 | honest on training instabilities |
| Rajbhandari et al., [*ZeRO*](https://arxiv.org/abs/1910.02054) | 2020 | memory sharding |
| Narayanan et al., [*Efficient Large-Scale LM Training on GPU Clusters*](https://arxiv.org/abs/2104.04473) (Megatron) | 2021 | 3-D parallelism |
| Lee et al., [*Deduplicating Training Data Makes LMs Better*](https://arxiv.org/abs/2107.06499) ⭐ | 2022 | dedup improves loss *and* cuts memorization |
| Penedo et al., [*The FineWeb Datasets*](https://arxiv.org/abs/2406.17557) | 2024 | an open, documented filtering pipeline |
| Yang et al., [*Tensor Programs V*](https://arxiv.org/abs/2310.02244) (μP) | 2022 | hyperparameter transfer across scale |
| Shazeer et al., [*Outrageously Large Neural Networks*](https://arxiv.org/abs/1701.06538) (MoE) | 2017 | sparse gating |
| Fedus et al., [*Switch Transformers*](https://arxiv.org/abs/2101.03961) | 2021 | top-1 routing at scale |
| Jiang et al., [*Mixtral of Experts*](https://arxiv.org/abs/2401.04088) | 2024 | open MoE; expert specialization analysis |
| DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) ⭐ | 2024 | fine-grained + shared experts; MLA; loss-free balancing |
| Dettmers et al., *LLM.int8()* | 2022 | the outlier discovery |
| Xiao et al., [*SmoothQuant*](https://arxiv.org/abs/2211.10438) | 2022 | migrate outliers into the weights |
| Frantar et al., [*GPTQ*](https://arxiv.org/abs/2210.17323) / Lin et al., *AWQ* | 2022/23 | 4-bit weight quantization |
| Ma et al., [*The Era of 1-bit LLMs*](https://arxiv.org/abs/2402.17764) (BitNet b1.58) | 2024 | ternary weights, trained natively |
| Hinton et al., [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) | 2015 | dark knowledge |
| Sun et al., [*Wanda*](https://arxiv.org/abs/2306.11695) | 2023 | pruning by $\|W\|\cdot\|X\|$; simple and effective |

## Fine-tuning, alignment and reasoning

| Paper | Year | Take-away |
|---|---|---|
| **Hu et al., *LoRA*** ⭐ | 2021 | low-rank adaptation |
| **Dettmers et al., *QLoRA*** ⭐ | 2023 | NF4 + double quant + paged optimizers |
| Liu et al., [*DoRA*](https://arxiv.org/abs/2402.09353) | 2024 | magnitude/direction decomposition |
| Biderman et al., [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673) | 2024 | the honest LoRA vs full-FT comparison |
| Zhou et al., [*LIMA*](https://arxiv.org/abs/2305.11206) | 2023 | 1,000 examples; the superficial alignment hypothesis |
| **Ouyang et al., *Training LMs to Follow Instructions with Human Feedback*** ⭐ | 2022 | InstructGPT / RLHF |
| **Rafailov et al., *Direct Preference Optimization*** ⭐ | 2023 | the derivation removing the reward model |
| Bai et al., [*Constitutional AI*](https://arxiv.org/abs/2212.08073) ⭐ | 2022 | AI feedback from written principles |
| Shao et al., [*DeepSeekMath*](https://arxiv.org/abs/2511.22570) | 2024 | GRPO |
| **DeepSeek-AI, *DeepSeek-R1*** ⭐ | 2025 | pure RL → emergent reasoning |
| Gao, Schulman & Hilton, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) ⭐ | 2022 | true preference peaks then declines |
| Casper et al., [*Open Problems and Fundamental Limitations of RLHF*](https://arxiv.org/abs/2307.15217) | 2023 | the honest critique |
| **Wei et al., *Chain-of-Thought Prompting*** ⭐ | 2022 | reasoning via intermediate tokens |
| Kojima et al., [*Large Language Models are Zero-Shot Reasoners*](https://arxiv.org/abs/2205.11916) | 2022 | "let's think step by step" |
| Wang et al., [*Self-Consistency*](https://arxiv.org/abs/2505.10772) ⭐ | 2022 | sample $n$, majority vote |
| Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050) ⭐ | 2023 | process reward models; PRM800K |
| Yao et al., [*Tree of Thoughts*](https://arxiv.org/abs/2305.10601) | 2023 | search over reasoning |
| Snell et al., [*Scaling LLM Test-Time Compute Optimally*](https://arxiv.org/abs/2408.03314) ⭐ | 2024 | the second scaling axis |
| Merrill & Sabharwal, [*The Expressive Power of Transformers with CoT*](https://arxiv.org/abs/2310.07923) ⭐ | 2024 | the complexity-theoretic justification for CoT |
| Turpin et al., *Language Models Don't Always Say What They Think* ⭐ | 2023 | CoT is often unfaithful |

## Inference and serving

| Paper | Year | Take-away |
|---|---|---|
| **Holtzman et al., *The Curious Case of Neural Text Degeneration*** ⭐ | 2019 | nucleus sampling; why MLE decoding fails |
| Leviathan et al., [*Fast Inference via Speculative Decoding*](https://arxiv.org/abs/2211.17192) ⭐ | 2023 | 2–3× speedup, **exact** distribution |
| **Kwon et al., *Efficient Memory Management with PagedAttention*** (vLLM) ⭐ | 2023 | OS paging for the KV cache |
| Pope et al., [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102) | 2022 | the arithmetic-intensity analysis |
| Xiao et al., [*Efficient Streaming LMs with Attention Sinks*](https://arxiv.org/abs/2309.17453) | 2023 | why token 0 matters |
| Willard & Louf, [*Efficient Guided Generation for LLMs*](https://arxiv.org/abs/2307.09702) (Outlines) | 2023 | FSM-based constrained decoding |
| Chen et al., [*Extending Context Window via Position Interpolation*](https://arxiv.org/abs/2306.15595) | 2023 | PI |
| Peng et al., [*YaRN*](https://arxiv.org/abs/2309.00071) | 2023 | the best context-extension method |

## Diffusion and vision

| Paper | Year | Take-away |
|---|---|---|
| Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* | 2015 | the idea, five years early |
| Song & Ermon, [*Generative Modeling by Estimating Gradients*](https://arxiv.org/abs/1907.05600) (NCSN) ⭐ | 2019 | annealed Langevin |
| **Ho et al., *Denoising Diffusion Probabilistic Models*** ⭐ | 2020 | the simple loss |
| Song et al., [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502) (DDIM) ⭐ | 2020 | deterministic sampling |
| **Song et al., *Score-Based Generative Modeling through SDEs*** ⭐ | 2021 | the unification; probability-flow ODE |
| Nichol & Dhariwal, [*Improved DDPM*](https://arxiv.org/abs/2102.09672) | 2021 | cosine schedule |
| **Dhariwal & Nichol, *Diffusion Models Beat GANs*** ⭐ | 2021 | classifier guidance; the changing of the guard |
| Ho & Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) ⭐ | 2022 | the CFG formula |
| **Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion***  ⭐⭐ | 2022 | Stable Diffusion |
| Saharia et al., [*Photorealistic Text-to-Image Diffusion*](https://arxiv.org/abs/2205.11487) (Imagen) | 2022 | the text encoder matters most |
| **Karras et al., *Elucidating the Design Space of Diffusion Models*** (EDM) ⭐ | 2022 | the best systematic study; the $\sigma$ schedule |
| Lu et al., [*DPM-Solver++*](https://arxiv.org/abs/2211.01095) | 2022 | exploiting the semi-linear ODE structure |
| Peebles & Xie, [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748) (DiT) ⭐ | 2023 | Transformers scale better than U-Nets |
| Zhang et al., [*Adding Conditional Control*](https://arxiv.org/abs/2302.05543) (ControlNet) ⭐ | 2023 | zero-init, frozen base |
| Song et al., [*Consistency Models*](https://arxiv.org/abs/2303.01469) | 2023 | 1-step generation |
| **Lipman et al., *Flow Matching for Generative Modeling*** ⭐ | 2023 | simulation-free CNFs |
| Liu et al., [*Flow Straight and Fast*](https://arxiv.org/abs/2209.03003) (Rectified Flow) ⭐ | 2022 | straight paths; reflow |
| Esser et al., [*Scaling Rectified Flow Transformers*](https://arxiv.org/abs/2403.03206) (SD3) ⭐ | 2024 | the practical details: logit-normal $t$, resolution shift, MMDiT |
| Dosovitskiy et al., [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) (ViT) ⭐ | 2020 | patches as tokens |
| **Radford et al., *Learning Transferable Visual Models*** (CLIP) ⭐ | 2021 | the shared embedding space |
| Zhai et al., [*Sigmoid Loss for Language Image Pre-Training*](https://arxiv.org/abs/2303.15343) (SigLIP) | 2023 | no global batch normalization |
| Liu et al., [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485) (LLaVA) ⭐ | 2023 | the cheap projection-based VLM recipe |
| Alayrac et al., [*Flamingo*](https://arxiv.org/abs/2204.14198) | 2022 | gated cross-attention VLMs |
| Défossez et al., [*EnCodec*](https://arxiv.org/abs/2210.13438) | 2022 | residual VQ audio tokens |

## Retrieval and agents

| Paper | Year | Take-away |
|---|---|---|
| Lewis et al., [*Retrieval-Augmented Generation*](https://arxiv.org/abs/2005.11401) ⭐ | 2020 | the original RAG |
| Karpukhin et al., [*Dense Passage Retrieval*](https://arxiv.org/abs/2004.04906) | 2020 | dense retrieval with hard negatives |
| Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* | 2009 | the sparse baseline you should still use |
| Malkov & Yashunin, [*HNSW*](https://arxiv.org/abs/1603.09320) | 2016 | the quality-default ANN index |
| Jégou et al., [*Product Quantization*](https://arxiv.org/abs/2411.12306) | 2011 | 32× index compression |
| Gao et al., [*Precise Zero-Shot Dense Retrieval*](https://arxiv.org/abs/2212.10496) (HyDE) | 2022 | embed a hypothetical answer |
| Kusupati et al., [*Matryoshka Representation Learning*](https://arxiv.org/abs/2205.13147) | 2022 | truncatable embeddings |
| **Liu et al., *Lost in the Middle*** ⭐ | 2023 | the U-shaped attention curve |
| Barnett et al., [*Seven Failure Points When Engineering a RAG System*](https://arxiv.org/abs/2401.05856) | 2024 | the failure taxonomy |
| Edge et al., [*From Local to Global*](https://arxiv.org/abs/2404.16130) (GraphRAG) | 2024 | corpus-wide questions |
| **Yao et al., *ReAct*** ⭐ | 2022 | reasoning + acting |
| Schick et al., [*Toolformer*](https://arxiv.org/abs/2302.04761) | 2023 | self-supervised tool learning |
| Shinn et al., [*Reflexion*](https://arxiv.org/abs/2303.11366) | 2023 | verbal self-critique |
| Jimenez et al., [*SWE-bench*](https://arxiv.org/abs/2310.06770) ⭐ | 2023 | realistic agentic evaluation |

## Evaluation, safety and security

| Paper | Year | Take-away |
|---|---|---|
| Hendrycks et al., [*MMLU*](https://arxiv.org/abs/2009.03300) | 2020 | the broad-knowledge standard |
| Chen et al., [*Evaluating LLMs Trained on Code*](https://arxiv.org/abs/2107.03374) (HumanEval) | 2021 | the pass@k estimator |
| Rein et al., [*GPQA*](https://arxiv.org/abs/2311.12022) | 2023 | genuinely hard, Google-proof |
| Zheng et al., [*Judging LLM-as-a-Judge*](https://arxiv.org/abs/2306.05685) ⭐ | 2023 | judge biases, measured |
| Chiang et al., [*Chatbot Arena*](https://arxiv.org/abs/2403.04132) | 2024 | preference Elo at scale |
| Liang et al., [*Holistic Evaluation of Language Models*](https://arxiv.org/abs/2211.09110) (HELM) | 2022 | the most thorough framework |
| Zhang et al., [*A Careful Examination of LLM Performance on Grade School Arithmetic*](https://arxiv.org/abs/2405.00332) (GSM1k) ⭐ | 2024 | contamination, quantified |
| Hsieh et al., [*RULER*](https://arxiv.org/abs/2404.06654) ⭐ | 2024 | effective vs advertised context |
| Schaeffer et al., [*Are Emergent Abilities of LLMs a Mirage?*](https://arxiv.org/abs/2304.15004) ⭐ | 2023 | metric discontinuity |
| **Wei et al., *Jailbroken: How Does LLM Safety Training Fail?*** ⭐ | 2023 | competing objectives; mismatched generalization |
| Zou et al., [*Universal and Transferable Adversarial Attacks*](https://arxiv.org/abs/2307.15043) (GCG) | 2023 | optimized adversarial suffixes |
| Anil et al., *Many-shot Jailbreaking* | 2024 | long context as an attack surface |
| Sharma et al., [*Towards Understanding Sycophancy*](https://arxiv.org/abs/2310.13548) ⭐ | 2023 | raters reward agreement |
| Kalai & Vempala, [*Calibrated Language Models Must Hallucinate*](https://arxiv.org/abs/2311.14648) | 2024 | a lower bound on hallucination |
| Carlini et al., [*Quantifying Memorization*](https://arxiv.org/abs/2202.07646) | 2022 | scales with duplicates and size |
| **Greshake et al., *Indirect Prompt Injection*** ⭐ | 2023 | the attack that has no complete fix |
| Nasr et al., [*Scalable Extraction of Training Data*](https://arxiv.org/abs/2311.17035) | 2023 | the divergence attack |
| Hubinger et al., [*Sleeper Agents*](https://arxiv.org/abs/2401.05566) ⭐ | 2024 | backdoors survive safety training |
| Templeton et al., [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/) ⭐ | 2024 | SAE features in a production model |
| Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) ⭐ | 2022 | why neurons are polysemantic |
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

> [!TIP]
> **How to read an ML paper efficiently**: abstract → figures → the main equation → the ablation
> table → related work. Read the method section in full only if you intend to implement it. Most
> papers have one idea; find it and move on.

---

**Next** → [Study roadmap](05-roadmap.md)
