# Math Cheatsheet

> Every formula in this wiki, on one page, grouped by topic, with a pointer to where it's derived.
> Intended for revision and for looking things up mid-implementation.

---

## Information theory

| Quantity | Formula |
|---|---|
| Entropy | $H(p) = -\sum_x p(x)\log p(x)$ |
| Cross-entropy | $H(p,q) = -\sum_x p(x)\log q(x)$ |
| **KL divergence** | $D_{\mathrm{KL}}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p) \ge 0$ |
| Jensen–Shannon | $D_{\mathrm{JS}}(p\|q) = \frac12 D_{\mathrm{KL}}(p\|m) + \frac12 D_{\mathrm{KL}}(q\|m),\ m=\frac{p+q}{2}$ |
| Mutual information | $I(X;Y) = H(X) - H(X\mid Y) = D_{\mathrm{KL}}(p(x,y)\|p(x)p(y))$ |
| **Perplexity** | $\mathrm{PPL} = \exp\!\big(H(p_{\text{data}}, p_\theta)\big)$ |
| Bits per character | $\mathrm{BPC} = \dfrac{\text{total nats}}{\ln 2 \times \#\text{chars}}$ |
| MLE ≡ min KL | $\arg\max_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta] = \arg\min_\theta D_{\mathrm{KL}}(p_{\text{data}}\|p_\theta)$ |

→ [Probability & information theory](../01-foundations/02-probability-and-information-theory.md)

---

## Gaussians

| Quantity | Formula |
|---|---|
| Density | $\mathcal{N}(x;\mu,\Sigma) = (2\pi)^{-d/2}\|\Sigma\|^{-1/2}\exp\!\big(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$ |
| Entropy | $H = \frac12\log(2\pi e\sigma^2)$ nats |
| **KL to $\mathcal{N}(0,I)$** | $\frac12\sum_j\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$ |
| General KL (diagonal) | $\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$ |
| Sum of independents | $\mathcal{N}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$ |
| **Reparameterization** | $z = \mu + \sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$ |
| Thin shell (dim $d$) | $\|z\|\approx\sqrt d$, std $\approx 1/\sqrt2$ |
| Random cosine similarity | $\mathbb{E}=0$, std $=1/\sqrt d$ |

---

## Core model equations

| Model | Objective / definition |
|---|---|
| **Autoregressive** | $p(x) = \prod_i p(x_i\mid x_{<i})$; loss $=-\sum_i\log p_\theta(x_i\mid x_{<i})$ |
| **VAE (ELBO)** | $\log p(x) \ge \mathbb{E}_{q}[\log p_\theta(x\mid z)] - D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$ |
| ELBO identity | $\log p_\theta(x) = \mathcal{L}_{\text{ELBO}} + D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))$ |
| **GAN** | $\min_G\max_D\ \mathbb{E}_{p_{\text{data}}}[\log D(x)] + \mathbb{E}_{p_z}[\log(1-D(G(z)))]$ |
| Optimal discriminator | $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$ |
| GAN ≡ JSD | $V(D^*,G) = 2D_{\mathrm{JS}}(p_{\text{data}}\|p_g) - 2\log 2$ |
| **WGAN** | $\min_G\max_{\|D\|_L\le1}\ \mathbb{E}_{p_{\text{data}}}[D(x)] - \mathbb{E}_{p_z}[D(G(z))]$ |
| **Normalizing flow** | $\log p_X(x) = \log p_Z(f^{-1}(x)) + \log\big\|\det\frac{\partial f^{-1}}{\partial x}\big\|$ |
| Affine coupling | $y_B = h_B\odot e^{s(h_A)} + t(h_A)$; $\log\|\det J\| = \sum_j s_j(h_A)$ |
| **EBM** | $p(x) = e^{-E(x)}/Z$; $\nabla_\theta\mathcal{L} = \mathbb{E}_{p_{\text{data}}}[\nabla_\theta E] - \mathbb{E}_{p_\theta}[\nabla_\theta E]$ |
| Score from energy | $\nabla_x\log p(x) = -\nabla_x E(x)$ |
| Langevin | $x_{t+1} = x_t + \frac{\eta}{2}\nabla_x\log p(x_t) + \sqrt{\eta}\,z_t$ |

---

## Attention and Transformer

| Quantity | Formula |
|---|---|
| **Attention** | $\operatorname{softmax}\!\big(\frac{QK^\top}{\sqrt{d_k}}\big)V$ |
| Why $\sqrt{d_k}$ | $\operatorname{Var}(q\cdot k) = d_k$ for unit-variance components |
| Multi-head | $\text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O$, $d_h = d/H$ |
| **Parameters** | $N \approx 12Ld^2 + Vd$ |
| Per layer | attention $4d^2$ + MLP $8d^2$ = $12d^2$ |
| **Training FLOPs** | $C \approx 6ND$ |
| Inference FLOPs | $\approx 2N$ per token |
| Per-layer FLOPs/token | $24d^2 + 4Td$ |
| **KV cache bytes** | $2 \cdot L \cdot H_{kv}\cdot d_h\cdot T\cdot B\cdot \text{bytes}$ |
| MFU | $\dfrac{6ND}{t\cdot F_{\text{peak}}\cdot n_{\text{GPU}}}$ |
| LayerNorm | $\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ |
| RMSNorm | $\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}$ |
| SwiGLU | $\big(\text{Swish}(xW_{\text{gate}})\big)\odot(xW_{\text{up}})$, then $W_{\text{down}}$ |
| GELU | $x\,\Phi(x)$ |
| **RoPE** | $\theta_i = 10000^{-2i/d}$; rotate pair $i$ by $m\theta_i$; $\langle R_mq, R_nk\rangle = q^\top R_{n-m}k$ |
| ALiBi | $\text{score}_{ij} - m_h\|i-j\|$, $m_h = 2^{-8h/H}$ |
| Softmax gradient | $\partial\mathcal{L}/\partial z = p - y$ |

→ [Transformer](../03-sequence-models/04-transformer.md)

---

## Diffusion and flow

| Quantity | Formula |
|---|---|
| Forward step | $q(x_t\mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\ \beta_t I)$ |
| $\alpha,\bar\alpha$ | $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s\le t}\alpha_s$ |
| **Closed form** | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| Posterior mean | $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$ |
| Posterior variance | $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ |
| **Simple loss** | $\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2$ |
| Cosine schedule | $\bar\alpha_t = \frac{f(t)}{f(0)},\ f(t)=\cos^2\!\big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\big)$ |
| $v$-prediction | $v = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ |
| **Score ↔ noise** | $s_\theta(x_t,t) = -\dfrac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$ |
| Reverse SDE | $dx = [f - g^2\nabla_x\log p_t(x)]dt + g\,d\bar w$ |
| **Probability-flow ODE** | $\frac{dx}{dt} = f - \frac12 g^2\nabla_x\log p_t(x)$ |
| Denoising score matching | $\mathbb{E}\big\|s_\theta(\tilde x) - \frac{x-\tilde x}{\sigma^2}\big\|^2$ |
| **CFG** | $\tilde\epsilon = \epsilon(\varnothing) + w\big(\epsilon(c) - \epsilon(\varnothing)\big)$ |
| Classifier guidance | $\nabla\log p(x\mid c) = \nabla\log p(x) + \nabla\log p(c\mid x)$ |
| **Flow matching** | $x_t = (1-t)x_0 + tx_1$; target $u = x_1 - x_0$; $\mathcal{L} = \|v_\theta(x_t,t) - (x_1-x_0)\|^2$ |
| Karras $\sigma$ schedule | $\sigma_i = \big(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})\big)^\rho,\ \rho=7$ |

→ [Diffusion](../05-diffusion-and-vision/01-diffusion-models.md), → [Flow matching](../05-diffusion-and-vision/04-flow-matching.md)

---

## Training and scaling

| Quantity | Formula |
|---|---|
| SGD + momentum | $v_t = \beta v_{t-1} + g_t$; $\theta \mathrel{-}= \eta v_t$ |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ |
| Bias correction | $\hat m_t = \frac{m_t}{1-\beta_1^t}$, $\hat v_t = \frac{v_t}{1-\beta_2^t}$ |
| Adam update | $\theta \mathrel{-}= \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ |
| **AdamW** | $\theta \mathrel{-}= \eta\big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta\big)$ |
| Cosine schedule | $\eta_t = \eta_{\min} + \frac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi\tau))$ |
| Gradient clipping | $g \mathrel{*}= \min(1, c/\|g\|_2)$ |
| Init (He) | $\sigma_w^2 = 2/n_{\text{in}}$; residual proj $\times 1/\sqrt{2L}$ |
| Batch scaling | steps$(B)$ / steps$(\infty) \approx 1 + B_{\text{crit}}/B$ |
| **Kaplan** | $L(N) = (N_c/N)^{0.076}$, $L(D) = (D_c/D)^{0.095}$ |
| **Chinchilla** | $L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$; $E{=}1.69,A{=}406.4,B{=}410.7,\alpha{=}0.34,\beta{=}0.28$ |
| Compute-optimal | $N \propto C^{0.45}$, $D\propto C^{0.55}$ → $D\approx 20N$ |
| From budget | $N = \sqrt{C/120}$, $D = 20N$ |
| Adam memory | 16 bytes/param (BF16 weights+grads, FP32 $m,v$, master) |
| ZeRO-3 memory | $16N/D$ bytes per GPU |
| Pipeline bubble | $\frac{P-1}{m+P-1}$ |

→ [Scaling laws](../04-large-language-models/03-scaling-laws.md)

---

## Alignment

| Quantity | Formula |
|---|---|
| Bradley–Terry | $P(y_w\succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$ |
| Reward model loss | $-\mathbb{E}\big[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big]$ |
| RLHF objective | $\max_\theta \mathbb{E}[r_\phi(x,y)] - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})$ |
| Optimal policy | $\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\big(\frac1\beta r(x,y)\big)$ |
| Implied reward | $r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$ |
| **DPO loss** | $-\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big)\Big]$ |
| PPO clip | $\mathbb{E}\big[\min(\rho A, \operatorname{clip}(\rho,1{-}\epsilon,1{+}\epsilon)A)\big]$ |
| GRPO advantage | $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ over a group of $G$ samples |
| MoE aux loss | $\mathcal{L}_{\text{aux}} = \lambda E\sum_i f_i P_i$ |
| LoRA | $W' = W_0 + \frac{\alpha}{r}BA$ |

→ [Alignment](../04-large-language-models/05-alignment.md)

---

## Retrieval and evaluation

| Quantity | Formula |
|---|---|
| Cosine similarity | $\frac{a\cdot b}{\|a\|\|b\|}$ |
| Normalized L2 ↔ cosine | $\|a-b\|^2 = 2 - 2\cos\theta$ for unit vectors |
| **BM25** | $\sum_{t\in q}\text{IDF}(t)\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\frac{\|d\|}{\overline{\|d\|}})}$ |
| **RRF** | $\sum_r \frac{1}{k + \text{rank}_r(d)},\ k=60$ |
| InfoNCE | $-\log\frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_j\exp(\text{sim}(q,d_j)/\tau)}$; MI bound $\le\log N$ |
| DCG@k | $\sum_{i=1}^{k}\frac{rel_i}{\log_2(i+1)}$; nDCG = DCG/IDCG |
| **pass@k** | $1 - \dfrac{\binom{n-c}{k}}{\binom{n}{k}}$ |
| **FID** | $\|\mu_r-\mu_g\|^2 + \operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ |
| Inception Score | $\exp\big(\mathbb{E}_x[D_{\mathrm{KL}}(p(y\mid x)\|p(y))]\big)$ |
| BLEU | $\text{BP}\cdot\exp(\sum_n w_n\log p_n)$ |
| ECE | $\sum_m \frac{\|B_m\|}{N}\big\|\text{acc}(B_m)-\text{conf}(B_m)\big\|$ |
| Elo | $P(A\succ B) = \big(1+10^{(R_B-R_A)/400}\big)^{-1}$ |
| Agent success | $p^k$ for $k$ steps at per-step reliability $p$ |
| Speculative decoding | $\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{k+1}}{1-\alpha}$ |

→ [Metrics](../07-evaluation/01-metrics.md)

---

## Quick numeric reference

| Quantity | Value |
|---|---|
| 1 nat | 1.4427 bits |
| English text | ~4 chars/token |
| English entropy (human) | ~0.6–1.3 bits/char |
| Chinchilla ratio | 20 tokens/param |
| Modern over-training | 500–2000 tokens/param |
| Adam memory | 16 bytes/param |
| Training FLOPs | $6N$ per token |
| Inference FLOPs | $2N$ per token |
| Good MFU | 40–50% |
| Typical LR (7B) | $3\times10^{-4}$ |
| Grad clip | 1.0 |
| Weight decay | 0.1 (matrices only) |
| Adam $\beta$ (LLM) | $(0.9, 0.95)$ |
| Warmup | 0.5–2% of steps |
| Diffusion steps (DDPM) | 1000 |
| Diffusion steps (DPM-Solver++) | 15–25 |
| CFG scale | 7–8 |
| LoRA rank | 16–32 |
| LoRA LR | $2\times10^{-4}$ |
| Top-$p$ | 0.9–0.95 |
| Chunk size (RAG) | 512 tokens, 64 overlap |
| RRF constant | 60 |

---

**Next** → [Timeline](03-timeline.md)
