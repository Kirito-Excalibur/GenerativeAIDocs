# Notation

> Symbol conventions used throughout this wiki. Where a symbol is overloaded across subfields
> (and several are), the disambiguation is noted.

---

## Probability and distributions

| Symbol | Meaning |
|---|---|
| $p(x)$, $q(x)$ | probability mass or density |
| $p_{\text{data}}$ | the true, unknown data distribution |
| $p_\theta$ | the model, parameterized by $\theta$ |
| $q_\phi(z\mid x)$ | approximate posterior / encoder, parameterized by $\phi$ |
| $\mathbb{E}_{x\sim p}[f(x)]$ | expectation of $f$ under $p$ |
| $\mathcal{N}(\mu, \Sigma)$ | Gaussian with mean $\mu$, covariance $\Sigma$ |
| $\mathcal{U}[a,b]$ | uniform distribution |
| $x \sim p$ | $x$ is sampled from $p$ |
| $\propto$ | proportional to (an unnormalized density) |
| $H(p)$ | entropy |
| $H(p,q)$ | cross-entropy |
| $D_{\mathrm{KL}}(p\|q)$ | Kullback–Leibler divergence |
| $D_{\mathrm{JS}}(p\|q)$ | Jensen–Shannon divergence |
| $I(X;Y)$ | mutual information |
| $\sigma(\cdot)$ | ⚠️ **overloaded**: the logistic sigmoid, *or* a standard deviation, *or* a diffusion noise level. Disambiguated by context. |

## Data and model dimensions

| Symbol | Meaning |
|---|---|
| $x$ | a data point (image, token sequence) |
| $z$ | a latent variable |
| $c$ | a conditioning variable (prompt, class, caption) |
| $y$ | a label or target |
| $d$, $d_{\text{model}}$ | model/residual-stream dimension |
| $d_k$, $d_h$ | per-head dimension ($=d/H$) |
| $d_{\text{ff}}$ | MLP hidden dimension |
| $L$ | number of layers |
| $H$ | number of attention heads |
| $H_{kv}$ | number of key/value heads (GQA) |
| $V$ | vocabulary size |
| $T$, $n$ | sequence length |
| $B$ | batch size |
| $N$ | ⚠️ **overloaded**: number of parameters (scaling laws) *or* dataset/batch size (statistics) |
| $D$ | ⚠️ **overloaded**: dataset size in tokens (scaling laws) *or* the discriminator (GANs) |
| $C$ | compute, in FLOPs |

## Neural networks

| Symbol | Meaning |
|---|---|
| $\theta$, $\phi$ | model parameters |
| $W$, $b$ | weight matrix, bias vector |
| $h^{(l)}$ | hidden state at layer $l$ |
| $\mathcal{L}$ | loss |
| $\eta$ | learning rate |
| $\nabla_\theta$ | gradient with respect to $\theta$ |
| $\odot$ | elementwise (Hadamard) product |
| $\|\cdot\|_p$ | $\ell_p$ norm |
| $\mathrm{sg}[\cdot]$ | stop-gradient |
| $\phi(\cdot)$ | ⚠️ an activation function (context) or the encoder parameters (VAE context) |

## Attention

| Symbol | Meaning |
|---|---|
| $Q, K, V$ | query, key, value matrices |
| $W_Q, W_K, W_V, W_O$ | the four attention projections |
| $A$ | the attention weight matrix, $(T\times T)$ |
| $\alpha_{ij}$ | attention weight from position $i$ to $j$ |
| $m$, $n$ | position indices (RoPE) |
| $\theta_i$ | ⚠️ RoPE frequency for dimension pair $i$ — **not** model parameters |

## Diffusion

| Symbol | Meaning |
|---|---|
| $x_0$ | ⚠️ **convention clash**: clean data in *diffusion*; **noise** in *flow matching* |
| $x_t$ | the noisy sample at time $t$ |
| $x_T$ | pure noise (diffusion) |
| $t$ | timestep — $\{1..T\}$ discrete, or $[0,1]$ continuous |
| $\beta_t$ | noise-schedule variance at step $t$ |
| $\alpha_t$ | $1 - \beta_t$ |
| $\bar\alpha_t$ | $\prod_{s\le t}\alpha_s$ — cumulative signal retention |
| $\epsilon$ | the Gaussian noise added |
| $\epsilon_\theta$ | the noise-prediction network |
| $s_\theta$ | the score network, $\nabla_x\log p$ |
| $v_\theta$ | the velocity network (v-prediction or flow matching) |
| $w$ | the classifier-free guidance scale |
| $\sigma(t)$ | noise level at time $t$ (VE / EDM parameterization) |

> [!WARNING]
> **The $x_0$ clash is the one that trips people up.** In diffusion, $x_0$ is the clean image and
> $x_T$ is noise; time runs *forward* toward noise. In flow matching, $x_0$ is noise and $x_1$ is
> data; time runs *forward* toward data. This wiki follows each field's own convention on its own
> page and flags the switch. Always check which direction a paper's time axis runs.

## Reinforcement learning and alignment

| Symbol | Meaning |
|---|---|
| $\pi_\theta$ | the policy (the model being trained) |
| $\pi_{\text{ref}}$ | the reference policy (frozen SFT model) |
| $r(x,y)$, $r_\phi$ | reward function / reward model |
| $A_t$ | advantage |
| $V_\psi$ | value function |
| $\beta$ | ⚠️ the KL penalty coefficient in RLHF/DPO — **not** a diffusion noise schedule |
| $y_w$, $y_l$ | the preferred ("won") and rejected ("lost") responses |
| $\rho_t$ | the PPO probability ratio $\pi_\theta/\pi_{\theta_{\text{old}}}$ |

## Common conventions

| Convention | Meaning |
|---|---|
| Bold lowercase $\mathbf{x}$ | a vector (this wiki mostly drops the bold; shapes are stated instead) |
| Capital $X$ | a matrix or a random variable |
| Subscript $x_i$ | the $i$-th element or the $i$-th position |
| Superscript $x^{(i)}$ | the $i$-th sample in a dataset |
| $\hat{y}$ | a predicted value |
| $\tilde{x}$ | a perturbed or noisy value |
| $x^*$ | an optimal value |
| $\bar{x}$ | a mean or a cumulative product (context-dependent) |
| $\log$ | natural logarithm, unless a base is given |

## Units

| Unit | Meaning |
|---|---|
| nats | information in base $e$ ($\log$ = $\ln$) |
| bits | information in base 2; 1 nat = 1.4427 bits |
| FLOPs | floating-point operations (the *count*) |
| FLOP/s | floating-point operations per second (the *rate*) |
| tokens | the model's discrete units, ~4 characters in English |

> [!WARNING]
> **FLOPs vs FLOP/s** are constantly confused in the literature. "$10^{25}$ FLOPs" is a training
> budget; "$10^{15}$ FLOP/s" is hardware throughput. This wiki always writes the rate with a slash.

---

**Back to** → [the index](../index.md)
