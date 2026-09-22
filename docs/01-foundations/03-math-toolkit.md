# The Math Toolkit

> **Summary** — The subset of linear algebra and calculus that actually appears in generative
> modelling code: matrix calculus for backprop, Jacobians and the change-of-variables formula for
> flows, SVD and low-rank structure for LoRA, and the geometry of high-dimensional spaces that
> makes latent variables behave counterintuitively.

**Prerequisites**: → [Probability & information theory](02-probability-and-information-theory.md) · **Next**: → [Neural networks refresher](04-neural-network-refresher.md)

---

## 1. Shapes: the thing that actually causes bugs

Before any math, internalize the shape conventions. Almost every deep learning bug is a shape bug.

| Object | Shape | Notes |
|---|---|---|
| Batch of token IDs | $(B, T)$ | batch, sequence length |
| Hidden states | $(B, T, d)$ | $d$ = model dimension |
| Attention $Q,K,V$ | $(B, H, T, d_h)$ | $H$ heads, $d_h = d/H$ |
| Attention scores | $(B, H, T, T)$ | ⚠️ quadratic in $T$ — this is *the* memory problem |
| Weight matrix | $(d_{\text{in}}, d_{\text{out}})$ | PyTorch `nn.Linear` stores $(d_{\text{out}}, d_{\text{in}})$ |
| Images | $(B, C, H, W)$ | PyTorch; TensorFlow uses $(B,H,W,C)$ |
| Latent (diffusion) | $(B, 4, H/8, W/8)$ | typical SD-style VAE latent |

🔢 **Concretely**: a batch of 8 sequences of length 2048 in a $d=4096$ model:
hidden states are $8 \times 2048 \times 4096 = 6.7\times10^7$ floats = 134 MB in BF16.
The attention score matrix for 32 heads is $8\times32\times2048\times2048 = 1.07\times10^9$ entries
= **2.1 GB for a single layer**. This number is why FlashAttention exists.
→ [Attention](../03-sequence-models/03-attention.md)

---

## 2. Matrix calculus: the six rules you need

Use **denominator layout**: $\partial \mathcal{L}/\partial W$ has the same shape as $W$. This is what
frameworks do, and it makes shape-checking your derivations trivial.

| # | Expression | Derivative | Shape check |
|---|---|---|---|
| 1 | $y = Wx$ | $\dfrac{\partial \mathcal{L}}{\partial x} = W^\top \dfrac{\partial \mathcal{L}}{\partial y}$ | $(n,m)^\top(m,) = (n,)$ ✓ |
| 2 | $y = Wx$ | $\dfrac{\partial \mathcal{L}}{\partial W} = \dfrac{\partial \mathcal{L}}{\partial y}x^\top$ | $(m,)(n,)^\top = (m,n)$ ✓ |
| 3 | $y = f(x)$ elementwise | $\dfrac{\partial \mathcal{L}}{\partial x} = f'(x) \odot \dfrac{\partial \mathcal{L}}{\partial y}$ | elementwise |
| 4 | $s = x^\top y$ | $\partial s/\partial x = y$, $\partial s/\partial y = x$ | |
| 5 | $y=\mathrm{softmax}(z)$, $\mathcal{L}=$ CE | $\partial \mathcal{L}/\partial z = y - y_{\text{true}}$ | the reason softmax+CE is standard |
| 6 | $\mathcal{L} = \|x\|_2^2$ | $\partial \mathcal{L}/\partial x = 2x$ | |

🧠 **The shape trick** — if you forget a rule, write down the shapes and there is usually only one
way to combine the tensors that type-checks. Rule 2: $\partial\mathcal{L}/\partial W$ must be
$(m,n)$; you have a $(m,)$ gradient and an $(n,)$ input; the only product that gives $(m,n)$ is the
outer product $\delta x^\top$.

📐 **Derivation of rule 5** (the softmax+cross-entropy gradient), because it's worth seeing once:

With $p_i = e^{z_i}/\sum_k e^{z_k}$ and $\mathcal{L} = -\sum_i y_i \log p_i$:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

$$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_i \frac{y_i}{p_i}\cdot p_i(\delta_{ij}-p_j)
= -\sum_i y_i(\delta_{ij}-p_j) = -y_j + p_j\underbrace{\sum_i y_i}_{=1} = p_j - y_j$$

∎ Predicted minus target. No division, no numerical hazards — which is exactly why frameworks fuse
these two operations into one kernel (`cross_entropy(logits, targets)`, never
`log(softmax(...))`).

---

## 3. Jacobians and the change-of-variables formula

If $z \sim p_Z$ and $x = f(z)$ with $f$ **invertible and differentiable**, then

$$\boxed{\;p_X(x) = p_Z(f^{-1}(x))\left|\det \frac{\partial f^{-1}}{\partial x}\right|
= p_Z(z)\left|\det\frac{\partial f}{\partial z}\right|^{-1}\;}$$

🧠 **Intuition** — probability is *conserved*. If a transformation stretches a region of space by a
factor $|\det J|$, the density in that region must drop by the same factor so that total mass stays
1. The determinant of the Jacobian is exactly the local volume-scaling factor.

```
   z-space (uniform)               x = f(z) stretches           density must drop
   ┌─┬─┬─┬─┬─┐                  ┌───┬───────┬─────────┐
   │▓│▓│▓│▓│▓│    ──f──►        │▓▓ │  ▒▒   │   ░░    │
   └─┴─┴─┴─┴─┘                  └───┴───────┴─────────┘
   equal cells,                 wide cells → low density
   equal density                narrow cells → high density
                                p_X(x)·Δx = p_Z(z)·Δz  (mass preserved)
```

🔢 **1-D worked example** — $z \sim \mathcal{U}(0,1)$, $x = f(z) = 2z$. Then $|f'| = 2$, so
$p_X(x) = 1/2$ on $[0,2]$. Total mass $= 2 \times 1/2 = 1$ ✓. Stretching by 2 halves the density.

**Why this matters**: → [Normalizing flows](../02-classical-models/04-normalizing-flows.md) use this
to compute *exact* likelihoods. The whole art is designing $f$ that is (a) expressive,
(b) invertible, and (c) has a Jacobian determinant computable in $O(d)$ instead of the generic
$O(d^3)$.

**The trick flows use**: make the Jacobian **triangular**, because the determinant of a triangular
matrix is just the product of its diagonal:

$$\det \begin{pmatrix} a & 0 & 0 \\ * & b & 0 \\ * & * & c\end{pmatrix} = abc$$

An affine coupling layer splits $z = (z_1, z_2)$ and sets $x_1 = z_1$, $x_2 = z_2 \odot
\exp(s(z_1)) + t(z_1)$. The Jacobian is triangular by construction, and
$\log|\det J| = \sum_j s_j(z_1)$ — a sum, computable in one forward pass, with $s$ and $t$
arbitrarily complex neural nets.

---

## 4. High-dimensional geometry: why your intuition is wrong

This section prevents real confusion about latent spaces.

### Fact 1: Gaussian mass lives on a thin shell

For $z \sim \mathcal{N}(0, I_d)$, the squared norm $\|z\|^2$ follows $\chi^2_d$, so

$$\mathbb{E}[\|z\|^2] = d, \qquad \|z\| \approx \sqrt{d} \pm \frac{1}{\sqrt{2}}$$

🔢 For $d = 512$: $\|z\| \approx 22.6$, with standard deviation ≈ 0.7. **Relative width shrinks as
$1/\sqrt{d}$.** Essentially *no* samples are near the origin — even though the origin is the mode
of the density.

```
   d = 2                           d = 512
   density is highest at 0,        density still highest at 0,
   and mass is spread out          but volume grows like r^(d-1)

   p(r)·r^(d-1)                    p(r)·r^(d-1)
      ╱▔▔▔╲                                      ▕▏
     ╱     ╲                                     ▕▏  ← razor-thin shell
    ╱       ╲__                    ____________▕ ▏__   at r ≈ √512 ≈ 22.6
   0    1    2   3  r              0        20  ▕▏ 25   r
```

⚠️ **Practical consequence** — linear interpolation between two latent codes $z_1, z_2$ passes
*through* the low-density interior. Midpoint norm is about $\sqrt{d/2}$ instead of $\sqrt{d}$,
i.e. the interpolant is off-distribution and decodes to mush. **Use spherical interpolation
(slerp)** instead:

$$\mathrm{slerp}(z_1,z_2;t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}z_1 + \frac{\sin(t\Omega)}{\sin\Omega}z_2,
\qquad \Omega = \arccos\frac{z_1\cdot z_2}{\|z_1\|\|z_2\|}$$

This keeps the norm roughly constant, staying on the shell where the decoder was trained.

### Fact 2: random high-dimensional vectors are nearly orthogonal

For random unit vectors in $\mathbb{R}^d$, $\mathbb{E}[\cos\theta] = 0$ and
$\mathrm{Std}[\cos\theta] = 1/\sqrt{d}$.

🔢 $d=768$: typical cosine similarity between two random vectors is $\pm 0.036$. So an observed
cosine of $0.3$ between two embeddings, which sounds low, is actually **8 standard deviations**
from random. This is why cosine-similarity thresholds must be calibrated per-model, never taken
from intuition. → [Embeddings](../06-applications/02-embeddings-and-vector-search.md)

### Fact 3: Johnson–Lindenstrauss — you can pack a lot in

You can embed $n$ points into $k = O(\log n / \epsilon^2)$ dimensions while preserving all pairwise
distances to within $(1\pm\epsilon)$.

🔢 One million points, 10% distortion: $k \approx 8\ln(10^6)/0.01 \approx 11{,}000$ — but in
practice, with real (non-adversarial) data, 768 dimensions suffice for far more points. Relatedly,
the number of *nearly* orthogonal directions in $\mathbb{R}^d$ grows **exponentially** in $d$,
which is the basis of the superposition hypothesis in interpretability: a $d$-dimensional residual
stream can represent far more than $d$ features if each is sparse.
→ [Safety](../08-safety-and-ethics/01-safety.md)

---

## 5. SVD and low-rank structure (the math behind LoRA)

Every matrix $W \in \mathbb{R}^{m\times n}$ factorizes as

$$W = U\Sigma V^\top = \sum_{i=1}^{r} \sigma_i u_i v_i^\top, \qquad \sigma_1 \ge \sigma_2 \ge \dots \ge 0$$

**Eckart–Young theorem**: the best rank-$k$ approximation in Frobenius (or spectral) norm is
obtained by keeping the top $k$ singular values:

$$\|W - W_k\|_F^2 = \sum_{i>k}\sigma_i^2$$

🧠 **Intuition + why LoRA works** — empirically, the *update* $\Delta W$ learned during fine-tuning
has a rapidly decaying singular value spectrum: it is approximately low-rank. So instead of
learning a full $m\times n$ update, learn $\Delta W = BA$ with $B\in\mathbb{R}^{m\times r}$,
$A \in \mathbb{R}^{r\times n}$, $r \ll \min(m,n)$.

🔢 **Parameter saving** — $d = 4096$ attention projection, $r = 8$:

| | Parameters |
|---|---|
| Full $W$ | $4096^2 = 16{,}777{,}216$ |
| LoRA $B, A$ | $2 \times 4096 \times 8 = 65{,}536$ |
| **Ratio** | **0.39%** — a 256× reduction |

→ [Fine-tuning & PEFT](../04-large-language-models/04-finetuning-peft.md) for the full treatment.

**Other places rank shows up:**
- **Attention** is low-rank by construction: $QK^\top$ has rank $\le d_h = d/H$, typically 64 or 128
  even when $T = 4096$. The softmax is the only thing making it non-linear.
- **Embedding matrices** can be factorized ($V\times d \to V\times k \to k\times d$), as ALBERT did.
- **Quantization error** analysis uses spectral norms.

---

## 6. Norms, and which one to use when

| Norm | Definition | Used for |
|---|---|---|
| $\ell_2$ / Euclidean | $\sqrt{\sum_i x_i^2}$ | gradient clipping, weight decay, distances |
| $\ell_1$ | $\sum_i |x_i|$ | sparsity-inducing regularization, pruning |
| $\ell_\infty$ | $\max_i |x_i|$ | adversarial perturbation budgets, quantization ranges |
| Frobenius | $\sqrt{\sum_{ij}W_{ij}^2}$ | matrix "size", low-rank error |
| Spectral | $\sigma_{\max}(W)$ | Lipschitz constants, spectral norm GAN regularization |
| Cosine "distance" | $1 - \frac{x\cdot y}{\|x\|\|y\|}$ | embedding retrieval |

⚠️ **Pitfall** — cosine similarity is *not* a metric (no triangle inequality). Many ANN index
structures assume a metric; they handle cosine by **L2-normalizing vectors first**, after which
$\|x-y\|^2 = 2 - 2\cos\theta$, making Euclidean and cosine ranking equivalent. If you forget to
normalize, your recall quietly degrades.
→ [Vector search](../06-applications/02-embeddings-and-vector-search.md)

---

## 7. Convexity, and why nobody worries about it any more

A function is convex if $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$. Convex
problems have a unique global minimum. **Neural network losses are wildly non-convex** —
permuting hidden units gives combinatorially many equivalent minima.

📊 Yet training works. The empirical explanation:

1. In high dimensions, **saddle points vastly outnumber local minima**. At a random critical point
   of a random high-dimensional function, the probability that all $d$ eigenvalues of the Hessian
   are positive is roughly $2^{-d}$ — vanishing. Most critical points are saddles, and SGD's noise
   escapes saddles easily.
2. Over-parameterized networks have **connected, nearly-flat minima basins** (mode connectivity:
   you can usually find a low-loss path between two independently-found solutions).
3. Loss landscapes with skip connections are dramatically smoother than without — this is a
   significant part of why ResNets and pre-norm Transformers train so much more reliably.

🧠 **The takeaway for a practitioner**: you are not searching for *the* minimum; you are looking for
*any* point in a wide, flat basin. Wide minima generalize better than sharp ones, which is the
motivation behind SAM, large-batch warmup schedules, and weight averaging (EMA, model soups).

---

## 8. Numerical precision: the practical constraint

| Format | Bits | Exponent/Mantissa | Range | Typical use |
|---|---|---|---|---|
| FP32 | 32 | 8 / 23 | $\pm3.4\times10^{38}$ | master weights, optimizer states |
| TF32 | 19* | 8 / 10 | same as FP32 | NVIDIA tensor core matmuls |
| **BF16** | 16 | 8 / 7 | same as FP32 | **default for LLM training** |
| FP16 | 16 | 5 / 10 | $\pm65{,}504$ | older mixed precision; needs loss scaling |
| FP8 (E4M3) | 8 | 4 / 3 | $\pm448$ | frontier training, inference |
| INT8 | 8 | integer | $[-128,127]$ | inference quantization |
| NF4 | 4 | normal-float | — | QLoRA weight storage |

🧠 **Why BF16 beat FP16 for training** — BF16 keeps FP32's 8 exponent bits, so it has the same
*dynamic range* and gradients never overflow or underflow to zero. It sacrifices mantissa bits
(precision), which matters far less because gradient noise from mini-batching already swamps the
low-order bits. FP16's narrow range required fragile loss-scaling machinery; BF16 mostly just
works.

⚠️ **Pitfall** — accumulate in FP32 even when multiplying in BF16. Summing $10^6$ BF16 values
naively loses significant accuracy (adding a small number to a large one is a no-op when the
exponent gap exceeds the mantissa width). Framework matmul kernels do this correctly by default;
hand-written reductions often do not.

🔢 **Memory math for a 7B model** (the number you will compute a hundred times):

| Component | Bytes/param | Total |
|---|---|---|
| Weights (BF16) | 2 | 14 GB |
| Gradients (BF16) | 2 | 14 GB |
| Adam $m$ (FP32) | 4 | 28 GB |
| Adam $v$ (FP32) | 4 | 28 GB |
| FP32 master weights | 4 | 28 GB |
| **Total (before activations)** | **16** | **112 GB** |

An 80 GB A100/H100 cannot train a 7B model naively. This single table motivates ZeRO sharding,
gradient checkpointing, 8-bit optimizers and LoRA.
→ [Pretraining](../04-large-language-models/02-pretraining.md), → [PEFT](../04-large-language-models/04-finetuning-peft.md)

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Track shapes first; most bugs are shape bugs. Attention scores are $(B,H,T,T)$ — quadratic and huge. |
| 2 | Denominator layout: gradients have the same shape as the thing they differentiate. Use shapes to reconstruct forgotten rules. |
| 3 | Softmax + cross-entropy gradient is $p - y$. Always use the fused implementation. |
| 4 | Change of variables: $p_X(x)=p_Z(z)|\det J|^{-1}$. Triangular Jacobians make flows tractable. |
| 5 | Gaussian mass in high dimensions lives on a shell of radius $\sqrt d$ — use slerp, not lerp, for latent interpolation. |
| 6 | Random high-dim vectors are near-orthogonal ($\mathrm{std}=1/\sqrt d$), so calibrate cosine thresholds. |
| 7 | Fine-tuning updates are approximately low-rank → LoRA gives ~250× parameter reduction. |
| 8 | BF16 for training (range), INT8/INT4 for inference (memory). Always accumulate in FP32. |
| 9 | Adam training needs ~16 bytes/param. Memorize the 7B → 112 GB calculation. |

---

## Further reading

- Petersen & Pedersen, *The Matrix Cookbook* — the lookup table for matrix calculus.
- Strang, *Linear Algebra and Learning from Data* (2019).
- Vershynin, *High-Dimensional Probability* (2018) — concentration and the thin-shell phenomenon.
- Micikevicius et al., *Mixed Precision Training* (2017).

**Next** → [Neural networks refresher](04-neural-network-refresher.md)
