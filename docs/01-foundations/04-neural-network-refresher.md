# Neural Networks Refresher

> **Summary** — The components that every generative architecture is assembled from: the MLP,
> backpropagation (worked by hand on real numbers), activation functions and why GELU/SwiGLU won,
> normalization layers and the pre-norm/post-norm distinction that decides whether your deep
> Transformer trains at all, residual connections, and initialization schemes.

**Prerequisites**: → [Math toolkit](03-math-toolkit.md) · **Next**: → [Optimization](05-optimization.md)

---

## 1. The MLP: the universal building block

$$h^{(l)} = \phi\!\left(W^{(l)}h^{(l-1)} + b^{(l)}\right)$$

A stack of these is a multilayer perceptron. The **universal approximation theorem** says a single
hidden layer with enough units can approximate any continuous function on a compact set —
but it says nothing about *how many* units, or whether gradient descent can find them.

> [!TIP]
> **Why depth beats width** — a depth-$L$ network with ReLU units can carve input space into
> exponentially many linear regions in $L$, but only polynomially many in width. Depth composes
> features: edges → textures → parts → objects. Width memorizes. Empirically, depth wins until
> optimization difficulty catches up — which is exactly what residual connections fixed.

In a modern Transformer, **roughly two-thirds of all parameters sit in MLP blocks**, not
attention. For $d_{\text{ff}} = 4d$: attention contributes $4d^2$ per layer, MLP $8d^2$. With
SwiGLU's three matrices at $d_{\text{ff}} = \frac{8}{3}d$, it's $3 \cdot d \cdot \frac83 d = 8d^2$
— same total, by design. → [Transformer](../03-sequence-models/04-transformer.md)

---

## 2. Backpropagation, worked by hand

Backprop is the chain rule with **memoization**. The key insight: computing gradients naively
costs $O(\text{params} \times \text{ops})$; reusing intermediate results costs $O(\text{ops})$ —
the backward pass is only ~2× the forward pass.

### The tiny network

$$x \xrightarrow{\;w_1\;} z_1 \xrightarrow{\;\tanh\;} a_1 \xrightarrow{\;w_2\;} z_2 = \hat y,
\qquad \mathcal{L} = \tfrac12(\hat y - y)^2$$

**Forward pass** with $x = 2.0$, $w_1 = 0.5$, $w_2 = -1.5$, target $y = 1.0$ (biases zero):

| Step | Computation | Value |
|---|---|---|
| $z_1 = w_1 x$ | $0.5 \times 2.0$ | $1.0$ |
| $a_1 = \tanh(z_1)$ | $\tanh(1.0)$ | $0.7616$ |
| $z_2 = w_2 a_1$ | $-1.5 \times 0.7616$ | $-1.1424$ |
| $\mathcal{L} = \frac12(z_2-y)^2$ | $\frac12(-2.1424)^2$ | $2.2949$ |

**Backward pass** — propagate $\delta = \partial\mathcal{L}/\partial(\cdot)$ right to left:

| Step | Rule | Value |
|---|---|---|
| $\delta_{z_2} = \hat y - y$ | derivative of squared error | $-2.1424$ |
| $\partial\mathcal{L}/\partial w_2 = \delta_{z_2}\cdot a_1$ | rule 2 | $-2.1424 \times 0.7616 = -1.6316$ |
| $\delta_{a_1} = \delta_{z_2}\cdot w_2$ | rule 1 | $-2.1424 \times -1.5 = 3.2136$ |
| $\delta_{z_1} = \delta_{a_1}(1 - a_1^2)$ | $\tanh' = 1-\tanh^2$ | $3.2136 \times (1-0.5800) = 1.3497$ |
| $\partial\mathcal{L}/\partial w_1 = \delta_{z_1}\cdot x$ | rule 2 | $1.3497 \times 2.0 = 2.6994$ |

**Gradient descent step** with $\eta = 0.1$:
$w_1 \leftarrow 0.5 - 0.1(2.6994) = 0.230$, $w_2 \leftarrow -1.5 - 0.1(-1.6316) = -1.337$.

New forward pass: $z_1 = 0.460$, $a_1 = 0.4301$, $\hat y = -0.5750$, $\mathcal{L} = 1.2402$.
Loss dropped from 2.295 to 1.240. ✓

```
FORWARD  ──────────────────────────────────────────────────►
   x=2.0 ──[×w₁]──► z₁=1.0 ──[tanh]──► a₁=0.762 ──[×w₂]──► ŷ=-1.142 ──► L=2.295
     │        │         │        │          │        │         │
     │     cache     cache    cache      cache    cache     cache
     ▼        ▼         ▼        ▼          ▼        ▼         ▼
◄────────────────────────────────────────────────────── BACKWARD
  δx     ∂L/∂w₁=2.699   δ_z₁=1.350    δ_a₁=3.214   ∂L/∂w₂=-1.632   δ_ŷ=-2.142
```

> [!WARNING]
> **The memory cost of "cache"** — every forward activation must be stored for the backward pass.
> For a Transformer, activation memory scales as $O(B \cdot T \cdot d \cdot L)$ and often *exceeds*
> parameter memory. **Gradient checkpointing** trades this for compute: store only layer boundaries,
> recompute the rest during backward. Cost: ~33% more compute; benefit: $O(\sqrt L)$ instead of
> $O(L)$ activation memory. → [Pretraining](../04-large-language-models/02-pretraining.md)

---

## 3. Activation functions

| Name | Formula | Derivative at 0 | Notes |
|---|---|---|---|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | 0.25 | saturates both ends; **max gradient 0.25** → vanishing |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 1.0 | zero-centered, still saturates |
| ReLU | $\max(0,x)$ | undefined | fast, sparse, but "dying ReLU" |
| LeakyReLU | $\max(\alpha x, x)$ | $\alpha$ | fixes dying ReLU |
| **GELU** | $x\,\Phi(x)$ | 0.5 | smooth; BERT, GPT-2/3 |
| **SiLU/Swish** | $x\,\sigma(x)$ | 0.5 | smooth; used inside SwiGLU |
| **SwiGLU** | $(\mathrm{Swish}(xW_1))\odot(xW_2)$ | — | **modern LLM default** |

```
  ReLU              GELU                 SiLU/Swish
    │    ╱            │    ╱               │    ╱
    │   ╱             │   ╱                │   ╱
 ───┼──╱──        ────┼──╱──           ────┼──╱──
    │ ╱ 0             │╲╱                  │╲╱
    │╱                │  (small negative   │  (dips to -0.278)
                      │   dip near -0.17)
  kink at 0        smooth everywhere    smooth, non-monotone
```

> [!TIP]
> **Why GELU over ReLU** — GELU is $x \cdot P(Z \le x)$ for $Z \sim \mathcal{N}(0,1)$: "scale the
> input by the probability that it is larger than a random draw." It is a soft, probabilistic gate.
> Practically, the benefit is smoothness — the gradient is continuous, which suits the very small
> learning rates and long schedules that Transformer training uses. The measured quality gain is
> small but consistent.

> [!TIP]
> **Why SwiGLU (the actual modern choice)** — a **gated linear unit** splits the projection in two
> and lets one half *multiply* the other:

$$\mathrm{SwiGLU}(x) = \big(\mathrm{Swish}(xW_{\text{gate}})\big) \odot (xW_{\text{up}})$$

The elementwise product is a **multiplicative interaction**, something a ReLU MLP can only
approximate. Gating lets the network learn "attend to feature A *only when* feature B is present"
in one layer. Cost: three weight matrices instead of two, so implementations shrink
$d_{\text{ff}}$ from $4d$ to $\frac{8}{3}d$ to keep the parameter count identical.

Noam Shazeer's *GLU Variants Improve Transformer* (2020) reports consistent perplexity gains
for SwiGLU/GeGLU at matched parameters. It is used in LLaMA, PaLM, Mistral, Qwen and most modern
open models. → [LLM architecture](../04-large-language-models/01-llm-architecture.md)

---

## 4. Normalization

### The zoo

| Method | Normalizes over | Batch-dependent? | Used in |
|---|---|---|---|
| BatchNorm | batch, per channel | ✅ yes | CNNs |
| **LayerNorm** | features, per token | ❌ no | Transformers |
| **RMSNorm** | features (no mean) | ❌ no | LLaMA-style LLMs |
| GroupNorm | channel groups | ❌ no | diffusion U-Nets |
| AdaGN / AdaLN | features, conditioned | ❌ no | diffusion, DiT |

```
Input tensor (Batch B, Tokens T, Features d)

BatchNorm: normalize down the batch axis        LayerNorm: normalize across features
                                                 
   feature →                                       feature →
 b ┌──┬──┬──┬──┐                                 b ┌────────────┐
 a │▓▓│░░│▒▒│██│  each column gets its own       a │▓▓▓▓▓▓▓▓▓▓▓▓│ each ROW gets its
 t │▓▓│░░│▒▒│██│  mean/var over the batch        t │░░░░░░░░░░░░│ own mean/var over
 c │▓▓│░░│▒▒│██│  ⇒ depends on batchmates        c │▒▒▒▒▒▒▒▒▒▒▒▒│ features
 h └──┴──┴──┴──┘                                 h └────────────┘  ⇒ independent
```

**LayerNorm**:
$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2+\epsilon}} + \beta,
\qquad \mu = \frac1d\sum_i x_i,\ \ \sigma^2 = \frac1d\sum_i (x_i-\mu)^2$$

**RMSNorm** — drop the mean subtraction entirely:
$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac1d\sum_i x_i^2 + \epsilon}}$$

> [!TIP]
> **Why RMSNorm won** — Zhang & Sennrich (2019) showed the re-centering does essentially nothing;
> the *re-scaling* is what stabilizes training. Removing the mean saves one pass over the feature
> dimension and one set of bias parameters — around 7–10% wall-clock in practice for zero quality
> loss. LLaMA popularized it and it is now standard.

**Worked example** — $x = [2, -1, 3, 0]$, $\gamma = 1$, $\epsilon = 0$:

*LayerNorm*: $\mu = 1.0$, centered $= [1,-2,2,-1]$, $\sigma^2 = \frac{1+4+4+1}{4}=2.5$,
$\sigma = 1.581$ → output $[0.632, -1.265, 1.265, -0.632]$. (Mean 0, variance 1. ✓)

*RMSNorm*: $\mathrm{RMS} = \sqrt{\frac{4+1+9+0}{4}} = \sqrt{3.5} = 1.871$
→ output $[1.069, -0.535, 1.604, 0]$. (Mean is **not** 0 — that's the point.)

### Pre-norm vs post-norm: the choice that decides if your model trains

```
  POST-NORM (original 2017 Transformer)     PRE-NORM (every modern model)

      x ──────────────┐                        x ──────────────┐
      │               │                        │               │
   ┌──▼───┐           │                     ┌──▼───┐           │
   │ Attn │           │                     │ Norm │           │
   └──┬───┘           │                     └──┬───┘           │
      │               │                     ┌──▼───┐           │
      ▼               │                     │ Attn │           │
     (+)◄─────────────┘                     └──┬───┘           │
      │                                        ▼               │
   ┌──▼───┐                                   (+)◄─────────────┘
   │ Norm │                                    │
   └──┬───┘                                    ▼
      ▼                                    clean residual highway
  norm sits ON the residual path           from input to output
```

**Why it matters mathematically** — in pre-norm, the residual stream is
$x_{L} = x_0 + \sum_{l=1}^{L} F_l(\mathrm{Norm}(x_{l-1}))$. The gradient
$\partial x_L / \partial x_0$ contains an explicit identity term, so gradients reach layer 0
undiminished regardless of depth. In post-norm, every residual addition is followed by a
normalization that *rescales* the whole stream, so the gradient gets multiplied by a
(typically <1) factor $L$ times → exponential decay with depth.

**Practical consequence**: post-norm Transformers deeper than ~12 layers require a careful
learning-rate warmup or they diverge. Pre-norm models train to 100+ layers with minimal warmup.
Every LLM since GPT-2 uses pre-norm. The tradeoff: pre-norm models have slightly worse final
quality at equal depth, which some architectures fix with **sandwich norm** (norm before *and*
after the sublayer) or QK-norm.

> [!WARNING]
> **Known failure mode** — the residual stream in a pre-norm model grows in magnitude with depth
> (each layer adds to it, nothing rescales it). In very deep or long-trained models this can cause
> activation outliers that break INT8 quantization.
> → [Efficiency](../04-large-language-models/07-efficiency.md)

---

## 5. Residual connections

$$y = x + F(x)$$

> [!TIP]
> **Three ways to see why this works:**

1. **Optimization**: learning $F(x) = 0$ (identity) is easy; learning $H(x) = x$ from scratch
   through nonlinearities is hard. Residuals make the identity the *default*.
2. **Gradient flow**: $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$ — the
   identity guarantees gradient magnitude never collapses, no matter how small $\partial F$ is.
3. **Ensemble view** (Veit et al., 2016): an $L$-layer residual network behaves like an implicit
   ensemble of $2^L$ paths of varying depth. Most of the gradient flows through the *short* paths,
   which is why removing a single residual layer barely hurts.

Before ResNet (2015), 20-layer plain networks performed *worse* than 8-layer ones — pure
optimization failure, not overfitting. After residuals: 152 layers, then 1000+.

**In Transformers the residual stream is a first-class object.** The interpretability view:
each layer *reads* from the stream, computes something, and *writes back* an additive update. The
stream is a shared communication bus, not a pipeline.
→ [Transformer](../03-sequence-models/04-transformer.md)

---

## 6. Initialization

The goal: keep activation variance ≈ constant across layers at step 0. Too large → explosion; too
small → the signal dies before reaching the output.

For $y = Wx$ with $W_{ij}$ i.i.d. mean-zero variance $\sigma_w^2$ and $x$ having variance
$\sigma_x^2$ per component:

$$\mathrm{Var}(y_i) = \sum_{j=1}^{n_{\text{in}}}\mathrm{Var}(W_{ij}x_j) = n_{\text{in}}\sigma_w^2\sigma_x^2$$

To preserve variance we need $\sigma_w^2 = 1/n_{\text{in}}$.

| Scheme | Variance | For |
|---|---|---|
| **Xavier/Glorot** | $\frac{2}{n_{\text{in}}+n_{\text{out}}}$ | tanh, sigmoid (symmetric activations) |
| **He/Kaiming** | $\frac{2}{n_{\text{in}}}$ | ReLU (the 2 compensates for zeroing half the inputs) |
| **LeCun** | $\frac{1}{n_{\text{in}}}$ | SELU |
| **GPT-2 style** | $\mathcal{N}(0, 0.02^2)$, residual projections scaled by $1/\sqrt{2L}$ | Transformers |

> [!TIP]
> **The GPT-2 residual trick** — with $L$ layers each adding to the residual stream, variance
> accumulates as $L\sigma^2$. Scaling the output projection of each block by $1/\sqrt{2L}$ keeps the
> stream's variance stable at initialization regardless of depth. Small detail, real stability gain.

In practice:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# then, for every residual output projection:
for name, p in model.named_parameters():
    if name.endswith('out_proj.weight') or name.endswith('down_proj.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

---

## 7. Regularization: what survives at scale

| Technique | What it does | Status in modern LLMs |
|---|---|---|
| **Dropout** | zero units with prob $p$ at train time | ⚠️ often **off** in large pretraining |
| **Weight decay** | penalize $\|W\|^2$ | ✅ standard, ~0.1 with AdamW |
| **Label smoothing** | target $1-\epsilon$ instead of 1 | ⚠️ used in translation, rare in LLMs |
| **Data augmentation** | transform inputs | ✅ dominant in vision |
| **Early stopping** | stop before overfit | ⚠️ irrelevant at <1 epoch |
| **Gradient clipping** | cap $\|g\|$ | ✅ essential (typically 1.0) |

> [!TIP]
> **Why dropout largely disappeared from pretraining** — dropout fights *overfitting*. Frontier
> pretraining runs see each token roughly once (or a handful of times); the model is
> under-fitting, not over-fitting. Adding dropout just wastes compute. It reappears during
> **fine-tuning**, where datasets are small and epochs are many.

> [!WARNING]
> **Pitfall** — weight decay in Adam is **not** the same as L2 regularization. Adam divides the
> gradient by $\sqrt{v}$, so an L2 term added to the gradient gets scaled differently for each
> parameter. **AdamW** applies decay directly to the weights, decoupled from the adaptive scaling.
> Always use AdamW. → [Optimization](05-optimization.md)

---

## 8. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Backprop = chain rule + caching; backward ≈ 2× forward cost, but needs all activations stored. |
| 2 | Gradient checkpointing trades ~33% compute for $O(\sqrt L)$ activation memory. |
| 3 | GELU/SiLU for smoothness; SwiGLU for multiplicative gating (modern default, $d_{\text{ff}}=\frac83 d$). |
| 4 | RMSNorm ≈ LayerNorm minus the mean, ~10% faster, no quality loss. |
| 5 | **Pre-norm** keeps a clean residual highway → trains at any depth. Post-norm needs careful warmup. |
| 6 | Residuals make identity the default and guarantee $\partial y/\partial x = I + \cdots$. |
| 7 | Init to preserve variance; scale residual projections by $1/\sqrt{2L}$. |
| 8 | Dropout is for fine-tuning, not for single-epoch pretraining. Use AdamW, not Adam+L2. |

---

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*](https://arxiv.org/abs/1607.00133), ch. 6–8.
- He et al., [*Deep Residual Learning*](https://arxiv.org/abs/1512.03385) (2015); Veit et al., [*Residual Networks Behave Like Ensembles*](https://arxiv.org/abs/1605.06431) (2016).
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (2020) — the pre/post-norm analysis.
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) (2020).
- Zhang & Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) (2019).

**Next** → [Optimization & training dynamics](05-optimization.md)
