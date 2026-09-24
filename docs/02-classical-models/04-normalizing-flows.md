# Normalizing Flows

> **Summary** — Build an invertible neural network $f$ mapping a simple base distribution to the
> data. The change-of-variables formula then gives the *exact* likelihood. The entire art is
> designing layers that are invertible **and** have a Jacobian determinant computable in $O(d)$
> rather than $O(d^3)$. Flows are the only family offering exact likelihood *and* fast sampling
> *and* exact inference — at the price of severe architectural constraints. Their continuous-time
> descendants (CNFs, flow matching) are now state of the art for images.

**Prerequisites**: → [Math toolkit §3](../01-foundations/03-math-toolkit.md#3-jacobians-and-the-change-of-variables-formula) · **Next**: → [Energy-based models](05-energy-based-models.md)

---

## 1. The core idea

$$z \sim p_Z = \mathcal{N}(0,I), \qquad x = f_\theta(z), \qquad z = f_\theta^{-1}(x)$$

By change of variables:

$$\boxed{\;\log p_X(x) = \log p_Z\!\big(f^{-1}(x)\big) + \log\left|\det \frac{\partial f^{-1}}{\partial x}\right|\;}$$

```
    BASE                      FLOW                        DATA
  p_Z = N(0,I)          f = f_K ∘ … ∘ f_1             p_X (complex)

     ╭───╮       f_1      ╭─────╮      f_2      ╭──╮  f_3   ╭╮  ╭╮
    │  ●  │ ──────────►  │  ●●  │ ──────────►  │●●  │ ────► ││  ││
     ╰───╯               ╰─────╯               ╰──╯         ╰╯  ╰╯
   round blob         stretched            bent            two modes

   ◄────────────────────────────────────────────────────────────────
     f^{-1}  (exact — every layer is invertible)

   Each step:  log p changes by  −log|det J_k|
   Total:      log p_X(x) = log p_Z(z) − Σ_k log|det J_k|
```

> [!TIP]
> **Intuition** — you are not learning a density directly; you are learning a *deformation of
> space*. Start with a perfectly round Gaussian blob of probability mass, then squeeze, stretch and
> fold it until it takes the shape of the data. The log-determinant term is the bookkeeping that
> keeps total mass at 1: wherever you stretch space, the density must drop proportionally.

**Composition is easy**, because determinants multiply:

$$\log\left|\det\frac{\partial f}{\partial z}\right| = \sum_{k=1}^{K}\log\left|\det\frac{\partial f_k}{\partial h_{k-1}}\right|$$

So the entire problem reduces to: **design one layer that is invertible with a cheap log-det**, then
stack many of them.

---

## 2. The determinant problem

For a general $d\times d$ Jacobian, computing $\det J$ costs $O(d^3)$. For $d = 3072$ (a
$32\times32$ RGB image) that is $2.9\times10^{10}$ operations **per sample per layer**. Completely
infeasible.

**The universal trick: make $J$ triangular.**

$$\det\begin{pmatrix} J_{11} & 0 & \cdots & 0\\ * & J_{22} & & \vdots \\ \vdots & & \ddots & 0 \\ * & \cdots & * & J_{dd}\end{pmatrix} = \prod_{i=1}^{d} J_{ii}$$

$O(d)$ instead of $O(d^3)$. Every flow architecture is a different way to force triangularity while
retaining expressive power.

---

## 3. Affine coupling layers (RealNVP)

The most important construction. Split the input in half: $h = (h_A, h_B)$.

$$
\begin{aligned}
y_A &= h_A &&\text{(pass through unchanged)}\\
y_B &= h_B \odot \exp\!\big(s_\theta(h_A)\big) + t_\theta(h_A) &&\text{(affine transform, conditioned on } h_A)
\end{aligned}
$$

```
     h_A ──────────────────────────────────────► y_A     (identity)
      │
      ├──► [ neural net s(·), t(·) ]  ← arbitrarily deep, NOT inverted
      │              │
      │         scale, shift
      │              ▼
     h_B ──────────► ⊗,⊕ ────────────────────► y_B
```

**Inverse** — trivial, and note that $s$ and $t$ are *never* inverted:

$$h_A = y_A, \qquad h_B = (y_B - t_\theta(y_A))\odot\exp(-s_\theta(y_A))$$

**Jacobian** — block triangular by construction:

$$J = \begin{pmatrix} I & 0 \\ \frac{\partial y_B}{\partial h_A} & \mathrm{diag}(\exp(s))\end{pmatrix}
\quad\Longrightarrow\quad \log|\det J| = \sum_j s_\theta(h_A)_j$$

> [!TIP]
> **Why this is clever** — $s_\theta$ and $t_\theta$ can be *any* network: a ResNet, a Transformer,
> anything. They are only ever evaluated in the forward direction. The invertibility constraint
> applies to the *coupling structure*, not to the networks inside it. You get arbitrary expressivity
> in the conditioning, with a $O(d)$ log-det.

> [!WARNING]
> **You must permute between layers.** Otherwise $h_A$ is never transformed — half the variables
> stay frozen forever. Alternating masks (or, better, learned $1\times1$ convolutions — see §4)
> ensure every dimension is both a conditioner and a target.

**Worked 2-D example.** Let $h = (h_1, h_2) = (1.0, 2.0)$, with
$s(h_1) = 0.5h_1 = 0.5$ and $t(h_1) = -h_1 = -1.0$:

$$y_1 = 1.0, \qquad y_2 = 2.0 \cdot e^{0.5} + (-1.0) = 2.0(1.6487) - 1.0 = 2.2974$$

$$\log|\det J| = s = 0.5$$

Inverse check: $h_2 = (2.2974 - (-1.0))\cdot e^{-0.5} = 3.2974 \times 0.6065 = 2.0$ ✓

Density: if $p_Z(1.0, 2.0) = \mathcal{N}$ evaluates to $\log p_Z = -\frac12(1+4) - \log(2\pi) = -4.338$,
then $\log p_X(y) = -4.338 - 0.5 = -4.838$. The transform stretched the space by $e^{0.5}$, so the
density dropped by exactly that factor. ✓

---

## 4. The architecture lineage

| Model | Year | Contribution |
|---|---|---|
| NICE | 2014 | additive coupling ($s = 0$); volume-preserving, $\log\det = 0$ |
| **RealNVP** | 2016 | affine coupling, multi-scale architecture, checkerboard + channel masks |
| **Glow** | 2018 | invertible $1\times1$ convolutions replace fixed permutations; ActNorm |
| MAF | 2017 | masked autoregressive flow: fast density, slow sampling |
| IAF | 2016 | inverse autoregressive flow: slow density, **fast sampling** |
| Neural Spline Flows | 2019 | monotonic rational-quadratic splines instead of affine — far more expressive |
| FFJORD | 2018 | continuous-time flow with a stochastic trace estimator |
| **Flow matching** | 2023 | continuous flow trained without simulating the ODE → [see V.4](../05-diffusion-and-vision/04-flow-matching.md) |

### Glow's invertible 1×1 convolution

A $1\times1$ conv over $c$ channels is a matrix $W \in \mathbb{R}^{c\times c}$ applied at each
spatial location. Its log-determinant over an $h\times w$ feature map is
$h \cdot w \cdot \log|\det W|$.

Computing $\det W$ for $c = 512$ still costs $O(c^3)$ — but only once per layer per step, not per
sample position. Glow further reduces it with an LU decomposition $W = PL(U + \mathrm{diag}(s))$,
making $\log|\det W| = \sum\log|s|$ an $O(c)$ operation.

> [!TIP]
> This is a **learned, soft generalization of the permutation** — instead of "swap halves", the
> model learns which linear mixture of channels to condition on. It was the main quality jump from
> RealNVP to Glow.

### MAF vs IAF: the duality worth understanding

| | Density evaluation $p(x)$ | Sampling |
|---|---|---|
| **MAF** (masked autoregressive flow) | ✅ 1 parallel pass | ❌ $d$ sequential steps |
| **IAF** (inverse autoregressive flow) | ❌ $d$ sequential steps | ✅ 1 parallel pass |

They are exact inverses of each other. **You cannot have both directions fast** with an
autoregressive flow — which is why Parallel WaveNet *distilled* a slow-sampling autoregressive
teacher into a fast-sampling IAF student. A clean early example of "train with the convenient
parameterization, deploy with the other one."

---

## 5. Training: it's just maximum likelihood

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\left[\log p_Z\!\big(f_\theta^{-1}(x^{(i)})\big) + \sum_{k}\log\left|\det J_k^{-1}\right|\right]$$

The whole training loop:

```python
def flow_nll(x, flow):
    z, log_det = flow.inverse(x)            # data -> base, accumulating log|det|
    log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * math.log(2 * math.pi)
    return -(log_pz + log_det).mean()       # negative log-likelihood, in nats

for x, _ in loader:
    loss = flow_nll(x, flow)
    opt.zero_grad(); loss.backward(); opt.step()

# sampling is the forward direction
with torch.no_grad():
    z = torch.randn(64, D)
    samples, _ = flow.forward(z)
```

No adversary, no bound, no sampling in the loss. **The number you print is a real log-likelihood in
nats** — which is exactly why flows are the standard tool when a calibrated density is required.

> [!WARNING]
> **Dequantization** — images are discrete ($\{0,\dots,255\}$) but flows model continuous
> densities. Fit a continuous density to discrete data and it will place infinitely tall spikes on
> the integers, giving unbounded (meaningless) likelihoods. Fix: add uniform noise,
> $\tilde x = (x + u)/256$ with $u\sim\mathcal{U}[0,1)^d$. This turns the objective into a valid
> lower bound on the discrete log-likelihood. **Variational dequantization** (Flow++) learns the
> noise distribution and improves bits/dim noticeably.

**Reported bits/dim on CIFAR-10** (lower is better):

| Model | bits/dim |
|---|---|
| RealNVP | 3.49 |
| Glow | 3.35 |
| Flow++ | 3.08 |
| **PixelCNN++ (autoregressive)** | **2.92** |
| Sparse Transformer (AR) | 2.80 |
| Diffusion (VDM, bound) | ~2.65 |

> [!TIP]
> **Note what this table says**: flows have *exact* likelihood but *worse* likelihood than
> autoregressive models. Invertibility costs expressivity. That tradeoff is the fundamental
> limitation of discrete flows.

---

## 6. The constraints, stated honestly

| Constraint | Consequence |
|---|---|
| $f$ must be invertible | **no dimensionality reduction** — the latent has the same size as the data |
| Every layer needs a cheap log-det | rules out most standard architectures |
| No bottleneck | no compression, no compact semantic latent |
| Needs many layers | Glow used 6 levels × 32 steps = 192 flow steps for 256×256 faces |
| Topology preservation | a continuous bijection cannot change the number of connected components |

> [!TIP]
> **The topology point is subtle and important.** A diffeomorphism maps connected sets to connected
> sets. If your data has $k$ disconnected modes and your base is a single Gaussian blob, a perfectly
> smooth flow *cannot* produce true separation — it must leave a thin bridge of probability between
> modes. In practice flows handle this by making the bridges very low-density, but it is a real
> structural limitation, and it is one reason flows produce samples "between classes."

```
   Base: one connected blob       Data: two disconnected modes
          ╭───╮                        ╭─╮     ╭─╮
         │  ●  │      ──f──►          │ ● │   │ ● │
          ╰───╯                        ╰─╯     ╰─╯
                                          ▲
                              a smooth bijection must keep
                              a (thin) bridge here — it cannot
                              tear space apart
```

---

## 7. Where flows are used

| Application | Why a flow |
|---|---|
| **Density estimation / anomaly detection** | exact calibrated $p(x)$; threshold on log-likelihood |
| **Simulation-based inference** (cosmology, particle physics) | need real posterior densities, not samples |
| **Variational inference** | a flow makes $q(z\mid x)$ far more expressive than a Gaussian, tightening the ELBO |
| **RL policies** | flexible, exactly-samplable continuous action distributions with tractable entropy |
| **Lossless compression** | a likelihood model + arithmetic coding = a codec (bits/dim is literally the rate) |
| **Audio synthesis** | Parallel WaveNet / WaveGlow: fast parallel sampling |
| **→ Flow matching** | the continuous-time descendant, now SOTA for images and video |

> [!WARNING]
> **Known failure mode worth knowing about** — Nalisnick et al. (2018), *Do Deep Generative Models
> Know What They Don't Know?*, showed that a flow trained on CIFAR-10 assigns **higher** likelihood to
> SVHN images than to CIFAR-10 images. High likelihood ≠ in-distribution. The usual explanation is
> that likelihood is dominated by low-level statistics (SVHN images are smoother, hence more
> "probable" under a model that learned local pixel correlations). **Do not use raw likelihood as an
> OOD detector without correction** (typicality tests and likelihood ratios against a background
> model are the standard fixes).

---

## 8. Continuous normalizing flows: the bridge to modern methods

Take the number of layers to infinity and each layer's change to zero. The discrete composition
becomes an ODE:

$$\frac{dz(t)}{dt} = v_\theta(z(t), t), \qquad z(0) \sim p_Z, \quad x = z(1)$$

The log-density evolves by the **instantaneous change of variables** formula:

$$\frac{d\log p(z(t))}{dt} = -\operatorname{tr}\!\left(\frac{\partial v_\theta}{\partial z}\right)$$

> [!TIP]
> **Note the huge simplification**: a log-*determinant* became a **trace**. Traces are cheap to
> estimate — Hutchinson's estimator gives $\operatorname{tr}(A) = \mathbb{E}_\epsilon[\epsilon^\top A\epsilon]$
> with one vector-Jacobian product. And $v_\theta$ has *no architectural constraints at all* — any
> network defines a valid flow, because the ODE is automatically invertible (just integrate
> backwards).

> [!WARNING]
> **The catch that killed neural ODEs for generation**: training required backpropagating through
> an adaptive ODE solver, which is slow and memory-hungry, and the learned trajectories were curved
> (requiring many solver steps).

✅ **Flow matching (2023) removed the catch**: instead of simulating the ODE during training,
*regress directly onto a known conditional velocity field*. Training becomes a simple
regression — as cheap and stable as diffusion — while sampling follows near-straight paths that
need few steps. This is the architecture behind Stable Diffusion 3, Flux and several video models.

→ [Flow matching & rectified flow](../05-diffusion-and-vision/04-flow-matching.md) — read this
page next if you care about modern image generation.

---

## 9. Exercises

**Problem 1 — the invertible-1×1-conv log-det.** A Glow-style $1\times1$ convolution has weight
matrix $W$ with $\det W = 2.5$, applied to a $16\times16$ feature map with $c$ channels. What is
the total $\log|\det J|$ contribution of this layer (per §4)? If the feature map were $32\times32$
instead, how would the contribution change, and why does this make spatial resolution a
*multiplier* on the layer's log-det rather than something that changes $\det W$ itself?

<details><summary>Solution</summary>

Per §4: $\log|\det J| = h\times w\times\log|\det W|$. At $16\times16$:
$256\times\log(2.5) = 256\times0.916=234.6$.

At $32\times32$: $1024\times\log(2.5)=938.6$ — exactly **4×** larger, matching the 4× increase in
spatial positions ($32^2/16^2=4$). This is because $W$ acts *independently and identically* at
every spatial location (it's a $1\times1$ conv — no spatial mixing), so the *same* per-location
$\log|\det W|$ term is simply summed over however many locations there are. $\det W$ itself never
changes with resolution; only the number of times it's applied does.

</details>

**Problem 2 — MAF/IAF trade-off, applied.** You need a model for a real-time synthesizer that
must generate audio samples fast, but you don't care about evaluating the likelihood of existing
recordings. Using §4's MAF/IAF table, which would you pick and why? Now suppose the requirement
flips: you need to score thousands of pre-recorded clips by likelihood as fast as possible, but
generation speed doesn't matter (e.g., an anomaly-detection pipeline). Which would you pick now?

<details><summary>Solution</summary>

Real-time synthesis (fast generation, don't care about density evaluation speed): **IAF** — "fast
sampling" is its whole advantage, at the cost of slow (sequential) density evaluation, which
you've said you don't need.

Batch likelihood scoring (fast density, generation speed irrelevant): **MAF** — "1 parallel pass"
for density evaluation is exactly what you want; its slow sequential sampling is irrelevant since
you're not generating anything.

This mirrors the real historical case in §4: Parallel WaveNet used exactly this reasoning in
reverse — trained an MAF-style teacher (fast to train/score) then **distilled it into an IAF
student** so that deployment (many users needing fast synthesis) could use the direction that's
fast to sample from, while training could use the direction that's fast to fit.

</details>

**Problem 3 — dequantization, why it matters.** A flow is trained directly on integer pixel
values $\{0,...,255\}$ without adding uniform noise. Using §5's warning, what happens to the
reported log-likelihood as training continues, and why is comparing this number against a
properly-dequantized model's log-likelihood meaningless?

<details><summary>Solution</summary>

Per §5: without dequantization, the model is fitting a *continuous* density to data supported
only on a discrete grid (the integers). The optimal continuous density for discrete support is a
sum of Dirac deltas at each observed integer — infinitely tall, infinitely thin spikes — so as
training continues, the model can push $\log p(x)$ at the training integers toward $+\infty$ by
concentrating density arbitrarily tightly around them. This isn't learning "the data is more
predictable"; it's an artifact of applying a continuous-density tool to discrete data.

The comparison is meaningless because a dequantized model's likelihood is a proper (finite) lower
bound on the *discrete* log-likelihood, while the non-dequantized model's number is an
unbounded, ill-defined quantity that has nothing to do with how well it models discrete pixel
values — the two numbers aren't measuring the same thing, even though they're both called
"log-likelihood."

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | $\log p_X(x) = \log p_Z(z) + \log|\det J_{f^{-1}}|$ — exact likelihood, no bound. |
| 2 | The whole design problem is making $\log|\det J|$ cheap → force the Jacobian triangular. |
| 3 | Affine coupling: transform half the variables conditioned on the other half. $s,t$ are never inverted, so they can be arbitrary networks. |
| 4 | Always permute (or use a learned $1\times1$ conv) between coupling layers. |
| 5 | MAF and IAF are inverses: fast density XOR fast sampling, never both. |
| 6 | Dequantize discrete data or your likelihoods are meaningless. |
| 7 | Invertibility forbids dimensionality reduction and constrains topology — the core limitation. |
| 8 | High likelihood ≠ in-distribution (the CIFAR→SVHN result). |
| 9 | The continuous-time limit turns $\log\det$ into a trace and removes all architectural constraints → CNFs → flow matching → modern SOTA. |

---

## Further reading

- Dinh et al., [*Density Estimation using Real NVP*](https://arxiv.org/abs/1605.08803) (2016).
- Kingma & Dhariwal, [*Glow*](https://arxiv.org/abs/1807.03039) (2018).
- Papamakarios et al., [*Normalizing Flows for Probabilistic Modeling and Inference*](https://arxiv.org/abs/1912.02762) (2021) — the definitive survey.
- Chen et al., [*Neural Ordinary Differential Equations*](https://arxiv.org/abs/1806.07366) (2018); Grathwohl et al., [*FFJORD*](https://arxiv.org/abs/1810.01367) (2018).
- Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?* (2018).
- Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747) (2023).

**Next** → [Energy-based models](05-energy-based-models.md)
