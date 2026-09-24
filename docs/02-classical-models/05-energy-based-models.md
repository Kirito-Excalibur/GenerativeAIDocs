# Energy-Based Models

> **Summary** — Define a scalar "energy" $E_\theta(x)$ for every configuration and set
> $p_\theta(x) \propto e^{-E_\theta(x)}$. Maximum architectural freedom — no invertibility, no
> latent, no ordering, no bound. The price is the normalizing constant $Z = \int e^{-E(x)}dx$,
> which is intractable and infects both training and sampling with MCMC. EBMs are rarely deployed
> directly, but they are the conceptual parent of score matching and diffusion, and the framework
> that explains *why* score-based models work.

**Prerequisites**: → [Probability & information theory](../01-foundations/02-probability-and-information-theory.md) · **Next**: Part III → [Tokenization](../03-sequence-models/01-tokenization.md)

---

## 1. The Boltzmann form

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \qquad Z_\theta = \int \exp(-E_\theta(x))\,dx$$

$E_\theta : \mathbb{R}^d \to \mathbb{R}$ is **any** neural network. Low energy = high probability.

> [!TIP]
> **Intuition — a landscape.** Think of $E$ as a physical terrain. Data points sit in valleys;
> implausible configurations sit on mountains. Sampling means dropping a marble somewhere and
> letting it roll downhill, with a bit of thermal jitter so it doesn't get stuck in the first dent it
> finds.

```
  E(x)  high energy = improbable
    │╲                            ╱│
    │ ╲      ╱╲        ╱╲        ╱ │
    │  ╲    ╱  ╲      ╱  ╲      ╱  │
    │   ╲__╱    ╲____╱    ╲____╱   │
    │    ▼       ▼         ▼       │
    └────●───────●─────────●───────┴──► x
       data     data      data
     (valleys = high probability)

  Training:  push DOWN on data,  push UP everywhere else
```

**The appeal**: you get to specify only a *relative* preference. No requirement that the network be
invertible (flows), produce a proper conditional (AR), or admit a tractable posterior (VAE). It is
the least constrained formulation possible.

**The problem**: $Z_\theta$ is a $d$-dimensional integral over an arbitrary neural network. For
$d = 3072$ it cannot be computed, estimated reliably, or differentiated.

---

## 2. Training: the contrastive gradient

**Derivation.** Maximize $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta$$

The second term:

$$
\begin{aligned}
\nabla_\theta \log Z_\theta &= \frac{1}{Z_\theta}\nabla_\theta\int e^{-E_\theta(x')}dx'
= \frac{1}{Z_\theta}\int e^{-E_\theta(x')}\big(-\nabla_\theta E_\theta(x')\big)dx'\\
&= -\int \underbrace{\frac{e^{-E_\theta(x')}}{Z_\theta}}_{p_\theta(x')}\nabla_\theta E_\theta(x')\,dx'
= -\mathbb{E}_{x'\sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]
\end{aligned}
$$

Therefore:

$$\boxed{\;\nabla_\theta \mathcal{L} = \underbrace{\mathbb{E}_{x\sim p_{\text{data}}}\big[\nabla_\theta E_\theta(x)\big]}_{\text{positive phase: push data energy DOWN}} - \underbrace{\mathbb{E}_{x'\sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]}_{\text{negative phase: push model energy UP}}\;}$$

> [!TIP]
> **Read the two phases.** The positive phase digs valleys under the real data. The negative phase
> fills in valleys wherever the *model currently thinks* data lives but it doesn't. At convergence
> the two cancel: the model's samples are distributed exactly like the data.

```
  Before training                    Gradient step                 After
  E(x)                               ↓ push down at data
   ╲___╱▔▔╲___╱                      ↑ push up at model samples
                                     
     ● ●        ○ ○                  ●=real  ○=model sample
                                     
                          ──────►    valleys form under ●,
                                     hills form under ○
```

> [!WARNING]
> **The problem is in the negative phase**: it requires samples from $p_\theta$, the very thing you
> are trying to learn. This is a chicken-and-egg loop, and it is why EBM training is hard.

---

## 3. Sampling: Langevin dynamics

$$x_{t+1} = x_t - \frac{\eta}{2}\nabla_x E_\theta(x_t) + \sqrt{\eta}\,\epsilon_t, \qquad \epsilon_t\sim\mathcal{N}(0,I)$$

> [!TIP]
> **Gradient descent with noise.** The drift term $-\nabla_x E$ rolls the sample downhill toward
> high probability; the noise term prevents it collapsing onto the single global minimum and gives
> it the right spread. As $\eta \to 0$ and $t\to\infty$, $x_t$ converges to a true sample from
> $p_\theta$ — **and note the formula never mentions $Z$**.

**Why $Z$ disappears** — this is the single most important observation in this page:

$$\nabla_x \log p_\theta(x) = \nabla_x\big(-E_\theta(x) - \log Z_\theta\big) = -\nabla_x E_\theta(x)$$

$\log Z_\theta$ is constant **in $x$**, so its gradient with respect to $x$ is zero. The
intractable constant is irrelevant to sampling and to any method that only needs the *score*
$\nabla_x \log p(x)$.

✅ **This is exactly the loophole that diffusion and score-based models exploit.** They never
compute a normalized density; they only ever estimate its gradient.
→ [Score-based models](../05-diffusion-and-vision/02-score-based-models.md)

> [!WARNING]
> **But MCMC in high dimensions is slow.** Langevin chains need thousands of steps to mix, and
> between well-separated modes they essentially never mix — the chain has to cross a high-energy
> barrier, which happens with probability $\propto e^{-\Delta E}$.

---

## 4. Contrastive divergence: the practical hack

Running Langevin to convergence in the inner loop of training is hopeless. **Contrastive
divergence (CD-$k$)**: run only $k$ steps, starting from the data.

```
  CD-k:   x_data ──[k Langevin steps]──► x_negative
          use x_negative in the negative phase. k is often 1.
```

It works surprisingly well and made RBMs trainable in the 2000s, but it is a **biased**
gradient: you're not sampling from $p_\theta$, you're sampling from a $k$-step-perturbed data
distribution.

| Variant | Idea |
|---|---|
| CD-$k$ | $k$ MCMC steps from the data |
| **PCD** (persistent CD) | keep a buffer of chains across parameter updates; chains stay near $p_\theta$ |
| **Replay buffer + reinit** | PCD with 5% of chains restarted from noise — prevents the buffer going stale |
| Short-run MCMC | accept the bias, treat the fixed-length chain as *defining* the model |

Modern EBM training (Du & Mordatch, 2019), in sketch:

```python
buffer = torch.rand(10000, *img_shape) * 2 - 1     # persistent chains

def sample_negatives(E, n, steps=60, step_size=10.0, noise=0.005):
    idx = torch.randint(0, len(buffer), (n,))
    x = buffer[idx].clone()
    x[torch.rand(n) < 0.05] = torch.rand_like(x[0]).expand(...) * 2 - 1   # 5% reinit
    x.requires_grad_(True)
    for _ in range(steps):
        grad = torch.autograd.grad(E(x).sum(), x)[0]
        x = x - step_size * grad + noise * torch.randn_like(x)
        x = x.clamp(-1, 1).detach().requires_grad_(True)
    buffer[idx] = x.detach()
    return x.detach()

for real in loader:
    fake = sample_negatives(E, real.size(0))
    loss = E(real).mean() - E(fake).mean()          # the two phases
    loss = loss + 1.0 * (E(real)**2 + E(fake)**2).mean()   # L2 reg on energies
    opt.zero_grad(); loss.backward(); opt.step()
```

> [!WARNING]
> The `E(real)**2 + E(fake)**2` regularizer is not optional. Without it the energies drift to
> $\pm\infty$ (the loss $E(\text{real}) - E(\text{fake})$ is unbounded below) and training diverges.

---

## 5. Score matching: training without MCMC at all

If MCMC is the problem, can we train an EBM without it? **Yes** — match the *score* instead of the
density.

$$\mathcal{L}_{\text{SM}} = \tfrac12\,\mathbb{E}_{p_{\text{data}}}\big[\|\nabla_x\log p_\theta(x) - \nabla_x\log p_{\text{data}}(x)\|^2\big]$$

We don't know $\nabla_x\log p_{\text{data}}$, but Hyvärinen (2005) showed integration by parts
removes it:

$$\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}\!\left[\operatorname{tr}\!\big(\nabla_x^2\log p_\theta(x)\big) + \tfrac12\|\nabla_x\log p_\theta(x)\|^2\right] + \text{const}$$

- ✅ No $Z$ (it's a score), no MCMC (it's an expectation under the data).
- ⚠️ But the trace of the Hessian costs $O(d)$ backward passes — infeasible for images.

**Denoising score matching (Vincent, 2011) fixes the cost.** Perturb the data with Gaussian noise
of scale $\sigma$ and match the score of the *noisy* distribution:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x,\tilde x}\left[\left\|s_\theta(\tilde x) - \frac{x - \tilde x}{\sigma^2}\right\|^2\right], \qquad \tilde x = x + \sigma\epsilon$$

> [!TIP]
> **The target $\frac{x - \tilde x}{\sigma^2} = -\frac{\epsilon}{\sigma}$ is just the noise you
> added.** No Hessian, no MCMC, no $Z$ — a plain regression problem.

$$\boxed{\text{Denoising score matching} = \text{the diffusion training objective}}$$

This is the direct lineage: EBM → score matching → denoising score matching → NCSN → DDPM →
Stable Diffusion. Every modern image generator is a descendant of this line, with the MCMC replaced
by a learned, noise-level-annealed sampler.
→ [Score-based models](../05-diffusion-and-vision/02-score-based-models.md)

---

## 6. The historical line

| Model | Year | Energy function | Note |
|---|---|---|---|
| Hopfield network | 1982 | $-\frac12 x^\top W x$ | associative memory; the 2024 physics Nobel |
| **Boltzmann machine** | 1985 | $-\frac12 x^\top Wx - b^\top x$ | fully connected, hopelessly slow |
| **RBM** | 2002 | bipartite visible/hidden | CD-1 made it trainable; powered deep belief nets (2006) |
| Deep Boltzmann Machine | 2009 | stacked RBMs | the pre-2012 "deep learning" |
| **NCSN** | 2019 | multi-scale score network | annealed Langevin; directly led to diffusion |
| **JEM** | 2019 | reinterpret a classifier's logits as an EBM | one model for $p(y\mid x)$ and $p(x)$ |
| IGEBM | 2019 | deep ConvNet energy + PCD | showed EBMs can work on images |

> [!TIP]
> **JEM is a lovely idea worth knowing.** A standard classifier outputs logits $f_\theta(x)[y]$.
> Define $E_\theta(x, y) = -f_\theta(x)[y]$. Then:

$$p(y\mid x) = \frac{e^{f_\theta(x)[y]}}{\sum_{y'}e^{f_\theta(x)[y']}} \quad\text{(the usual softmax)}$$
$$p(x) = \frac{\sum_y e^{f_\theta(x)[y]}}{Z} \quad\text{(logsumexp over classes = free energy)}$$

**Your classifier was secretly a generative model the whole time.** Training with the joint
objective gave better calibration, adversarial robustness and OOD detection — at the cost of
EBM-grade training instability.

---

## 7. Honest assessment

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Total architectural freedom | $Z$ intractable → no likelihood, no perplexity, no clean model comparison |
| Naturally compositional: $E_{\text{total}} = \sum_i E_i$ multiplies the distributions | MCMC sampling is slow and mixes badly between modes |
| One model serves classification, generation, OOD, inpainting | Training is unstable; needs regularizers and replay buffers |
| Directly expresses constraints and preferences | Heavily outperformed on sample quality |

> [!TIP]
> **The compositionality point deserves a mention** because nothing else offers it so cleanly:
> if $E_1$ encodes "is a cat" and $E_2$ encodes "is outdoors", then $E_1 + E_2$ is a valid energy for
> "cat outdoors" — no retraining, no conditioning mechanism. Compositional visual generation work has
> exploited this, and classifier guidance in diffusion is exactly the same algebra:
> $\nabla\log p(x\mid c) = \nabla\log p(x) + \nabla\log p(c\mid x)$ is a sum of two scores, i.e. a
> sum of two energies. → [Latent diffusion](../05-diffusion-and-vision/03-latent-diffusion.md)

---

## 8. Exercises

**Problem 1 — Langevin step, by hand.** Let $E(x) = x^2$ (so $\nabla_x E = 2x$), $\eta=0.1$,
current point $x_t=0.5$, and a drawn noise value $z_t=0.3$. Compute $x_{t+1}$ using §3's update
rule. Which direction did the drift term push $x$ (toward or away from the energy minimum at
$x{=}0$), and would a *larger* $\eta$ make the noise term relatively more or less influential?

<details><summary>Solution</summary>

$$x_{t+1} = x_t - \frac{\eta}{2}\nabla_xE(x_t) + \sqrt\eta\,z_t
= 0.5 - 0.05(1.0) + \sqrt{0.1}(0.3) = 0.5 - 0.05 + 0.0949 = 0.545$$

The drift term ($-0.05$) pushed *toward* 0 (the minimum of $E(x)=x^2$), as expected — gradient
descent on the energy. The noise term ($+0.0949$) happened to push the other way and dominated
this particular step, which is normal (Langevin dynamics is stochastic, not monotonically
descending).

Larger $\eta$: the drift term scales linearly in $\eta$ while the noise term scales as
$\sqrt\eta$ — so as $\eta$ grows, drift grows *faster* than noise, meaning **noise becomes
relatively less influential** at large $\eta$ (and vice versa: at small $\eta$, noise dominates
relative to drift). This is why $\eta\to0$ is needed for the chain to actually sample from
$p_\theta$ correctly (per §3) — at large step sizes the balance between drift and noise is off
from what the correct continuous-time process requires.

</details>

**Problem 2 — energy composition.** Two energy functions score two candidate images $a, b$:
$E_1(a)=1.0, E_1(b)=3.0$ (a "cat" detector — lower is more cat-like) and $E_2(a)=2.0,
E_2(b)=0.5$ (an "outdoor" detector). Using §7's compositionality property, which image does the
*combined* energy $E_1+E_2$ favor as "cat outdoors," and does either individual detector agree
with the combined verdict?

<details><summary>Solution</summary>

$E_{\text{total}}(a) = 1.0+2.0=3.0$; $E_{\text{total}}(b)=3.0+0.5=3.5$. Lower energy is more
probable, so **image $a$** is favored as "cat outdoors" overall.

Individually: $E_1$ alone prefers $a$ (1.0 < 3.0, more cat-like) — **agrees** with the combined
verdict. $E_2$ alone prefers $b$ (0.5 < 2.0, more outdoor-like) — **disagrees**. So the composed
energy doesn't simply follow either detector; it's a genuine trade-off, exactly the
"no retraining, no conditioning mechanism" compositional algebra described in §7 — you get a
principled answer to "cat AND outdoors" just by adding two independently-trained energies.

</details>

**Problem 3 — CD-$k$ bias, intuition check.** Contrastive divergence with $k=1$ starts Langevin
chains from real data points and runs one step. Using §4, explain why this produces a *biased*
gradient (compare with the unbiased "true" MLE gradient in §2, which requires negative-phase
samples from the *actual* model distribution $p_\theta$, not from a 1-step perturbation of the
data). What would happen to the bias as $k\to\infty$?

<details><summary>Solution</summary>

The unbiased gradient (§2) needs $\mathbb{E}_{x'\sim p_\theta}[\nabla_\theta E_\theta(x')]$ — an
expectation under the model's *true, converged* distribution. CD-$k$ substitutes a **1-step
Langevin perturbation of a real data point** for that expectation — which is a sample from
"data slightly nudged by the model's current gradient field," not a sample from $p_\theta$ itself
(unless the chain has fully mixed, which 1 step never achieves). Because the negative-phase
samples stay anchored near the data manifold instead of exploring wherever $p_\theta$ actually
puts mass, the gradient systematically under-corrects regions of $x$-space that are far from any
training point but where the model might currently assign high probability.

As $k\to\infty$, the chain has arbitrarily long to mix, so its distribution converges to the true
stationary distribution of the Langevin process, which (per §3) *is* $p_\theta$ — so the CD-$k$
gradient becomes the exact, unbiased MLE gradient from §2 in the limit. This is exactly why §4
frames CD-$k$ as a "practical hack": it trades bias (finite $k$) for tractability (you can't
actually run $k=\infty$ inside every training step).

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | $p(x) = e^{-E(x)}/Z$. Any network can be an energy function — maximal freedom, maximal trouble. |
| 2 | The MLE gradient has two phases: lower energy at data, raise it at model samples. |
| 3 | The negative phase needs samples from the model → MCMC → the central difficulty. |
| 4 | **$\nabla_x\log p(x) = -\nabla_x E(x)$**: the score doesn't depend on $Z$. This is the loophole everything modern uses. |
| 5 | Langevin dynamics samples using only the score, but mixes slowly across modes. |
| 6 | Denoising score matching removes the MCMC *and* the Hessian → it is literally the diffusion loss. |
| 7 | EBM → score matching → DSM → NCSN → DDPM is a single continuous lineage. |
| 8 | Energies add, so EBMs compose — the same algebra as classifier guidance. |

---

## Further reading

- LeCun et al., *A Tutorial on Energy-Based Learning* (2006) — still the best conceptual introduction.
- Hinton, *Training Products of Experts by Minimizing Contrastive Divergence* (2002).
- Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching* (2005).
- Vincent, *A Connection Between Score Matching and Denoising Autoencoders* (2011) — the key bridge.
- Du & Mordatch, *Implicit Generation and Generalization in Energy-Based Models* (2019).
- Grathwohl et al., [*Your Classifier is Secretly an Energy Based Model*](https://arxiv.org/abs/1912.03263) (JEM, 2019).
- Song & Kingma, [*How to Train Your Energy-Based Models*](https://arxiv.org/abs/2101.03288) (2021) — the modern survey.

**Next** → Part III: [Tokenization](../03-sequence-models/01-tokenization.md)
