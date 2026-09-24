# Score-Based Models and SDEs

> **Summary** — The same algorithm as DDPM, discovered independently from a different direction.
> Instead of "learn to denoise", the framing is "learn $\nabla_x\log p(x)$, the score, then follow
> it uphill." This view unifies diffusion with energy-based models, reveals that the discrete
> chain is a discretized stochastic differential equation, and produces the **probability-flow
> ODE** — a deterministic sampler that turns a diffusion model into an invertible flow and enables
> 20-step sampling.

**Prerequisites**: → [Diffusion models](01-diffusion-models.md), → [Energy-based models](../02-classical-models/05-energy-based-models.md) · **Next**: → [Latent diffusion](03-latent-diffusion.md)

---

## 1. The score function

$$s(x) = \nabla_x \log p(x)$$

**The gradient of the log-density with respect to the input** — a vector field pointing in the
direction of increasing probability.

```
   density p(x)                    score ∇ log p(x)
                                   
        ╱▔▔╲                        →→→→  ←←←←
       ╱    ╲                      →→        ←←
   ___╱      ╲___                 →            ←
                                  arrows point UPHILL toward the mode,
   ──────●────────                and are LONGEST on the steep slopes
```

> [!TIP]
> **Why the score, and not the density?** Recall
> $\nabla_x\log p(x) = \nabla_x\left(-E(x) - \log Z\right) = -\nabla_x E(x)$.
> **The intractable normalizing constant vanishes.** You can learn the score of an
> arbitrarily complex distribution without ever computing or approximating $Z$. This is the single
> most important idea on this page.

---

## 2. From score to samples: Langevin dynamics

$$x_{t+1} = x_t + \frac{\eta}{2}\,s_\theta(x_t) + \sqrt{\eta}\,z_t, \qquad z_t \sim \mathcal{N}(0,I)$$

Gradient ascent on log-probability, plus noise. As $\eta\to0$, $t\to\infty$, the iterates converge
to samples from $p$.

> [!WARNING]
> **Two problems kill naive Langevin sampling in high dimensions:**

**(a) The score is undefined off the data manifold.** Real data occupies a thin manifold; in the
vast empty regions $p(x) \approx 0$ and $\nabla\log p$ is both numerically meaningless and never
trained (no training data lands there). Starting from random noise, your sample is in exactly such
a region, and the score points nowhere useful.

**(b) Mixing between modes is exponentially slow.** To move from one mode to another the chain must
cross a low-density valley, which it does with probability $\propto e^{-\Delta}$.

---

## 3. The fix: noise conditioning

> [!TIP]
> **Song & Ermon's insight (2019)**: perturb the data with noise at *many* scales. Large noise
> smooths the distribution so its support covers the whole space and the score is defined
> everywhere. Small noise preserves fine detail. Learn the score at every noise level, then
> **anneal** from large to small during sampling.

```
  σ = 1.0 (heavy noise)     σ = 0.3            σ = 0.05 (light)
  ╱▔▔▔▔▔▔▔▔▔╲              ╱▔▔╲  ╱▔▔╲          ╱╲    ╱╲
 ╱           ╲            ╱    ╲╱    ╲        ╱  ╲  ╱  ╲
 broad, smooth,           modes emerging      sharp, true structure
 score defined            score still good    score undefined off-manifold
 EVERYWHERE

  ANNEALED SAMPLING: start at σ=1.0 (find the right region),
                     decrease σ (refine within it),
                     end at σ≈0 (add fine detail)
```

**The training objective** is denoising score matching at every noise level:

$$\mathcal{L} = \mathbb{E}_{\sigma}\,\mathbb{E}_{x\sim p_{\text{data}}}\,\mathbb{E}_{\tilde x\sim\mathcal{N}(x,\sigma^2I)}
\left[\lambda(\sigma)\left\|s_\theta(\tilde x, \sigma) + \frac{\tilde x - x}{\sigma^2}\right\|^2\right]$$

**The target $-\frac{\tilde x - x}{\sigma^2}$ is exactly the score of the Gaussian that produced
$\tilde x$**: for $\tilde x \sim \mathcal{N}(x, \sigma^2 I)$,
$\nabla_{\tilde x}\log q(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma}$.
No Hessian, no MCMC, no $Z$ — just regression onto scaled noise.

---

## 4. The equivalence with DDPM

Two independently-developed methods turn out to be the same algorithm.

In DDPM, $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, so
$q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ and

$$\nabla_{x_t}\log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

$$\boxed{\;s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}\;}$$

**The noise-prediction network and the score network are the same network, up to a known scalar.**

| DDPM framing | Score framing |
|---|---|
| predict the noise $\epsilon$ | estimate the score $\nabla\log p$ |
| discrete chain of $T$ steps | continuous SDE |
| variational bound (ELBO) | denoising score matching |
| ancestral sampling | annealed Langevin |
| $\bar\alpha_t$ schedule | $\sigma(t)$ schedule |

> [!TIP]
> Neither framing is more correct. The DDPM view makes training obvious; the score/SDE view makes
> *sampling* improvable, which is where all the subsequent progress came from.

---

## 5. The continuous-time SDE view

Take $T\to\infty$ and the discrete chain becomes a stochastic differential equation:

$$dx = f(x,t)\,dt + g(t)\,dw \qquad\text{(forward: data → noise)}$$

where $dw$ is Brownian motion.

**Anderson's theorem (1982)** gives the time-reversal:

$$\boxed{\;dx = \left[f(x,t) - g(t)^2\nabla_x\log p_t(x)\right]dt + g(t)\,d\bar w\;}$$

> [!TIP]
> **This is the central result.** The reverse of *any* diffusion SDE is another SDE, and the only
> unknown in it is the **score**. Learn the score and you can run any diffusion backwards.

**The two canonical SDEs:**

| Name | Forward SDE | Discrete analogue |
|---|---|---|
| **VP** (variance preserving) | $dx = -\frac12\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$ | DDPM |
| **VE** (variance exploding) | $dx = \sqrt{\frac{d\sigma^2(t)}{dt}}\,dw$ | SMLD / NCSN |

VP keeps total variance at 1 (signal shrinks as noise grows). VE keeps the signal and lets variance
grow to a large $\sigma_{\max}$. They are related by a simple rescaling of $x$ and $t$ — an
important unification, since it means DDPM and NCSN were never really different models.

### The probability flow ODE

Every SDE has a corresponding **deterministic** ODE with *identical marginal distributions*
$p_t(x)$ at every time:

$$\boxed{\;\frac{dx}{dt} = f(x,t) - \tfrac12 g(t)^2\nabla_x\log p_t(x)\;}$$

```
   SDE sampling (stochastic)          ODE sampling (deterministic)

   noise ──┬─╮ ╭─┬──► sample A        noise ─────────────────► sample A
           ╰─┴─╯ │                           one smooth path
           │ ╭───┴──► sample B        
           ╰─╯                        same x_T always gives the same x_0
   different runs → different samples
   
   same MARGINAL distribution at every t
```

> [!TIP]
> **Why this is so useful — three consequences:**

1. **Fewer steps.** ODEs can be solved with sophisticated numerical methods (Runge-Kutta, multistep
   solvers) that take far larger steps than a stochastic chain allows. 1000 → 20 steps.
2. **Exact likelihoods.** An ODE is an invertible map, so the instantaneous change-of-variables
   formula applies. **A diffusion model is secretly a continuous normalizing flow.**
   → [Normalizing flows §8](../02-classical-models/04-normalizing-flows.md#8-continuous-normalizing-flows-the-bridge-to-modern-methods)
3. **Meaningful latents.** A deterministic, invertible map means you can encode a real image to its
   noise vector, edit that vector, and decode — the basis of several editing techniques.

**DDIM** (Song et al., 2020) is exactly a discretization of this ODE:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted } x_0}
+ \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t,t)}_{\text{direction pointing to } x_t} + \sigma_t z$$

With $\sigma_t = 0$ it is fully deterministic. And crucially, **DDIM works with a subsequence of
timesteps** — sample $t \in \{0, 50, 100, \dots, 1000\}$ and you get a 20-step sampler from a model
trained with $T = 1000$, with no retraining.

---

## 6. Samplers: the practical menu

Steps needed for good quality on a standard latent diffusion model:

| Sampler | Steps | Type | Notes |
|---|---|---|---|
| DDPM (ancestral) | 1000 | SDE | the original; slow |
| **DDIM** | 20–50 | ODE | deterministic, the first big speedup |
| Euler | 20–30 | ODE | simplest ODE solver |
| Euler-a (ancestral) | 20–30 | SDE | adds noise each step; more variation |
| Heun | 15–25 | ODE | 2nd-order; 2 model evals per step |
| **DPM-Solver++ (2M)** | **15–25** | ODE | multistep, exploits the semi-linear structure |
| UniPC | 10–20 | ODE | unified predictor–corrector |
| **LCM / consistency** | **2–8** | distilled | requires a distilled model |
| **Rectified flow / distilled** | **1–4** | distilled | → [Flow matching](04-flow-matching.md) |

> [!TIP]
> **Why DPM-Solver is so much better than Euler.** The probability-flow ODE is **semi-linear**: it
> has an analytically solvable linear part plus a nonlinear part involving $\epsilon_\theta$. A
> generic solver treats the whole thing as a black box; DPM-Solver solves the linear part *exactly*
> and only approximates the nonlinear remainder. The error at a given step count drops
> substantially.

> [!WARNING]
> **SDE vs ODE samplers is a real quality trade-off, not just speed.** Stochastic samplers inject
> fresh noise at each step, which lets the model *correct* accumulated errors — so they often reach
> better final quality given enough steps. Deterministic samplers are faster and reproducible but
> compound their errors. Karras et al. (2022) analysed this carefully and recommend a *little*
> stochasticity (their "churn" parameter) as the best of both.

**Karras schedule** — as important as the sampler choice. Instead of uniform timesteps, space the
noise levels as

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right)\right)^{\rho},\qquad \rho = 7$$

This allocates more steps to the low-noise end, where small errors are perceptually visible.
Combined with a good solver it is worth several steps of budget for free.

---

## 7. Distillation to very few steps

| Method | Idea | Steps |
|---|---|---|
| **Progressive distillation** | student learns to take 2 teacher steps in 1; repeat $\log_2$ times | 1000 → 4 |
| **Consistency models** | train $f(x_t,t)$ to map *any* point on a trajectory to its endpoint | 1–4 |
| **LCM** | consistency distillation applied in latent space | 2–8 |
| **ADD / LADD** | adversarial loss + distillation (a GAN discriminator on outputs) | 1–4 |
| **Rectified flow (reflow)** | straighten the trajectories, then distill | 1–2 |

> [!TIP]
> **The consistency-model idea is elegant.** The probability-flow ODE defines a trajectory from
> noise to image. Define $f(x_t, t) = x_0$ — the endpoint — for every point on that trajectory. Train
> $f$ to be **self-consistent**: $f(x_t,t) = f(x_{t'},t')$ for any two points on the same trajectory.
> Then a single evaluation of $f$ jumps straight from noise to image.

```
   Multi-step diffusion              Consistency model
   
   x_T ──►x_T-1──►...──►x_1──►x_0     x_T ────────────────────► x_0
    each arrow = 1 network eval        x_t ───────────────────► x_0
                                       ALL points map directly
                                       to the same endpoint
```

Quality at 1 step still lags the multi-step teacher (fine detail and prompt adherence suffer),
but 4-step distilled models are close enough for most production use, and this is where
real-time image generation comes from.

---

## 8. Implementation

A DPM-Solver++(2M) sampler, which is what most production systems actually use:

```python
@torch.no_grad()
def dpm_solver_pp_2m(model, x, sigmas, **kwargs):
    """Multistep 2nd-order DPM-Solver++. `sigmas` is a decreasing schedule
    ending at 0, e.g. from the Karras formula."""
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i], **kwargs)      # predicted x0
        t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            # first step, or final step: 1st-order (DDIM) update
            x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised
        else:
            h_last = t - (-sigmas[i - 1].log())
            r = h_last / h
            # 2nd-order correction using the previous denoised estimate
            d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * d
        old_denoised = denoised
    return x

def karras_sigmas(n, sigma_min=0.03, sigma_max=14.6, rho=7.0):
    ramp = torch.linspace(0, 1, n)
    min_inv, max_inv = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])     # append 0 for the final step
```

> [!TIP]
> Note the "multistep" part: the 2nd-order correction reuses the *previous* step's denoised
> estimate rather than calling the model twice. You get 2nd-order accuracy at 1st-order cost, which
> is why DPM-Solver++(2M) is the default in most inference stacks.

---

## 9. Exercises

**Problem 1 — score from noise prediction.** Using §4's boxed identity
$s_\theta(x_t,t)=-\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$, if a trained model predicts
$\epsilon_\theta=0.6$ at a timestep where $\sqrt{1-\bar\alpha_t}=0.3$, what is the implied score?
What does the *sign* of the score tell you about which direction increases $\log p(x_t)$?

<details markdown="1"><summary>Solution</summary>

$$s_\theta = -0.6/0.3 = -2.0$$

The score is $\nabla_x\log p(x)$ — it points in the direction that *increases* log-probability. A
score of $-2.0$ (negative) means moving $x$ in the *negative* direction (of whichever coordinate
this represents) increases $\log p(x_t)$ locally — i.e., Langevin dynamics (§2's update rule)
would nudge $x$ *downward* along this component. The specific sign here is a direct, deterministic
consequence of the noise prediction's sign: a positive $\epsilon_\theta$ always yields a negative
score (and vice versa), since the noise-to-score conversion is just a negative scalar multiple.

</details>

**Problem 2 — VP vs VE, applied.** A colleague trains a score model using the VE
(variance-exploding) SDE, then tries to sample using a DDIM-style update rule (which assumes the
VP parameterization's $\bar\alpha_t$ schedule). Using §5's table, explain why this is likely to
fail or require modification, referencing what VP and VE actually preserve/allow to grow.

<details markdown="1"><summary>Solution</summary>

Per §5's table, **VP** (variance preserving) keeps total variance at 1 throughout the forward
process — signal shrinks as noise grows, governed by $\bar\alpha_t$, which is exactly the
quantity DDIM's update rule (§5, borrowed from → [Diffusion models
§6](01-diffusion-models.md#6-parameterization-choices)) is built around. **VE** (variance
exploding) instead keeps the *signal* fixed and lets total variance grow via $\sigma(t)$, with no
$\bar\alpha_t$-style shrinking schedule at all.

Using a DDIM update (which reads off $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$ from a VP
schedule) on a VE-trained model is a category error: the VE model's score network was trained
under an entirely different noise-injection convention, and the DDIM formula's implicit
assumptions about signal-to-noise ratio at each $t$ simply don't hold. §5 notes VP and VE are
related "by a simple rescaling of $x$ and $t$" — so the fix is to apply that rescaling
explicitly (or use a sampler written for the VE convention, such as the annealed Langevin
approach the VE formulation was originally paired with) rather than mixing conventions.

</details>

**Problem 3 — DPM-Solver's multistep trick, why it's free.** §8's DPM-Solver++(2M) implementation
reuses the *previous* step's denoised estimate rather than calling the model twice per step to
get 2nd-order accuracy. Using the code and §6's discussion, explain in one or two sentences why
this doesn't compromise correctness compared to a "true" 2nd-order method that evaluates the
model twice per step.

<details markdown="1"><summary>Solution</summary>

A true single-step 2nd-order method (like Heun's) needs two model evaluations *within the same
step* because it has no other source of a second data point at a nearby noise level. But a
multistep solver, running many steps in sequence, has already computed the denoised estimate at
the *previous* step as a byproduct of normal operation — that stored value is itself a valid
data point near the current one (at an adjacent, already-visited noise level), so reusing it for
the 2nd-order correction costs nothing extra: it's "free" information that a single-step method
would have to purchase with a second network call. This is exactly why §8 notes "2nd-order
accuracy at 1st-order cost" — the trick doesn't reduce accuracy, it just relocates where the
second data point comes from (history instead of an extra forward pass).

</details>

## 10. Key takeaways

| # | Takeaway |
|---|---|
| 1 | The score $\nabla_x\log p(x)$ does not depend on the normalizing constant $Z$ — the whole reason this approach works. |
| 2 | Naive Langevin fails because the score is undefined off the data manifold; multi-scale noise fixes it. |
| 3 | $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$ — DDPM and score matching are the same model. |
| 4 | The reverse of any diffusion SDE needs only the score (Anderson's theorem). |
| 5 | Every SDE has a probability-flow ODE with identical marginals → deterministic sampling, exact likelihoods, invertibility. |
| 6 | DDIM is that ODE discretized, and it works on a *subsequence* of timesteps — 1000 → 20 steps free. |
| 7 | DPM-Solver exploits the ODE's semi-linear structure; the Karras $\sigma$ schedule is worth several steps by itself. |
| 8 | Stochastic samplers self-correct and can reach higher quality; deterministic ones are faster and reproducible. |
| 9 | Consistency distillation maps any trajectory point directly to the endpoint → 1–4 step generation. |

---

## Further reading

- Song & Ermon, [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1907.05600) (2019).
- Song et al., [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456) (2021) — the unification.
- Song et al., [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502) (DDIM, 2020).
- Karras et al., [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364) (EDM, 2022) — the best systematic study.
- Lu et al., [*DPM-Solver++*](https://arxiv.org/abs/2211.01095) (2022).
- Song et al., [*Consistency Models*](https://arxiv.org/abs/2303.01469) (2023).
- Anderson, *Reverse-time diffusion equation models* (1982) — the 40-year-old theorem it all rests on.

**Next** → [Latent diffusion & conditioning](03-latent-diffusion.md)
