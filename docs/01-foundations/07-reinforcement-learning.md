# Reinforcement Learning Basics

> **Summary**: Reinforcement learning (RL) trains a policy to take actions that maximize a reward
> signal, when you can score outcomes but cannot write down the correct answer. It is how LLMs are
> aligned with human preferences (RLHF) and how reasoning models are trained (RLVR). This page covers
> exactly the RL you need for that: the policy-gradient theorem derived from scratch, why baselines
> and advantages cut variance, actor-critic methods, PPO's clipped objective, and how text
> generation maps onto an RL problem.

**Prerequisites**: → [Probability & information theory](02-probability-and-information-theory.md), → [Optimization](05-optimization.md) · **Next**: Part II → [Autoregressive models](../02-classical-models/01-autoregressive-models.md) · **Used by**: → [Alignment](../04-large-language-models/05-alignment.md), → [Reasoning](../04-large-language-models/10-reasoning.md)

---

## 1. When supervised learning isn't enough

Supervised learning needs the right answer for every input. Often you don't have it:

| Situation | You can't write down... | ...but you can score |
|---|---|---|
| Chat assistant | the one ideal reply | which of two replies a person prefers |
| Math reasoning | the ideal chain of thought | whether the final answer is correct |
| Code generation | the one correct program | whether the tests pass |
| Game playing | the best move in every position | whether you won |

RL learns from **scores of outcomes** instead of **demonstrations of answers**. The price is that
the score says nothing about *which* of your many decisions caused it. Working that out, the
**credit assignment problem**, is what most of RL's machinery is for.

> [!TIP]
> **Intuition.** Supervised learning is a teacher who shows you the correct answer. RL is a
> teacher who only says "better" or "worse" after you've finished. You learn by trying things,
> keeping what earned praise, and dropping what didn't. It is slower and noisier, but it works
> even when nobody knows the correct answer, and it can find answers better than any demonstration.

---

## 2. The vocabulary

An agent interacts with an environment over time:

```
        ┌─────────── action aₜ ───────────┐
        │                                  ▼
   ┌─────────┐                       ┌─────────────┐
   │  AGENT  │                       │ ENVIRONMENT │
   │ policy π│◄── state sₜ₊₁, ───────│             │
   └─────────┘    reward rₜ₊₁        └─────────────┘
```

| Term | Symbol | Meaning |
|---|---|---|
| State | $s_t$ | what the agent observes at time $t$ |
| Action | $a_t$ | what it does |
| Reward | $r_t$ | a scalar score after acting |
| Policy | $\pi_\theta(a\mid s)$ | a probability distribution over actions: the thing we train |
| Trajectory | $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ | one full episode |
| Return | $G_t = \sum_{k\ge 0}\gamma^k r_{t+k}$ | total (discounted) future reward from time $t$ |
| Discount | $\gamma \in [0,1]$ | how much less a reward counts for each step it is delayed |
| Value | $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ | expected return from a state |
| Action value | $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$ | expected return after taking $a$ in $s$ |
| Advantage | $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | how much better $a$ is than the policy's average action |

**The objective** is to maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\big[G_0\big] = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t \gamma^t r_t\Big]$$

**Worked example: discounting.** Rewards $(0, 0, 0, 10)$ over four steps with $\gamma = 0.9$:

$$G_0 = 0 + 0.9(0) + 0.81(0) + 0.729(10) = 7.29$$

The same reward arriving one step earlier is worth $0.81 \times 10 = 8.1$. Discounting encodes a
preference for sooner rewards and keeps infinite sums finite. For LLM training, where an episode
is one response and the reward arrives at the end, $\gamma = 1$ is the usual choice.

---

## 3. The policy-gradient theorem

We want $\nabla_\theta J(\theta)$, but $J$ is an expectation over trajectories whose *distribution*
depends on $\theta$. The environment is a black box (you cannot differentiate a game engine or a
human rater), so the reparameterization trick is unavailable. The solution is the score-function
trick from → [Probability & information theory §10](02-probability-and-information-theory.md#10-monte-carlo-estimation-and-the-two-gradient-estimators).

**Derivation.** Let $p_\theta(\tau)$ be the probability of a trajectory and $R(\tau)$ its return:

$$
\begin{aligned}
\nabla_\theta J &= \nabla_\theta \int p_\theta(\tau) R(\tau)\,d\tau
= \int \nabla_\theta p_\theta(\tau)\,R(\tau)\,d\tau \\
&= \int p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,R(\tau)\,d\tau
\qquad\text{(using } \nabla p = p\,\nabla\log p\text{)}\\
&= \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta \log p_\theta(\tau)\,R(\tau)\big]
\end{aligned}
$$

Now expand the trajectory probability. It is a product of environment transitions (unknown, but
not depending on $\theta$) and policy choices:

$$p_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$$

$$\log p_\theta(\tau) = \log p(s_0) + \sum_t \log\pi_\theta(a_t\mid s_t) + \sum_t \log p(s_{t+1}\mid s_t,a_t)$$

Only the middle sum depends on $\theta$, so **the environment's dynamics drop out of the gradient**:

$$\boxed{\;\nabla_\theta J = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\;R(\tau)\Big]\;}$$

> [!TIP]
> **Read the formula.** $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ is the direction that makes
> action $a_t$ more likely. The formula says: take every action you actually took, push its
> probability up, and weight the push by how good the outcome was. Good trajectories get
> reinforced and bad ones suppressed. You never need to know *how* the environment works, only
> what it returned.

**Causality refinement.** An action at time $t$ cannot affect rewards that came before it, so
replace $R(\tau)$ with the return from $t$ onward. This drops terms that are zero in expectation
but add noise:

$$\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big]$$

This estimator, used with Monte Carlo samples, is **REINFORCE** (Williams, 1992).

---

## 4. Variance, and why baselines fix it

REINFORCE is unbiased but extremely noisy. The problem is visible in a simple case: if every
return is positive (say rewards between 90 and 100), *every* action gets pushed up, and the
algorithm can only learn from the small differences between large numbers.

**The fix: subtract a baseline** $b(s_t)$ that doesn't depend on the action:

$$\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big]$$

**Proof that this adds no bias.** The extra term has expectation zero:

$$\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big]
= b(s)\sum_a \pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
= b(s)\,\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1} = 0$$

The expected gradient is unchanged, but the variance can drop enormously. The natural choice is
$b(s) = V(s)$, which turns $G_t - b(s_t)$ into an estimate of the **advantage**: was this action
better or worse than what the policy usually does here?

**Worked example: a two-armed bandit.** One state, two actions. Arm A pays 10, arm B pays 11.
The policy currently picks each with probability 0.5, using logits $z_A = z_B = 0$.

For a softmax policy, $\frac{\partial \log\pi(a)}{\partial z_j} = \mathbb{1}[a=j] - \pi(j)$.

| Sample | Return | Gradient on $z_B$ (no baseline) | With baseline $b = 10.5$ |
|---|---|---|---|
| pulled A | 10 | $(0 - 0.5)\times 10 = -5.0$ | $(0-0.5)\times(-0.5) = +0.25$ |
| pulled B | 11 | $(1 - 0.5)\times 11 = +5.5$ | $(1-0.5)\times(+0.5) = +0.25$ |
| **mean** | | **+0.25** | **+0.25** |
| **spread between samples** | | **10.5** | **0** |

Both estimators agree on average (push toward B by 0.25), which is exactly the unbiasedness
proof above. Without a baseline, individual samples swing between −5.0 and +5.5, so you need many
of them to see the signal. With the baseline, every sample points the same way.

---

## 5. Actor-critic methods

Where does $V(s)$ come from? Learn it. An **actor-critic** method trains two things:

| Component | Learns | Loss |
|---|---|---|
| **Actor** $\pi_\theta$ | the policy | policy gradient weighted by advantage |
| **Critic** $V_\psi$ | the value of each state | regression: $\big(V_\psi(s_t) - \hat G_t\big)^2$ |

The critic also enables **bootstrapping**: rather than waiting for the full return, estimate it
from one real reward plus the critic's guess about what follows. The one-step **TD error** is

$$\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

and it is itself an advantage estimate.

**The bias–variance dial.** Monte Carlo returns are unbiased but noisy; TD errors are low-variance
but biased by whatever the critic gets wrong. **Generalized Advantage Estimation** (GAE) blends
them with a parameter $\lambda$:

$$\hat A_t^{\text{GAE}} = \sum_{k\ge 0}(\gamma\lambda)^k\,\delta_{t+k}$$

| $\lambda$ | Behaviour |
|---|---|
| 0 | pure one-step TD: lowest variance, most bias |
| 1 | Monte Carlo return minus baseline: unbiased, highest variance |
| 0.95 | the common default |

---

## 6. PPO: small, safe steps

Policy gradients have a nasty failure mode. One overly large update can wreck the policy, and then
the data it collects next is bad, so it never recovers. **Trust-region** methods limit how far each
update can move. PPO (Schulman et al., 2017) does it with a simple clip.

Let $\rho_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$,
the ratio between the new and the data-collecting policy. PPO maximizes

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\;\operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big], \qquad \epsilon \approx 0.2$$

**Worked example: what the clip does.** With $\epsilon = 0.2$:

| Advantage | Ratio $\rho$ | Unclipped $\rho A$ | Clipped | Objective (the min) | Effect |
|---|---|---|---|---|---|
| $+2$ | 1.1 | 2.2 | 2.2 | 2.2 | still rewarded for raising the probability |
| $+2$ | 1.5 | 3.0 | 2.4 | **2.4** | capped: no gain from pushing past 1.2× |
| $-2$ | 0.9 | −1.8 | −1.8 | −1.8 | still rewarded for lowering it |
| $-2$ | 0.5 | −1.0 | −1.6 | **−1.6** | capped: no gain from pushing below 0.8× |

> [!TIP]
> **Intuition.** The clip removes the *incentive* to move any single action's probability more
> than 20% in one update. It isn't a hard constraint, but it makes big jumps pointless, so you can
> safely reuse one batch of data for several gradient steps. That data reuse is PPO's main
> practical win over plain policy gradients.

> [!WARNING]
> **The four models of RLHF-PPO.** Applied to LLMs, PPO needs the policy, a frozen reference
> model (for the KL penalty), a reward model, and a critic, all in memory at once. That cost is
> why GRPO (which drops the critic) and DPO (which drops RL entirely) became popular. See
> → [Alignment](../04-large-language-models/05-alignment.md).

---

## 7. Text generation as an RL problem

The mapping is direct:

| RL concept | For an LLM |
|---|---|
| State $s_t$ | the prompt plus the tokens generated so far |
| Action $a_t$ | the next token |
| Policy $\pi_\theta(a_t\mid s_t)$ | the model's next-token distribution |
| Transition | deterministic: append the token |
| Episode | one full response |
| Reward | usually a single score at the end: a reward model's rating, or 1 if the answer is correct |

Three features make LLM RL unusual:

1. **The action space is huge**: about 100,000 tokens per step.
2. **Rewards are sparse**: one number for a response hundreds of tokens long. Every token shares
   the credit, which is why advantage estimation matters so much.
3. **The policy starts out good.** Unlike a game agent that begins random, a pretrained model
   already produces sensible text. RL is refining a strong prior, not searching from scratch. The
   KL penalty toward the reference model protects that prior.

**GRPO, in these terms.** Instead of a learned critic, sample a *group* of $G$ responses to the same
prompt and use their mean reward as the baseline:

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}$$

This is §4's baseline idea with an empirical estimate of $V(s)$: the average reward of other
attempts at the same prompt. It needs no critic network and fits verifiable rewards naturally.

**Worked example.** Eight attempts at a math problem, three correct (reward 1), five wrong (reward
0). Mean 0.375, standard deviation 0.484. Correct answers get advantage $(1-0.375)/0.484 = +1.29$;
wrong ones get $(0-0.375)/0.484 = -0.77$. If all eight were correct (or all wrong), every
advantage would be 0: the group teaches nothing, because there is no contrast to learn from.

---

## 8. Implementation

REINFORCE with a baseline on the two-armed bandit from §4, small enough to read in full:

```python
import numpy as np

rng = np.random.default_rng(0)
PAYOUT = np.array([10.0, 11.0])      # mean reward of arm A and arm B
NOISE = 2.0                          # rewards are noisy
z = np.zeros(2)                      # policy logits
baseline, lr, beta = 0.0, 0.05, 0.05

def policy(z):
    p = np.exp(z - z.max())
    return p / p.sum()

for step in range(2000):
    p = policy(z)
    a = rng.choice(2, p=p)                           # act
    r = PAYOUT[a] + NOISE * rng.standard_normal()    # observe reward
    adv = r - baseline                               # advantage estimate
    grad_logp = -p; grad_logp[a] += 1.0              # d log pi(a) / dz
    z += lr * adv * grad_logp                        # policy-gradient ascent
    baseline += beta * (r - baseline)                # running-average baseline

print(policy(z))   # most probability ends up on arm B
```

Set `baseline` to stay at 0 and compare: learning becomes much slower and less stable, which is
§4's table in action.

> [!WARNING]
> **Sign conventions.** Policy gradients *ascend* the objective. When you implement this as a
> loss for an optimizer that *minimizes*, use `loss = -(logp * advantage).mean()`, and make sure
> the advantage is treated as a constant (`.detach()` in PyTorch). Getting either wrong silently
> trains the policy to do worse.

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | RL learns from scores of outcomes, not demonstrations. The cost is credit assignment. |
| 2 | Policy gradient: $\mathbb{E}[\sum_t \nabla\log\pi(a_t\mid s_t)\,G_t]$. The environment's dynamics drop out, so it works with black boxes. |
| 3 | Subtracting a baseline leaves the gradient unbiased and can cut the variance enormously. |
| 4 | With the value function as the baseline, you are weighting by the advantage: better or worse than usual. |
| 5 | Actor-critic learns the baseline; GAE's $\lambda$ trades bias against variance. |
| 6 | PPO's clip removes the incentive for big policy jumps, which makes data reuse safe. |
| 7 | For LLMs: state = context, action = token, reward usually arrives only at the end. |
| 8 | GRPO uses the group's mean reward as the baseline. Groups where every answer scores the same teach nothing. |

---

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018): the standard textbook, free online.
- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (1992): REINFORCE.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Shao et al., [*DeepSeekMath*](https://arxiv.org/abs/2511.22570) (2024): introduces GRPO.
- OpenAI, *Spinning Up in Deep RL*: a practical, well-written tutorial.

**Next** → Part II: [Autoregressive models](../02-classical-models/01-autoregressive-models.md)
