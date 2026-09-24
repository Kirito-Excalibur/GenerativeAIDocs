# Latent Diffusion and Conditioning

> **Summary** — Diffusing directly in pixel space is wasteful: most pixels are perceptually
> redundant. Latent diffusion compresses images 8× per side with a VAE (64× fewer spatial
> positions, 48× fewer values), runs diffusion in that small space, and decodes at the end. Combined with cross-attention for text conditioning and
> classifier-free guidance for prompt adherence, this is the architecture behind essentially every
> text-to-image system. This page also covers ControlNet, inpainting, and image editing.

**Prerequisites**: → [Diffusion models](01-diffusion-models.md), → [VAE](../02-classical-models/02-vae.md) · **Next**: → [Flow matching](04-flow-matching.md)

---

## 1. The compression argument

A $512\times512\times3$ image has 786,432 dimensions. Running 50 denoising steps of a U-Net over
that is enormously expensive — and most of the work is spent on imperceptible high-frequency
detail.

**Latent diffusion**: compress first.

```
   PIXEL DIFFUSION                    LATENT DIFFUSION

   512×512×3 = 786,432 dims           512×512×3
        │                                  │
        ▼                             ┌────▼────┐
   ┌──────────┐                       │ VAE enc │  (trained once, then frozen)
   │ 50 steps │                       └────┬────┘
   │ of U-Net │                            ▼
   │ at FULL  │                      64×64×4 = 16,384 dims   ← 48× smaller
   │ res      │                            │
   └──────────┘                       ┌────▼────┐
        │                             │ 50 steps│  ← each step ~48× cheaper
        ▼                             │ of U-Net│
      image                           └────┬────┘
                                           ▼
                                      ┌────────┐
                                      │VAE dec │
                                      └────┬───┘
                                           ▼
                                         image
```

**The saving**: $786{,}432 / 16{,}384 = 48\times$ fewer dimensions. Attention cost, which is
quadratic in the number of spatial positions, drops even more: $(64\times64)^2$ vs
$(512\times512)^2$ is a factor of 4096.

This is why Stable Diffusion runs on a consumer GPU while pixel-space models of comparable
quality needed a datacenter.

### Why this doesn't destroy quality

> [!TIP]
> **The perceptual compression / semantic compression split.** Rombach et al.'s framing: image
> information divides into
> 1. **perceptual detail** — high-frequency texture, sensor noise, exact pixel values. High entropy,
>    low semantic content. A VAE learns this cheaply and reconstructs it convincingly.
> 2. **semantic content** — what objects are present, their layout, style. Low-dimensional, and this
>    is the hard part.

Let the VAE handle (1) and the diffusion model spend all its capacity on (2). It is a division of
labour matched to the structure of the problem.

**The VAE is not a plain VAE.** The latent-diffusion autoencoder is trained with:
- an L1/L2 reconstruction loss,
- a **perceptual (LPIPS)** loss,
- a **patch-based adversarial** loss,
- a *very* small KL weight (or vector quantization).

> [!WARNING]
> The tiny KL weight is deliberate: the goal is a good *compressor*, not a samplable prior — a
> diffusion model will learn the latent distribution anyway. This is precisely the fix for the
> VAE's prior/aggregate-posterior gap discussed in
> → [VAE §5](../02-classical-models/02-vae.md#5-the-two-classic-failure-modes).

**Downsampling factor** is the key hyperparameter:

| $f$ | Latent size (from 512²) | Trade-off |
|---|---|---|
| 4 | 128×128 | better detail, slower |
| **8** | **64×64** | **the standard** — best quality/compute balance |
| 16 | 32×32 | fast, visible reconstruction artifacts |
| 32 | 16×16 | too lossy |

> [!WARNING]
> **The VAE is a hard ceiling on quality.** Whatever the autoencoder cannot reconstruct, the
> diffusion model can never produce. SD 1.x's 4-channel VAE notoriously struggled with small faces
> and legible text; SDXL improved the VAE, and SD3/Flux moved to 16 channels — a direct quality
> improvement with no change to the diffusion model at all.

---

## 2. Text conditioning via cross-attention

```
   "a photo of an astronaut riding a horse"
                    │
            ┌───────▼────────┐
            │  Text encoder  │   CLIP text / T5-XXL — FROZEN
            └───────┬────────┘
                    ▼
            (77, 768) token embeddings
                    │
    ┌───────────────┴──────────────────────────┐
    │  injected as K, V into EVERY U-Net block │
    └──────────────────────────────────────────┘

   inside a U-Net block:
      image latent ──► Q  ┐
                          ├──► cross-attention ──► modulated latent
      text embeddings ──► K, V ┘

   "which words should this image region attend to?"
```

**The text encoder choice matters enormously:**

| Model | Text encoder | Effect |
|---|---|---|
| SD 1.5 | CLIP ViT-L/14 | weak compositionality; ignores word order |
| SDXL | CLIP ViT-L + OpenCLIP ViT-bigG | better, two encoders concatenated |
| Imagen | **T5-XXL** (4.6 B) | much better prompt adherence |
| SD3 / Flux | CLIP ×2 **+ T5-XXL** | best; handles text rendering and complex prompts |

> [!TIP]
> **Why T5 beats CLIP for conditioning.** CLIP's text encoder was trained to match images at the
> *whole-caption* level, with a contrastive objective. It learns "which bag of concepts is in this
> image" and is famously weak at compositional structure — "a red cube on a blue sphere" and "a blue
> cube on a red sphere" get nearly identical CLIP embeddings. T5 was trained on language modelling,
> so it actually represents syntax and binding. Imagen's central finding was that **scaling the text
> encoder helps more than scaling the diffusion model.**

---

## 3. Classifier-free guidance

**Start from Bayes.** To sample from $p(x\mid c)$ instead of $p(x)$:

$$\nabla_x\log p(x\mid c) = \nabla_x\log p(x) + \nabla_x\log p(c\mid x)$$

**Classifier guidance** (Dhariwal & Nichol) trains a separate classifier on noisy images and uses
its gradient. It works, but requires training and maintaining a noise-aware classifier.

**Classifier-free guidance** (Ho & Salimans) eliminates the classifier. Rearranging Bayes:

$$\nabla_x\log p(c\mid x) = \nabla_x\log p(x\mid c) - \nabla_x\log p(x)$$

Substitute back and scale by a guidance weight $w$:

$$\boxed{\;\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t,\varnothing) + w\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t,\varnothing)\big)\;}$$

**The trick in training**: randomly drop the conditioning (replace $c$ with a null embedding)
~10% of the time. The *same* network then learns both the conditional and unconditional score.

```
   ε(∅)  unconditional   ────────────►  generic image
                                 ╲
                                  ╲ the difference is the
                                   ╲ "direction of the prompt"
   ε(c)  conditional     ──────────►  prompt-matching image
                                     ╱
   extrapolate past it: ε(∅) + w·(ε(c) − ε(∅))
                                          ╱
                                    ────►  exaggeratedly prompt-matching
```

> [!TIP]
> **Guidance is extrapolation, not interpolation.** With $w > 1$ you push *beyond* the conditional
> prediction, amplifying whatever makes the conditional differ from the unconditional. This
> sharpens prompt adherence and increases saturation and contrast.

**Guidance scale effects:**

| $w$ | Behaviour |
|---|---|
| 0 | unconditional — ignores the prompt |
| 1 | the true conditional — surprisingly *weak* prompt adherence |
| **7–8** | **the standard for SD-family models** |
| 12–20 | strong adherence, oversaturated, loses diversity |
| >20 | artifacts, blown-out colours |

> [!WARNING]
> **CFG doubles inference cost** — every step requires two forward passes (conditional and
> unconditional). Implementations batch them together. Distilled models (LCM, Flux-schnell) bake the
> guidance in and skip this.

> [!WARNING]
> **CFG trades diversity for fidelity.** High $w$ collapses the output distribution toward the
> "most prototypical" image for the prompt. This is a real, measurable mode-seeking effect — the same
> phenomenon as reverse-KL collapse in
> → [alignment](../04-large-language-models/05-alignment.md). If your outputs all look the same,
> lower the guidance.

**Refinements**: *dynamic thresholding* (Imagen) rescales the prediction to prevent saturation at
high $w$; *guidance intervals* apply CFG only in the middle timesteps, where it helps most;
*CFG-Zero / rescaling* corrects the variance inflation that CFG introduces.

---

## 4. ControlNet and spatial conditioning

Text says *what*. ControlNet says *where*.

```
   ┌──────────────────────────────────────┐
   │  FROZEN U-Net (the base model)       │
   │   in ──► block1 ──► block2 ──► out   │
   └──────────▲──────────▲────────────────┘
              │ (+)      │ (+)         ← outputs added to the frozen path
     ┌────────┴──────────┴────────┐
     │  TRAINABLE COPY of the     │
     │  encoder blocks            │
     │  + zero-initialized convs  │
     └────────────▲───────────────┘
                  │
          control image (depth map, pose skeleton,
                         Canny edges, segmentation)
```

> [!TIP]
> **The two design choices that make ControlNet work:**

1. **The base model is frozen.** You cannot damage it. The ControlNet is a pure addition.
2. **Zero-initialized convolutions.** At step 0 the ControlNet contributes exactly nothing, so
   training starts from the unmodified base model's behaviour and the control signal is introduced
   gradually. Without this, the random initial output would destroy the base model's predictions
   before it could learn anything useful.

The ControlNet paper reports that training is robust with small datasets (under 50k pairs) as
well as large ones, and feasible on a single consumer GPU — remarkably cheap for the capability it
adds ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543)).

| Control type | Use |
|---|---|
| Canny edges | preserve composition, restyle |
| Depth map | preserve 3-D layout |
| **OpenPose skeleton** | control human pose exactly |
| Segmentation map | control the layout of regions |
| Scribble | rough sketch → finished image |
| Normal map | control surface geometry |

**Related adapters**: T2I-Adapter (lighter, similar idea), IP-Adapter (condition on a *reference
image's style or subject* via an image encoder + decoupled cross-attention).

---

## 5. Editing and inpainting

### Inpainting: the no-training approach

> [!TIP]
> At every denoising step, replace the *known* region with a correctly-noised version of the
> original image, and let the model fill in the rest:

```python
for t in reversed(range(T)):
    x = denoise_step(model, x, t)
    # the known region: re-noise the original to level t and paste it in
    x_known = sqrt_alpha_bar[t] * x_orig + sqrt_one_minus_alpha_bar[t] * torch.randn_like(x)
    x = mask * x + (1 - mask) * x_known         # mask = 1 where we're generating
```

- ✅ **Works with any pretrained diffusion model.** No retraining.
- ⚠️ Boundaries can be inconsistent, because the generated region sees the known region only through
  the network's receptive field at each step. Dedicated inpainting models (trained with masked
  inputs as an extra conditioning channel) do better.

### SDEdit (img2img)

Add noise to an existing image up to timestep $t_0 < T$, then denoise from there.

```
   strength = t₀/T

   0.2 │  minor changes, structure preserved
   0.5 │  significant restyling, layout kept
   0.8 │  loosely inspired by the original
   1.0 │  ignores the input entirely
```

> [!TIP]
> The intuition is clean: partial noising destroys high-frequency detail first and global
> structure last, so the "strength" parameter literally selects *which scale of information* to
> preserve.

### Other editing methods

| Method | Idea |
|---|---|
| **DDIM inversion** | run the deterministic ODE *backwards* to find the noise that produces a given image; edit the prompt; re-generate |
| **Prompt-to-Prompt** | inject the original's cross-attention maps into the edited generation, preserving layout |
| **InstructPix2Pix** | train on (image, instruction, edited image) triples → direct instruction following |
| **Null-text inversion** | optimize the unconditional embedding so that inversion is exact under CFG |
| **LoRA / DreamBooth** | fine-tune on a few images of a subject to inject a new concept |

> [!TIP]
> **DDIM inversion is only possible because of the probability-flow ODE** — a deterministic,
> invertible map from noise to image. Stochastic samplers cannot be inverted.
> → [Score-based models §5](02-score-based-models.md#5-the-continuous-time-sde-view)

---

## 6. Personalization

| Method | Trains | Data needed | Output size |
|---|---|---|---|
| **Textual Inversion** | one new token embedding | 3–5 images | ~10 KB |
| **DreamBooth** | the whole U-Net (+ a prior-preservation loss) | 3–20 images | full model |
| **LoRA** | low-rank updates to attention | 10–50 images | 10–200 MB |
| **IP-Adapter** | a decoupled image cross-attention module | 0 (zero-shot at inference) | ~50 MB, reusable |

> [!WARNING]
> **DreamBooth's language drift problem**: fine-tuning on 5 photos of your dog with the prompt "a
> photo of [V] dog" causes the model to associate *all* dogs with your dog. The fix is the
> **prior-preservation loss** — simultaneously train on model-generated images of generic dogs to
> anchor the original concept.

**LoRA is the practical default**: small files, composable (stack a style LoRA with a character
LoRA), and fast to train. The image-generation community's adapter ecosystem is built on it.
→ [Fine-tuning & PEFT](../04-large-language-models/04-finetuning-peft.md)

---

## 7. Implementation

A complete text-to-image loop, showing every piece:

```python
import torch

@torch.no_grad()
def txt2img(unet, vae, text_encoder, tokenizer, scheduler, prompt,
            negative_prompt="", steps=30, guidance=7.5, height=512, width=512, seed=None):
    device = unet.device
    g = torch.Generator(device).manual_seed(seed) if seed is not None else None

    # ---- 1. Encode the prompt (and the negative / unconditional prompt) ----
    def encode(text):
        ids = tokenizer(text, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        return text_encoder(ids)[0]

    cond   = encode(prompt)
    uncond = encode(negative_prompt)          # "" gives the unconditional embedding
    emb    = torch.cat([uncond, cond])        # batch both for one forward pass

    # ---- 2. Start from noise in LATENT space (8x smaller than the image) ----
    latents = torch.randn((1, 4, height // 8, width // 8), generator=g, device=device)
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma

    # ---- 3. Denoising loop with classifier-free guidance ----
    for t in scheduler.timesteps:
        inp = torch.cat([latents] * 2)                      # duplicate for cond + uncond
        inp = scheduler.scale_model_input(inp, t)
        noise_uncond, noise_cond = unet(inp, t, encoder_hidden_states=emb).sample.chunk(2)
        # CFG: extrapolate from unconditional toward conditional
        noise = noise_uncond + guidance * (noise_cond - noise_uncond)
        latents = scheduler.step(noise, t, latents).prev_sample

    # ---- 4. Decode latents to pixels ----
    image = vae.decode(latents / vae.config.scaling_factor).sample
    return ((image / 2 + 0.5).clamp(0, 1) * 255).byte()
```

> [!WARNING]
> **The `scaling_factor` (0.18215 for SD 1.x)** is easy to forget and produces washed-out or
> blown-out images when wrong. It normalizes the VAE latents to roughly unit variance, which the
> diffusion model expects. Every latent-diffusion model has its own value.

> [!TIP]
> **Negative prompts are free** — you were computing the unconditional branch anyway, so replacing
> $\varnothing$ with "blurry, low quality, watermark" costs nothing and steers *away* from those
> concepts. This is a nice example of a capability that falls out of the CFG mechanism rather than
> being designed in.

---

## 8. Exercises

**Problem 1 — compression, a different downsample factor.** Using §1's method, compute the
spatial and total compression for a $512\times512$ RGB image compressed with downsample factor
$f{=}4$ into a 16-channel latent (rather than §1's $f{=}8$, 4-channel example). Is this more or
less aggressive compression than the standard $f{=}8$ setup?

<details><summary>Solution</summary>

Spatial compression: $f^2=16\times$ (vs $f{=}8$'s $64\times$).

Pixel dims: $512\times512\times3=786{,}432$. Latent dims:
$(512/4)\times(512/4)\times16=128\times128\times16=262{,}144$.

Total compression: $786432/262144=3.0\times$ — **far less aggressive** than the $f{=}8$,
4-channel setup's 48× reduction (§1). Even though $f{=}4$ vs $f{=}8$ only halves the *linear*
downsample, spatial compression is quartered ($16\times$ vs $64\times$, since it's $f^2$); and
using 16 channels instead of 4 partially offsets that (more channels per spatial location means
less compression per location), compounding into a much smaller overall ratio. This is a
worked illustration of §1's downsample-factor table: $f{=}4$ trades "better quality, slower" for
giving up most of the compute savings that make latent diffusion fast in the first place.

</details>

**Problem 2 — classifier-free guidance, a numeric case.** Suppose at some pixel/channel,
$\epsilon_\theta(x_t,\varnothing)=0.2$ (unconditional prediction) and $\epsilon_\theta(x_t,c)=0.5$
(conditional, given the prompt). Compute the guided noise prediction at $w=1$, $w=7$, and $w=15$
using §3's formula. At which $w$ does the prediction start to look like *extrapolation beyond*
the conditional prediction itself (i.e., further from $\varnothing$ than $c$ is)?

<details><summary>Solution</summary>

$\tilde\epsilon = \epsilon(\varnothing)+w(\epsilon(c)-\epsilon(\varnothing)) = 0.2+w(0.3)$.

$w{=}1$: $0.2+0.3=0.5$ — exactly recovers $\epsilon(c)$ (the true conditional prediction),
matching §3's note that "$w{=}1$: the true conditional."

$w{=}7$: $0.2+7(0.3)=0.2+2.1=2.3$.

$w{=}15$: $0.2+15(0.3)=0.2+4.5=4.7$.

Already at $w{=}7$ (§3's stated typical value for SD-family models), $2.3$ is well beyond
$\epsilon(c){=}0.5$ — this is **extrapolation past the conditional prediction itself**, starting
immediately for any $w>1$, not just at large $w$. This matches §3's framing precisely: "guidance
is extrapolation, not interpolation" — even the "standard" $w{\approx}7$ setting is already
pushing the prediction several multiples past where the model's honest conditional estimate
sits, which is exactly the mechanism behind CFG's oversaturation and reduced-diversity effects
at high $w$.

</details>

**Problem 3 — ControlNet's zero-init, why it matters mechanically.** §4 states ControlNet's
added connections are zero-initialized so training "starts from the unmodified base model's
behaviour." Using the general zero-init reasoning (also seen in RL's advantage-baseline problem
and elsewhere in this wiki), explain specifically what would go wrong at step 0 of training if the
zero-init convolutions were instead randomly initialized like normal layers.

<details><summary>Solution</summary>

At step 0, a randomly-initialized added connection contributes a *random, untrained* signal to
the frozen base model's forward pass — effectively corrupting the base model's carefully
pretrained representations with noise from day one, before the ControlNet branch has learned
anything useful about the control signal (depth map, pose, etc.). Per §4's explanation, this
would "destroy the base model's predictions before it could learn anything useful": early
training would have to first *undo* the damage from the random injection before it could start
extracting a useful gradient signal about the actual control task, likely making training slower,
less stable, and at risk of permanently degrading the base model's quality if the corruption is
severe enough early on. Zero-initializing sidesteps this entirely: at step 0 the ControlNet branch
contributes exactly nothing (mathematically identical to the unmodified base model), so training
can smoothly *introduce* the control signal's influence rather than having to *repair* damage
from an uninformed random one.

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Latent diffusion compresses 8× spatially (48× in dimension) before diffusing — the reason SD runs on a laptop. |
| 2 | Split the problem: the VAE handles perceptual detail, the diffusion model handles semantics. |
| 3 | The VAE is a hard quality ceiling — more latent channels (4 → 16) directly improves faces and text. |
| 4 | Cross-attention injects text into every U-Net block; the text encoder's quality dominates prompt adherence. |
| 5 | T5 beats CLIP for conditioning because it actually represents compositional structure. |
| 6 | CFG: $\epsilon(\varnothing) + w(\epsilon(c)-\epsilon(\varnothing))$, trained by dropping conditioning 10% of the time. |
| 7 | CFG is extrapolation — it trades diversity for fidelity. $w \approx 7$; higher means samey, saturated output. |
| 8 | ControlNet works because the base is frozen and the connections are zero-initialized. |
| 9 | Inpainting needs no retraining: re-noise the known region and paste it in at every step. |
| 10 | DDIM inversion (and hence prompt-based editing) requires the deterministic ODE sampler. |

---

## Further reading

- Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) (2022) — Stable Diffusion.
- Ho & Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598) (2022).
- Dhariwal & Nichol, [*Diffusion Models Beat GANs on Image Synthesis*](https://arxiv.org/abs/2105.05233) (2021) — classifier guidance.
- Saharia et al., [*Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*](https://arxiv.org/abs/2205.11487) (Imagen, 2022).
- Zhang et al., [*Adding Conditional Control to Text-to-Image Diffusion Models*](https://arxiv.org/abs/2302.05543) (ControlNet, 2023).
- Meng et al., [*SDEdit*](https://arxiv.org/abs/2108.01073) (2021); Hertz et al., [*Prompt-to-Prompt*](https://arxiv.org/abs/2208.01626) (2022).
- Ruiz et al., [*DreamBooth*](https://arxiv.org/abs/2208.12242) (2022); Ye et al., [*IP-Adapter*](https://arxiv.org/abs/2308.06721) (2023).
- Esser et al., [*Scaling Rectified Flow Transformers*](https://arxiv.org/abs/2403.03206) (SD3, 2024).

**Next** → [Flow matching & rectified flow](04-flow-matching.md)
