# Latent Diffusion & Conditioning

> **Summary** — Diffusing directly in pixel space is wasteful: most pixels are perceptually
> redundant. Latent diffusion compresses images 8× with a VAE, runs diffusion in that 64×-smaller
> space, and decodes at the end. Combined with cross-attention for text conditioning and
> classifier-free guidance for prompt adherence, this is the architecture behind essentially every
> text-to-image system. This page also covers ControlNet, inpainting, and image editing.

**Prerequisites**: → [Diffusion models](01-diffusion-models.md), → [VAE](../02-classical-models/02-vae.md) · **Next**: → [Flow matching](04-flow-matching.md)

---

## 1. The compression argument

🔢 A $512\times512\times3$ image has 786,432 dimensions. Running 50 denoising steps of a U-Net over
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

🔢 **The saving**: $786{,}432 / 16{,}384 = 48\times$ fewer dimensions. Attention cost, which is
quadratic in the number of spatial positions, drops even more: $(64\times64)^2$ vs
$(512\times512)^2$ is a factor of 4096.

📊 This is why Stable Diffusion runs on a consumer GPU while pixel-space models of comparable
quality needed a datacenter.

### Why this doesn't destroy quality

🧠 **The perceptual compression / semantic compression split.** Rombach et al.'s framing: image
information divides into
1. **perceptual detail** — high-frequency texture, sensor noise, exact pixel values. High entropy,
   low semantic content. A VAE learns this cheaply and reconstructs it convincingly.
2. **semantic content** — what objects are present, their layout, style. Low-dimensional, and this
   is the hard part.

Let the VAE handle (1) and the diffusion model spend all its capacity on (2). It is a division of
labour matched to the structure of the problem.

📊 **The VAE is not a plain VAE.** The latent-diffusion autoencoder is trained with:
- an L1/L2 reconstruction loss,
- a **perceptual (LPIPS)** loss,
- a **patch-based adversarial** loss,
- a *very* small KL weight (or vector quantization).

⚠️ The tiny KL weight is deliberate: the goal is a good *compressor*, not a samplable prior — a
diffusion model will learn the latent distribution anyway. This is precisely the fix for the
VAE's prior/aggregate-posterior gap discussed in
→ [VAE §5](../02-classical-models/02-vae.md#5-the-two-classic-failure-modes).

🔢 **Downsampling factor** is the key hyperparameter:

| $f$ | Latent size (from 512²) | Trade-off |
|---|---|---|
| 4 | 128×128 | better detail, slower |
| **8** | **64×64** | **the standard** — best quality/compute balance |
| 16 | 32×32 | fast, visible reconstruction artifacts |
| 32 | 16×16 | too lossy |

⚠️ **The VAE is a hard ceiling on quality.** Whatever the autoencoder cannot reconstruct, the
diffusion model can never produce. SD 1.x's 4-channel VAE notoriously struggled with small faces
and legible text; SDXL improved the VAE, and SD3/Flux moved to 16 channels — a direct quality
improvement with no change to the diffusion model at all.

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

📊 **The text encoder choice matters enormously:**

| Model | Text encoder | Effect |
|---|---|---|
| SD 1.5 | CLIP ViT-L/14 | weak compositionality; ignores word order |
| SDXL | CLIP ViT-L + OpenCLIP ViT-bigG | better, two encoders concatenated |
| Imagen | **T5-XXL** (4.6 B) | much better prompt adherence |
| SD3 / Flux | CLIP ×2 **+ T5-XXL** | best; handles text rendering and complex prompts |

🧠 **Why T5 beats CLIP for conditioning.** CLIP's text encoder was trained to match images at the
*whole-caption* level, with a contrastive objective. It learns "which bag of concepts is in this
image" and is famously weak at compositional structure — "a red cube on a blue sphere" and "a blue
cube on a red sphere" get nearly identical CLIP embeddings. T5 was trained on language modelling,
so it actually represents syntax and binding. Imagen's central finding was that **scaling the text
encoder helps more than scaling the diffusion model.**

---

## 3. Classifier-free guidance

📐 **Start from Bayes.** To sample from $p(x\mid c)$ instead of $p(x)$:

$$\nabla_x\log p(x\mid c) = \nabla_x\log p(x) + \nabla_x\log p(c\mid x)$$

**Classifier guidance** (Dhariwal & Nichol) trains a separate classifier on noisy images and uses
its gradient. It works, but requires training and maintaining a noise-aware classifier.

📐 **Classifier-free guidance** (Ho & Salimans) eliminates the classifier. Rearranging Bayes:

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

🧠 **Guidance is extrapolation, not interpolation.** With $w > 1$ you push *beyond* the conditional
prediction, amplifying whatever makes the conditional differ from the unconditional. This
sharpens prompt adherence and increases saturation and contrast.

📊 **Guidance scale effects:**

| $w$ | Behaviour |
|---|---|
| 0 | unconditional — ignores the prompt |
| 1 | the true conditional — surprisingly *weak* prompt adherence |
| **7–8** | **the standard for SD-family models** |
| 12–20 | strong adherence, oversaturated, loses diversity |
| >20 | artifacts, blown-out colours |

⚠️ **CFG doubles inference cost** — every step requires two forward passes (conditional and
unconditional). Implementations batch them together. Distilled models (LCM, Flux-schnell) bake the
guidance in and skip this.

⚠️ **CFG trades diversity for fidelity.** High $w$ collapses the output distribution toward the
"most prototypical" image for the prompt. This is a real, measurable mode-seeking effect — the same
phenomenon as reverse-KL collapse in
→ [alignment](../04-large-language-models/05-alignment.md). If your outputs all look the same,
lower the guidance.

📊 **Refinements**: *dynamic thresholding* (Imagen) rescales the prediction to prevent saturation at
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

🧠 **The two design choices that make ControlNet work:**

1. **The base model is frozen.** You cannot damage it. The ControlNet is a pure addition.
2. **Zero-initialized convolutions.** At step 0 the ControlNet contributes exactly nothing, so
   training starts from the unmodified base model's behaviour and the control signal is introduced
   gradually. Without this, the random initial output would destroy the base model's predictions
   before it could learn anything useful.

📊 ControlNet trains on as few as ~50k image/control pairs and takes a day on one GPU — remarkably
cheap for the capability it adds.

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

🧠 At every denoising step, replace the *known* region with a correctly-noised version of the
original image, and let the model fill in the rest:

```python
for t in reversed(range(T)):
    x = denoise_step(model, x, t)
    # the known region: re-noise the original to level t and paste it in
    x_known = sqrt_alpha_bar[t] * x_orig + sqrt_one_minus_alpha_bar[t] * torch.randn_like(x)
    x = mask * x + (1 - mask) * x_known         # mask = 1 where we're generating
```

✅ **Works with any pretrained diffusion model.** No retraining.
⚠️ Boundaries can be inconsistent, because the generated region sees the known region only through
the network's receptive field at each step. Dedicated inpainting models (trained with masked
inputs as an extra conditioning channel) do better.

### SDEdit / img2img

Add noise to an existing image up to timestep $t_0 < T$, then denoise from there.

```
   strength = t₀/T

   0.2 │  minor changes, structure preserved
   0.5 │  significant restyling, layout kept
   0.8 │  loosely inspired by the original
   1.0 │  ignores the input entirely
```

🧠 The intuition is clean: partial noising destroys high-frequency detail first and global
structure last, so the "strength" parameter literally selects *which scale of information* to
preserve.

### Other editing methods

| Method | Idea |
|---|---|
| **DDIM inversion** | run the deterministic ODE *backwards* to find the noise that produces a given image; edit the prompt; re-generate |
| **Prompt-to-Prompt** | inject the original's cross-attention maps into the edited generation, preserving layout |
| **InstructPix2Pix** | train on (image, instruction, edited image) triples → direct instruction following |
| **Null-text inversion** | optimize the unconditional embedding so that inversion is exact under CFG |
| **LoRA / DreamBooth** | fine-tune on a few images of a subject to inject a new concept |

🧠 **DDIM inversion is only possible because of the probability-flow ODE** — a deterministic,
invertible map from noise to image. Stochastic samplers cannot be inverted.
→ [Score-based models §5](02-score-based-models.md#5-the-continuous-time-sde-view)

---

## 6. Personalization

| Method | Trains | Data needed | Output size |
|---|---|---|---|
| **Textual Inversion** | one new token embedding | 3–5 images | ~10 KB |
| **DreamBooth** | the whole U-Net (+ a prior-preservation loss) | 3–20 images | full model |
| **LoRA** | low-rank updates to attention | 10–50 images | 10–200 MB |
| **IP-Adapter** | a decoupled image cross-attention module | 0 (zero-shot at inference) | ~50 MB, reusable |

⚠️ **DreamBooth's language drift problem**: fine-tuning on 5 photos of your dog with the prompt "a
photo of [V] dog" causes the model to associate *all* dogs with your dog. The fix is the
**prior-preservation loss** — simultaneously train on model-generated images of generic dogs to
anchor the original concept.

📊 **LoRA is the practical default**: small files, composable (stack a style LoRA with a character
LoRA), and fast to train. The image-generation community's adapter ecosystem is built on it.
→ [Fine-tuning & PEFT](../04-large-language-models/04-finetuning-peft.md)

---

## 7. Implementation

💻 A complete text-to-image loop, showing every piece:

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

⚠️ **The `scaling_factor` (0.18215 for SD 1.x)** is easy to forget and produces washed-out or
blown-out images when wrong. It normalizes the VAE latents to roughly unit variance, which the
diffusion model expects. Every latent-diffusion model has its own value.

🧠 **Negative prompts are free** — you were computing the unconditional branch anyway, so replacing
$\varnothing$ with "blurry, low quality, watermark" costs nothing and steers *away* from those
concepts. This is a nice example of a capability that falls out of the CFG mechanism rather than
being designed in.

---

## 8. Key takeaways

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

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) — Stable Diffusion.
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022).
- Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021) — classifier guidance.
- Saharia et al., *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* (Imagen, 2022).
- Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet, 2023).
- Meng et al., *SDEdit* (2021); Hertz et al., *Prompt-to-Prompt* (2022).
- Ruiz et al., *DreamBooth* (2022); Ye et al., *IP-Adapter* (2023).
- Esser et al., *Scaling Rectified Flow Transformers* (SD3, 2024).

**Next** → [Flow matching & rectified flow](04-flow-matching.md)
