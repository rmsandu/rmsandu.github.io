---
date: 2026-08-18
title: Multi-View Image Generation with In-Context LoRA
subtitle: An honest, work-in-progress attempt at teaching FLUX to keep an object's identity straight across four views
cover-img: img/multiview-composite-example.jpeg
thumbnail-img: img/multiview-example-output1.jpeg
tags: [genai, lora, flux, diffusion, computer vision, physical ai]
---

Ever since finishing up the wrinkle segmentation project, I've been circling a bigger question: not just recognizing what's flat in a single image, but understanding an object well enough to picture it from angles you haven't actually shown the model. That's the "physical AI" itch I keep mentioning on my homepage — how do machines understand and reason about our messy 3D world from a handful of 2D photos? This post is my newest (and still very much unfinished) attempt at chipping away at that, and I'd rather write it up honestly mid-flight than wait for a tidy ending that may be months away. Code's here: [In-Context-multiview-img-generation](https://github.com/rmsandu/In-Context-multiview-img-generation).

## The question I'm actually testing

The approach builds on [In-Context LoRA](https://ali-vilab.github.io/In-Context-LoRA-Page/), a technique for adapting diffusion transformer (DiT) models — I'm using [FLUX](https://github.com/black-forest-labs/flux) — to multi-image outputs without touching the model architecture. The trick is deceptively simple: concatenate several images of the same object into one big composite (a 2×2 grid, in my case), write a single joint caption describing all four sub-images, and fine-tune on a small dataset with LoRA. Because LoRA only inserts trainable low-rank matrices into the attention layers and freezes everything else, the whole thing trains on a single RTX 4090 instead of a GPU cluster — the same reason I like it for the fine-tuning work I described in my [SDXL post](/blog/2024-06-15-sdxlfinetuning.html).

The original In-Context LoRA paper only evaluated results qualitatively — someone eyeballed the generated grids and decided if they looked coherent. That bugged me, because "looks fine" is exactly how you miss a model quietly duplicating the same view four times and calling it four angles. So the actual research question I'm chasing is: **does LoRA preserve object identity across views, or does it just learn to copy-paste the input with minor noise?** Answering that properly means building a real evaluation pipeline, not just staring at grids like the cover photo above — more on that below.

## Building the dataset

The source data comes from [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet), a large multi-view image dataset. From it, I deterministically select four well-spaced views per object and concatenate them into a composite. Writing a single caption that accurately describes all four sub-images by hand doesn't scale, so I automated it with `gemini-3.5-flash`, constrained to a Pydantic schema (`MultiviewAnnotation`) that forces the model to return a structured viewpoint, side, vertical angle, framing, and confidence for each tile rather than free-form text. Annotations the model flags as `indeterminate` get routed to a separate abstention set instead of silently entering the training data — better an honest "I don't know" than a wrong pose label baked into the dataset.

The most recent full run (July 2026) processed every eligible instance under `data/`:

| Item | Result |
| --- | ---: |
| Eligible four-view instances | 529 |
| Selected source images | 2,116 |
| Accepted image/caption pairs | 423 |
| Abstention composites | 106 |

Here's what the accepted pairs actually look like — the composite next to the exact structured caption the pipeline generated:

<figure class="image">
    <img src="/static/img/multiview-sanity-check-2views.png" alt="Sanity-check contact sheet showing two accepted four-view composites (a table tennis racket and a red handbag) next to their structured Gemini-generated captions" />
</figure>
<figcaption>Two accepted composites with their structured [FOUR-VIEWS] / [TOP-LEFT] / [TOP-RIGHT] / [BOTTOM-LEFT] / [BOTTOM-RIGHT] captions.</figcaption>

<figure class="image">
    <img src="/static/img/multiview-sanity-check-4views.png" alt="Sanity-check contact sheet showing four accepted four-view composites (flowers, a toy motorcycle, a toy pistol, and a plush toy) next to their structured captions" />
</figure>
<figcaption>Four more accepted composites — this is the "sanity check" step I run before trusting a captioning batch enough to train on it.</figcaption>

## Where things stand right now

To be upfront: this is not a finished project, and I don't want to dress it up as one. What's actually done:

- Deterministic four-view selection and composite construction, with duplicate-image hashing.
- The Gemini-based structured captioning pipeline above, including a resumable Batch API workflow for running it over hundreds of composites without babysitting it.
- A reproducible 90/10 train/holdout split of the 423 accepted pairs (seed 17 → 381 train / 42 holdout), with leakage checks so no source instance or image hash leaks across the split.
- A minimal Study 1 pilot LoRA config (rank 16, 500 steps) trained via [AI-toolkit](https://github.com/ostris/ai-toolkit), plus a harness that generates paired base-FLUX-vs-LoRA outputs and a blinded scorecard for human raters.

What I'm explicitly *not* claiming yet is a finished evaluation — the blinded scoring hasn't been run to completion, and I don't have DINOv2/DreamSim/LPIPS numbers to show you instead of "trust me, it looks okay." The two example generations below predate the current structured-captioning pipeline entirely, and I'm keeping them purely as historical curiosities, not as evidence of anything:

<div class="grid is-col-min-7">
    <div class="cell">
        <figure class="image is-square">
            <img src="/static/img/multiview-example-output1.jpeg" alt="Four generated views of a red desk lamp from different angles" />
        </figure>
        <figcaption>An early four-view generation of a red desk lamp, from before the structured captioning pipeline existed.</figcaption>
    </div>
    <div class="cell">
        <figure class="image is-square">
            <img src="/static/img/multiview-example-output2.jpeg" alt="Four generated views of a bedroom interior from different angles" />
        </figure>
        <figcaption>Same story, a bedroom this time — consistent-looking, but generated under the old free-form caption format.</figcaption>
    </div>
</div>

You can try the current LoRA checkpoint yourself without any of this setup: [Hugging Face Model Card](https://huggingface.co/rmsandu/fourviews-incontext-lora) and [Hugging Face Demo Space](https://huggingface.co/spaces/rmsandu/fourviews-incontext-lora).

## What's next

The honest roadmap, straight from the repo:

- Review the 106 abstentions and spot-check the 423 accepted annotations before trusting them as clean training data.
- Build and adjudicate a 100-composite gold benchmark to actually measure captioning quality.
- Benchmark a handful of Gemini models against each other for viewpoint annotation.
- Train pose-conditioned and appearance-only LoRA variants, three seeds each, so I can separate "does it know the pose" from "does it keep the object looking like itself."
- Run the Study 1 pilot training through to a completed, blinded human scoring pass.
- Add DINOv2, DreamSim, LPIPS, and perceptual-hash metrics so identity preservation is a number, not a vibe.

If any of this overlaps with something you're working on — multi-view generation, structured VLM captioning, or evaluating diffusion models beyond "it looks fine to me" — I'd genuinely like to compare notes. Details on how to reach me are on the [consulting page](/consulting.html).

## Useful resources

- [In-Context-multiview-img-generation](https://github.com/rmsandu/In-Context-multiview-img-generation) — full pipeline code, configs, and the `PLANS.md` write-up of the research questions.
- [In-Context LoRA](https://ali-vilab.github.io/In-Context-LoRA-Page/) — the original method this builds on.
- [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) — the source multi-view dataset.
- [AI-toolkit](https://github.com/ostris/ai-toolkit) — the LoRA training harness I used for FLUX.
- [Hugging Face Model Card](https://huggingface.co/rmsandu/fourviews-incontext-lora) and [Demo Space](https://huggingface.co/spaces/rmsandu/fourviews-incontext-lora).
