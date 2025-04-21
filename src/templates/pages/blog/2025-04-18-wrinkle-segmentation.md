---
date: 2025-04-18
title: Segmentation of Fine Facial Wrinkles with U-Net
subtitle: Segmenting wrinkles is still unsolved task with many applications
cover-img: img/young-old-wrinkles.png
thumbnail-img: img/portrait_ai.png
tags: [segmentation, machine learning, ai, deep learning, Unet]
---

## Table of Contents

- [Introduction](#introduction)
- [What are fine facial wrinkles and why are these so hard to classify?](#background)
- [Model architecture](#model-architecture)
- [Dataset Preparation and Augmentation](#dataset-preparation)
- [Evaluating an imbalanced dataset - tips & tricks](#implementation-challenges)
- [Results and Outlook](#results-and-evaluation)
- [Demo](#demo)

<a id="introduction"></a>

## Introduction

Facial wrinkle analysis is essential for a range of applications, from cosmetic product evaluation to dermatological interventions, invasive or non-invasive. In the past, traditional methods for detecting fine lines on a human face often rely on classical image processing—think edge detection or thresholding. While useful as a baseline, these methods tend to struggle with varying lighting conditions, complex skin textures, and subtle wrinkles, or telling the difference between facial hair and a wrinkle (damn edges!.)
After some literature research, I was surprised to discover that the classic U-Net, a popular deep learning architecture initially designed for biomedical image segmentation (e.g., isolating cells, organs, and more) still renders the best results for facial wrinkle segmentation. In this blog post, I’ll discuss if U-Net for automatic wrinkle segmentation is good enough to get a decent IoU & Dice score, lessons learned, and next tips.

I documented my training and evaluation algorithm in a Github repository: [FFHQ-detect-face-wrinkles](https://github.com/rmsandu/FFHQ-detect-face-wrinkles). The demo is up on
Google Colab.

<a id="background"></a>

## Background

Wrinkles are a pain in the neck for both humans and deep learning models:

- **Tiny structures**: Fine wrinkles are just a few pixels wide. Detecting these thin lines requires high resolution and keen eyesight (or heavy magnification for the model).
- **No clear definition**: There’s no consensus on what counts as a wrinkle versus a fold versus a fine line. One person’s “character line” is another’s “wrinkle.” Even deep vs. superficial wrinkles have fuzzy boundaries.
- **Annotation variability**: Ask two dermatologists (or grad students) to hand-label wrinkles, and you’ll get different answers. Inherent subjectivity leads to inconsistent ground truth masks.
- **Visual ambiguity**: Wrinkles can be hard to see depending on lighting, facial expression, or image resolution. A harsh shadow might mimic a wrinkle. A light wrinkle might vanish under bright light. Think about why everyone started getting a ring light.
- **Data Scarcity**: Very few labeled wrinkle datasets exist. It’s not every day that people volunteer high-resolution photos of their wrinkles along with meticulous annotations.
- **Class imbalance**: On any face, the area covered by wrinkles is tiny compared to smooth skin. This leads to a highly imbalanced dataset (maybe 99% of pixels are “no-wrinkle”). The model can get lazy and just predict “no-wrinkle” everywhere and still achieve 99% accuracy – while completely failing at the actual task.

**In short, wrinkles are the needle in the haystack of facial features. This sets the stage for why a specialized approach (and some clever tricks) are needed for segmentation.**

<a id=model-architecture></a>

## Model architecture

- **`unet/unet_model.py`**: Defines the U-Net model architecture.
- **`unet/unet_parts.py`**: Contains the building blocks of the U-Net model, such as convolutional layers, downsampling, and upsampling blocks.

### Encoder

Following the steps in literature, the model I chose is a U-Net with a ResNet50 encoder (backbone) which was pretrained on ImageNet. This transfer learning approach could give the model a head start in recognizing low-level patterns (like edges and textures). There is the option to freeze or unfreeze a few layers or the entire ResNet50 encoder. When you freeze the entire encoder (i.e., no ResNet weights get updated), the network is effectively stuck with ImageNet features. But if your data is substantially different (like face wrinkles vs. typical ImageNet classes of cats and dogs), a frozen encoder never adapts to the new domain. This typically leads to poor segmentation performance—especially when your target masks are extremely imbalanced (like our very few wrinkle pixels.)

**So I opted to fine-tune the ResNet50 encoder, allowing it to gradually learn wrinkle-relevant features.**

### Decoder

The decoder follows the standard U-Net upsampling path, with skip connections from the encoder. Each skip connection from the encoder is filtered through an Attention Gate before merging. This essentially builds an Attention U-Net – the attention gates learn to emphasize only the relevant features from the encoder (e.g. those pixels that likely belong to wrinkles) and suppress noise (like backgrounds or unrelated textures). Recent research indicates attention can significantly improve segmentation of fine lines. For instance, Yang et al. (2024) introduced Striped WriNet, which uses a Striped Attention Module to focus on long, thin wrinkle patterns. Inspired by that, I integrated attention into my U-Net to help distinguish wrinkles from the surrounding skin by amplifying the telltale features of wrinkles. This is especially useful for such extremely small targets; attention acts like a spotlight on the wrinkles so they don’t get lost in the vast expanse of forehead.

Optionally one can use attention mechanisms to improve the results.Recent research indicates attention can significantly improve segmentation of fine lines. For instance, [Yang et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1746809423012508) introduced Striped WriNet, which features a Striped Attention Module (SAM) to focus on long, thin wrinkle patterns. Adding attention to a U-Net could help the model distinguish wrinkles from skin texture by emphasizing relevant features. This is especially useful for extremely small targets like wrinkles, as attention can amplify their signal relative to the vast background.

**The decoder uses transposed convolutions (learned upsampling) to get back to full 512×512 resolution.**

**Learning Rate and Optimizer**: AdamW has the advantage of decoupling weight decay (regularization) from the learning rate schedule. The inclusion of weight decay is a positive regularization step, helping to prevent overfitting given the small fraction of positive pixels (wrinkles). I also employed a learning rate scheduler (ReduceLROnPlateau) that reduces lr when validation loss plateaus, and gradient clipping to avoid exploding gradients that can arise due to the focal loss on hard examples. A _dropout rate with p=0.2_ has also been introduced to further reduce overfitting. The idea was to stabilize the training in the case of imbalanced segmentation tasks.

<a id="dataset-preparation"></a>

## Dataset Preparation and Augmentation

The project leverages the [FFHQ-Wrinkle dataset](https://github.com/labhai/ffhq-wrinkle-dataset) used by [Moon et al. (2024) Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision)](https://arxiv.org/abs/2408.10060) and [Kim et al. (2024), A Facial Wrinkle Segmentation Method Based on Unet++ Model](https://github.com/jun01pd2015/wrinkle_dataset).

- **FFHQ-Wrinkle dataset (1000 masks)** – Introduced by Moon et al. (2024), this dataset is an extension of NVIDIA’s FFHQ (Flickr-Faces-HQ) dataset. FFHQ-Wrinkle is an extension of the [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) dataset, specifically designed to include additional features related to facial wrinkles. It provides 1,000 high-res face images with manual wrinkle annotations (drawn by humans painstakingly marking every visible wrinkle) and a whopping 50,000 images with automatically generated weak labels. The weak labels are essentially wrinkles predicted by an algorithm (texture extractor) – not perfect, but a clever way to get lots of training data without manual effort. Moon et al.’s idea was to pretrain on those 50k weak labels and then fine-tune on the 1k real labels (something that I should probably do next.)

- **Kim et al.’s wrinkle dataset (820 masks)** – The authors of the UNet++ paper provided their own dataset of manually annotated fine wrinkles across the entire face. I found their data via their project GitHub. I primarily used the manual annotations from this set too and removed the masks that overlapped with the FFHQ wrinkle dataset.

I modified the **`face_masking.py`** to a new version of **`face_parsing_extraction.py`** usign [BiSeNET](https://github.com/CoinCheung/BiSeNet) to crop faces given an input folder and provided face-parsed labels for the face images corresponding to the manual wrinkle labels as 512x512 numpy arrays, which were obtained using [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). This way the user can create new manual labels and generate new face-parsed images.

<div class="grid is-col-min-7">
    <div class="cell">
        <figure class="image is-square">
            <img src="/static/img/00001.png" />
        </figure>
    </div>
    <div class="cell">
        <figure class="image is-square">
            <img src="/static/img/00001_faceMask.png" />
        </figure>
    </div>
</div>

The pre-trained face weights can be downloaded from [here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) . The pre-trained wrinkle weights can be downloaded from [here](https://drive.google.com/file/d/1ZxBPlfPWjx1ZUpaDl61CC8LQhel77-wk/view?usp=sharing). Both ".pth" weight files should be saved in "res/cp/".

### Data transformation:

Since my combined dataset is modest in size, augmentation was crucial to teach the model invariances and to see wrinkles in various conditions. I applied a suite of augmentations randomly during training, including:

- **Normalization**: All augmented images were normalized to the standard ImageNet mean/std, since we’re using a ResNet backbone that expects roughly normalized input.

- **Flips and rotations**: Horizontal flip (because a wrinkle on the left eye vs right eye should be treated the same) and slight rotations (±15°) to simulate different camera angles or head tilts.

- **Scale and translation**: Small scaling (zoom in/out by 10%) and shifts (up to 10% of image) via affine transforms, to make the model robust to distance and centering variations.

- **Lighting and blur**: I used CLAHE (Contrast Limited Adaptive Histogram Equalization) to randomly adjust contrast in local patches (sometimes wrinkles pop more under certain lighting). I added Gaussian blur or noise occasionally because a slight blur can simulate a lower-quality photo where wrinkles are softer, while noise can simulate sensor noise or skin texture variance.

- **Elastic deformation**: (Just made things worse at first try) I tried a slight elastic distortion to mimic the stretching of skin or different facial expressions. However my results were worse, i.e. IoU decreased when I applied elastic transforms.

**One potentially useful augmentation trick: mask dilation**.

Because wrinkles are so thin, one idea was to slightly dilate (thicken) the ground truth wrinkle masks during some training iterations. The thought was to make the model’s job a tiny bit easier (the target line a bit thicker), so it learns to at least hit the vicinity of the wrinkle. This is an optional step and I kept it modest (a 3×3 kernel once). It’s like giving the model a wider paintbrush to aim with initially. In later evaluation, I always measure the thin original masks. This trick of “fattening” the target can sometimes help with extremely thin structures (in theory). I can’t say it was a game-changer here for my final IoU and Dice values, but it’s a handy trick to keep in the toolbox.

Here is the workflow for the face pre-processing:

![Face Pre-Processing Workflow](img/workflow-unet.png)

<a id="results-and-evaluation"></a>

## Results and evaluation

### Dealing with class imbalance

The extreme class imbalance (skin pixels vastly outnumber wrinkle pixels) was the biggest challenge in training. Given the extreme class imbalance, underfitting (lack of signal) is more a concern than overfitting, so regularization beyond data augmentation was applied judiciously.

**Focal + Dice loss evaluation**
[Kim et al. (2024), Automated Facial Wrinkle Segmentation Scheme Using UNet++](https://itiis.org/digital-library/manuscript/file/101103/TIIS%20Vol%2018,%20No%208-15.pdf) also trained with Dice + Focal to tackle the tiny proportion of wrinkle pixels​. In their study, this combination outperformed alternatives (like binary cross-entropy + focal) in terms of IoU and F1. Dice loss is ideal for segmentation imbalance since it directly maximizes overlap (treating the tiny positive region with equal importance), while focal loss down-weights easy negatives and focuses learning on the hard positives (wrinkle pixels).

However, one has to pay attention to choosing the optimal parameters for these losses. Focal loss can blow up if nearly all pixels are negative (or if your model saturates its outputs to 0 or 1 early on). Even with your small alpha=0.75, if the model sees almost no positive pixels in a batch, the log(1 - p) or log(p) can get extreme, producing Infs / NaNs in half‐precision.

In practice, the combined loss = Dice + Focal pushed the model to pay attention to wrinkles without totally neglecting the overall structure. During training, I monitored these losses separately. Typically, Dice loss would start high (since overlap was initially poor) and improve, while focal loss would ensure the model didn’t just predict all zeros.

**Hyperparameters**
All the hyperparameters can be set in **`config.yaml`** file to set the desired training parameters. [Weights&Biases](https://wandb.ai/site) is used for tracking.

- `evaluate.py` and `losses.py` : Evaluates the trained model on the validation dataset. It computes various metrics such as Dice score, precision, recall, F1 score, and AUC. It also logs and saves images for visualization. For Dice score, a combined loss of average Dice Loss + average Focal Loss has been used due to the high class-imbalace. This loss function balances class imbalance and model confidence by penalizing incorrect predictions more aggressively. Itersection-over-Union (IoU or Jaccard Index) is the primary metric of interest to improve upon. IoU directly measures how well the predicted wrinkle mask overlaps with ground truth, which is crucial when the positive region is very small​.

The evaluation code computes not only IoU but also precision, recall, F1, and AUC, which is excellent. Monitoring F1-score (which for segmentation is the same as Dice coefficient) is useful since it’s insensitive to true negatives (dominant here) and gives a sense of balance between precision and recall. In Kim et al.’s experiments, Dice+Focal training yielded an F1 of 0.6157 and IoU ~0.45​, whereas Moon et al. report higher F1 (around 0.64) and Jaccard (~0.48) after their advanced pretrainig. By examining precision-recall curve alongside IoU, we can understand if improvements should focus on reducing false positives or recovering more missed wrinkles. A balanced F1 and high IoU are the end-goals.

### Outlook

**Class balancing and patching**: Next I will consider approaches like oversampling wrinkle regions or training in patches focusing on areas likely to have wrinkles (like around eyes, forehead, mouth.). One idea is to sample patches of the image during training that contain at least one wrinkle, to avoid too many all-background patches. A patch-based refinement (zooming into high-importance areas) is something else that I marked as a potential future improvement.

**Weakly supervised texture maps**:Moon et al. (2024) approached the imbalance from a different angle – they leveraged 50k weakly-labeled images to pretrain their model. In their two-stage strategy, the first stage teaches the model “what a wrinkle could look like” by predicting texture maps on a large dataset, and the second stage fine-tunes on the smaller true dataset. By doing so, they achieved an F1-score around 0.64 and Jaccard Index (IoU) ~0.48, which is currently one of the best reported results for this task​. I didn’t use the weak data for training in this project (to keep things simpler and purely supervised), but that’s my next stepy: more data (even if noisy) can help if used carefully.

<a id="demo"></a>

## Demo

What’s a project without a cool demo? There is an interactive demo where you can upload a face photo (or use one from FFHQ) and see the model highlight the wrinkles. It’s both fascinating and a bit frightening. I used Gradio and Google Colab.

![Demo Screenshot](img/demo_screenshot.png)

**Lessons learned**:

Wrinkle segmentation is hard. It taught me to pay attention (literally, via attention mechanisms) to the little things, to never underestimate the impact of a good loss function, and to always be ready to improvise when the data is limited. If you made it this far, thanks for reading – now excuse me while I go apply some moisturizer and retinol; all this talk of wrinkles has reminded me to take better care of my skin!

![Young vs. Old generated with Midjourney](img/young-old-wrinkles.png)
