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

Facial wrinkle analysis is essential for a range of applications, from cosmetic product evaluation to dermatological interventions, invasive or non-invasive. In the past, traditional methods for detecting fine lines on a human face often rely on classical image processing—think edge detection or thresholding. While useful as a baseline, these methods tend to struggle with varying lighting conditions, complex skin textures, and subtle wrinkles, or telling the difference between facial hair and a wrinkle (damn edges.)
After some literature research, I was surprise to discover that the classic U-Net, a popular deep learning architecture initially designed for biomedical image segmentation (e.g., isolating cells, organs, and more) still renders the best results for facial wrinkle segmentation. In this blog post, I’ll discuss how I adapted U-Net for automatic wrinkle segmentation on facial images, highlighting why the topic of wrinkle segmentation is extremely challenging.

I documented my training and evaluation algorithm in a Github repository: [FFHQ-detect-face-wrinkles](https://github.com/rmsandu/FFHQ-detect-face-wrinkles). The demo will also be up soon.

<a id="background"></a>

## Background

Wrinkles are notoriously difficult to detect and segment for various reasons:

- tiny structures
- no consensus on what is a wrinkle, what is a deep wrinkle or a superficial wrinkle
- mismatch in labelling between annotators
- difficult to see based on lighting, angle, photo resolutions
- very few available labelled datasets
- a tiny surface of our face covered by wrinkles which leads to a highly imbalanced negative vs. postive class dataset

<a id=model-architecture></a>

## Model architecture

- **`unet/unet_model.py`**: Defines the U-Net model architecture.
- **`unet/unet_parts.py`**: Contains the building blocks of the U-Net model, such as convolutional layers, downsampling, and upsampling blocks.

Following the steps in literature, the model I chose is a U-Net with a ResNet50 encoder (backbone) which was pretrained on ImageNet. This transfer learning approach could give the model a head start in recognizing low-level patterns. There is the option to freeze or unfreeze a few layers or the entire ResNet50 encoder. When you freeze the entire encoder (i.e., no ResNet weights get updated), the network is effectively stuck with ImageNet features. But if your data is substantially different (like face wrinkles vs. typical ImageNet classes of cats and dogs), the encoder never adapts to the new domain. This typically leads to poor segmentation performance—especially when your target masks are extremely imbalanced (like our very few wrinkle pixels.)
Optionally one can use attention mechanisms to improve the results.Recent research indicates attention can significantly improve segmentation of fine lines. For instance, Yang et al. (2024) introduced Striped WriNet, which features a Striped Attention Module (SAM) to focus on long, thin wrinkle patterns. Adding attention to a U-Net could help the model distinguish wrinkles from skin texture by emphasizing relevant features. This is especially useful for extremely small targets like wrinkles, as attention can amplify their signal relative to the vast background.

**Learning Rate and Optimizer**: AdamW has the advantage of decoupling weight decay (regularization) from the learning rate schedule. The inclusion of weight decay is a positive regularization step, helping to prevent overfitting given the small fraction of positive pixels (wrinkles). I also employed a learning rate scheduler (ReduceLROnPlateau) that reduces lr when validation loss plateaus, and gradient clipping to avoid exploding gradients that can arise due to the focal loss on hard examples. A dropout rate with p=0.2 has also been introduced to further reduce overfitting. The idea was to stabilize the training in the case of imbalanced segmentation tasks.

## Dealing with extreme class imbalance

Given the extreme class imbalance, however, underfitting (lack of signal) is more a concern than overfitting, so regularization beyond data augmentation should be applied judiciously.

1. **Focal + Dice loss evaluation**

Kim et al. (2024) also trained with Dice + Focal to tackle the tiny proportion of wrinkle pixels​. In their study, this combination outperformed alternatives (like binary cross-entropy + focal) in terms of IoU and F1. Dice loss is ideal for segmentation imbalance since it directly maximizes overlap (treating the tiny positive region with equal importance), while focal loss down-weights easy negatives and focuses learning on the hard positives (wrinkle pixels).

However, one has to pay attention to choosing the optimal parameters for these losses. Focal loss can blow up if nearly all pixels are negative (or if your model saturates its outputs to 0 or 1 early on). Even with your small alpha=0.75, if the model sees almost no positive pixels in a batch, the log(1 - p) or log(p) can get extreme, producing Infs / NaNs in half‐precision. 2. Augmentation techniques

2. **Image augmentations**

Mask dilations, etc

3. **Patch-based segmentation**
   TODO

<a id="dataset-preparation"></a>

## Dataset Preparation and Augmentation

The project leverages the [FFHQ-Wrinkle dataset](https://github.com/labhai/ffhq-wrinkle-dataset) and [A Facial Wrinkle Segmentation Method Based on Unet++ Model](https://github.com/jun01pd2015/wrinkle_dataset), which includes both manually labeled and weakly labeled wrinkle masks. However, I have only used manually labeled wrinkle masks from both datasets. The training code and the weights for the manually segmented wrinkles have been provided in this latest update. FFHQ-Wrinkle is an extension of the [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) dataset, specifically designed to include additional features related to facial wrinkles. This dataset aims to support research and development in facial recognition, aging simulation, and other related fields.

I modified the **`face_masking.py`** to a new version of **`face_parsing_extraction.py`** usign [BiSeNET](https://github.com/CoinCheung/BiSeNet) to crop faces given an input folder and provided face-parsed labels for the face images corresponding to the manual wrinkle labels as 512x512 numpy arrays, which were obtained using [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch). This way the user can create new manual labels and generate new face-parsed images.
The pre-trained face weights can be downloaded from [here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) . The pre-trained wrinkle weights can be downloaded from [here](https://drive.google.com/file/d/1C_PtOD5UFlluqf5-kYC0xTUCU8UPY12H/view?usp=sharing). Both ".pth" weight files should be saved in "res/cp/".

Normalization: I normalized each image so that pixel values lie within a specific range, removing variability from lighting conditions.
Augmentations: Since the dataset wasn’t huge, data augmentation was key. Horizontal flips, rotations, contrast enhancement, slight elastic distortion and slight blurs/sharpening improved generalizability.

Here is the workflow for the face pre-processing:

![Face Pre-Processing Workflow](img/workflow-unet.png)

<a id="results-and-evaluation"></a>

## Results and evaluation

All the hyperparameters can be set in **`config.yaml`** file to set the desired training parameters. [Weights&Biases](https://wandb.ai/site) is used for tracking.

- `evaluate.py` and `losses.py` : Evaluates the trained model on the validation dataset. It computes various metrics such as Dice score, precision, recall, F1 score, and AUC. It also logs and saves images for visualization. For Dice score, a combined loss of average Dice Loss + average Focal Loss has been used due to the high class-imbalace. This loss function balances class imbalance and model confidence by penalizing incorrect predictions more aggressively. Itersection-over-Union (IoU or Jaccard Index) is the primary metric of interest to improve upon. IoU directly measures how well the predicted wrinkle mask overlaps with ground truth, which is crucial when the positive region is very small​.

The evaluation code computes not only IoU but also precision, recall, F1, and AUC, which is excellent. Monitoring F1-score (which for segmentation is the same as Dice coefficient) is useful since it’s insensitive to true negatives (dominant here) and gives a sense of balance between precision and recall. In Kim et al.’s experiments, Dice+Focal training yielded an F1 of 0.6157 and IoU ~0.45​, whereas Moon et al. report higher F1 (around 0.64) and Jaccard (~0.48) after their advanced pretrainig. By examining precision-recall curve alongside IoU, we can understand if improvements should focus on reducing false positives or recovering more missed wrinkles. A balanced F1 and high IoU are the end-goals.

<a id="demo"></a>

## Demo

There is cool demo and repository that you can try out yourself. To be continued...

![Face Pre-Processing Workflow](img/demo_screenshot.png)
