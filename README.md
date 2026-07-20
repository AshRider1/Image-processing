# Robustness of Vision Algorithms Under Image Distortions

Image Processing & Computer Vision, Course Project

## What is this project about?

Real-world cameras don't produce perfect images. Sensor noise, motion blur, and bad weather all degrade image quality. But most vision models are trained and tested on clean data. So what happens when we throw distorted images at them? Do they break? Can we fix them?

In this project we take three different vision tasks, apply three types of image distortions, and measure how badly performance drops. Then we try two recovery strategies: classical image enhancement and model fine-tuning.

---

## The Dataset

We use BDD100K, a large-scale driving dataset with 70K training and 10K validation images. Every image comes from a dashboard camera and has multiple annotations:

- Bounding boxes for cars, people, trucks, etc.
- Drivable area polygons showing where the car can drive
- Scene labels like city street, highway, or residential

This means we can run three completely different tasks on the same images.

---

## The Three Tasks

### Task 1: Object Detection (YOLOv8n)
Find and label objects in the image. We use a pretrained YOLOv8n, a deep learning model trained on COCO. It knows how to detect people, cars, buses, trucks, motorcycles, and traffic lights.

### Task 2: Drivable Area Segmentation (SegFormer-b0)
Label every pixel as road or not road. We use SegFormer-b0, a deep learning model pretrained on Cityscapes (another driving dataset), so it already understands what a road looks like.

### Task 3: Scene Classification (HOG + SVM)
Classify the entire image as city street, highway, or residential. This is our classical (non-DL) task. We extract HOG features (which capture edge patterns) and feed them to an SVM classifier trained on BDD100K labels.

This gives us two high-level DL tasks and one low-level classical task, as required by the project.

---

## The Three Distortions

We apply three types of image degradation. All of them preserve image geometry, so the ground truth annotations stay valid.

| Distortion | What it simulates | How it works |
|------------|-------------------|-------------|
| Gaussian Noise | Camera sensor noise | Random pixel values added to the image |
| Motion Blur | Camera or vehicle movement | Directional kernel smears the pixels |
| Rain | Rainy weather | Semi-transparent streaks drawn over the image |

We measure severity using SNR (Signal-to-Noise Ratio) in dB. Higher SNR means a cleaner image.

---

## The Three Enhancements

Each distortion is paired with a classical technique designed to undo the damage:

| Distortion | Enhancement | How it helps |
|------------|-------------|-------------|
| Noise | Non-local means denoising | Averages similar patches to smooth noise while keeping edges |
| Motion Blur | Unsharp mask | Amplifies edges to recover some sharpness |
| Rain | Median blur + bilateral filter | Median removes streaks, bilateral smooths the result |

These are simple preprocessing steps. The question is whether they are enough to help the models.

---

## The Metrics

| Task | Metric | Why |
|------|--------|-----|
| Detection | mAP0.5 (per class) | Captures both correct detections and false alarms |
| Detection | Avg confidence | Shows how certain the model is, even when mAP looks ok |
| Segmentation | Pixel IoU | How much predicted road overlaps with actual road |
| Segmentation | Precision | Whether the model is predicting road where there is none |
| Classification | Accuracy (per class) | How often the model gets the scene right |

---

## The Experiment

We follow five steps:

1. **Baseline**: run each model on clean images
2. **Distortion**: apply each distortion and measure the drop
3. **Enhancement**: enhance the distorted images and measure recovery
4. **Fine-tuning**: retrain the DL models on distorted images (3 epochs) to see if they can adapt
5. **SNR sweep**: vary distortion intensity across 7 levels and plot metric vs SNR

---

## Results

### Distortion Samples

First, let's see what the distortions actually look like on real images. Each row shows clean, distorted, and enhanced side by side on three different images.

#### Noise

![Noise](./results/noise.png)

The noise makes the image grainy, especially in darker regions. The denoising step cleans up most of it, though some fine detail is lost in the process.

#### Motion Blur

![Motion Blur](./results/motion_blur.png)

Everything is smeared horizontally. Signs become unreadable and small objects blend into the background. The unsharp mask brings back some edge contrast, but the lost detail can't be fully recovered since the information was destroyed by the blur kernel.

#### Rain

![Rain](./results/rain.png)

Rain streaks cover large parts of the image. The median + bilateral filter removes most of the visible streaks, though in darker areas some traces remain.

---

### Object Detection (YOLOv8)

#### Ground Truth

This shows a sample image with the human-annotated bounding boxes drawn in green. These are the objects we expect YOLOv8 to find.

![Sample GT](./results/detection/sample_gt.png)

#### Clean vs Distorted vs Enhanced

Each row applies a different distortion. Left column is clean with YOLOv8 predictions, middle is distorted, right is after enhancement. The fourth row stacks all three distortions together.

![All Variants](./results/detection/all_variants.png)

On clean images the model finds most objects confidently. Under noise and rain, some detections vanish or drop in confidence. Motion blur is the most destructive: the model barely finds anything. Enhancement brings some detections back, but the recovery is partial.

#### Baseline Per-Class AP

This chart shows per-class average precision on clean images before any distortion is applied. The red dashed line is the overall mAP0.5.

![Baseline Per-class AP](./results/detection/Per_class_AP_on_clean_images.png)

Larger objects like buses and cars score highest. Smaller or less frequent objects like traffic lights and motorcycles are harder to detect even on clean images.

#### mAP0.5 vs SNR (Performance per Distortion Level)

These three plots show how detection performance changes as distortion gets stronger (moving right means lower SNR, worse image). Solid = distorted, dashed = enhanced, dotted = fine-tuned.

![Performance per SNR](./results/detection/performance_per_snr.png)

For noise, the enhanced line stays above the distorted line, confirming that denoising helps. For motion blur, all three lines are close together, meaning neither enhancement nor fine-tuning can fix the damage. For rain, fine-tuning gives a small advantage at moderate severity levels.

#### mAP0.5 Comparison (Clean vs Distorted vs Enhanced vs Fine-tuned)

This is the summary chart. Four bars per distortion showing the overall mAP0.5 for each condition.

![Comparison](./results/detection/comparison.png)

The gap between green (clean) and red (distorted) shows how much damage each distortion causes. Blue (enhanced) recovers part of it, with the biggest improvement on rain and noise. Orange (fine-tuned) is competitive with enhancement and sometimes better.

#### Per-Class AP0.5 Comparison

These break down the comparison chart by object class for each distortion type.

![Per-class Noise](./results/detection/comparison_per_class_noise.png)
![Per-class Motion Blur](./results/detection/comparison_per_class_motion_blur.png)
![Per-class Rain](./results/detection/comparison_per_class_rain.png)

Not all classes are affected equally. Buses and cars survive distortion better than smaller objects. Motorcycles drop to near zero under all distortions. Enhancement helps most for larger objects that have enough pixels to benefit from denoising or sharpening.

#### Average Detection Confidence (Clean vs Distorted vs Enhanced vs Fine-tuned)

This chart shows the average confidence of all predictions, grouped by distortion.

![Confidence](./results/detection/confidence.png)

Interestingly, confidence stays relatively stable compared to mAP. This tells us something important: the model doesn't just make fewer predictions, it also becomes less sure about the ones it does make. Rain causes the biggest confidence drop. Fine-tuning on noise actually pushes confidence above the clean baseline.

---

### Drivable Area Segmentation (SegFormer)

#### Ground Truth

This shows a sample image with the annotated drivable area overlaid in green. This is what the segmentation model should predict.

![Sample GT](./results/segmentation/sample_gt.png)

#### Clean vs Distorted vs Enhanced

Each row applies a different distortion to the same image. The green overlay is the model's predicted road area.

![All Variants](./results/segmentation/all_variants.png)

On clean images the model captures the road surface well. Under noise, the prediction becomes patchy. Motion blur causes the model to either over-predict or under-predict road areas. After enhancement, the predictions generally look closer to the clean version.

#### IoU vs SNR (Performance per Distortion Level)

How segmentation IoU changes as distortion gets stronger. Solid = distorted, dashed = enhanced, dotted = fine-tuned.

![Performance per SNR](./results/segmentation/performance_per_snr.png)

The fine-tuned models (dotted) maintain stable IoU across severity levels, showing that adapting to distorted data helps the model stay robust even as image quality drops.

#### IoU Comparison (Clean vs Distorted vs Enhanced vs Fine-tuned)

Four bars per distortion showing overall pixel IoU.

![Comparison](./results/segmentation/comparison.png)

The clean baseline sits around 0.48. Distortion drops it to 0.35-0.41 depending on the type. Enhancement recovers some of the loss. Fine-tuning on motion blur performs best, nearly matching the clean baseline.

#### Precision (Clean vs Distorted vs Enhanced vs Fine-tuned)

What fraction of the predicted road is actually road, grouped by distortion.

![Precision](./results/segmentation/precision.png)

Precision tells a different story than IoU. Even when IoU drops, precision can stay high if the model simply predicts less road rather than predicting road in wrong places.

---

### Scene Classification (HOG + SVM)

#### Sample Images

Three examples from each scene class. City streets have dense buildings and traffic, highways are open with clear lane markings, and residential areas have houses and quieter roads.

![Sample GT](./results/classification/sample_gt.png)

#### HOG Feature Visualization

These show the HOG features extracted from clean, distorted, and enhanced images. HOG captures edge directions and is what the SVM uses to classify scenes.

![All Variants](./results/classification/all_variants.png)

On clean images the edges are sharp and structured. Noise adds random edges everywhere. Motion blur smears them directionally. Rain introduces vertical streak patterns. The enhanced versions recover some of the original structure, but not perfectly.

#### Accuracy vs SNR (Performance per Distortion Level)

These plots show how classification accuracy changes with increasing distortion severity. Solid = distorted, dashed = enhanced.

![Performance per SNR](./results/classification/performance_per_snr.png)

Classification turns out to be more robust than detection. This makes sense: HOG captures global edge statistics rather than fine object details, so moderate distortions don't change the overall scene structure much. Noise has the biggest impact, while motion blur barely affects accuracy.

#### Accuracy Comparison (Clean vs Distorted vs Enhanced)

Overall accuracy for clean, distorted, and enhanced across all distortions. No fine-tuning bar since HOG + SVM is classical.

![Comparison](./results/classification/comparison.png)

The accuracy drop is much smaller than what we saw in detection. Noise and rain reduce accuracy by a few percent, and enhancement recovers most of it. Motion blur barely changes accuracy at all.

#### Per-Scene Accuracy Comparison

Accuracy broken down by scene class for each distortion.

![Per-class Comparison](./results/classification/comparison_per_class.png)

City streets are classified reliably under all conditions thanks to their distinctive features (buildings, dense traffic). Highway accuracy varies more since open roads have fewer HOG features. Residential scenes are the hardest to classify in general.

---

## Project Structure

```
Image-processing/
├── main.py              # Entry point (python main.py [task])
├── distortions.py       # Noise, motion blur, rain
├── enhancement.py       # Denoise, deblur, derain
├── detection.py         # YOLOv8 + mAP
├── segmentation.py      # SegFormer + IoU
├── classification.py    # HOG + SVM + accuracy
├── requirements.txt
├── README.md
├── dataset/             # BDD100K (not included)
│   ├── train/
│   ├── val/
│   └── test/
└── results/             # Generated figures
    ├── detection/
    ├── segmentation/
    └── classification/
```

---

## How to Run

Download BDD100K from https://www.kaggle.com/datasets/marquis03/bdd100k and place it in `dataset/`.

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything
python main.py

# Run a specific task
python main.py detection
python main.py segmentation
python main.py classification
python main.py distortions
```
