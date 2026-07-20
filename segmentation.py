import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from distortions import add_noise, add_motion_blur, add_rain
from enhancement import denoise, deblur, derain

DATASET_DIR = "dataset"
RESULTS_DIR = os.path.join("results", "segmentation")
NUM_EVAL = 100

# Cityscapes class 0 = road, fine-tuned class 1 = road
ROAD_CLASS = 0
FT_ROAD_CLASS = 1

DISTORTIONS = {
    "noise":       (add_noise,       denoise),
    "motion_blur": (add_motion_blur, deblur),
    "rain":        (add_rain,        derain),
}

PROCESSOR = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

# Load annotations
def load_annotations(split="val"):
    path = os.path.join(DATASET_DIR, split, "annotations", f"bdd100k_labels_images_{split}.json")
    with open(path) as f:
        data = json.load(f)
    annotations = {}
    for item in data:
        areas = {"direct": [], "alternative": []}
        for label in item.get("labels", []):
            if label["category"] == "drivable area" and "poly2d" in label:
                area_type = label.get("attributes", {}).get("areaType", "direct")
                for poly in label["poly2d"]:
                    pts = np.array(poly["vertices"], dtype=np.int32)
                    if area_type in areas:
                        areas[area_type].append(pts)
        if areas["direct"] or areas["alternative"]:
            annotations[item["name"]] = areas
    return annotations

# Rasterize polygons into binary mask (both direct + alternative = drivable)
def make_mask(areas, h=720, w=1280):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in areas.get("alternative", []):
        cv2.fillPoly(mask, [pts], 1)
    for pts in areas.get("direct", []):
        cv2.fillPoly(mask, [pts], 1)
    return mask

# Rasterize with separate classes for visualization (1=direct, 2=alternative)
def make_mask_colored(areas, h=720, w=1280):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in areas.get("alternative", []):
        cv2.fillPoly(mask, [pts], 2)
    for pts in areas.get("direct", []):
        cv2.fillPoly(mask, [pts], 1)
    return mask

# Plot sample with GT overlay
def plot_sample(annotations, idx=0):
    name = list(annotations.keys())[idx]
    img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = make_mask(annotations[name], img.shape[0], img.shape[1])
    overlay = img.copy()
    overlay[mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    plt.figure(figsize=(12, 7))
    plt.imshow(blended)
    plt.title("Drivable area GT")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "sample_gt.png"), dpi=150)
    plt.close()

# Run SegFormer on an image, return binary road mask
def run_segmentation(model, img, road_class=ROAD_CLASS):
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = PROCESSOR(images=inp, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    road_mask = (pred == road_class).astype(np.uint8)
    return cv2.resize(road_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Compute pixel IoU
def compute_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    union = int((p | g).sum())
    return float((p & g).sum()) / union if union else 0

# Compute mean IoU and precision across n images
def compute_ious(model, annotations, distort_fn=None, enhance_fn=None, road_class=ROAD_CLASS, n=NUM_EVAL):
    images_dir = os.path.join(DATASET_DIR, "val", "images")
    ious, precs = [], []
    for name in list(annotations.keys())[:n]:
        img = cv2.imread(os.path.join(images_dir, name))
        gt = make_mask(annotations[name], img.shape[0], img.shape[1])
        if distort_fn:
            img = distort_fn(img)
        if enhance_fn:
            img = enhance_fn(img)
        pred = run_segmentation(model, img, road_class=road_class)
        ious.append(compute_iou(pred, gt))
        p, g = pred.astype(bool), gt.astype(bool)
        precs.append(float((p & g).sum()) / p.sum() if p.sum() > 0 else 0)
    return np.mean(ious) if ious else 0, np.mean(precs) if precs else 0

# Compute mean IoU with a specific distortion severity
def compute_miou(model, annotations, distort_fn=None, n=NUM_EVAL):
    iou, _ = compute_ious(model, annotations, distort_fn=distort_fn, n=n)
    return iou

# Compute SNR in dB between clean and distorted image
def compute_snr(clean, distorted):
    clean_f = clean.astype(np.float64)
    noise = clean_f - distorted.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

# Performance per SNR (IoU vs SNR in dB) with distorted + enhanced + fine-tuned
def plot_performance_per_snr(model, annotations, ft_models=None):
    noise_sev = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    blur_sev = [3, 5, 9, 13, 17, 21, 25]
    rain_sev = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    sample_name = list(annotations.keys())[0]
    sample_img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", sample_name))

    noise_snrs = [compute_snr(sample_img, add_noise(sample_img, severity=s)) for s in noise_sev]
    blur_snrs = [compute_snr(sample_img, add_motion_blur(sample_img, kernel_size=k)) for k in blur_sev]
    rain_snrs = [compute_snr(sample_img, add_rain(sample_img, intensity=i)) for i in rain_sev]

    noise_dist = [compute_miou(model, annotations, lambda img, s=s: add_noise(img, severity=s)) for s in noise_sev]
    blur_dist = [compute_miou(model, annotations, lambda img, k=k: add_motion_blur(img, kernel_size=k)) for k in blur_sev]
    rain_dist = [compute_miou(model, annotations, lambda img, i=i: add_rain(img, intensity=i)) for i in rain_sev]

    noise_enh = [compute_miou(model, annotations, lambda img, s=s: denoise(add_noise(img, severity=s))) for s in noise_sev]
    blur_enh = [compute_miou(model, annotations, lambda img, k=k: deblur(add_motion_blur(img, kernel_size=k))) for k in blur_sev]
    rain_enh = [compute_miou(model, annotations, lambda img, i=i: derain(add_rain(img, intensity=i))) for i in rain_sev]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(noise_snrs, noise_dist, "o-", label="Distorted")
    axes[0].plot(noise_snrs, noise_enh, "o--", label="Enhanced")
    axes[1].plot(blur_snrs, blur_dist, "s-", label="Distorted")
    axes[1].plot(blur_snrs, blur_enh, "s--", label="Enhanced")
    axes[2].plot(rain_snrs, rain_dist, "^-", label="Distorted")
    axes[2].plot(rain_snrs, rain_enh, "^--", label="Enhanced")

    if ft_models:
        noise_ft = [compute_miou(ft_models["noise"], annotations, lambda img, s=s: add_noise(img, severity=s)) for s in noise_sev]
        blur_ft = [compute_miou(ft_models["motion_blur"], annotations, lambda img, k=k: add_motion_blur(img, kernel_size=k)) for k in blur_sev]
        rain_ft = [compute_miou(ft_models["rain"], annotations, lambda img, i=i: add_rain(img, intensity=i)) for i in rain_sev]
        axes[0].plot(noise_snrs, noise_ft, "o:", label="Fine-tuned")
        axes[1].plot(blur_snrs, blur_ft, "s:", label="Fine-tuned")
        axes[2].plot(rain_snrs, rain_ft, "^:", label="Fine-tuned")

    axes[0].set_xlabel("SNR (dB) <- noisier")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Noise")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[1].set_xlabel("SNR (dB) <- blurrier")
    axes[1].set_title("Motion Blur")
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[2].set_xlabel("SNR (dB) <- rainier")
    axes[2].set_title("Rain")
    axes[2].invert_xaxis()
    axes[2].legend()

    plt.suptitle("Performance per SNR", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_per_snr.png"), dpi=150)
    plt.close()

# Comparison: baseline vs distorted vs enhanced vs fine-tuned
def plot_comparison(results):
    groups = list(DISTORTIONS.keys())
    baseline = results["clean"]
    distorted = [results[f"{g}_distorted"] for g in groups]
    enhanced = [results[f"{g}_enhanced"] for g in groups]
    finetuned = [results[f"{g}_finetuned"] for g in groups]

    x = np.arange(len(groups))
    w = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*w, [baseline] * len(groups), w, label="Clean", color="green")
    plt.bar(x - 0.5*w, distorted, w, label="Distorted", color="red")
    plt.bar(x + 0.5*w, enhanced, w, label="Enhanced", color="blue")
    plt.bar(x + 1.5*w, finetuned, w, label="Fine-tuned", color="orange")
    plt.xticks(x, groups)
    plt.ylabel("mIoU")
    plt.title("Comparison: Clean vs Distorted vs Enhanced vs Fine-tuned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()

# Precision grouped by distortion
def plot_precision(results):
    groups = list(DISTORTIONS.keys())
    baseline = results["clean"]
    distorted = [results[f"{g}_distorted"] for g in groups]
    enhanced = [results[f"{g}_enhanced"] for g in groups]
    finetuned = [results[f"{g}_finetuned"] for g in groups]

    x = np.arange(len(groups))
    w = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*w, [baseline] * len(groups), w, label="Clean", color="green")
    plt.bar(x - 0.5*w, distorted, w, label="Distorted", color="red")
    plt.bar(x + 0.5*w, enhanced, w, label="Enhanced", color="blue")
    plt.bar(x + 1.5*w, finetuned, w, label="Fine-tuned", color="orange")
    plt.xticks(x, groups)
    plt.ylabel("Precision")
    plt.title("Segmentation: Precision per condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision.png"), dpi=150)
    plt.close()


# Show one image with segmentation overlay: per distortion
def plot_all_variants(model, annotations, idx=0):
    name = list(annotations.keys())[idx]
    img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", name))

    def draw(version):
        pred = run_segmentation(model, version)
        display = cv2.cvtColor(version, cv2.COLOR_BGR2RGB).copy()
        display[pred == 1] = [0, 255, 0]
        return display

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for row, (dname, (distort_fn, enhance_fn)) in enumerate(DISTORTIONS.items()):
        distorted = distort_fn(img)
        enhanced = enhance_fn(distorted)
        axes[row][0].imshow(draw(img))
        axes[row][0].set_title("Clean")
        axes[row][1].imshow(draw(distorted))
        axes[row][1].set_title(dname)
        axes[row][2].imshow(draw(enhanced))
        axes[row][2].set_title(f"{dname} enhanced")

    # Row 3: all distortions combined, all enhancements combined
    all_distorted = img.copy()
    for distort_fn, _ in DISTORTIONS.values():
        all_distorted = distort_fn(all_distorted)
    all_enhanced = all_distorted.copy()
    for _, enhance_fn in DISTORTIONS.values():
        all_enhanced = enhance_fn(all_enhanced)

    axes[3][0].imshow(draw(img))
    axes[3][0].set_title("Clean")
    axes[3][1].imshow(draw(all_distorted))
    axes[3][1].set_title("All distortions")
    axes[3][2].imshow(draw(all_enhanced))
    axes[3][2].set_title("All enhanced")

    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle("SegFormer Segmentation: Clean vs Distorted vs Enhanced", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_variants.png"), dpi=150)
    plt.close()

# Create pseudo-labels from pretrained model on clean images, then distort
def create_finetune_data(model, distort_fn, out_dir, n=100):
    images_dir = os.path.join(DATASET_DIR, "train", "images")
    img_out = os.path.join(out_dir, "images")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    filenames = os.listdir(images_dir)[:n]
    for name in filenames:
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        # Get pseudo-label from pretrained model on clean image
        mask = run_segmentation(model, img)
        # Save distorted image + clean pseudo-label mask
        distorted = distort_fn(img)
        cv2.imwrite(os.path.join(img_out, name), distorted)
        cv2.imwrite(os.path.join(mask_out, name.replace(".jpg", ".png")), mask * 255)

def fine_tune(model, distort_fn, distort_name, epochs=3):
    out_dir = os.path.join("seg_data", distort_name)
    create_finetune_data(model, distort_fn, out_dir)

    class SegDataset(Dataset):
        def __init__(self, img_dir, mask_dir):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.names = os.listdir(img_dir)
        def __len__(self):
            return len(self.names)
        def __getitem__(self, i):
            name = self.names[i]
            img = cv2.imread(os.path.join(self.img_dir, name))
            img = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.mask_dir, name.replace(".jpg", ".png")), 0)
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.int64)
            inputs = PROCESSOR(images=img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), torch.LongTensor(mask)

    dataset = SegDataset(os.path.join(out_dir, "images"), os.path.join(out_dir, "masks"))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    ft_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ft_model.decode_head.classifier = nn.Conv2d(256, 2, 1)
    ft_model.train()
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for imgs, masks in loader:
            logits = ft_model(pixel_values=imgs).logits
            logits = nn.functional.interpolate(logits, size=masks.shape[1:], mode="bilinear")
            loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ft_model.eval()
    return ft_model


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model.eval()
    annotations = load_annotations("val")
    results = {}

    # Sample visualizations
    plot_sample(annotations)
    plot_all_variants(model, annotations)

    iou_results, prec_results = {}, {}

    # Baseline on clean images
    iou_results["clean"], prec_results["clean"] = compute_ious(model, annotations)

    # Measure degradation on distorted images
    for name, (distort_fn, _) in DISTORTIONS.items():
        iou_results[f"{name}_distorted"], prec_results[f"{name}_distorted"] = compute_ious(model, annotations, distort_fn=distort_fn)

    # Measure improvement on enhanced images
    for name, (distort_fn, enhance_fn) in DISTORTIONS.items():
        iou_results[f"{name}_enhanced"], prec_results[f"{name}_enhanced"] = compute_ious(model, annotations, distort_fn=distort_fn, enhance_fn=enhance_fn)

    # Fine-tune on each distortion
    ft_models = {}
    for name, (distort_fn, _) in DISTORTIONS.items():
        ft_models[name] = fine_tune(model, distort_fn, name)
        iou_results[f"{name}_finetuned"], prec_results[f"{name}_finetuned"] = compute_ious(ft_models[name], annotations, distort_fn=distort_fn, road_class=FT_ROAD_CLASS)

    # Performance across distortion intensities (distorted + enhanced + fine-tuned)
    plot_performance_per_snr(model, annotations, ft_models=ft_models)

    # Comparison charts
    plot_comparison(iou_results)
    plot_precision(prec_results)
