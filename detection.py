import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from distortions import add_noise, add_motion_blur, add_rain
from enhancement import denoise, deblur, derain

DATASET_DIR = "dataset"
RESULTS_DIR = os.path.join("results", "detection")
NUM_EVAL = 100

# BDD100K to COCO class mapping
BDD_TO_COCO = {
    "person": 0, "car": 2, "motor": 3,
    "bus": 5, "truck": 7, "traffic light": 9,
}
COCO_TO_BDD = {v: k for k, v in BDD_TO_COCO.items()}

DISTORTIONS = {
    "noise":       (add_noise,       denoise),
    "motion_blur": (add_motion_blur, deblur),
    "rain":        (add_rain,        derain),
}

# Load annotations
def load_annotations(split="val"):
    path = os.path.join(DATASET_DIR, split, "annotations", f"bdd100k_labels_images_{split}.json")
    with open(path) as f:
        data = json.load(f)
    annotations = {}
    for item in data:
        boxes = []
        for label in item.get("labels", []):
            if label["category"] in BDD_TO_COCO and "box2d" in label:
                b = label["box2d"]
                boxes.append({"class": BDD_TO_COCO[label["category"]],
                              "name": label["category"],
                              "bbox": [b["x1"], b["y1"], b["x2"], b["y2"]]})
        if boxes:
            annotations[item["name"]] = boxes
    return annotations

# Plot sample with GT boxes
def plot_sample(annotations):
    name = list(annotations.keys())[0]
    img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for gt in annotations[name]:
        x1, y1, x2, y2 = map(int, gt["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, gt["name"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.title("Sample with GT labels")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "sample_gt.png"), dpi=150)
    plt.close()

# Run YOLOv8 on an image
def run_detection(model, img):
    preds = []
    for r in model(img, verbose=False):
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if int(cls) in COCO_TO_BDD:
                preds.append({"class": int(cls), "name": COCO_TO_BDD[int(cls)],
                              "bbox": box.tolist(), "conf": float(conf)})
    return preds

# Compute IoU between two boxes
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0

# Compute AP0.5 for one class: sort preds by confidence, match to GT, compute precision-recall
def compute_ap(predictions, ground_truths, class_id):
    preds, n_gt = [], 0
    for img in ground_truths:
        n_gt += sum(1 for g in ground_truths[img] if g["class"] == class_id)
        for p in predictions.get(img, []):
            if p["class"] == class_id:
                preds.append((img, p))
    if n_gt == 0:
        return None

    preds.sort(key=lambda x: x[1]["conf"], reverse=True)
    tp, matched = [], {img: set() for img in ground_truths}

    for img, pred in preds:
        ious = [compute_iou(pred["bbox"], g["bbox"]) for g in ground_truths[img] if g["class"] == class_id]
        best = max(range(len(ious)), key=lambda j: ious[j]) if ious else -1
        if best >= 0 and ious[best] >= 0.5 and best not in matched[img]:
            tp.append(1)
            matched[img].add(best)
        else:
            tp.append(0)

    tp = np.cumsum(tp)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / n_gt
    return sum(precision[recall >= t].max() if (recall >= t).any() else 0 for t in np.arange(0, 1.1, 0.1)) / 11


# Compute per-class mAP0.5 (optionally distort/enhance each image)
def compute_map(model, annotations, distort_fn=None, enhance_fn=None, n=NUM_EVAL):
    images_dir = os.path.join(DATASET_DIR, "val", "images")
    filenames = list(annotations.keys())[:n]
    predictions = {}
    for name in filenames:
        img = cv2.imread(os.path.join(images_dir, name))
        if distort_fn:
            img = distort_fn(img)
        if enhance_fn:
            img = enhance_fn(img)
        predictions[name] = run_detection(model, img)
    gt_subset = {k: annotations[k] for k in filenames}
    per_class = {}
    for cls_name, cls_id in BDD_TO_COCO.items():
        ap = compute_ap(predictions, gt_subset, cls_id)
        per_class[cls_name] = ap if ap is not None else 0.0
    return per_class


# Compute mean mAP with a specific distortion severity
def compute_mmap(model, annotations, distort_fn=None, n=NUM_EVAL):
    per_class = compute_map(model, annotations, distort_fn=distort_fn, n=n)
    return np.mean(list(per_class.values())) if per_class else 0


# Plot per-class AP bar chart, optionally with clean baseline side by side
def plot_map(per_class, title="Per class AP", clean_per_class=None):
    classes = sorted(per_class, key=per_class.get, reverse=True)
    values = [per_class[c] for c in classes]
    mmap = np.mean(values)
    x = np.arange(len(classes))
    w = 0.35 if clean_per_class else 0.7
    plt.figure(figsize=(12, 6))
    if clean_per_class:
        clean_values = [clean_per_class.get(c, 0) for c in classes]
        plt.bar(x - w/2, clean_values, w, label="Clean", color="green", alpha=0.5)
        plt.bar(x + w/2, values, w, label="Current")
    else:
        plt.bar(x, values, w)
    plt.axhline(y=mmap, color="red", linestyle="--", label=f"mAP0.5 = {mmap:.3f}")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("AP0.5")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}.png"), dpi=150)
    plt.close()

# Compute SNR in dB between clean and distorted image
def compute_snr(clean, distorted):
    clean_f = clean.astype(np.float64)
    noise = clean_f - distorted.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# Performance per SNR (IoU vs SNR in dB)
def plot_performance_per_snr(model, annotations):
    noise_sev = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    blur_sev = [3, 5, 9, 13, 17, 21, 25]
    rain_sev = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    # Use first image to compute SNR values
    sample_name = list(annotations.keys())[0]
    sample_img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", sample_name))

    noise_snrs = [compute_snr(sample_img, add_noise(sample_img, severity=s)) for s in noise_sev]
    blur_snrs = [compute_snr(sample_img, add_motion_blur(sample_img, kernel_size=k)) for k in blur_sev]
    rain_snrs = [compute_snr(sample_img, add_rain(sample_img, intensity=i)) for i in rain_sev]

    noise_maps = [compute_mmap(model, annotations, lambda img, s=s: add_noise(img, severity=s)) for s in noise_sev]
    blur_maps = [compute_mmap(model, annotations, lambda img, k=k: add_motion_blur(img, kernel_size=k)) for k in blur_sev]
    rain_maps = [compute_mmap(model, annotations, lambda img, i=i: add_rain(img, intensity=i)) for i in rain_sev]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(noise_snrs, noise_maps, "o-")
    axes[0].set_xlabel("SNR (dB) <- noisier")
    axes[0].set_ylabel("mAP0.5")
    axes[0].set_title("Noise")
    axes[0].invert_xaxis()

    axes[1].plot(blur_snrs, blur_maps, "s-")
    axes[1].set_xlabel("SNR (dB) <- blurrier")
    axes[1].set_title("Motion Blur")
    axes[1].invert_xaxis()

    axes[2].plot(rain_snrs, rain_maps, "^-")
    axes[2].set_xlabel("SNR (dB) <- rainier")
    axes[2].set_title("Rain")
    axes[2].invert_xaxis()

    plt.suptitle("Performance per SNR", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_per_snr.png"), dpi=150)
    plt.close()

# Comparison: baseline vs distorted vs enhanced
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
    plt.ylabel("mAP0.5")
    plt.title("Detection: Clean vs Distorted vs Enhanced vs Fine-tuned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()

# Per-class bar chart with clean/distorted/enhanced/finetuned side by side
def plot_comparison_per_class(clean_pc, dist_pc, enh_pc, ft_pc, dist_name):
    classes = sorted(clean_pc, key=clean_pc.get, reverse=True)
    x = np.arange(len(classes))
    w = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*w, [clean_pc[c] for c in classes], w, label="Clean", color="green")
    plt.bar(x - 0.5*w, [dist_pc[c] for c in classes], w, label="Distorted", color="red")
    plt.bar(x + 0.5*w, [enh_pc[c] for c in classes], w, label="Enhanced", color="blue")
    plt.bar(x + 1.5*w, [ft_pc[c] for c in classes], w, label="Fine-tuned", color="orange")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("AP0.5")
    plt.title(f"Per class AP comparison - {dist_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"comparison_per_class_{dist_name}.png"), dpi=150)
    plt.close()


# Show one image with detections: per distortion + all combined
def plot_all_variants(model, annotations):
    name = list(annotations.keys())[0]
    img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", name))

    def draw(version):
        display = cv2.cvtColor(version, cv2.COLOR_BGR2RGB).copy()
        for p in run_detection(model, version):
            if p["conf"] > 0.3:
                x1, y1, x2, y2 = map(int, p["bbox"])
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display, f"{p['name']} {p['conf']:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return display

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # Rows 0-2: per distortion
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
    plt.suptitle("YOLOv8 Detection: Clean vs Distorted vs Enhanced", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_variants.png"), dpi=150)
    plt.close()

# Save YOLO format label file
def save_yolo_label(txt_path, boxes_xyxy, cls_ids, w, h):
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


# Create pseudo-labels from pretrained model on clean images, then distort
def create_finetune_data(model, distort_fn, out_dir, n=100):
    images_dir = os.path.join(DATASET_DIR, "train", "images")
    img_out = os.path.join(out_dir, "images", "train")
    lbl_out = os.path.join(out_dir, "labels", "train")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    filenames = os.listdir(images_dir)[:n]
    for name in filenames:
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]

        r = model.predict(img, conf=0.35, iou=0.5, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        distorted = distort_fn(img)
        cv2.imwrite(os.path.join(img_out, name), distorted)
        save_yolo_label(os.path.join(lbl_out, name.replace(".jpg", ".txt")), xyxy, cls, w, h)


def fine_tune(model, distort_fn, distort_name, epochs=3):
    from pathlib import Path

    out_dir = os.path.join("yolo_data", distort_name)
    create_finetune_data(model, distort_fn, out_dir)

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/train\n")
        f.write(f"nc: 80\nnames: {list(model.names.values())}\n")

    ft_model = YOLO("yolov8n.pt")
    ft_model.train(data=str(yaml_path), imgsz=640, epochs=epochs, batch=2, device="cpu", verbose=False)

    best = Path(ft_model.trainer.best) if hasattr(ft_model, "trainer") else Path("runs/detect/train/weights/best.pt")
    return YOLO(str(best))


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = YOLO("yolov8n.pt")
    annotations = load_annotations("val")
    results = {}
    all_per_class = {}

    # Sample visualizations
    plot_sample(annotations)
    plot_all_variants(model, annotations)

    # Baseline on clean images
    clean_pc = compute_map(model, annotations)
    plot_map(clean_pc, "Per class AP on clean images")
    results["clean"] = np.mean(list(clean_pc.values()))
    all_per_class["clean"] = clean_pc

    # Measure degradation on distorted images
    for name, (distort_fn, _) in DISTORTIONS.items():
        per_class = compute_map(model, annotations, distort_fn=distort_fn)
        plot_map(per_class, f"Per class AP on {name}", clean_per_class=clean_pc)
        results[f"{name}_distorted"] = np.mean(list(per_class.values()))
        all_per_class[f"{name}_distorted"] = per_class

    # Performance across distortion intensities
    plot_performance_per_snr(model, annotations)

    # Measure improvement on enhanced images
    for name, (distort_fn, enhance_fn) in DISTORTIONS.items():
        per_class = compute_map(model, annotations, distort_fn=distort_fn, enhance_fn=enhance_fn)
        plot_map(per_class, f"Per class AP on {name} enhanced", clean_per_class=clean_pc)
        results[f"{name}_enhanced"] = np.mean(list(per_class.values()))
        all_per_class[f"{name}_enhanced"] = per_class

    # Fine-tune on each distortion
    for name, (distort_fn, _) in DISTORTIONS.items():
        ft_model = fine_tune(model, distort_fn, name)
        per_class = compute_map(ft_model, annotations, distort_fn=distort_fn)
        plot_map(per_class, f"Per class AP on {name} finetuned", clean_per_class=clean_pc)
        results[f"{name}_finetuned"] = np.mean(list(per_class.values()))
        all_per_class[f"{name}_finetuned"] = per_class

    plot_comparison(results)

    # Per-class comparison for each distortion
    for name in DISTORTIONS:
        plot_comparison_per_class(clean_pc, all_per_class[f"{name}_distorted"],
                                  all_per_class[f"{name}_enhanced"],
                                  all_per_class[f"{name}_finetuned"], name)

