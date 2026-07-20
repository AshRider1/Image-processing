import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from distortions import add_noise, add_motion_blur, add_rain
from enhancement import denoise, deblur, derain

DATASET_DIR = "dataset"
RESULTS_DIR = os.path.join("results", "classification")
NUM_EVAL = 100
IMG_SIZE = 128

DISTORTIONS = {
    "noise":       (add_noise,       denoise),
    "motion_blur": (add_motion_blur, deblur),
    "rain":        (add_rain,        derain),
}


CLASSES = {"city street": 0, "highway": 1, "residential": 2}

# Load annotations with scene labels
def load_annotations(split="val"):
    path = os.path.join(DATASET_DIR, split, "annotations", f"bdd100k_labels_images_{split}.json")
    with open(path) as f:
        data = json.load(f)
    annotations = {}
    for item in data:
        scene = item.get("attributes", {}).get("scene", "")
        if scene in CLASSES:
            annotations[item["name"]] = CLASSES[scene]
    return annotations


# Extract HOG features from an image
def extract_hog(img):
    gray = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))


# Load images and extract HOG features
def get_features(annotations, split="val", distort_fn=None, enhance_fn=None, n=NUM_EVAL):
    images_dir = os.path.join(DATASET_DIR, split, "images")
    features, labels = [], []
    for name in list(annotations.keys())[:n]:
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        if distort_fn:
            img = distort_fn(img)
        if enhance_fn:
            img = enhance_fn(img)
        features.append(extract_hog(img))
        labels.append(annotations[name])
    return np.array(features), np.array(labels)


# Train SVM on clean images
def train_svm(train_ann):
    X, y = get_features(train_ann, split="train", n=500)
    svm = SVC(kernel="rbf")
    svm.fit(X, y)
    return svm


# Evaluate accuracy (overall + per class)
def evaluate(svm, annotations, distort_fn=None, enhance_fn=None):
    X, y = get_features(annotations, distort_fn=distort_fn, enhance_fn=enhance_fn)
    preds = svm.predict(X)
    result = {"overall": accuracy_score(y, preds)}
    for cls_name, cls_id in CLASSES.items():
        mask = y == cls_id
        result[cls_name] = accuracy_score(y[mask], preds[mask]) if mask.any() else 0
    return result


# Compute SNR in dB
def compute_snr(clean, distorted):
    clean_f = clean.astype(np.float64)
    noise = clean_f - distorted.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# Plot sample: show examples per scene class
def plot_sample(annotations):
    images_dir = os.path.join(DATASET_DIR, "val", "images")
    class_names = list(CLASSES.keys())
    samples = {c: [] for c in class_names}
    for name, label in annotations.items():
        for cls_name, cls_id in CLASSES.items():
            if label == cls_id and len(samples[cls_name]) < 3:
                samples[cls_name].append(name)

    fig, axes = plt.subplots(len(class_names), 3, figsize=(15, 4 * len(class_names)))
    for row, cls_name in enumerate(class_names):
        for col, name in enumerate(samples[cls_name][:3]):
            img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, name)), cv2.COLOR_BGR2RGB)
            axes[row][col].imshow(img)
            axes[row][col].set_title(cls_name)
            axes[row][col].axis("off")
    plt.suptitle("Sample: Scene Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_gt.png"), dpi=150)
    plt.close()


# Show HOG visualization for clean, distorted, enhanced
def plot_all_variants(annotations):
    images_dir = os.path.join(DATASET_DIR, "val", "images")
    name = list(annotations.keys())[0]
    img = cv2.imread(os.path.join(images_dir, name))

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for row, (dname, (dfn, efn)) in enumerate(DISTORTIONS.items()):
        dist = dfn(img)
        for col, (title, v) in enumerate([("Clean", img), (dname, dist), (f"{dname} enhanced", efn(dist))]):
            gray = cv2.cvtColor(cv2.resize(v, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
            _, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            axes[row][col].imshow(hog_img, cmap="gray")
            axes[row][col].set_title(title)
            axes[row][col].axis("off")
    plt.suptitle("HOG features: Clean vs Distorted vs Enhanced", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_variants.png"), dpi=150)
    plt.close()


# Performance per SNR
def plot_performance_per_snr(svm, annotations):
    noise_sev = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    blur_sev = [3, 5, 9, 13, 17, 21, 25]
    rain_sev = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    sample_img = cv2.imread(os.path.join(DATASET_DIR, "val", "images", list(annotations.keys())[0]))
    noise_snrs = [compute_snr(sample_img, add_noise(sample_img, severity=s)) for s in noise_sev]
    blur_snrs = [compute_snr(sample_img, add_motion_blur(sample_img, kernel_size=k)) for k in blur_sev]
    rain_snrs = [compute_snr(sample_img, add_rain(sample_img, intensity=i)) for i in rain_sev]

    noise_dist = [evaluate(svm, annotations, lambda img, s=s: add_noise(img, severity=s))["overall"] for s in noise_sev]
    blur_dist = [evaluate(svm, annotations, lambda img, k=k: add_motion_blur(img, kernel_size=k))["overall"] for k in blur_sev]
    rain_dist = [evaluate(svm, annotations, lambda img, i=i: add_rain(img, intensity=i))["overall"] for i in rain_sev]

    noise_enh = [evaluate(svm, annotations, lambda img, s=s: denoise(add_noise(img, severity=s)))["overall"] for s in noise_sev]
    blur_enh = [evaluate(svm, annotations, lambda img, k=k: deblur(add_motion_blur(img, kernel_size=k)))["overall"] for k in blur_sev]
    rain_enh = [evaluate(svm, annotations, lambda img, i=i: derain(add_rain(img, intensity=i)))["overall"] for i in rain_sev]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(noise_snrs, noise_dist, "o-", label="Distorted")
    axes[0].plot(noise_snrs, noise_enh, "o--", label="Enhanced")
    for ax in axes:
        ax.set_ylim(0, 1)

    axes[0].set_xlabel("SNR (dB) <- noisier")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Noise")
    axes[0].invert_xaxis()
    axes[0].legend()

    axes[1].plot(blur_snrs, blur_dist, "s-", label="Distorted")
    axes[1].plot(blur_snrs, blur_enh, "s--", label="Enhanced")
    axes[1].set_xlabel("SNR (dB) <- blurrier")
    axes[1].set_title("Motion Blur")
    axes[1].invert_xaxis()
    axes[1].legend()

    axes[2].plot(rain_snrs, rain_dist, "^-", label="Distorted")
    axes[2].plot(rain_snrs, rain_enh, "^--", label="Enhanced")
    axes[2].set_xlabel("SNR (dB) <- rainier")
    axes[2].set_title("Rain")
    axes[2].invert_xaxis()
    axes[2].legend()

    plt.suptitle("Performance per SNR", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_per_snr.png"), dpi=150)
    plt.close()


# Comparison bar chart (overall accuracy)
def plot_comparison(results):
    groups = list(DISTORTIONS.keys())
    baseline = results["clean"]["overall"]
    distorted = [results[f"{g}_distorted"]["overall"] for g in groups]
    enhanced = [results[f"{g}_enhanced"]["overall"] for g in groups]

    x = np.arange(len(groups))
    w = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - w, [baseline] * len(groups), w, label="Clean", color="green")
    plt.bar(x, distorted, w, label="Distorted", color="red")
    plt.bar(x + w, enhanced, w, label="Enhanced", color="blue")
    plt.xticks(x, groups)
    plt.ylabel("Accuracy")
    plt.title("Classification: Clean vs Distorted vs Enhanced (overall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()


# Per-class accuracy comparison
def plot_comparison_per_class(results):
    groups = list(DISTORTIONS.keys())
    class_names = list(CLASSES.keys())

    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 6))
    for i, cls in enumerate(class_names):
        baseline = results["clean"][cls]
        distorted = [results[f"{g}_distorted"][cls] for g in groups]
        enhanced = [results[f"{g}_enhanced"][cls] for g in groups]
        x = np.arange(len(groups))
        w = 0.25
        axes[i].bar(x - w, [baseline] * len(groups), w, label="Clean", color="green")
        axes[i].bar(x, distorted, w, label="Distorted", color="red")
        axes[i].bar(x + w, enhanced, w, label="Enhanced", color="blue")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(groups)
        axes[i].set_ylabel("Accuracy")
        axes[i].set_title(cls)
        axes[i].legend()
    plt.suptitle("Classification: Per-class accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_per_class.png"), dpi=150)
    plt.close()


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_ann = load_annotations("train")
    val_ann = load_annotations("val")
    results = {}

    # Sample visualizations
    plot_sample(val_ann)

    # Train SVM on clean images
    svm = train_svm(train_ann)

    # Show HOG features
    plot_all_variants(val_ann)

    # Baseline on clean images
    results["clean"] = evaluate(svm, val_ann)

    # Measure degradation on distorted images
    for name, (distort_fn, _) in DISTORTIONS.items():
        results[f"{name}_distorted"] = evaluate(svm, val_ann, distort_fn=distort_fn)

    # Performance across distortion intensities
    plot_performance_per_snr(svm, val_ann)

    # Measure improvement on enhanced images
    for name, (distort_fn, enhance_fn) in DISTORTIONS.items():
        results[f"{name}_enhanced"] = evaluate(svm, val_ann, distort_fn=distort_fn, enhance_fn=enhance_fn)

    # Comparison charts
    plot_comparison(results)
    plot_comparison_per_class(results)
