import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from distortions import add_noise, add_motion_blur, add_rain
from enhancement import denoise, deblur, derain

DATASET_DIR = "dataset"
RESULTS_DIR = os.path.join("results", "geometric_features")
NUM_EVAL = 20

DISTORTIONS = {
    "noise":       (add_noise,       denoise),
    "motion_blur": (add_motion_blur, deblur),
    "rain":        (add_rain,        derain),
}

# Load images
def load_images(split="val", n=NUM_EVAL):
    images_dir = os.path.join(DATASET_DIR, split, "images")
    filenames = os.listdir(images_dir)[:n]
    images = {}
    for name in filenames:
        img = cv2.imread(os.path.join(images_dir, name))
        if img is not None:
            images[name] = img
    return images

# Extract geometric corners using Shi-Tomasi
def get_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.array([])

# Compute match accuracy (repeatability) for feature detection
def compute_repeatability(corners_clean, corners_test, threshold=3.0):
    if len(corners_clean) == 0 or len(corners_test) == 0:
        return 0.0
    
    matched = 0
    for pt1 in corners_clean:
        dists = np.sqrt(np.sum((corners_test - pt1)**2, axis=1))
        if len(dists) > 0 and np.min(dists) <= threshold:
            matched += 1
    return matched / len(corners_clean)

# Evaluate match accuracy across the dataset
def evaluate_dataset(images, distort_fn=None, enhance_fn=None):
    scores = []
    for name, img in images.items():
        corners_clean = get_corners(img)
        test_img = img.copy()
        if distort_fn:
            test_img = distort_fn(test_img)
        if enhance_fn:
            test_img = enhance_fn(test_img)
        corners_test = get_corners(test_img)
        scores.append(compute_repeatability(corners_clean, corners_test))
    return np.mean(scores) if scores else 0
# Compute SNR in dB
def compute_snr(clean, distorted):
    clean_f = clean.astype(np.float64)
    noise = clean_f - distorted.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

# Show feature visualization for clean, distorted, enhanced
def plot_all_variants(images):
    name = list(images.keys())[0]
    img = images[name]
    
    def draw(version):
        display = cv2.cvtColor(version, cv2.COLOR_BGR2RGB).copy()
        corners = get_corners(version)
        for pt in corners:
            cv2.circle(display, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        return display

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for row, (dname, (dfn, efn)) in enumerate(DISTORTIONS.items()):
        dist = dfn(img)
        for col, (title, v) in enumerate([("Clean", img), (dname, dist), (f"{dname} enhanced", efn(dist))]):
            axes[row][col].imshow(draw(v))
            axes[row][col].set_title(title)
            axes[row][col].axis("off")
            
    plt.suptitle("Geometric Features: Clean vs Distorted vs Enhanced", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "all_variants.png"), dpi=150)
    plt.close()

# Performance per SNR
def plot_performance_per_snr(images):
    noise_sev = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    blur_sev = [3, 5, 9, 13, 17, 21, 25]
    rain_sev = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    sample_img = list(images.values())[0]

    noise_snrs = [compute_snr(sample_img, add_noise(sample_img, severity=s)) for s in noise_sev]
    blur_snrs = [compute_snr(sample_img, add_motion_blur(sample_img, kernel_size=k)) for k in blur_sev]
    rain_snrs = [compute_snr(sample_img, add_rain(sample_img, intensity=i)) for i in rain_sev]

    noise_dist = [evaluate_dataset(images, lambda img, s=s: add_noise(img, severity=s)) for s in noise_sev]
    blur_dist = [evaluate_dataset(images, lambda img, k=k: add_motion_blur(img, kernel_size=k)) for k in blur_sev]
    rain_dist = [evaluate_dataset(images, lambda img, i=i: add_rain(img, intensity=i)) for i in rain_sev]

    noise_enh = [evaluate_dataset(images, lambda img, s=s: add_noise(img, severity=s), enhance_fn=denoise) for s in noise_sev]
    blur_enh = [evaluate_dataset(images, lambda img, k=k: add_motion_blur(img, kernel_size=k), enhance_fn=deblur) for k in blur_sev]
    rain_enh = [evaluate_dataset(images, lambda img, i=i: add_rain(img, intensity=i), enhance_fn=derain) for i in rain_sev]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(noise_snrs, noise_dist, "o-", label="Distorted")
    axes[0].plot(noise_snrs, noise_enh, "o--", label="Enhanced")
    for ax in axes:
        ax.set_ylim(0, 1)
    
    axes[0].set_xlabel("SNR (dB) <- noisier")
    axes[0].set_ylabel("Match Accuracy")
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
    baseline = results["clean"]
    distorted = [results[f"{g}_distorted"] for g in groups]
    enhanced = [results[f"{g}_enhanced"] for g in groups]

    x = np.arange(len(groups))
    w = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - w, [baseline] * len(groups), w, label="Clean", color="green")
    plt.bar(x, distorted, w, label="Distorted", color="red")
    plt.bar(x + w, enhanced, w, label="Enhanced", color="blue")
    plt.xticks(x, groups)
    plt.ylabel("Match Accuracy")
    plt.title("Geometric Features: Clean vs Distorted vs Enhanced (overall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = load_images()
    results = {}

    # Sample visualizations
    plot_all_variants(images)
    
    # Baseline on clean images
    results["clean"] = evaluate_dataset(images)

    # Measure degradation on distorted images
    for name, (distort_fn, _) in DISTORTIONS.items():
        results[f"{name}_distorted"] = evaluate_dataset(images, distort_fn=distort_fn)

    # Performance across distortion intensities
    plot_performance_per_snr(images)

    # Measure improvement on enhanced images
    for name, (distort_fn, enhance_fn) in DISTORTIONS.items():
        results[f"{name}_enhanced"] = evaluate_dataset(images, distort_fn=distort_fn, enhance_fn=enhance_fn)

    # Comparison charts
    plot_comparison(results)

    # Print results tables
    print("\nGeometric Features: Match Accuracy")
    print(f"{'Condition':<15} {'Noise':<10} {'Motion Blur':<15} {'Rain':<10}")
    for cond in ["clean", "distorted", "enhanced"]:
        label = cond.title()
        n = results.get(f"noise_{cond}" if cond != "clean" else "clean", 0)
        b = results.get(f"motion_blur_{cond}" if cond != "clean" else "clean", 0)
        r = results.get(f"rain_{cond}" if cond != "clean" else "clean", 0)
        print(f"{label:<15} {n:<10.2f} {b:<15.2f} {r:<10.2f}")

if __name__ == "__main__":
    run()
