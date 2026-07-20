import os
import cv2
import matplotlib.pyplot as plt
from distortions import add_noise, add_motion_blur, add_rain
from enhancement import denoise, deblur, derain

DATASET_DIR = os.path.join("dataset", "val", "images")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DISTORTIONS = {
    "noise":       (add_noise,       denoise),
    "motion_blur": (add_motion_blur, deblur),
    "rain":        (add_rain,        derain),
}


def load_images():
    names = os.listdir(DATASET_DIR)[:3]
    images = []
    for name in names:
        img = cv2.imread(os.path.join(DATASET_DIR, name))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


def show_distortions():
    images = load_images()

    for dist_name, (distort_fn, enhance_fn) in DISTORTIONS.items():
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(dist_name.replace("_", " ").title(), fontsize=16)

        for row, image in enumerate(images):
            distorted = distort_fn(image)
            enhanced = enhance_fn(distorted)

            axes[row][0].imshow(image)
            axes[row][0].set_title("Clean")
            axes[row][1].imshow(distorted)
            axes[row][1].set_title("Distorted")
            axes[row][2].imshow(enhanced)
            axes[row][2].set_title("Enhanced")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{dist_name}.png"), dpi=150)
        plt.show()


if __name__ == "__main__":
    show_distortions()
