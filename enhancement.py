import numpy as np
import cv2


# Non-local means denoising (stronger h=15)
def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)


# Unsharp mask to counteract motion blur (stronger)
def deblur(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 5)
    sharpened = cv2.addWeighted(image, 2.5, blurred, -1.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# Median + bilateral for rain removal (larger kernel)
def derain(image):
    median = cv2.medianBlur(image, 9)
    return cv2.bilateralFilter(median, 9, 75, 75)
