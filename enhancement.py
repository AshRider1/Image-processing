import numpy as np
import cv2


# Non-local means denoising
def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 25, 25, 7, 21)


# Unsharp mask for motion blur
def deblur(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# Median + bilateral for rain removal (larger kernel)
def derain(image):
    median = cv2.medianBlur(image, 9)
    return cv2.bilateralFilter(median, 9, 75, 75)
