import numpy as np
import cv2


# Add Gaussian noise
def add_noise(image, severity=0.2):
    sigma = severity * 255
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


# Add motion blur
def add_motion_blur(image, kernel_size=25, angle=0):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return cv2.filter2D(image, -1, kernel)


# Add rain
def add_rain(image, intensity=0.15, slant=10, drop_length=20):
    h, w = image.shape[:2]
    rain = np.zeros_like(image)
    for _ in range(int(intensity * h * w / drop_length)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(rain, (x, y), (min(x + slant, w - 1), min(y + drop_length, h - 1)), (200, 200, 200), 1)
    return cv2.addWeighted(image, 1.0, rain, 0.5, 0)
