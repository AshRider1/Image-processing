import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import rank
from skimage.morphology import disk


# Non local means for removing noise
def denoise(image):
    sigma = estimate_sigma(image, channel_axis=2)
    return (denoise_nl_means(image / 255.0, h=0.8 * np.mean(sigma), sigma=np.mean(sigma),
                             fast_mode=True, channel_axis=2) * 255).astype(np.uint8)


# Wiener deconvolution for removing motion blur
def deblur(image, kernel_size=15, angle=0, snr=20):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    channels = cv2.split(image)
    restored = []
    for ch in channels:
        f_img = np.fft.fft2(ch.astype(np.float64))
        f_kernel = np.fft.fft2(kernel, s=ch.shape)
        wiener = np.conj(f_kernel) / (np.abs(f_kernel) ** 2 + 1.0 / snr)
        result = np.abs(np.fft.ifft2(f_img * wiener))
        restored.append(np.clip(result, 0, 255).astype(np.uint8))
    return cv2.merge(restored)


# Median + bilateral filters for removing rain
def derain(image):
    median = cv2.medianBlur(image, 5)
    return cv2.bilateralFilter(median, 9, 75, 75)
