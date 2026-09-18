import cv2
import numpy as np


def scan_document(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    big_kernel = max(h, w) // 3
    if big_kernel % 2 == 0:
        big_kernel += 1
    bg = cv2.GaussianBlur(gray, (big_kernel, big_kernel), 0)
    normalized = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    return denoised
