"""
preprocessing.py
Image preprocessing utilities
"""

import cv2
import numpy as np

IMG_SIZE = (224, 224)


def load_image(path):
    """Load image from disk"""

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    return img


def resize_image(img, size=IMG_SIZE):
    """Resize image safely"""

    if img is None:
        raise ValueError("Image is None")

    return cv2.resize(img, size)


def to_grayscale(img):
    """Convert BGR image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize(img):
    """Normalize image pixels"""
    return img.astype(np.float32) / 255.0


def preprocess_for_ssim(path):
    """Prepare image for SSIM"""

    img = load_image(path)

    img = resize_image(img)

    gray = to_grayscale(img)

    return gray


def preprocess_for_cnn(path):
    """Prepare image for CNN"""

    img = load_image(path)

    img = resize_image(img)

    img = normalize(img)

    return img