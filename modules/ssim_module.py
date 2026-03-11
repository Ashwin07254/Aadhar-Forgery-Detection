"""
ssim_module.py
SSIM based document similarity detection
"""

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from modules.preprocessing import preprocess_for_ssim

SSIM_THRESHOLD = 0.75


def compute_ssim(img1, img2):

    score, diff = ssim(img1, img2, full=True)

    diff = (diff * 255).astype(np.uint8)

    return score, diff


def check_ssim(test_image_path, template_path):

    test_img = preprocess_for_ssim(test_image_path)

    template_img = preprocess_for_ssim(template_path)

    score, diff = compute_ssim(test_img, template_img)

    verdict = "Genuine" if score >= SSIM_THRESHOLD else "Forged"

    print(f"[SSIM] Score: {score:.4f} → {verdict}")

    return {
        "ssim_score": round(score, 4),
        "verdict": verdict,
        "diff_image": diff
    }


def highlight_differences(diff_image):

    _, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, thresh