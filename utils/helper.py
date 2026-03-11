"""
helper.py
Utility functions used across the project.
"""


def combine_verdicts(ssim_verdict, ocr_valid, verhoeff_valid, cnn_verdict):
    """
    Combine outputs from all modules to produce final decision.

    Args:
        ssim_verdict (str): "Genuine" or "Forged"
        ocr_valid (bool): OCR field validation result
        verhoeff_valid (bool): Aadhaar checksum validation
        cnn_verdict (str): "Genuine" or "Forged"

    Returns:
        str: Final verdict ("Genuine" or "Forged")
    """

    score = 0

    # SSIM result
    if ssim_verdict == "Genuine":
        score += 1

    # OCR validation
    if ocr_valid:
        score += 1

    # Verhoeff checksum
    if verhoeff_valid:
        score += 1

    # CNN prediction
    if cnn_verdict == "Genuine":
        score += 1

    # Majority voting
    if score >= 3:
        return "Genuine"
    else:
        return "Forged"