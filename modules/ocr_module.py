"""
ocr_module.py
OCR extraction and Aadhaar field validation
"""

import pytesseract
import cv2
import re
from modules.preprocessing import load_image, resize_image

# Path to Tesseract OCR (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """Extract text from Aadhaar card using Tesseract."""

    img = load_image(image_path)

    # Resize image for better OCR accuracy
    img = resize_image(img, (800, 500))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    config = "--psm 6"

    raw_text = pytesseract.image_to_string(gray, config=config)

    return raw_text


def validate_aadhaar_number(text):
    """Check Aadhaar number pattern."""

    pattern = r"\b\d{4}\s\d{4}\s\d{4}\b"
    matches = re.findall(pattern, text)

    return len(matches) > 0, matches


def validate_dob(text):
    """Check Date of Birth."""

    pattern = r"\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}\b"

    matches = re.findall(pattern, text)

    return len(matches) > 0, matches


def validate_gender(text):
    """Check gender field."""

    pattern = r"\b(MALE|FEMALE|TRANSGENDER)\b"

    matches = re.findall(pattern, text.upper())

    return len(matches) > 0, matches


def validate_name(text):
    """Find possible name lines."""

    lines = text.split("\n")

    name_candidates = []

    for line in lines:

        line = line.strip()

        if re.match(r"^[A-Za-z]+(\s[A-Za-z]+)+$", line) and len(line) > 4:
            name_candidates.append(line)

    return len(name_candidates) > 0, name_candidates


def validate_pincode(text):
    """Detect Indian PIN code."""

    pattern = r"\b[1-9][0-9]{5}\b"

    matches = re.findall(pattern, text)

    return len(matches) > 0, matches


def extract_aadhaar_number_digits(text):
    """Extract 12-digit Aadhaar number."""

    pattern = r"\b(\d{4})\s(\d{4})\s(\d{4})\b"

    match = re.search(pattern, text)

    if match:
        return match.group(1) + match.group(2) + match.group(3)

    return None


def run_ocr_validation(image_path):
    """Run full OCR validation pipeline."""

    text = extract_text(image_path)

    aadhaar_valid, _ = validate_aadhaar_number(text)
    dob_valid, _ = validate_dob(text)
    gender_valid, _ = validate_gender(text)
    name_valid, _ = validate_name(text)
    pin_valid, _ = validate_pincode(text)

    aadhaar_digits = extract_aadhaar_number_digits(text)

    overall = aadhaar_valid and dob_valid and gender_valid

    return {
        "raw_text": text,
        "aadhaar_number": aadhaar_digits,
        "has_valid_aadhaar": aadhaar_valid,
        "has_valid_dob": dob_valid,
        "has_valid_gender": gender_valid,
        "has_valid_name": name_valid,
        "has_valid_pincode": pin_valid,
        "overall_ocr_valid": overall
    }