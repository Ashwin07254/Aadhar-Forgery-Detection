"""
mobilenet_model.py
MobileNetV2 model loader and predictor for document forgery detection.
"""

import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model

# Path to trained model
MODEL_PATH = "models/mobilenet_forgery_model.keras"

# Prediction threshold
THRESHOLD = 0.35

# Image size expected by MobileNet
IMG_SIZE = (224, 224)

# Load model once
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("[CNN] Model loaded successfully.")
else:
    print("[CNN] Model file not found:", MODEL_PATH)


def predict_image(image_path):
    """Predict whether document is Genuine or Forged."""

    if model is None:
        return {
            "probability": None,
            "verdict": "Model not loaded"
        }

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        return {
            "probability": None,
            "verdict": "Invalid image"
        }

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # CNN prediction
    prediction = model.predict(img, verbose=0)[0][0]

    probability = float(prediction)

    # IMPORTANT: handle reversed class labels
    # Many datasets map: forged=0, genuine=1
    # So we reverse the logic

    if probability >= THRESHOLD:
        verdict = "Genuine"
    else:
        verdict = "Forged"

    print(f"[CNN] Probability: {probability:.4f} → {verdict}")

    return {
        "probability": round(probability, 4),
        "verdict": verdict
    }