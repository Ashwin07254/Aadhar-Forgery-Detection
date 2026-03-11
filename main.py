import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from PIL import Image

# allow importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ssim_module import check_ssim
from modules.ocr_module import run_ocr_validation
from modules.verhoeff import check_id_integrity
from modules.mobilenet_model import predict_image
from utils.helper import combine_verdicts


# Page config
st.set_page_config(
    page_title="Aadhaar Forgery Detector",
    page_icon="🪪",
    layout="wide"
)

st.title("🪪 Aadhaar Card Forgery Detection System")
st.write("Detect whether an Aadhaar card is **Genuine or Forged**")


# Sidebar settings
st.sidebar.header("Settings")

use_ssim = st.sidebar.checkbox("Enable SSIM", True)
use_ocr = st.sidebar.checkbox("Enable OCR", True)
use_verhoeff = st.sidebar.checkbox("Enable Verhoeff", True)
use_cnn = st.sidebar.checkbox("Enable CNN", True)

st.sidebar.markdown("---")

template_file = st.sidebar.file_uploader(
    "Upload Template Image (optional)",
    type=["jpg", "png", "jpeg"]
)


# Choose input method
st.markdown("### Select Input Method")

input_method = st.radio(
    "Choose how to provide image",
    ["Upload Image", "Use Camera"]
)

image = None


# Upload image
if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Aadhaar Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)


# Capture from camera
elif input_method == "Use Camera":

    camera_image = st.camera_input("Capture Aadhaar Card")

    if camera_image:
        image = Image.open(camera_image)
        image = np.array(image)


# Run detection if image exists
if image is not None:

    st.image(image, caption="Input Image", use_column_width=True)

    # save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        test_path = tmp.name

    # save template if uploaded
    template_path = None
    if template_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            tmp2.write(template_file.read())
            template_path = tmp2.name

    results = {}

    st.subheader("Running Analysis...")


    # SSIM
    if use_ssim and template_path:
        ssim_res = check_ssim(test_path, template_path)
        results["ssim_verdict"] = ssim_res["verdict"]
        results["ssim_score"] = ssim_res["ssim_score"]
    else:
        results["ssim_verdict"] = "Genuine"
        results["ssim_score"] = "N/A"


    # OCR
    if use_ocr:
        ocr_res = run_ocr_validation(test_path)
        results["ocr_valid"] = ocr_res["overall_ocr_valid"]
        results["aadhaar_number"] = ocr_res["aadhaar_number"]
        results["raw_text"] = ocr_res["raw_text"]
    else:
        results["ocr_valid"] = True


    # Verhoeff
    if use_verhoeff:
        aadhaar_num = results.get("aadhaar_number") or "123456789012"
        v = check_id_integrity(aadhaar_num)
        results["verhoeff_valid"] = v["is_valid"]
    else:
        results["verhoeff_valid"] = True


    # CNN
    if use_cnn:
        cnn_res = predict_image(test_path)
        results["cnn_verdict"] = cnn_res["verdict"]
        results["cnn_probability"] = cnn_res["probability"]
    else:
        results["cnn_verdict"] = "Genuine"


    # Final verdict
    final = combine_verdicts(
        results["ssim_verdict"],
        results["ocr_valid"],
        results["verhoeff_valid"],
        results["cnn_verdict"]
    )

    st.markdown("---")

    if final == "Genuine":
        st.success("✅ Aadhaar Card Appears Genuine")
    else:
        st.error("❌ Aadhaar Card Appears Forged")


    # Show module results
    st.markdown("### Module Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("SSIM Score", results["ssim_score"])

    with col2:
        st.write("OCR Valid:", results["ocr_valid"])

    with col3:
        st.write("Verhoeff Valid:", results["verhoeff_valid"])

    with col4:
        st.write("CNN Verdict:", results["cnn_verdict"])


    # OCR text
    st.markdown("### Extracted OCR Text")
    st.text(results.get("raw_text", ""))


else:

    st.info("Upload an image or capture using camera to start detection.")