import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import time
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ssim_module import check_ssim
from modules.ocr_module import run_ocr_validation
from modules.verhoeff import check_id_integrity
from modules.mobilenet_model import predict_image
from utils.helper import combine_verdicts


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Forgery Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — Light Techy Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #EDF2F7;
    --surface:     #FFFFFF;
    --surface2:    #F0F4FA;
    --border:      #C8D6E8;
    --accent:      #0057FF;
    --accent2:     #00C2A8;
    --danger:      #E53E3E;
    --success:     #16A34A;
    --text:        #0D1B2A;
    --text-muted:  #5A6A7E;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'Syne', sans-serif;
    --radius:      12px;
    --shadow:      0 4px 24px rgba(0,87,255,0.08);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1.5px solid var(--border);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0057FF 0%, #0099FF 60%, #00C2A8 100%);
    border-radius: var(--radius);
    padding: 40px 48px 36px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,87,255,0.22);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(90deg, rgba(255,255,255,.04) 0px, rgba(255,255,255,.04) 1px, transparent 1px, transparent 48px),
        repeating-linear-gradient(0deg, rgba(255,255,255,.04) 0px, rgba(255,255,255,.04) 1px, transparent 1px, transparent 48px);
    pointer-events: none;
}
.hero-title {
    font-family: var(--sans);
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
    margin: 0 0 8px;
}
.hero-sub {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: rgba(255,255,255,0.72);
    letter-spacing: 0.12em;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: #fff;
    letter-spacing: 0.1em;
    margin-bottom: 18px;
}

/* ── Module toggle cards in sidebar ── */
.stCheckbox label {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px !important;
    font-family: var(--mono);
    font-size: 0.82rem;
    cursor: pointer;
    transition: all .18s;
    display: flex !important;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px !important;
}
.stCheckbox label:hover {
    border-color: var(--accent);
    background: #EAF0FF;
    box-shadow: 0 0 0 3px rgba(0,87,255,0.08);
}

/* ── Input section card ── */
.input-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
    transition: border-color .2s, box-shadow .2s;
}
.input-card:hover {
    border-color: var(--accent);
    box-shadow: 0 6px 32px rgba(0,87,255,0.12);
}
.section-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Scan button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0057FF, #0099FF) !important;
    color: #fff !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all .22s !important;
    box-shadow: 0 4px 16px rgba(0,87,255,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,87,255,0.38) !important;
    background: linear-gradient(90deg, #0040CC, #0082E6) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Verdict cards ── */
.verdict-genuine {
    background: linear-gradient(135deg, #F0FFF4 0%, #DCFCE7 100%);
    border: 2px solid var(--success);
    border-radius: var(--radius);
    padding: 36px 40px;
    text-align: center;
    box-shadow: 0 6px 32px rgba(22,163,74,0.15);
    animation: popIn .4s cubic-bezier(.34,1.56,.64,1);
}
.verdict-forged {
    background: linear-gradient(135deg, #FFF5F5 0%, #FEE2E2 100%);
    border: 2px solid var(--danger);
    border-radius: var(--radius);
    padding: 36px 40px;
    text-align: center;
    box-shadow: 0 6px 32px rgba(229,62,62,0.15);
    animation: popIn .4s cubic-bezier(.34,1.56,.64,1);
}
.verdict-icon { font-size: 3.5rem; margin-bottom: 12px; }
.verdict-title {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.verdict-genuine .verdict-title { color: var(--success); }
.verdict-forged  .verdict-title { color: var(--danger); }
.verdict-sub {
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    opacity: .65;
}

@keyframes popIn {
    from { opacity: 0; transform: scale(.92); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Progress / scan animation ── */
.scan-block {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 28px;
    margin-bottom: 24px;
}
.scan-row {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    padding: 6px 0;
}
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}
.dot.running  { background: var(--accent); animation: pulse 1s infinite; }
.dot.done-ok  { background: var(--success); }
.dot.done-err { background: var(--danger); }
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:.3; }
}

/* ── Parameter cards ── */
.param-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 20px 0 8px;
}
.param-card {
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
    transition: transform .18s, box-shadow .18s, border-color .18s;
    cursor: default;
}
.param-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,87,255,0.12);
    border-color: var(--accent);
}
.param-card.pass {
    border-color: #86EFAC;
    background: #F0FFF4;
}
.param-card.fail {
    border-color: #FCA5A5;
    background: #FFF5F5;
}
.param-card.skip {
    border-color: var(--border);
    background: var(--surface2);
    opacity: .7;
}
.param-icon { font-size: 1.5rem; margin-bottom: 6px; }
.param-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.param-value {
    font-family: var(--sans);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text);
}
.param-card.pass .param-value { color: var(--success); }
.param-card.fail .param-value { color: var(--danger); }

/* ── Image preview ── */
.img-preview {
    border-radius: var(--radius);
    border: 1.5px solid var(--border);
    overflow: hidden;
    box-shadow: var(--shadow);
}

/* ── Sidebar labels ── */
.sidebar-title {
    font-family: var(--sans);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.sidebar-hint {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}

/* ── Radio ── */
.stRadio label {
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    transition: color .15s;
}
.stRadio label:hover { color: var(--accent) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface2) !important;
    transition: border-color .2s, background .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: #EAF0FF !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">▸ FORENSIC ANALYSIS SYSTEM v2.0</div>
    <div class="hero-title">🔬 Document Forgery Detector</div>
    <div class="hero-sub">MULTI-LAYER DOCUMENT AUTHENTICATION · OCR · VERHOEFF · CNN CLASSIFIER</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title">⚙ Analysis Modules</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-hint">TOGGLE FORENSIC LAYERS</div>', unsafe_allow_html=True)

use_ssim      = st.sidebar.checkbox("🖼  SSIM Similarity", True)
use_ocr       = st.sidebar.checkbox("🔤  OCR Text Extraction", True)
use_verhoeff  = st.sidebar.checkbox("🔢  Verhoeff Checksum", True)
use_cnn       = st.sidebar.checkbox("🧠  CNN Deep Classifier", True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-title">📄 Template</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-hint">OPTIONAL REFERENCE IMAGE</div>', unsafe_allow_html=True)

template_file = st.sidebar.file_uploader("Upload Template", type=["jpg","png","jpeg"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#5A6A7E;line-height:1.8;">
SYS STATUS: <span style="color:#16A34A">■ ONLINE</span><br>
MODULES LOADED: 3 / 3<br>
ENGINE: MobileNetV2<br>
OCR: Tesseract 5.x
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TOP: Input (left) + Image Preview (right)
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

image = None

with col_left:
    st.markdown('<div class="section-label">▸ INPUT SOURCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    input_method = st.radio(
        "Input method",
        ["📁  Upload Image", "📷  Use Camera"],
        label_visibility="collapsed"
    )

    if "Upload" in input_method:
        uploaded_file = st.file_uploader(
            "Drop Aadhaar image here",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image = np.array(Image.open(uploaded_file).convert("RGB"))
    else:
        camera_image = st.camera_input("Capture", label_visibility="collapsed")
        if camera_image:
            image = np.array(Image.open(camera_image).convert("RGB"))

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">▸ IMAGE PREVIEW</div>', unsafe_allow_html=True)
    if image is not None:
        st.markdown('<div class="img-preview">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background:#F0F4FA;
            border:1.5px dashed #C8D6E8;
            border-radius:12px;
            padding:52px 40px;
            text-align:center;
            color:#5A6A7E;
            font-family:'IBM Plex Mono',monospace;
            font-size:0.78rem;
            letter-spacing:0.1em;
        ">
            <div style="font-size:2.2rem;margin-bottom:10px;">🪪</div>
            NO IMAGE LOADED<br>
            <span style="opacity:.5;font-size:0.68rem;">UPLOAD OR CAPTURE TO PREVIEW</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MIDDLE: Run button — full width
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if image is not None:
    run_btn = st.button("⚡  RUN FORENSIC ANALYSIS")
else:
    st.info("Upload or capture an Aadhaar card image to begin.")
    run_btn = False

st.markdown("---")

# ─────────────────────────────────────────────
# BOTTOM: Analysis — full width
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">▸ ANALYSIS OUTPUT</div>', unsafe_allow_html=True)

if image is not None and run_btn:

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        test_path = tmp.name

    template_path = None
    if template_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            tmp2.write(template_file.read())
            template_path = tmp2.name

    results = {}

    # ── Scan progress block ──
    st.markdown('<div class="scan-block">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">RUNNING MODULES</div>', unsafe_allow_html=True)

    p_ssim     = st.empty()
    p_ocr      = st.empty()
    p_verhoeff = st.empty()
    p_cnn      = st.empty()

    def row(placeholder, dot_class, label, status=""):
        placeholder.markdown(
            f'<div class="scan-row"><div class="dot {dot_class}"></div>{label} &nbsp;<span style="color:#0057FF">{status}</span></div>',
            unsafe_allow_html=True
        )

    row(p_ssim,     "" if not (use_ssim and template_path) else "running",
        "SSIM Structural Similarity", "waiting" if not (use_ssim and template_path) else "scanning...")
    row(p_ocr,      "running", "OCR Text Extraction", "scanning...")
    row(p_verhoeff, "",        "Verhoeff Checksum",   "waiting")
    row(p_cnn,      "",        "CNN Classifier",      "waiting")

    # ── SSIM ──
    if use_ssim and template_path:
        try:
            ssim_res = check_ssim(test_path, template_path)
            results["ssim_verdict"] = ssim_res["verdict"]
            results["ssim_score"]   = round(ssim_res["ssim_score"], 4)
            row(p_ssim, "done-ok", "SSIM Structural Similarity", "✓ complete")
        except Exception as e:
            results["ssim_verdict"] = "Unknown"
            results["ssim_score"]   = "Error"
            row(p_ssim, "done-err", "SSIM Structural Similarity", "✗ error")
    else:
        results["ssim_verdict"] = "Skipped"
        results["ssim_score"]   = "N/A"
        msg = "skipped — no template" if use_ssim else "skipped"
        row(p_ssim, "done-ok", "SSIM Structural Similarity", msg)

    # ── OCR ──
    if use_ocr:
        try:
            ocr_res = run_ocr_validation(test_path)
            results["ocr_valid"]      = ocr_res["overall_ocr_valid"]
            results["aadhaar_number"] = ocr_res["aadhaar_number"]
            results["raw_text"]       = ocr_res["raw_text"]
            row(p_ocr, "done-ok", "OCR Text Extraction", "✓ complete")
        except:
            results["ocr_valid"]      = False
            results["aadhaar_number"] = None
            results["raw_text"]       = ""
            row(p_ocr, "done-err", "OCR Text Extraction", "✗ failed")
    else:
        results["ocr_valid"]      = True
        results["aadhaar_number"] = None
        results["raw_text"]       = ""
        row(p_ocr, "done-ok", "OCR Text Extraction", "skipped")

    # ── Verhoeff ──
    row(p_verhoeff, "running", "Verhoeff Checksum", "validating...")
    time.sleep(0.3)

    if use_verhoeff:
        aadhaar_num = results.get("aadhaar_number")
        if aadhaar_num:
            v = check_id_integrity(aadhaar_num)
            results["verhoeff_valid"] = v["is_valid"]
            row(p_verhoeff, "done-ok", "Verhoeff Checksum", "✓ complete")
        else:
            results["verhoeff_valid"] = False
            row(p_verhoeff, "done-err", "Verhoeff Checksum", "✗ no ID found")
    else:
        results["verhoeff_valid"] = True
        row(p_verhoeff, "done-ok", "Verhoeff Checksum", "skipped")

    # ── CNN ──
    row(p_cnn, "running", "CNN Classifier", "inferencing...")
    time.sleep(0.3)

    if use_cnn:
        try:
            cnn_res = predict_image(test_path)
            results["cnn_verdict"]     = cnn_res["verdict"]
            results["cnn_probability"] = cnn_res["probability"]
            row(p_cnn, "done-ok", "CNN Classifier", "✓ complete")
        except:
            results["cnn_verdict"]     = "Error"
            results["cnn_probability"] = None
            row(p_cnn, "done-err", "CNN Classifier", "✗ error")
    else:
        results["cnn_verdict"]     = "Skipped"
        results["cnn_probability"] = None
        row(p_cnn, "done-ok", "CNN Classifier", "skipped")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Parameter result cards — native st.columns (avoids CSS class rendering issues) ──
    st.markdown("#### 🧪 Module Results")

    cnn_verdict  = results.get("cnn_verdict", "Skipped")
    cnn_prob     = results.get("cnn_probability")
    cnn_display  = cnn_verdict + (f" ({cnn_prob:.0%})" if cnn_prob is not None else "")
    ocr_display  = "Valid" if results.get("ocr_valid") else ("Skipped" if not use_ocr else "Invalid")
    ver_display  = "Valid" if results.get("verhoeff_valid") else ("Skipped" if not use_verhoeff else "Invalid")
    ssim_score   = results.get("ssim_score", "N/A")
    ssim_verdict = results.get("ssim_verdict", "Skipped")
    ssim_display = f"{ssim_score}" if ssim_score != "N/A" else "No template"

    # Build card list — include SSIM only if template was provided
    cards = [
        ("🔤", "OCR Check",         ocr_display,  results.get("ocr_valid"),         not use_ocr),
        ("🔢", "Verhoeff Checksum",  ver_display,  results.get("verhoeff_valid"),     not use_verhoeff),
        ("🧠", "CNN Classifier",     cnn_display,  cnn_verdict == "Genuine",          cnn_verdict in ("Skipped", "Error")),
    ]
    if use_ssim and template_path:
        cards.append(("🖼", "SSIM Similarity", ssim_display, ssim_verdict == "Genuine", ssim_verdict == "Skipped"))

    cols = st.columns(len(cards), gap="small")

    for col, (icon, label, value, is_pass, is_skip) in zip(cols, cards):
        if is_skip:
            bg, border, val_color, badge = "#F0F4FA", "#C8D6E8", "#5A6A7E", "⬜ SKIPPED"
        elif is_pass:
            bg, border, val_color, badge = "#F0FFF4", "#86EFAC", "#16A34A", "✅ PASS"
        else:
            bg, border, val_color, badge = "#FFF5F5", "#FCA5A5", "#E53E3E", "❌ FAIL"

        with col:
            st.markdown(f"""
            <div style="
                background:{bg};
                border:2px solid {border};
                border-radius:12px;
                padding:22px 14px 18px;
                text-align:center;
            ">
                <div style="font-size:1.9rem;margin-bottom:8px;">{icon}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.14em;color:#5A6A7E;text-transform:uppercase;
                    margin-bottom:10px;">{label}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                    font-weight:700;color:{val_color};margin-bottom:8px;">{value}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    color:{val_color};opacity:.85;">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Final verdict — full width ──
    final = combine_verdicts(
        results["ocr_valid"],
        results["verhoeff_valid"],
        results["cnn_verdict"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if final == "Genuine":
        st.markdown("""
        <div class="verdict-genuine">
            <div class="verdict-icon">✅</div>
            <div class="verdict-title">GENUINE</div>
            <div class="verdict-sub">DOCUMENT AUTHENTICATED · ALL CHECKS PASSED</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict-forged">
            <div class="verdict-icon">🚫</div>
            <div class="verdict-title">FORGED</div>
            <div class="verdict-sub">DOCUMENT REJECTED · ANOMALIES DETECTED</div>
        </div>
        """, unsafe_allow_html=True)

    # Cleanup
    try:
        os.remove(test_path)
        if template_path:
            os.remove(template_path)
    except:
        pass

elif image is None:
    st.markdown("""
    <div style="
        background:#F0F4FA;
        border:1.5px dashed #C8D6E8;
        border-radius:12px;
        padding:60px 40px;
        text-align:center;
        color:#5A6A7E;
        font-family:'IBM Plex Mono',monospace;
        font-size:0.8rem;
        letter-spacing:0.1em;
    ">
        <div style="font-size:2.5rem;margin-bottom:12px;">🔬</div>
        AWAITING IMAGE INPUT<br>
        <span style="opacity:.5;font-size:0.7rem;">RESULTS WILL APPEAR HERE</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="
        background:#F0F4FA;
        border:1.5px dashed #C8D6E8;
        border-radius:12px;
        padding:60px 40px;
        text-align:center;
        color:#5A6A7E;
        font-family:'IBM Plex Mono',monospace;
        font-size:0.8rem;
        letter-spacing:0.1em;
    ">
        <div style="font-size:2rem;margin-bottom:12px;">⚡</div>
        CLICK "RUN FORENSIC ANALYSIS"<br>
        <span style="opacity:.5;font-size:0.7rem;">TO START DOCUMENT VERIFICATION</span>
    </div>
    """, unsafe_allow_html=True)
