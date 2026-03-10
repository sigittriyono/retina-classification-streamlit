import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import onnxruntime as ort

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaAI — Ocular Disease Classifier",
    page_icon="👁️",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f0f4f8;
    color: #1a2733;
  }

  /* Header */
  .header-container {
    background: linear-gradient(135deg, #0a3d62 0%, #1565a8 60%, #1e8bc3 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 8px 32px rgba(21,101,168,0.18);
  }
  .header-icon { font-size: 3rem; }
  .header-title {
    font-size: 2rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .header-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.72);
    margin-top: 4px;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border-radius: 14px;
    padding: 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    border: 1px solid #e2eaf2;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6b8caa;
    margin-bottom: 14px;
  }

  /* Prediction result */
  .prediction-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0a3d62, #1565a8);
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    padding: 10px 24px;
    border-radius: 10px;
    letter-spacing: 1px;
    margin: 8px 0;
  }
  .confidence-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.6rem;
    font-weight: 500;
    color: #1565a8;
    line-height: 1;
  }
  .confidence-label {
    font-size: 0.8rem;
    color: #6b8caa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }

  /* Confidence bar */
  .conf-bar-bg {
    background: #e2eaf2;
    border-radius: 8px;
    height: 10px;
    margin-top: 10px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1565a8, #1e8bc3);
    transition: width 0.6s ease;
  }

  /* Disease info */
  .disease-info {
    background: #f0f7ff;
    border-left: 4px solid #1565a8;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 0.88rem;
    color: #1a2733;
    line-height: 1.6;
  }

  /* Upload zone */
  .upload-hint {
    text-align: center;
    color: #6b8caa;
    font-size: 0.9rem;
    padding: 8px 0 4px;
  }

  /* Section divider */
  .divider {
    height: 1px;
    background: #e2eaf2;
    margin: 20px 0;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  .stFileUploader > div { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Disease Info ────────────────────────────────────────────────────────────────
CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

DISEASE_INFO = {
    "AMD":    ("Age-Related Macular Degeneration",  "Degenerasi makula terkait usia yang menyerang bagian tengah retina. Dapat menyebabkan kehilangan penglihatan pusat secara bertahap."),
    "CNV":    ("Choroidal Neovascularization",       "Pertumbuhan pembuluh darah baru abnormal di bawah retina. Sering merupakan komplikasi AMD basah."),
    "CSR":    ("Central Serous Retinopathy",         "Penumpukan cairan di bawah makula yang menyebabkan penglihatan kabur atau terdistorsi. Sering terkait dengan stres."),
    "DME":    ("Diabetic Macular Edema",             "Pembengkakan makula akibat kebocoran pembuluh darah pada penderita diabetes. Penyebab utama kebutaan pada diabetisi."),
    "DR":     ("Diabetic Retinopathy",               "Komplikasi diabetes yang merusak pembuluh darah retina. Dapat berkembang tanpa gejala pada stadium awal."),
    "DRUSEN": ("Drusen Deposits",                    "Endapan kuning kecil di bawah retina. Penanda awal AMD dan dapat meningkatkan risiko kehilangan penglihatan."),
    "MH":     ("Macular Hole",                       "Lubang kecil pada makula yang menyebabkan penglihatan pusat kabur atau hilang. Umumnya terjadi pada lansia."),
    "NORMAL": ("Normal Retina",                      "Tidak ditemukan tanda-tanda kelainan patologis pada retina. Kondisi retina dalam batas normal."),
}

# ─── Model ───────────────────────────────────────────────────────────────────────
MODEL_PATH = "retina_model.onnx"

def download_model():
    url = "https://drive.google.com/uc?id=1O001N_8sAC4RQ8xV4g5sZtrdzFnttWP_"  # ← ganti ID
    with st.spinner("⏳ Mengunduh model..."):
        output = gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    if output is None or not os.path.exists(MODEL_PATH):
        st.error("❌ Gagal mengunduh model. Periksa koneksi dan pastikan file Google Drive bersifat publik.")
        st.stop()

if not os.path.exists(MODEL_PATH):
    download_model()

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

try:
    session = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ─── Grad-CAM (ONNX approximation via occlusion sensitivity) ────────────────────
def compute_gradcam_occlusion(session, img_array, patch_size=56, stride=28):
    """
    Occlusion-based saliency map sebagai pengganti Grad-CAM untuk model ONNX.
    Menutup area gambar secara bertahap dan mengukur perubahan prediksi.
    """
    input_name = session.get_inputs()[0].name
    base_pred = session.run(None, {input_name: img_array})[0][0]
    pred_class = int(np.argmax(base_pred))
    base_score = base_pred[pred_class]

    h, w = 224, 224
    saliency = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            occluded = img_array.copy()
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            occluded[0, y:y2, x:x2, :] = 0.5
            occ_pred = session.run(None, {input_name: occluded})[0][0]
            drop = base_score - occ_pred[pred_class]
            saliency[y:y2, x:x2] += drop
            count_map[y:y2, x:x2] += 1

    count_map = np.maximum(count_map, 1)
    saliency = saliency / count_map
    saliency = np.maximum(saliency, 0)
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency

def overlay_gradcam(original_pil, saliency_map):
    orig = np.array(original_pil.resize((224, 224)))
    heatmap = cv2.resize(saliency_map, (224, 224))
    heatmap_color = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig, 0.55, heatmap_color, 0.45, 0)
    return Image.fromarray(overlay)

# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
  <div class="header-icon">👁️</div>
  <div>
    <div class="header-title">RetinaAI</div>
    <div class="header-sub">Ocular Disease Classification · 8 Categories · Grad-CAM Visualization</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload Citra Retina (OCT)</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Format: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar retina", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Gambar Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if uploaded_file:
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB → BGR
        img_array -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # Predict
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: img_array})[0]
        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        pred_class = CLASSES[pred_index]
        full_name, description = DISEASE_INFO[pred_class]

        # ── Result Card ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Hasil Prediksi</div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f'<div class="prediction-badge">{pred_class}</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.82rem;color:#6b8caa;margin-top:6px'>{full_name}</div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="confidence-value">{confidence*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="conf-bar-bg">
              <div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="disease-info">ℹ️ {description}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Grad-CAM Card ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Grad-CAM Visualization</div>', unsafe_allow_html=True)

        with st.spinner("Menghitung saliency map..."):
            saliency = compute_gradcam_occlusion(session, img_array)
            cam_image = overlay_gradcam(image, saliency)

        g1, g2 = st.columns(2)
        with g1:
            st.image(image.resize((224, 224)), caption="Original (224×224)", use_container_width=True)
        with g2:
            st.image(cam_image, caption="Grad-CAM Overlay", use_container_width=True)

        # Colorbar legend
        fig, ax = plt.subplots(figsize=(5, 0.35))
        fig.patch.set_facecolor('#ffffff')
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap='jet')
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['Rendah', 'Sedang', 'Tinggi'], fontsize=7, color='#6b8caa')
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        fig.tight_layout(pad=0.1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div style="font-size:0.78rem;color:#6b8caa;margin-top:6px">🔴 Area merah = region paling berpengaruh terhadap prediksi</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px 28px;color:#6b8caa">
          <div style="font-size:3rem;margin-bottom:16px">🔬</div>
          <div style="font-size:1rem;font-weight:500;color:#1a2733">Belum ada gambar</div>
          <div style="font-size:0.85rem;margin-top:8px">Upload citra OCT retina di panel kiri untuk memulai analisis</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.78rem;color:#a0b4c4;padding:12px 0">
  RetinaAI · Klasifikasi 8 Penyakit Retina · Untuk keperluan akademik
</div>
""", unsafe_allow_html=True)
