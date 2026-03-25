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
    page_title="RetinaAI — OCT Disease Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

  :root {
    --bg:        #05080f;
    --bg2:       #0c1220;
    --bg3:       #111926;
    --border:    rgba(255,255,255,0.07);
    --accent:    #00d4ff;
    --accent2:   #0099cc;
    --gold:      #f5c842;
    --green:     #00e5a0;
    --red:       #ff5a5a;
    --text:      #e8f0f7;
    --muted:     #5a7a96;
    --card-glow: rgba(0,212,255,0.06);
  }

  html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
  }

  .stApp { background: var(--bg) !important; }
  section[data-testid="stSidebar"] { display: none; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Hero ── */
  .hero {
    position: relative;
    padding: 40px 48px;
    margin-bottom: 32px;
    border-radius: 20px;
    background: linear-gradient(135deg, #071828 0%, #0a1f35 50%, #071828 100%);
    border: 1px solid rgba(0,212,255,0.2);
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(245,200,66,0.06) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
    opacity: 0.9;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 8px 0;
    line-height: 1.1;
  }
  .hero-title span { color: var(--accent); }
  .hero-sub {
    font-size: 0.92rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 500px;
    line-height: 1.6;
  }
  .hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
  }
  .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    letter-spacing: 0.5px;
  }
  .badge.active {
    border-color: rgba(0,212,255,0.4);
    color: var(--accent);
    background: rgba(0,212,255,0.06);
  }

  /* ── Panels ── */
  .panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
  }
  .panel:hover { border-color: rgba(0,212,255,0.15); }
  .panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Upload ── */
  .stFileUploader > div {
    background: var(--bg3) !important;
    border: 1.5px dashed rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s !important;
  }
  .stFileUploader > div:hover {
    border-color: rgba(0,212,255,0.5) !important;
  }

  /* ── Prediction grid ── */
  .pred-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .pred-class-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,153,204,0.04));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 18px 20px;
  }
  .pred-class-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .pred-class-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.5px;
    line-height: 1;
  }
  .pred-class-full {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 6px;
    font-weight: 300;
  }
  .pred-conf-box {
    background: linear-gradient(135deg, rgba(0,229,160,0.06), rgba(0,229,160,0.02));
    border: 1px solid rgba(0,229,160,0.15);
    border-radius: 12px;
    padding: 18px 20px;
  }
  .pred-conf-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--green);
    letter-spacing: -0.5px;
    line-height: 1;
  }
  .pred-conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .conf-track {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    margin-top: 12px;
    overflow: hidden;
  }
  .conf-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00e5a0, #00d4ff);
  }

  /* ── Info box ── */
  .info-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.84rem;
    color: #8fb3cc;
    line-height: 1.65;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .info-icon {
    color: var(--accent);
    font-size: 0.9rem;
    margin-top: 1px;
    flex-shrink: 0;
  }

  /* ── Probability bars ── */
  .prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
  }
  .prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    width: 54px;
    flex-shrink: 0;
    letter-spacing: 0.5px;
  }
  .prob-track {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
  }
  .prob-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, rgba(0,212,255,0.5), rgba(0,212,255,0.9));
  }
  .prob-fill.top {
    background: linear-gradient(90deg, #00e5a0, #00d4ff);
  }
  .prob-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
  }

  /* ── CAM legend ── */
  .cam-legend {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
  }
  .cam-legend-bar {
    flex: 1;
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00008b, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000);
  }
  .cam-legend-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.5px;
    flex-shrink: 0;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    padding: 20px 0 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 1px;
    opacity: 0.6;
  }

  img { border-radius: 10px; }
  .stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

DISEASE_INFO = {
    "AMD":    ("Age-Related Macular Degeneration",  "Degenerasi makula terkait usia yang menyerang bagian tengah retina. Dapat menyebabkan kehilangan penglihatan pusat secara bertahap."),
    "CNV":    ("Choroidal Neovascularization",       "Pertumbuhan pembuluh darah baru abnormal di bawah retina. Sering merupakan komplikasi AMD basah yang membutuhkan penanganan segera."),
    "CSR":    ("Central Serous Retinopathy",         "Penumpukan cairan di bawah makula yang menyebabkan penglihatan kabur atau terdistorsi. Sering berkorelasi dengan tingkat stres tinggi."),
    "DME":    ("Diabetic Macular Edema",             "Pembengkakan makula akibat kebocoran pembuluh darah pada penderita diabetes. Penyebab utama kebutaan pada pasien diabetisi."),
    "DR":     ("Diabetic Retinopathy",               "Komplikasi diabetes yang merusak pembuluh darah retina. Dapat berkembang tanpa gejala pada stadium awal — deteksi dini sangat penting."),
    "DRUSEN": ("Drusen Deposits",                    "Endapan kuning kecil di bawah retina. Merupakan penanda awal AMD dan dapat meningkatkan risiko kehilangan penglihatan jangka panjang."),
    "MH":     ("Macular Hole",                       "Lubang kecil pada makula yang menyebabkan penglihatan pusat kabur atau hilang. Umumnya terjadi pada lansia di atas 60 tahun."),
    "NORMAL": ("Normal Retina",                      "Tidak ditemukan tanda-tanda kelainan patologis pada retina. Kondisi retina dalam batas normal — tidak diperlukan tindakan klinis."),
}

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH = "retina_model.onnx"

def download_model():
    url = "https://drive.google.com/uc?id=1O001N_8sAC4RQ8xV4g5sZtrdzFnttWP_"
    with st.spinner("⏳ Mengunduh model ONNX..."):
        output = gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    if output is None or not os.path.exists(MODEL_PATH):
        st.error("❌ Gagal mengunduh model.")
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

# ─── Saliency Map ────────────────────────────────────────────────────────────
def compute_saliency(session, img_array, patch_size=56, stride=28):
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
            y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
            occluded[0, y:y2, x:x2, :] = 0.5
            occ_pred = session.run(None, {input_name: occluded})[0][0]
            drop = base_score - occ_pred[pred_class]
            saliency[y:y2, x:x2] += drop
            count_map[y:y2, x:x2] += 1
    count_map = np.maximum(count_map, 1)
    saliency /= count_map
    saliency = np.maximum(saliency, 0)
    saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency

def make_overlay(original_pil, saliency_map):
    orig = np.array(original_pil.resize((224, 224)))
    heatmap = cv2.resize(saliency_map, (224, 224))
    heatmap_color = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig, 0.52, heatmap_color, 0.48, 0)
    return Image.fromarray(overlay)

# ─── Validasi Retina ─────────────────────────────────────────────────────────
def is_valid_retina(probs, pil_image,
                    confidence_threshold=0.50,
                    entropy_threshold=1.8,
                    top2_gap_threshold=0.10):
    from scipy.stats import entropy as scipy_entropy

    max_conf = float(np.max(probs))
    if max_conf < confidence_threshold:
        return False, f"Confidence terlalu rendah ({max_conf*100:.1f}% < {confidence_threshold*100:.0f}%)"

    ent = float(scipy_entropy(probs))
    if ent > entropy_threshold:
        return False, f"Model tidak dapat mengenali pola retina (entropy={ent:.2f})"

    sorted_probs = np.sort(probs)[::-1]
    top2_gap = float(sorted_probs[0] - sorted_probs[1])
    if top2_gap < top2_gap_threshold:
        return False, f"Prediksi tidak meyakinkan — model ragu antar kelas (gap={top2_gap:.2f})"

    return True, "OK"

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">// Medical Imaging · Deep Learning</div>
  <div class="hero-title">Retina<span>AI</span></div>
  <div class="hero-sub">Klasifikasi penyakit retina berbasis citra OCT menggunakan Convolutional Neural Network + Grad-CAM Visualization</div>
  <div class="hero-badges">
    <span class="badge active">8 Kelas Penyakit</span>
    <span class="badge active">ResNet-50</span>
    <span class="badge active">Grad-CAM</span>
    <span class="badge active">OCT Imaging</span>
    <span class="badge active">Academic Use</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.5], gap="large")

# ══ LEFT ══
with col_left:
    st.markdown('<div class="panel"><div class="panel-label">01 · Upload Citra OCT</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload gambar retina OCT",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('<div style="font-size:0.75rem;color:#3d5a70;margin-top:8px;font-family:\'IBM Plex Mono\',monospace;">FORMAT: JPG · JPEG · PNG · MAX 200MB</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="panel"><div class="panel-label">02 · Preview Citra</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        w_img, h_img = image.size
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-top:12px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--muted);">
            <span style="color:#3d5a70">DIMENSI</span><br>{w_img} × {h_img} px
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--muted);">
            <span style="color:#3d5a70">FILE</span><br>{uploaded_file.name}
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--muted);">
            <span style="color:#3d5a70">MODE</span><br>RGB
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══ RIGHT ══
with col_right:
    if uploaded_file:
        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array[:, :, ::-1]
        img_array -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        input_name = session.get_inputs()[0].name
        raw_output = session.run(None, {input_name: img_array})[0]
        probs = raw_output[0]

        # ── Validasi gambar ──
        is_valid, reason = is_valid_retina(probs, image)  # ← tambah image
        if not is_valid:
            st.markdown(f"""
            <div class="panel" style="border-color:rgba(255,90,90,0.3);">
              <div class="panel-label" style="color:#ff5a5a;">⚠ · Gambar Tidak Valid</div>
              <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="font-size:2rem;opacity:0.6;">🚫</div>
                <div>
                  <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                              color:#ff5a5a;margin-bottom:6px;">
                    Bukan Citra Retina OCT
                  </div>
                  <div style="font-size:0.83rem;color:#5a7a96;line-height:1.6;">
                    Gambar yang diupload tidak terdeteksi sebagai citra retina OCT yang valid.<br>
                    <span style="color:#3d5a70;font-family:'IBM Plex Mono',monospace;
                                 font-size:0.75rem;">ALASAN: {reason}</span>
                  </div>
                  <div style="margin-top:14px;padding:12px 14px;background:rgba(255,90,90,0.05);
                              border:1px solid rgba(255,90,90,0.15);border-radius:8px;
                              font-size:0.78rem;color:#3d5a70;line-height:1.6;">
                    ◈ Pastikan gambar adalah citra OCT retina<br>
                    ◈ Format yang didukung: JPG, JPEG, PNG<br>
                    ◈ Hindari gambar foto biasa, selfie, atau objek non-medis
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        pred_index = int(np.argmax(probs))
        confidence = float(np.max(probs))
        pred_class = CLASSES[pred_index]
        full_name, description = DISEASE_INFO[pred_class]

        # ── Result ──

        # ── Result ──
        st.markdown('<div class="panel"><div class="panel-label">03 · Hasil Prediksi Model</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-grid">
          <div class="pred-class-box">
            <div class="pred-class-label">Diagnosed Class</div>
            <div class="pred-class-value" style="font-size:1.1rem;line-height:1.3;">{full_name}</div>
            <div class="pred-class-full">{pred_class}</div>
          </div>
          <div class="pred-conf-box">
            <div class="pred-conf-label">Confidence Score</div>
            <div class="pred-conf-num">{confidence*100:.1f}%</div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{confidence*100:.1f}%"></div>
            </div>
          </div>
        </div>
        <div class="info-box">
          <span class="info-icon">◈</span>
          <span>{description}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<div class="panel"><div class="panel-label">04 · Distribusi Probabilitas</div>', unsafe_allow_html=True)
        sorted_idx = np.argsort(probs)[::-1]
        for i, idx in enumerate(sorted_idx):
            cls = CLASSES[idx]
            val = float(probs[idx])
            is_top = (i == 0)
            fill_class = "top" if is_top else ""
            name_color = "color:var(--accent)" if is_top else ""
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-name" style="{name_color}">{cls}</div>
              <div class="prob-track">
                <div class="prob-fill {fill_class}" style="width:{val*100:.1f}%"></div>
              </div>
              <div class="prob-val" style="{name_color}">{val*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Grad-CAM ──
        st.markdown('<div class="panel"><div class="panel-label">05 · Grad-CAM Saliency Map</div>', unsafe_allow_html=True)
        with st.spinner("Menghitung saliency map..."):
            saliency = compute_saliency(session, img_array)
            cam_image = make_overlay(image, saliency)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(image.resize((224, 224)), caption="Original", use_container_width=True)
        with g2:
            heatmap_np = (cm.jet(saliency)[:, :, :3] * 255).astype(np.uint8)
            st.image(Image.fromarray(heatmap_np), caption="Heatmap", use_container_width=True)
        with g3:
            st.image(cam_image, caption="Overlay", use_container_width=True)

        st.markdown("""
        <div class="cam-legend">
          <div class="cam-legend-label">RENDAH</div>
          <div class="cam-legend-bar"></div>
          <div class="cam-legend-label">TINGGI</div>
        </div>
        <div style="font-size:0.73rem;color:var(--muted);margin-top:8px;font-family:'IBM Plex Mono',monospace;">
          ◈ Area merah = region paling berpengaruh terhadap prediksi model
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:72px 28px;">
          <div style="font-size:3.5rem;margin-bottom:20px;opacity:0.25">👁</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;color:#2a4a62;margin-bottom:8px;">
            Belum ada citra untuk dianalisis
          </div>
          <div style="font-size:0.82rem;color:#2a4a62;line-height:1.6;max-width:320px;margin:0 auto;">
            Upload citra OCT retina di panel kiri untuk memulai klasifikasi dan visualisasi Grad-CAM
          </div>
          <div style="margin-top:28px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#1a3040;letter-spacing:1px;">
            MENDUKUNG · AMD · CNV · CSR · DME · DR · DRUSEN · MH · NORMAL
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  RETINA AI &nbsp;·&nbsp; 8-CLASS OCT DISEASE CLASSIFIER &nbsp;·&nbsp; RESNET-50 + GRAD-CAM &nbsp;·&nbsp; ACADEMIC USE ONLY
</div>
""", unsafe_allow_html=True)
