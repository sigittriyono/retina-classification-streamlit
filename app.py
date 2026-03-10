import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import onnxruntime as ort

st.set_page_config(page_title="Retina Classification", layout="centered")
st.title("Retina Disease Classification")
st.write("Upload gambar retina untuk melakukan klasifikasi penyakit.")

MODEL_PATH = "retina_model.onnx"

def download_model():
    url = "https://drive.google.com/uc?id=1O001N_8sAC4RQ8xV4g5sZtrdzFnttWP_"
    with st.spinner("Downloading model..."):
        output = gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    if output is None:
        st.error("❌ Gagal download model.")
        st.stop()

if not os.path.exists(MODEL_PATH):
    download_model()

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

try:
    session = load_model()
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: img})[0]

    classes = ["Normal", "Diabetic Retinopathy"]
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    result = classes[predicted_index]
    st.subheader("Prediction Result")
    st.success(f"**{result}** ({confidence:.2f}% confidence)")
