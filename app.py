import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

st.set_page_config(page_title="Retina Classification", layout="centered")
st.title("Retina Disease Classification")
st.write("Upload gambar retina untuk melakukan klasifikasi penyakit.")

MODEL_PATH = "retina_model.tflite"

def download_model():
    # Ganti dengan ID file .tflite yang baru
    url = "https://drive.google.com/uc?id=1OHR5bZpfFVVLRRCfCLkeoNaVkAadighm"
    with st.spinner("Downloading model..."):
        output = gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    if output is None:
        st.error("❌ Gagal download model.")
        st.stop()

if not os.path.exists(MODEL_PATH):
    download_model()

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
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

    # Inference dengan TFLite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    classes = ["Normal", "Diabetic Retinopathy"]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    result = classes[predicted_index]
    st.subheader("Prediction Result")
    st.success(f"**{result}** ({confidence:.2f}% confidence)")
