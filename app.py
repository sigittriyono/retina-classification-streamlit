import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Retina Classification", layout="centered")

st.title("Retina Disease Classification")
st.write("Upload gambar retina untuk melakukan klasifikasi penyakit.")

MODEL_PATH = "retina_model.h5"

# download model jika belum ada
def download_model():
    url = "https://drive.google.com/uc?id=1XsQi3KnKMmYAF2k-NURdMRYzX0D2lVDd"
    gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    download_model()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    classes = ["Normal","Diabetic Retinopathy"]

    result = classes[np.argmax(prediction)]

    st.subheader("Prediction Result")
    st.success(result)
