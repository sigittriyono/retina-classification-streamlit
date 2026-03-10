readme = """
# Retina Disease Classification

Aplikasi Streamlit untuk klasifikasi penyakit retina menggunakan model Deep Learning.

## Cara Menjalankan

Install dependencies:

pip install -r requirements.txt

Jalankan aplikasi:

streamlit run app.py

## Fitur
- Upload gambar retina
- Prediksi penyakit retina
- Menampilkan hasil klasifikasi
"""

with open("README.md","w") as f:
    f.write(readme)
