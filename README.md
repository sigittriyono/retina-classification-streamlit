# RetinaAI — Ocular Disease Classification

Aplikasi web berbasis Streamlit yang digunakan untuk melakukan klasifikasi penyakit retina dari citra OCT (Optical Coherence Tomography). Model yang digunakan merupakan model deep learning berbasis ResNet50 dan dilengkapi dengan visualisasi Grad-CAM untuk membantu melihat bagian gambar yang paling berpengaruh terhadap prediksi model.

Project ini dibuat sebagai bagian dari implementasi model deep learning untuk klasifikasi citra medis sekaligus sebagai contoh deployment model ke dalam bentuk aplikasi web sederhana.

---

## Daftar Isi

* Demo
* Fitur
* Kategori Penyakit
* Arsitektur Model
* Instalasi Lokal
* Deploy ke Streamlit Cloud
* Kendala dan Solusi
* Struktur Proyek

---

## Demo

Aplikasi dapat dijalankan secara online melalui Streamlit Cloud.

[Open Streamlit App](https://your-app-url.streamlit.app)

---

## Fitur

Beberapa fitur yang tersedia pada aplikasi ini antara lain:

* Klasifikasi citra OCT retina ke dalam **8 kategori penyakit**
* Menampilkan **confidence score** dari setiap kelas prediksi
* Visualisasi **Grad-CAM** untuk melihat area gambar yang paling mempengaruhi prediksi model
* Tampilan antarmuka sederhana dengan gaya **medical dashboard**
* Informasi singkat mengenai setiap penyakit retina

---

## Kategori 8 Penyakit

Model yang digunakan mampu mengklasifikasikan citra retina ke dalam 8 kelas berikut:

| Kode   | Nama Penyakit                    |
| ------ | -------------------------------- |
| AMD    | Age-Related Macular Degeneration |
| CNV    | Choroidal Neovascularization     |
| CSR    | Central Serous Retinopathy       |
| DME    | Diabetic Macular Edema           |
| DR     | Diabetic Retinopathy             |
| DRUSEN | Drusen Deposits                  |
| MH     | Macular Hole                     |
| NORMAL | Retina Normal                    |

---

## Arsitektur Model

Model yang digunakan merupakan **ResNet50 pretrained ImageNet** yang kemudian dilakukan fine-tuning untuk klasifikasi citra retina.

Spesifikasi model:

* Base model : ResNet50
* Input size : 224 × 224 × 3
* Preprocessing : `preprocess_input` dari ResNet50
* Output : 8 kelas dengan aktivasi softmax
* Format deployment : **ONNX**

Penggunaan format ONNX dipilih karena lebih kompatibel dengan environment deployment di Streamlit Cloud dibandingkan TensorFlow.

---

## Instalasi Lokal

Jika ingin menjalankan aplikasi secara lokal, ikuti langkah berikut.

### 1. Clone Repository

```bash
git clone https://github.com/username/retina-classification-streamlit.git
cd retina-classification-streamlit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Setelah itu aplikasi akan berjalan di browser melalui alamat:

```
http://localhost:8501
```

---

## ☁️ Deploy ke Streamlit Cloud

Untuk menjalankan aplikasi secara online dapat menggunakan Streamlit Cloud.

Langkah-langkah:

1. Upload repository ke GitHub
2. Masuk ke Streamlit Cloud
3. Hubungkan dengan repository GitHub
4. Pilih file utama `app.py`
5. Klik deploy

Model disimpan di Google Drive dan akan otomatis diunduh saat aplikasi pertama kali dijalankan.

---

## Kendala dan Solusi

Selama proses pengembangan dan deployment aplikasi ini terdapat beberapa kendala yang muncul.

### 1. TensorFlow tidak bisa diinstall di Streamlit Cloud

Masalah ini muncul karena Streamlit Cloud menggunakan versi Python yang lebih baru sementara TensorFlow belum sepenuhnya mendukung versi tersebut.

Solusi yang digunakan adalah **mengubah format model dari `.h5` menjadi `.onnx`** sehingga proses inference dapat menggunakan `onnxruntime` yang lebih kompatibel.

---

### 2. Hasil prediksi selalu mengarah ke satu kelas

Masalah ini biasanya terjadi karena preprocessing gambar yang digunakan saat inference tidak sama dengan preprocessing yang digunakan saat training.

Model ResNet50 membutuhkan preprocessing khusus menggunakan `preprocess_input`, bukan sekadar normalisasi nilai piksel.

---

### 3. Grad-CAM berjalan sangat lambat

Grad-CAM dengan metode occlusion sensitivity membutuhkan banyak proses inference sehingga waktu komputasi menjadi cukup lama.

Untuk mengatasi hal ini digunakan ukuran patch yang lebih besar dan stride yang lebih lebar agar jumlah proses inference berkurang.

---

## Struktur Proyek

Struktur folder pada project ini adalah sebagai berikut:

```
retina-classification-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
└── retina_model.onnx
```

Penjelasan:

* **app.py** : script utama aplikasi Streamlit
* **requirements.txt** : daftar library yang dibutuhkan
* **README.md** : dokumentasi project
* **retina_model.onnx** : file model yang digunakan untuk inference

---

## Dependencies

Beberapa library utama yang digunakan dalam project ini:

| Library                | Fungsi                            |
| ---------------------- | --------------------------------- |
| streamlit              | framework untuk membuat web app   |
| onnxruntime            | menjalankan model ONNX            |
| gdown                  | mengunduh model dari Google Drive |
| Pillow                 | membaca dan memproses gambar      |
| numpy                  | operasi array                     |
| opencv-python-headless | membuat visualisasi heatmap       |
| matplotlib             | visualisasi tambahan              |

---

## Catatan

Project ini dibuat untuk keperluan pembelajaran dan tugas akademik terkait implementasi deep learning pada citra medis serta deployment model ke dalam bentuk aplikasi web sederhana.
