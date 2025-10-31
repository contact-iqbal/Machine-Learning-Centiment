# Analisis Sentimen Survei Kepuasan Siswa SMK
Proyek ini dibuat untuk memahami dasar **Data Science** dan **Machine Learning** melalui studi kasus analisis sentimen komentar siswa SMK.  

---
##  Struktur Folder

| Folder / File | Deskripsi |
|----------------|------------|
| `data/raw/` | Dataset mentah (`Survey Kepuasan SMK_Train.csv`, `Survey Kepuasan SMK_Test.csv`) |
| `data/processed/` | Dataset setelah dibersihkan dan siap dipakai model |
| `src/preprocessing.py` | Script pembersihan teks (lowercase, stopword, stemming) |
| `src/train_model.py` | Melatih model TF-IDF + Logistic Regression |
| `src/evaluate_model.py` | Menguji performa model menggunakan data test |
| `model/` | Menyimpan model dan vectorizer dalam format `.joblib` |
| `app/app.py` | Aplikasi Streamlit sederhana untuk uji prediksi |
| `reports/` | Hasil evaluasi model (confusion matrix, laporan akurasi) |
| `requirements.txt` | Daftar library yang digunakan |
| `README.md` | Penjelasan proyek ini |

---

##  Alur Kerja Proyek

| Langkah | File Terkait | Deskripsi Singkat |
|----------|---------------|------------------|
| 1️ | `src/preprocessing.py` | Bersihkan teks dari data mentah |
| 2️ | `src/train_model.py` | Lakukan TF-IDF dan latih model |
| 3️ | `src/evaluate_model.py` | Uji performa model pada data test |
| 4️ | `src/predict.py` | Coba prediksi kalimat baru |
| 5️ | `app/app.py` | Jalankan aplikasi interaktif Streamlit |

---

##  Library yang Digunakan

| Library | Fungsi Utama |
|----------|---------------|
| `pandas` | Membaca dan mengolah data CSV |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluasi model |
| `Sastrawi` | Stemming Bahasa Indonesia |
| `joblib` | Menyimpan dan memuat model |
| `streamlit` | Membuat aplikasi web sederhana untuk prediksi |

---

##  Dataset
Proyek menggunakan file utama:

| File | Fungsi | Deskripsi |
|------|---------|------------|
| `Survey Kepuasan SMK_Train.csv` | Training Data | Model belajar pola kata dan sentimen |

---

##  Cara Menjalankan Proyek

```bash
# 1. Instal dependensi
pip install -r requirements.txt

# 2. run preprocessing.py
python src/controller/preprocessing.py

# 3. run training model
python src/controller/train_model.py

# 4. run evaluate
python src/controller/evaluate_model.py

# 5. run Streamlit
streamlit run src/view/app.py
```

---