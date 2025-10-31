# 🧠 Analisis Sentimen Survei Kepuasan Siswa SMK

## 🎯 Tujuan Proyek
Proyek ini dibuat untuk memahami dasar **Data Science** dan **Machine Learning** melalui studi kasus analisis sentimen komentar siswa SMK.  
Fokusnya bukan pada akurasi tinggi, melainkan memahami alur kerja *end-to-end* mulai dari data mentah hingga aplikasi sederhana.

---

## 📂 Struktur Folder

| Folder / File | Deskripsi |
|----------------|------------|
| `data/raw/` | Dataset mentah (`Survey Kepuasan SMK_Train.csv`, `Survey Kepuasan SMK_Test.csv`) |
| `data/processed/` | Dataset setelah dibersihkan dan siap dipakai model |
| `src/preprocessing.py` | Script pembersihan teks (lowercase, stopword, stemming) |
| `src/train_model.py` | Melatih model TF-IDF + Logistic Regression |
| `src/evaluate_model.py` | Menguji performa model menggunakan data test |
| `src/predict.py` | Memprediksi teks baru menggunakan model tersimpan |
| `model/` | Menyimpan model dan vectorizer dalam format `.joblib` |
| `app/app.py` | Aplikasi Streamlit sederhana untuk uji prediksi |
| `reports/` | Hasil evaluasi model (confusion matrix, laporan akurasi) |
| `requirements.txt` | Daftar library yang digunakan |
| `README.md` | Penjelasan proyek ini |

---

## ⚙️ Alur Kerja Proyek

| Langkah | File Terkait | Deskripsi Singkat |
|----------|---------------|------------------|
| 1️⃣ | `src/preprocessing.py` | Bersihkan teks dari data mentah |
| 2️⃣ | `src/train_model.py` | Lakukan TF-IDF dan latih model |
| 3️⃣ | `src/evaluate_model.py` | Uji performa model pada data test |
| 4️⃣ | `src/predict.py` | Coba prediksi kalimat baru |
| 5️⃣ | `app/app.py` | Jalankan aplikasi interaktif Streamlit |

---

## 📦 Library yang Digunakan

| Library | Fungsi Utama |
|----------|---------------|
| `pandas` | Membaca dan mengolah data CSV |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluasi model |
| `Sastrawi` | Stemming Bahasa Indonesia |
| `joblib` | Menyimpan dan memuat model |
| `streamlit` | Membuat aplikasi web sederhana untuk prediksi |

---

## 🧩 Dataset
Proyek menggunakan dua file utama:

| File | Fungsi | Deskripsi |
|------|---------|------------|
| `Survey Kepuasan SMK_Train.csv` | Training Data | Model belajar pola kata dan sentimen |
| `Survey Kepuasan SMK_Test.csv` | Testing Data | Menguji performa model pada data baru |

---

## 🚀 Cara Menjalankan Proyek

```bash
# 1. Instal dependensi
pip install -r requirements.txt

# 2. Jalankan training model
python src/train_model.py

# 3. Jalankan evaluasi
python src/evaluate_model.py

# 4. Jalankan aplikasi Streamlit
streamlit run app/app.py
```

---

## 📊 Output yang Dihasilkan

| Jenis Output | Lokasi | Keterangan |
|---------------|---------|-------------|
| Model Terlatih | `model/sentiment_model.joblib` | Model klasifikasi sentimen |
| TF-IDF Vectorizer | `model/tfidf_vectorizer.joblib` | Objek pembobotan kata |
| Confusion Matrix | `reports/confusion_matrix.png` | Visualisasi performa model |
| Laporan Akurasi | `reports/evaluation_report.txt` | Ringkasan metrik model |
| Aplikasi Streamlit | `app/app.py` | UI sederhana untuk uji coba prediksi |

---

## 📚 Tujuan Pembelajaran
- Mengenal konsep **sentimen positif / negatif**.  
- Memahami proses **pembersihan data teks**.  
- Mengetahui cara kerja **TF-IDF dan model ML sederhana**.  
- Melihat bagaimana model diuji dan digunakan secara nyata.  

---

> 💡 Proyek ini ditujukan untuk **kelas 11 semester 2 SMK**, sebagai pengantar praktis pemrosesan teks dan pembelajaran mesin dalam konteks sederhana.
