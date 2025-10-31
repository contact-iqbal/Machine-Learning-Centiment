# Analisis Sentimen Survei Kepuasan Siswa SMK
Proyek ini dibuat untuk memahami dasar **Data Science** dan **Machine Learning** melalui studi kasus analisis sentimen komentar siswa SMK.  

---

## 📁 Folder Structure

```
project/
├── data/
│   ├── raw/
│   │   └── Survey Kepuasan SMK_Train.csv
│   └── processed/
│       └── train_clean.csv
│
├── reports/
│   ├── confusion_matrix.png
│   ├── evaluation_report.txt
│   └── sample_prediction.csv
│
├── src/
│   ├── controller/
│   │   ├── preprocessing.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── predict.py
│   │
│   ├── model/
│   │   ├── sentiment_model.joblib
│   │   └── tfidf_vectorizer.joblib
│   │
│   └── view/
│       └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Project Flow

1. **`src/controller/preprocessing.py`**  
   Membersihkan teks dari data mentah agar siap digunakan untuk pelatihan.

2. **`src/controller/train_model.py`**  
   Melakukan proses TF-IDF dan melatih model untuk mengenali sentimen.

3. **`src/controller/evaluate_model.py`**  
   Menguji performa model menggunakan data uji untuk melihat tingkat akurasi.

4. **`src/controller/predict.py`**  
   Menggunakan model yang sudah dilatih untuk memprediksi sentimen dari kalimat baru.

5. **`src/view/app.py`**  
   Menjalankan aplikasi interaktif berbasis Streamlit untuk melakukan analisis sentimen secara visual.

---

## 🧩 Library Included

| Library | Fungsi Utama |
|----------|---------------|
| `pandas` | Membaca dan mengolah data CSV |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluasi model |
| `Sastrawi` | Stemming Bahasa Indonesia |
| `joblib` | Menyimpan dan memuat model |
| `streamlit` | Membuat aplikasi web sederhana untuk prediksi |

---

## 📄 Dataset

**File:** `Survey Kepuasan SMK_Train.csv`  
**Fungsi:** Training Data  
**Deskripsi:** Digunakan untuk melatih model agar dapat mempelajari pola kata dan menentukan sentimen (positif atau negatif).

---

## 🚀 How to Run

```bash
# 1. Instal dependensi
pip install -r requirements.txt

# 2. Jalankan preprocessing
python src/controller/preprocessing.py

# 3. Latih model
python src/controller/train_model.py

# 4. Evaluasi model
python src/controller/evaluate_model.py

# 5. Jalankan aplikasi Streamlit
streamlit run src/view/app.py
```

---

## ✨ Catatan
- Pastikan file dataset (`Survey Kepuasan SMK_Train.csv`) sudah ada di folder `data/raw/`.
- Semua output hasil preprocessing, model, dan evaluasi akan tersimpan di folder `data/processed/`, `src/model/`, dan `reports/`.
- Aplikasi web bisa diakses di browser setelah menjalankan perintah Streamlit.
