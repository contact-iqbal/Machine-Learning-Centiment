# 🚀 Panduan Cepat - Analisis Sentimen SMK

Panduan ini akan membantu Anda menjalankan proyek dengan cepat!

## ⚡ Langkah Cepat (5 Menit)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Semua Tahap Sekaligus
```bash
# Preprocessing data
python src/preprocessing.py

# Training model
python src/train_model.py

# Evaluasi model
python src/evaluate_model.py

# Jalankan aplikasi
streamlit run app/app.py
```

## 📋 Checklist Eksekusi

- [ ] Install semua library (pip install -r requirements.txt)
- [ ] Data CSV tersedia di folder data/raw/
- [ ] Jalankan preprocessing.py
- [ ] Jalankan train_model.py
- [ ] Jalankan evaluate_model.py
- [ ] Buka aplikasi Streamlit
- [ ] Coba prediksi beberapa komentar

## 🎯 Apa yang Harus Dilihat?

### Setelah Preprocessing
Cek folder `data/processed/` - harus ada:
- train_clean.csv
- test_clean.csv

### Setelah Training
Cek folder `model/` - harus ada:
- sentiment_model.joblib
- tfidf_vectorizer.joblib

### Setelah Evaluasi
Cek folder `reports/` - harus ada:
- confusion_matrix.png
- evaluation_report.txt
- sample_prediction.csv

### Di Aplikasi Streamlit
- Tab Prediksi: coba input komentar baru
- Tab Statistik: lihat akurasi dan confusion matrix
- Tab Tentang: baca penjelasan lengkap

## 💡 Tips Debugging

### Error: Module not found
```bash
pip install -r requirements.txt
```

### Error: File not found
Pastikan Anda di root folder proyek (ada folder src/, data/, dll)

### Error: Model not found
Jalankan training terlebih dahulu:
```bash
python src/train_model.py
```

### Streamlit tidak mau jalan
Pastikan sudah install:
```bash
pip install streamlit
```

## 🧪 Contoh Komentar untuk Dicoba

**Positif:**
- "Sekolah ini luar biasa! Guru-gurunya sangat baik"
- "Fasilitas lengkap dan lingkungan belajar nyaman"
- "Sangat puas dengan pelayanan sekolah"

**Negatif:**
- "Fasilitas rusak dan tidak terawat"
- "Guru sering tidak masuk kelas"
- "Sangat kecewa dengan kondisi sekolah"

## 📊 Target Hasil

Akurasi yang diharapkan: **75-90%**

Jangan khawatir jika tidak mencapai target - yang penting memahami prosesnya!

## 🆘 Butuh Bantuan?

1. Baca README.md untuk penjelasan detail
2. Lihat komentar di setiap file .py
3. Cek bagian FAQ di README.md
4. Tanya guru atau teman sekelas

## 🎓 Setelah Berhasil

Coba tantangan ini:
1. Tambahkan lebih banyak data training
2. Ubah parameter model
3. Coba model machine learning lain
4. Buat visualisasi tambahan
5. Export hasil ke Excel

---

**Selamat mencoba! Semangat belajar! 🔥**
