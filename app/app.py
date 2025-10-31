"""
app.py
======
Aplikasi Streamlit untuk Analisis Sentimen Survei Kepuasan Siswa SMK

Aplikasi ini memungkinkan pengguna untuk:
1. Memasukkan komentar baru
2. Melihat prediksi sentimen secara real-time
3. Melihat statistik model
4. Memahami cara kerja model secara interaktif
"""

import streamlit as st
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text
from src.predict import predict_sentiment


st.set_page_config(
    page_title="Analisis Sentimen SMK",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model_cache():
    """Memuat model dengan caching untuk performa lebih baik."""
    model = joblib.load('model/sentiment_model.joblib')
    vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    return model, vectorizer


def main():
    st.title("📊 Analisis Sentimen Survei Kepuasan Siswa SMK")
    st.markdown("---")

    st.markdown("""
    ### Selamat Datang di Aplikasi Analisis Sentimen!

    Aplikasi ini menggunakan **Machine Learning** untuk menganalisis sentimen komentar siswa
    dalam survei kepuasan sekolah. Model akan memprediksi apakah komentar bersifat **Positif** atau **Negatif**.
    """)

    try:
        model, vectorizer = load_model_cache()
        st.success("✓ Model berhasil dimuat!")
    except FileNotFoundError:
        st.error("❌ Model belum tersedia. Jalankan training terlebih dahulu!")
        st.info("Jalankan: `python src/train_model.py`")
        return

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📈 Statistik", "ℹ️ Tentang"])

    with tab1:
        st.header("Prediksi Sentimen")

        input_method = st.radio(
            "Pilih metode input:",
            ["Teks Tunggal", "Batch (Multiple)"]
        )

        if input_method == "Teks Tunggal":
            user_input = st.text_area(
                "Masukkan komentar:",
                height=150,
                placeholder="Contoh: Sekolah ini sangat bagus, guru-gurunya ramah dan fasilitas lengkap!"
            )

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                predict_button = st.button("🔍 Prediksi", type="primary", use_container_width=True)

            with col2:
                clear_button = st.button("🗑️ Bersihkan", use_container_width=True)

            if predict_button and user_input.strip():
                with st.spinner("Menganalisis sentimen..."):
                    sentiment, confidence = predict_sentiment(user_input, model, vectorizer)

                    cleaned = clean_text(user_input)

                    if sentiment == "Positif":
                        st.success(f"### 😊 Sentimen: **{sentiment}**")
                        st.progress(confidence / 100)
                        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                    else:
                        st.error(f"### 😞 Sentimen: **{sentiment}**")
                        st.progress(confidence / 100)
                        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")

                    with st.expander("🔬 Lihat Proses Preprocessing"):
                        st.write("**Teks Asli:**")
                        st.text(user_input)
                        st.write("**Teks Setelah Dibersihkan:**")
                        st.text(cleaned)

            elif predict_button:
                st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")

            if clear_button:
                st.rerun()

        else:
            st.write("**Masukkan beberapa komentar (satu per baris):**")
            batch_input = st.text_area(
                "Komentar:",
                height=200,
                placeholder="Masukkan komentar, satu per baris"
            )

            if st.button("🔍 Prediksi Semua", type="primary"):
                if batch_input.strip():
                    comments = [c.strip() for c in batch_input.split('\n') if c.strip()]

                    results = []
                    for comment in comments:
                        sentiment, confidence = predict_sentiment(comment, model, vectorizer)
                        results.append({
                            'Komentar': comment,
                            'Sentimen': sentiment,
                            'Keyakinan (%)': f"{confidence:.2f}"
                        })

                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)

                    positive_count = len([r for r in results if r['Sentimen'] == 'Positif'])
                    negative_count = len(results) - positive_count

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("😊 Positif", positive_count)
                    with col2:
                        st.metric("😞 Negatif", negative_count)
                else:
                    st.warning("⚠️ Silakan masukkan komentar terlebih dahulu!")

    with tab2:
        st.header("📈 Statistik Model")

        try:
            with open('reports/evaluation_report.txt', 'r', encoding='utf-8') as f:
                report_content = f.read()

            lines = report_content.split('\n')
            accuracy_line = [line for line in lines if 'Akurasi:' in line]
            if accuracy_line:
                accuracy_value = float(accuracy_line[0].split(':')[1].strip().replace('%', ''))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Akurasi Model", f"{accuracy_value:.2f}%")
                with col2:
                    st.metric("📊 Jumlah Fitur", f"{vectorizer.get_feature_names_out().shape[0]}")
                with col3:
                    st.metric("🏷️ Kelas", "2 (Positif/Negatif)")

            st.subheader("Confusion Matrix")
            try:
                st.image('reports/confusion_matrix.png', use_container_width=True)
            except:
                st.info("Confusion matrix belum tersedia. Jalankan evaluasi terlebih dahulu.")

            with st.expander("📄 Laporan Lengkap"):
                st.text(report_content)

        except FileNotFoundError:
            st.info("Statistik belum tersedia. Jalankan evaluasi model terlebih dahulu.")
            st.code("python src/evaluate_model.py")

    with tab3:
        st.header("ℹ️ Tentang Proyek")

        st.markdown("""
        ### 🎓 Proyek Akhir Kelas 11 Semester 2

        **Judul:** Analisis Sentimen Survei Kepuasan Siswa SMK

        #### 🎯 Tujuan Pembelajaran:
        - Memahami konsep dasar **Data Science** dan **Machine Learning**
        - Belajar memproses data teks (Text Preprocessing)
        - Memahami cara kerja model klasifikasi
        - Mengevaluasi performa model

        #### 🔧 Teknologi yang Digunakan:
        - **Python** - Bahasa pemrograman
        - **Pandas** - Pengolahan data
        - **Scikit-learn** - Machine Learning
        - **Sastrawi** - Stemming Bahasa Indonesia
        - **Streamlit** - Aplikasi Web

        #### 📊 Alur Kerja:
        1. **Preprocessing** - Membersihkan data teks
        2. **Feature Extraction** - TF-IDF Vectorization
        3. **Training** - Melatih model Logistic Regression
        4. **Evaluation** - Menguji performa model
        5. **Deployment** - Aplikasi web ini!

        #### 📚 Konsep yang Dipelajari:
        - **TF-IDF**: Mengubah teks menjadi angka
        - **Logistic Regression**: Model klasifikasi sederhana
        - **Train-Test Split**: Pentingnya data terpisah
        - **Accuracy & Confusion Matrix**: Metrik evaluasi
        """)

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Dibuat dengan ❤️ untuk pembelajaran Data Science di SMK</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
