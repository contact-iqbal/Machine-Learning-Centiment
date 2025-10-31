import streamlit as st
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from controller.preprocessing import clean_text
    from controller.predict import predict_sentiment
except ModuleNotFoundError as e:
    st.error(f"Modul tidak ditemukan: {e}")
    st.stop()

st.set_page_config(page_title="Analisis Sentimen SMK", layout="wide")

@st.cache_resource
def load_model_cache():
    model = joblib.load('src/model/sentiment_model.joblib')
    vectorizer = joblib.load('src/model/tfidf_vectorizer.joblib')
    return model, vectorizer


def main():
    st.title("Analisis Sentimen Survei Kepuasan Siswa SMK")
    st.markdown("---")

    try:
        model, vectorizer = load_model_cache()
    except FileNotFoundError:
        st.error("Model belum tersedia. Jalankan training terlebih dahulu.")
        return

    # --------------------------------------------------------------
    st.subheader("Input Komentar atau File CSV")
    st.markdown("Masukkan komentar siswa atau unggah file CSV untuk menganalisis sentimen.")

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        input_text = st.text_area(
            "Masukkan komentar:",
            height=150,
            placeholder="Contoh: Guru sangat baik dalam menjelaskan materi."
        )
    with col_input2:
        uploaded_file = st.file_uploader("Atau upload file CSV", type=['csv'])

    analyze_button = st.button("Analisis Sentimen", use_container_width=True)

    chart_col, table_col = st.columns([2, 1])

    # --------------------------------------------------------------
    if analyze_button:
        if uploaded_file is not None:
            try:
                sample = uploaded_file.read().decode('utf-8-sig')
                delimiter = ';' if sample.count(';') > sample.count(',') else ','
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=delimiter, encoding='utf-8-sig', on_bad_lines='skip')
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
                return

            if 'Komentar' not in df.columns:
                st.error("File CSV harus memiliki kolom 'Komentar'.")
                return

            df = df[df['Komentar'].notna()]
            if df.empty:
                st.warning("Tidak ada komentar valid dalam file CSV.")
                return

            df['clean_text'] = df['Komentar'].apply(clean_text)
            df['Prediksi'], df['Kepercayaan'] = zip(*df['Komentar'].apply(
                lambda x: predict_sentiment(str(x), model, vectorizer)
            ))


            with table_col:
                st.subheader("Laporan Data (Tabel)")
                st.dataframe(df[['Komentar', 'Prediksi', 'Kepercayaan']], use_container_width=True)


            with chart_col:
                st.subheader("Distribusi Sentimen (Dari Data Tabel)")

                # Hitung jumlah dari kolom Prediksi
                sentiment_counts = df['Prediksi'].value_counts()

                if sentiment_counts.empty:
                    st.warning("Tidak ada data sentimen yang bisa divisualisasikan.")
                else:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(
                        sentiment_counts.index,
                        sentiment_counts.values,
                        color=['#4CAF50' if s == 'Positif' else '#F44336' for s in sentiment_counts.index],
                        width=0.6
                    )

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.5,
                            f'{int(height)}',
                            ha='center',
                            va='bottom',
                            fontsize=11,
                            color='black'
                        )

                    ax.set_ylabel("Jumlah Komentar", fontsize=11)
                    ax.set_xlabel("Kategori Sentimen", fontsize=11)
                    ax.set_title("Distribusi Sentimen Berdasarkan Data Tabel", fontsize=13, weight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', linestyle='--', alpha=0.6)
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)

        elif input_text.strip():
            sentiment, confidence = predict_sentiment(input_text, model, vectorizer)
            cleaned = clean_text(input_text)

            with chart_col:
                st.subheader("Hasil Prediksi Sentimen")
                st.write(f"Sentimen terdeteksi: **{sentiment}**")
                st.write(f"Tingkat keyakinan: **{confidence:.2f}%**")

            with table_col:
                st.subheader("Detil Data")
                st.write("Teks Asli:")
                st.text(input_text)
                st.write("Setelah Preprocessing:")
                st.text(cleaned)

        else:
            st.warning("Masukkan teks atau unggah file CSV terlebih dahulu.")


if __name__ == "__main__":
    main()
