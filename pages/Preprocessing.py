import streamlit as st
from contextlib import redirect_stdout
import io
import os
import pandas as pd
from src.controller.preprocessing import preprocess_dataset
st.set_page_config(page_title="Training Model Sentiment")
st.title("Preprocess Data")
if st.button("Mulai Preprocessing data"):
    log_box = st.code("Waiting...", language=None)
    f = io.StringIO()
    
    with redirect_stdout(f):
        with st.spinner("Preprocessing di mulai, jangan menutup file atau program..."):
            preprocess_dataset(
                input_path='data/raw/Survey Kepuasan SMK_Train.csv',
                output_path='data/processed/train_clean.csv',
                text_column='Survey',
                label_column='Label'
            )
            
    logs = f.getvalue()
    log_box.code(logs, language=None)
    st.success("Training selesai!")

st.write("---")

processed_path = 'data/processed/train_clean.csv'
if os.path.exists(processed_path):
    df = pd.read_csv(processed_path)
    st.info(f"File '{processed_path}' ditemukan.")
    if st.button("Hapus File Preprocessed"):
        os.remove(processed_path)
        st.success(f"File '{processed_path}' telah dihapus.")
else:
    st.error(f"File '{processed_path}' belum ada. Silakan lakukan preprocessing data terlebih dahulu.")