import streamlit as st
from contextlib import redirect_stdout
import io
from src.controller.preprocessing import preprocess_dataset
st.set_page_config(page_title="Training Model Sentiment")
st.title("Preprocess Data")
if st.button("Mulai Preprocessing data"):
    log_box = st.code("Waiting...", language=None)
    f = io.StringIO()
    
    with redirect_stdout(f):
        with st.spinner("Preprocessing in progress..."):
            preprocess_dataset(
                input_path='data/raw/Survey Kepuasan SMK_Train.csv',
                output_path='data/processed/train_clean.csv',
                text_column='Survey',
                label_column='Label'
            )
            
    logs = f.getvalue()
    log_box.code(logs, language=None)
    st.success("Training selesai!")