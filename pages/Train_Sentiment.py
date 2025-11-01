from contextlib import redirect_stdout
import io
import re
import streamlit as st
from src.controller.train_model import train_sentiment_model
from src.controller.evaluate_model import evaluate_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'data/processed/train_clean.csv'

st.set_page_config(page_title="Training Model Sentiment")
st.title("Training Model")
if st.button("Mulai Training"):
    log_box = st.code("Waiting...", language=None)
    f = io.StringIO()
    
    with redirect_stdout(f):
        with st.spinner("Training in progress..."):
            train_sentiment_model(path)
            evaluate_model(path)
            
    logs = f.getvalue()
    log_box.code(logs, language=None)
    with open ("reports/evaluation_report.txt", "r", encoding="utf-8") as r:
        text = r.read()
        
    accuracy_match = re.search(r"Akurasi:\s*([\d.]+)%", text)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    
    cm_match = re.search(r"Confusion Matrix:\s*\n(\[\[.*?\]\])", text, re.S)

    if cm_match:
        cm_text = cm_match.group(1)
        nums = list(map(int, re.findall(r"\d+", cm_text)))
        cm = np.array(nums).reshape(2, 2)
    else:
        cm = None
    
    report_section = text.split("Classification Report:")[-1].strip()
    lines = [line.strip() for line in report_section.splitlines() if line.strip()]
    
    data = []
    for line in lines[1:]:  
        parts = line.split()
        if len(parts) >= 5 and re.match(r"^[A-Za-z]", parts[0]):
            label = parts[0] + ("" if len(parts) == 5 else " " + parts[1])
            numbers = parts[-4:]
            try:
                precision, recall, f1, support = map(float, numbers)
                data.append((label, precision, recall, f1, int(support)))
            except ValueError:
                continue
            
    report_df = pd.DataFrame(data, columns=["Label", "Precision", "Recall", "F1-score", "Support"])
    
    true_positive = cm[1, 1]
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]

    labels = ["Predict Positive", "Predict Negative", "Salah Predict Positive", "Salah Predict Negative"]
    values = [true_positive, true_negative, false_positive, false_negative]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    st.title("Model Evaluation Results")
    if accuracy:
        st.metric("Accuracy", f"{accuracy:.2f}%")
    if cm is not None:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        fig.set_facecolor((1,1,1,0))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': 'white'})
        ax.axis("equal")
        st.pyplot(fig)
    if not report_df.empty:
        st.subheader("Classification Report")
        st.dataframe(report_df.style.format(precision=2))
        
    st.success("Training selesai!")