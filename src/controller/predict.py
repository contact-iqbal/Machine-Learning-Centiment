import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller.preprocessing import clean_text


def load_model(model_path='src/model/sentiment_model.joblib',
               vectorizer_path='src/model/tfidf_vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_sentiment(text, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    cleaned = clean_text(text)

    if cleaned.strip() == "":
        return "Tidak dapat diprediksi (teks kosong)", 0.0

    text_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    confidence = max(probability) * 100

    return prediction, confidence
