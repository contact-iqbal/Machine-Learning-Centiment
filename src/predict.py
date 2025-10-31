import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text


def load_model(model_path='model/sentiment_model.joblib',
               vectorizer_path='model/tfidf_vectorizer.joblib'):
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


if __name__ == "__main__":
    print("=" * 60)
    print("PREDIKSI SENTIMEN - MODE INTERAKTIF")
    print("=" * 60)
    print("Masukkan teks untuk memprediksi sentimennya.")
    print("Ketik 'exit' untuk keluar.\n")

    model, vectorizer = load_model()

    while True:
        text = input("Masukkan komentar: ")

        if text.lower() == 'exit':
            print("Terima kasih! Program selesai.")
            break

        if text.strip() == "":
            print("Teks tidak boleh kosong!\n")
            continue

        sentiment, confidence = predict_sentiment(text, model, vectorizer)

        print(f"\nHasil Prediksi:")
        print(f"  Sentimen: {sentiment}")
        print(f"  Keyakinan: {confidence:.2f}%\n")
