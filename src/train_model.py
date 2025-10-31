import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def train_sentiment_model(train_path, model_path='model/sentiment_model.joblib',
                          vectorizer_path='model/tfidf_vectorizer.joblib', verbose=True):
    if verbose:
        print(f"Training model...")

    df_train = pd.read_csv(train_path)
    X_train = df_train['clean_text']
    y_train = df_train['sentimen']

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred_train = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    if verbose:
        print(f"Selesai: Akurasi {train_accuracy * 100:.2f}%")

    return model, vectorizer


if __name__ == "__main__":
    print("=" * 60)
    print("TAHAP 2: TRAINING MODEL")
    print("=" * 60)
    print("Pada tahap ini, model akan belajar dari data training")
    print("untuk mengenali pola kata-kata yang menunjukkan sentimen.\n")

    train_sentiment_model('data/processed/train_clean.csv')

    print("\n" + "=" * 60)
    print("Model siap digunakan!")
    print("Lanjut ke tahap evaluasi dengan menjalankan: python src/evaluate_model.py")
    print("=" * 60)
