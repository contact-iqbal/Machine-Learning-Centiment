import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_sentiment_model(
    train_path,
    model_path='src/model/sentiment_model.joblib',
    vectorizer_path='src/model/tfidf_vectorizer.joblib',
    verbose=True
):
    if verbose:
        print("mulai")

    df_train = pd.read_csv(train_path, encoding='utf-8')

    X_train = df_train['clean_text']
    y_train = df_train['Label']

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    y_pred_train = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Simpan model dan vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    if verbose:
        print(f"Model dan vectorizer disimpan di '{model_path}' dan '{vectorizer_path}'")
        print(f"Akurasi pada data training: {train_accuracy * 100:.2f}%")
        
    return model, vectorizer

if __name__ == "__main__":
    train_sentiment_model('data/processed/train_clean.csv')
    # train_sentiment_model('data/raw/Survey Kepuasan SMK_Train.csv')
    print("\nSelesai")