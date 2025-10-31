import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(test_path, model_path='model/sentiment_model.joblib',
                   vectorizer_path='model/tfidf_vectorizer.joblib', verbose=True):
    if verbose:
        print(f"Evaluasi model...")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df_test = pd.read_csv(test_path)

    X_test = df_test['clean_text']
    y_test = df_test['sentimen']

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Analisis Sentimen')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)

    report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'])

    with open('reports/evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LAPORAN EVALUASI MODEL ANALISIS SENTIMEN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Akurasi: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    sample_results = df_test[['komentar', 'sentimen']].copy()
    sample_results['prediksi'] = y_pred
    sample_results['benar'] = sample_results['sentimen'] == sample_results['prediksi']
    sample_results.head(20).to_csv('reports/sample_prediction.csv', index=False)

    if verbose:
        print(f"Selesai: Akurasi {accuracy * 100:.2f}%")

    return accuracy, cm, report


if __name__ == "__main__":
    print("=" * 60)
    print("TAHAP 3: EVALUASI MODEL")
    print("=" * 60)
    print("Pada tahap ini, kita akan menguji seberapa baik model")
    print("memprediksi sentimen pada data yang belum pernah dilihat.\n")

    evaluate_model('data/processed/test_clean.csv')

    print("\n" + "=" * 60)
    print("Evaluasi selesai!")
    print("Lihat folder reports/ untuk melihat hasil lengkap.")
    print("Selanjutnya, coba aplikasi dengan: streamlit run app/app.py")
    print("=" * 60)
