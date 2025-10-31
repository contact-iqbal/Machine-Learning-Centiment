import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    test_path='data/processed/train_clean.csv',
    model_path='src/model/sentiment_model.joblib',
    vectorizer_path='src/model/tfidf_vectorizer.joblib',
    report_dir='reports',
    verbose=True
):
    if verbose:
        print("Evaluasi model...")

    # Pastikan file ada
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"File {test_path} tidak ditemukan. Pastikan sudah menjalankan preprocessing dan training terlebih dahulu.")

    # Load model dan vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Baca dataset test
    df_test = pd.read_csv(test_path, encoding='utf-8')

    # Pastikan kolom sesuai
    if 'clean_text' not in df_test.columns or 'Label' not in df_test.columns:
        raise ValueError(f"Kolom 'clean_text' dan 'Label' harus ada di {test_path}. Kolom ditemukan: {list(df_test.columns)}")

    X_test = df_test['clean_text']
    y_test = df_test['Label']

    # Transform data test
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    # Evaluasi hasil
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['negatif', 'positif'])

    # Pastikan folder report ada
    os.makedirs(report_dir, exist_ok=True)

    # Simpan confusion matrix visual
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Analisis Sentimen')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Simpan laporan evaluasi
    with open(os.path.join(report_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Akurasi: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    # Simpan hasil prediksi contoh
    sample_results = df_test[['Survey', 'Label']].copy() if 'Survey' in df_test.columns else df_test.copy()
    sample_results['Prediksi'] = y_pred
    sample_results['Benar'] = sample_results['Label'] == sample_results['Prediksi']
    sample_results.head(20).to_csv(os.path.join(report_dir, 'sample_prediction.csv'), index=False)

    if verbose:
        print(f"Akurasi model: {accuracy * 100:.2f}%")
        print(f"Hasil evaluasi disimpan di folder: {report_dir}")

    return accuracy, cm, report


if __name__ == "__main__":
    evaluate_model('data/processed/train_clean.csv')

