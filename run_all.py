import os
import sys


def main():
    print("=" * 60)
    print("PROYEK ANALISIS SENTIMEN SMK")
    print("=" * 60)
    print()

    data_files = [
        'data/raw/Survey Kepuasan SMK_Train.csv',
        'data/raw/Survey Kepuasan SMK_Test.csv'
    ]

    print("[1/4] Cek data mentah...")
    for file in data_files:
        if not os.path.exists(file):
            print(f"ERROR: {file} tidak ditemukan!")
            sys.exit(1)
    print("      Data tersedia\n")

    print("[2/4] Preprocessing...")
    from src.preprocessing import preprocess_dataset
    preprocess_dataset('data/raw/Survey Kepuasan SMK_Train.csv',
                      'data/processed/train_clean.csv', verbose=False)
    preprocess_dataset('data/raw/Survey Kepuasan SMK_Test.csv',
                      'data/processed/test_clean.csv', verbose=False)
    print("      Preprocessing selesai\n")

    print("[3/4] Training model...")
    from src.train_model import train_sentiment_model
    train_sentiment_model('data/processed/train_clean.csv', verbose=False)
    print("      Model berhasil dilatih\n")

    print("[4/4] Evaluasi model...")
    from src.evaluate_model import evaluate_model
    accuracy, cm, report = evaluate_model('data/processed/test_clean.csv', verbose=False)
    print(f"      Akurasi: {accuracy * 100:.2f}%\n")

    print("=" * 60)
    print("SELESAI!")
    print("=" * 60)
    print()
    print("Hasil:")
    print("  - Model: model/sentiment_model.joblib")
    print("  - Laporan: reports/evaluation_report.txt")
    print("  - Grafik: reports/confusion_matrix.png")
    print()
    print("Jalankan aplikasi:")
    print("  streamlit run app/app.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
