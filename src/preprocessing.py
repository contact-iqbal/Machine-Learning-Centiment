import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)

    return text


def preprocess_dataset(input_path, output_path, text_column='komentar', label_column='sentimen', verbose=True):
    if verbose:
        print(f"Preprocessing {input_path}...")

    try:
        df = pd.read_csv(input_path, encoding='utf-8', quoting=1, on_bad_lines='skip')
    except Exception as e:
        if verbose:
            print(f"Error: {e}")

        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue

            last_comma = line.rfind(',')

            if last_comma != -1:
                komentar = line[:last_comma].strip()
                sentimen = line[last_comma+1:].strip()

                if sentimen.lower() in ['positif', 'negatif']:
                    data.append({
                        'komentar': komentar,
                        'sentimen': sentimen.capitalize()
                    })
                else:
                    if verbose:
                        print(f"Baris {i}: Label tidak valid")
            else:
                if verbose:
                    print(f"Baris {i}: Format tidak valid")

        df = pd.DataFrame(data)

    if text_column not in df.columns or label_column not in df.columns:
        print(f"ERROR: Kolom '{text_column}' atau '{label_column}' tidak ditemukan!")
        return

    df['clean_text'] = df[text_column].apply(clean_text)

    df_clean = df[[text_column, 'clean_text', label_column]].copy()
    df_clean = df_clean[df_clean['clean_text'].str.strip() != '']

    df_clean.to_csv(output_path, index=False)

    if verbose:
        print(f"Selesai: {len(df_clean)} data")


if __name__ == "__main__":
    print("=" * 60)
    print("TAHAP 1: PREPROCESSING DATA")
    print("=" * 60)
    print("Preprocessing adalah proses membersihkan data teks")
    print("agar lebih mudah dipahami oleh model machine learning.\n")

    preprocess_dataset(
        input_path='data/raw/Survey Kepuasan SMK_Train.csv',
        output_path='data/processed/train_clean.csv'
    )

    preprocess_dataset(
        input_path='data/raw/Survey Kepuasan SMK_Test.csv',
        output_path='data/processed/test_clean.csv'
    )

    print("Silakan cek folder data/processed/ untuk melihat hasilnya.")
