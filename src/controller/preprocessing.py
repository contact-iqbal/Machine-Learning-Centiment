import pandas as pd
import re
import os
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
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    negative_words = ['tidak', 'kurang', 'belum', 'buruk', 'jelek', 'gagal']
    for word in negative_words:
        text = text.replace(word, f"{word}_keep")

    text = stopword_remover.remove(text)

    for word in negative_words:
        text = text.replace(f"{word}_keep", word)
    text = stemmer.stem(text)

    return text

def preprocess_dataset(input_path, output_path, 
                       text_column='Survey', 
                       label_column='Label'
                       ):
    print("Mulai")
    print("reading file:", input_path)

    #delimiter
    #with open(input_path, 'r', encoding='utf-8') as f:
    #    sample = f.readline()
    # sep = ';' if sample.count(';') > sample.count(',') else ','
    sep = ';'

    try:
        df = pd.read_csv(input_path, sep=sep, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print("Gagal membaca file:", e)
        return

    if text_column not in df.columns or label_column not in df.columns:
        print(f"Kolom '{text_column}' atau '{label_column}' tidak ditemukan.")
        print({list(df.columns)})
        return

    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    df_clean = df[[text_column, 'clean_text', label_column]].copy()
    df_clean = df_clean[df_clean['clean_text'].str.strip() != '']
    
    df_clean[label_column] = df_clean[label_column].str.lower().str.strip()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"file disimpan di: {output_path}")
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("Selesai")
    

if __name__ == "__main__":
    preprocess_dataset(
        input_path='data/raw/Survey Kepuasan SMK_Train.csv',
        output_path='data/processed/train_clean.csv',
        text_column='Survey',
        label_column='Label'
    )

print("selesai")