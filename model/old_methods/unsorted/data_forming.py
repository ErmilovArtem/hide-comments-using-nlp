import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Загрузка стоп-слов
stop_words = set(stopwords.words('english'))

print(stop_words)

# Создание объекта PorterStemmer для стемминга слов
ps = PorterStemmer()

for elem in ['test_negative.csv', 'train_negative.csv']:
    df = pd.read_csv(elem, header=None)
    df = df.sample(frac=1, random_state=42).head(30000).reset_index(drop=True)

    df[0] = df[0] - 1

    df = df[[0, 2]]

    df.columns = ['Class', 'Comments']
    df['Class'] = df['Class'].replace({0: 1, 1: 0})
    # Обработка текста в столбце 'Comments'
    df['Comments'] = df['Comments']

    df.columns = ['Class', 'Comments']
    processed_file_name = elem.split(".")[0] + '_processed.csv'
    df.to_csv(processed_file_name, index=False)

    class_counts = df['Class'].value_counts()

    # Вывод результатов
    print("Количество классов со значением 1 и 0:")
    print(class_counts)

    print(df)
    print(f"Processed data saved to {processed_file_name}")