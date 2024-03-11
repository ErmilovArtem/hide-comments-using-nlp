import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import fasttext

# Загрузка стоп-слов
stop_words = set(stopwords.words('english'))

# Создание объекта PorterStemmer для стемминга слов
ps = PorterStemmer()


def preprocess_text(text):
    # Токенизация текста
    words = word_tokenize(text.lower())

    # Унификация (стемминг) и удаление стоп-слов
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Сбор обработанных слов обратно в строку
    processed_text = ' '.join(words)

    return processed_text

df1 = pd.read_csv("train_toxic_no_processed.csv")
df2 = pd.read_csv("train_no_processed.csv")

for df in [df1, df2]:
    df["Comments"] = df1["Comments"].apply(preprocess_text)
    df["Class"] = "__label__" + df1["Class"].astype("str")
    df["Class_Comments"] = df1["Class"] + " " + df1["Comments"]
# print(df["Class_Comments"])

train_1, test_1 = train_test_split(df1, test_size=0.1)
train_2, test_2 = train_test_split(df2, test_size=0.1)

test_1.to_csv("fasttext_negative_test_no_processed.csv")
test_2.to_csv("fasttext_toxic_test_no_processed.csv")

train_1.to_csv("train_fasttext_negative.train", columns=["Class_Comments"], index=False, header=False)
train_2.to_csv("train_fasttext_toxic.train", columns=["Class_Comments"], index=False, header=False)
test_1.to_csv("train_fasttext_negative.test", columns=["Class_Comments"], index=False, header=False)
test_2.to_csv("train_fasttext_toxic.test", columns=["Class_Comments"], index=False, header=False)
