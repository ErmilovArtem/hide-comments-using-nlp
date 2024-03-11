import pickle

import pandas as pd
import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import load_model



# Загрузка стоп-слов
stop_words = set(stopwords.words('english'))

print(stop_words)

# Создание объекта PorterStemmer для стемминга слов
ps = PorterStemmer()


def preprocess_text(text):
    # Токенизация текста
    words = word_tokenize(text.lower())

    # Унификация (стемминг) и удаление стоп-слов
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]

    # Сбор обработанных слов обратно в строку
    processed_text = ' '.join(words)

    return processed_text


df = pd.read_csv('test_no_processed.csv')



X = df["Comments"]

# X = list(df["Comments"][:15])
print(X)

# X = [preprocess_text(str(elem)) for elem in X]
X = [word_tokenize(str(elem)) for elem in X]

y = df["Class"]

# y = df["Class"][:15]
with open('saved_data_no_proc.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

X_val_seq = tokenizer.texts_to_sequences(X)

# Добавление нулей для уравнивания длины последовательностей
max_len = 100  # Максимальная длина последовательности
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

model = load_model("end_model_normal_no_proc.h5")

print(X_val_pad)

# Предсказание на проверочном наборе данных
y_pred_prob = model.predict(X_val_pad)
y_pred = (y_pred_prob > 0.5).astype(int)


# Вывод комментариев, предсказанных классов и фактических классов
for comment, predicted_class, actual_class in zip(X, y_pred, y):
    print(f"Comment: {comment}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Actual Class: {actual_class}")
    print("------------")

# Вычисление метрик
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")