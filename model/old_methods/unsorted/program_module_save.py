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


# df = pd.read_csv('test_processed.csv')



X = [
"This product exceeded my expectations! It's fantastic!",
"I love how this product is easy to use and provides excellent results.",
"The design of this product is absolutely beautiful!",
"It's worth every penny. This product is high-quality and durable.",
"This is my new favorite product! I can't get enough of it.",
"I've tried many similar products, but this one stands out. It's amazing!",
"This product has made my life so much easier. I'm impressed!",
"The customer service for this product is top-notch. They really care about their customers.",
"Using this product brings a smile to my face every time.",
"I highly recommend this product to anyone looking for a reliable solution.",
"This product is a game-changer! It has exceeded all my expectations.",
"The performance of this product is outstanding. It gets the job done perfectly.",
"I'm always excited to show off this product to my friends. It's that good!",
"This product offers a wide variety of options, making it versatile and appealing.",
"The user-friendly interface of this product makes it a breeze to navigate and use.",
"I'm disappointed with this product. It didn't live up to its claims.",
"This product is poorly made and feels cheap. I expected better quality.",
"I encountered several issues with this product. It was frustrating to deal with.",
"The customer service for this product was unhelpful and unsupportive.",
"I do not recommend this product. It didn't meet my expectations at all.",
"The design of this product is not appealing. It's lackluster and plain.",
"This product didn't work for me. It was a waste of my time and money.",
"Using this product was a hassle. It was difficult to set up and operate.",
"I regret purchasing this product. It's not worth the price tag.",
"This product broke easily, even with careful handling.",
"The performance of this product was subpar. I expected better results.",
"The packaging of this product was damaged upon arrival, which was disappointing.",
"I found this product to be confusing and not intuitive to use.",
"This product didn't deliver the promised benefits. It fell short of my expectations.",
"I experienced compatibility issues with this product. It didn't work well with other devices or systems."
]

# X = list(df["Comments"][:15])
print(X)

# X = [preprocess_text(str(elem)) for elem in X]
X = [word_tokenize(str(elem)) for elem in X]

y = [0] * 15 + [1] * 15

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