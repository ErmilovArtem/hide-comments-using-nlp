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
    "This product is complete garbage, just toss it in the trash and forget about it.",
    "What a disgrace! I don't know how anyone could create something so pathetic.",
    "The creator of this product probably has no clue what quality means.",
    "This is just shameful. I would prefer not to even see this junk.",
    "Terrible choice! Someone clearly didn't think when making this.",
    "I'd rather spend my money on something more useful than this rubbish.",
    "A product not worthy of a single glance. Total disappointment.",
    "It seems the manufacturer decided to mock their customers by offering such trash.",
    "This is not a product, it's just pure disappointment and a waste of time.",
    "I wouldn't recommend anyone spend a penny on this. Complete letdown.",
    "Horrible quality! Could they not create something better?",
    "The worst thing I've ever seen. It's not even worth discussing.",
    "The manufacturer of this product probably thought customers wouldn't notice. Wrong.",
    "It's incomprehensible how anyone could create something so tasteless and pointless.",
    "This product is best ignored. Don't waste your time and money on it.",
    "I am impressed by the quality and functionality of this product.",
    "Great job on creating such a useful and well-designed product!",
    "This product exceeded my expectations. Highly recommended!",
    "I appreciate the effort put into making this product. It's fantastic!",
    "I'm delighted with my purchase. This product is exactly what I needed.",
    "The attention to detail in this product is commendable. Well done!",
    "This product has greatly improved my daily routine. Thank you!",
    "I've been using this product for a while now, and it's been excellent.",
    "Kudos to the team behind this product. It's truly remarkable!",
    "I couldn't be happier with this purchase. It's worth every penny.",
    "This product has made my life so much easier. I'm grateful for it.",
    "The quality of this product speaks for itself. I'm impressed!",
    "I have nothing but positive things to say about this product. It's superb!",
    "I've recommended this product to all my friends. It's that good!",
    "I'm amazed by how well this product works. It's definitely a winner."

]

# X = list(df["Comments"][:15])
print(X)

# X = [preprocess_text(str(elem)) for elem in X]
X = [word_tokenize(str(elem)) for elem in X]

y = [1] * 15 + [0] * 15

# y = df["Class"][:15]
with open('saved_data_toxic_no_proc.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

X_val_seq = tokenizer.texts_to_sequences(X)

# Добавление нулей для уравнивания длины последовательностей
max_len = 100  # Максимальная длина последовательности
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

model = load_model("end_model_normal_toxic_no_proc.h5")

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