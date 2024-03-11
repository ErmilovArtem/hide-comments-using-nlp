import fasttext
import pickle

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# from save import tokenizer
#
# print(42)
#
# # Загрузка обработанных данных
# df = pd.read_csv('train_no_processed.csv')
#
# # Разделение на обучающий и проверочный наборы данных
# X = df['Comments']
# y = df['Class']
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка предобученной модели Word2Vec
ft = fasttext.load_model('C:\\Users\\march\\Desktop\\cc.en.300.bin')

def generateVector(sentence):
    return [int(elem) for elem in list(np.asarray((ft.get_sentence_vector(sentence) * 500 + 123)))]
# Загрузка обработанных данных
df = pd.read_csv('train_no_processed.csv')

# Токенизация комментариев
df['Comments'] = df['Comments'].apply(lambda x: generateVector(str(x)))

# Разделение на обучающий и проверочный наборы данных
X = list(df['Comments'])
y = list(df['Class'])

print(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Добавление нулей для уравнивания длины последовательностей
max_len = 300  # Максимальная длина последовательности
# X_train_pad = X_train #pad_sequences(X_train, maxlen=max_len)

# X_train_pad = tf.convert_to_tensor([x for x in X_train], dtype=tf.float32)
# # X_val_pad =  X_val #pad_sequences(X_val, maxlen=max_len)
# X_val_pad = tf.convert_to_tensor([x for x in X_val], dtype=tf.float32)

# print(X_train)

# X_train_np = np.array([np.array(comment) for comment in X_train]).astype(np.float32)
# X_train_np = pad_sequences(X_train, maxlen=max_len)
# X_val_np = np.array([np.array(comment) for comment in X_val]).astype(np.float32)
# X_val_np = pad_sequences(X_val, maxlen=max_len)

X_train_pad = pad_sequences(X_train, maxlen=max_len)
X_val_pad = pad_sequences(X_val, maxlen=max_len) #, dtype=tf.float32

# print(X_train_pad)

# Создание модели BiLSTM
embedding_dim = 300
vocab_size = 300

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.metrics_names)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(len(X_train), len(y_train))
print(len(X_val), len(y_val))

print(min(len(sublist) for sublist in X_train))
print(min(len(sublist) for sublist in X_val))
print(max(len(sublist) for sublist in X_train))
print(max(len(sublist) for sublist in X_val))

print(min(min(sublist) for sublist in X_train))
print(min(min(sublist) for sublist in X_train))
print(max(max(sublist) for sublist in X_val))
print(max(max(sublist) for sublist in X_val))

# Обучение модели
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

model.save("end_model_real_fasttext_no_proc.h5")

print(X_val_pad)

# Предсказание на проверочном наборе данных
y_pred_prob = model.predict(X_val_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

# Вывод комментариев, предсказанных классов и фактических классов
for comment, predicted_class, actual_class in zip(X_val, y_pred, y_val):
    print(f"Comment: {comment}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Actual Class: {actual_class}")
    print("------------")

# Вычисление метрик
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")




