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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка обработанных данных
df = pd.read_csv('train_no_processed.csv')

# Токенизация комментариев
df['Comments'] = df['Comments'].apply(lambda x: word_tokenize(str(x)))

# Разделение на обучающий и проверочный наборы данных
X = df['Comments']
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование текста в последовательности индексов
max_words = 10000  # Максимальное количество слов для токенизатора
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

with open('saved_data_no_proc.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Добавление нулей для уравнивания длины последовательностей
max_len = 100  # Максимальная длина последовательности
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

# Создание модели BiLSTM
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.metrics_names)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_val_pad, y_val))

model.save("end_model_normal_no_proc.h5")

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