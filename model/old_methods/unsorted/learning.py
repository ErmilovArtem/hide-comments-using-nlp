import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from gensim.models import KeyedVectors

from save import tokenizer

print(42)

# Загрузка обработанных данных
df = pd.read_csv('train_no_processed.csv')

# Токенизация комментариев
df['Comments'] = df['Comments'].apply(lambda x: word_tokenize(str(x)))

# Разделение на обучающий и проверочный наборы данных
X = df['Comments']
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка предобученной модели Word2Vec
word_vectors = KeyedVectors.load_word2vec_format('cc.en.300.bin', binary=True)

# Преобразование текста в последовательности векторов
max_len = 100  # Максимальная длина последовательности
X_train_seq = []
for sentence in X_train:
    vectors = [word_vectors[word] for word in sentence if word in word_vectors]
    X_train_seq.append(vectors)

X_val_seq = []
for sentence in X_val:
    vectors = [word_vectors[word] for word in sentence if word in word_vectors]
    X_val_seq.append(vectors)

# Добавление нулей для уравнивания длины последовательностей
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', dtype='float32')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', dtype='float32')

# Создание модели BiLSTM
embedding_dim = 300  # Устанавливаем размерность векторов в соответствии с размерностью предобученной модели
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_len, embedding_dim)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Загрузка предобученных весов в Embedding-слое
vocab_size = len(word_vectors.vocab)
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False  # Замораживаем слой с предобученными весами

# Компиляция и обучение модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_val_pad, y_val))

model.save("end_model_no_proc_fasttext.h5")


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