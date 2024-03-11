import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import fasttext
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Загрузка обработанных данных
df = pd.read_csv('train_processed.csv')

# Токенизация комментариев
df['Comments'] = df['Comments'].apply(lambda x: word_tokenize(str(x)))

# Разделение на обучающий и проверочный наборы данных
X = df['Comments']
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка предобученной модели FastText
model_path = 'cc.en.300.bin'  
fasttext_model = fasttext.load_model(model_path)

# Преобразование текста в средний вектор каждого комментария
def average_word_vectors(tokens, vector_size=300):
    vectors = [fasttext_model[word] for word in tokens if word in fasttext_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

X_train_vec = [average_word_vectors(tokens) for tokens in X_train]
X_val_vec = [average_word_vectors(tokens) for tokens in X_val]

# Преобразование в массив NumPy
X_train_vec = np.array(X_train_vec)
X_val_vec = np.array(X_val_vec)

# Создание модели BiLSTM
model = Sequential()
model.add(Dense(64, input_shape=(300,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_vec, y_train, epochs=10, batch_size=32, validation_data=(X_val_vec, y_val))

model.save("end_model_fasttext.h5")



# Предсказание на проверочном наборе данных
y_pred_prob = model.predict(X_val_vec)
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