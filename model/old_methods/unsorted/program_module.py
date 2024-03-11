import pickle

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
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



X = ["i hate this product",
     "i love this product",
     "Don't buy it for anything! Deception",
     "I liked it, and at a discount the product brought me even more joy",
     "This supplement was a big disappointment for me.",
     "No matter what, I think it's worth purchasing", # Не смотря ни на что, я думаю что стоит приобретать
     "It's hard to say, it's more likely no than yes", # Сложно сказать, скорее нет чем да
     "definitely yes",
     "If you don't want to be poisoned, don't buy",
     "The tablet tastes like I ate licorice as a child, wonderful!",
     "I want to fuck you",
     "I liked it and will buy more",
     "I will never return again and will not recommend it to anyone",
     "cool book",
     "shit so bad"
     ]

# X = list(df["Comments"][:15])
print(X)

# X = [preprocess_text(str(elem)) for elem in X]
X = [word_tokenize(str(elem)) for elem in X]

y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

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