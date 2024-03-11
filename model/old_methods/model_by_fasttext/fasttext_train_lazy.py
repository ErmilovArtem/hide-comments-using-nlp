import fasttext
import pandas as pd
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_negative = fasttext.train_supervised(input = "train_fasttext_negative.train", epoch=25, lr=0.1, wordNgrams=2)
model_negative.test("train_fasttext_negative.test")

model_toxic = fasttext.train_supervised(input = "train_fasttext_toxic.train", epoch=25, lr=0.1, wordNgrams=2)
model_toxic.test("train_fasttext_toxic.test")

print(model_negative.predict("happy happy happy!"))
print(model_toxic.predict("happy happy happy!"))



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


# df = pd.read_csv('test                  _processed.csv')

# X = [preprocess_text(str(elem)) for elem in X]

df_negative = pd.read_csv("fasttext_negative_test_no_processed.csv")
df_toxic = pd.read_csv("C:\\Users\\march\\PycharmProjects\\pythonProject1\\fasttext_toxic_test_no_processed.csv")

y_negative = df_negative["Class"]
y_toxic = df_toxic["Class"]

# Предсказание на проверочном наборе данных
# y_pred_prob = model.predict(list(df["Comments"])) #.apply(preprocess_text)))
print([i for i, x in enumerate([type(elem) == str for elem in (list(df_negative["Comments"]))]) if x==False])
# df_negative["Comments"][1439] = "hello world"
# print(df_negative["Comments"].min())
df_negative['Comments'] = df_negative['Comments'].fillna("Hello world")
y_pred_prob_negative = model_negative.predict(list(df_negative["Comments"])) #.apply(preprocess_text)))
y_pred_prob_toxic = model_toxic.predict(list(df_toxic["Comments"])) #.apply(preprocess_text)))
# y_pred = (y_pred_prob > 0.5).astype(int)

y_pred_negative = [int(elem[0][9:]) for elem in y_pred_prob_negative[0]]
y_pred_toxic = [int(elem[0][9:]) for elem in y_pred_prob_toxic[0]]

# print(list(y))
y_negative = [int(elem[9:]) for elem in list(y_negative)]
y_toxic = [int(elem[9:]) for elem in list(y_toxic)]

# # Вывод комментариев, предсказанных классов и фактических классов
# for comment, predicted_class, actual_class in zip(df_negative["Comments"], y_pred_negative, y_negative):
#     print(f"Comment: {comment}")
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Actual Class: {actual_class}")
#     print("------------")

# Вычисление метрик
accuracy = accuracy_score(y_negative, y_pred_negative)
precision = precision_score(y_negative, y_pred_negative)
recall = recall_score(y_negative, y_pred_negative)
f1 = f1_score(y_negative, y_pred_negative)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# # Вывод комментариев, предсказанных классов и фактических классов
# for comment, predicted_class, actual_class in zip(df_negative["Comments"], y_pred_toxic, y_toxic):
#     print(f"Comment: {comment}")
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Actual Class: {actual_class}")
#     print("------------")

# Вычисление метрик
accuracy = accuracy_score(y_toxic, y_pred_toxic)
precision = precision_score(y_toxic, y_pred_toxic)
recall = recall_score(y_toxic, y_pred_toxic)
f1 = f1_score(y_toxic, y_pred_toxic)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

model_negative.save_model("final_negative_model.bin")
model_toxic.save_model("final_toxic_model.bin")