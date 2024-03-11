import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Загрузка стоп-слов
stop_words = set(stopwords.words('english'))

# Создание объекта PorterStemmer для стемминга слов
ps = PorterStemmer()


df = pd.read_csv("C:\\Users\\march\\PycharmProjects\\pythonProject1\\train_TOXIC.csv")
# df = df.sample(frac=1, random_state=42).head(30000).reset_index(drop=True)

# Выбор необходимых столбцов
selected_columns = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[selected_columns]

# Функция для создания столбца 'Class'
def create_class_column(row):
    if any(row[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == 1):
        return 1
    else:
        return 0

# Применение функции к каждой строке и создание столбца 'Class'
df['Class'] = df.apply(create_class_column, axis=1)

df['Comments'] = df['comment_text']
# Вывод результата
df = df[['Class', 'Comments']]

processed_file_name = 'train_toxic_no_processed.csv'

# Разделите DataFrame на два подмножества по классу
class_0 = df[df['Class'] == 0]
class_1 = df[df['Class'] == 1]

# Выберите случайные 15000 объектов из каждого класса
class_0_sampled = class_0.sample(n=15000, random_state=42)
class_1_sampled = class_1.sample(n=15000, random_state=42)

# Объедините образцы в новый DataFrame
balanced_df = pd.concat([class_0_sampled, class_1_sampled])

# Перемешайте строки, чтобы объекты каждого класса были случайно распределены
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Выведите информацию о количестве объектов каждого класса
print(balanced_df['Class'].value_counts())

# Выведите первые несколько строк нового DataFrame
print(balanced_df.head())

balanced_df.to_csv(processed_file_name, index=False)

class_counts = balanced_df['Class'].value_counts()

# Вывод результатов
print("Количество классов со значением 1 и 0:")
print(class_counts)

print(balanced_df)
print(f"Processed data saved to {processed_file_name}")