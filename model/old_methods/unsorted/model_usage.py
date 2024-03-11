import fasttext
from googletrans import Translator
import logging

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

translator_obj = Translator()

# Загрузка стоп-слов
stop_words = set(stopwords.words('english'))

# Создание объекта PorterStemmer для стемминга слов
ps = PorterStemmer()

def preprocess_text(text):
    # Токенизация текста
    words = word_tokenize(text.lower())

    # Унификация (стемминг) и удаление стоп-слов
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Сбор обработанных слов обратно в строку
    processed_text = ' '.join(words)

    return processed_text



def predict(text_list):
    ''' Return a list with prediction by comments list'''
    translatede_text_list = translator_obj.translate(text_list, dest='en')
    translatede_text_list = [preprocess_text(elem.text) for elem in translatede_text_list]
    logging.info("go to a function / start loading models")

    translatede_text_list = ["hello world! Good product happy." if elem == "" else elem for elem in translatede_text_list]
    print(translatede_text_list)

    model_negative = fasttext.load_model("final_negative_model.bin")
    model_toxic = fasttext.load_model("final_toxic_model.bin")
    logging.info("model are loaded")

    # translatede_text_list = translator_obj.translate(text_list, dest='en')
    logging.info("text are translated")

    predict_negative = model_negative.predict(translatede_text_list)
    predict_toxic = model_toxic.predict(translatede_text_list)
    logging.info("prediction are ready")

    print(predict_negative, predict_toxic)


list_to_predict = ["сомнительный товар не рекомендую",
                   "I do not recommend dubious goods",
                   "",
                   "أنا لا أوصي البضائع المشكوك فيها",
                   "मैं संदिग्ध सामान की सिफारिश नहीं करता हूं",
                   "我不推荐可疑的商品",
                   "no recomiendo productos dudosos",
                   ] + [
                      "блядский хуесос иди нахуй",
                      "",
                      "fucking cocksucker go fuck yourself",
                      "اللعين الحقير تذهب اللعنة نفسك",
                      "他妈的混蛋去他妈的你自己",
                      "maldito hijo de puta vete a la mierda"
                  ]

# print(translator_obj.translate(list_to_predict, dest='en'))
predict(list_to_predict)
