from googletrans import Translator
import logging
import fasttext
from nltk import PorterStemmer, word_tokenize
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
    translatede_text_list = [preprocess_text(elem.text) for elem in translator_obj.translate(text_list, dest='en')]
    translatede_text_list= ["hello world! Good product happy love." if elem == "" else elem for elem in translatede_text_list]
    print(translatede_text_list)
    logging.info("go to a function / start loading models")

    # text_list = ["hello world! Good product happy." if elem == "" else elem for elem in text_list]

    print(translatede_text_list)

    model_negative = fasttext.load_model("final_negative_model.bin")
    model_toxic = fasttext.load_model("final_toxic_model.bin")
    logging.info("model are loaded")

    # translatede_text_list = translator_obj.translate(text_list, dest='en')
    logging.info("text are translated")

    # print(model_negative.predict(translatede_text_list)[0])
    predict_negative = [int(elem[0][9:]) for elem in model_negative.predict(translatede_text_list)[0]]
    predict_toxic = [int(elem[0][9:]) for elem in model_toxic.predict(translatede_text_list)[0]]
    print(predict_negative)
    print(predict_toxic)

    print([max(predict_negative[index], predict_toxic[index]) for index in range(len(predict_negative))])
    logging.info("prediction are ready")

list_to_predict = ["плохой товар, я разочарован",
                   "",
                   "Bad product, I'm disappointed",
                   "منتج سيء ، أشعر بخيبة أمل",
                   "मखराब उत्पाद, मैं निराश हूं",
                   "糟糕的产品，我很失望",
                   "mala mercancía, estoy decepcionado",
                   ] + [
                      "блядский мудак иди нахуй",
                      "",
                      "fucking cocksucker go fuck yourself",
                      "اللعين الحقير تذهب اللعنة نفسك",
                      "कमबख्त मुर्गा, अपने आप को बकवास जाओ",
                      "他妈的混蛋去他妈的你自己",
                      "maldito hijo de puta vete a la mierda"
                  ] + [
                      "я люблю эту штуку! это делает меня счастливым",
                      "",
                      "I love this thing! It makes me happy.",
                      "أنا أحب هذا الشيء! يجعلني سعيدا.",
                      "我喜欢这东西！ 这让我很高兴。",
                      "मुझे यह बात पसंद है! यह मुझे खुश करता है ।",
                      "¡me encanta esta cosa! me hace feliz"
                  ]

predict(list_to_predict)
print([1,0,1,1,1,1,1, 1,0,1,1,1,1,1,0,0,0,0,0,0,0])