import os

from googletrans import Translator
import logging
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
import pickle

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

file_log = logging.FileHandler('model.log', encoding='utf-8')
console_out = logging.StreamHandler()

logging.basicConfig(handlers=(file_log, console_out),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

translator_obj = Translator()


def predict(text_list):
    ''' Return a list with prediction by comments list'''
    translatede_text_list = [word_tokenize(elem.text) for elem in translator_obj.translate(text_list, dest='en')]
    translatede_text_list= ["hello world! Good product happy love." if elem == "" else elem for elem in translatede_text_list]
    logging.info(translatede_text_list)

    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

    with open(os.path.join(BASE_DIR, 'saved_data_no_proc.pkl'), 'rb') as file:
        logging.info(file)
        tokenizer_negative = pickle.load(file)
    with open(os.path.join(BASE_DIR, 'saved_data_toxic_no_proc.pkl'), 'rb') as file:
        tokenizer_toxic = pickle.load(file)

    translatede_tokenizer_negative = tokenizer_negative.texts_to_sequences(translatede_text_list)
    translatede_tokenizer_toxic = tokenizer_toxic.texts_to_sequences(translatede_text_list)

    max_len = 100  # Максимальная длина последовательности

    translatede_pad_negative = pad_sequences(translatede_tokenizer_negative, maxlen=max_len)
    translatede_pad_toxic = pad_sequences(translatede_tokenizer_toxic, maxlen=max_len)

    model_negative = load_model(os.path.join(BASE_DIR, 'end_model_normal_no_proc.h5'))
    model_toxic = load_model(os.path.join(BASE_DIR, 'end_model_normal_toxic_no_proc.h5'))

    logging.info("go to a function / start loading models")

    logging.info(translatede_text_list)
    logging.info("model are loaded")
    # translatede_text_list = translator_obj.translate(text_list, dest='en')
    logging.info("text are translated")

    # logging.info(model_negative.predict(translatede_text_list)[0])
    predict_negative = model_negative.predict(translatede_pad_negative)
    predict_toxic = model_toxic.predict(translatede_pad_toxic)

    predict_negative = (predict_negative > 0.5).astype(int)
    predict_toxic = (predict_toxic > 0.5).astype(int)

    predict_negative = [elem[0] for elem in predict_negative]
    predict_toxic = [elem[0] for elem in predict_toxic]

    logging.info("prediction are ready")

    logging.info(predict_negative)
    logging.info(predict_toxic)
    return [max(predict_negative[index], predict_toxic[index]) for index in range(len(predict_negative))]


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
if __name__ == '__main__':
    liste = predict(list_to_predict)
    logging.info(([bool(elem) for elem in liste]))