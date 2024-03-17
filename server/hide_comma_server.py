import logging
import os
import sys
import time

from flask import Flask, request, jsonify
import requests
import json

from model.model_usage_my_models import predict

app = Flask(__name__)

import requests
import json

file_log = logging.FileHandler('hide_comma_server.log', encoding='utf-8')
console_out = logging.StreamHandler()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
logging.basicConfig(handlers=(file_log, console_out),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)


class InvalidToken(Exception):
    pass

# здесь не прикручена пагинация, хз сколько коментов он обрабатывает и сколько постов
class FacebookPostRuner:
    def __init__(self, token : str, site = "graph.facebook.com", version = "v18.0"):
        self.token = "access_token=" + token
        self.site = site if "https://" in site else "https://" + site
        self.version = version
        self.url = f'{self.site}/{version}/'
        self.fields_url = f'{self.site}/{version}/me?fields='

    def get_json_by_str_fields(self, fields = "posts"):
        return requests.get(self.fields_url + fields + "&" + self.token).text

    def get_json_by_id(self, id = "122106155180052382_122101224734052382", sup_param = "/comments"):
        return requests.get(self.url + id + sup_param + "?" + self.token).text

    def get_user_posts_in_list_format(self):
        ''' получаем список постов (!БЕЗ ПАГИНАЦИИ!)'''
        data = json.loads(self.get_json_by_str_fields(fields="posts"))
        if "error" in data.keys():
            raise InvalidToken(str(data["error"]))
        posts_data = data["posts"]["data"]
        return posts_data

    def get_posts_comments(self):
        ''' вызываем отдельно для каждого поста (много малых json)
        список коменатриев (много json чуть побольше) (!БЕЗ ПАГИНАЦИИ!)'''
        id_list = [elem["id"] for elem in self.get_user_posts_in_list_format()]
        return [json.loads(self.get_json_by_id(elem, sup_param = "/comments"))["data"] for elem in id_list]

    def get_posts_comments(self):
        ''' вызываем отдельно для каждого поста (много малых json)
        список коменатриев (много json чуть побольше) (!БЕЗ ПАГИНАЦИИ!)'''
        id_list = [elem["id"] for elem in self.get_user_posts_in_list_format()]
        return [json.loads(self.get_json_by_id(elem, sup_param = "/comments"))["data"] for elem in id_list]

    def valdiate_message_by_neuro(self, comments_to_validate, token = "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"):
        ''' валидируем сообщения. Вернется булевый список.'''
        predict_list = predict(comments_to_validate)
        return list(map(bool, predict_list))

    def only_non_valid_comments_list(self, comments_lsit, gpt_token = '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'):
        ''' Разделил на разные функции для дальнейшего переиспользования '''
        comments_to_validate = [elem["message"] for elem in comments_lsit]
        logging.info("not valideted comma" + str(comments_to_validate))

        bool_list = []

        for sublist in [comments_to_validate[i:i+20] for i in range(0, len(comments_to_validate), 20)]:
            while True:
                try:
                    bool_sublist = self.valdiate_message_by_neuro(sublist, gpt_token)
                except Exception as e:
                    if "Rate limit reached for" in str(e):
                        time.sleep(80)
                    else:
                        raise e
                if len(sublist) == len(bool_sublist):
                    bool_list += bool_sublist
                    break
                elif len(sublist) == 1 and bool_sublist:
                    bool_list += [bool_sublist[0]]
                    break

        logging.info("bool list of valideted comma" + str(bool_list))

        problems = []
        for index in range(len(bool_list)):
            if bool_list[index]:
                problems.append(comments_lsit[index])

        logging.info("problem comma")
        logging.info(str(problems))

        return problems

    def post_hided_by_list_id(self, comments_lsit = [], hide = False, gpt_token = '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'):
        ''' удаляем комментарии по списку id или просто по id
        можно передавать и не индексированные по id списки, если так будет проще валидировать '''
        non_valid_comments = self.only_non_valid_comments_list(comments_lsit, gpt_token)

        request_list = []
        for elem in non_valid_comments:
          url = f'{self.site}/{elem["id"]}?is_hidden={str(hide).lower()}&' + self.token
          logging.info("url " + url)
          req = requests.post(url)
          request_list.append((req.text, elem["id"], elem["message"], req.status_code))
        return request_list

@app.route('/process_post_data', methods=['POST'])
def process_post_data():
    # Проверяем, есть ли токен в теле POST-запроса
    try:
        data = request.json

        if 'facebook_token' not in data:
            return jsonify({'error': 'Facebook Token is missing'}), 400  # Возвращаем ошибку, если токен отсутствует

        if 'gpt_token' not in data:
            return jsonify({'error': 'GPT Token is missing'}), 400  # Возвращаем ошибку, если токен отсутствует

        fr_user = FacebookPostRuner(data['facebook_token'])
        all_pages = json.loads(fr_user.get_json_by_str_fields(fields="accounts"))

        all_pages_tokens = [elem["access_token"] for elem in all_pages["accounts"]["data"]]

        to_req_client = []

        for page_token in all_pages_tokens:
            fr_page = FacebookPostRuner(page_token)
            for comments_in_page in [elem for elem in fr_page.get_posts_comments() if elem]:
                logging.info("all comma in page - " + str(comments_in_page))
                to_req_client.append(fr_page.post_hided_by_list_id(comments_lsit = comments_in_page, hide = False, gpt_token=data["gpt_token"]))

        return jsonify({"Succsess": True, "List_of_comments": to_req_client})
    except Exception as err:
        logging.info("ERROR:  " + str(err))
        if str(err) == "accounts":
            return jsonify({"Succsess": False, "error": "error of facebook token"})
        if "Incorrect API key provided" in str(err):
            return jsonify({"Succsess": False, "error": "error of gpt token"})
        return jsonify({"Succsess": False, "error": str(err)})

@app.route('/hello', methods=['get'])
def hello():
    return jsonify("hello")

if __name__ == '__main__':
    app.run(port=5000, debug=True)