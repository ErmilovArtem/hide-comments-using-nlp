# hide-comments-using-nlp
Сокрытие комментариев с помощью BiLSTM/Flask

Этот код представляет серверную часть приложения по сокрытию комментариев к товарам facebook (исполняемый файл - \server\hide_comma_server.py)
Код посылает api запросы к facebook странице, получает список комментариев и валидирует их при помощи нейронной сети (отдельно идет классификация по недовольсту продуктом и токсичностью комментария). Отчет о проделанной работе возвращается в Json формате на страничку фронта

Для экономии места единиственные экземпляры данных хранятся в папках model\negative и model\toxic без каких-либо промежуточных файлов.

Все те варианты моделей, что были хуже представленной в корне папки model хранятся в папке model\old_methods (в том числе и модель fasttext - meta)

Для запуска и использования модели все необходимые файлы хранятся в корне model, а зависимости - в файле requirements.txt.
Пример использования можно увидеть в \server\hide_comma_server.py
