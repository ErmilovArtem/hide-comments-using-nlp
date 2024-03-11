import logging

import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

file_log = logging.FileHandler('../../../../PycharmProjects/facebook/keitaro_server.log', encoding='utf-8')
console_out = logging.StreamHandler()

logging.basicConfig(handlers=(file_log, console_out),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)


@app.route('/hello', methods=['get'])
def hello():
    return jsonify("hello")

@app.route('/get_all_user_and_profit', methods=['get'])
def get_all_user_and_profit():
    period = request.args.get('period')  # Чтение параметра запроса 'period'
    logging.info('request to /get_all_user_and_profit with - %s' % period)
    list_of_non_valid = ["all_time", "lastweek", "lastmonth", "lastyear"] # today and yesterday is valid anyway
    list_of_valid = ["all_time", "last_monday", "1_month_ago", "first_day_of_this_year"]

    if period in list_of_non_valid:
        period = list_of_valid[list_of_non_valid.index(period)]

    try:
        # Чтение данных из файла (например, data.json)
        with open(period + "_report.json", 'r') as file:
            data = file.read()
        logging.info('SUCCESS - %s' % period)
        # Возвращение данных как JSON-ответ
        return jsonify({"success": True, "data": data})

    except Exception as e:
        logging.info('FAILE is non valid period?- %s', str(e))
        return jsonify({"success": False, "error": 'non valid period. Valid is - "all time"/"all_time", "last week"/"last_monday_report", "last month"/"1_month_ago_report", "last year"/"first_day_of_this_year_report" ,today , yesterday)'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)