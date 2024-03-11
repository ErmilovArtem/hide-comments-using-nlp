import logging
import time

import requests
import json

file_log = logging.FileHandler('keitaro_data.log', encoding='utf-8')
console_out = logging.StreamHandler()

logging.basicConfig(handlers=(file_log, console_out),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)

# Входные данные
MAIN_URL = "http://unionaff.com/admin_api/v1/"
report_url = MAIN_URL + "report/build" # by post

headers = {
    "Api-Key": "e82c8429749e568b5448ab2ff3a81032"
}

while True:
    for date_elem in ["today", "yesterday", "last_monday", "1_month_ago", "first_day_of_this_year", "all_time"]: # + "all_time"
        logging.info('time interval of report building - %s' % date_elem)
        report_by_date = []
        report_payload = {
            "range": {
                "interval": date_elem,
                "timezone": "Europe/Moscow",
                "from": None,
                "to": None
            },
            "columns": [],
            "metrics": ["profit_confirmed"],
            "grouping": ["source"],
            "sort": [
                {"name": "profit_confirmed", "order": "desc"},
            ],
            "filters": [
                {"name": "offer_group_id", "operator": "EQUALS", "expression": "191"},
            ],
            "summary": True,
            "limit": 1000,
            "offset": 0
        }
        response = requests.post(report_url, headers=headers, json=report_payload)
        response_to_dict = json.loads(response.text)
        for elem in response_to_dict["rows"]:
            report_by_date.append({"id": response_to_dict["rows"].index(elem), "name": elem["source"], "profit": elem["profit_confirmed"]})
        rep_to_save = sorted(report_by_date, key=lambda x: x['profit'], reverse=True)
        with open(date_elem + '_report.json', 'w', encoding="utf-8") as json_file:
            json.dump(rep_to_save, json_file)
    logging.info('!all report done! - %s' % date_elem)
    time.sleep(1200)


