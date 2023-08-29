import sys
sys.path.insert(0, 'src/vendor/')

from typing import Callable

import json
import requests
from lxml import html


def lambda_handler(event, context):
    if "pathParameters" in event and "symbol" in event["pathParameters"]:
        symbol = event["pathParameters"]["symbol"]
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "symbol is required"})
        }
    fs, ks = get_from_quickfs(lambda: do_request(symbol))
    return {
        "statusCode": 200,
        "body": json.dumps({
            "symbol": symbol,
            "fs": fs,
            "ks": ks
            })
    }


def do_request(symbol: str) -> dict:
    url = f"https://api.quickfs.net/stocks/{symbol}/ovr/Annual/?sortOrder=ASC&maxPeriods=11"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        r.raise_for_status()
    return r.json()


def get_from_quickfs(request_f: Callable) -> tuple[dict, dict]:
    j = request_f()
    return parse_html_table_to_dict(j["datasets"]["ovr"]), j["datasets"]["metadata"]


def parse_html_table_to_dict(table_html):
    tree = html.fromstring(table_html)
    rows = tree.xpath('//tr')
    
    if not rows:
        return None
    
    years = [header.text_content().strip() for header in rows[0].xpath('td|th')][1:]
    years = [int(y) for y in years]
    row_headers = []
    data = []
    
    for row in rows[1:]:
        row_data = [cell.text_content().strip() for cell in row.xpath('td')]
        if row_data:
            row_headers.append(row_data[0])
            data.append(row_data[1:])
    
    result = {}

    for row_title, row in zip(row_headers, data):
        result[row_title] = dict(zip(years, row))
    
    return result
