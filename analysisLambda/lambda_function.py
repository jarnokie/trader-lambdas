import sys
sys.path.insert(0, 'src/vendor/')

from typing import Callable

import json
import requests
import pandas

def lambda_handler(event, context):
    if "pathParameters" in event and "symbol" in event["pathParameters"]:
        symbol = event["pathParameters"]["symbol"]
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "symbol is required"})
        }
    return {
        "statusCode": 200,
        "body": json.dumps({
            "symbol": symbol,
            "data": to_json(get_from_quickfs(lambda: do_request(symbol))[0])
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

def get_from_quickfs(request_f: Callable) -> tuple[pandas.DataFrame, dict]:
    j = request_f()
    return pandas.read_html(j["datasets"]["ovr"], header=0, index_col=0)[0], j["datasets"]["metadata"]

def to_json(quickfs_df: pandas.DataFrame) -> dict[dict]:
    data = {}
    for year in quickfs_df.columns:
        data[year] = {k: v for k, v in data.items() if v is not None}
    return data