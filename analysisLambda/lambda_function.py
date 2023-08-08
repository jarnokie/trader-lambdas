import sys
sys.path.insert(0, 'src/vendor/')

import json

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
        "body": json.dumps({"symbol": symbol})
    }

