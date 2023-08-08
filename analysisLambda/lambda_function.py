import sys
sys.path.insert(0, 'src/vendor/')

import json

def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "hello world",
                            "event": event})
    }

