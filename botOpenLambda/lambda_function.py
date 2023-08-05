import sys
sys.path.insert(0, 'src/vendor')

import os
os.environ["LD_LIBRARY_PATH"] = "./src/vendor/psycopg2_binary.libs/"

import json
import psycopg2

def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Hello from trader!"})
    }
