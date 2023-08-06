import sys
sys.path.insert(0, 'src/vendor/')

import os
import json
import psycopg2

def lambda_handler(event, context):
    database = os.environ["PG_DB"]
    host = os.environ["PG_HOST"]
    user = os.environ["PG_USER"]
    password = os.environ["PG_PASSWORD"]
    table = os.environ["PG_TABLE"]

    return {
        "statusCode": 200,
        "body": json.dumps({"closed": get_closed_trades(database, host, user, password, table)}, sort_keys=True, default=str)
    }

def sort_trades(trades: list[dict]):
    return list(reversed(sorted(trades, key=lambda x: x["opendate"])))

def get_closed_trades(database, host, user, password, table):
    keys = ("id", "symbol", "status", "prediction", "opendate", "openprice", "closedate", "closeprice", "type")
    with create_connection(database, host, user, password) as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE status = FALSE")
        return sort_trades([
            {k: v for k, v in zip(keys, values)} for values in cursor.fetchall()
        ])
    
def create_connection(database, host, user, password) -> psycopg2.extensions.connection:
    return psycopg2.connect(database=database,
                            host=host,
                            user=user,
                            password=password)
