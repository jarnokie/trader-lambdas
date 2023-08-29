import os
import logging
import psycopg2
import datetime
import random

_trades_table_def = """
CREATE TABLE IF NOT EXISTS %s (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    status BOOLEAN,
    prediction REAL,
    opendate TIMESTAMP NOT NULL,
    openprice REAL NOT NULL,
    closedate TIMESTAMP,
    closeprice REAL,
    type BOOLEAN,
    model VARCHAR(16)
);
"""

_trade_insert_def = """
INSERT INTO #TABLE# (symbol, status, opendate, openprice, type, prediction, model, closedate, closeprice) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

def get_table_name() -> str:
    return f"test_table_{int(datetime.datetime.utcnow().timestamp())}_{random.randint(1, 10000):05d}"

def create_connection(database, host, user, password) -> psycopg2.extensions.connection:
    return psycopg2.connect(database=database,
                            host=host,
                            user=user,
                            password=password)

def load_test_trades():
    result = []
    with open("test_trades.csv", "r") as f:
        header = []
        for line in f:
            if not header:
                header = line.split(",")
                continue
            s = line.split(",")
            result.append({k.strip(): v.strip() for k, v in zip(header, s)})
    return result

def create_trades_table(table: str):
    test_trades = load_test_trades()
    commands = [(_trades_table_def % table, None)]
    for trade in test_trades:
        commands.append((_trade_insert_def.replace("#TABLE#", table), (
            trade["symbol"] if trade["symbol"] else None,
            trade["status"] if trade["status"] else None,
            trade["opendate"] if trade["opendate"] else None,
            trade["openprice"] if trade["openprice"] else None,
            trade["type"] if trade["type"] else None,
            trade["prediction"] if trade["prediction"] else None,
            trade["model"] if trade["model"] else None,
            trade["closedate"] if trade["closedate"] else None,
            trade["closeprice"] if trade["closeprice"] else None)))
    execute_sql(commands)
    
def get_database_config() -> dict:
    if "PG_HOST" not in os.environ:
        logging.error("PG_HOST not set!")
        assert False
    if "PG_USER" not in os.environ:
        logging.error("PG_USER not set!")
        assert False
    if "PG_DB" not in os.environ:
        logging.error("PG_DB not set!")
        assert False
    if "PG_PASSWORD" not in os.environ:
        logging.error("PG_PASSWORD not set!")
        assert False
    return {
        "PG_HOST": os.environ["PG_HOST"],
        "PG_USER": os.environ["PG_USER"],
        "PG_DB": os.environ["PG_DB"],
        "PG_PASSWORD": os.environ["PG_PASSWORD"]
    }

def execute_sql(commands: list[str]):

    db_config = get_database_config()

    with create_connection(db_config["PG_DB"], db_config["PG_HOST"], db_config["PG_USER"], db_config["PG_PASSWORD"]) as connection:
        cursor = connection.cursor()
        for command in commands:
            cursor.execute(command[0], command[1])
        connection.commit()
