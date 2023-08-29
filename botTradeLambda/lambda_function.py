import sys

sys.path.insert(0, "src/vendor/")

import os
import json

import psycopg2

import torch
import torch.nn.functional as F

try:
    from helpers import predict
except ImportError:
    from .helpers import predict

MODEL_FILE = "model.pt"


def lambda_handler(event, context):
    database = os.environ["PG_DB"]
    host = os.environ["PG_HOST"]
    user = os.environ["PG_USER"]
    password = os.environ["PG_PASSWORD"]
    table = os.environ["PG_TABLE"]
    alpaca_key = os.environ["ALPACA_KEY"]
    alpaca_secret = os.environ["ALPACA_SECRET"]

    model = Model()
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    predictions = predict(model, alpaca_key, alpaca_secret)

    return {
        "statusCode": 200,
        "body": json.dumps({"todo": "todo"}, sort_keys=True, default=str),
    }


def create_connection(database, host, user, password) -> psycopg2.extensions.connection:
    return psycopg2.connect(database=database, host=host, user=user, password=password)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=4, hidden_size=120, num_layers=2, batch_first=True
        )
        self.linear1 = torch.nn.Linear(120 + 7 + 1, 128)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(128, 32)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linear3 = torch.nn.Linear(32, 8)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.linear4 = torch.nn.Linear(8, 1)

    def forward(self, prices, weekday, dayofyear):
        x = self.lstm(prices)[0][:, -1, :]
        x = torch.cat((x, weekday, dayofyear), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)
