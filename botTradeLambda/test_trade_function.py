import os

import pytest
import datetime

import torch

try:
    from botTradeLambda import Model, get_weekday, predict, TICKERS
except ImportError:
    from lambda_function import Model, get_weekday, predict, TICKERS


def test_weekday():
    """Test get weekday"""
    # For tuesday should return wednesday
    d = datetime.datetime.strptime("2023-08-29", "%Y-%m-%d")
    w = get_weekday(d)
    assert w.size() == torch.Size([7])
    assert w.numpy().tolist() == [0, 0, 1, 0, 0, 0, 0]

    # For friday should return monday
    d = datetime.datetime.strptime("2023-08-25", "%Y-%m-%d")
    w = get_weekday(d)
    assert w.size() == torch.Size([7])
    assert w.numpy().tolist() == [1, 0, 0, 0, 0, 0, 0]


def test_predict():
    alpaca_key = os.environ["ALPACA_KEY"]
    alpaca_secret = os.environ["ALPACA_SECRET"]
    model = Model()
    model.load_state_dict(torch.load("botTradeLambda/model.pt", map_location="cpu"))
    prediction = predict(model, alpaca_key, alpaca_secret)
    for ticker in TICKERS:
        assert ticker in prediction.keys()
        assert type(prediction[ticker]) == float
