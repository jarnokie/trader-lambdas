import datetime
import torch
import numpy as np
import pytz

from alpaca.data import StockHistoricalDataClient
from alpaca.trading import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import BarSet
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order

TICKERS = {
    "aapl": {"shares": 15730000000},
    "adbe": {"shares": 455800000},
    "amd": {"shares": 1610000000},
    "amzn": {"shares": 10260000000},
    "asml": {"shares": 394590000},
    "avgo": {"shares": 412680000},
    "cost": {"shares": 443150000},
    "gme": {"shares": 304750000},
    "goog": {"shares": 5870000000},
    "jnj": {"shares": 2600000000},
    "ko": {"shares": 4320000000},
    "meta": {"shares": 2190000000},
    "mmm": {"shares": 551670000},
    "msft": {"shares": 7430000000},
    "nflx": {"shares": 444540000},
    "nvda": {"shares": 2470000000},
    "pep": {"shares": 1370000000},
    "pg": {"shares": 2360000000},
    "t": {"shares": 7150000000},
    "tsla": {"shares": 3170000000},
    "wba": {"shares": 863260000},
    "wmt": {"shares": 2690000000},
    "xom": {"shares": 4040000000},
}

HOLIDAYS = {
    "2023-01-02": "new year's eve",
    "2023-01-16": "birthday of martin luther king jr",
    "2023-02-20": "president's day",
    "2023-04-07": "good friday",
    "2023-05-29": "memorial day",
    "2023-06-19": "juneteenth",
    "2023-07-04": "indipendence day",
    "2023-09-04": "labor day",
    "2023-11-23": "thanksgiving day",
    "2023-12-25": "christmas",
}


def get_weekday(date: datetime.datetime) -> torch.Tensor:
    next_trading_day = get_next_trading_day(
        date,
        list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), HOLIDAYS.keys())),
    )
    w = np.zeros(7, dtype=np.float32)
    w[next_trading_day.weekday()] = 1
    return torch.from_numpy(w).to(torch.float32)


def get_next_trading_day(
    current_date: datetime.datetime, holidays: list[datetime.datetime]
) -> datetime.datetime:
    c = _strip_time(current_date) + datetime.timedelta(days=1)
    while c.weekday() > 4 or c in holidays:
        c += datetime.timedelta(days=1)
    return c


def get_daysofyear(date: datetime.datetime) -> torch.Tensor:
    d = np.array([float(date.strftime("%j")) / 365.0], dtype=np.float32)
    return torch.from_numpy(d).to(torch.float32)


def _strip_time(date: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(date.year, date.month, date.day)


def get_prices(symbols: dict, alpaca_key: str, alpaca_secret: str) -> dict:
    client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
    tickers = list(symbols.keys())
    tickers.append("SPY")
    tickers.append("VIXY")
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=_strip_time(
            datetime.datetime.now(pytz.timezone("America/New_York"))
            - datetime.timedelta(days=10)
        ),
    )
    result = {}
    bars = client.get_stock_bars(request)
    assert type(bars) is BarSet
    for symbol, data in bars.dict().items():
        last = data[0]["close"]
        result[symbol] = []
        for bar in data:
            result[symbol].append(
                (
                    (bar["close"] / last - 1) * 100,
                    bar["volume"] / symbols[symbol]["shares"]
                    if symbol in symbols and "shares" in symbols[symbol]
                    else 0,
                )
            )
            last = bar["close"]
        result[symbol] = np.array(result[symbol])
    return result


def predict(model: torch.nn.Module, alpaca_key: str, alpaca_secret: str) -> dict:
    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    weekday = get_weekday(now)
    dayofyear = get_daysofyear(now)
    tickers: list[str] = list(TICKERS.keys())
    prices = get_prices(TICKERS, alpaca_key, alpaca_secret)
    price_tensors = []
    weekday_tensors = torch.from_numpy(np.array([weekday] * len(tickers)))
    daysofyear_tensors = torch.from_numpy(np.array([dayofyear] * len(tickers)))
    with torch.no_grad():
        for ticker in tickers:
            price_array = np.array(
                [
                    [a[0], b[0], c[0], a[1]]
                    for a, b, c in zip(
                        prices[ticker.upper()], prices["SPY"], prices["VIXY"]
                    )
                ][-5:]
            )
            price_tensors.append(price_array)
    price_tensors = torch.from_numpy(np.array(price_tensors)).to(torch.float32)
    yp = model(price_tensors, weekday_tensors, daysofyear_tensors)
    return {ticker: prediction[0] for ticker, prediction in zip(tickers, yp.tolist())}


def open_trades_positions(
    client: TradingClient,
    predictions: dict[str, float],
    trading_limit: float,
    notional=100.0,
) -> list[Order]:
    orders = []
    for symbol, prediction in predictions.items():
        if prediction > trading_limit:
            order_data = MarketOrderRequest(
                symbol=symbol.upper(),
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            orders.append(client.submit_order(order_data))
    return orders
