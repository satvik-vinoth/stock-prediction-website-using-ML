# stock_utils.py

import yfinance as yf
import pandas_ta as ta
from datetime import datetime
import numpy as np
import pandas as pd

def fetch_stock_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol.upper())
        end_date = datetime.today().strftime("%Y-%m-%d")  

        ohlc_data = ticker.history(start="2023-01-01", end=end_date)

        if ohlc_data.empty:
            return None

        ohlc_data = ohlc_data.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close",
            "Volume": "Volume", "Adj Close": "Adj Close"
        })

        ohlc_data.index = ohlc_data.index.tz_localize(None)
        ohlc_data["SMA_50"] = ohlc_data["Close"].rolling(window=50).mean()
        ohlc_data["EMA_50"] = ohlc_data["Close"].ewm(span=50, adjust=False).mean()
        ohlc_data.ta.macd(append=True)
        ohlc_data.ta.rsi(length=14, append=True)

        ohlc_data["OBV"] = ohlc_data["Volume"].copy()
        for i in range(1, len(ohlc_data)):
            if ohlc_data.iloc[i]["Close"] > ohlc_data.iloc[i - 1]["Close"]:
                ohlc_data.iloc[i, ohlc_data.columns.get_loc("OBV")] += ohlc_data.iloc[i - 1]["OBV"]
            elif ohlc_data.iloc[i]["Close"] < ohlc_data.iloc[i - 1]["Close"]:
                ohlc_data.iloc[i, ohlc_data.columns.get_loc("OBV")] -= ohlc_data.iloc[i - 1]["OBV"]

        ohlc_data["Money Flow Multiplier"] = ((ohlc_data["Close"] - ohlc_data["Low"]) - (ohlc_data["High"] - ohlc_data["Close"])) / (ohlc_data["High"] - ohlc_data["Low"])
        ohlc_data["Money Flow Volume"] = ohlc_data["Money Flow Multiplier"] * ohlc_data["Volume"]
        ohlc_data["ADI"] = ohlc_data["Money Flow Volume"].cumsum()

        final_columns = [
            "Open", "High", "Low", "Close", "Volume", "Adj Close",
            "SMA_50", "EMA_50", "MACD_12_26_9", "RSI_14",
            "OBV", "ADI"
        ]

        ohlc_data = ohlc_data[[col for col in final_columns if col in ohlc_data.columns]].dropna()
        return ohlc_data.tail(30).to_dict(orient="index")

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
def fetch_stock_data_training(symbol: str):
    try:
        ticker = yf.Ticker(symbol.upper())
        end_date = datetime.today().strftime("%Y-%m-%d")  

        ohlc_data = ticker.history(start="2023-01-01", end=end_date)

        if ohlc_data.empty:
            return None

        ohlc_data = ohlc_data.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close",
            "Volume": "Volume", "Adj Close": "Adj Close"
        })

        ohlc_data.index = ohlc_data.index.tz_localize(None)
        ohlc_data["SMA_50"] = ohlc_data["Close"].rolling(window=50).mean()
        ohlc_data["EMA_50"] = ohlc_data["Close"].ewm(span=50, adjust=False).mean()
        ohlc_data.ta.macd(append=True)
        ohlc_data.ta.rsi(length=14, append=True)

        ohlc_data["OBV"] = ohlc_data["Volume"].copy()
        for i in range(1, len(ohlc_data)):
            if ohlc_data.iloc[i]["Close"] > ohlc_data.iloc[i - 1]["Close"]:
                ohlc_data.iloc[i, ohlc_data.columns.get_loc("OBV")] += ohlc_data.iloc[i - 1]["OBV"]
            elif ohlc_data.iloc[i]["Close"] < ohlc_data.iloc[i - 1]["Close"]:
                ohlc_data.iloc[i, ohlc_data.columns.get_loc("OBV")] -= ohlc_data.iloc[i - 1]["OBV"]

        ohlc_data["Money Flow Multiplier"] = ((ohlc_data["Close"] - ohlc_data["Low"]) - (ohlc_data["High"] - ohlc_data["Close"])) / (ohlc_data["High"] - ohlc_data["Low"])
        ohlc_data["Money Flow Volume"] = ohlc_data["Money Flow Multiplier"] * ohlc_data["Volume"]
        ohlc_data["ADI"] = ohlc_data["Money Flow Volume"].cumsum()

        final_columns = [
            "Open", "High", "Low", "Close", "Volume", "Adj Close",
            "SMA_50", "EMA_50", "MACD_12_26_9", "RSI_14",
            "OBV", "ADI"
        ]

        ohlc_data = ohlc_data[[col for col in final_columns if col in ohlc_data.columns]].dropna()
        return ohlc_data.to_dict(orient="index")


    except Exception as e:
        print(f"Error fetching data: {e}")
        return None



def prepare_close_sequence(symbol: str, seq_length=60):
    data_dict = fetch_stock_data_training(symbol)
    if data_dict is None:
        return None, None, None

    df = pd.DataFrame(data_dict).T
    close = df["Close"].values.reshape(-1, 1)

    data_min = close.min()
    data_max = close.max()
    scaled = (close - data_min) / (data_max - data_min)

    if len(scaled) < seq_length:
        return None, None, None
    return scaled, data_min, data_max,close