# stock_utils.py

import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from db.mongo import stock_collection
from datetime import datetime

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


async def fetch_stock_data(symbol: str):
    symbol = symbol.upper()
    today = datetime.today().strftime("%Y-%m-%d")
    cached = await stock_collection.find_one({"symbol": symbol, "date": today})
    if cached:
        return dict(list(cached["data"].items())[-30:])
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
        ohlc_data["RSI_14"] = compute_rsi(ohlc_data["Close"], 14)

        macd_line, signal_line, hist = compute_macd(ohlc_data["Close"])
        ohlc_data["MACD_12_26_9"] = macd_line

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

        final_data = ohlc_data[[col for col in final_columns if col in ohlc_data.columns]].dropna()
        final_dict = {k.strftime("%Y-%m-%d"): v for k, v in final_data.to_dict(orient="index").items()}

        # 2. Store in MongoDB
        await stock_collection.insert_one({
            "symbol": symbol,
            "date": today,
            "data": final_dict
        })

        return dict(list(final_dict.items())[-30:])




    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
async def fetch_stock_data_training(symbol: str):
    symbol = symbol.upper()
    today = datetime.today().strftime("%Y-%m-%d")
    cached = await stock_collection.find_one({"symbol": symbol, "date": today})
    if cached:
        return cached["data"]
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
        ohlc_data["RSI_14"] = compute_rsi(ohlc_data["Close"], 14)

        macd_line, signal_line, hist = compute_macd(ohlc_data["Close"])
        ohlc_data["MACD_12_26_9"] = macd_line

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

        final_data = ohlc_data[[col for col in final_columns if col in ohlc_data.columns]].dropna()
        final_dict = {k.strftime("%Y-%m-%d"): v for k, v in final_data.to_dict(orient="index").items()}

        await stock_collection.insert_one({
            "symbol": symbol,
            "date": today,
            "data": final_dict
        })

        return final_dict


    except Exception as e:
        print(f"Error fetching data: {e}")
        return None



async def prepare_close_sequence(symbol: str, seq_length=60):
    data_dict = await fetch_stock_data_training(symbol)
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