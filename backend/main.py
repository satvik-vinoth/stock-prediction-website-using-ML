# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from stock_utils import fetch_stock_data
from inference import predict_next_close


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stock/{symbol}")
def get_stock_data(symbol: str):
    data = fetch_stock_data(symbol)
    if data is None:
        raise HTTPException(status_code=404, detail="Stock data not found.")
    return data

@app.get("/predict/{model}")
def predict_stock(model: str, symbol: str):
    prediction = predict_next_close(symbol, model) 
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction failed.")
    return prediction