# main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from stock_utils import fetch_stock_data
from inference import predict_next_close
from routes import auth
from utils.deps import get_current_user



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://stock-prediction-g3d2yvak6-satvik-vinoths-projects.vercel.app","https://stock-prediction-git-main-satvik-vinoths-projects.vercel.app","https://stock-vision-prediction.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    data = await fetch_stock_data(symbol)
    if data is None:
        raise HTTPException(status_code=404, detail="Stock data not found.")
    return data

@app.get("/predict/{model}")
async def predict_stock(model: str, symbol: str, user=Depends(get_current_user)):
    prediction = await predict_next_close(symbol, model) 
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction failed.")
    return {
        "prediction": prediction,
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}