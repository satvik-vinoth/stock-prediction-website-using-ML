# Stock-prediction-website-using-ML

This project is a stock price prediction website that uses machine learning (ML) models such as Transformer, GRU, and LSTM to forecast future stock prices. It is designed to help users predict stock trends in real-time through a web-based interface.

## Models Used

1. **Transformer**: A deep learning model that uses self-attention mechanisms for sequential data, making it highly effective for time-series forecasting.
2. **GRU (Gated Recurrent Unit)**: A type of recurrent neural network (RNN) that is effective for sequence prediction, especially in financial data.
3. **LSTM (Long Short-Term Memory)**: Another variant of RNNs designed to handle long-range dependencies, making it ideal for stock price prediction.

## Tech Stack

### Frontend
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white) ![Chart.js](https://img.shields.io/badge/Chart.js-F6B23E?style=flat&logo=chart.js&logoColor=black)

### Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

# Running the Stock Prediction Website

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/satvik-vinoth/stock-prediction-website-using-ML.git
```
### Step 2: Naviagte to backend
```
cd stock-prediction-website-using-ML/backend
```
### Run Backend server
```
uvicorn main:app --reload
```

### Navigate to frontend
```
cd stock-prediction-website-using-ML/frontend
```
### Run the website
```
npm run dev
```






