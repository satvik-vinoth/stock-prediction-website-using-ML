import os
import torch
from datetime import datetime
from stock_utils import prepare_close_sequence
from trainer import train_model
from gru_model import GRUPredictor
from transformer_model import TransformerPredictor, TransformerConfig
from lstm_model import LSTMPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_class(model: str):
    if model == "gru":
        return GRUPredictor, {}  
    elif model == "lstm":
        return LSTMPredictor, {}
    elif model == "transformer":
        config = TransformerConfig(hidden_size=64,
                                        num_hidden_layers=2,
                                        num_attention_heads=8,
                                        intermediate_size=128,
                                        hidden_dropout_prob=0.1,
                                        max_position_embeddings=1000)
        return TransformerPredictor, {
            "config": config,
            "input_dim": 1,
            "seq_length": 1000
        }
    else:
        raise ValueError("Invalid model name")

def get_model_path(symbol: str, model: str):
    today_str = datetime.today().strftime("%Y%m%d")
    return f"model_weights/{symbol.upper()}_{model.lower()}_{today_str}.pth"

def delete_old_models(symbol: str, model: str):
    for file in os.listdir("model_weights"):
        if file.startswith(f"{symbol.upper()}_{model.lower()}") and not file.endswith(f"{datetime.today().strftime('%Y%m%d')}.pth"):
            os.remove(os.path.join("model_weights", file))

def predict_next_close(symbol: str, model: str):
    model_path = get_model_path(symbol, model)

    # Step 1: Train if today's model doesn't exist
    if not os.path.exists(model_path):
        print(f"📢 No model found for {symbol}-{model} today. Retraining...")
        train_model(symbol, model)
        delete_old_models(symbol, model)
        if not os.path.exists(model_path):
            return None

    # Step 2: Get data
    seq, data_min, data_max, all_y_true = prepare_close_sequence(symbol)
    print(seq.shape)  # Should print: (num_samples, 1) (only feature sequence)

    if seq is None:
        return None

    # Prepare X and y
    X = []
    y = []
    for i in range(0, len(seq) - 60):  # Use 60 days as the sequence length
        X.append(seq[i:i+60])  # X: 60 days of stock prices
        y.append(seq[i+60, 0])  # y: next day's stock price (target)

    X = np.array(X)
    y = np.array(y)

    # Convert to tensor
    input_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    print(input_tensor.shape)  # Check if the shape is correct

    # Step 3: Load model and predict
    model_class, model_kwargs = get_model_class(model)
    net = model_class(**model_kwargs).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Create empty list to store predictions
    predictions = []

    with torch.no_grad():
        for i in range(len(X)):
            pred = net(input_tensor[i:i+1])  # Predict for one sequence at a time
            predictions.append(pred.item())  # Collect predicted values

    # Inverse scaling
    predicted_prices = np.array(predictions) * (data_max - data_min) + data_min
    actual_prices = all_y_true

    # Calculate RMSE & MAPE (on last n samples)
    rmse = np.sqrt(mean_squared_error(actual_prices[-60:], predicted_prices[-60:]))
    mape = np.mean(np.abs((actual_prices[-60:] - predicted_prices[-60:]) / actual_prices[-60:])) * 100

    return {
        "symbol": symbol.upper(),
        "model": model.upper(),
        "predicted_close": round(predicted_prices[-1], 2),  # Next day's predicted close
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "recent_actual": actual_prices[-60:].tolist(),
        "recent_predicted": predicted_prices[-60:].tolist()
    }
