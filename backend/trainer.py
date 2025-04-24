from gru_model import GRUPredictor
from lstm_model import LSTMPredictor
from transformer_model import TransformerPredictor, TransformerConfig
import torch
import numpy as np
from datetime import datetime
from stock_utils import prepare_close_sequence

def train_model(symbol: str, model_type: str):
    print(f"📈 Training {model_type.upper()} model for {symbol.upper()}")

    seq, data_min, data_max,y_all= prepare_close_sequence(symbol)
    print(seq.shape)
    if seq is None or len(seq) < 61:
        print("❌ Not enough data to train.")
        return
    X = []
    y = []
    for i in range(0, len(seq) - 60):
        X.append(seq[i:i+60])
        y.append(seq[i+60, 0])
    X = np.array(X)
    y = np.array(y)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "gru":
        model = GRUPredictor()
    elif model_type == "lstm":
        model = LSTMPredictor()
    elif model_type == "transformer":
        trans_config = TransformerConfig(hidden_size=64,
                                        num_hidden_layers=2,
                                        num_attention_heads=8,
                                        intermediate_size=128,
                                        hidden_dropout_prob=0.1,
                                        max_position_embeddings=1000)

        model = TransformerPredictor(trans_config, input_dim=1, seq_length=1000)

    else:
        raise ValueError("Invalid model type")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    batch_size=4
    model.train()
    for epoch in range(50):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train[indices].to(device)
            batch_y = y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    today_str = datetime.today().strftime("%Y%m%d")
    save_path = f"model_weights/{symbol.upper()}_{model_type.lower()}_{today_str}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved model at {save_path}")
