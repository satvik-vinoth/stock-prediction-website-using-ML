# gru_model.py

import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_size=256, num_layers=2, dropout=0.3):
        super(GRUPredictor, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
