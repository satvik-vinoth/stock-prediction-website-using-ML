import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_size=256, num_layers=2, dropout=0.3):
        super(LSTMPredictor, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_size, num_layers=1,
                             batch_first=True, dropout=dropout, bidirectional=True)

        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1,
                             batch_first=True, dropout=dropout, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out