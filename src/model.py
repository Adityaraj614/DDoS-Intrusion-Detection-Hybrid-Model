# model.py

import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # Better CNN
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.conv2 = nn.Conv1d(64, 128, 3)

        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(128, 64, batch_first=True)

        # Fully connected
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x