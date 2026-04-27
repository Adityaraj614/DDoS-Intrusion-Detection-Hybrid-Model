# phase5_cnn_lstm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load PCA data (IMPORTANT: use PCA here)
X_train = np.load("data/X_train_pca.npy")
X_test = np.load("data/X_test_pca.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32)
y_test = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32)

# Add channel dimension for CNN
X_train = X_train.unsqueeze(2)
X_test = X_test.unsqueeze(2)

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channel, seq)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)  # (batch, seq, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = CNN_LSTM().to(device)

# Loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training
epochs = 5
batch_size = 512

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X_train[idx]
        batch_y = y_train[idx].unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X_train[idx]
        batch_y = y_train[idx].unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/cnn_lstm.pth")