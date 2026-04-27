# phase5_cnn_lstm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import model (IMPORTANT)
from model import CNN_LSTM

# -----------------------------
# Load Data (PCA version)
# -----------------------------
X_train = np.load("data/X_train_pca.npy")
y_train = np.load("data/y_train.npy")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Add channel dimension → (batch, seq_len, 1)
X_train = X_train.unsqueeze(2)

# -----------------------------
# Device (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train = X_train.to(device)
y_train = y_train.to(device)

# -----------------------------
# Model
# -----------------------------
model = CNN_LSTM().to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# -----------------------------
# Training Parameters
# -----------------------------
epochs = 20
batch_size = 512

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))

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

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "models/cnn_lstm.pth")

print("\nModel training complete and saved!")