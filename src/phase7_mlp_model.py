# phase7_mlp_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# Load PCA Data
# -----------------------------
X_train = np.load("data/X_train_pca.npy")
X_test = np.load("data/X_test_pca.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# -----------------------------
# MLP Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP(X_train.shape[1]).to(device)

# -----------------------------
# Training
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 20
batch_size = 512

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))

    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]

        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs > 0.5).float()

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\n=== MLP Model Performance ===\n")
print(classification_report(y_test.cpu(), preds.cpu()))
print("Accuracy:", accuracy_score(y_test.cpu(), preds.cpu()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test.cpu(), preds.cpu()))

# Save model
torch.save(model.state_dict(), "models/mlp_model.pth")