# phase6_evaluation.py

import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -----------------------------
# Load Test Data (PCA version)
# -----------------------------
X_test = np.load("data/X_test_pca.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensor
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load Model (IMPORTANT FIX)
# -----------------------------
from model import CNN_LSTM   # ✅ only import model, NOT training file

model = CNN_LSTM().to(device)
model.load_state_dict(torch.load("models/cnn_lstm.pth", map_location=device))
model.eval()

# Move data to device
X_test = X_test.to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs > 0.5).float().cpu().numpy()

# -----------------------------
# Evaluation
# -----------------------------
print("\n=== CNN-LSTM Model Performance ===\n")

print(classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))