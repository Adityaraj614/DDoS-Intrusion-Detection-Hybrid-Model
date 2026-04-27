# phase9_multiclass.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# SAFE DATA LOADING (FINAL FIX)
# -----------------------------
def load_data(file_path):
    data = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            # last element = difficulty → ignore
            # second last = label
            label = parts[-2]
            features = parts[:-2]

            data.append(features)
            labels.append(label)

    return pd.DataFrame(data), pd.Series(labels)


train_df, train_labels = load_data("data/KDDTrain+.txt")
test_df, test_labels = load_data("data/KDDTest+.txt")

# attach label
train_df['label'] = train_labels
test_df['label'] = test_labels

# -----------------------------
# EXTRACT LABEL (SECOND LAST COLUMN)
# -----------------------------
train_labels = train_df.iloc[:, -2]
test_labels = test_df.iloc[:, -2]

# -----------------------------
# DROP LAST 2 COLUMNS (label + difficulty)
# -----------------------------
train_df = train_df.iloc[:, :-2]
test_df = test_df.iloc[:, :-2]

# -----------------------------
# ADD CLEAN LABEL COLUMN
# -----------------------------
train_df['label'] = train_labels.astype(str)
test_df['label'] = test_labels.astype(str)

# -----------------------------
# ENCODE CATEGORICAL FEATURES
# Columns: 1=protocol, 2=service, 3=flag
# -----------------------------
categorical_cols = [1, 2, 3]

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)

    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# -----------------------------
# ENCODE LABELS (MULTICLASS)
# -----------------------------
label_encoder = LabelEncoder()

combined_labels = pd.concat([train_df['label'], test_df['label']])
label_encoder.fit(combined_labels)

train_df['label'] = label_encoder.transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# -----------------------------
# DEBUG CHECK (MUST BE CORRECT)
# -----------------------------
num_classes = len(label_encoder.classes_)

print("Num classes:", num_classes)
print("Min label:", train_df['label'].min())
print("Max label:", train_df['label'].max())

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# -----------------------------
# NORMALIZATION
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# CONVERT TO TENSOR
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# -----------------------------
# MODEL
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(X_train.shape[1], num_classes).to(device)

# -----------------------------
# TRAINING
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 20
batch_size = 512

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))

    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]

        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -----------------------------
# EVALUATION
# -----------------------------
model.eval()

with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)

print("\n=== FINAL MULTICLASS RESULT ===")
print("Accuracy:", accuracy_score(y_test.cpu(), preds.cpu()))