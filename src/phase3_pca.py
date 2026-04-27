# phase3_pca.py

import numpy as np
from sklearn.decomposition import PCA
import joblib

# Load preprocessed data from Phase 2
# (For now we will directly copy variables — later we can modularize)

from phase2_preprocess import X_train, X_test, y_train, y_test

# -----------------------------
# Apply PCA
# -----------------------------
n_components = 30  # you can tune later (25–35)

pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# -----------------------------
# Check variance retained
# -----------------------------
explained_variance = np.sum(pca.explained_variance_ratio_)

print("Original shape:", X_train.shape)
print("Reduced shape:", X_train_pca.shape)
print("Variance retained:", explained_variance)

# -----------------------------
# Save PCA model (important for pipeline)
# -----------------------------
joblib.dump(pca, "models/pca_model.pkl")

# Save processed data (optional but useful)
np.save("data/X_train_pca.npy", X_train_pca)
np.save("data/X_test_pca.npy", X_test_pca)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)