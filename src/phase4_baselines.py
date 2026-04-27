# phase4_baselines.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import preprocessed data (NO PCA)
from phase2_preprocess import X_train, X_test, y_train, y_test

# -----------------------------
# Random Forest
# -----------------------------
print("\n=== Random Forest ===")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

joblib.dump(rf, "models/random_forest.pkl")


# -----------------------------
# SVM
# -----------------------------
print("\n=== SVM ===")

# If slow → change to kernel='linear'
svm = SVC(kernel='rbf', C=10, gamma='scale')

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))

joblib.dump(svm, "models/svm.pkl")