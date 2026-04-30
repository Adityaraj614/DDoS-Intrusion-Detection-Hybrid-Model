import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# =========================
# 1. Accuracy Comparison
# =========================
models = ['RF', 'SVM', 'CNN-LSTM', 'MLP', 'Enhanced', 'Proposed']
accuracy = [0.77, 0.78, 0.76, 0.78, 0.79, 0.86]

plt.figure()
plt.plot(models, accuracy, marker='o')
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.savefig("results/accuracy_plot.png")
plt.close()


# =========================
# 2. Confusion Matrix
# =========================
import seaborn as sns

cm = np.array([[9430, 281],
               [4655, 8178]])

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()


# =========================
# 3. Training Loss Plot
# =========================
loss = [19.7, 8.2, 6.9, 6.3, 5.9, 5.6, 5.8, 5.3, 4.8, 4.7,
        4.4, 4.5, 4.2, 4.0, 4.2, 4.2, 3.8, 3.6, 3.5, 3.7]

epochs = list(range(1, 21))

plt.figure()
plt.plot(epochs, loss)
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("results/loss_plot.png")
plt.close()


# =========================
# 4. Precision / Recall Plot
# =========================
metrics = ['Precision', 'Recall', 'F1-score']
normal = [0.68, 0.97, 0.80]
attack = [0.97, 0.65, 0.78]

x = np.arange(len(metrics))

plt.figure()
plt.bar(x - 0.2, normal, width=0.4, label='Normal')
plt.bar(x + 0.2, attack, width=0.4, label='Attack')

plt.xticks(x, metrics)
plt.title("Precision, Recall, F1 Comparison")
plt.legend()
plt.savefig("results/metrics_plot.png")
plt.close()


# =========================
# 5. PCA Variance Plot (OPTIONAL)
# =========================
try:
    X_train = np.load("data/X_train.npy")

    pca = PCA()
    pca.fit(X_train)

    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.title("PCA Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Variance")
    plt.savefig("results/pca_variance.png")
    plt.close()

except:
    print("PCA plot skipped (data not found)")


print("All plots saved in results/")