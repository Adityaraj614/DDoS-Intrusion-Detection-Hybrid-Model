import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Your final confusion matrix
cm = np.array([[9430, 281],
               [4655, 8178]])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png")
plt.close()

print("Saved confusion_matrix.png")