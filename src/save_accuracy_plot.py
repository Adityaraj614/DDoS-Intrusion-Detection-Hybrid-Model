import matplotlib.pyplot as plt

models = ['RF', 'SVM', 'CNN-LSTM', 'MLP', 'Enhanced', 'Proposed']
accuracy = [0.77, 0.78, 0.76, 0.78, 0.79, 0.86]

plt.figure(figsize=(7,5))
plt.plot(models, accuracy, marker='o')

plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.savefig("results/accuracy_plot.png")
plt.close()

print("Saved accuracy_plot.png")