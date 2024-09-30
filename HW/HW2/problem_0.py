import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return np.mean((y_pred >= 0.5) == y_true)

# Generate 3 sets of synthetic data
np.random.seed(42)
n_samples = 20

data_sets = []
for _ in range(3):
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.rand(n_samples)
    
    # Adjust y_pred to create anti-correlation
    for i in range(n_samples):
        if y_true[i] == 1:
            y_pred[i] = 1 - y_pred[i]**2  # Push predictions for positive samples towards 0
        else:
            y_pred[i] = y_pred[i]**2  # Push predictions for negative samples towards 1
    
    data_sets.append((y_true, y_pred))

# Calculate loss and accuracy for each dataset
results = []
for y_true, y_pred in data_sets:
    loss = cross_entropy(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    results.append((loss, acc))

# Plotting
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i, (loss, acc) in enumerate(results):
    plt.scatter(loss, acc, c=colors[i], label=f'Dataset {i+1}')

plt.xlabel('Cross-Entropy Loss')
plt.ylabel('Accuracy')
plt.title('Cross-Entropy Loss vs Accuracy (Anti-correlated)')
plt.legend()
plt.grid(True)
# plt.show()

print("Results:")
for i, (loss, acc) in enumerate(results):
    print(f"Dataset {i+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
    
# Show the predictions for the first dataset
print("\nPredictions for Dataset 1:")
print("True labels:", data_sets[0][0])
print("Predictions:", np.round(data_sets[0][1],3))

# Show the predictions for the second dataset
print("\nPredictions for Dataset 2:")
print("True labels:", data_sets[1][0])
print("Predictions:", np.round(data_sets[1][1],3))

# Show the predictions for the third dataset
print("\nPredictions for Dataset 3:")
print("True labels:", data_sets[2][0])
print("Predictions:", np.round(data_sets[2][1],3))