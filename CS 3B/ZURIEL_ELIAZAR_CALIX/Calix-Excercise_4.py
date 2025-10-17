import numpy as np

# Activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (X: features, y: labels)
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Example inputs

y = np.array([[0],[1],[1],[0]])  # XOR output (non-linearly separable, for illustration)

# Initialize weights and bias randomly
np.random.seed(1)
weights = 2 * np.random.random((2,1)) - 1
bias = 0.0

# Hyperparameters
learning_rate = 0.1
decay = 0.001   # Learning rate decay per epoch
momentum = 0.9
adaptive = True

# Initialize momentum and adaptive caches
delta_weights_prev = np.zeros_like(weights)
adaptive_cache = np.zeros_like(weights)

epochs = 1000

for epoch in range(epochs):
    # Apply learning rate decay
    lr = learning_rate / (1 + decay * epoch)

    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    # Calculate error
    error = y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)

    # Gradient for weights and bias
    grad_weights = np.dot(X.T, d_output)
    grad_bias = np.sum(d_output)

    # Adaptive gradient (RMSProp-like)
    if adaptive:
        adaptive_cache = 0.9 * adaptive_cache + 0.1 * (grad_weights ** 2)
        adjusted_grad = grad_weights / (np.sqrt(adaptive_cache) + 1e-8)
    else:
        adjusted_grad = grad_weights

    # Momentum update
    delta_weights = momentum * delta_weights_prev + lr * adjusted_grad
    weights += delta_weights
    bias += lr * grad_bias
    delta_weights_prev = delta_weights

    # Print accuracy every 100 epochs
    if (epoch + 1) % 100 == 0:
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y) * 100
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)
