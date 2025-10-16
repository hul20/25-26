import numpy as np

# ============================================================
# LAYER CLASS
# ============================================================

class Layer_Dense:
    """
    Dense (fully connected) layer with random weight initialization.
    
    Attributes:
        weights: weight matrix of shape (n_inputs, n_neurons)
        biases: bias vector of shape (1, n_neurons)
        inputs: stored inputs for backward pass
        output: stored output for backward pass
        dweights: gradient of loss w.r.t. weights
        dbiases: gradient of loss w.r.t. biases
        dinputs: gradient of loss w.r.t. inputs
    """
    
    def __init__(self, n_inputs, n_neurons, name=None):
        """
        Initialize layer with random weights and zero biases.
        
        Parameters:
            n_inputs: number of input features
            n_neurons: number of neurons in the layer
            name: optional name for the layer
        """
        # Initialize weights with small random values (He initialization scaled down)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zero
        self.biases = np.zeros((1, n_neurons))
        self.name = name or f"Dense_Layer({n_inputs}→{n_neurons})"
        
        print(f"[INFO] Initialized {self.name}")
        print(f"       >> weights.shape = {self.weights.shape}, biases.shape = {self.biases.shape}\n")
    
    def forward(self, inputs):
        """
        Forward pass: compute output = inputs · weights + biases
        
        Parameters:
            inputs: input data of shape (batch_size, n_inputs) or (n_inputs,)
        
        Returns:
            output: layer output of shape (batch_size, n_neurons)
        """
        # Store inputs for backward pass
        self.inputs = inputs
        # Compute output
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, dvalues):
        """
        Backward pass: compute gradients w.r.t. weights, biases, and inputs.
        
        Parameters:
            dvalues: gradient of loss w.r.t. layer output (batch_size, n_neurons)
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

class ActivationLinear:
    """
    Linear (identity) activation function.
    f(x) = x
    f'(x) = 1
    """
    
    def __init__(self, name="Linear"):
        self.name = name
    
    def forward(self, inputs):
        """Forward pass: output = input"""
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        """Backward pass: gradient passes through unchanged"""
        self.dinputs = dvalues.copy()


class ActivationSigmoid:
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + e^(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    def __init__(self, name="Sigmoid"):
        self.name = name
    
    def forward(self, inputs):
        """Forward pass: apply sigmoid function"""
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, dvalues):
        """Backward pass: compute gradient using sigmoid derivative"""
        # Derivative: sigmoid * (1 - sigmoid)
        self.dinputs = dvalues * self.output * (1 - self.output)


class ActivationTanh:
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    f'(x) = 1 - tanh²(x)
    """
    
    def __init__(self, name="Tanh"):
        self.name = name
    
    def forward(self, inputs):
        """Forward pass: apply tanh function"""
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output
    
    def backward(self, dvalues):
        """Backward pass: compute gradient using tanh derivative"""
        # Derivative: 1 - tanh²(x)
        self.dinputs = dvalues * (1 - self.output ** 2)


class ActivationReLU:
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    f'(x) = 1 if x > 0, else 0
    """
    
    def __init__(self, name="ReLU"):
        self.name = name
    
    def forward(self, inputs):
        """Forward pass: apply ReLU function"""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        """Backward pass: compute gradient using ReLU derivative"""
        # Derivative: 1 where input > 0, else 0
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    """
    Softmax activation function for multi-class classification.
    f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Converts logits to probability distribution.
    """
    
    def __init__(self, name="Softmax"):
        self.name = name
    
    def forward(self, inputs):
        """Forward pass: apply softmax function"""
        self.inputs = inputs
        # Subtract max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        """Backward pass: compute gradient using softmax derivative"""
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Calculate gradient for each sample
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class LossMSE:
    """
    Mean Squared Error loss function.
    L = (1/n) * Σ(y_true - y_pred)²
    
    Commonly used for regression problems.
    """
    
    def __init__(self, name="MSE"):
        self.name = name
    
    def forward(self, y_pred, y_true):
        """
        Forward pass: calculate MSE loss.
        
        Parameters:
            y_pred: predicted values
            y_true: true values
        
        Returns:
            loss: mean squared error
        """
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return np.mean(sample_losses)
    
    def backward(self, y_pred, y_true):
        """
        Backward pass: compute gradient of MSE loss.
        
        Parameters:
            y_pred: predicted values
            y_true: true values
        """
        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        outputs = len(y_pred[0])
        
        # Gradient: -2/n * (y_true - y_pred)
        self.dinputs = -2 * (y_true - y_pred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossBinaryCrossentropy:
    """
    Binary Cross-Entropy loss function.
    L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
    
    Used for binary classification problems.
    """
    
    def __init__(self, name="BinaryCrossentropy"):
        self.name = name
    
    def forward(self, y_pred, y_true):
        """
        Forward pass: calculate binary cross-entropy loss.
        
        Parameters:
            y_pred: predicted probabilities
            y_true: true binary labels
        
        Returns:
            loss: binary cross-entropy
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + 
                         (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        
        return np.mean(sample_losses)
    
    def backward(self, y_pred, y_true):
        """
        Backward pass: compute gradient of binary cross-entropy loss.
        
        Parameters:
            y_pred: predicted probabilities
            y_true: true binary labels
        """
        # Number of samples
        samples = len(y_pred)
        # Number of outputs
        outputs = len(y_pred[0])
        
        # Clip predictions to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -(y_true / y_pred_clipped - 
                        (1 - y_true) / (1 - y_pred_clipped)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossCategoricalCrossentropy:
    """
    Categorical Cross-Entropy loss function.
    L = -Σ(y_true * log(y_pred))
    
    Used for multi-class classification problems with one-hot encoded labels.
    """
    
    def __init__(self, name="CategoricalCrossentropy"):
        self.name = name
    
    def forward(self, y_pred, y_true):
        """
        Forward pass: calculate categorical cross-entropy loss.
        
        Parameters:
            y_pred: predicted probabilities (after softmax)
            y_true: true labels (one-hot encoded or class indices)
        
        Returns:
            loss: categorical cross-entropy
        """
        # Number of samples
        samples = len(y_pred)
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Handle both one-hot encoded and sparse labels
        if len(y_true.shape) == 1:
            # Sparse labels (class indices)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # One-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, y_pred, y_true):
        """
        Backward pass: compute gradient of categorical cross-entropy loss.
        
        Parameters:
            y_pred: predicted probabilities
            y_true: true labels (one-hot encoded or class indices)
        """
        # Number of samples
        samples = len(y_pred)
        # Number of labels
        labels = len(y_pred[0])
        
        # If labels are sparse, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / y_pred
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# ============================================================
# OPTIMIZER
# ============================================================

class OptimizerSGD:
    """
    Stochastic Gradient Descent optimizer.
    Updates weights using: w = w - learning_rate * gradient
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        """
        Update layer parameters using SGD.
        
        Parameters:
            layer: layer object with weights, biases, dweights, dbiases
        """
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# ============================================================
# EXAMPLE USAGE: XOR PROBLEM
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEURAL NETWORK TRAINING: XOR PROBLEM")
    print("=" * 60)
    print()
    
    # XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    print("[INFO] Dataset: XOR Problem")
    print(f"       >> X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"       >> Inputs:\n{X}")
    print(f"       >> Targets:\n{y}\n")
    
    # Create network layers
    print("[INFO] Building Neural Network Architecture...")
    layer1 = Layer_Dense(2, 4, name="Hidden_Layer_1")
    activation1 = ActivationTanh(name="Tanh_1")
    
    layer2 = Layer_Dense(4, 1, name="Output_Layer")
    activation2 = ActivationSigmoid(name="Sigmoid_Output")
    
    # Loss function and optimizer
    loss_function = LossBinaryCrossentropy()
    optimizer = OptimizerSGD(learning_rate=0.5)
    
    print(f"[INFO] Loss Function: {loss_function.name}")
    print(f"[INFO] Optimizer: SGD (learning_rate={optimizer.learning_rate})")
    print(f"[INFO] Training for 1000 epochs...\n")
    print("=" * 60)
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # ========== FORWARD PASS ==========
        # Layer 1
        layer1.forward(X)
        activation1.forward(layer1.output)
        
        # Layer 2
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        
        # Calculate loss
        loss = loss_function.forward(activation2.output, y)
        
        # Print progress every 100 epochs
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        
        # ========== BACKWARD PASS ==========
        # Loss gradient
        loss_function.backward(activation2.output, y)
        
        # Layer 2 backward
        activation2.backward(loss_function.dinputs)
        layer2.backward(activation2.dinputs)
        
        # Layer 1 backward
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        
        # ========== UPDATE WEIGHTS ==========
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
    
    print("=" * 60)
    print("\n[INFO] Training Complete!\n")
    
    # ========== FINAL PREDICTIONS ==========
    print("=" * 60)
    print("FINAL PREDICTIONS")
    print("=" * 60)
    
    # Forward pass for final predictions
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    predictions = activation2.output
    
    print("\nInput | Target | Prediction | Rounded")
    print("-" * 45)
    for i in range(len(X)):
        pred_val = predictions[i][0]
        rounded = 1 if pred_val >= 0.5 else 0
        print(f"{X[i]} |   {y[i][0]}    |   {pred_val:.4f}   |    {rounded}")
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Loss: {loss:.6f}")
    print(f"Total Epochs: {epochs}")
    print(f"Learning Rate: {optimizer.learning_rate}")
    print("=" * 60)
