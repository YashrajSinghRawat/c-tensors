import numpy as np


class SelfAttentionLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights
        self.W_Q = np.random.randn(output_dim, input_dim)
        self.W_K = np.random.randn(output_dim, input_dim)
        self.W_V = np.random.randn(output_dim, input_dim)
        # Initialize biases
        self.b_Q = np.zeros((output_dim, 1))
        self.b_K = np.zeros((output_dim, 1))
        self.b_V = np.zeros((output_dim, 1))

    def forward(self, X):
        # X.shape = (input_dim, num_samples)
        self.X = X  # (seq_len, input_n)
        # Compute Q, K, V
        self.Q = np.dot(self.W_Q, X.T) + self.b_Q  # (output_n, seq_len)
        self.K = np.dot(self.W_K, X.T) + self.b_K  # (output_n, seq_len)
        self.V = np.dot(self.W_V, X.T) + self.b_V  # (output_n, seq_len)
        # Compute attention scores
        self.S = np.dot(self.Q.T, self.K) / \
            np.sqrt(self.input_dim)  # (seq_len, seq_len)
        # Softmax
        self.attn_weights = np.exp(
            self.S) / np.sum(np.exp(self.S), axis=1, keepdims=True)  # (seq_len, seq_len)
        # Weighted sum of values
        return np.dot(self.attn_weights, self.V.T)  # (seq_len, output_n)

    def backward(self, output_d, learning_rate=0.001):
        # Gradient of output w.r.t. V
        V_d = np.dot(output_d.T, self.attn_weights)  # (output_n, seq_len)
        # Gradient of output w.r.t. attention_weights
        attn_weights_d = np.dot(self.V.T, output_d.T)  # (seq_len, seq_len)
        # Gradient of output w.r.t. S
        S_d = attn_weights_d * self.attn_weights * \
            (1 - self.attn_weights)  # (seq_len, seq_len)
        # Gradient of S w.r.t. Q and K
        Q_d = np.dot(self.K, S_d.T)  # (output_n, seq_len)
        K_d = np.dot(self.Q, S_d.T)  # (output_n, seq_len)
        # Gradient of Q and K w.r.t. W_Q, W_K, b_Q, and b_K
        W_Q_d = np.dot(Q_d, self.X)  # (output_n, input_n)
        B_Q_d = np.sum(Q_d, axis=1, keepdims=True)  # (output_n, 1)
        W_K_d = np.dot(K_d, self.X)  # (output_n, input_n)
        B_K_d = np.sum(K_d, axis=1, keepdims=True)  # (output_n, 1)
        W_V_d = np.dot(V_d, self.X)  # (output_n, input_n)
        B_V_d = np.sum(V_d, axis=1, keepdims=True)  # (output_n, 1)
        # Update weights
        self.W_Q -= learning_rate * W_Q_d
        self.b_Q -= learning_rate * B_Q_d
        self.W_K -= learning_rate * W_K_d
        self.b_K -= learning_rate * B_K_d
        self.W_V -= learning_rate * W_V_d
        self.b_V -= learning_rate * B_V_d
        return np.dot(V_d.T, self.W_V)  # (seq_len, input_n)


# Define the input data
X_train = np.array([[1, 2], [3, 4]])

# Define hyperparameters
input_dim = 2
attention_dim = 2
learning_rate = 0.001
num_epochs = 1000

# Create an instance of the SelfAttentionLayer
attention_layer = SelfAttentionLayer(input_dim, attention_dim)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = attention_layer.forward(X_train)

    # Compute loss (mean squared error)
    loss = np.mean((output - X_train) ** 2)

    # Backward pass
    d_loss = 2 * (output - X_train) / len(X_train)
    attention_layer.backward(d_loss, learning_rate)

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss}")

# Evaluate on training data
predicted_output = attention_layer.forward(X_train)
print("\nPredicted output:")
print(predicted_output)
