import math
import numpy as np
import json

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, output_neurons, activation="relu",random_init=True):
        self.layers = [input_neurons] + hidden_layers + [output_neurons]
        self.weights = []
        self.biases = []
        
        if random_init:
            for i in range(len(self.layers) - 1):
                self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]))
                self.biases.append(np.random.randn(self.layers[i + 1]))
        else:
            for i in range(len(self.layers) - 1):
                self.weights.append(np.zeros((self.layers[i], self.layers[i + 1])))
                self.biases.append(np.zeros(self.layers[i + 1]))
        
        # Set activation function
        if activation == "relu":
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        elif activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_prime = self.sigmoid_prime

    def relu(self, x):
        return np.maximum(0.01*x, x)
    
    def relu_prime(self, x):
        return np.where(x > 0, 1, 0.01)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        return x * (1 - x)

    def save_to_file(self, filename):
        data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    def load_from_file(cls, filename, input_neurons, hidden_layers, output_neurons, activation="relu"):
        with open(filename, 'r') as file:
            data = json.load(file)
        
        nn = cls(input_neurons, hidden_layers, output_neurons, activation, random_init=False)
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        
        return nn
    
    def forward(self, x):
        self.a = [x]
        for i in range(len(self.weights) - 1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.a.append(self.activation(z))
        z_out = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.a.append(self.sigmoid(z_out))
        return self.a[-1]
    
    def compute_loss(self, y):
        m = y.shape[0]
        return (1 / (2 * m)) * np.sum((self.a[-1] - y) ** 2)
    
    def backward(self, x, y, learning_rate=0.1):
        m = x.shape[0]
        self.dz = [(1 / m) * (self.a[-1] - y)]
        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ self.dz[-1]
            db = np.sum(self.dz[-1], axis=0)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            if i != 0:
                dz_next = self.dz[-1] @ self.weights[i].T * self.activation_prime(self.a[i])
                self.dz.append(dz_next)
                
    def train(self, x, y, epochs=10000, learning_rate=0.01, optimizer=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if optimizer == "ADAM":
            # Initialize Adam-specific variables
            m_weights = [np.zeros_like(w) for w in self.weights]
            v_weights = [np.zeros_like(w) for w in self.weights]
            m_biases = [np.zeros_like(b) for b in self.biases]
            v_biases = [np.zeros_like(b) for b in self.biases]
            t = 0
        
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y)
            
            if optimizer == "ADAM":
                t += 1
                for i in range(len(self.weights)):
                    # Compute gradients for weights and biases
                    grad_w = self.a[i].T @ self.dz[-(i+1)]
                    grad_b = np.sum(self.dz[-(i+1)], axis=0)
                    
                    # Update first moment for weights and biases
                    m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * grad_w
                    m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * grad_b
                    
                    # Update second moment for weights and biases
                    v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * grad_w ** 2
                    v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * grad_b ** 2
                    
                    # Bias-corrected moments
                    m_weights_corr = m_weights[i] / (1 - beta1**t)
                    m_biases_corr = m_biases[i] / (1 - beta1**t)
                    v_weights_corr = v_weights[i] / (1 - beta2**t)
                    v_biases_corr = v_biases[i] / (1 - beta2**t)
                    
                    # Update weights and biases
                    self.weights[i] -= learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + epsilon)
                    self.biases[i] -= learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + epsilon)
            else:
                # Standard gradient descent
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * self.a[i].T @ self.dz[-(i+1)]
                    self.biases[i] -= learning_rate * np.sum(self.dz[-(i+1)], axis=0)
            if epoch%math.floor(epochs/100)==0:
                print(f"Epoch {epoch}/{epochs}")

