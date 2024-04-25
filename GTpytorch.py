import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import category_encoders as ce
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

class GTpytorch(nn.Module):
    def __init__(self, config):
        super(GTpytorch, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        # Configuring the model architecture
        self.output_layer = GradTreeBlock(self.config)
        self.objective = self.config.get('objective', 'classification')

    def forward(self, x):
        return self.output_layer(x)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        # Data preprocessing
        X_train, X_val = self.preprocess_data(X_train, y_train, X_val)
        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = self.get_loss_function()
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        # Training loop
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def preprocess_data(self, X_train, y_train, X_val):
        # Placeholder for comprehensive preprocessing based on the TensorFlow example
        # Assume encoding and normalization steps here
        return X_train, X_val

    def get_loss_function(self):
        if self.objective == 'classification':
            return nn.CrossEntropyLoss()
        elif self.objective == 'binary':
            return nn.BCEWithLogitsLoss()
        elif self.objective == 'regression':
            return nn.MSELoss()
        return nn.CrossEntropyLoss()  # Default case

def hardmax(logits):
    # Create a one-hot tensor with the maximum entry set to 1
    return (logits == logits.max(dim=-1, keepdim=True)[0]).float()

def entmax15(logits, dim=-1, alpha=1.5):
    # Approximation to the entmax activation; 1.5 aims for a middle ground between softmax (2) and sparsemax (1)
    t = torch.relu(logits)
    return F.softmax(t ** alpha, dim=dim)

class GradTreeBlock(nn.Module):
    def __init__(self, config):
        super(GradTreeBlock, self).__init__()
        self.depth = config.get('depth', 6)
        self.n_estimators = config.get('n_estimators', 1)  # Number of trees if ensemble method is used
        self.input_dim = config.get('input_dim', 10)
        self.num_classes = config.get('output_dim', 2)  # Output dimension or number of classes

        # Weights for internal nodes
        self.T = nn.Parameter(torch.randn(self.n_estimators, 2**self.depth - 1, self.input_dim))
        # Weights for leaf nodes
        self.L = nn.Parameter(torch.randn(self.n_estimators, 2**self.depth, self.num_classes))

    def forward(self, x):
        # Applying entmax and the ST operator
        I = entmax15(self.T)
        c1 = I - hardmax(I)  # ST operator
        I = I - c1

        y_hat = torch.zeros((x.size(0), self.num_classes), device=x.device)

        # Loop over all leaf nodes
        for l in range(2**self.depth):
            p = torch.tensor(1.0, device=x.device)
            # Loop over depth of the tree
            for j in range(1, self.depth + 1):
                idx = int(2**(j-1) + (l // 2**(self.depth - (j - 1))) - 1)
                s = torch.einsum('bi,ij->bj', self.T[:, idx, :], I[:, idx, :]) - torch.einsum('bi,ij->bj', x, I[:, idx, :])
                c2 = s - torch.floor(s)  # ST operator
                s = s - c2
                p = p * ((1 - ((l // 2**(self.depth - j)) % 2)) * s + ((l // 2**(self.depth - j)) % 2) * (1 - s))
            y_hat += self.L[:, l, :] * p.unsqueeze(-1)  # Equation 1

        print("Shape of self.L[:, l, :]:", self.L[:, l, :].shape)
        print("Shape of p.unsqueeze(-1):", p.unsqueeze(-1).shape)

        return F.softmax(y_hat, dim=-1)  # Softmax to get probability distribution




