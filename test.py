import openml
import torch
import pandas as pd
from GTpytorch import GTpytorch
from sklearn.model_selection import train_test_split
import numpy as np
import category_encoders as ce

# Load data
dataset = openml.datasets.get_dataset(40536)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_valid))
print("Test set size:", len(X_test))

categorical_feature_indices = [attribute_names[i] for i, is_cat in enumerate(categorical_indicator) if is_cat]
existing_categorical_features = [col for col in categorical_feature_indices if col in X_train.columns]

encoder = ce.OrdinalEncoder(cols=existing_categorical_features)
X_train = encoder.fit_transform(X_train)
X_valid = encoder.transform(X_valid)
X_test = encoder.transform(X_test)

# Ensure y data is in the correct format
if isinstance(y_train, pd.Series):
    y_train = y_train.values  # Convert to numpy array if it's a pandas Series
if isinstance(y_valid, pd.Series):
    y_valid = y_valid.values
if isinstance(y_test, pd.Series):
    y_test = y_test.values

# Assuming y data needs to be encoded as float and is already in numpy format
y_train = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
y_valid = torch.tensor(y_valid.astype(np.float32), dtype=torch.float32)
y_test = torch.tensor(y_test.astype(np.float32), dtype=torch.float32)

# Convert DataFrames to numpy arrays and then to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

# Model configuration
config = {
    'input_dim': X_train.shape[1],
    'output_dim': 2,  # Assuming binary classification; adjust as necessary
    'depth': 6,
    'learning_rate': 0.01,
    'objective': 'binary'
}

# Create and train the model
model = GTpytorch(config)
model.fit(X_train=X_train, y_train=y_train, X_val=X_valid, y_val=y_valid)

# Predictions
preds_gradtree = model.predict(X_test)

print("Training complete")

