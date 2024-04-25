import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Assuming the models are classes named GradTree and GradTreePy in their respective files
from GradTree import GradTree
from GTpytorch import GradTreePy

# Load the data
iris_data_path = "iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(iris_data_path, header=None, names=column_names)

# Handle missing values (if any)
iris_df.fillna(iris_df.median(), inplace=True)

# Encode class labels as integers
label_encoder = LabelEncoder()
iris_df['class'] = label_encoder.fit_transform(iris_df['class'])

X = iris_df.drop('class', axis=1)
y = iris_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

config = {
    'input_dim': 4,
    'output_dim': 3,
    'epochs': 10
}

# Prepare TensorFlow model
tf_model = GradTree(config=config)
# Ensure that your TensorFlow model is designed to handle integer labels if it is not expecting one-hot encoding
tf_model.train(X_train, y_train, epochs=config['epochs'])

# Prepare PyTorch model
torch_model = GradTreePy(config=config)
torch_model.train(X_train, y_train, epochs=config['epochs'])

y_pred_tf = tf_model.predict(X_test)
y_pred_torch = torch_model.predict(X_test)

# Calculate accuracy
accuracy_tf = accuracy_score(y_test, np.argmax(y_pred_tf, axis=1)) if len(y_pred_tf.shape) > 1 else accuracy_score(y_test, y_pred_tf)
accuracy_torch = accuracy_score(y_test, y_pred_torch)

# Print comparison
print(f"TensorFlow Model Accuracy: {accuracy_tf}")
print(f"PyTorch Model Accuracy: {accuracy_torch}")
