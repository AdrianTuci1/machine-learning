import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. Load and Investigate the Data
dataset = pd.read_csv('admissions_data.csv')
print(dataset.head())
print(dataset.describe())

# 2. Features and Labels Split
# 'Serial No.' is just an index and doesn't help with prediction, so we drop it.
# 'Chance of Admit' is our continuous target label.
features = dataset.iloc[:, 1:-1] 
labels = dataset.iloc[:, -1]

# Check for categorical variables (In this dataset, all are numerical)
print(features.dtypes)

# 3. Train-Test Split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# 4. Scaling the Data
# Since GRE (340) and University Rating (5) have different scales, we normalize.
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# 5. Create the Neural Network Model
def create_model():
    model = Sequential()
    # Input layer based on the number of features
    model.add(InputLayer(input_shape=(features.shape[1],)))
    
    # Hidden layers
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    
    # Output layer (1 neuron for regression, no activation for continuous output)
    model.add(Dense(1))
    
    # Compile with Adam optimizer and Mean Squared Error loss
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

model = create_model()

# 6 & 7. Fit and Hyperparameter Tuning
# We use a batch size of 8 and 40 epochs. 
# Validation split allows us to see how it performs on unseen data during training.
history = model.fit(
    features_train_scaled, 
    labels_train, 
    epochs=40, 
    batch_size=8, 
    verbose=1, 
    validation_split=0.2
)

# 8. Evaluate the model
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)
print(f"Final Test Mean Absolute Error: {res_mae}")

# 9. Plotting performance with Matplotlib
fig = plt.figure(figsize=(12, 5))

# Plot MAE
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(history.history['mae'], label='train')
ax1.plot(history.history['val_mae'], label='validation')
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('Epoch')
ax1.legend()

# Plot Loss (MSE)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='validation')
ax2.set_title('Model Loss (MSE)')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

# Save the plot as required
plt.tight_layout()
fig.savefig('static/images/my_plots.png')

# 10. R-Squared Evaluation
predictions = model.predict(features_test_scaled)
print(f"R-squared Score: {r2_score(labels_test, predictions)}")
