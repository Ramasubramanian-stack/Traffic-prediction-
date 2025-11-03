import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ✅ Path to your extracted dataset
file_path = r"C:\Users\kaviya\Downloads\archive (5)\METR-LA.h5"

# --- STEP 1: Inspect the file structure ---
with h5py.File(file_path, 'r') as f:
    print("Available keys in dataset:", list(f.keys()))
    key_name = list(f.keys())[0]
    print(f"Using key: '{key_name}'")
    print("Subkeys inside this key:", list(f[key_name].keys()))

# --- STEP 2: Load the actual dataset ---
with h5py.File(file_path, 'r') as f:
    group = f['df']
    print("\nAvailable sub-datasets in 'df':", list(group.keys()))

    # Try to locate where the data actually is
    if 'block0_values' in group.keys():
        data = group['block0_values'][:]  # Main numerical data
        print("Loaded data shape:", data.shape)
    else:
        raise KeyError("Expected dataset 'block0_values' not found in 'df'")

# --- STEP 3: Preprocess ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

sequence_length = 10
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])
X, y = np.array(X), np.array(y)

# --- STEP 4: Split ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- STEP 5: LSTM Model ---
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(y_train.shape[1])
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- STEP 6: Train ---
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# --- STEP 7: Predict ---
pred = model.predict(X_test)
pred_rescaled = scaler.inverse_transform(pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# --- STEP 8: Visualize ---
plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled[:200, 0], label='Actual')
plt.plot(pred_rescaled[:200, 0], label='Predicted')
plt.title("Traffic Speed Prediction")
plt.legend()
plt.show()
