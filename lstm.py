import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("usage.csv")

# Select important features for system behavior
features = [
    "cpu_percent",
    "memory_percent",
    "gpu_percent",
    "disk_percent",
    "cpu_temp_c",
    "gpu_temp_c"
]

df = df[features]

# Handle missing values
df = df.fillna(method="ffill")

# -----------------------------
# 2. Normalize Data
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# -----------------------------
# 3. Feature Engineering
# Create Time-Series Sequences
# -----------------------------
sequence_length = 10

X = []
y = []

for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length][0])  # predict CPU

X = np.array(X)
y = np.array(y)

print("Input shape:", X.shape)

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 5. Build LSTM Model
# -----------------------------
model = Sequential()

model.add(LSTM(
    units=64,
    return_sequences=False,
    input_shape=(X_train.shape[1], X_train.shape[2])
))

model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # CPU prediction

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------
# 6. Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 7. Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# 8. Plot Results
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual CPU Usage")
plt.plot(predictions, label="Predicted CPU Usage")
plt.legend()
plt.title("CPU Usage Prediction using LSTM")
plt.show()