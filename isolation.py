import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("usage.csv")

# Select important system metrics
features = [
    "cpu_percent",
    "memory_percent",
    "gpu_percent",
    "disk_percent",
    "cpu_temp_c",
    "gpu_temp_c"
]

data = df[features]

# Handle missing values
data = data.fillna(method="ffill")

# -----------------------------
# 2. Train Isolation Forest
# -----------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # 5% anomalies expected
    random_state=42
)

model.fit(data)

# -----------------------------
# 3. Predict Anomalies
# -----------------------------
df["anomaly"] = model.predict(data)

# convert labels
# 1 = normal
# -1 = anomaly

# -----------------------------
# 4. Extract Anomalies
# -----------------------------
anomalies = df[df["anomaly"] == -1]

print("Number of anomalies detected:", len(anomalies))

# -----------------------------
# 5. Visualization
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(df["cpu_percent"], label="CPU Usage")

plt.scatter(
    anomalies.index,
    anomalies["cpu_percent"],
    color="red",
    label="Anomaly"
)

plt.legend()
plt.title("Isolation Forest Anomaly Detection (CPU Usage)")
plt.xlabel("Time")
plt.ylabel("CPU %")
plt.show()