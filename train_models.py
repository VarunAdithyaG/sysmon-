import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_FILE = Path(__file__).parent / "usage.csv"
SCALER_FILE = Path(__file__).parent / "scaler.pkl"
LSTM_FILE = Path(__file__).parent / "lstm_model.pth"
ISO_FILE = Path(__file__).parent / "iso_forest.pkl"

# Number of past samples in one sequence window.
SEQ_LEN = 10
PRED_HORIZON = 1  # predict 1 step ahead
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

FEATURE_COLS = [
    "cpu_percent",
    "gpu_percent",
    "memory_percent",
    "disk_percent",
    "cpu_temp_c",
    "gpu_temp_c",
]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"No data file at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate().bfill().ffill()
    return df


def build_sequences(df: pd.DataFrame, scaler: StandardScaler | None = None, fit_scaler: bool = True):
    X = df[FEATURE_COLS].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    seqs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(len(X_scaled) - SEQ_LEN - PRED_HORIZON + 1):
        seq = X_scaled[i : i + SEQ_LEN]
        target_idx = i + SEQ_LEN + PRED_HORIZON - 1
        y = df.loc[target_idx, ["cpu_percent", "memory_percent"]].values.astype(np.float32)
        seqs.append(seq)
        targets.append(y)

    if not seqs:
        raise ValueError("Not enough data to build sequences. Collect more usage samples first.")

    X_seqs = np.stack(seqs)
    y_arr = np.stack(targets)
    return X_seqs, y_arr, scaler, X_scaled


def train_lstm(X_seqs: np.ndarray, y_arr: np.ndarray) -> LSTMRegressor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_seqs.shape[-1]
    model = LSTMRegressor(input_dim=input_dim, output_dim=2).to(device)

    dataset = TensorDataset(torch.from_numpy(X_seqs), torch.from_numpy(y_arr))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS}, loss={epoch_loss:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
        },
        LSTM_FILE,
    )
    return model


def train_isolation_forest(X_scaled: np.ndarray) -> IsolationForest:
    windows: list[np.ndarray] = []
    for i in range(len(X_scaled) - SEQ_LEN + 1):
        win = X_scaled[i : i + SEQ_LEN].reshape(-1)
        windows.append(win)

    if not windows:
        raise ValueError("Not enough data to train IsolationForest. Collect more usage samples first.")

    X_win = np.stack(windows)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
    )
    iso.fit(X_win)
    with open(ISO_FILE, "wb") as f:
        pickle.dump(iso, f)
    return iso


def main() -> None:
    df = load_data()
    X_seqs, y_arr, scaler, X_scaled = build_sequences(df, scaler=None, fit_scaler=True)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {SCALER_FILE}")

    _ = train_lstm(X_seqs, y_arr)
    print(f"Saved LSTM model to {LSTM_FILE}")

    _ = train_isolation_forest(X_scaled)
    print(f"Saved IsolationForest to {ISO_FILE}")


if __name__ == "__main__":
    main()

