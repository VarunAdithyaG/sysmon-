# SysMon Predictor

SysMon Predictor is a Linux system monitoring and control prototype that collects runtime metrics, learns usage patterns from historical data, predicts near-term load, and flags anomalous behavior.

The project is built around three stages:

- `monitor.py` logs system metrics into `usage.csv`
- `train_models.py` trains prediction and anomaly-detection models from that data
- `controller.py` runs the full loop: collect, retrain, predict, detect, and react

## What It Tracks

Each sample can include:

- CPU usage
- memory usage
- disk usage
- GPU usage when available
- CPU and GPU temperature when available
- top process IDs by recent CPU activity

The monitor reads memory information from `/proc/sysmon/stats` when a compatible kernel module is present, and otherwise falls back to standard Linux procfs data.

## Main Files

- `monitor.py`: samples system metrics and appends them to `usage.csv`
- `train_models.py`: trains a scaler, an LSTM regressor, and an Isolation Forest model
- `decision_engine.py`: loads trained artifacts and computes anomaly / action signals
- `controller.py`: coordinates monitoring, prediction, anomaly detection, and resource decisions
- `inject_heavy_usage.py`: appends synthetic heavy-load samples for testing
- `usage.csv`: collected monitoring history

## Model Flow

`train_models.py` uses the collected metrics in `usage.csv` to produce:

- a `StandardScaler`
- an LSTM model that predicts future CPU and memory usage
- an Isolation Forest model that scores unusual behavior across recent windows

The controller uses these outputs together with time-of-day baselines from the CSV history to decide whether the system appears to be entering a high-pressure state.

## Running The Project

Install dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
```

Start collecting data:

```bash
python3 monitor.py
```

Train the models:

```bash
python3 train_models.py
```

Run the controller:

```bash
python3 controller.py
```

If you run it from the parent directory and want explicit environment variables, use:

```bash
PYTHONPATH=sysmon_predictor SYSMON_ENABLE_KILL=1 ./.venv/bin/python sysmon_predictor/controller.py
```

That command:

- makes `sysmon_predictor` importable via `PYTHONPATH`
- enables live kill behavior with `SYSMON_ENABLE_KILL=1`
- runs the controller with the project virtual environment

Generate synthetic heavy-usage samples for testing:

```bash
python3 inject_heavy_usage.py
```

## Runtime Notes

- This project is Linux-specific.
- Some features depend on `/proc`, `/sys`, cgroup support, and CPU governor tools.
- GPU telemetry is optional and only appears when supported by the host.
- `torch` is required for training and controller execution, even though it is not currently listed in `requirements.txt`.
- The controller can interact with process priorities and CPU policy, so it should be tested carefully on a non-critical machine first.
- If you recreate a virtual environment, the recommended path is `./.venv/` at the project root.

## Current State

This repository is a prototype-oriented project rather than a packaged library. The code is organized around scripts that operate on local files in the project directory, especially `usage.csv` and the generated model artifacts.
