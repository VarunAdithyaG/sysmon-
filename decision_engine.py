import json
import os
import pickle
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_models import (
    DATA_FILE,
    FEATURE_COLS,
    ISO_FILE,
    LSTM_FILE,
    SCALER_FILE,
    SEQ_LEN,
    LSTMRegressor,
)

PROCESS_SAMPLE_SECONDS = 0.35
CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
MEMORY_FILE = Path(__file__).parent / "process_memory.json"
LEARN_TRUST_GAIN = 0.08
LEARN_TRUST_DECAY = 0.25
LEARN_TRUST_MAX = 0.95
MIN_SIGHTINGS_BEFORE_KILL = 4


def _read_comm(pid: int) -> str:
    try:
        return (Path(f"/proc/{pid}/comm")).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _is_protected_pid(pid: int, protected_pids: set[int]) -> bool:
    return pid in protected_pids


def _read_pid_stat(pid: int) -> tuple[int, str] | None:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        after = stat_text.rsplit(")", 1)[1].strip().split()
        state = after[0]
        utime = int(after[11])
        stime = int(after[12])
    except (OSError, IndexError, ValueError):
        return None

    if state == "Z":
        return None
    try:
        os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None
    return utime + stime, state


def _read_process_meta(pid: int) -> dict[str, Any] | None:
    proc_dir = Path(f"/proc/{pid}")
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
        name = stat_text.split("(", 1)[1].rsplit(")", 1)[0].strip().lower()
        after = stat_text.rsplit(")", 1)[1].strip().split()
        state = after[0]
        ppid = int(after[1])
        pgrp = int(after[2])
        session = int(after[3])
        tty_nr = int(after[4])
        num_threads = int(after[17])
    except (OSError, IndexError, ValueError):
        return None

    if state == "Z":
        return None

    status_path = proc_dir / "status"
    uid = None
    vm_rss_kb = 0
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Uid:"):
                uid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                vm_rss_kb = int(line.split()[1])
    except (OSError, IndexError, ValueError):
        return None

    try:
        exe_path = os.readlink(proc_dir / "exe")
    except OSError:
        return None

    cmdline0 = ""
    try:
        raw_cmdline = (proc_dir / "cmdline").read_bytes().split(b"\x00")
        if raw_cmdline and raw_cmdline[0]:
            cmdline0 = raw_cmdline[0].decode("utf-8", errors="ignore").strip().lower()
    except OSError:
        cmdline0 = ""

    return {
        "pid": pid,
        "ppid": ppid,
        "pgrp": pgrp,
        "session": session,
        "tty_nr": tty_nr,
        "uid": uid,
        "name": name,
        "state": state,
        "threads": num_threads,
        "vm_rss_kb": vm_rss_kb,
        "exe": exe_path,
        "cmdline0": cmdline0,
    }


def _process_signature(meta: dict[str, Any]) -> str:
    exe = str(meta.get("exe", "")).strip().lower()
    cmdline0 = str(meta.get("cmdline0", "")).strip().lower()
    name = str(meta.get("name", "")).strip().lower()
    uid = str(meta.get("uid", ""))
    base = Path(exe).name if exe else name
    launcher = Path(cmdline0).name if cmdline0 else base
    return f"{uid}:{base}:{launcher}"


def _load_process_memory() -> dict[str, dict[str, Any]]:
    if not MEMORY_FILE.exists():
        return {}
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for sig, record in data.items():
        if not isinstance(sig, str) or not isinstance(record, dict):
            continue
        cleaned[sig] = {
            "trust": float(record.get("trust", 0.0)),
            "seen": int(record.get("seen", 0)),
            "spared": int(record.get("spared", 0)),
            "killed": int(record.get("killed", 0)),
            "last_seen": str(record.get("last_seen", "")),
            "name": str(record.get("name", "")),
            "exe": str(record.get("exe", "")),
        }
    return cleaned


def _save_process_memory(memory: dict[str, dict[str, Any]]) -> None:
    try:
        MEMORY_FILE.write_text(json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _get_learned_trust(memory: dict[str, dict[str, Any]], signature: str) -> float:
    record = memory.get(signature, {})
    try:
        trust = float(record.get("trust", 0.0))
    except (TypeError, ValueError):
        trust = 0.0
    return float(np.clip(trust, 0.0, LEARN_TRUST_MAX))


def _get_seen_count(memory: dict[str, dict[str, Any]], signature: str) -> int:
    record = memory.get(signature, {})
    try:
        return max(0, int(record.get("seen", 0)))
    except (TypeError, ValueError):
        return 0


def _update_process_memory(
    memory: dict[str, dict[str, Any]],
    signature: str,
    meta: dict[str, Any],
    observed_at: str,
    learned: bool,
    killed: bool,
) -> float:
    record = memory.setdefault(
        signature,
        {
            "trust": 0.0,
            "seen": 0,
            "spared": 0,
            "killed": 0,
            "last_seen": "",
            "name": str(meta.get("name", "")),
            "exe": str(meta.get("exe", "")),
        },
    )
    record["seen"] = int(record.get("seen", 0)) + 1
    record["last_seen"] = observed_at
    record["name"] = str(meta.get("name", ""))
    record["exe"] = str(meta.get("exe", ""))

    trust = float(record.get("trust", 0.0))
    if killed:
        record["killed"] = int(record.get("killed", 0)) + 1
        trust -= LEARN_TRUST_DECAY
    elif learned:
        record["spared"] = int(record.get("spared", 0)) + 1
        trust += LEARN_TRUST_GAIN

    record["trust"] = float(np.clip(trust, 0.0, LEARN_TRUST_MAX))
    return float(record["trust"])


def _ancestor_chain(pid: int, max_depth: int = 32) -> set[int]:
    ancestors: set[int] = set()
    current = pid
    for _ in range(max_depth):
        meta = _read_process_meta(current)
        if meta is None:
            break
        ppid = int(meta["ppid"])
        if ppid <= 0 or ppid in ancestors:
            break
        ancestors.add(ppid)
        current = ppid
    return ancestors


def _essentiality_score(
    meta: dict[str, Any],
    protected_pids: set[int],
    controller_uid: int,
    controller_session: int,
    controller_ancestors: set[int],
) -> float:
    pid = int(meta["pid"])
    score = 0.0

    if pid in protected_pids:
        return 1.0
    if pid in controller_ancestors:
        return 1.0

    if meta["uid"] is None or int(meta["uid"]) != controller_uid:
        score += 0.6
    if int(meta["session"]) == controller_session:
        score += 0.35
    if int(meta["tty_nr"]) != 0:
        score += 0.35
    if pid == int(meta["session"]) or pid == int(meta["pgrp"]):
        score += 0.25
    if int(meta["ppid"]) <= 1:
        score += 0.3
    if int(meta["threads"]) >= 16:
        score += 0.1
    if int(meta["vm_rss_kb"]) >= 1_000_000:
        score -= 0.1

    return float(np.clip(score, 0.0, 1.0))


def _kill_candidate_score(
    cpu: float,
    threshold: float,
    predicted_cpu: float,
    predicted_mem: float,
    is_anomaly: bool,
    essentiality: float,
    learned_trust: float,
) -> float:
    if cpu < threshold:
        return -1.0

    pressure_bonus = max(0.0, (cpu - threshold) / max(threshold, 1.0))
    model_bonus = max(0.0, predicted_cpu / 100.0) + max(0.0, predicted_mem / 100.0)
    anomaly_bonus = 0.2 if is_anomaly else 0.0
    expendability = 1.0 - np.clip(essentiality + (0.7 * learned_trust), 0.0, 0.98)
    return (1.0 + pressure_bonus + model_bonus + anomaly_bonus) * expendability


def _parse_allowlist() -> set[str]:
    raw = os.environ.get("SYSMON_KILL_ALLOW", "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _kill_cpu_threshold(default: float) -> float:
    raw = os.environ.get("SYSMON_KILL_CPU", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _sample_top_processes(limit: int = 12) -> list[dict[str, Any]]:
    start = time.time()
    snap_1: dict[int, int] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        stat = _read_pid_stat(pid)
        if stat is None:
            continue
        ticks, _ = stat
        snap_1[pid] = ticks
    time.sleep(PROCESS_SAMPLE_SECONDS)
    elapsed = max(time.time() - start, 1e-6)
    rows: list[dict[str, Any]] = []
    for pid, ticks_1 in snap_1.items():
        stat = _read_pid_stat(pid)
        if stat is None:
            continue
        ticks_2, _ = stat
        tick_delta = ticks_2 - ticks_1
        if tick_delta <= 0:
            continue
        cpu = (tick_delta / CLK_TCK) / elapsed * 100.0
        if cpu <= 0:
            continue
        comm = _read_comm(pid).lower()
        rows.append({"pid": pid, "cpu": round(cpu, 1), "comm": comm})

    rows.sort(key=lambda row: row["cpu"], reverse=True)
    return rows[:limit]


def _dynamic_kill_threshold(df: pd.DataFrame, predicted_cpu: float, is_anomaly: bool) -> float:
    top_cpu_hist = pd.to_numeric(df.get("top_cpu_percent", pd.Series(dtype=float)), errors="coerce").dropna()
    cpu_hist = pd.to_numeric(df.get("cpu_percent", pd.Series(dtype=float)), errors="coerce").dropna()

    baseline_top = float(top_cpu_hist.quantile(0.75)) if len(top_cpu_hist) >= 20 else 60.0
    baseline_cpu_p75 = float(cpu_hist.quantile(0.75)) if len(cpu_hist) >= 20 else 45.0

    dynamic_threshold = baseline_top

    # If forecasted system pressure is high, react earlier.
    if predicted_cpu >= baseline_cpu_p75:
        dynamic_threshold -= 10.0

    # During anomalies, be more aggressive.
    if is_anomaly:
        dynamic_threshold -= 15.0

    return float(np.clip(dynamic_threshold, 25.0, 75.0))


@dataclass
class DecisionResult:
    timestamp: str
    predicted_cpu: float
    predicted_mem: float
    anomaly: bool
    anomaly_score: float
    top_pids: list[int]
    kill_pids: list[int]
    top_cpu_percent: float
    kill_cpu_threshold: float
    kill_allow: list[str]
    kill_candidate_pid: int
    kill_candidate_comm: str
    kill_block_reason: str
    kill_candidate_signature: str
    kill_candidate_learned_trust: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "predicted_cpu": self.predicted_cpu,
            "predicted_mem": self.predicted_mem,
            "anomaly": self.anomaly,
            "anomaly_score": self.anomaly_score,
            "top_pids": self.top_pids,
            "kill_pids": self.kill_pids,
            "top_cpu_percent": self.top_cpu_percent,
            "kill_cpu_threshold": self.kill_cpu_threshold,
            "kill_allow": self.kill_allow,
            "kill_candidate_pid": self.kill_candidate_pid,
            "kill_candidate_comm": self.kill_candidate_comm,
            "kill_block_reason": self.kill_block_reason,
            "kill_candidate_signature": self.kill_candidate_signature,
            "kill_candidate_learned_trust": self.kill_candidate_learned_trust,
        }


def _load_models():
    if not all(p.exists() for p in [SCALER_FILE, LSTM_FILE, ISO_FILE, DATA_FILE]):
        raise FileNotFoundError("Required model/scaler/data files are missing. Run train_models.py first.")

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    checkpoint = torch.load(LSTM_FILE, map_location="cpu")
    input_dim = checkpoint["input_dim"]
    model = LSTMRegressor(input_dim=input_dim, output_dim=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(ISO_FILE, "rb") as f:
        iso = pickle.load(f)

    return scaler, model, iso


def _prepare_latest_window(df: pd.DataFrame, scaler) -> tuple[np.ndarray, pd.Series]:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate().bfill().ffill()

    if len(df) < SEQ_LEN:
        raise ValueError("Not enough data to build one window for decision engine.")

    recent = df.iloc[-SEQ_LEN:]
    X = recent[FEATURE_COLS].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    window = X_scaled.reshape(1, SEQ_LEN, -1)
    last_row = recent.iloc[-1]
    return window, last_row


def make_decision(kill_threshold_cpu_pct: float = 50.0, dry_run: bool = True) -> DecisionResult:
    scaler, lstm_model, iso = _load_models()
    df = pd.read_csv(DATA_FILE)

    window, last_row = _prepare_latest_window(df, scaler)

    with torch.no_grad():
        x_tensor = torch.from_numpy(window)
        pred = lstm_model(x_tensor).numpy()[0]
    predicted_cpu = float(pred[0])
    predicted_mem = float(pred[1])

    flat_window = window.reshape(1, -1)
    anomaly_score = float(iso.decision_function(flat_window)[0])
    anomaly_label = int(iso.predict(flat_window)[0])  # 1 = normal, -1 = anomaly
    is_anomaly = anomaly_label == -1

    timestamp_str = str(last_row["timestamp"])
    process_memory = _load_process_memory()

    live_top = _sample_top_processes(limit=12)
    top_pids = [row["pid"] for row in live_top[:5]]
    top_cpu_pct = float(live_top[0]["cpu"]) if live_top else 0.0

    kill_pids: list[int] = []
    dynamic_threshold = _dynamic_kill_threshold(df, predicted_cpu, is_anomaly)
    kill_threshold_cpu_pct = _kill_cpu_threshold(dynamic_threshold)
    controller_pid = os.getpid()
    controller_meta = _read_process_meta(controller_pid)
    controller_session = int(controller_meta["session"]) if controller_meta is not None else os.getsid(0)
    controller_uid = os.getuid()
    controller_ancestors = _ancestor_chain(controller_pid)
    protected_pids = {1, controller_pid, os.getppid(), *controller_ancestors}
    allowlist = _parse_allowlist()
    kill_candidate_pid = -1
    kill_candidate_comm = ""
    kill_block_reason = ""
    kill_candidate_signature = ""
    kill_candidate_learned_trust = 0.0
    best_candidate_score = -1.0
    observed_candidates: list[tuple[str, dict[str, Any], bool, bool]] = []
    warmup_blocked = False

    if not live_top:
        kill_block_reason = "no_live_processes"
    elif top_cpu_pct < kill_threshold_cpu_pct:
        kill_block_reason = "top_cpu_below_dynamic_threshold"
    else:
        for row in live_top:
            pid = int(row["pid"])
            cpu = float(row["cpu"])
            comm = str(row["comm"]).lower()

            if _is_protected_pid(pid, protected_pids):
                continue
            if allowlist and comm not in allowlist:
                continue
            meta = _read_process_meta(pid)
            if meta is None:
                continue
            signature = _process_signature(meta)
            learned_trust = _get_learned_trust(process_memory, signature)
            seen_count = _get_seen_count(process_memory, signature)
            should_learn = (
                cpu >= (0.75 * kill_threshold_cpu_pct)
                and (
                    not is_anomaly
                    or seen_count < MIN_SIGHTINGS_BEFORE_KILL
                    or learned_trust > 0.0
                )
            )
            observed_candidates.append((signature, meta, should_learn, False))

            essentiality = _essentiality_score(
                meta,
                protected_pids=protected_pids,
                controller_uid=controller_uid,
                controller_session=controller_session,
                controller_ancestors=controller_ancestors,
            )
            candidate_score = _kill_candidate_score(
                cpu=cpu,
                threshold=kill_threshold_cpu_pct,
                predicted_cpu=predicted_cpu,
                predicted_mem=predicted_mem,
                is_anomaly=is_anomaly,
                essentiality=essentiality,
                learned_trust=learned_trust,
            )
            warmup_allowed = (
                seen_count < MIN_SIGHTINGS_BEFORE_KILL
                and not is_anomaly
                and cpu < (kill_threshold_cpu_pct + 15.0)
                and top_cpu_pct < (kill_threshold_cpu_pct + 25.0)
            )
            if warmup_allowed:
                warmup_blocked = True
                continue
            if essentiality >= 0.85 or candidate_score <= best_candidate_score or candidate_score <= 0:
                continue

            kill_candidate_pid = pid
            kill_candidate_comm = comm
            kill_candidate_signature = signature
            kill_candidate_learned_trust = learned_trust
            best_candidate_score = candidate_score

        if kill_candidate_pid <= 1:
            kill_block_reason = "learning_warmup" if warmup_blocked else "no_safe_dynamic_candidate"
        else:
            if not dry_run:
                try:
                    os.kill(kill_candidate_pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.kill(kill_candidate_pid, 0)
                    os.kill(kill_candidate_pid, signal.SIGKILL)
                except PermissionError:
                    kill_block_reason = "permission_error"
                except ProcessLookupError:
                    pass
            if kill_block_reason == "":
                kill_pids.append(kill_candidate_pid)

    for signature, meta, should_learn, _ in observed_candidates:
        was_killed = int(meta["pid"]) in kill_pids
        _update_process_memory(
            process_memory,
            signature=signature,
            meta=meta,
            observed_at=timestamp_str,
            learned=should_learn and not was_killed,
            killed=was_killed,
        )
        if signature == kill_candidate_signature:
            kill_candidate_learned_trust = _get_learned_trust(process_memory, signature)

    _save_process_memory(process_memory)

    return DecisionResult(
        timestamp=timestamp_str,
        predicted_cpu=predicted_cpu,
        predicted_mem=predicted_mem,
        anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        top_pids=top_pids,
        kill_pids=kill_pids,
        top_cpu_percent=top_cpu_pct,
        kill_cpu_threshold=kill_threshold_cpu_pct,
        kill_allow=sorted(list(allowlist)),
        kill_candidate_pid=kill_candidate_pid,
        kill_candidate_comm=kill_candidate_comm,
        kill_block_reason=kill_block_reason,
        kill_candidate_signature=kill_candidate_signature,
        kill_candidate_learned_trust=kill_candidate_learned_trust,
    )


def main() -> None:
    result = make_decision()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
