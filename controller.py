import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from monitor import log_once
from decision_engine import make_decision
from train_models import main as train_models_main


LOG_FILE = Path(__file__).parent / "monitor.log"
USAGE_FILE = Path(__file__).parent / "usage.csv"
INTERVAL_SECONDS = 20
CGROUP_ROOT = Path("/sys/fs/cgroup")


def append_log(entry: dict) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def build_time_profile() -> tuple[dict[tuple[int, int], dict[str, float]], dict[str, float]]:
    if not USAGE_FILE.exists():
        return {}, {"cpu_p50": 30.0, "cpu_p75": 50.0, "mem_p50": 45.0, "mem_p75": 65.0}
    df = pd.read_csv(USAGE_FILE, parse_dates=["timestamp"])
    if df.empty:
        return {}, {"cpu_p50": 30.0, "cpu_p75": 50.0, "mem_p50": 45.0, "mem_p75": 65.0}

    df["cpu_percent"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    df["memory_percent"] = pd.to_numeric(df["memory_percent"], errors="coerce")
    cpu = df["cpu_percent"].dropna()
    mem = df["memory_percent"].dropna()

    baseline = {
        "cpu_p50": float(cpu.quantile(0.50)) if not cpu.empty else 30.0,
        "cpu_p75": float(cpu.quantile(0.75)) if len(cpu) >= 20 else 50.0,
        "mem_p50": float(mem.quantile(0.50)) if not mem.empty else 45.0,
        "mem_p75": float(mem.quantile(0.75)) if len(mem) >= 20 else 65.0,
    }
    df["hour"] = df["timestamp"].dt.hour
    df["minute_bucket"] = (df["timestamp"].dt.minute // 5) * 5
    grouped = (
        df.groupby(["hour", "minute_bucket"])[["cpu_percent", "memory_percent"]]
        .mean()
        .reset_index()
    )
    profile: dict[tuple[int, int], dict[str, float]] = {}
    for _, row in grouped.iterrows():
        key = (int(row["hour"]), int(row["minute_bucket"]))
        profile[key] = {
            "cpu": float(row["cpu_percent"]),
            "mem": float(row["memory_percent"]),
        }
    return profile, baseline


def set_cpu_governor(mode: str) -> None:
    try:
        subprocess.run(
            ["cpupower", "frequency-set", "-g", mode],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def ensure_cgroup(path: Path, cpu_weight: int) -> None:
    try:
        path.mkdir(exist_ok=True)
        cpu_weight_file = path / "cpu.weight"
        if cpu_weight_file.exists():
            cpu_weight_file.write_text(str(cpu_weight))
    except OSError:
        return


def assign_pid_to_cgroup(pid: int, high: bool) -> None:
    if not CGROUP_ROOT.exists():
        return
    group_name = "sysmon_high" if high else "sysmon_low"
    weight = 1000 if high else 1
    group_path = CGROUP_ROOT / group_name
    ensure_cgroup(group_path, weight)
    procs_file = group_path / "cgroup.procs"
    try:
        procs_file.write_text(str(pid))
    except OSError:
        try:
            nice_val = -5 if high else 5
            os.setpriority(os.PRIO_PROCESS, pid, nice_val)
        except OSError:
            return


def main() -> None:
    print("Adaptive Controller - collecting metrics, predicting, and detecting anomalies")
    print(f"Logging controller decisions to: {LOG_FILE}")
    print(f"Interval: {INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.\n")

    print("Training models from existing usage.csv (automatic)...")
    try:
        train_models_main()
    except Exception as exc:
        print(f"Model training failed: {exc}")

    while True:
        try:
            log_once()
            kill_enabled = os.environ.get("SYSMON_ENABLE_KILL", "0") == "1"
            decision = make_decision(dry_run=not kill_enabled)

            profile, baseline = build_time_profile()
            now_dt = datetime.now()
            bucket_minute = (now_dt.minute // 5) * 5
            slot_key = (now_dt.hour, bucket_minute)
            expected = profile.get(slot_key)

            preallocate = False
            expected_cpu = None
            expected_mem = None

            expected_cpu = expected["cpu"] if expected is not None else baseline["cpu_p50"]
            expected_mem = expected["mem"] if expected is not None else baseline["mem_p50"]

            cpu_signal = max(decision.predicted_cpu, expected_cpu)
            mem_signal = max(decision.predicted_mem, expected_mem)

            cpu_trigger = baseline["cpu_p75"]
            mem_trigger = baseline["mem_p75"]
            if decision.anomaly:
                cpu_trigger *= 0.9
                mem_trigger *= 0.9

            preallocate = (cpu_signal >= cpu_trigger) or (mem_signal >= mem_trigger)
            if preallocate:
                set_cpu_governor("performance")
                for pid in decision.top_pids:
                    assign_pid_to_cgroup(pid, high=True)
            else:
                set_cpu_governor("powersave")
                for pid in decision.top_pids:
                    assign_pid_to_cgroup(pid, high=False)

            now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            summary = {
                "time": now,
                "predicted_cpu": decision.predicted_cpu,
                "predicted_mem": decision.predicted_mem,
                "expected_cpu_slot": expected_cpu,
                "expected_mem_slot": expected_mem,
                "prealloc_cpu_trigger": cpu_trigger,
                "prealloc_mem_trigger": mem_trigger,
                "preallocate": preallocate,
                "kill_enabled": kill_enabled,
                "kill_allow": os.environ.get("SYSMON_KILL_ALLOW", ""),
                "kill_cpu_threshold": os.environ.get("SYSMON_KILL_CPU", ""),
                "anomaly": decision.anomaly,
                "anomaly_score": decision.anomaly_score,
                "top_pids": decision.top_pids,
                "top_cpu_percent": decision.top_cpu_percent,
                "kill_candidate_pid": decision.kill_candidate_pid,
                "kill_candidate_comm": decision.kill_candidate_comm,
                "kill_candidate_signature": decision.kill_candidate_signature,
                "kill_candidate_learned_trust": decision.kill_candidate_learned_trust,
                "kill_block_reason": decision.kill_block_reason,
                "kill_pids": decision.kill_pids,
            }

            append_log(summary)

            status = "ANOMALY" if decision.anomaly else "normal"
            pre = "PREALLOC" if preallocate else "normal"
            print(
                f"{now} [{status}/{pre}] pred_cpu={decision.predicted_cpu:.1f}% "
                f"pred_mem={decision.predicted_mem:.1f}% "
                f"exp_cpu_slot={expected_cpu:.1f} exp_mem_slot={expected_mem:.1f} "
                f"dyn_cpu_trigger={cpu_trigger:.1f} dyn_mem_trigger={mem_trigger:.1f} "
                f"anomaly_score={decision.anomaly_score:.4f} "
                f"top_cpu={decision.top_cpu_percent:.1f}% "
                f"top_pids={decision.top_pids} kill_pids={decision.kill_pids} "
                f"learned_trust={decision.kill_candidate_learned_trust:.2f} "
                f"kill_reason={decision.kill_block_reason}"
            )

            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nController stopped.")
            break


if __name__ == "__main__":
    main()
