#!/usr/bin/env python3
"""
System Monitor - Reads from kernel module and logs to CSV
Kernel module provides memory stats, this script adds CPU, GPU, temps
"""

import os
import re
import time
import csv
import subprocess
from datetime import datetime
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "usage.csv"
KERNEL_STATS = "/proc/sysmon/stats"
INTERVAL = 20  # seconds
PROCESS_SAMPLE_SECONDS = 0.35
CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


def _read_total_cpu_ticks():
    with open("/proc/stat") as f:
        line = f.readline()
    parts = line.split()[1:]
    return sum(int(part) for part in parts)


def _read_pid_cpu_snapshot():
    snapshot = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        stat_path = proc_dir / "stat"
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
        except OSError:
            continue

        try:
            after = stat_text.rsplit(")", 1)[1].strip().split()
            state = after[0]
            utime = int(after[11])
            stime = int(after[12])
        except (IndexError, ValueError):
            continue

        if state == "Z":
            continue
        try:
            os.readlink(proc_dir / "exe")
        except OSError:
            continue

        snapshot[int(proc_dir.name)] = utime + stime
    return snapshot


def read_kernel_stats():
    """Read stats from kernel module."""
    try:
        with open(KERNEL_STATS) as f:
            line = f.read().strip()
        parts = line.split(",")
        if len(parts) >= 3:
            return {
                "timestamp": int(parts[0]),
                "mem_total_kb": int(parts[1]),
                "mem_free_kb": int(parts[2])
            }
    except (IOError, ValueError):
        pass
    return None


def get_cpu_usage():
    """Read current CPU usage over a short interval."""
    total_1 = _read_total_cpu_ticks()
    with open("/proc/stat") as f:
        idle_1 = sum(int(x) for x in f.readline().split()[4:6])
    time.sleep(0.2)
    total_2 = _read_total_cpu_ticks()
    with open("/proc/stat") as f:
        idle_2 = sum(int(x) for x in f.readline().split()[4:6])

    total_delta = total_2 - total_1
    idle_delta = idle_2 - idle_1
    if total_delta <= 0:
        return 0.0
    used_pct = (1.0 - (idle_delta / total_delta)) * 100.0
    return round(max(0.0, used_pct), 1)


def get_gpu_usage():
    """Get GPU usage percentage."""
    # AMD/Intel GPU via /sys/class/drm
    for card in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"):
        try:
            val = int(card.read_text().strip())
            return round(float(val), 1)
        except (ValueError, OSError):
            continue
    
    # NVIDIA GPU via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip().split("\n")[0]), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def get_cpu_temp():
    """Read CPU temperature from /sys/class/thermal."""
    temps = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temp = int(zone.read_text().strip()) / 1000.0
            if 0 <= temp <= 120:
                temps.append(temp)
        except (ValueError, OSError):
            continue
    return round(max(temps), 1) if temps else None


def get_top_pids(limit=5):
    """Get top N user-space process IDs by recent CPU usage."""
    start = time.time()
    snap_1 = _read_pid_cpu_snapshot()
    time.sleep(PROCESS_SAMPLE_SECONDS)
    elapsed = max(time.time() - start, 1e-6)
    snap_2 = _read_pid_cpu_snapshot()
    procs = []
    for pid, ticks_2 in snap_2.items():
        ticks_1 = snap_1.get(pid)
        if ticks_1 is None:
            continue
        tick_delta = ticks_2 - ticks_1
        if tick_delta <= 0:
            continue
        cpu = (tick_delta / CLK_TCK) / elapsed * 100.0
        procs.append((pid, cpu))

    procs.sort(key=lambda x: x[1], reverse=True)
    return [str(p[0]) for p in procs[:limit]], (round(procs[0][1], 1) if procs else 0.0)


def get_disk_usage():
    """Get disk usage percentage."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    pct_str = parts[4].replace("%", "")
                    return round(float(pct_str), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_gpu_temp():
    """Get GPU temperature."""
    # AMD/Intel GPU temp
    for card in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/temp1_input"):
        try:
            temp = int(card.read_text().strip()) / 1000.0
            if 0 <= temp <= 120:
                return round(temp, 1)
        except (ValueError, OSError):
            continue
    
    # NVIDIA GPU temp
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip().split("\n")[0]), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def log_once():
    """Collect stats and append to CSV."""
    kernel = read_kernel_stats()
    cpu = get_cpu_usage()
    gpu = get_gpu_usage()
    cpu_temp = get_cpu_temp()
    gpu_temp = get_gpu_temp()
    disk = get_disk_usage()
    top_pids, top_cpu_pct = get_top_pids(5)
    
    if kernel:
        mem_total = kernel["mem_total_kb"]
        mem_free = kernel["mem_free_kb"]
        mem_used = mem_total - mem_free
        mem_pct = round(mem_used * 100 / mem_total, 1) if mem_total > 0 else 0
    else:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    mem[key.strip()] = int(val.split()[0])
        mem_total = mem.get("MemTotal", 0)
        mem_available = mem.get("MemAvailable", mem.get("MemFree", 0))
        mem_used = mem_total - mem_available
        mem_pct = round(mem_used * 100 / mem_total, 1) if mem_total > 0 else 0
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_exists = False
    try:
        with open(OUTPUT_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "cpu_percent",
                    "gpu_percent",
                    "memory_percent",
                    "mem_used_kb",
                    "mem_total_kb",
                    "disk_percent",
                    "cpu_temp_c",
                    "gpu_temp_c",
                    "top_pids",
                    "top_cpu_percent",
                ]
            )
        writer.writerow(
            [
                now,
                cpu,
                gpu if gpu is not None else "",
                mem_pct,
                mem_used,
                mem_total,
                disk if disk is not None else "",
                cpu_temp if cpu_temp is not None else "",
                gpu_temp if gpu_temp is not None else "",
                "|".join(top_pids),
                top_cpu_pct,
            ]
        )
        f.flush()
    
    gpu_str = f"{gpu}%" if gpu is not None else "N/A"
    disk_str = f"{disk}%" if disk is not None else "N/A"
    cpu_temp_str = f"{cpu_temp}°C" if cpu_temp is not None else "N/A"
    gpu_temp_str = f"{gpu_temp}°C" if gpu_temp is not None else "N/A"
    mem_mb = mem_used // 1024
    kernel_status = "✓" if kernel else "✗"
    pids_str = ",".join(top_pids)
    print(f"{now} [{kernel_status}] CPU: {cpu}%  GPU: {gpu_str}  Mem: {mem_pct}%  Disk: {disk_str}  Top PIDs: {pids_str}")


def main():
    print("System Monitor - Reading from kernel module")
    print(f"Kernel module: {'loaded' if Path(KERNEL_STATS).exists() else 'not loaded (using fallback)'}")
    print(f"Logging to: {OUTPUT_FILE}")
    print(f"Interval: {INTERVAL} seconds. Press Ctrl+C to stop.\n")
    
    while True:
        try:
            log_once()
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
