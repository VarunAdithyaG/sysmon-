from datetime import datetime, timedelta
from pathlib import Path
import csv


OUTPUT_FILE = Path(__file__).parent / "usage.csv"


def main() -> None:
    if not OUTPUT_FILE.exists():
        print(f"{OUTPUT_FILE} does not exist yet.")
        return

    now = datetime.now()
    start = now - timedelta(minutes=10)

    rows: list[list] = []
    t = start
    while t <= now:
        rows.append(
            [
                t.strftime("%Y-%m-%d %H:%M:%S"),
                90.0,  # cpu_percent
                50.0,  # gpu_percent
                85.0,  # memory_percent
                0,  # mem_used_kb (ignored by controller)
                0,  # mem_total_kb
                10.0,  # disk_percent
                70.0,  # cpu_temp_c
                60.0,  # gpu_temp_c
                "11541|7650|14975|14013|11952",  # fake top_pids
            ]
        )
        t += timedelta(minutes=1)

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)

    print(f"Injected {len(rows)} synthetic heavy-usage rows into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

