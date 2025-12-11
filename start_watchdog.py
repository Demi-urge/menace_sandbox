# start_watchdog.py
import json
import os
import time

WATCHDOG_PATH = "/tmp/menace_bootstrap_watchdog.json"

os.makedirs("/tmp", exist_ok=True)


def write_heartbeat(path: str = WATCHDOG_PATH) -> None:
    payload = {
        "ts": time.time(),
        "pid": os.getpid(),
        "status": "alive",
    }
    with open(path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    print("[watchdog] heartbeat emitter running...")
    while True:
        write_heartbeat()
        time.sleep(2)
