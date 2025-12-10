import json
import os
import time

WATCHDOG_PATH = "/tmp/menace_bootstrap_watchdog.json"

os.makedirs("/tmp", exist_ok=True)


def write_heartbeat() -> None:
    data = {
        "ts": time.time(),
        "pid": os.getpid(),
        "status": "ok",
    }
    with open(WATCHDOG_PATH, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    print("[watchdog] heartbeat emitter running...")
    while True:
        write_heartbeat()
        time.sleep(2)
