# start_watchdog.py
import json
import os
import time

WATCHDOG_PATH = "/tmp/menace_bootstrap_watchdog.json"

os.makedirs("/tmp", exist_ok=True)

print("[watchdog] heartbeat emitter running...")

while True:
    with open(WATCHDOG_PATH, "w") as f:
        json.dump({"heartbeat": time.time()}, f)
    time.sleep(2)
