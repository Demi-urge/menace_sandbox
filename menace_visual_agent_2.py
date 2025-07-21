import cv2
import numpy as np
import pytesseract
import mss
import time
import pyautogui
import os
from datetime import datetime
# --- new imports ---
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn
import secrets
import os
import tempfile
from filelock import FileLock, Timeout
from collections import deque
import json
from pathlib import Path
# ------------------------------------------------------------------
# 0️⃣  CONFIG -------------------------------------------------------
API_TOKEN = os.getenv("VISUAL_AGENT_TOKEN", "tombalolosvisualagent123")
HTTP_PORT = int(os.getenv("MENACE_AGENT_PORT", 8001))
DEVICE_ID  = "desktop"

# ------------------------------------------------------------------
# 1️⃣  FASTAPI SCHEMA ----------------------------------------------
class TaskIn(BaseModel):
    prompt: str               # text that will be typed in Codex
    branch: str | None = None # optional git branch selector

app = FastAPI(title="Menace-Visual-Agent")

_running_lock = threading.Lock()      # ensures only one job at a time
_current_job  = {"active": False}
GLOBAL_LOCK_PATH = os.getenv(
    "VISUAL_AGENT_LOCK_FILE",
    os.path.join(tempfile.gettempdir(), "visual_agent.lock"),
)
_global_lock = FileLock(GLOBAL_LOCK_PATH)

# Queue management
import hashlib

DATA_DIR = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
QUEUE_FILE = DATA_DIR / "visual_agent_queue.json"
HASH_FILE = QUEUE_FILE.with_suffix(QUEUE_FILE.suffix + ".sha256")
BACKUP_COUNT = 3
task_queue = deque()
job_status = {}
_exit_event = threading.Event()

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_state_locked() -> None:
    """Persist queue and status atomically."""
    data = {"queue": list(task_queue), "status": job_status}
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=QUEUE_FILE.parent, encoding="utf-8"
    ) as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
        tmp_path = Path(fh.name)
    os.replace(tmp_path, QUEUE_FILE)
    HASH_FILE.write_text(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _recover_queue_file_locked() -> None:
    """Backup corrupt queue file and reset state."""
    if not QUEUE_FILE.exists():
        return

    backups = [QUEUE_FILE.with_suffix(QUEUE_FILE.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    hash_backups = [HASH_FILE.with_suffix(HASH_FILE.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    try:
        for i in range(BACKUP_COUNT - 1, 0, -1):
            if backups[i - 1].exists():
                if backups[i].exists():
                    backups[i].unlink()
                os.replace(backups[i - 1], backups[i])
            if hash_backups[i - 1].exists():
                if hash_backups[i].exists():
                    hash_backups[i].unlink()
                os.replace(hash_backups[i - 1], hash_backups[i])
        if QUEUE_FILE.exists():
            os.replace(QUEUE_FILE, backups[0])
        if HASH_FILE.exists():
            os.replace(HASH_FILE, hash_backups[0])
    except OSError:
        try:
            QUEUE_FILE.unlink()
        except OSError:
            pass
        try:
            HASH_FILE.unlink()
        except OSError:
            pass
    task_queue.clear()
    job_status.clear()
    _save_state_locked()


def _load_state_locked() -> None:
    if not QUEUE_FILE.exists():
        return
    if not HASH_FILE.exists():
        _recover_queue_file_locked()
        return

    try:
        expected = HASH_FILE.read_text().strip()
        data_bytes = QUEUE_FILE.read_bytes()
        if hashlib.sha256(data_bytes).hexdigest() != expected:
            raise ValueError("checksum mismatch")
        data = json.loads(data_bytes.decode("utf-8"))
    except Exception:
        _recover_queue_file_locked()
        return

    if not isinstance(data, dict):
        _recover_queue_file_locked()
        return

    raw_queue = data.get("queue")
    if not isinstance(raw_queue, list):
        raw_queue = []

    raw_status = data.get("status")
    if not isinstance(raw_status, dict):
        raw_status = {}

    valid_queue = []
    for item in raw_queue:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("prompt"), str):
            branch = item.get("branch") if isinstance(item.get("branch"), str) or item.get("branch") is None else None
            valid_queue.append({"id": item["id"], "prompt": item["prompt"], "branch": branch})

    valid_status = {}
    for tid, info in raw_status.items():
        if not isinstance(tid, str) or not isinstance(info, dict):
            continue
        status = info.get("status", "queued")
        if status not in {"queued", "running", "completed", "cancelled", "failed"}:
            status = "queued"
        prompt = info.get("prompt", "") if isinstance(info.get("prompt", ""), str) else ""
        branch = info.get("branch") if isinstance(info.get("branch"), str) or info.get("branch") is None else None
        valid_status[tid] = {"status": status, "prompt": prompt, "branch": branch}

    task_queue.clear()
    job_status.clear()
    for item in valid_queue:
        task_queue.append(item)
    job_status.update(valid_status)
    for tid, info in job_status.items():
        if info.get("status") not in {"completed", "cancelled", "failed"}:
            info["status"] = "queued"
            if not any(t.get("id") == tid for t in task_queue):
                task_queue.append({"id": tid, "prompt": info.get("prompt", ""), "branch": info.get("branch")})

    _save_state_locked()


def _persist_state() -> None:
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        return
    try:
        _save_state_locked()
    finally:
        try:
            _global_lock.release()
        except Exception:
            pass


def _queue_worker():
    while not _exit_event.is_set():
        if not task_queue:
            _exit_event.wait(0.1)
            continue
        task = task_queue.popleft()
        tid = task["id"]
        _running_lock.acquire()
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            _running_lock.release()
            job_status[tid]["status"] = "failed"
            continue
        job_status[tid]["status"] = "running"
        _save_state_locked()
        try:
            _current_job["active"] = True
            run_menace_pipeline(task["prompt"], task["branch"])
            job_status[tid]["status"] = "completed"
        except Exception:
            job_status[tid]["status"] = "failed"
        finally:
            _current_job["active"] = False
            _running_lock.release()
            _save_state_locked()
            try:
                _global_lock.release()
            except Exception:
                pass

        _persist_state()

        if _exit_event.is_set():
            break


def _autosave_worker() -> None:
    while not _exit_event.wait(5):
        _persist_state()


try:
    _global_lock.acquire(timeout=0)
    try:
        _load_state_locked()
    finally:
        _global_lock.release()
except Timeout:
    pass

_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
_worker_thread.start()
_autosave_thread = threading.Thread(target=_autosave_worker, daemon=True)
_autosave_thread.start()

# ------------------------------------------------------------------
# 2️⃣  END-POINT ----------------------------------------------------
@app.post("/run", status_code=202)
async def run_task(task: TaskIn, x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        raise HTTPException(status_code=409, detail="Agent busy")

    try:
        if _running_lock.locked() or task_queue:
            raise HTTPException(status_code=409, detail="Agent busy")

        task_id = secrets.token_hex(8)
        job_status[task_id] = {"status": "queued", "prompt": task.prompt, "branch": task.branch}
        task_queue.append({"id": task_id, "prompt": task.prompt, "branch": task.branch})
        _save_state_locked()
        response = {"id": task_id, "status": "queued"}
    finally:
        try:
            _global_lock.release()
        except Exception:
            pass

    return response

# Tesseract path (change this if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dataset directory
DATASET_DIR = os.getenv("VA_DATASET_DIR", r"C:\\menace_training_dataset")

# Create dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# Screen capture dimensions
MONITOR = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

# Failure retry count
MAX_RETRIES = 3

# Text triggers for detection
TRIGGERS = {
    "prompt": "Describe a task",
    "code_button": "Code",
    "working": ["Working on your task", "starting container", "downloading repo", "thinking", "running a command"],
    "cant_help": "I'm sorry, but I can't help with that.",
    "update_branch": "Update Branch",
    "create_pr": "Create Pull Request",
    "view_pr": "View Pull Request",
    "resolve_conflicts": "Resolve conflicts",
    "merge_pr": "Merge pull request",
    "confirm_merge": "Confirm merge",
    "commit_merge": "Commit merge"
}


def capture_screen():
    with mss.mss() as sct:
        time.sleep(1)
        img = np.array(sct.grab(MONITOR))
        return img


def save_screenshot(img, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(DATASET_DIR, f"{label}_{timestamp}.png")
    cv2.imwrite(filename, img)


def ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def find_text_coordinates(img, target_text, y_min=None, y_max=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    for i, word in enumerate(data['text']):
        if target_text.lower() in word.lower():
            y = data['top'][i] + data['height'][i] // 2
            if y_min is not None and y < y_min:
                continue
            if y_max is not None and y > y_max:
                continue
            x = data['left'][i] + data['width'][i] // 2
            return x, y
    return None


def click_target(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()


def safe_find_and_click(target_text, y_min=None, y_max=None, tolerance=100):
    retries = 0
    time.sleep(1)
    img = None

    while retries < MAX_RETRIES:
        time.sleep(1)
        img = capture_screen()
        save_screenshot(img, f"attempt_{target_text}")

        coords = find_text_coordinates(img, target_text, y_min=y_min, y_max=y_max)
        if coords:
            print(f"[SUCCESS] Found '{target_text}' at {coords}")
            click_target(*coords)
            return True
        elif target_text == "Describe a task":
            print(f"[FALLBACK] Could not detect '{target_text}'. Clicking hardcoded coordinates (526, 382).")
            click_target(603, 304)
            return True
        else:
            print(f"[FAILURE] '{target_text}' not found. Retrying...")
            retries += 1
            time.sleep(1)

    if img is not None:
        save_screenshot(img, f"failure_{target_text}")
    print(f"[ABORT] Failed after {MAX_RETRIES} retries.")
    return False


def _wait_for_build(timeout: int, refresh: int) -> bool:
    """Wait for Codex to finish building and trigger the PR view.

    Args:
        timeout: Maximum number of seconds to wait.
        refresh: Interval in seconds to trigger a browser refresh.

    Returns:
        True if the build completed before timing out, False otherwise.
    """
    start = time.time()
    last_refresh = time.time()

    while True:
        time.sleep(1)
        img = capture_screen()
        text = ocr_image(img)
        print("[OCR TEXT]", text)
        save_screenshot(img, "build_loop")

        if "diff" in text.lower():
            print("[READY] Found 'diff' – proceeding to PR.")
            click_target(1746, 154)
            time.sleep(20)
            click_target(1746, 154)
            time.sleep(3)
            return True

        if time.time() - start > timeout:
            mins = timeout // 60
            print(f"[TIMEOUT] Gave up after {mins} minutes. Returning nothing.")
            return False

        if time.time() - last_refresh >= refresh:
            mins = refresh // 60
            print(f"[CTRL+R] Refresh triggered after {mins} minutes.")
            pyautogui.hotkey('ctrl', 'r')
            last_refresh = time.time()

        print("…still building – re-checking in 10 s")
        time.sleep(10)


def run_menace_pipeline(prompt: str, branch: str | None = None):
    try:
        if not safe_find_and_click(TRIGGERS['prompt']):
            print("[ERROR] Could not locate prompt field.")
            return

        time.sleep(1.5)
        pyautogui.typewrite(prompt)
        pyautogui.press('enter')
        time.sleep(1)

        print("[FALLBACK] Skipping OCR for 'Code'."
              " Clicking hardcoded coordinates (1387, 557).")
        click_target(1304, 749)
        time.sleep(10)
        click_target(665, 565)
        time.sleep(0.5)

        # ② WAIT UNTIL Codex finishes building / thinking
        if not _wait_for_build(timeout=1800, refresh=1200):
            return

        # Continue exactly as before
        click_target(562, 510)          # open the PR task row
        time.sleep(1)
        pyautogui.press('down', presses=15)  # 5️⃣ Down-arrow spam
        coordinates = [
            (450, 421), (452, 435), (452, 446), (451, 463),
            (450, 480), (452, 497), (450, 520), (451, 538),
            (453, 560), (452, 578), (456, 595), (454, 621),
            (452, 648), (456, 664), (455, 695), (457, 720)
        ]

        delay_between_clicks = 0.3  # seconds

        print("Eshgooo G 🔥 Starting click sweep...")

        for x, y in coordinates:
            pyautogui.moveTo(x, y)
            pyautogui.click()
            print(f"Clicked at: ({x}, {y})")
            time.sleep(delay_between_clicks)

        print("✅ Sweep complete. Surely Balolos-coded.")
        time.sleep(3)
        pyautogui.hotkey('ctrl', 'w')
        time.sleep(3)
        click_target(29, 152)
        pyautogui.hotkey('win', '5')
        pyautogui.write("git fetch origin")
        pyautogui.press('enter')
        time.sleep(10)
        pyautogui.write('git reset --hard origin/main')
        pyautogui.press('enter')
        time.sleep(3)
        pyautogui.hotkey('win', '1')
        time.sleep(2)
        print("Menace pipeline run completed.")


    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        time.sleep(1)
        img = capture_screen()
        save_screenshot(img, "fatal_error")

@app.post("/revert", status_code=202)
async def revert_patch(x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    if not _running_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Agent busy")

    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        _running_lock.release()
        raise HTTPException(status_code=409, detail="Agent busy")

    def _revert_worker():
        try:
            _current_job["active"] = True
            pyautogui.hotkey('win', '5')  # switch to terminal window
            time.sleep(2)
            pyautogui.write("git reset --hard HEAD~1")
            pyautogui.press('enter')
            time.sleep(3)
            pyautogui.write("git push --force")
            pyautogui.press('enter')
            time.sleep(5)
            pyautogui.hotkey('win', '1')  # return to browser
        finally:
            _current_job["active"] = False
            _running_lock.release()
            try:
                _global_lock.release()
            except Exception:
                pass

    threading.Thread(target=_revert_worker, daemon=True).start()
    return {"status": "revert triggered"}

@app.post("/clone", status_code=202)
async def clone_repo(x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    if not _running_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Agent busy")

    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        _running_lock.release()
        raise HTTPException(status_code=409, detail="Agent busy")

    def _clone_worker():
        try:
            _current_job["active"] = True
            pyautogui.hotkey('win', '5')  # switch to terminal window
            time.sleep(2)
            pyautogui.write("git clone https://github.com/Demi-urge/menace_sandbox")
            pyautogui.press('enter')
            time.sleep(3)
            pyautogui.write("git push --force")
            pyautogui.press('enter')
            time.sleep(5)
            pyautogui.hotkey('win', '1')  # return to browser
        finally:
            _current_job["active"] = False
            _running_lock.release()
            try:
                _global_lock.release()
            except Exception:
                pass

@app.post("/cancel/{task_id}", status_code=202)
async def cancel_task(task_id: str, x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    if task_id not in job_status:
        raise HTTPException(status_code=404, detail="Not found")

    if not _running_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Agent busy")

    try:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            _running_lock.release()
            raise HTTPException(status_code=409, detail="Agent busy")

        try:
            if job_status[task_id]["status"] != "queued":
                raise HTTPException(status_code=409, detail="Task already running")

            for i, t in enumerate(list(task_queue)):
                if t["id"] == task_id:
                    del task_queue[i]
                    break

            job_status[task_id]["status"] = "cancelled"
            _save_state_locked()
            return {"id": task_id, "status": "cancelled"}
        finally:
            _global_lock.release()
    finally:
        _running_lock.release()

@app.get("/status")
async def status():
    return {"active": _current_job["active"], "queue": len(task_queue)}


@app.get("/status/{task_id}")
async def task_status(task_id: str):
    if task_id in job_status:
        info = job_status[task_id].copy()
        info["id"] = task_id
        return info
    raise HTTPException(status_code=404, detail="Not found")


@app.post("/flush", status_code=200)
async def flush_queue(x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        raise HTTPException(status_code=409, detail="Agent busy")
    try:
        task_queue.clear()
        job_status.clear()
        if QUEUE_FILE.exists():
            QUEUE_FILE.unlink()
    finally:
        _global_lock.release()
    return {"status": "flushed"}


@app.post("/recover", status_code=200)
async def recover_queue(x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        raise HTTPException(status_code=409, detail="Agent busy")
    try:
        task_queue.clear()
        job_status.clear()
        _load_state_locked()
    finally:
        _global_lock.release()
    return {"status": "recovered", "queued": len(task_queue)}


# ------------------------------------------------------------------
# 3️⃣  BOOT SERVER  -------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import signal

    parser = argparse.ArgumentParser(description="Menace Visual Agent")
    parser.add_argument("--flush-queue", action="store_true", help="Clear persistent queue and exit")
    parser.add_argument("--recover-queue", action="store_true", help="Reload queue from disk and exit")
    args = parser.parse_args()

    if args.flush_queue:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            print("Agent busy", file=sys.stderr)
            sys.exit(1)
        try:
            task_queue.clear()
            job_status.clear()
            if QUEUE_FILE.exists():
                QUEUE_FILE.unlink()
        finally:
            _global_lock.release()
        print("Queue flushed")
        sys.exit(0)

    if args.recover_queue:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            print("Agent busy", file=sys.stderr)
            sys.exit(1)
        try:
            task_queue.clear()
            job_status.clear()
            _load_state_locked()
        finally:
            _global_lock.release()
        print(f"Recovered {len(task_queue)} tasks")
        sys.exit(0)

    print(f"👁️  Menace Visual Agent listening on :{HTTP_PORT}  token={API_TOKEN[:8]}...")

    config = uvicorn.Config(app, host="0.0.0.0", port=HTTP_PORT, workers=1)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    def _handle_signal(sig, frame):
        _exit_event.set()
        server.handle_exit(sig, frame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    server.run()
    _exit_event.set()
    _worker_thread.join()
    _autosave_thread.join()
