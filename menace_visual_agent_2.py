# Standard library imports
import logging
import os
import time
from datetime import datetime

# Third party imports
import cv2
import numpy as np
import pytesseract
import mss
import pyautogui
# --- new imports ---
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn
import secrets
import tempfile
from filelock import FileLock, Timeout
from collections import deque
import json
from pathlib import Path
import atexit
import psutil
# ------------------------------------------------------------------
# 0️⃣  CONFIG -------------------------------------------------------
API_TOKEN = os.getenv("VISUAL_AGENT_TOKEN", "tombalolosvisualagent123")
HTTP_PORT = int(os.getenv("MENACE_AGENT_PORT", 8001))
DEVICE_ID  = "desktop"

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1️⃣  FASTAPI SCHEMA ----------------------------------------------
class TaskIn(BaseModel):
    prompt: str               # text that will be typed in Codex
    branch: str | None = None # optional git branch selector

app = FastAPI(title="Menace-Visual-Agent")

@app.on_event("startup")
def _startup_load_state() -> None:
    _setup_pid_file()
    _initialize_state()
    _start_background_threads()

_running_lock = threading.Lock()      # ensures only one job at a time
_current_job  = {"active": False}
GLOBAL_LOCK_PATH = os.getenv(
    "VISUAL_AGENT_LOCK_FILE",
    os.path.join(tempfile.gettempdir(), "visual_agent.lock"),
)
_global_lock = FileLock(GLOBAL_LOCK_PATH)

# PID file setup
PID_FILE_PATH = os.getenv(
    "VISUAL_AGENT_PID_FILE",
    os.path.join(tempfile.gettempdir(), "visual_agent.pid"),
)

def _remove_pid_file() -> None:
    path = Path(PID_FILE_PATH)
    try:
        if path.exists():
            existing = int(path.read_text().strip())
            if existing == os.getpid():
                path.unlink()
    except Exception as exc:
        logger.warning("failed to remove pid file %s: %s", path, exc)


def _setup_pid_file() -> None:
    path = Path(PID_FILE_PATH)
    if path.exists():
        try:
            existing = int(path.read_text().strip())
            if existing != os.getpid() and psutil.pid_exists(existing):
                raise SystemExit(
                    f"Another instance of menace_visual_agent_2 is running (PID {existing})"
                )
        except Exception as exc:
            logger.warning("failed reading pid from %s: %s", path, exc)
        try:
            path.unlink()
        except Exception as exc:
            logger.warning("failed to remove stale pid file %s: %s", path, exc)
    try:
        path.write_text(str(os.getpid()))
    finally:
        atexit.register(_remove_pid_file)

# Queue management
import hashlib

DATA_DIR = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
QUEUE_FILE = DATA_DIR / "visual_agent_queue.jsonl"
STATE_FILE = DATA_DIR / "visual_agent_state.json"
HASH_FILE = STATE_FILE.with_suffix(STATE_FILE.suffix + ".sha256")
BACKUP_COUNT = 3


class PersistentQueue:
    """Persistent queue stored as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._queue: deque[dict] = deque()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> None:
        with self._lock:
            self._queue.clear()
            if not self.path.exists():
                return
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            self._queue.append(item)
            except Exception:
                self._queue.clear()

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for item in self._queue:
                fh.write(json.dumps(item) + "\n")
        os.replace(tmp, self.path)

    def append(self, item: dict) -> None:
        with self._lock:
            self._queue.append(item)
            self.save()

    def appendleft(self, item: dict) -> None:
        with self._lock:
            self._queue.appendleft(item)
            self.save()

    def popleft(self) -> dict:
        with self._lock:
            item = self._queue.popleft()
            self.save()
            return item

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            if self.path.exists():
                try:
                    os.remove(self.path)
                except Exception:
                    pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return len(self) > 0

    def __iter__(self):
        with self._lock:
            return iter(list(self._queue))


task_queue = PersistentQueue(QUEUE_FILE)
job_status = {}
last_completed_ts = 0.0
_exit_event = threading.Event()

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_state_locked() -> None:
    """Persist queue and status atomically."""
    global last_completed_ts
    task_queue.save()
    data = {
        "status": job_status,
        "last_completed": last_completed_ts,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=STATE_FILE.parent, encoding="utf-8"
    ) as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
        tmp_path = Path(fh.name)
    os.replace(tmp_path, STATE_FILE)
    HASH_FILE.write_text(hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _recover_queue_file_locked() -> None:
    """Backup corrupt queue file and reset state."""
    if not STATE_FILE.exists() and not QUEUE_FILE.exists():
        return

    backups = [STATE_FILE.with_suffix(STATE_FILE.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    queue_backups = [QUEUE_FILE.with_suffix(QUEUE_FILE.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    hash_backups = [HASH_FILE.with_suffix(HASH_FILE.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    try:
        for i in range(BACKUP_COUNT - 1, 0, -1):
            if backups[i - 1].exists():
                if backups[i].exists():
                    backups[i].unlink()
                os.replace(backups[i - 1], backups[i])
            if queue_backups[i - 1].exists():
                if queue_backups[i].exists():
                    queue_backups[i].unlink()
                os.replace(queue_backups[i - 1], queue_backups[i])
            if hash_backups[i - 1].exists():
                if hash_backups[i].exists():
                    hash_backups[i].unlink()
                os.replace(hash_backups[i - 1], hash_backups[i])
        if STATE_FILE.exists():
            os.replace(STATE_FILE, backups[0])
        if QUEUE_FILE.exists():
            os.replace(QUEUE_FILE, queue_backups[0])
        if HASH_FILE.exists():
            os.replace(HASH_FILE, hash_backups[0])
    except OSError:
        for p in (STATE_FILE, QUEUE_FILE, HASH_FILE):
            try:
                p.unlink()
            except OSError:
                pass
    task_queue.clear()
    job_status.clear()
    global last_completed_ts
    last_completed_ts = 0.0
    _save_state_locked()


def _load_state_locked() -> None:
    task_queue.load()
    if not STATE_FILE.exists():
        return
    if not HASH_FILE.exists():
        try:
            data_bytes = STATE_FILE.read_bytes()
            data = json.loads(data_bytes.decode("utf-8"))
            HASH_FILE.write_text(hashlib.sha256(data_bytes).hexdigest())
        except Exception:
            _recover_queue_file_locked()
            return
    else:
        try:
            expected = HASH_FILE.read_text().strip()
            data_bytes = STATE_FILE.read_bytes()
            if hashlib.sha256(data_bytes).hexdigest() != expected:
                raise ValueError("checksum mismatch")
            data = json.loads(data_bytes.decode("utf-8"))
        except Exception:
            _recover_queue_file_locked()
            return

    if not isinstance(data, dict):
        _recover_queue_file_locked()
        return

    global last_completed_ts

    raw_status = data.get("status")
    if not isinstance(raw_status, dict):
        raw_status = {}

    last_completed = data.get("last_completed")
    if isinstance(last_completed, (int, float)):
        last_completed_ts = float(last_completed)
    else:
        last_completed_ts = 0.0

    valid_status = {}
    for tid, info in raw_status.items():
        if not isinstance(tid, str) or not isinstance(info, dict):
            continue
        status = info.get("status", "queued")
        if status not in {"queued", "running", "completed", "cancelled", "failed"}:
            status = "queued"
        prompt = info.get("prompt", "") if isinstance(info.get("prompt", ""), str) else ""
        branch = info.get("branch") if isinstance(info.get("branch"), str) or info.get("branch") is None else None
        error_msg = info.get("error") if isinstance(info.get("error"), str) else None
        entry = {"status": status, "prompt": prompt, "branch": branch}
        if error_msg:
            entry["error"] = error_msg
        valid_status[tid] = entry

    job_status.clear()
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
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)


def _queue_worker():
    while not _exit_event.is_set():
        if not task_queue:
            _exit_event.wait(0.1)
            continue
        task = task_queue.popleft()
        if _exit_event.is_set():
            task_queue.appendleft(task)
            break
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
        success = False
        try:
            _current_job["active"] = True
            run_menace_pipeline(task["prompt"], task["branch"])
            job_status[tid]["status"] = "completed"
            job_status[tid].pop("error", None)
            success = True
        except Exception as exc:
            logger.exception("run_menace_pipeline failed for task %s", tid)
            job_status[tid]["status"] = "failed"
            job_status[tid]["error"] = str(exc)
            _save_state_locked()
        finally:
            _current_job["active"] = False
            _running_lock.release()
            if success:
                global last_completed_ts
                last_completed_ts = time.time()
            _save_state_locked()
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)

        _persist_state()

        if _exit_event.is_set():
            break


def _autosave_worker() -> None:
    while not _exit_event.wait(5):
        _persist_state()


def _initialize_state() -> None:
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        return
    try:
        _load_state_locked()
    finally:
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)


_worker_thread = None
_autosave_thread = None

def _start_background_threads() -> None:
    global _worker_thread, _autosave_thread
    if _worker_thread is not None:
        return
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
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)

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
        _persist_state()
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
    _persist_state()
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
            _persist_state()
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
        _persist_state()


    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        time.sleep(1)
        img = capture_screen()
        save_screenshot(img, "fatal_error")
        _persist_state()

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
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)

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
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)

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
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
    finally:
        try:
            _running_lock.release()
        except Exception as exc:
            logger.warning("failed to release running lock: %s", exc)

@app.get("/status")
async def status():
    return {"active": _current_job["active"], "queue": len(task_queue)}


@app.get("/metrics")
async def metrics():
    return {"queue": len(task_queue), "last_completed": last_completed_ts}


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
        for p in (QUEUE_FILE, STATE_FILE, HASH_FILE):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
    finally:
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
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
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
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
            for p in (QUEUE_FILE, STATE_FILE, HASH_FILE):
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        finally:
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
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
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
        print(f"Recovered {len(task_queue)} tasks")
        sys.exit(0)


    _setup_pid_file()
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
    if _worker_thread:
        _worker_thread.join()
    if _autosave_thread:
        _autosave_thread.join()
