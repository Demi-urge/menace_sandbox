# Standard library imports
import logging
import os
import time
import sqlite3

import platform
import hashlib
import shutil
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
from contextlib import suppress
try:
    from .lock_utils import SandboxLock, is_lock_stale, Timeout
    _ContextFileLock = SandboxLock  # backward compatibility
except ImportError:  # pragma: no cover - allow running as script
    from lock_utils import SandboxLock, is_lock_stale, Timeout
    _ContextFileLock = SandboxLock
import json
from pathlib import Path
import atexit
import psutil
try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore
try:
    from .visual_agent_queue import VisualAgentQueue
except ImportError:  # pragma: no cover - allow running as script
    from visual_agent_queue import VisualAgentQueue
try:
    from . import metrics_exporter
except Exception:  # pragma: no cover - allow running as script
    import metrics_exporter
# ------------------------------------------------------------------
# 0️⃣  CONFIG -------------------------------------------------------
_token = "tombalolosvisualagent123"
API_TOKEN = _token
API_TOKEN_HASH = hashlib.sha256(API_TOKEN.encode()).hexdigest()
HTTP_PORT = int(os.getenv("MENACE_AGENT_PORT", 8001))
# Optional TLS configuration
SSL_CERT_PATH = os.getenv("VISUAL_AGENT_SSL_CERT")
SSL_KEY_PATH = os.getenv("VISUAL_AGENT_SSL_KEY")
DEVICE_ID  = "desktop"

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Return True if the given PID appears to be running."""
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _extract_token(x_token: str | None, authorization: str | None) -> str:
    """Return the token from ``authorization`` or ``x_token`` headers."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    return x_token or ""


def _verify_token(x_token: str = "", authorization: str = "") -> None:
    token = _extract_token(x_token, authorization)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if token_hash != API_TOKEN_HASH:
        raise HTTPException(status_code=401, detail="Bad token")

# ------------------------------------------------------------------
# 1️⃣  FASTAPI SCHEMA ----------------------------------------------
class TaskIn(BaseModel):
    prompt: str               # text that will be typed in Codex
    branch: str | None = None # optional git branch selector

app = FastAPI(title="Menace-Visual-Agent")

@app.on_event("startup")
def _startup_load_state() -> None:
    _cleanup_stale_files()
    _setup_pid_file()
    _setup_instance_lock()
    recovered = task_queue.check_integrity()
    if os.path.exists(GLOBAL_LOCK_PATH) and _global_lock.is_lock_stale():
        with suppress(Exception):
            os.remove(GLOBAL_LOCK_PATH)
    if AUTO_RECOVER_ON_STARTUP:
        while True:
            try:
                _global_lock.acquire(timeout=0)
                break
            except Timeout:
                if _global_lock.is_lock_stale():
                    logger.info("recovered stale global lock")
                    with suppress(Exception):
                        os.remove(GLOBAL_LOCK_PATH)
                    continue
                raise SystemExit("Agent busy")
        try:
            task_queue.clear()
            job_status.clear()
            if not recovered and QUEUE_DB.exists():
                try:
                    QUEUE_DB.unlink()
                except PermissionError as exc:  # pragma: no cover - win32
                    logger.warning(
                        "failed to remove queue db %s: %s", QUEUE_DB, exc
                    )
            _load_state_locked()
        finally:
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
        if task_queue:
            logger.info("auto recovered %s queued tasks", len(task_queue))
            _log_recovery_metrics(len(task_queue))
    else:
        _initialize_state()
    _start_background_threads()

_running_lock = threading.Lock()      # ensures only one job at a time
_queue_lock   = threading.Lock()      # protects task queue and job_status
_current_job  = {"active": False, "id": None}
GLOBAL_LOCK_PATH = os.getenv(
    "VISUAL_AGENT_LOCK_FILE",
    os.path.join(tempfile.gettempdir(), "visual_agent.lock"),
)
try:
    if os.path.exists(GLOBAL_LOCK_PATH) and is_lock_stale(GLOBAL_LOCK_PATH):
        os.remove(GLOBAL_LOCK_PATH)
except Exception:  # pragma: no cover - fs errors
    logger.exception("failed to remove stale lock %s", GLOBAL_LOCK_PATH)

# File-based lock preventing multiple agent instances
_global_lock = SandboxLock(GLOBAL_LOCK_PATH)
INSTANCE_LOCK_PATH = os.getenv(
    "VISUAL_AGENT_INSTANCE_LOCK",
    os.path.join(tempfile.gettempdir(), "visual_agent.lock.tmp"),
)
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
    except Exception:
        logger.exception("failed to remove pid file %s", path)


def _read_pid_file(path: Path) -> int | None:
    """Return the PID stored in ``path`` or ``None`` if unreadable."""
    try:
        return int(path.read_text().strip())
    except Exception as exc:
        logger.error("unable to read pid from %s: %s", path, exc)
        return None


def _remove_instance_lock() -> None:
    path = Path(INSTANCE_LOCK_PATH)
    try:
        if path.exists():
            existing_pid = int(path.read_text().split(",")[0])
            if existing_pid == os.getpid():
                path.unlink()
    except Exception:
        logger.exception("failed to remove instance lock %s", path)


def _read_instance_pid(path: Path) -> int | None:
    """Return the PID stored in ``path`` or ``None`` if unreadable."""
    try:
        return int(path.read_text().split(",")[0])
    except Exception as exc:
        logger.error("unable to read lock pid from %s: %s", path, exc)
        return None


def _check_existing_instance_lock(path: Path) -> None:
    """Validate or remove an existing instance lock."""
    if not path.exists():
        return
    pid = _read_instance_pid(path)
    if pid is None or not _pid_alive(pid):
        with suppress(Exception):
            path.unlink()
        return
    if pid != os.getpid():
        raise SystemExit(
            f"Another instance of menace_visual_agent_2 is running (PID {pid})"
        )


def _setup_pid_file() -> None:
    if os.path.exists(GLOBAL_LOCK_PATH) and is_lock_stale(GLOBAL_LOCK_PATH):
        try:
            os.remove(GLOBAL_LOCK_PATH)
        except Exception:
            logger.exception("failed to remove stale lock %s", GLOBAL_LOCK_PATH)
    path = Path(PID_FILE_PATH)
    if path.exists():
        existing = _read_pid_file(path)
        if existing and existing != os.getpid() and _pid_alive(existing):
            raise SystemExit(
                f"Another instance of menace_visual_agent_2 is running (PID {existing})"
            )
        try:
            path.unlink()
        except Exception:
            logger.exception("failed to remove stale pid file %s", path)
    try:
        path.write_text(str(os.getpid()))
    finally:
        atexit.register(_cleanup_stale_files)
        atexit.register(_remove_pid_file)


def _setup_instance_lock() -> None:
    """Create a crash-resistant instance lock for this process."""
    path = Path(INSTANCE_LOCK_PATH)

    # If a lock already exists validate it before proceeding
    if path.exists():
        pid = _read_instance_pid(path)
        if pid is not None:
            if _pid_alive(pid):
                if pid != os.getpid():
                    raise SystemExit(
                        f"Another instance of menace_visual_agent_2 is running (PID {pid})"
                    )
            else:
                # Stale lock from crashed process
                with suppress(Exception):
                    path.unlink()
        else:
            # Corrupt lock file
            with suppress(Exception):
                path.unlink()

    # Write our PID and timestamp atomically
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()},{time.time()}")
        os.replace(tmp, path)
    finally:
        atexit.register(_remove_instance_lock)


def _cleanup_stale_files() -> None:
    """Remove leftover lock and PID files."""
    try:
        if os.path.exists(GLOBAL_LOCK_PATH) and (
            is_lock_stale(GLOBAL_LOCK_PATH) or _global_lock.is_lock_stale()
        ):
            os.remove(GLOBAL_LOCK_PATH)
    except Exception:  # pragma: no cover - fs errors
        logger.exception("failed to remove lock %s", GLOBAL_LOCK_PATH)

    path = Path(INSTANCE_LOCK_PATH)
    try:
        _check_existing_instance_lock(path)
    except Exception:  # pragma: no cover - fs errors
        logger.exception("failed to remove instance lock %s", path)
    path = Path(PID_FILE_PATH)
    try:
        if path.exists():
            pid = _read_pid_file(path)
            if pid is None or not _pid_alive(pid):
                path.unlink()
    except Exception:  # pragma: no cover - fs errors
        logger.exception("failed to remove pid file %s", path)

    try:
        task_queue.reset_running_tasks()
    except Exception:  # pragma: no cover - fs errors
        logger.exception("failed to reset running tasks")

# Queue management

# Database paths
DATA_DIR = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
QUEUE_FILE = DATA_DIR / "visual_agent_queue.jsonl"  # legacy path
QUEUE_DB = DATA_DIR / "visual_agent_queue.db"
RECOVERY_METRICS_FILE = DATA_DIR / "visual_agent_recovery.json"
STATE_FILE = DATA_DIR / "visual_agent_state.json"
# When true, queued tasks will be restored from disk on startup.
AUTO_RECOVER_ON_STARTUP = os.getenv("VISUAL_AGENT_AUTO_RECOVER", "1") != "0"


def _log_recovery_metrics(count: int, watchdog: bool = False) -> None:
    """Update recovery metrics file and gauges."""
    metrics = {
        "recovery_count": 0,
        "watchdog_recoveries": 0,
        "last_recovery_time": 0.0,
        "last_recovered": 0,
    }
    try:
        if RECOVERY_METRICS_FILE.exists():
            metrics.update(json.loads(RECOVERY_METRICS_FILE.read_text()))
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed reading metrics %s: %s", RECOVERY_METRICS_FILE, exc)
    metrics["recovery_count"] = float(metrics.get("recovery_count", 0)) + 1
    if watchdog:
        metrics["watchdog_recoveries"] = float(metrics.get("watchdog_recoveries", 0)) + 1
    metrics["last_recovery_time"] = time.time()
    metrics["last_recovered"] = count
    try:
        RECOVERY_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECOVERY_METRICS_FILE.write_text(json.dumps(metrics))
        try:
            gauge = getattr(metrics_exporter, "visual_agent_recoveries_total", None)
            if gauge is not None:
                gauge.set(metrics["recovery_count"])
            metrics_exporter.visual_agent_watchdog_recoveries_total.set(
                metrics["watchdog_recoveries"]
            )
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed writing metrics %s: %s", RECOVERY_METRICS_FILE, exc)


VisualAgentQueue.migrate_from_jsonl(QUEUE_DB, QUEUE_FILE)
task_queue = VisualAgentQueue(QUEUE_DB)
job_status = {}
last_completed_ts = 0.0
_exit_event = threading.Event()

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_state_locked() -> None:
    if AUTO_RECOVER_ON_STARTUP:
        task_queue.reset_running_tasks()
    global last_completed_ts
    job_status.clear()
    job_status.update(task_queue.get_status())
    last_completed_ts = task_queue.get_last_completed()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if isinstance(data, dict):
                status = data.get("status")
                if isinstance(status, dict):
                    for tid, info in status.items():
                        if isinstance(info, dict) and "status" in info:
                            job_status[tid] = dict(info)
                lc = data.get("last_completed")
                if isinstance(lc, (int, float)):
                    last_completed_ts = float(lc)
        except Exception as exc:  # pragma: no cover - fs errors
            logger.warning("failed loading state %s: %s", STATE_FILE, exc)

    _validate_job_status()


def _validate_job_status() -> None:
    """Ensure ``job_status`` entries match the persistent queue."""
    db_status = task_queue.get_status()
    changed = False
    for tid, info in list(job_status.items()):
        db_info = db_status.get(tid)
        if db_info is None:
            if info.get("status") in {"queued", "running"}:
                task_queue.append(
                    {
                        "id": tid,
                        "prompt": info.get("prompt", ""),
                        "branch": info.get("branch"),
                    }
                )
                job_status[tid]["status"] = "queued"
                changed = True
            else:
                job_status.pop(tid, None)
                changed = True
        else:
            if db_info.get("status") != info.get("status"):
                job_status[tid] = dict(db_info)
                changed = True

    if changed:
        _write_legacy_queue()


def _write_legacy_queue() -> None:
    """Persist queued tasks to the legacy JSONL file."""
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as fh:
            for item in task_queue.load_all():
                fh.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "prompt": item.get("prompt", ""),
                            "branch": item.get("branch"),
                        }
                    )
                    + "\n"
                )
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed updating legacy queue %s: %s", QUEUE_FILE, exc)


def _persist_state() -> None:
    """Persist queue status and last completed timestamp to ``STATE_FILE``."""
    state = {
        "status": task_queue.get_status(),
        "last_completed": last_completed_ts,
    }
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, STATE_FILE)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed persisting state %s: %s", STATE_FILE, exc)
        with suppress(Exception):
            if tmp.exists():
                tmp.unlink()


def _recover_queue(*, watchdog: bool = False) -> None:
    """Attempt to rebuild the queue database after a failure."""
    logger.warning("recovering task queue")
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        if _global_lock.is_lock_stale():
            logger.info("recovered stale global lock")
            with suppress(Exception):
                os.remove(GLOBAL_LOCK_PATH)
            _global_lock.acquire(timeout=0)
        else:
            logger.error("unable to acquire global lock for recovery")
            return
    try:
        with _queue_lock:
            rebuilt = task_queue.check_integrity()
            if rebuilt:
                _load_state_locked()
            task_queue.reset_running_tasks()
            job_status.clear()
            job_status.update(task_queue.get_status())
            global last_completed_ts
            last_completed_ts = task_queue.get_last_completed()
            _log_recovery_metrics(len(task_queue), watchdog)
    finally:
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
    logger.info("queue recovery finished with %s tasks", len(task_queue))


def _queue_worker():
    while not _exit_event.is_set():
        try:
            queue_empty = not task_queue
        except sqlite3.DatabaseError:
            logger.exception("queue database error on len")
            _recover_queue()
            continue
        if queue_empty:
            _exit_event.wait(0.1)
            continue
        with _queue_lock:
            try:
                task = task_queue.popleft()
            except sqlite3.DatabaseError:
                logger.exception("queue database error on popleft")
                _recover_queue()
                continue
        if _exit_event.is_set():
            with _queue_lock:
                try:
                    task_queue.update_status(task["id"], "queued")
                except sqlite3.DatabaseError:
                    logger.exception("queue database error on update_status")
                    _recover_queue()
            break
        tid = task["id"]
        _running_lock.acquire()
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            _running_lock.release()
            with _queue_lock:
                job_status[tid]["status"] = "failed"
                try:
                    task_queue.update_status(tid, "failed")
                except sqlite3.DatabaseError:
                    logger.exception("queue database error on update_status")
                    _recover_queue()
            continue
        with _queue_lock:
            job_status[tid]["status"] = "running"
            try:
                task_queue.update_status(tid, "running")
            except sqlite3.DatabaseError:
                logger.exception("queue database error on update_status")
                _recover_queue()
                _running_lock.release()
                continue
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
        success = False
        try:
            _current_job["active"] = True
            _current_job["id"] = tid
            run_menace_pipeline(task["prompt"], task["branch"])
            success = True
        except Exception as exc:
            logger.exception("run_menace_pipeline failed for task %s", tid)
            with _queue_lock:
                job_status[tid]["status"] = "failed"
                job_status[tid]["error"] = str(exc)
                try:
                    task_queue.update_status(tid, "failed", str(exc))
                except sqlite3.DatabaseError:
                    logger.exception("queue database error on update_status")
                    _recover_queue()
        finally:
            _current_job["active"] = False
            _current_job["id"] = None
            _running_lock.release()
            if success:
                global last_completed_ts
                last_completed_ts = time.time()
                try:
                    task_queue.set_last_completed(last_completed_ts)
                except sqlite3.DatabaseError:
                    logger.exception("queue database error on set_last_completed")
                    _recover_queue()

        with _queue_lock:
            if success:
                job_status[tid]["status"] = "completed"
                job_status[tid].pop("error", None)
                try:
                    task_queue.update_status(tid, "completed")
                except sqlite3.DatabaseError:
                    logger.exception("queue database error on update_status")
                    _recover_queue()

        if _exit_event.is_set():
            break


def _initialize_state() -> None:
    acquired = False
    try:
        _global_lock.acquire(timeout=0)
        acquired = True
    except Timeout:
        if _global_lock.is_lock_stale():
            logger.info("recovered stale global lock")
            with suppress(Exception):
                os.remove(GLOBAL_LOCK_PATH)
            _global_lock.acquire(timeout=0)
            acquired = True
        else:
            job_status.clear()
            job_status.update(task_queue.get_status())
            return
    try:
        _load_state_locked()
    finally:
        if acquired:
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)


_worker_thread = None
_watchdog_thread = None
_watchdog = None


class QueueWatchdog(threading.Thread):
    """Background watchdog monitoring the queue worker."""

    def __init__(self, interval: float = float(os.getenv("VA_WATCHDOG_INTERVAL", "5"))):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.restarts = 0

    def run(self) -> None:  # pragma: no cover - timing sensitive
        while not _exit_event.is_set():
            time.sleep(self.interval)
            alive = _worker_thread.is_alive() if _worker_thread else False
            rebuilt = False
            try:
                rebuilt = task_queue.check_integrity()
            except sqlite3.DatabaseError:
                logger.exception("queue integrity check failed")
                rebuilt = True
            if not alive or rebuilt:
                logger.warning("queue watchdog triggered recovery")
                _recover_queue(watchdog=True)
                self.restarts += 1
                _start_background_threads()

def _start_background_threads() -> None:
    global _worker_thread, _watchdog_thread, _watchdog
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_queue_worker, daemon=True)
        _worker_thread.start()
    if _watchdog_thread is None or not _watchdog_thread.is_alive():
        _watchdog = QueueWatchdog()
        _watchdog_thread = _watchdog
        _watchdog_thread.start()

# ------------------------------------------------------------------
# 2️⃣  END-POINT ----------------------------------------------------
@app.post("/run", status_code=202)
async def run_task(
    task: TaskIn,
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)

    task_id = secrets.token_hex(8)
    with _queue_lock:
        job_status[task_id] = {"status": "queued", "prompt": task.prompt, "branch": task.branch}
        task_queue.append({"id": task_id, "prompt": task.prompt, "branch": task.branch})
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            response = {"id": task_id, "status": "queued"}
        else:
            try:
                pass
            finally:
                try:
                    _global_lock.release()
                except Exception as exc:
                    logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
            response = {"id": task_id, "status": "queued"}

    return response

def _configure_tesseract() -> None:
    """Set ``pytesseract.tesseract_cmd`` depending on the host OS."""
    env_path = os.getenv("TESSERACT_CMD")
    if env_path:
        pytesseract.pytesseract.tesseract_cmd = env_path
    else:
        system = platform.system()
        if system == "Windows":
            default = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        elif system == "Darwin":
            default = "/usr/local/bin/tesseract"
        else:
            # Linux and other platforms
            candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
            path = shutil.which("tesseract")
            if path:
                candidates.insert(0, path)
            for cand in candidates:
                if os.path.exists(cand):
                    default = cand
                    break
            else:
                default = candidates[0]

        pytesseract.pytesseract.tesseract_cmd = default

    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logger.warning(
            "tesseract executable not found: %s",
            pytesseract.pytesseract.tesseract_cmd,
        )

_configure_tesseract()

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
            logger.info("found %s at %s", target_text, coords)
            click_target(*coords)
            return True
        elif target_text == "Describe a task":
            logger.warning(
                "could not detect %s; clicking fallback coordinates (603, 304)",
                target_text,
            )
            click_target(603, 304)
            return True
        else:
            logger.warning("%s not found; retrying", target_text)
            retries += 1
            time.sleep(1)

    if img is not None:
        save_screenshot(img, f"failure_{target_text}")
    logger.error("failed to find %s after %s retries", target_text, MAX_RETRIES)
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
        logger.info("ocr text: %s", text)
        save_screenshot(img, "build_loop")

        if "diff" in text.lower():
            logger.info("found 'diff' in ocr; proceeding to PR")
            click_target(1746, 154)
            time.sleep(20)
            click_target(1746, 154)
            time.sleep(3)
            return True

        if time.time() - start > timeout:
            mins = timeout // 60
            logger.error("build timeout after %s minutes", mins)
            return False

        if time.time() - last_refresh >= refresh:
            mins = refresh // 60
            logger.info("refreshing browser after %s minutes", mins)
            pyautogui.hotkey('ctrl', 'r')
            last_refresh = time.time()

        logger.info("build in progress; re-checking in 10 s")
        time.sleep(10)


def run_menace_pipeline(prompt: str, branch: str | None = None):
    _persist_state()
    try:
        if not safe_find_and_click(TRIGGERS['prompt']):
            logger.error("could not locate prompt field")
            return

        time.sleep(1.5)
        pyautogui.typewrite(prompt)
        pyautogui.press('enter')
        time.sleep(1)

        logger.info(
            "skipping OCR for 'Code'; using fallback coordinates (1304, 749)"
        )
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

        logger.info("starting click sweep")

        for x, y in coordinates:
            pyautogui.moveTo(x, y)
            pyautogui.click()
            logger.info("clicked at (%s, %s)", x, y)
            _persist_state()
            time.sleep(delay_between_clicks)

        logger.info("click sweep complete")
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
        logger.info("menace pipeline run completed")
        _persist_state()


    except Exception as e:
        logger.error("fatal error during pipeline: %s", e)
        time.sleep(1)
        img = capture_screen()
        save_screenshot(img, "fatal_error")
        _persist_state()

@app.post("/revert", status_code=202)
async def revert_patch(
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)

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
            _current_job["id"] = "clone"
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
            _current_job["id"] = None
            _running_lock.release()
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)

    threading.Thread(target=_revert_worker, daemon=True).start()
    return {"status": "revert triggered"}

@app.post("/clone", status_code=202)
async def clone_repo(
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)

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
async def cancel_task(
    task_id: str,
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)

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
            task_queue.update_status(task_id, "cancelled")
            job_status[task_id]["status"] = "cancelled"
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
    return {
        "active": _current_job["active"],
        "queue": len(task_queue),
        "running": _current_job.get("id"),
    }


@app.get("/metrics")
async def metrics():
    return {"queue": len(task_queue), "last_completed": last_completed_ts}


@app.get("/health")
async def health():
    """Return basic health information for the monitor."""
    return {
        "queue": len(task_queue),
        "worker_alive": bool(_worker_thread and _worker_thread.is_alive()),
        "watchdog_restarts": getattr(_watchdog, "restarts", 0),
    }


@app.get("/status/{task_id}")
async def task_status(task_id: str):
    if task_id in job_status:
        info = job_status[task_id].copy()
        info["id"] = task_id
        return info
    raise HTTPException(status_code=404, detail="Not found")


@app.post("/flush", status_code=200)
async def flush_queue(
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        raise HTTPException(status_code=409, detail="Agent busy")
    try:
        task_queue.clear()
        job_status.clear()
        for p in (QUEUE_DB, QUEUE_FILE):
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
async def recover_queue(
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    _verify_token(x_token, authorization)
    try:
        _global_lock.acquire(timeout=0)
    except Timeout:
        raise HTTPException(status_code=409, detail="Agent busy")
    try:
        task_queue.check_integrity()
        task_queue.clear()
        job_status.clear()
        if QUEUE_DB.exists():
            QUEUE_DB.unlink()
        _load_state_locked()
    finally:
        try:
            _global_lock.release()
        except Exception as exc:
            logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
    return {"status": "recovered", "queued": len(task_queue)}


@app.post("/integrity", status_code=200)
async def queue_integrity(
    x_token: str = Header(default=""),
    authorization: str = Header(default=""),
):
    """Validate queue DB and rebuild if corrupt."""
    _verify_token(x_token, authorization)
    rebuilt = task_queue.check_integrity()
    return {"rebuilt": rebuilt}


# ------------------------------------------------------------------
# 3️⃣  BOOT SERVER  -------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import signal

    parser = argparse.ArgumentParser(description="Menace Visual Agent")
    parser.add_argument("--flush-queue", action="store_true", help="Clear persistent queue and exit")
    parser.add_argument("--recover-queue", action="store_true", help="Reload queue from disk and exit")
    parser.add_argument("--repair-running", action="store_true", help="Reset tasks marked as running to queued and exit")
    parser.add_argument("--cleanup", action="store_true", help="Remove stale lock and PID files then exit")
    parser.add_argument("--resume", action="store_true", help="Reload queue and process without starting server")
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        default=None,
        help="Automatically recover queued tasks on startup",
    )
    parser.add_argument(
        "--no-auto-recover",
        action="store_true",
        help="Disable automatic queue recovery on startup",
    )
    args = parser.parse_args()

    if args.cleanup:
        _cleanup_stale_files()
        sys.exit(0)

    if args.flush_queue:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            logger.error("agent busy while flushing queue")
            sys.exit(1)
        try:
            task_queue.clear()
            job_status.clear()
            for p in (QUEUE_DB, QUEUE_FILE):
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
        logger.info("queue flushed")
        sys.exit(0)

    if args.recover_queue:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            logger.error("agent busy while recovering queue")
            sys.exit(1)
        try:
            task_queue.clear()
            job_status.clear()
            if QUEUE_DB.exists():
                QUEUE_DB.unlink()
            VisualAgentQueue.migrate_from_jsonl(QUEUE_DB, QUEUE_FILE)
            task_queue._init_db()
            job_status.update(task_queue.get_status())
            last_completed_ts = task_queue.get_last_completed()
        finally:
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
        logger.info("recovered %s tasks", len(task_queue))
        sys.exit(0)

    if args.repair_running:
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            logger.error("agent busy while repairing queue")
            sys.exit(1)
        try:
            with _queue_lock:
                task_queue.reset_running_tasks()
                job_status.clear()
                job_status.update(task_queue.get_status())
        finally:
            try:
                _global_lock.release()
            except Exception as exc:
                logger.warning("failed to release lock %s: %s", GLOBAL_LOCK_PATH, exc)
        logger.info("repaired running tasks")
        sys.exit(0)

    if args.resume:
        AUTO_RECOVER_ON_STARTUP = True
        _cleanup_stale_files()
        _setup_pid_file()
        _setup_instance_lock()
        _initialize_state()
        _start_background_threads()
        try:
            while task_queue or _current_job.get("active"):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        _exit_event.set()
        if _worker_thread:
            _worker_thread.join()
        sys.exit(0)

    if args.auto_recover is True:
        AUTO_RECOVER_ON_STARTUP = True
    elif args.no_auto_recover:
        AUTO_RECOVER_ON_STARTUP = False

    _cleanup_stale_files()
    _setup_pid_file()
    _setup_instance_lock()
    logger.info(
        "Menace Visual Agent listening on :%s  token=%s",
        HTTP_PORT,
        API_TOKEN[:8],
    )

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=HTTP_PORT,
        workers=1,
        ssl_certfile=SSL_CERT_PATH,
        ssl_keyfile=SSL_KEY_PATH,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    def _handle_signal(sig, frame):
        _exit_event.set()
        server.handle_exit(sig, frame)
        _remove_instance_lock()
        _remove_pid_file()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    try:
        server.run()
    finally:
        _exit_event.set()
        if _worker_thread:
            _worker_thread.join()
        _remove_instance_lock()
        _remove_pid_file()
