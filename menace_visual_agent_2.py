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
# ------------------------------------------------------------------
# 0️⃣  CONFIG -------------------------------------------------------
API_TOKEN = "tombalolosvisualagent123"
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
task_queue = deque()
job_status = {}


def _queue_worker():
    while True:
        if not task_queue:
            time.sleep(0.1)
            continue
        task = task_queue.popleft()
        tid = task["id"]
        job_status[tid]["status"] = "running"
        _running_lock.acquire()
        try:
            _global_lock.acquire(timeout=0)
        except Timeout:
            _running_lock.release()
            job_status[tid]["status"] = "failed"
            continue
        try:
            _current_job["active"] = True
            run_menace_pipeline(task["prompt"], task["branch"])
            job_status[tid]["status"] = "completed"
        except Exception:
            job_status[tid]["status"] = "failed"
        finally:
            _current_job["active"] = False
            _running_lock.release()
            try:
                _global_lock.release()
            except Exception:
                pass


_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
_worker_thread.start()

# ------------------------------------------------------------------
# 2️⃣  END-POINT ----------------------------------------------------
@app.post("/run", status_code=202)
async def run_task(task: TaskIn, x_token: str = Header(default="")):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")

    task_id = secrets.token_hex(8)
    job_status[task_id] = {"status": "queued", "prompt": task.prompt}
    task_queue.append({"id": task_id, "prompt": task.prompt, "branch": task.branch})
    return {"id": task_id, "status": "queued"}

# Tesseract path (change this if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dataset directory
DATASET_DIR = r'C:\menace_training_dataset'

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
        start = time.time()
        last_refresh = time.time()
        TIMEOUT = 1800  # 30 minutes
        last_refresh = time.time()
        start = time.time()

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
                break

            if time.time() - start > TIMEOUT:
                print("[TIMEOUT] Gave up after 30 minutes. Returning nothing.")
                return  # ✨ quit the entire function with no further steps

            if time.time() - last_refresh >= 1200:
                print("[CTRL+R] Refresh triggered after 20 minutes.")
                pyautogui.hotkey('ctrl', 'r')
                last_refresh = time.time()

            print("…still building – re-checking in 10 s")
            time.sleep(10)

            # 💡 Check for timeout
            if time.time() - start > TIMEOUT:
                print("[TIMEOUT] build never finished.")
                return

            # ⏱️ Inject ctrl+r every 20 minutes (1200 seconds)
            if time.time() - last_refresh >= 1200:
                print("[CTRL+R] Refresh triggered after 20 minutes.")
                pyautogui.hotkey('ctrl', 'r')
                last_refresh = time.time()

            print("…still building – re-checking in 10 s")
            time.sleep(10)

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

@app.get("/status")
async def status():
    return {"active": _current_job["active"]}


@app.get("/status/{task_id}")
async def task_status(task_id: str):
    if task_id in job_status:
        info = job_status[task_id].copy()
        info["id"] = task_id
        return info
    raise HTTPException(status_code=404, detail="Not found")


# ------------------------------------------------------------------
# 3️⃣  BOOT SERVER  -------------------------------------------------
if __name__ == "__main__":
    print(f"👁️  Menace Visual Agent listening on :{HTTP_PORT}  token={API_TOKEN[:8]}...")
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, workers=1)
