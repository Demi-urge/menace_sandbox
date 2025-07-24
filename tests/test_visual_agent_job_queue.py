import time
import threading
from menace.visual_agent_job_queue import VisualAgentJobQueue


class DummyClient:
    def __init__(self) -> None:
        self.active = 0
        self.overlap: list[bool] = []
        self.lock = threading.Lock()

    def ask(self, messages):
        with self.lock:
            self.active += 1
            if self.active > 1:
                self.overlap.append(True)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return {"choices": [{"message": {"content": "ok"}}]}

    def revert(self):
        return True


def test_queue_serialises_requests():
    client = DummyClient()
    queue = VisualAgentJobQueue(client)
    results = []

    def run():
        results.append(queue.ask([{"content": "x"}]))

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    queue.stop()

    assert len(results) == 2
    assert not client.overlap
