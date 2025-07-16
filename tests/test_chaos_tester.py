import sys
import time
import sqlite3
import subprocess

from menace.chaos_tester import ChaosTester


class DummyBot:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.rolled_back = False

    def check(self) -> None:
        if self.proc.poll() is not None:
            self.rollback()

    def rollback(self) -> None:
        self.rolled_back = True


class DBBot:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.recovered = False

    def operate(self) -> None:
        try:
            self.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            self.recover()

    def recover(self) -> None:
        self.recovered = True


def test_kill_process_triggers_rollback():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    bot = DummyBot(proc)
    tester = ChaosTester()
    tester.chaos_monkey(processes=[proc])
    time.sleep(0.1)
    bot.check()
    assert bot.rolled_back


def test_drop_connection_triggers_recovery():
    conn = sqlite3.connect(":memory:")
    bot = DBBot(conn)
    ChaosTester.drop_db_connection(conn)
    bot.operate()
    assert bot.recovered


def test_corrupt_network_data():
    data = b"hello"
    corrupted = ChaosTester.corrupt_network_data(data)
    assert corrupted != data
    assert len(corrupted) == len(data)


def test_validate_recovery():
    class Bot:
        def __init__(self):
            self.rolled_back = False

        def rollback(self):
            self.rolled_back = True

    bot = Bot()
    tester = ChaosTester()

    def fail():
        bot.rollback()
        raise RuntimeError("boom")

    assert tester.validate_recovery(bot, fail)
