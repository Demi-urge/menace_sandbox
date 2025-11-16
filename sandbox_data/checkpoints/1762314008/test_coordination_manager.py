import menace.coordination_manager as cm


def test_send_receive(tmp_path):
    log = cm.MessageLog(tmp_path / "m.db")
    manager = cm.CoordinationManager(log=log)
    msg = cm.Message(sender="a", recipient="b", task="t", payload="p")
    manager.send(msg)
    got = manager.receive()
    assert got and got.sender == "a"
    rows = log.fetch()
    assert rows and rows[0][0] == "a"


def test_task_distributor():
    dist = cm.TaskDistributor()
    tasks = ["t1", "t2", "t3"]
    bots = ["b1", "b2"]
    assigns = dist.assign(tasks, bots)
    assert assigns == [("b1", "t1"), ("b2", "t2"), ("b1", "t3")]
