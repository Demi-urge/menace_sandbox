from menace_sandbox.menace_tasks import ping

if __name__ == "__main__":
    result = ping.delay()
    print("Task sent. Waiting for result...")
    print(result.get(timeout=10))
