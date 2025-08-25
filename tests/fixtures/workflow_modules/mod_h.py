from pathlib import Path

def read_file():
    return Path("shared.txt").read_text()
