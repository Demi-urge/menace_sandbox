from pathlib import Path

def write_file():
    Path("shared.txt").write_text("hi")
