import os
import pytest

# Skip entire module when hardware is unavailable
if not os.environ.get("MENACE_HARDWARE"):
    pytest.skip("hardware not available", allow_module_level=True)


class MockSerialPort:
    """Simple buffer-based serial port stub."""

    def __init__(self):
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    def read(self, size: int) -> bytes:
        out = self.buffer[:size]
        del self.buffer[:size]
        return bytes(out)


class MockGPIO:
    """Very small GPIO stub tracking pin states."""

    def __init__(self):
        self.pins = {}

    def setup(self, pin: int, mode: str) -> None:
        self.pins[pin] = 0

    def output(self, pin: int, value: int) -> None:
        self.pins[pin] = value

    def input(self, pin: int) -> int:
        return self.pins.get(pin, 0)


def test_serial_roundtrip():
    port = MockSerialPort()
    port.write(b"hello")
    assert port.read(5) == b"hello"


def test_gpio_roundtrip():
    gpio = MockGPIO()
    gpio.setup(1, "out")
    gpio.output(1, 1)
    assert gpio.input(1) == 1
