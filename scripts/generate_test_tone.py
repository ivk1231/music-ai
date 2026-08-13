"""Create a public-domain-free synthetic WAV for the local smoke test."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import struct


def main() -> None:
    output = Path("examples/two_notes.wav")
    output.parent.mkdir(exist_ok=True)
    rate, duration = 22050, 2.0
    frames = []
    for index in range(int(rate * duration)):
        time = index / rate
        frequency = 440 if time < 1 else 523.251
        amplitude = 0.25 * math.sin(2 * math.pi * frequency * time)
        frames.append(struct.pack("<h", int(amplitude * 32767)))
    with wave.open(str(output), "w") as wav:
        wav.setparams((1, 2, rate, len(frames), "NONE", "not compressed"))
        wav.writeframes(b"".join(frames))
    print(output)


if __name__ == "__main__":
    main()
