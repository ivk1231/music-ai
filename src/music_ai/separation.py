"""Audio separation adapters. Replace this module to introduce a custom model."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class SeparatorError(RuntimeError):
    pass


class DemucsSeparator:
    """Adapter around Demucs' stable command-line interface."""

    def __init__(self, model: str = "htdemucs", device: str = "mps") -> None:
        self.model = model
        self.device = device

    def separate(self, audio_path: Path, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "demucs", "-d", self.device, "-n", self.model,
                   "-o", str(output_dir), str(audio_path)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise SeparatorError(f"Demucs failed:\n{result.stderr.strip()}")

        stem_dir = output_dir / self.model / audio_path.stem
        stems = {path.stem: path for path in stem_dir.glob("*.wav")}
        if not stems:
            raise SeparatorError(f"Demucs did not produce stems in {stem_dir}")
        return stems


class PassthroughSeparator:
    """One-part separator for a quick end-to-end test or already-isolated audio."""

    def separate(self, audio_path: Path, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / audio_path.name
        if audio_path.resolve() != destination.resolve():
            shutil.copy2(audio_path, destination)
        return {"full_mix": destination}
