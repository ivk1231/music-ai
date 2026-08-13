"""Stable, model-agnostic representation used between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start_seconds: float
    end_seconds: float
    velocity: int = 100
    confidence: float | None = None


@dataclass
class Part:
    id: str
    label: str
    source_audio: str
    midi_program: int
    notes: list[NoteEvent] = field(default_factory=list)
    label_confidence: float | None = None
    label_source: str | None = None


@dataclass
class ScoreArtifact:
    schema_version: str
    source_audio: str
    parts: list[Part]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        import json

        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "ScoreArtifact":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        for raw_part in data["parts"]:
            notes = [NoteEvent(**raw_note) for raw_note in raw_part.get("notes", [])]
            parts.append(Part(
                id=raw_part["id"], label=raw_part["label"], source_audio=raw_part["source_audio"],
                midi_program=raw_part["midi_program"], notes=notes,
                label_confidence=raw_part.get("label_confidence"),
                label_source=raw_part.get("label_source"),
            ))
        return cls(data["schema_version"], data["source_audio"], parts, data.get("metadata", {}))
