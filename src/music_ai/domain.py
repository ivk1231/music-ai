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


@dataclass(frozen=True)
class BeatAnchor:
    """One detected pulse expressed in wall-clock and musical time."""

    time_seconds: float
    position_quarters: float
    is_downbeat: bool = False
    measure_index: int | None = None
    beat_in_measure: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class TimeSignatureChange:
    position_quarters: float
    numerator: int
    denominator: int
    confidence: float | None = None
    source: str = "inferred"


@dataclass
class TimingMap:
    """Derived timing data; raw note seconds remain the canonical input."""

    source: str
    beat_unit_quarters: float
    anchors: list[BeatAnchor]
    time_signatures: list[TimeSignatureChange] = field(default_factory=list)
    fallback_bpm: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuantizedNote:
    pitch: int
    onset_quarters: float
    duration_quarters: float
    voice: int = 1
    staff: int = 1
    velocity: int = 100
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    quantization_error_seconds: float | None = None


@dataclass
class QuantizedPart:
    part_id: str
    notes: list[QuantizedNote] = field(default_factory=list)


@dataclass
class ScoreArtifact:
    schema_version: str
    source_audio: str
    parts: list[Part]
    metadata: dict[str, Any] = field(default_factory=dict)
    timing_map: TimingMap | None = None
    quantized_parts: list[QuantizedPart] = field(default_factory=list)

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
        raw_timing = data.get("timing_map")
        timing_map = None
        if raw_timing:
            timing_map = TimingMap(
                source=raw_timing["source"],
                beat_unit_quarters=raw_timing["beat_unit_quarters"],
                anchors=[BeatAnchor(**item) for item in raw_timing.get("anchors", [])],
                time_signatures=[
                    TimeSignatureChange(**item)
                    for item in raw_timing.get("time_signatures", [])
                ],
                fallback_bpm=raw_timing.get("fallback_bpm"),
                warnings=list(raw_timing.get("warnings", [])),
            )
        quantized_parts = [
            QuantizedPart(
                part_id=item["part_id"],
                notes=[QuantizedNote(**note) for note in item.get("notes", [])],
            )
            for item in data.get("quantized_parts", [])
        ]
        return cls(
            data["schema_version"], data["source_audio"], parts,
            data.get("metadata", {}), timing_map, quantized_parts,
        )
