"""Readable, editable sheet-music export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from .domain import ScoreArtifact


def write_musicxml(score: ScoreArtifact, path: Path, bpm: int = 120, grid: float = 0.25) -> None:
    """Write a quantized 4/4 score; `grid=0.25` means sixteenth notes."""
    from music21 import chord, instrument, meter, note, stream, tempo

    document = stream.Score(id="music-ai-score")
    document.metadata = None
    for source_part in score.parts:
        if not source_part.notes:
            continue
        part = stream.Part(id=source_part.id)
        part.partName = source_part.label.title()
        try:
            part.insert(0, instrument.instrumentFromMidiProgram(source_part.midi_program))
        except Exception:
            generic = instrument.Instrument()
            generic.instrumentName = source_part.label.title()
            generic.midiProgram = source_part.midi_program
            part.insert(0, generic)
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, tempo.MetronomeMark(number=bpm))

        grouped: dict[float, list] = defaultdict(list)
        for event in source_part.notes:
            offset = round((event.start_seconds * bpm / 60) / grid) * grid
            grouped[offset].append(event)
        offsets = sorted(grouped)
        for index, offset in enumerate(offsets):
            events = grouped[offset]
            durations = [max(grid, round(((event.end_seconds - event.start_seconds) * bpm / 60) / grid) * grid)
                         for event in events]
            duration = median(durations)
            if index + 1 < len(offsets):
                duration = min(duration, max(grid, offsets[index + 1] - offset))
            pitches = sorted({event.pitch for event in events})
            element = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            element.duration.quarterLength = duration
            part.insert(offset, element)
        part.makeMeasures(inPlace=True)
        document.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.write("musicxml", fp=str(path))
