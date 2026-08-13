"""Readable, editable sheet-music export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from .domain import QuantizedPart, ScoreArtifact
from .timing import tempo_changes


def write_musicxml(score: ScoreArtifact, path: Path, bpm: int = 120, grid: float = 0.25) -> None:
    """Write editable notation, preferring detected musical time when present."""
    from music21 import chord, instrument, layout, meter, note, stream, tempo

    if score.timing_map is not None and any(
        anchor.position_quarters < -1e-9 for anchor in score.timing_map.anchors
    ):
        raise ValueError(
            "Detected pickup measures are not exported yet; set the first downbeat "
            "manually before creating MIDI or MusicXML."
        )

    document = stream.Score(id="music-ai-score")
    document.metadata = None
    quantized_by_part: dict[str, QuantizedPart] = {
        part.part_id: part for part in score.quantized_parts
    }
    emitted_global_tempo = False
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
        quantized = quantized_by_part.get(source_part.id)
        if score.timing_map is not None and quantized is not None:
            is_piano = source_part.id == "piano" or source_part.label.lower() == "piano"
            targets = [part]
            if is_piano:
                right = stream.PartStaff(id=f"{source_part.id}-rh")
                left = stream.PartStaff(id=f"{source_part.id}-lh")
                right.partName = source_part.label.title()
                left.partName = source_part.label.title()
                targets = [right, left]
            signatures = score.timing_map.time_signatures or []
            if not signatures:
                signatures = [type("Signature", (), {"position_quarters": 0.0, "numerator": 4, "denominator": 4})()]
            for signature in signatures:
                for target in targets:
                    target.insert(
                        signature.position_quarters,
                        meter.TimeSignature(f"{signature.numerator}/{signature.denominator}"),
                    )
            if not emitted_global_tempo:
                changes = tempo_changes(score.timing_map, max_error_seconds=0.04)
                for index, change in enumerate(changes):
                    mark = (
                        tempo.MetronomeMark(number=change.qpm)
                        if index == 0
                        else tempo.MetronomeMark(number=None, numberSounding=change.qpm)
                    )
                    targets[0].insert(change.position_quarters, mark)
                emitted_global_tempo = True

            grouped: dict[tuple[int, int, float, float], list[int]] = defaultdict(list)
            for event in quantized.notes:
                grouped[(event.staff, event.voice, event.onset_quarters, event.duration_quarters)].append(event.pitch)
            voices: dict[tuple[int, int], stream.Voice] = {}
            for (staff, voice_number, offset, duration), pitches in sorted(grouped.items()):
                voice_stream = voices.setdefault((staff, voice_number), stream.Voice(id=f"{staff}-{voice_number}"))
                unique = sorted(set(pitches))
                element = note.Note(unique[0]) if len(unique) == 1 else chord.Chord(unique)
                element.duration.quarterLength = duration
                element.staffNumber = staff
                voice_stream.insert(offset, element)
            for (staff, _), voice_stream in voices.items():
                targets[min(staff - 1, len(targets) - 1)].insert(0, voice_stream)
            for target in targets:
                target.makeMeasures(inPlace=True)
                target.makeTies(inPlace=True)
                document.append(target)
            if is_piano:
                document.insert(0, layout.StaffGroup(targets, symbol="brace", barTogether=True))
            continue

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
