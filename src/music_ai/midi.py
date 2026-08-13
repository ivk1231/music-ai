"""MIDI rendering from the stable event schema."""

from __future__ import annotations

from pathlib import Path

import mido

from .domain import QuantizedPart, ScoreArtifact
from .timing import tempo_changes


def write_multitrack_midi(score: ScoreArtifact, path: Path, ticks_per_beat: int = 480, bpm: int = 120) -> None:
    if score.timing_map is not None and any(
        anchor.position_quarters < -1e-9 for anchor in score.timing_map.anchors
    ):
        raise ValueError(
            "Detected pickup measures are not exported yet; set the first downbeat "
            "manually before creating MIDI or MusicXML."
        )
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Music AI arrangement", time=0))
    meta_events: list[tuple[int, int, mido.MetaMessage]] = []
    if score.timing_map is not None:
        for change in tempo_changes(score.timing_map):
            meta_events.append((
                max(0, round(change.position_quarters * ticks_per_beat)),
                1,
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(change.qpm), time=0),
            ))
        for signature in score.timing_map.time_signatures:
            meta_events.append((
                max(0, round(signature.position_quarters * ticks_per_beat)),
                0,
                mido.MetaMessage(
                    "time_signature", numerator=signature.numerator,
                    denominator=signature.denominator, time=0,
                ),
            ))
    else:
        meta_events.append((0, 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)))
        meta_events.append((0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)))
    previous_meta_tick = 0
    for tick, _, message in sorted(meta_events, key=lambda event: (event[0], event[1])):
        message.time = tick - previous_meta_tick
        meta.append(message)
        previous_meta_tick = tick
    midi.tracks.append(meta)
    seconds_per_tick = (60 / bpm) / ticks_per_beat
    quantized_by_part: dict[str, QuantizedPart] = {
        part.part_id: part for part in score.quantized_parts
    }
    if score.timing_map is not None:
        missing = [part.id for part in score.parts if part.notes and part.id not in quantized_by_part]
        if missing:
            raise ValueError(
                "Tempo-aware MIDI requires quantized notes for every non-empty part; missing "
                + ", ".join(missing)
            )
    melodic_channels = iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15])

    for part in score.parts:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=part.label, time=0))
        is_drum = part.label == "drums"
        # Program changes are channel-scoped. Using channel 0 for every track
        # lets some DAWs (notably Logic) collapse all parts to Grand Piano.
        try:
            channel = 9 if is_drum else next(melodic_channels)
        except StopIteration as exc:
            raise ValueError("Standard MIDI supports at most 15 melodic instrument channels.") from exc
        if not is_drum:
            track.append(mido.Message("program_change", program=part.midi_program, channel=channel, time=0))
        messages = []
        quantized = quantized_by_part.get(part.id)
        render_notes = quantized.notes if quantized is not None else part.notes
        for note in render_notes:
            if quantized is not None:
                start = max(0, round(note.onset_quarters * ticks_per_beat))
                end = max(start + 1, round((note.onset_quarters + note.duration_quarters) * ticks_per_beat))
            else:
                start = max(0, round(note.start_seconds / seconds_per_tick))
                end = max(start + 1, round(note.end_seconds / seconds_per_tick))
            messages.extend([(start, 1, note), (end, 0, note)])
        previous_tick = 0
        for tick, is_on, note in sorted(messages, key=lambda msg: (msg[0], msg[1])):
            track.append(mido.Message("note_on" if is_on else "note_off", note=note.pitch,
                                      velocity=note.velocity if is_on else 0, channel=channel,
                                      time=tick - previous_tick))
            previous_tick = tick
        midi.tracks.append(track)
    midi.save(path)
