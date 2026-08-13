"""MIDI rendering from the stable event schema."""

from __future__ import annotations

from pathlib import Path

import mido

from .domain import ScoreArtifact


def write_multitrack_midi(score: ScoreArtifact, path: Path, ticks_per_beat: int = 480, bpm: int = 120) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(bpm)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Music AI arrangement", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    midi.tracks.append(meta)
    seconds_per_tick = (60 / bpm) / ticks_per_beat
    melodic_channels = iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15])

    for part in score.parts:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=part.label, time=0))
        is_drum = part.label == "drums"
        # Program changes are channel-scoped. Using channel 0 for every track
        # lets some DAWs (notably Logic) collapse all parts to Grand Piano.
        channel = 9 if is_drum else next(melodic_channels)
        if not is_drum:
            track.append(mido.Message("program_change", program=part.midi_program, channel=channel, time=0))
        messages = []
        for note in part.notes:
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
