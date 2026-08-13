from pathlib import Path

import mido

from music_ai.domain import NoteEvent, Part, ScoreArtifact
from music_ai.midi import write_multitrack_midi


def test_multitrack_midi_is_written(tmp_path: Path) -> None:
    score = ScoreArtifact("1.0", "test.wav", [
        Part("piano", "piano", "test.wav", 0, [NoteEvent(69, 0, 1)]),
        Part("bass", "bass", "test.wav", 33, [NoteEvent(45, 0, 1)]),
    ])
    path = tmp_path / "arrangement.mid"
    write_multitrack_midi(score, path)
    loaded = mido.MidiFile(path)
    assert len(loaded.tracks) == 3
    assert any(message.type == "note_on" for message in loaded.tracks[1])
    channels = [next(message.channel for message in track if message.type == "note_on") for track in loaded.tracks[1:]]
    assert channels == [0, 1]
