from pathlib import Path

import mido

from music_ai.domain import (
    BeatAnchor,
    NoteEvent,
    Part,
    QuantizedNote,
    QuantizedPart,
    ScoreArtifact,
    TimeSignatureChange,
    TimingMap,
)
from music_ai.midi import write_multitrack_midi
from music_ai.notation import write_musicxml


def variable_tempo_score() -> ScoreArtifact:
    part = Part("piano", "piano", "test.wav", 0, [NoteEvent(60, 0.0, 3.2)])
    timing = TimingMap(
        "test",
        1.0,
        [
            BeatAnchor(0.0, 0.0, True, 0, 1),
            BeatAnchor(0.5, 1.0, False, 0, 2),
            BeatAnchor(1.1, 2.0, False, 0, 3),
            BeatAnchor(1.8, 3.0, False, 0, 4),
            BeatAnchor(2.6, 4.0, True, 1, 1),
            BeatAnchor(3.5, 5.0, False, 1, 2),
        ],
        [TimeSignatureChange(0.0, 4, 4, 1.0, "test")],
    )
    quantized = QuantizedPart("piano", [
        QuantizedNote(60, 0.0, 5.0, voice=1, staff=1),
        QuantizedNote(48, 1.0, 1.0, voice=1, staff=2),
    ])
    return ScoreArtifact("2.0", "test.wav", [part], {}, timing, [quantized])


def test_midi_emits_changing_tempo_and_meter(tmp_path: Path) -> None:
    path = tmp_path / "score.mid"
    write_multitrack_midi(variable_tempo_score(), path)

    midi = mido.MidiFile(path)
    meta = midi.tracks[0]
    assert sum(message.type == "set_tempo" for message in meta) > 1
    assert sum(message.type == "time_signature" for message in meta) == 1
    assert any(message.type == "note_on" and message.note == 48 for message in midi.tracks[1])


def test_midi_rejects_mixed_raw_and_quantized_coordinates(tmp_path: Path) -> None:
    score = variable_tempo_score()
    score.quantized_parts = []

    try:
        write_multitrack_midi(score, tmp_path / "unsafe.mid")
    except ValueError as exc:
        assert "quantized notes" in str(exc)
    else:
        raise AssertionError("mixed coordinate systems must be rejected")


def test_musicxml_has_measures_voices_and_tie(tmp_path: Path) -> None:
    path = tmp_path / "score.musicxml"
    write_musicxml(variable_tempo_score(), path)
    xml = path.read_text(encoding="utf-8")

    assert xml.count("<measure ") >= 2
    assert "<voice>" in xml
    assert "<staves>2</staves>" in xml
    assert "<staff>2</staff>" in xml
    assert '<tie type="start"' in xml


def test_pickup_is_rejected_instead_of_corrupted(tmp_path: Path) -> None:
    score = variable_tempo_score()
    score.timing_map.anchors = [
        BeatAnchor(-1.0, -2.0, False, -1, 3),
        BeatAnchor(-0.5, -1.0, False, -1, 4),
        *score.timing_map.anchors,
    ]

    for writer, filename in (
        (write_multitrack_midi, "pickup.mid"),
        (write_musicxml, "pickup.musicxml"),
    ):
        try:
            writer(score, tmp_path / filename)
        except ValueError as exc:
            assert "pickup measures" in str(exc)
        else:
            raise AssertionError("pickup export must fail safely until anacrusis support exists")
