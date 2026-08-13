from pathlib import Path

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


def test_v2_timing_and_quantized_data_roundtrip(tmp_path: Path) -> None:
    score = ScoreArtifact(
        "2.0",
        "source.wav",
        [Part("piano", "piano", "source.wav", 0, [NoteEvent(60, 0.1, 0.9)])],
        {"backend": "test"},
        TimingMap(
            "test",
            1.0,
            [BeatAnchor(0.0, 0.0, True, 0, 0), BeatAnchor(0.5, 1.0, False, 0, 1)],
            [TimeSignatureChange(0.0, 4, 4, 0.8, "inferred")],
        ),
        [QuantizedPart("piano", [QuantizedNote(60, 0.25, 1.5, source_start_seconds=0.1)])],
    )
    path = tmp_path / "events.json"
    score.write_json(path)

    loaded = ScoreArtifact.read_json(path)

    assert loaded == score


def test_v1_json_without_timing_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(
        '{"schema_version":"1.0","source_audio":"old.wav","parts":[],"metadata":{}}\n',
        encoding="utf-8",
    )

    loaded = ScoreArtifact.read_json(path)

    assert loaded.timing_map is None
    assert loaded.quantized_parts == []
