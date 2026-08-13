from dataclasses import dataclass
from pathlib import Path

import mido
import pytest

from music_ai.muscriptor_transcription import MuscriptorTranscriber, instrument_spec


@dataclass
class Start:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class End:
    end_time: float
    start_event: Start


def test_piano_and_violin_are_independent_invariant_parts() -> None:
    events = [
        End(1.0, Start(60, 0.0, 1, "acoustic_piano")),
        End(1.5, Start(69, 0.5, 2, "violin")),
        End(2.0, Start(72, 1.0, 3, "string_ensemble")),
    ]
    parts = MuscriptorTranscriber._parts_from_events(events, Path("mix.wav"))
    by_id = {part.id: part for part in parts}

    assert by_id["piano"].label == "piano"
    assert by_id["piano"].midi_program == 0
    assert by_id["violin"].label == "violin"
    assert by_id["violin"].midi_program == 40
    assert by_id["string_ensemble"].midi_program == 48
    assert "other" not in by_id


def test_unknown_instrument_never_becomes_piano_violin_or_other() -> None:
    spec = instrument_spec("new_model_group")
    assert spec.part_id == "unassigned_new_model_group"
    assert spec.part_id not in {"piano", "violin", "other"}


def test_small_is_the_default_model() -> None:
    transcriber = MuscriptorTranscriber(model_loader=lambda *args, **kwargs: object())
    assert transcriber.model_size == "small"


class FakeModel:
    def transcribe(self, audio: str, instruments: list[str] | None = None):
        assert instruments == ["acoustic_piano", "violin"]
        return [
            End(1.0, Start(60, 0.0, 1, "acoustic_piano")),
            End(1.5, Start(69, 0.5, 2, "violin")),
        ]


def test_run_publishes_json_midi_musicxml_with_separate_channels(tmp_path: Path) -> None:
    transcriber = MuscriptorTranscriber(model_loader=lambda *args, **kwargs: FakeModel())
    destination = tmp_path / "muscriptor" / "run"

    score = transcriber.run(
        tmp_path / "mix.wav",
        destination,
        instruments=["acoustic_piano", "violin"],
    )

    assert (destination / "events.json").is_file()
    assert (destination / "arrangement.mid").is_file()
    assert (destination / "score.musicxml").is_file()
    assert score.metadata["instrument_conditioning"] == ["acoustic_piano", "violin"]
    midi = mido.MidiFile(destination / "arrangement.mid")
    channels = [
        next(message.channel for message in track if message.type == "note_on")
        for track in midi.tracks[1:]
    ]
    assert channels == [0, 1]


def test_run_never_overwrites_accepted_output(tmp_path: Path) -> None:
    destination = tmp_path / "accepted"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("accepted", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MuscriptorTranscriber(model_loader=lambda *args, **kwargs: FakeModel()).run(
            tmp_path / "mix.wav", destination
        )

    assert marker.read_text(encoding="utf-8") == "accepted"


def test_failed_inference_does_not_publish_partial_output(tmp_path: Path) -> None:
    class BrokenModel:
        def transcribe(self, audio: str, instruments: list[str] | None = None):
            raise RuntimeError("model failed")

    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="model failed"):
        MuscriptorTranscriber(model_loader=lambda *args, **kwargs: BrokenModel()).run(
            tmp_path / "mix.wav", destination
        )
    assert not destination.exists()
