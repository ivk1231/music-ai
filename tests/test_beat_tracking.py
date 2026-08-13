import builtins
from pathlib import Path

import pytest

from music_ai.beat_tracking import (
    BeatThisTracker,
    BeatTrackingError,
    BeatTrackingUnavailableError,
    _load_beat_this_model,
)


def test_adapter_loads_final0_on_cpu_without_dbn() -> None:
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return lambda path: ([0.0, 0.5, 1.0], [0.0])

    tracker = BeatThisTracker(model_loader=loader)
    tracker.track("music.wav", numerator=3)

    assert calls == [{"checkpoint_path": "final0", "device": "cpu", "dbn": False}]


def test_normalizes_times_aligns_downbeats_and_infers_stable_meter() -> None:
    def predict(path: str):
        assert path == "music.wav"
        beats = [2.0, 0.0, 1.0, 0.5, 1.5, 2.5, 3.0, 3.5, 4.0, float("nan"), 1.0]
        downbeats = [0.02, 2.01, 4.02]
        return beats, downbeats

    timing = BeatThisTracker(predictor=predict).track("music.wav")

    assert [anchor.time_seconds for anchor in timing.anchors] == [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
    ]
    assert [anchor.beat_in_measure for anchor in timing.anchors] == [1, 2, 3, 4, 1, 2, 3, 4, 1]
    assert [anchor.is_downbeat for anchor in timing.anchors] == [
        True, False, False, False, True, False, False, False, True
    ]
    assert timing.time_signatures[0].numerator == 4
    assert timing.time_signatures[0].confidence == 1.0
    assert timing.fallback_bpm == pytest.approx(120.0)
    assert any("invalid beat" in warning for warning in timing.warnings)


def test_unmatched_downbeat_is_ignored_instead_of_inventing_a_pulse() -> None:
    timing = BeatThisTracker(
        predictor=lambda path: ([0.0, 0.5, 1.0], [0.0, 0.75])
    ).track("music.wav", numerator=4)

    assert [anchor.time_seconds for anchor in timing.anchors] == [0.0, 0.5, 1.0]
    assert [anchor.is_downbeat for anchor in timing.anchors] == [True, False, False]
    assert any("Ignored 1 downbeat" in warning for warning in timing.warnings)


def test_ambiguous_meter_uses_fallback_and_warns() -> None:
    beats = [index * 0.5 for index in range(11)]
    downbeats = [beats[index] for index in [0, 3, 7, 10]]  # intervals 3, 4, 3
    timing = BeatThisTracker(predictor=lambda path: (beats, downbeats)).track("music.wav")

    # 3/4 has a stable majority despite a single outlier.
    assert timing.time_signatures[0].numerator == 3
    assert timing.time_signatures[0].confidence == pytest.approx(2 / 3)
    assert any("ignored inconsistent" in warning for warning in timing.warnings)

    tied = BeatThisTracker(
        predictor=lambda path: (beats, [beats[index] for index in [0, 3, 7]])
    ).track("music.wav")
    assert tied.time_signatures[0].numerator == 4
    assert tied.time_signatures[0].source == "fallback_unverified"
    assert any("ambiguous" in warning for warning in tied.warnings)


def test_even_high_confidence_inferred_meter_is_provisional() -> None:
    beats = [index * 0.5 for index in range(9)]
    timing = BeatThisTracker(
        predictor=lambda path: (beats, [beats[0], beats[4], beats[8]])
    ).track("music.wav")

    assert timing.time_signatures[0].confidence == 1.0
    assert timing.time_signatures[0].source == "inferred_provisional"
    assert any("provisional" in warning for warning in timing.warnings)


def test_pickup_positions_first_downbeat_at_zero_and_marks_pickup_measure() -> None:
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    timing = BeatThisTracker(
        predictor=lambda path: (beats, [1.0, 3.0])
    ).track("pickup.wav", numerator=4, denominator=4)

    assert [anchor.position_quarters for anchor in timing.anchors] == [
        -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0
    ]
    assert timing.anchors[0].measure_index == -1
    assert timing.anchors[0].beat_in_measure == 3
    assert timing.anchors[2].is_downbeat
    assert timing.anchors[2].measure_index == 0
    assert timing.anchors[2].beat_in_measure == 1
    assert timing.time_signatures[0].position_quarters == -2.0


def test_explicit_meter_and_beat_unit_overrides_win() -> None:
    timing = BeatThisTracker(
        predictor=lambda path: ([0.0, 0.75, 1.5, 2.25], [])
    ).track("music.wav", numerator=6, denominator=8, beat_unit_quarters=1.5)

    signature = timing.time_signatures[0]
    assert (signature.numerator, signature.denominator, signature.source) == (6, 8, "override")
    assert timing.beat_unit_quarters == 1.5
    assert [anchor.position_quarters for anchor in timing.anchors] == [0.0, 1.5, 3.0, 4.5]
    assert timing.fallback_bpm == pytest.approx(120.0)
    assert not any("Assumed each detected pulse" in warning for warning in timing.warnings)
    assert not any("Could not infer meter" in warning for warning in timing.warnings)


def test_known_bach_six_eight_uses_dotted_quarter_pulses() -> None:
    beats = [0.0, 0.75, 1.5, 2.25, 3.0]
    timing = BeatThisTracker(
        predictor=lambda path: (beats, [0.0, 1.5, 3.0])
    ).track("bach.wav", numerator=6, denominator=8, beat_unit_quarters=1.5)

    assert [anchor.position_quarters for anchor in timing.anchors] == [0.0, 1.5, 3.0, 4.5, 6.0]
    assert [anchor.beat_in_measure for anchor in timing.anchors] == [1, 4, 1, 4, 1]
    assert [anchor.measure_index for anchor in timing.anchors] == [0, 0, 1, 1, 2]
    assert timing.time_signatures[0].source == "override"


def test_known_dvorak_four_four_uses_quarter_note_pulses() -> None:
    beats = [0.0, 0.6, 1.2, 1.8, 2.4]
    timing = BeatThisTracker(
        predictor=lambda path: (beats, [0.0, 2.4])
    ).track("dvorak.wav", numerator=4, denominator=4, beat_unit_quarters=1.0)

    assert [anchor.beat_in_measure for anchor in timing.anchors] == [1, 2, 3, 4, 1]
    assert [anchor.position_quarters for anchor in timing.anchors] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert timing.time_signatures[0].source == "override"


def test_no_usable_beats_is_an_error() -> None:
    tracker = BeatThisTracker(predictor=lambda path: ([float("nan"), -1], []))
    with pytest.raises(BeatTrackingError, match="no usable beat"):
        tracker.track(Path("silent.wav"))


def test_missing_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("beat_this"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(BeatTrackingUnavailableError, match="beat-this==1.1.0"):
        _load_beat_this_model(checkpoint_path="final0", device="cpu", dbn=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numerator": 0},
        {"denominator": 3},
        {"beat_unit_quarters": 0},
        {"beat_unit_quarters": float("inf")},
    ],
)
def test_invalid_overrides_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        BeatThisTracker(predictor=lambda path: ([0.0], [])).track("music.wav", **kwargs)
