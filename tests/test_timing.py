import pytest

from music_ai.domain import BeatAnchor, TimingMap
from music_ai.timing import TimingWarp, simplify_anchors, tempo_changes


def timing(*points: tuple[float, float]) -> TimingMap:
    return TimingMap("test", 1.0, [BeatAnchor(seconds, quarters) for seconds, quarters in points])


def test_piecewise_warp_is_exact_reversible_and_extrapolates() -> None:
    warp = TimingWarp(timing((1.0, 0.0), (1.5, 1.0), (2.5, 2.0)))
    assert warp.seconds_to_quarters(1.25) == pytest.approx(0.5)
    assert warp.seconds_to_quarters(2.0) == pytest.approx(1.5)
    assert warp.seconds_to_quarters(0.5) == pytest.approx(-1.0)
    assert warp.quarters_to_seconds(3.0) == pytest.approx(3.5)
    for seconds in (0.5, 1.0, 1.7, 3.0):
        assert warp.quarters_to_seconds(warp.seconds_to_quarters(seconds)) == pytest.approx(seconds)


@pytest.mark.parametrize("points", [
    ((0.0, 0.0),),
    ((0.0, 0.0), (0.0, 1.0)),
    ((0.0, 0.0), (1.0, 0.0)),
    ((1.0, 0.0), (0.0, 1.0)),
])
def test_invalid_maps_are_rejected(points: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        TimingWarp(timing(*points))


def test_simplification_obeys_wall_clock_error_bound() -> None:
    original = [
        BeatAnchor(0.00, 0), BeatAnchor(0.51, 1), BeatAnchor(1.00, 2),
        BeatAnchor(1.70, 3), BeatAnchor(2.40, 4),
    ]
    simplified = simplify_anchors(original, max_error_seconds=0.025)
    warp = TimingWarp(TimingMap("simple", 1.0, simplified))
    assert len(simplified) < len(original)
    assert max(abs(warp.quarters_to_seconds(p.position_quarters) - p.time_seconds) for p in original) <= 0.025


def test_tempo_changes_follow_retained_anchor_intervals() -> None:
    changes = tempo_changes(timing((0, 0), (0.5, 1), (1.5, 2)), max_error_seconds=0)
    assert [(item.position_quarters, item.qpm) for item in changes] == pytest.approx([(0, 120), (1, 60)])

