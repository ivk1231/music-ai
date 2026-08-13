"""Musical-time warping derived from detected beat anchors.

The transcription model's seconds remain authoritative.  This module supplies
the reversible, piecewise-linear mapping needed by notation and MIDI exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .domain import BeatAnchor, TimingMap


@dataclass(frozen=True)
class TempoChange:
    """A MIDI-compatible tempo change in quarter-notes per minute."""

    position_quarters: float
    qpm: float


class TimingWarp:
    """Strict monotonic interpolation between wall-clock and score time."""

    def __init__(self, timing_map: TimingMap):
        self.timing_map = timing_map
        self.anchors = tuple(timing_map.anchors)
        self._validate()

    def _validate(self) -> None:
        if len(self.anchors) < 2:
            raise ValueError("A timing map requires at least two beat anchors.")
        for anchor in self.anchors:
            if not isfinite(anchor.time_seconds) or not isfinite(anchor.position_quarters):
                raise ValueError("Beat anchor coordinates must be finite.")
        for left, right in zip(self.anchors, self.anchors[1:]):
            if right.time_seconds <= left.time_seconds:
                raise ValueError("Beat anchor seconds must be strictly increasing.")
            if right.position_quarters <= left.position_quarters:
                raise ValueError("Beat anchor score positions must be strictly increasing.")

    @staticmethod
    def _interpolate(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
        return y0 + (value - x0) * (y1 - y0) / (x1 - x0)

    @staticmethod
    def _segment(value: float, coordinates: tuple[float, ...]) -> int:
        """Return the bracketing segment, using boundary segments to extrapolate."""
        if value <= coordinates[0]:
            return 0
        if value >= coordinates[-1]:
            return len(coordinates) - 2
        lo, hi = 0, len(coordinates) - 1
        while lo + 1 < hi:
            middle = (lo + hi) // 2
            if coordinates[middle] <= value:
                lo = middle
            else:
                hi = middle
        return lo

    def seconds_to_quarters(self, seconds: float) -> float:
        if not isfinite(seconds):
            raise ValueError("Seconds must be finite.")
        times = tuple(anchor.time_seconds for anchor in self.anchors)
        index = self._segment(seconds, times)
        left, right = self.anchors[index : index + 2]
        return self._interpolate(
            seconds, left.time_seconds, right.time_seconds,
            left.position_quarters, right.position_quarters,
        )

    def quarters_to_seconds(self, quarters: float) -> float:
        if not isfinite(quarters):
            raise ValueError("Quarter-note position must be finite.")
        positions = tuple(anchor.position_quarters for anchor in self.anchors)
        index = self._segment(quarters, positions)
        left, right = self.anchors[index : index + 2]
        return self._interpolate(
            quarters, left.position_quarters, right.position_quarters,
            left.time_seconds, right.time_seconds,
        )


def simplify_anchors(
    anchors: Iterable[BeatAnchor], max_error_seconds: float = 0.02,
) -> list[BeatAnchor]:
    """Reduce a tempo curve while bounding error at every supplied beat.

    This is Ramer-Douglas-Peucker in the ``score position -> seconds`` plane.
    Consequently every omitted beat is reconstructed within
    ``max_error_seconds`` by the retained anchors.
    """
    points = list(anchors)
    if max_error_seconds < 0 or not isfinite(max_error_seconds):
        raise ValueError("Tempo simplification error must be finite and non-negative.")
    # Reuse the strict validator, including the two-anchor minimum.
    TimingWarp(TimingMap("simplification", 1.0, points))

    keep = {0, len(points) - 1}

    def visit(first: int, last: int) -> None:
        if last <= first + 1:
            return
        left, right = points[first], points[last]
        worst_index = -1
        worst_error = -1.0
        for index in range(first + 1, last):
            point = points[index]
            estimate = TimingWarp._interpolate(
                point.position_quarters,
                left.position_quarters,
                right.position_quarters,
                left.time_seconds,
                right.time_seconds,
            )
            error = abs(estimate - point.time_seconds)
            if error > worst_error:
                worst_error, worst_index = error, index
        if worst_error > max_error_seconds:
            keep.add(worst_index)
            visit(first, worst_index)
            visit(worst_index, last)

    visit(0, len(points) - 1)
    return [points[index] for index in sorted(keep)]


def tempo_changes(
    timing_map: TimingMap, max_error_seconds: float = 0.02,
) -> list[TempoChange]:
    """Return a simplified step-tempo curve suitable for MIDI emission."""
    anchors = simplify_anchors(timing_map.anchors, max_error_seconds)
    changes: list[TempoChange] = []
    for left, right in zip(anchors, anchors[1:]):
        qpm = 60.0 * (right.position_quarters - left.position_quarters) / (
            right.time_seconds - left.time_seconds
        )
        changes.append(TempoChange(left.position_quarters, qpm))
    return changes

