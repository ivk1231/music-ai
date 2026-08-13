"""Beat/downbeat tracking adapter and conservative meter inference.

The tracker deliberately keeps audio time as the source of truth.  It only
builds anchors for the later timing-warp and notation stages.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
import math
from pathlib import Path
from statistics import median
from typing import Any

from music_ai.domain import BeatAnchor, TimeSignatureChange, TimingMap


class BeatTrackingError(RuntimeError):
    """Raised when a tracker cannot produce a usable pulse sequence."""


class BeatTrackingUnavailableError(BeatTrackingError):
    """Raised when the optional Beat This dependency is not installed."""


Predictor = Callable[[str], tuple[Iterable[float], Iterable[float]]]
ModelLoader = Callable[..., Predictor]


class BeatThisTracker:
    """Direct-Python adapter for Beat This with injectable model loading.

    Beat This returns beat and downbeat times, but not an unambiguous notated
    time signature.  Meter inference here is intentionally conservative and
    can always be replaced by explicit user overrides.
    """

    def __init__(
        self,
        *,
        checkpoint: str = "final0",
        device: str = "cpu",
        predictor: Predictor | None = None,
        model_loader: ModelLoader | None = None,
        fallback_numerator: int = 4,
        fallback_denominator: int = 4,
        downbeat_snap_tolerance_seconds: float = 0.07,
    ) -> None:
        _validate_numerator(fallback_numerator)
        _validate_denominator(fallback_denominator)
        if downbeat_snap_tolerance_seconds <= 0:
            raise ValueError("downbeat snap tolerance must be positive")
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = predictor
        self._model_loader = model_loader
        self.fallback_numerator = fallback_numerator
        self.fallback_denominator = fallback_denominator
        self.downbeat_snap_tolerance_seconds = downbeat_snap_tolerance_seconds

    def track(
        self,
        audio_path: str | Path,
        *,
        numerator: int | None = None,
        denominator: int | None = None,
        beat_unit_quarters: float | None = None,
    ) -> TimingMap:
        """Track an audio file and return normalized musical-time anchors.

        ``beat_unit_quarters`` states how many quarter notes one detected pulse
        represents.  It defaults to 1.0 independently of the time-signature
        denominator because compound meter is ambiguous from audio alone.
        """

        if numerator is not None:
            _validate_numerator(numerator)
        if denominator is not None:
            _validate_denominator(denominator)
        unit = 1.0 if beat_unit_quarters is None else float(beat_unit_quarters)
        if not math.isfinite(unit) or unit <= 0:
            raise ValueError("beat_unit_quarters must be a positive finite number")

        prediction = self._get_predictor()(str(audio_path))
        try:
            raw_beats, raw_downbeats = prediction
        except (TypeError, ValueError) as exc:
            raise BeatTrackingError(
                "Beat This must return a (beats, downbeats) pair"
            ) from exc

        warnings: list[str] = []
        beats, rejected_beats = _clean_times(raw_beats)
        downbeats, rejected_downbeats = _clean_times(raw_downbeats)
        if rejected_beats:
            warnings.append(f"Ignored {rejected_beats} invalid beat time(s).")
        if rejected_downbeats:
            warnings.append(f"Ignored {rejected_downbeats} invalid downbeat time(s).")
        if not beats:
            raise BeatTrackingError("Beat This returned no usable beat times")

        downbeat_indices, ignored_downbeats = _align_downbeats(
            beats, downbeats, self.downbeat_snap_tolerance_seconds
        )
        if ignored_downbeats:
            warnings.append(
                f"Ignored {ignored_downbeats} downbeat prediction(s) that did not "
                "align with a detected beat."
            )

        if numerator is None:
            inferred_numerator, meter_confidence, meter_warning, meter_fell_back = (
                _infer_numerator(downbeat_indices, self.fallback_numerator)
            )
        else:
            inferred_numerator = numerator
            meter_confidence = 1.0
            meter_warning = None
            meter_fell_back = False
        if meter_warning:
            warnings.append(meter_warning)
        chosen_numerator = numerator or inferred_numerator
        chosen_denominator = denominator or self.fallback_denominator
        if beat_unit_quarters is None:
            warnings.append(
                "Assumed each detected pulse is one quarter note; compound and "
                "half/double-tempo interpretations may need correction."
            )

        meter_source = "override" if numerator is not None else "inferred_provisional"
        if numerator is None:
            warnings.append(
                "The inferred meter is provisional and does not verify the written "
                "time signature; review or override it before final notation."
            )
        if numerator is None and meter_fell_back:
            meter_source = "fallback_unverified"

        origin = downbeat_indices[0] if downbeat_indices else 0
        detected_downbeats = set(downbeat_indices)
        measure_length_quarters = chosen_numerator * 4.0 / chosen_denominator
        denominator_beat_quarters = 4.0 / chosen_denominator
        anchors: list[BeatAnchor] = []
        for index, time_seconds in enumerate(beats):
            position_quarters = (index - origin) * unit
            measure_index = math.floor(position_quarters / measure_length_quarters)
            within_measure = position_quarters - measure_index * measure_length_quarters
            anchors.append(
                BeatAnchor(
                    time_seconds=time_seconds,
                    position_quarters=position_quarters,
                    is_downbeat=index in detected_downbeats,
                    measure_index=measure_index,
                    beat_in_measure=int(math.floor(
                        (within_measure + 1e-9) / denominator_beat_quarters
                    )) + 1,
                )
            )

        intervals = [later - earlier for earlier, later in zip(beats, beats[1:])]
        fallback_bpm = None
        if intervals:
            fallback_bpm = 60.0 * unit / median(intervals)

        return TimingMap(
            source=f"beat_this:{self.checkpoint}",
            beat_unit_quarters=unit,
            anchors=anchors,
            time_signatures=[
                TimeSignatureChange(
                    position_quarters=anchors[0].position_quarters,
                    numerator=chosen_numerator,
                    denominator=chosen_denominator,
                    confidence=meter_confidence,
                    source=meter_source,
                )
            ],
            fallback_bpm=fallback_bpm,
            warnings=warnings,
        )

    def _get_predictor(self) -> Predictor:
        if self._predictor is not None:
            return self._predictor
        loader = self._model_loader or _load_beat_this_model
        self._predictor = loader(
            checkpoint_path=self.checkpoint,
            device=self.device,
            dbn=False,
        )
        return self._predictor


def _load_beat_this_model(**kwargs: Any) -> Predictor:
    try:
        from beat_this.inference import File2Beats
    except (ImportError, ModuleNotFoundError) as exc:
        raise BeatTrackingUnavailableError(
            "Beat tracking requires the optional 'beat-this==1.1.0' package."
        ) from exc
    return File2Beats(**kwargs)


def _clean_times(values: Iterable[float]) -> tuple[list[float], int]:
    clean: list[float] = []
    rejected = 0
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise BeatTrackingError("Beat predictions must be iterable") from exc
    for value in iterator:
        try:
            time_seconds = float(value)
        except (TypeError, ValueError):
            rejected += 1
            continue
        if not math.isfinite(time_seconds) or time_seconds < 0:
            rejected += 1
            continue
        clean.append(time_seconds)
    clean.sort()
    unique: list[float] = []
    for value in clean:
        if not unique or value - unique[-1] > 1e-6:
            unique.append(value)
    rejected += len(clean) - len(unique)
    return unique, rejected


def _align_downbeats(
    beats: list[float], downbeats: list[float], tolerance: float
) -> tuple[list[int], int]:
    aligned_indices: set[int] = set()
    ignored = 0
    for downbeat in downbeats:
        nearest_index = min(range(len(beats)), key=lambda index: abs(beats[index] - downbeat))
        if abs(beats[nearest_index] - downbeat) <= tolerance:
            aligned_indices.add(nearest_index)
        else:
            ignored += 1
    return sorted(aligned_indices), ignored


def _infer_numerator(
    downbeat_indices: list[int], fallback: int
) -> tuple[int, float | None, str | None, bool]:
    if len(downbeat_indices) < 2:
        return fallback, None, (
            f"Could not infer meter from fewer than two downbeats; using fallback "
            f"numerator {fallback}."
        ), True
    counts = [
        later - earlier
        for earlier, later in zip(downbeat_indices, downbeat_indices[1:])
        if 2 <= later - earlier <= 12
    ]
    if not counts:
        return fallback, None, (
            f"Downbeat spacing was not a plausible stable meter; using fallback "
            f"numerator {fallback}."
        ), True
    frequencies = Counter(counts)
    top_frequency = max(frequencies.values())
    winners = sorted(value for value, count in frequencies.items() if count == top_frequency)
    confidence = top_frequency / len(counts)
    if len(winners) > 1 or confidence < 0.6:
        return fallback, confidence, (
            f"Downbeat spacing was ambiguous ({counts}); using fallback numerator "
            f"{fallback}."
        ), True
    numerator = winners[0]
    warning = None
    if confidence < 1.0:
        warning = (
            f"Inferred {numerator} beats per measure from {top_frequency}/{len(counts)} "
            "downbeat intervals; ignored inconsistent intervals."
        )
    return numerator, confidence, warning, False


def _validate_numerator(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32:
        raise ValueError("time-signature numerator must be an integer from 1 to 32")


def _validate_denominator(value: int) -> None:
    if isinstance(value, bool) or value not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError("time-signature denominator must be a power of two from 1 to 64")
