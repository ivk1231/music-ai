"""Notation-aware, non-destructive note quantization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import floor
from statistics import median

from .domain import NoteEvent, Part, QuantizedNote, QuantizedPart, TimingMap
from .timing import TimingWarp


_MONOPHONIC_IDS = {
    # Bowed strings are intentionally absent: double stops and divisi are
    # ordinary notation and must not be silently deleted.  Callers may still
    # request ``monophonic=True`` for a known solo line.
    "flutes", "flute", "oboe", "english_horn", "bassoon", "clarinet",
    "soprano_and_alto_sax", "tenor_sax", "baritone_sax",
}


def _candidates(value: float) -> set[float]:
    """Nearby straight-sixteenth and eighth-triplet positions."""
    result: set[float] = set()
    for denominator in (4, 3):
        center = floor(value * denominator)
        result.update((center + shift) / denominator for shift in range(-2, 4))
    return result


def _is_triplet_only(value: float) -> bool:
    return (
        abs(value * 3 - round(value * 3)) < 1e-8
        and abs(value * 4 - round(value * 4)) >= 1e-8
    )


def _snap(value: float, structural_positions: tuple[float, ...] = ()) -> float:
    # Prefer ordinary notation when two candidates are effectively equivalent.
    return min(
        _candidates(value),
        key=lambda candidate: (
            abs(candidate - value)
            + (0.008 if _is_triplet_only(candidate) else 0)
            # Near a detected downbeat, prefer the actual bar boundary over a
            # marginally closer subdivision. This prevents bar-line drift.
            + (
                0.35 * min(abs(candidate - position) for position in structural_positions)
                if structural_positions
                and min(abs(value - position) for position in structural_positions) <= 0.15
                else 0
            ),
            abs(candidate - value),
            candidate,
        ),
    )


def _cluster_onsets(
    items: list[tuple[NoteEvent, float, float]],
    threshold_quarters: float,
    structural_positions: tuple[float, ...],
) -> dict[int, float]:
    ordered = sorted(enumerate(items), key=lambda item: item[1][1])
    clustered: dict[int, float] = {}
    cluster: list[tuple[int, float]] = []
    for index, (_, onset, _) in ordered:
        # Compare with the first member, not the previous member.  Adjacent
        # chaining could otherwise turn a long arpeggio into one giant chord.
        if cluster and onset - cluster[0][1] > threshold_quarters:
            target = _snap(median(value for _, value in cluster), structural_positions)
            clustered.update((member, target) for member, _ in cluster)
            cluster = []
        cluster.append((index, onset))
    if cluster:
        target = _snap(median(value for _, value in cluster), structural_positions)
        clustered.update((member, target) for member, _ in cluster)
    return clustered


def _assign_polyphonic_voices(notes: list[QuantizedNote]) -> list[QuantizedNote]:
    """Greedily color onset groups so sustained material remains independent."""
    by_staff: dict[int, list[QuantizedNote]] = defaultdict(list)
    for note in notes:
        by_staff[note.staff].append(note)
    result: list[QuantizedNote] = []
    for staff, staff_notes in by_staff.items():
        groups: dict[float, list[QuantizedNote]] = defaultdict(list)
        for note in staff_notes:
            groups[note.onset_quarters].append(note)
        voice_ends: list[float] = []
        for onset, chord_notes in sorted(groups.items()):
            # Notes may share a MusicXML voice/chord only when their durations
            # agree. Unequal chord tones require separate voices (or ties),
            # otherwise importers silently adopt one duration for all pitches.
            by_duration: dict[float, list[QuantizedNote]] = defaultdict(list)
            for note in chord_notes:
                by_duration[round(note.duration_quarters, 9)].append(note)
            for duration, duration_group in sorted(by_duration.items(), reverse=True):
                available = next(
                    (index for index, end in enumerate(voice_ends) if end <= onset + 1e-9),
                    None,
                )
                if available is None:
                    available = len(voice_ends)
                    voice_ends.append(onset)
                voice_ends[available] = onset + duration
                result.extend(replace(note, voice=available + 1) for note in duration_group)
    return sorted(result, key=lambda note: (note.onset_quarters, note.staff, note.voice, note.pitch))


def _enforce_monophonic(notes: list[QuantizedNote], minimum_duration: float) -> list[QuantizedNote]:
    """Collapse simultaneous alternatives and remove sequential overlaps."""
    by_onset: dict[float, list[QuantizedNote]] = defaultdict(list)
    for note in notes:
        by_onset[note.onset_quarters].append(note)
    selected = [
        max(group, key=lambda note: (note.duration_quarters, note.velocity, note.pitch))
        for _, group in sorted(by_onset.items())
    ]
    result: list[QuantizedNote] = []
    for current in selected:
        if result:
            previous = result[-1]
            available = current.onset_quarters - previous.onset_quarters
            if previous.duration_quarters > available + 1e-9:
                if available >= minimum_duration:
                    result[-1] = replace(previous, duration_quarters=available)
                else:
                    # Do not erase a real preceding note merely because two
                    # detected attacks quantized too close together. Preserve
                    # it at minimum readable length and move the next attack.
                    preserved_end = previous.onset_quarters + minimum_duration
                    result[-1] = replace(previous, duration_quarters=minimum_duration)
                    current = replace(current, onset_quarters=preserved_end)
        result.append(replace(current, voice=1, staff=1))
    return result


def quantize_part(
    part: Part,
    timing_map: TimingMap,
    *,
    monophonic: bool | None = None,
    minimum_duration: float = 0.25,
    chord_threshold_quarters: float = 0.08,
) -> QuantizedPart:
    """Map raw seconds to editable notation without modifying ``part.notes``."""
    if minimum_duration <= 0:
        raise ValueError("Minimum duration must be positive.")
    if chord_threshold_quarters < 0:
        raise ValueError("Chord clustering threshold cannot be negative.")
    warp = TimingWarp(timing_map)
    converted = [
        (event, warp.seconds_to_quarters(event.start_seconds),
         warp.seconds_to_quarters(event.end_seconds))
        for event in part.notes
        if event.end_seconds > event.start_seconds
    ]
    structural_positions = tuple(
        anchor.position_quarters for anchor in timing_map.anchors if anchor.is_downbeat
    )
    clustered = _cluster_onsets(converted, chord_threshold_quarters, structural_positions)
    notes: list[QuantizedNote] = []
    is_piano = part.id == "piano" or part.label.lower() == "piano"
    for index, (event, raw_onset, raw_end) in enumerate(converted):
        onset = clustered[index]
        raw_duration = max(minimum_duration, raw_end - onset)
        duration = max(minimum_duration, _snap(raw_duration))
        end = onset + duration
        error = max(
            abs(warp.quarters_to_seconds(onset) - event.start_seconds),
            abs(warp.quarters_to_seconds(end) - event.end_seconds),
        )
        notes.append(QuantizedNote(
            pitch=event.pitch,
            onset_quarters=onset,
            duration_quarters=duration,
            staff=1 if not is_piano or event.pitch >= 60 else 2,
            velocity=event.velocity,
            source_start_seconds=event.start_seconds,
            source_end_seconds=event.end_seconds,
            quantization_error_seconds=error,
        ))

    if monophonic is None:
        monophonic = part.id in _MONOPHONIC_IDS or part.label.lower() in _MONOPHONIC_IDS
    finished = (
        _enforce_monophonic(notes, minimum_duration)
        if monophonic else _assign_polyphonic_voices(notes)
    )
    return QuantizedPart(part.id, finished)


def quantize_parts(parts: list[Part], timing_map: TimingMap) -> list[QuantizedPart]:
    return [quantize_part(part, timing_map) for part in parts]
