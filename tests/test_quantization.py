from copy import deepcopy

import pytest

from music_ai.domain import BeatAnchor, NoteEvent, Part, TimingMap
from music_ai.quantization import quantize_part


def timing() -> TimingMap:
    # Expressive second beat: mapping must use anchors rather than a fixed BPM.
    return TimingMap("test", 1.0, [BeatAnchor(0.0, 0.0), BeatAnchor(0.5, 1.0), BeatAnchor(1.5, 2.0)])


def part(part_id: str, notes: list[NoteEvent]) -> Part:
    return Part(part_id, part_id, "test.wav", 0, notes)


def test_quantization_uses_timing_map_and_preserves_raw_notes() -> None:
    source = part("piano", [NoteEvent(60, 1.0, 1.5)])
    before = deepcopy(source.notes)
    result = quantize_part(source, timing())
    assert result.notes[0].onset_quarters == pytest.approx(1.5)
    assert source.notes == before
    assert result.notes[0].source_start_seconds == 1.0


def test_triplet_candidates_are_available() -> None:
    source = part("flutes", [NoteEvent(72, 1 / 6, 1 / 3)])
    result = quantize_part(source, timing(), monophonic=True)
    assert result.notes[0].onset_quarters == pytest.approx(1 / 3)
    assert result.notes[0].duration_quarters == pytest.approx(1 / 3)


def test_near_simultaneous_notes_cluster_as_a_chord() -> None:
    source = part("piano", [NoteEvent(60, 0.50, 1.0), NoteEvent(64, 0.52, 1.0)])
    result = quantize_part(source, timing())
    assert {note.onset_quarters for note in result.notes} == {1.0}
    assert {note.voice for note in result.notes} == {1}


def test_chord_clustering_span_is_bounded_instead_of_chained() -> None:
    source = part("piano", [
        NoteEvent(60, 0.000, 0.5),
        NoteEvent(64, 0.035, 0.5),  # 0.07 quarters from C
        NoteEvent(67, 0.070, 0.5),  # close to E, but 0.14 from C
    ])
    result = quantize_part(source, timing(), chord_threshold_quarters=0.08)
    assert len({note.onset_quarters for note in result.notes}) == 2


def test_sustained_polyphony_gets_separate_voices_and_piano_staves() -> None:
    source = part("piano", [NoteEvent(48, 0, 1.5), NoteEvent(72, 0.5, 1.0)])
    result = quantize_part(source, timing())
    by_pitch = {note.pitch: note for note in result.notes}
    assert by_pitch[48].duration_quarters == 2.0
    assert by_pitch[48].staff == 2
    assert by_pitch[72].staff == 1


def test_polyphony_on_same_staff_uses_independent_voices() -> None:
    source = part("string_ensemble", [NoteEvent(60, 0, 1.5), NoteEvent(67, 0.5, 1.0)])
    result = quantize_part(source, timing(), monophonic=False)
    assert {note.voice for note in result.notes} == {1, 2}


def test_same_onset_unequal_duration_chord_tones_get_valid_separate_voices() -> None:
    source = part("piano", [NoteEvent(60, 0, 1.5), NoteEvent(64, 0, 0.5)])
    result = quantize_part(source, timing())
    by_pitch = {note.pitch: note for note in result.notes}
    assert by_pitch[60].onset_quarters == by_pitch[64].onset_quarters
    assert by_pitch[60].duration_quarters != by_pitch[64].duration_quarters
    assert by_pitch[60].voice != by_pitch[64].voice


def test_violin_double_stop_is_preserved_by_default() -> None:
    source = part("violin", [NoteEvent(67, 0, 1.0), NoteEvent(74, 0, 1.0)])
    result = quantize_part(source, timing())
    assert {note.pitch for note in result.notes} == {67, 74}
    assert len(result.notes) == 2


def test_monophonic_policy_removes_overlaps_and_enforces_minimum_duration() -> None:
    source = part("violin", [NoteEvent(60, 0, 1.0), NoteEvent(62, 0.5, 0.55)])
    result = quantize_part(source, timing(), monophonic=True, minimum_duration=0.25)
    assert all(note.duration_quarters >= 0.25 for note in result.notes)
    assert all(
        left.onset_quarters + left.duration_quarters <= right.onset_quarters
        for left, right in zip(result.notes, result.notes[1:])
    )


def test_monophonic_tiny_overlap_preserves_prior_note() -> None:
    source = part("flutes", [NoteEvent(60, 0, 0.5), NoteEvent(62, 0.12, 0.4)])
    result = quantize_part(source, timing(), minimum_duration=0.5)
    assert [note.pitch for note in result.notes] == [60, 62]
    assert result.notes[0].duration_quarters == 0.5
    assert result.notes[1].onset_quarters >= 0.5


def test_near_downbeat_prefers_bar_boundary_over_marginally_closer_grid() -> None:
    marked = TimingMap("test", 1.0, [
        BeatAnchor(0.0, 0.0, is_downbeat=True),
        BeatAnchor(0.5, 1.0), BeatAnchor(1.0, 2.0), BeatAnchor(1.5, 3.0),
        BeatAnchor(2.0, 4.0, is_downbeat=True),
    ])
    # 3.87 is slightly closer to straight 3.75 than to the known downbeat 4.
    source = part("piano", [NoteEvent(60, 1.935, 2.2)])
    result = quantize_part(source, marked)
    assert result.notes[0].onset_quarters == 4.0
