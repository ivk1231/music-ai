"""Audio-to-note adapters. Keep model specifics out of the orchestration layer."""

from __future__ import annotations

from pathlib import Path

from .domain import NoteEvent


class BasicPitchTranscriber:
    """Spotify Basic Pitch adapter; works best on tonal, isolated stems."""

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise RuntimeError("Basic Pitch is not installed. Run `pip install -e .`.") from exc

        _, midi_data, _ = predict(str(audio_path), ICASSP_2022_MODEL_PATH)
        events: list[NoteEvent] = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                events.append(NoteEvent(
                    pitch=int(note.pitch), start_seconds=round(float(note.start), 6),
                    end_seconds=round(float(note.end), 6), velocity=int(note.velocity),
                ))
        return sorted(events, key=lambda note: (note.start_seconds, note.pitch))
