"""MuScriptor adapter for full-mix, instrument-aware transcription.

This backend is deliberately independent from the Demucs + Basic Pitch
pipeline.  It reads the original mix and returns one ``Part`` per instrument,
so a separator bucket such as ``other`` can never be mistaken for an
instrument.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .domain import NoteEvent, Part, ScoreArtifact


class MuscriptorUnavailableError(RuntimeError):
    """Raised when the optional MuScriptor runtime is not installed."""


@dataclass(frozen=True)
class InstrumentSpec:
    part_id: str
    label: str
    midi_program: int


# MuScriptor uses the MT3_FULL_PLUS instrument groups.  Programs below are
# the representative General MIDI programs for those groups.  Piano and
# violin are intentionally explicit invariants rather than inferred aliases.
INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "acoustic_piano": InstrumentSpec("piano", "piano", 0),
    "electric_piano": InstrumentSpec("electric_piano", "electric piano", 4),
    "chromatic_percussion": InstrumentSpec("chromatic_percussion", "chromatic percussion", 8),
    "organ": InstrumentSpec("organ", "organ", 16),
    "acoustic_guitar": InstrumentSpec("acoustic_guitar", "acoustic guitar", 24),
    "clean_electric_guitar": InstrumentSpec("clean_electric_guitar", "clean electric guitar", 27),
    "distorted_electric_guitar": InstrumentSpec("distorted_electric_guitar", "distorted electric guitar", 30),
    "acoustic_bass": InstrumentSpec("acoustic_bass", "acoustic bass", 32),
    "electric_bass": InstrumentSpec("electric_bass", "electric bass", 33),
    "violin": InstrumentSpec("violin", "violin", 40),
    "viola": InstrumentSpec("viola", "viola", 41),
    "cello": InstrumentSpec("cello", "cello", 42),
    "contrabass": InstrumentSpec("contrabass", "contrabass", 43),
    "orchestral_harp": InstrumentSpec("orchestral_harp", "orchestral harp", 46),
    "timpani": InstrumentSpec("timpani", "timpani", 47),
    "string_ensemble": InstrumentSpec("string_ensemble", "string ensemble", 48),
    "synth_strings": InstrumentSpec("synth_strings", "synth strings", 50),
    "voice": InstrumentSpec("voice", "voice", 52),
    "orchestra_hit": InstrumentSpec("orchestra_hit", "orchestra hit", 55),
    "trumpet": InstrumentSpec("trumpet", "trumpet", 56),
    "trombone": InstrumentSpec("trombone", "trombone", 57),
    "tuba": InstrumentSpec("tuba", "tuba", 58),
    "french_horn": InstrumentSpec("french_horn", "french horn", 60),
    "brass_section": InstrumentSpec("brass_section", "brass section", 61),
    "soprano_and_alto_sax": InstrumentSpec("soprano_and_alto_sax", "soprano and alto sax", 64),
    "tenor_sax": InstrumentSpec("tenor_sax", "tenor sax", 66),
    "baritone_sax": InstrumentSpec("baritone_sax", "baritone sax", 67),
    "oboe": InstrumentSpec("oboe", "oboe", 68),
    "english_horn": InstrumentSpec("english_horn", "english horn", 69),
    "bassoon": InstrumentSpec("bassoon", "bassoon", 70),
    "clarinet": InstrumentSpec("clarinet", "clarinet", 71),
    "flutes": InstrumentSpec("flutes", "flutes", 73),
    "synth_lead": InstrumentSpec("synth_lead", "synth lead", 80),
    "synth_pad": InstrumentSpec("synth_pad", "synth pad", 88),
    "drums": InstrumentSpec("drums", "drums", 0),
}


def instrument_spec(name: str) -> InstrumentSpec:
    """Return a safe score identity for a MuScriptor instrument group."""
    if name in INSTRUMENT_SPECS:
        return INSTRUMENT_SPECS[name]
    if name.startswith("program_"):
        program = int(name.removeprefix("program_"))
        return InstrumentSpec(name, name.replace("_", " "), program)
    # Preserve an unknown model label without ever converting it to piano,
    # violin, or the separator-specific name `other`.
    safe_id = f"unassigned_{name}".replace(" ", "_").lower()
    return InstrumentSpec(safe_id, name.replace("_", " "), 0)


class MuscriptorTranscriber:
    """Full-mix MuScriptor backend tuned for Apple Silicon laptops."""

    ALLOWED_MODELS = {"small", "medium"}

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        dtype: str | None = None,
        weights_path: str | Path | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if model_size not in self.ALLOWED_MODELS:
            raise ValueError("MuScriptor model must be 'small' or 'medium'.")
        if device not in {"auto", "mps", "cpu"}:
            raise ValueError("MuScriptor device must be auto, mps, or cpu.")
        self.model_size = model_size
        self.device = device
        self.dtype = dtype
        self.weights_path = weights_path
        self._model_loader = model_loader
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        loader = self._model_loader
        if loader is None:
            try:
                from muscriptor import TranscriptionModel
            except ImportError as exc:
                raise MuscriptorUnavailableError(
                    "MuScriptor is optional. Install it with `pip install -e '.[muscriptor]'`."
                ) from exc
            loader = TranscriptionModel.load_model

        device = None if self.device == "auto" else self.device
        try:
            self._model = loader(
                self.weights_path or self.model_size, device=device, dtype=self.dtype
            )
        except Exception as exc:
            message = str(exc).lower()
            if "gated" in message or "401" in message or "403" in message:
                raise MuscriptorUnavailableError(
                    "MuScriptor weights are gated. Accept the model license on Hugging Face "
                    "and run `hf auth login`, then retry."
                ) from exc
            raise
        return self._model

    def transcribe_parts(
        self,
        audio_path: Path,
        instruments: list[str] | None = None,
    ) -> list[Part]:
        model = self._load_model()
        stream = model.transcribe(str(audio_path), instruments=instruments)
        return self._parts_from_events(stream, audio_path)

    @staticmethod
    def _parts_from_events(events: Iterable[Any], audio_path: Path) -> list[Part]:
        notes_by_instrument: dict[str, list[NoteEvent]] = defaultdict(list)
        for event in events:
            # Progress events have neither of these fields and are ignored.
            if not hasattr(event, "end_time") or not hasattr(event, "start_event"):
                continue
            start = event.start_event
            notes_by_instrument[start.instrument].append(NoteEvent(
                pitch=int(start.pitch),
                start_seconds=round(float(start.start_time), 6),
                end_seconds=round(float(event.end_time), 6),
                velocity=100,
            ))

        parts: list[Part] = []
        for instrument_name, notes in sorted(notes_by_instrument.items()):
            spec = instrument_spec(instrument_name)
            parts.append(Part(
                id=spec.part_id,
                label=spec.label,
                source_audio=str(audio_path),
                midi_program=spec.midi_program,
                notes=sorted(notes, key=lambda note: (note.start_seconds, note.pitch)),
                label_source=f"muscriptor-{instrument_name}",
            ))

        MuscriptorTranscriber._validate_invariants(parts)
        return parts

    @staticmethod
    def _validate_invariants(parts: list[Part]) -> None:
        ids = [part.id for part in parts]
        if len(ids) != len(set(ids)):
            raise ValueError("MuScriptor produced duplicate instrument part identifiers.")
        for part in parts:
            if part.id == "piano" and (part.label != "piano" or part.midi_program != 0):
                raise ValueError("Piano invariant violated: piano must use id/label piano and Program 0.")
            if part.id == "violin" and (part.label != "violin" or part.midi_program != 40):
                raise ValueError("Violin invariant violated: violin must use id/label violin and Program 40.")
            if part.id == "other":
                raise ValueError("Instrument-aware output may not use the separator bucket `other`.")

    def run(
        self,
        audio_path: Path,
        output_dir: Path,
        bpm: int = 120,
        instruments: list[str] | None = None,
        detect_timing: bool = False,
        meter_numerator: int | None = None,
        meter_denominator: int | None = None,
        beat_unit_quarters: float | None = None,
        beat_checkpoint: str = "final0",
    ) -> ScoreArtifact:
        from .midi import write_multitrack_midi
        from .notation import write_musicxml

        # A model experiment must never replace an accepted transcription.
        # Build all artifacts privately and publish only after every exporter
        # succeeds. Reusing a destination is always an explicit error.
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        try:
            parts = self.transcribe_parts(audio_path, instruments=instruments)
            timing_map = None
            quantized_parts = []
            if detect_timing:
                from .beat_tracking import BeatThisTracker
                from .quantization import quantize_parts

                timing_map = BeatThisTracker(checkpoint=beat_checkpoint).track(
                    audio_path,
                    numerator=meter_numerator,
                    denominator=meter_denominator,
                    beat_unit_quarters=beat_unit_quarters,
                )
                quantized_parts = quantize_parts(parts, timing_map)
            score = ScoreArtifact("2.0" if timing_map else "1.0", str(audio_path), parts, {
                "backend": "muscriptor",
                "model": self.model_size,
                "device": self.device,
                "instrument_conditioning": instruments,
                "bpm_assumption": None if timing_map else bpm,
                "timing_backend": timing_map.source if timing_map else "fixed-bpm",
                "experimental": True,
            }, timing_map, quantized_parts)
            score.write_json(staging / "events.json")
            write_multitrack_midi(score, staging / "arrangement.mid", bpm=bpm)
            write_musicxml(score, staging / "score.musicxml", bpm=bpm)
            staging.rename(output_dir)
            return score
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
