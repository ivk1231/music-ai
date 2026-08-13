"""Pipeline orchestration; models communicate only through simple adapter methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import Part, ScoreArtifact
from .labeling import StemLabeler
from .midi import write_multitrack_midi
from .notation import write_musicxml


@dataclass
class Pipeline:
    separator: object
    transcriber: object
    labeler: StemLabeler
    instrument_detector: object | None = None

    def run(self, audio_path: Path, output_dir: Path, bpm: int = 120) -> ScoreArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = self.separator.separate(audio_path, output_dir / "stems")
        parts = []
        for stem_name, stem_path in sorted(stems.items()):
            label = self.labeler.label(stem_name)
            label_confidence = None
            label_source = "separator"
            # A separator bucket is not an instrument. In particular, never
            # turn `other` into piano or violin; instrument-aware models emit
            # their own independent parts through the MuScriptor backend.
            # Percussion transcription needs a dedicated model; preserve the part for later.
            notes = [] if label.label == "drums" else self.transcriber.transcribe(stem_path)
            parts.append(Part(id=stem_name, label=label.label, source_audio=str(stem_path),
                              midi_program=label.midi_program, notes=notes,
                              label_confidence=label_confidence, label_source=label_source))
        score = ScoreArtifact("1.0", str(audio_path), parts, {"bpm_assumption": bpm})
        score.write_json(output_dir / "events.json")
        write_multitrack_midi(score, output_dir / "arrangement.mid", bpm=bpm)
        write_musicxml(score, output_dir / "score.musicxml", bpm=bpm)
        return score
