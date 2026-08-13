"""Automatic instrument recognition for ambiguous separator stems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstrumentPrediction:
    label: str
    midi_program: int
    confidence: float


class AudioSetInstrumentDetector:
    """AudioSet-trained AST adapter, used only for ambiguous `other` stems."""

    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    TARGETS = {
        "violin": (40, ("violin", "fiddle")),
        "viola": (41, ("viola",)),
        "cello": (42, ("cello",)),
        "string ensemble": (48, ("string section", "bowed string")),
        "piano": (0, ("piano",)),
        "guitar": (24, ("guitar",)),
        "organ": (19, ("organ",)),
        "flute": (73, ("flute",)),
        "saxophone": (65, ("saxophone",)),
        "trumpet": (56, ("trumpet",)),
        "synth pad": (88, ("synthesizer",)),
    }

    def __init__(self, device: str = "mps", minimum_confidence: float = 0.02,
                 minimum_margin_ratio: float = 1.15) -> None:
        self.device = device
        self.minimum_confidence = minimum_confidence
        self.minimum_margin_ratio = minimum_margin_ratio
        self._extractor = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
            self._model.to(self.device)
            self._model.eval()
        return self._extractor, self._model

    def rank(self, audio_path: Path) -> list[InstrumentPrediction]:
        import librosa
        import numpy as np
        import torch

        extractor, model = self._load_model()
        audio, _ = librosa.load(str(audio_path), sr=16_000, mono=True)
        window = 10 * 16_000
        if len(audio) <= window:
            clips = [audio]
        else:
            starts = np.linspace(0, len(audio) - window, num=min(5, max(2, len(audio) // window)), dtype=int)
            clips = [audio[start:start + window] for start in starts]

        probabilities = []
        with torch.no_grad():
            for clip in clips:
                inputs = extractor(clip, sampling_rate=16_000, return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                probabilities.append(torch.sigmoid(model(**inputs).logits).cpu().numpy()[0])
        averaged = np.mean(probabilities, axis=0)
        id_to_label = model.config.id2label

        candidates: list[InstrumentPrediction] = []
        for target, (program, needles) in self.TARGETS.items():
            matching = [float(averaged[index]) for index, label in id_to_label.items()
                        if any(needle in label.lower() for needle in needles)]
            if matching:
                candidates.append(InstrumentPrediction(target, program, max(matching)))
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)

    def predict(self, audio_path: Path) -> InstrumentPrediction | None:
        candidates = self.rank(audio_path)
        if not candidates:
            return None
        best = candidates[0]
        if best.confidence < self.minimum_confidence:
            return None
        if len(candidates) > 1 and best.confidence < candidates[1].confidence * self.minimum_margin_ratio:
            return None
        return best
