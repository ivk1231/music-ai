"""Part labeling policy. This is intentionally swappable for a learned classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartLabel:
    label: str
    midi_program: int


class StemLabeler:
    """Maps separator stems to a human-editable label and General MIDI program."""

    DEFAULTS = {
        "bass": PartLabel("bass", 33),
        "drums": PartLabel("drums", 0),
        "guitar": PartLabel("guitar", 25),
        "piano": PartLabel("piano", 0),
        "vocals": PartLabel("vocals", 52),
        "other": PartLabel("other", 0),
        "full_mix": PartLabel("unclassified mix", 0),
    }

    def label(self, stem_name: str) -> PartLabel:
        return self.DEFAULTS.get(stem_name.lower(), PartLabel(stem_name.replace("_", " "), 0))
