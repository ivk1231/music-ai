"""Local human-review interface for Music AI transcription drafts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from music_ai.domain import NoteEvent, Part, ScoreArtifact
from music_ai.midi import write_multitrack_midi


GM_PROGRAMS = {
    "Acoustic Grand Piano": 0,
    "Electric Piano": 4,
    "Acoustic Guitar": 24,
    "Electric Guitar": 27,
    "Acoustic Bass": 32,
    "Electric Bass": 33,
    "Violin": 40,
    "Cello": 42,
    "Choir Aahs": 52,
    "Voice Oohs": 53,
    "Synth Pad": 88,
    "Drum Kit (channel 10)": 0,
}


def input_directory() -> Path:
    if len(sys.argv) < 2:
        st.error("Open this app with `music-ai review outputs/your-run`.")
        st.stop()
    folder = Path(sys.argv[1]).expanduser().resolve()
    if not (folder / "events.json").is_file():
        st.error(f"No events.json found in {folder}")
        st.stop()
    return folder


@st.cache_data(show_spinner=False)
def load_data(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def note_rows(part: dict) -> list[dict]:
    return [
        {
            "pitch": note["pitch"],
            "start_seconds": note["start_seconds"],
            "end_seconds": note["end_seconds"],
            "velocity": note["velocity"],
        }
        for note in part["notes"]
    ]


def quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 6)


def build_score(data: dict, editors: dict, bpm: int, grid: str) -> ScoreArtifact:
    divisions = {"Off": None, "1/4": 1, "1/8": 2, "1/16": 4, "1/32": 8}
    division = divisions[grid]
    step = (60 / bpm) / division if division else None
    parts: list[Part] = []
    for original in data["parts"]:
        state = editors[original["id"]]
        if not state["include"]:
            continue
        notes = []
        for row in state["notes"]:
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            if step:
                start = quantize(start, step)
                end = max(start + step, quantize(end, step))
            notes.append(NoteEvent(
                pitch=max(0, min(127, int(row["pitch"]) + state["transpose"])),
                start_seconds=start,
                end_seconds=end,
                velocity=max(1, min(127, int(row["velocity"]))),
            ))
        parts.append(Part(original["id"], state["label"], original["source_audio"], state["program"], notes))
    metadata = {**data.get("metadata", {}), "reviewed": True, "review_bpm": bpm, "quantization": grid}
    return ScoreArtifact(data["schema_version"], data["source_audio"], parts, metadata)


st.set_page_config(page_title="Music AI Review", layout="wide")
folder = input_directory()
data = load_data(str(folder / "events.json"))
st.title("Music AI Review")
st.caption("Keep the useful parts, remove false positives, and export a corrected MIDI draft.")

with st.sidebar:
    st.header("Export settings")
    bpm = st.number_input("Tempo for MIDI export", min_value=20, max_value=300,
                          value=int(data.get("metadata", {}).get("bpm_assumption", 120)))
    grid = st.selectbox("Quantize notes", ["Off", "1/4", "1/8", "1/16", "1/32"], index=0)
    st.caption("Quantization is applied only to the exported review copy.")

editors: dict = {}
for part in data["parts"]:
    part_id = part["id"]
    with st.expander(f"{part['label'].title()} — {len(part['notes'])} notes", expanded=True):
        left, right = st.columns([1, 2])
        with left:
            audio_path = Path(part["source_audio"])
            if audio_path.is_file():
                st.audio(str(audio_path), format="audio/wav")
            else:
                st.warning("Stem audio file is unavailable.")
            include = st.checkbox("Include this part", value=True, key=f"include-{part_id}")
            label = st.text_input("Part name", value=part["label"], key=f"label-{part_id}")
            choices = list(GM_PROGRAMS)
            current = next((name for name, program in GM_PROGRAMS.items() if program == part["midi_program"]), choices[0])
            preset = st.selectbox("Playback instrument", choices, index=choices.index(current), key=f"program-{part_id}")
            transpose = st.number_input("Transpose semitones", min_value=-24, max_value=24, value=0, key=f"transpose-{part_id}")
        with right:
            edited = st.data_editor(
                note_rows(part),
                num_rows="dynamic",
                hide_index=True,
                key=f"notes-{part_id}",
                column_config={
                    "pitch": st.column_config.NumberColumn("MIDI pitch", min_value=0, max_value=127, step=1),
                    "start_seconds": st.column_config.NumberColumn("Start (s)", min_value=0.0, step=0.01),
                    "end_seconds": st.column_config.NumberColumn("End (s)", min_value=0.0, step=0.01),
                    "velocity": st.column_config.NumberColumn("Velocity", min_value=1, max_value=127, step=1),
                },
            )
        editors[part_id] = {
            "include": include,
            "label": label.strip() or part["label"],
            "program": GM_PROGRAMS[preset],
            "transpose": int(transpose),
            "notes": edited.to_dict("records"),
        }

st.divider()
if st.button("Export corrected MIDI + JSON", type="primary"):
    score = build_score(data, editors, int(bpm), grid)
    destination = folder / "reviewed"
    destination.mkdir(exist_ok=True)
    score.write_json(destination / "events.json")
    write_multitrack_midi(score, destination / "arrangement.mid", bpm=int(bpm))
    st.success(f"Exported {len(score.parts)} parts to {destination}")
    st.download_button("Download corrected MIDI", (destination / "arrangement.mid").read_bytes(),
                       file_name="music-ai-corrected.mid", mime="audio/midi")
