# Music AI

Local, modular baseline for turning a music file into editable musical data:

```text
MP3/WAV → instrument notes + beat/downbeat map → quantized voices → JSON + MIDI + MusicXML
```

This is a first vertical slice, not a claim of score-perfect transcription. Its durable output is `events.json`: raw note seconds remain unchanged, while detected musical timing and quantized notation are stored as derived, replaceable data. `arrangement.mid` and `score.musicxml` open in MuseScore, Sibelius, or a DAW.

## Tempo-aware notation

The score path uses Beat This! 1.1.0 (`final0`, CPU, no DBN) to detect beats
and downbeats. A monotonic timing warp maps the original note seconds into
musical quarter-note positions. Notes are then quantized against straight and
triplet grids, with chord grouping, independent voices, and minimum readable
durations. MIDI receives a simplified changing-tempo curve; MusicXML receives
measures, meter, voices, and ties across barlines.

Install the timing dependency with the transcription model:

```bash
python -m pip install -e ".[dev,muscriptor,timing]"
```

Tempo-aware output is the default for `transcribe-score`. Expected-instrument
selection remains available because automatic instrument identity is not yet
reliable:

```bash
music-ai transcribe-score song.mp3 \
  --profile small \
  --instruments acoustic_piano,flutes,violin \
  --output outputs/muscriptor/song
```

Add changing tempo to an existing transcription without rerunning MuScriptor:

```bash
music-ai retime-score outputs/old/events.json song.mp3 \
  --output outputs/old-tempo-aware
```

Beat tracking cannot always distinguish meter notation such as 3/4 versus
6/8, or a quarter-note pulse from a dotted-quarter pulse. Ambiguous analysis
is reported as a warning. Use explicit overrides when the score is known:

```bash
music-ai transcribe-score song.mp3 --meter 6/8 --beat-unit 1.5 \
  --output outputs/muscriptor/song-6-8
```

Use `--fixed-tempo --bpm 120` only for the legacy constant-tempo behavior.

Pickup/anacrusis detection is preserved in the timing JSON but is not yet
exported. The command stops with an explicit message instead of collapsing
negative pickup positions into the first downbeat.

## Instrument-specific score transcription (MuScriptor)

The preferred score-first path now sends the complete mix to MuScriptor. It
creates instrument-labelled note events directly, so Piano and Violin become
separate score/MIDI parts instead of being placed in Demucs's mixed `other`
stem. The older Demucs + Basic Pitch command remains available for audio stems
and as a lightweight fallback.

MuScriptor profiles:

- `small` (103M parameters): default, intended for the M1 Air and other lower-memory Macs.
- `medium` (307M parameters): optional accuracy upgrade for the 16 GB M1 Pro.

Piano is always stored as part ID `piano`, label `piano`, GM Program 0. Violin
is always part ID `violin`, label `violin`, GM Program 40. A part named `other`
is never promoted to either instrument.

The code is MIT licensed, while the model weights are CC BY-NC 4.0. Personal
and family use fits the intended non-commercial use described by the model
publisher. The weights are gated: accept the license for the desired model on
Hugging Face and authenticate once. This project does not request, store, or
download gated credentials for you.

```bash
cd "/Users/immanuelkoshy/Documents/Music AI"
source .venv/bin/activate
python -m pip install -e ".[dev,muscriptor,timing]"

# After you have accepted the model license, authenticate once:
hf auth login

# M1 Air / default: let the model detect all instruments
music-ai transcribe-score song.mp3 \
  --profile small \
  --output outputs/muscriptor/song-small

# When you know the recording contains piano and violin, conditioning improves
# instrument consistency and constrains the result to those two instruments:
music-ai transcribe-score song.mp3 \
  --profile small \
  --instruments acoustic_piano,violin \
  --output outputs/muscriptor/song-piano-violin

# M1 Pro 16 GB quality profile:
music-ai transcribe-score song.mp3 \
  --profile medium \
  --output outputs/muscriptor/song-medium
```

Each run writes `events.json`, `arrangement.mid`, and `score.musicxml` into a
new directory under `outputs/muscriptor/`. Existing output directories are
never overwritten, and files are published only after every export succeeds.
Open `score.musicxml` in MuseScore or Sibelius for staff-based correction.

MuScriptor creates separate symbolic parts, not isolated instrument WAV files.
Demucs remains useful for listening stems, but its `other` stem may contain
both piano and violin and must not be treated as an instrument identity.

## Apple Silicon setup

## Native personal Mac app

`music_ai.gui` provides a normal macOS window for choosing audio and a result
folder, selecting the Small/Medium profile, optionally listing expected
instruments, and confirming meter/pulse interpretation. The personal packaged
build can include already-authorized local Small and Beat This weights for
fully offline use; model-containing app bundles must not be committed to this
repository.

For a development launch:

```bash
python -m pip install -e ".[dev,muscriptor,timing,ui]"
music-ai-gui
```

Use a **native ARM Python 3.11**. It is selected deliberately because the audio-ML dependencies are more reliable there than on the system Python 3.12. Avoid an Intel/Rosetta interpreter: check that the command below prints `arm64` before creating the environment.

```bash
# One-time, if a native Python 3.11 is not already present:
arch -arm64 /opt/homebrew/bin/brew install python@3.11

/opt/homebrew/opt/python@3.11/bin/python3.11 -c "import platform; print(platform.machine())"
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The first transcription may download model weights. The project defaults to Demucs's four-source `htdemucs` model because it produced the best piano transcription in local testing. It separates `vocals`, `drums`, `bass`, and `other`; piano and strings can coexist in `other`. The experimental `htdemucs_6s` model remains available through `--model htdemucs_6s`, but is not recommended here because its piano stem introduced audible degradation. The separator is memory-intensive but workable on an M1 Pro with 16 GB for normal-length tracks; close other apps for long tracks.

## Run it

```bash
music-ai transcribe /path/to/song.mp3 --output outputs/song
```

On an Apple Silicon Mac the separator uses the MPS accelerator by default. Use `--device cpu` only if MPS is unavailable.

Output:

- `outputs/song/stems/` — separated WAV stems
- `outputs/song/events.json` — editable intermediate representation
- `outputs/song/arrangement.mid` — one MIDI track per part
- `outputs/song/score.musicxml` — editable sheet music for MuseScore or Sibelius

The current part labels come from Demucs stem names. A learned audio classifier can replace `StemLabeler` without changing the pipeline. Drums are retained as a part but intentionally emit no notes until we add a drum-aware transcriber.

## Correct the draft as sheet music

Open `score.musicxml` in MuseScore Studio. It shows conventional staves and lets you select, move, delete, or re-enter notes like a normal score editor. Save the corrected score as `.mscz`, then export MIDI or MusicXML whenever needed.

For an existing output made before MusicXML and automatic instrument detection were added:

```bash
music-ai finalize outputs/bright-places-full
```

This preserves the existing stem identities and writes
`finalized/score.musicxml`, `finalized/arrangement.mid`, and
`finalized/events.json`. It never renames a mixed `other` stem to Piano or
Violin.

## Optional browser review screen

The browser review screen remains available for listening to raw stems and inspecting event data, but it is not the primary score-correction workflow.

```bash
pip install -e ".[review]"
music-ai review outputs/bright-places-full
```

The app opens locally in your browser and writes reviewed files to `outputs/bright-places-full/reviewed/`.

## Safe local smoke test

This creates a two-note synthetic WAV—no copyrighted audio needed—and runs the same transcription and output stages, skipping separation because the input is already one isolated part.

```bash
python scripts/generate_test_tone.py
music-ai transcribe examples/two_notes.wav --separator passthrough --output outputs/smoke
pytest
```

## Architecture and next steps

Adapters are deliberately narrow:

- `separation.py`: `DemucsSeparator.separate()` returns `{stem_name: Path}`.
- `labeling.py`: `StemLabeler.label()` provides names and General MIDI program suggestions.
- `transcription.py`: `BasicPitchTranscriber.transcribe()` returns `NoteEvent` values.
- `domain.py`: versioned JSON schema boundary for correction, quantization and exporters.

The remaining notation work is grand-staff engraving, richer time-signature
change handling, confidence-guided manual beat correction, and a native Mac
interface. Drum transcription also remains separate. Replace any adapter with
a custom-trained model when evaluation says it is worthwhile.
