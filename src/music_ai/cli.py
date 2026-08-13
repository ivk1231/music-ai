"""Command-line entrypoint."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Annotated

import typer

from .labeling import StemLabeler
from .pipeline import Pipeline
from .separation import DemucsSeparator, PassthroughSeparator
from .transcription import BasicPitchTranscriber
from .midi import write_multitrack_midi
from .notation import write_musicxml
from .domain import ScoreArtifact
from .muscriptor_transcription import MuscriptorTranscriber, MuscriptorUnavailableError
from .beat_tracking import BeatThisTracker, BeatTrackingError
from .quantization import quantize_parts

app = typer.Typer(no_args_is_help=True, help="Audio to editable MIDI baseline.")


@app.callback()
def main() -> None:
    """Audio to editable MIDI baseline."""


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(exists=True, readable=True, help="Input MP3, WAV, or other Demucs-supported audio.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Directory for stems, JSON, and MIDI.")] = Path("outputs/run"),
    separator: Annotated[str, typer.Option(help="demucs (default) or passthrough for isolated/test audio.")] = "demucs",
    model: Annotated[str, typer.Option(help="Demucs model name.")] = "htdemucs",
    device: Annotated[str, typer.Option(help="Demucs device: mps for Apple Silicon, or cpu.")] = "mps",
    detect_instruments: Annotated[bool, typer.Option("--detect-instruments/--no-detect-instruments", help="Experimental recognition after a model has produced separate instrument stems.")] = False,
    bpm: Annotated[int, typer.Option(min=20, max=300, help="Temporary MIDI timing assumption.")] = 120,
) -> None:
    chosen_separator = DemucsSeparator(model, device) if separator == "demucs" else PassthroughSeparator()
    if separator not in {"demucs", "passthrough"}:
        raise typer.BadParameter("Choose demucs or passthrough.")
    if detect_instruments:
        typer.echo("Instrument relabeling is disabled for mixed separator stems; use transcribe-score instead.")
    score = Pipeline(chosen_separator, BasicPitchTranscriber(), StemLabeler()).run(audio, output, bpm)
    typer.echo(f"Created JSON, MIDI, and MusicXML in {output} ({len(score.parts)} parts).")


@app.command("transcribe-score")
def transcribe_score(
    audio: Annotated[Path, typer.Argument(exists=True, readable=True, help="Input MP3, WAV, FLAC, OGG, or M4A.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="New directory for this MuScriptor run.")] = Path("outputs/muscriptor/run"),
    profile: Annotated[str, typer.Option(help="small for an M1 Air; medium for an M1 Pro.")] = "small",
    instruments: Annotated[str, typer.Option(help="Optional comma-separated MuScriptor instrument names, e.g. acoustic_piano,violin.")] = "",
    device: Annotated[str, typer.Option(help="auto selects Apple Metal; mps and cpu are also valid.")] = "auto",
    bpm: Annotated[int, typer.Option(min=20, max=300, help="Temporary notation timing assumption.")] = 120,
    detect_timing: Annotated[bool, typer.Option("--detect-timing/--fixed-tempo", help="Detect beats, downbeats, and changing tempo for notation.")] = True,
    meter: Annotated[str, typer.Option(help="Optional meter override such as 3/4 or 6/8.")] = "",
    beat_unit: Annotated[float | None, typer.Option(help="Quarter lengths per detected pulse; use 1.5 for compound dotted-quarter beats.")] = None,
) -> None:
    """Transcribe a full mix directly into instrument-specific score parts."""
    conditioned = [item.strip() for item in instruments.split(",") if item.strip()] or None
    meter_numerator = meter_denominator = None
    if meter:
        try:
            meter_numerator, meter_denominator = (int(value) for value in meter.split("/", 1))
        except (ValueError, TypeError) as exc:
            raise typer.BadParameter("Meter must look like 3/4, 4/4, or 6/8.") from exc
    try:
        score = MuscriptorTranscriber(model_size=profile, device=device).run(
            audio, output, bpm=bpm, instruments=conditioned,
            detect_timing=detect_timing, meter_numerator=meter_numerator,
            meter_denominator=meter_denominator, beat_unit_quarters=beat_unit,
        )
    except (FileExistsError, MuscriptorUnavailableError, BeatTrackingError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    note_count = sum(len(part.notes) for part in score.parts)
    for warning in score.timing_map.warnings if score.timing_map else []:
        typer.echo(f"Timing warning: {warning}", err=True)
    typer.echo(
        f"Created instrument-specific JSON, MIDI, and MusicXML in {output} "
        f"({len(score.parts)} parts, {note_count} notes)."
    )


@app.command("retime-score")
def retime_score(
    events: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="An existing events.json transcription.")],
    audio: Annotated[Path, typer.Argument(exists=True, readable=True, help="The original audio used for the transcription.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="New directory for tempo-aware JSON, MIDI, and MusicXML.")],
    meter: Annotated[str, typer.Option(help="Optional meter override such as 3/4 or 6/8.")] = "",
    beat_unit: Annotated[float | None, typer.Option(help="Quarter lengths per detected pulse; 1.5 means a dotted-quarter pulse.")] = None,
) -> None:
    """Add detected beats, changing tempo, and notation quantization to an existing score."""
    numerator = denominator = None
    if meter:
        try:
            numerator, denominator = (int(value) for value in meter.split("/", 1))
        except (ValueError, TypeError) as exc:
            raise typer.BadParameter("Meter must look like 3/4, 4/4, or 6/8.") from exc
    if output.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        score = ScoreArtifact.read_json(events)
        timing_map = BeatThisTracker().track(
            audio, numerator=numerator, denominator=denominator,
            beat_unit_quarters=beat_unit,
        )
        score.schema_version = "2.0"
        score.timing_map = timing_map
        score.quantized_parts = quantize_parts(score.parts, timing_map)
        score.metadata.update({"timing_backend": timing_map.source, "bpm_assumption": None})
        score.write_json(staging / "events.json")
        write_multitrack_midi(score, staging / "arrangement.mid")
        write_musicxml(score, staging / "score.musicxml")
        staging.rename(output)
    except (BeatTrackingError, ValueError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise typer.BadParameter(str(exc)) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for warning in timing_map.warnings:
        typer.echo(f"Timing warning: {warning}", err=True)
    typer.echo(f"Created tempo-aware JSON, MIDI, and MusicXML in {output}.")


@app.command()
def finalize(
    output: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Existing pipeline output directory.")],
    bpm: Annotated[int, typer.Option(min=20, max=300)] = 120,
    device: Annotated[str, typer.Option(help="mps for Apple Silicon, or cpu.")] = "mps",
) -> None:
    """Auto-label ambiguous stems and create corrected MIDI + MusicXML."""
    score = ScoreArtifact.read_json(output / "events.json")
    typer.echo("Kept separator stem identities unchanged; mixed stems are never relabeled as instruments.")
    destination = output / "finalized"
    destination.mkdir(exist_ok=True)
    score.metadata.update({"bpm_assumption": bpm, "instrument_detection": "not-applied-to-separator-buckets"})
    score.write_json(destination / "events.json")
    write_multitrack_midi(score, destination / "arrangement.mid", bpm=bpm)
    write_musicxml(score, destination / "score.musicxml", bpm=bpm)
    typer.echo(f"Created editable score, MIDI, and JSON in {destination}.")


@app.command()
def review(
    output: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A pipeline output directory containing events.json.")],
) -> None:
    """Open the local correction interface for a transcription output."""
    if not (output / "events.json").is_file():
        raise typer.BadParameter("This folder does not contain events.json.")
    app_path = Path(__file__).with_name("review_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--", str(output.resolve())], check=True)


if __name__ == "__main__":
    app()
