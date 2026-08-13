"""Native desktop front end for non-technical transcription testing."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
import traceback


def _safe_name(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    return value[:60] or "transcription"


def run_worker(args: argparse.Namespace) -> int:
    """Run in a child process so model inference cannot freeze the window."""
    try:
        from music_ai.muscriptor_transcription import MuscriptorTranscriber

        numerator = denominator = None
        if args.meter != "automatic":
            numerator, denominator = (int(item) for item in args.meter.split("/"))
        instruments = [item.strip() for item in args.instruments.split(",") if item.strip()] or None
        print("Starting transcription…", flush=True)
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        bundled_weights = bundle_root / "models" / "muscriptor-small" / "model.safetensors"
        bundled_beat = bundle_root / "models" / "beat-this-final0.ckpt"
        weights = bundled_weights if bundled_weights.is_file() else None
        beat_checkpoint = str(bundled_beat) if bundled_beat.is_file() else "final0"
        score = MuscriptorTranscriber(
            model_size=args.profile, device="auto", weights_path=weights
        ).run(
            Path(args.audio), Path(args.output), instruments=instruments,
            detect_timing=True, meter_numerator=numerator,
            meter_denominator=denominator, beat_unit_quarters=args.beat_unit,
            beat_checkpoint=beat_checkpoint,
        )
        for warning in score.timing_map.warnings if score.timing_map else []:
            print(f"TIMING WARNING: {warning}", flush=True)
        print(f"COMPLETE: {args.output}", flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="small")
    parser.add_argument("--instruments", default="")
    parser.add_argument("--meter", default="automatic")
    parser.add_argument("--beat-unit", type=float, default=None)
    return parser


def launch_gui() -> int:
    from PySide6.QtCore import QProcess, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
        QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
        QProgressBar, QPushButton, QVBoxLayout, QWidget,
    )

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Music AI — Audio to Sheet Music")
            self.resize(720, 650)
            self.process: QProcess | None = None
            self.result_dir: Path | None = None

            root = QWidget()
            layout = QVBoxLayout(root)
            title = QLabel("Turn an MP3 or WAV into editable sheet music")
            title.setStyleSheet("font-size: 22px; font-weight: 600; margin-bottom: 8px;")
            layout.addWidget(title)
            explanation = QLabel(
                "Creates MusicXML for MuseScore/Sibelius, multitrack MIDI, and detailed JSON. "
                "The first run downloads the selected non-commercial model."
            )
            explanation.setWordWrap(True)
            layout.addWidget(explanation)

            form = QFormLayout()
            self.audio = QLineEdit()
            audio_button = QPushButton("Choose audio…")
            audio_button.clicked.connect(self.choose_audio)
            audio_row = QHBoxLayout()
            audio_row.addWidget(self.audio)
            audio_row.addWidget(audio_button)
            form.addRow("Audio file", audio_row)

            self.destination = QLineEdit(str(Path.home() / "Documents" / "Music AI Results"))
            destination_button = QPushButton("Choose folder…")
            destination_button.clicked.connect(self.choose_destination)
            destination_row = QHBoxLayout()
            destination_row.addWidget(self.destination)
            destination_row.addWidget(destination_button)
            form.addRow("Save results in", destination_row)

            self.profile = QComboBox()
            self.profile.addItem("Small — bundled, recommended for M1 Air", "small")
            form.addRow("Model", self.profile)

            self.instruments = QLineEdit()
            self.instruments.setPlaceholderText("Blank = automatic; example: acoustic_piano,voice,drums")
            form.addRow("Expected instruments", self.instruments)

            self.meter = QComboBox()
            self.meter.addItems(["automatic", "4/4", "3/4", "6/8"])
            self.meter.currentTextChanged.connect(self.update_beat_unit)
            form.addRow("Written meter", self.meter)

            self.beat_unit = QComboBox()
            self.beat_unit.addItem("Automatic pulse interpretation", None)
            self.beat_unit.addItem("Eighth-note pulse", 0.5)
            self.beat_unit.addItem("Quarter-note pulse", 1.0)
            self.beat_unit.addItem("Dotted-quarter pulse", 1.5)
            form.addRow("Detected pulse means", self.beat_unit)

            layout.addLayout(form)

            local_note = QLabel(
                "This personal build contains the Small transcription and beat models. "
                "It runs locally and does not require an account or upload your music."
            )
            local_note.setWordWrap(True)
            layout.addWidget(local_note)

            self.run_button = QPushButton("Transcribe music")
            self.run_button.setStyleSheet("font-size: 17px; padding: 10px; font-weight: 600;")
            self.run_button.clicked.connect(self.run_transcription)
            layout.addWidget(self.run_button)

            self.progress = QProgressBar()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            layout.addWidget(self.progress)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setPlaceholderText("Progress and timing warnings will appear here.")
            layout.addWidget(self.log, 1)

            buttons = QHBoxLayout()
            self.open_score_button = QPushButton("Open score")
            self.open_folder_button = QPushButton("Open result folder")
            self.open_score_button.setEnabled(False)
            self.open_folder_button.setEnabled(False)
            self.open_score_button.clicked.connect(self.open_score)
            self.open_folder_button.clicked.connect(self.open_folder)
            buttons.addWidget(self.open_score_button)
            buttons.addWidget(self.open_folder_button)
            layout.addLayout(buttons)
            self.setCentralWidget(root)

        def choose_audio(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose music", str(Path.home()),
                "Audio (*.mp3 *.wav *.flac *.ogg *.oga *.m4a);;All files (*)",
            )
            if selected:
                self.audio.setText(selected)

        def choose_destination(self) -> None:
            selected = QFileDialog.getExistingDirectory(
                self, "Choose result folder", self.destination.text() or str(Path.home())
            )
            if selected:
                self.destination.setText(selected)

        def update_beat_unit(self, meter: str) -> None:
            if meter == "6/8" and self.beat_unit.currentData() is None:
                self.beat_unit.setCurrentIndex(1)

        def run_transcription(self) -> None:
            audio = Path(self.audio.text()).expanduser()
            base = Path(self.destination.text()).expanduser()
            if not audio.is_file():
                QMessageBox.warning(self, "Choose audio", "Please choose an MP3, WAV, or other audio file.")
                return
            base.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.result_dir = base / f"{_safe_name(audio)}-{stamp}"
            command = [
                sys.executable, "--worker", "--audio", str(audio),
                "--output", str(self.result_dir), "--profile", self.profile.currentData(),
                "--instruments", self.instruments.text(), "--meter", self.meter.currentText(),
            ]
            if self.beat_unit.currentData() is not None:
                command.extend(["--beat-unit", str(self.beat_unit.currentData())])
            self.log.clear()
            self.log.appendPlainText(f"Results: {self.result_dir}\n")
            self.run_button.setEnabled(False)
            self.open_score_button.setEnabled(False)
            self.open_folder_button.setEnabled(False)
            self.progress.setRange(0, 0)
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.start(command[0], command[1:])

        def read_output(self) -> None:
            if self.process:
                value = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
                if value:
                    self.log.appendPlainText(value.rstrip())

        def finished(self, exit_code: int) -> None:
            self.read_output()
            self.progress.setRange(0, 1)
            self.run_button.setEnabled(True)
            success = exit_code == 0 and self.result_dir is not None
            self.progress.setValue(1 if success else 0)
            self.open_score_button.setEnabled(success)
            self.open_folder_button.setEnabled(success)
            if success:
                QMessageBox.information(self, "Transcription complete", "The editable score is ready.")
            else:
                QMessageBox.warning(
                    self, "Transcription stopped",
                    "Read the final message in the progress box. Existing results were not overwritten.",
                )

        def open_score(self) -> None:
            if self.result_dir:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_dir / "score.musicxml")))

        def open_folder(self) -> None:
            if self.result_dir:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_dir)))

    application = QApplication(sys.argv)
    application.setApplicationName("Music AI")
    window = Window()
    window.show()
    return application.exec()


def main() -> int:
    if "--worker" in sys.argv:
        return run_worker(_worker_parser().parse_args())
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
