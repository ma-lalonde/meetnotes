# meetnotes - local meeting recorder with live transcription and notes
# Copyright (C) 2026 Marc-Antoine Lalonde
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path

import tempfile
import threading

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from . import artifacts, audio, hardware, llm, models, outputs, prompts, store, theme


# (label, mode, language codes)
LANGUAGE_CHOICES = [
    ("French and English", "restrict", ("fr", "en")),
    ("Mainly French, some English", "primary", ("fr",)),
    ("Mainly English, some French", "primary", ("en",)),
    ("French only", "primary", ("fr",)),
    ("English only", "primary", ("en",)),
    ("Detect anything", "auto", ()),
]


class RecordScreen(QWidget):
    level = Signal(str, float)

    def __init__(self, cfg, session, window):
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.window = window
        self.colors: dict[str, str] = {}
        self.meters: dict[str, audio.Meter] = {}
        self.level.connect(self._on_level)

        self.title = QLineEdit()
        self.title.setPlaceholderText("Meeting title (defaults to both names)")
        self.mic = QComboBox()
        self.system = QComboBox()
        # Speaker names: yours is stable and gets saved, the other side changes
        # per meeting and does not.
        self.mic_name = QLineEdit(cfg.capture.mic_label)
        self.mic_name.setPlaceholderText("Your name")
        self.mic_name.setMaximumWidth(180)
        self.system_name = QLineEdit(cfg.capture.system_label)
        self.system_name.setPlaceholderText("Who you are meeting")
        self.system_name.setMaximumWidth(180)
        self.mic_level = self._make_meter()
        self.system_level = self._make_meter()
        self.button = QPushButton("Start recording")
        self.button.clicked.connect(self.toggle)
        self.elapsed = QLabel("00:00:00")
        clock_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        clock_font.setPointSize(clock_font.pointSize() + 6)
        self.elapsed.setFont(clock_font)

        mic_row = QHBoxLayout()
        mic_row.addWidget(self.mic, 3)
        mic_row.addWidget(self.mic_level, 2)
        mic_row.addWidget(self.mic_name, 0)
        system_row = QHBoxLayout()
        system_row.addWidget(self.system, 3)
        system_row.addWidget(self.system_level, 2)
        system_row.addWidget(self.system_name, 0)

        top = QFormLayout()
        top.addRow("Title", self.title)
        top.addRow("Microphone", mic_row)
        top.addRow("System audio", system_row)
        top.addRow("", self._hint(
            "Names on the right label each speaker in the transcript and summary. "
            "Yours is remembered; the other changes per meeting."
        ))

        self.engine = QLabel("")
        theme.muted(self.engine)
        self.refresh_engine()

        controls = QHBoxLayout()
        controls.addWidget(self.button)
        controls.addWidget(self.elapsed)
        controls.addStretch()
        controls.addWidget(self.engine)

        self.transcript = QListWidget()
        self.transcript.setWordWrap(True)
        self.notes = QListWidget()
        self.notes.setWordWrap(True)
        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Note, then Enter")
        self.note_input.returnPressed.connect(self.add_note)
        theme.comfortable(
            self.title, self.mic, self.system, self.note_input,
            self.mic_name, self.system_name,
        )

        left = QVBoxLayout()
        left.addWidget(QLabel("Live transcript"))
        left.addWidget(self.transcript)
        right = QVBoxLayout()
        right.addWidget(QLabel("Notes"))
        right.addWidget(self.notes)
        right.addWidget(self.note_input)

        panes = QHBoxLayout()
        panes.addLayout(left, 3)
        panes.addLayout(right, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addLayout(panes)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.reload()
        # Persist on change, not only on Start, so picking a source and walking
        # away still leaves it selected next launch.
        self.mic.currentIndexChanged.connect(lambda _: self._persist_sources())
        self.system.currentIndexChanged.connect(lambda _: self._persist_sources())

    def reload(self):
        for combo, kind, key in (
            (self.mic, "mic", "mic_source"),
            (self.system, "system", "system_source"),
        ):
            current = getattr(self.cfg.capture, key)
            combo.blockSignals(True)
            combo.clear()
            for source in audio.list_sources():
                if source.kind == kind:
                    combo.addItem(source.label, source.target)
            index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def refresh_engine(self):
        """Always show what will actually run, so a bad result is explainable."""
        plan = hardware.plan(self.cfg)
        languages = (
            "/".join(self.cfg.asr.languages)
            if self.cfg.asr.language_mode == "restrict"
            else (self.cfg.asr.language or "auto")
        )
        self.engine.setText(
            f"{plan['live_model']} on {plan['device']} ({plan['compute_type']}), {languages}"
        )

    def _make_meter(self) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(-60, 0)
        bar.setValue(-60)
        bar.setFormat("%v dBFS")
        bar.setTextVisible(True)
        return bar

    def _on_level(self, kind: str, dbfs: float):
        bar = self.mic_level if kind == "mic" else self.system_level
        bar.setValue(int(max(-60.0, min(0.0, dbfs))))

    def showEvent(self, event):
        super().showEvent(event)
        # Qt only delivers this when the widget is becoming visible, so it is
        # the signal itself. Testing isVisible() here answered False during the
        # first show, which left the meters never started until a tab change
        # sent a second showEvent.
        self.start_meters()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.stop_meters()

    def start_meters(self):
        """Run the level meters whenever this screen is visible.

        While recording the levels come from the recording itself rather than
        a second capture stream, so nothing extra touches the devices.
        """
        self.stop_meters()
        work = Path(tempfile.mkdtemp(prefix="meetnotes-levels-"))

        if self.session.recording and self.session.recorder is not None:
            # Track keys are the speaker names chosen for this meeting, which
            # may differ from the stored defaults.
            names = list(self.session.recorder.paths)
            for kind, label in zip(("mic", "system"), names):
                path = self.session.recorder.paths.get(label)
                if path is None:
                    continue
                meter = audio.FileMeter(path, lambda db, k=kind: self.level.emit(k, db))
                self.meters[kind] = meter
                meter.start()
            return

        known = {s.target: s for s in audio.list_sources()}
        for kind, key in (("mic", "mic_source"), ("system", "system_source")):
            source = known.get(getattr(self.cfg.capture, key))
            if source is None:
                continue
            meter = audio.Meter(
                self.cfg.capture.record_cmd, self.cfg.capture.sample_rate, source, work,
                lambda db, k=kind: self.level.emit(k, db),
            )
            self.meters[kind] = meter
            meter.start()

    def stop_meters(self):
        for meter in self.meters.values():
            meter.stop()
        for meter in self.meters.values():
            meter.join(timeout=3)
        self.meters.clear()
        self.mic_level.setValue(-60)
        self.system_level.setValue(-60)

    def _persist_sources(self):
        if self.mic.currentData():
            self.cfg.capture.mic_source = self.mic.currentData()
        if self.system.currentData():
            self.cfg.capture.system_source = self.system.currentData()
        # Your own name is stable, so it becomes the default. The other side is
        # different every meeting and is deliberately not remembered.
        mine = self.mic_name.text().strip()
        if mine:
            self.cfg.capture.mic_label = mine
        self.cfg.save()

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        theme.muted(label)
        return label

    def toggle(self):
        if self.session.recording:
            self.session.stop()
            self.timer.stop()
            self.button.setText("Start recording")
            self.start_meters()
            return
        self._persist_sources()
        self.transcript.clear()
        self.notes.clear()
        try:
            self.session.start(
                self.title.text().strip(),
                mic_label=self.mic_name.text().strip(),
                system_label=self.system_name.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Cannot record", str(exc))
            return
        self.timer.start()
        self.button.setText("Stop")
        self.note_input.setFocus()
        self.start_meters()

    def tick(self):
        self.elapsed.setText(outputs.clock(self.session.elapsed()))

    def add_note(self):
        text = self.note_input.text().strip()
        if not text or not self.session.recording:
            return
        self.session.add_note(text)
        self.note_input.clear()

    def on_segment(self, segment: dict):
        if segment["speaker"] not in self.colors:
            palette = theme.speaker_colors(self.transcript)
            self.colors[segment["speaker"]] = palette[len(self.colors) % len(palette)]
        item = QListWidgetItem(
            f"[{outputs.clock(segment['start'])}] {segment['speaker']}: {segment['text']}"
        )
        item.setForeground(QColor(self.colors[segment["speaker"]]))
        at_bottom = self.transcript.verticalScrollBar().value() >= (
            self.transcript.verticalScrollBar().maximum() - 4
        )
        self.transcript.addItem(item)
        if at_bottom:
            self.transcript.scrollToBottom()

    def on_note(self, note: dict):
        self.notes.addItem(f"[{outputs.clock(note['at'])}] {note['text']}")
        self.notes.scrollToBottom()


class LibraryScreen(QWidget):
    HEADERS = ["Meeting", "State", "Length", "Artifacts", ""]

    def __init__(self, cfg, session, window):
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.window = window

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        process = QPushButton("Post-process")
        process.clicked.connect(lambda: self.run(False))
        force = QPushButton("Force regenerate")
        force.clicked.connect(lambda: self.run(True))
        reveal = QPushButton("Open folder")
        reveal.clicked.connect(self.open_folder)
        import_audio = QPushButton("Import audio...")
        import_audio.clicked.connect(self.import_audio)

        buttons = QHBoxLayout()
        for widget in (refresh, process, force, reveal, import_audio):
            buttons.addWidget(widget)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

        # A running job writes its state to disk, but nothing tells the table.
        # Poll only while something is actually running.
        self.ticker = QTimer(self)
        self.ticker.setInterval(1000)
        self.ticker.timeout.connect(self._tick)
        self.reload()

    RUNNING_STATES = {"recording", "transcribing", "summarizing", "pending"}

    def _tick(self):
        # reload() stops the timer itself once nothing is running, so this
        # avoids scanning the meetings directory twice per tick.
        self.reload()

    def reload(self):
        meetings = store.list_meetings(self.cfg.root)
        self.table.setRowCount(len(meetings))
        running = False
        for row, meta in enumerate(meetings):
            state = meta.get("state", "")
            step = self.session.active.get(meta["id"], "")
            if step:
                percent = self.session.fraction.get(meta["id"])
                suffix = f" ({percent * 100:.0f}%)" if percent is not None else ""
                state = f"{state} - {step}{suffix}"
                running = True
            elif state in self.RUNNING_STATES:
                running = True
            self.table.setItem(row, 0, QTableWidgetItem(meta["id"]))
            self.table.setItem(row, 1, QTableWidgetItem(state))
            self.table.setItem(row, 2, QTableWidgetItem(outputs.clock(meta.get("duration", 0))))
            self.table.setItem(row, 3, QTableWidgetItem(self._artifact_summary(meta)))
            self.table.setItem(row, 4, QTableWidgetItem(meta.get("error", "")[:60]))
            self.table.item(row, 0).setData(Qt.UserRole, meta["path"])
        if running and not self.ticker.isActive():
            self.ticker.start()
        elif not running and self.ticker.isActive():
            self.ticker.stop()

    def _artifact_summary(self, meta: dict) -> str:
        path = Path(meta["path"])
        recorded = meta.get("artifacts", {})
        if not recorded:
            return "-"
        edited = []
        for name, record in recorded.items():
            target = path / name
            if name == "segments" or not target.exists():
                continue
            if artifacts.file_hash(target) != record.get("output_hash"):
                edited.append(name)
        count = len([n for n in recorded if n != "segments"])
        return f"{count} files" + (f", {len(edited)} hand-edited" if edited else "")

    def selected(self) -> Path | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return Path(self.table.item(row, 0).data(Qt.UserRole))

    def run(self, force: bool):
        path = self.selected()
        if not path:
            QMessageBox.information(self, "No meeting selected", "Pick a row first.")
            return
        if force:
            confirm = QMessageBox.question(
                self, "Force regenerate",
                f"Overwrite every generated file in {path.name}, including any you edited by hand?",
            )
            if confirm != QMessageBox.Yes:
                return
        self.session.process_async(path, force=force)

    def open_folder(self):
        path = self.selected()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def import_audio(self):
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Import audio", str(Path.home()), "Audio (*.wav *.mp3 *.m4a *.flac *.ogg)"
        )
        if not chosen:
            return
        source = Path(chosen)
        self.cfg.root.mkdir(parents=True, exist_ok=True)
        path = store.new_meeting(self.cfg.root, source.stem)
        target = path / "audio" / f"{self.cfg.capture.system_label}{source.suffix}"
        target.write_bytes(source.read_bytes())
        store.update_meta(
            path, tracks={self.cfg.capture.system_label: target.name}, state="pending"
        )
        self.reload()
        self.session.process_async(path)


class SettingsScreen(QWidget):
    gpu_log = Signal(str)

    def __init__(self, cfg, session, window):
        super().__init__()
        self.cfg = cfg
        self.window = window
        self.gpu_log.connect(self._append_gpu_log)

        self.data_dir = QLineEdit(str(cfg.root))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.pick_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.data_dir)
        folder_row.addWidget(browse)

        self.auto = QCheckBox("Post-process automatically when recording stops")
        self.auto.setChecked(cfg.auto_process)
        self.start_in_tray = QCheckBox("Start in the tray without opening the window")
        self.start_in_tray.setChecked(cfg.start_in_tray)
        self.minimize_on_quit = QCheckBox("Closing the window minimizes to the tray")
        self.minimize_on_quit.setChecked(cfg.minimize_on_quit)

        self.profile = QComboBox()
        self.profile.addItems(["auto", "gpu", "cpu"])
        self.profile.setCurrentText(cfg.asr.profile)
        self.language = QComboBox()
        for label, mode, codes in LANGUAGE_CHOICES:
            self.language.addItem(label, (mode, codes))
        index = self.language.findData((cfg.asr.language_mode, tuple(cfg.asr.languages)))
        if index < 0 and cfg.asr.language_mode == "primary":
            index = self.language.findData(("primary", (cfg.asr.language,)))
        self.language.setCurrentIndex(max(index, 0))
        theme.comfortable(self.profile, self.language, self.data_dir)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(cfg.llm.temperature)
        theme.comfortable(self.temperature)
        self.keep_loaded = QCheckBox("Keep the speech model loaded while summarizing")
        self.keep_loaded.setChecked(cfg.llm.keep_asr_loaded)

        self.summary_prompt = QPlainTextEdit(cfg.llm.summary_prompt)
        self.actions_prompt = QPlainTextEdit(cfg.llm.actions_prompt)

        self.multilingual = QCheckBox("Detect per segment when set to 'Detect anything'")
        self.multilingual.setChecked(cfg.asr.multilingual)

        self.diarize = QCheckBox("Enable speaker diarization (not implemented yet)")
        self.diarize.setEnabled(False)

        self.gpu_status = QLabel("")
        self.gpu_status.setWordWrap(True)
        self.gpu_button = QPushButton("Install GPU support")
        self.gpu_button.clicked.connect(self.install_gpu)
        self.gpu_output = QPlainTextEdit()
        self.gpu_output.setReadOnly(True)
        self.gpu_output.setMaximumHeight(110)
        self.gpu_output.hide()
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_status, 1)
        gpu_row.addWidget(self.gpu_button)
        self.refresh_gpu()

        general = QWidget()
        general_form = QFormLayout(general)
        general_form.addRow("Output folder", folder_row)
        general_form.addRow("", self.auto)
        general_form.addRow("", self.start_in_tray)
        general_form.addRow("", self.minimize_on_quit)

        speech = QWidget()
        speech_form = QFormLayout(speech)
        speech_form.addRow("Profile", self.profile)
        speech_form.addRow("", QLabel("Model choices live in the Models tab."))
        speech_form.addRow("Language", self.language)
        speech_form.addRow("", self.multilingual)
        speech_form.addRow("Acceleration", gpu_row)
        speech_form.addRow("", self.gpu_output)
        speech_form.addRow("Diarization", self.diarize)
        speech_form.addRow(
            "", QLabel(
                "Dual-track capture already labels speakers for one-on-one calls.\n"
                "Diarization only matters when several people share one stream."
            )
        )

        model = QWidget()
        model_form = QFormLayout(model)
        model_form.addRow("", QLabel("Server and model selection live in the Models tab."))
        model_form.addRow("Temperature", self.temperature)
        model_form.addRow("", self.keep_loaded)
        model_form.addRow("Summary prompt", self.summary_prompt)
        model_form.addRow("Actions prompt", self.actions_prompt)

        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        refresh_diag = QPushButton("Refresh")
        refresh_diag.clicked.connect(self.refresh_diagnostics)
        copy_diag = QPushButton("Copy")
        copy_diag.clicked.connect(self.copy_diagnostics)
        diag_buttons = QHBoxLayout()
        diag_buttons.addWidget(refresh_diag)
        diag_buttons.addWidget(copy_diag)
        diag_buttons.addStretch()
        diagnostics = QWidget()
        diag_layout = QVBoxLayout(diagnostics)
        diag_layout.addWidget(self.diagnostics)
        diag_layout.addLayout(diag_buttons)

        tabs = QTabWidget()
        tabs.addTab(general, "General")
        tabs.addTab(speech, "Speech")
        tabs.addTab(model, "Language model")
        tabs.addTab(diagnostics, "Diagnostics")
        tabs.currentChanged.connect(
            lambda i: self.refresh_diagnostics() if tabs.tabText(i) == "Diagnostics" else None
        )

        save = QPushButton("Save")
        save.clicked.connect(self.save)
        reset = QPushButton("Reset prompts")
        reset.clicked.connect(self.reset_prompts)
        self.status = QLabel("")

        buttons = QHBoxLayout()
        buttons.addWidget(save)
        buttons.addWidget(reset)
        buttons.addWidget(self.status)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addLayout(buttons)

    def pick_folder(self):
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.data_dir.text())
        if chosen:
            self.data_dir.setText(chosen)

    def refresh_diagnostics(self):
        """Everything the CLI subcommands report, in one pasteable block."""
        lines = []
        report = hardware.report(self.cfg)
        width = max(len(k) for k in report)
        lines.append("== environment ==")
        lines += [f"{k.ljust(width)}  {v}" for k, v in report.items()]

        lines.append("\n== gpu ==")
        rows = hardware.cuda_diagnostics()
        gwidth = max(len(k) for k, _ in rows)
        lines += [f"{k.ljust(gwidth)}  {v}" for k, v in rows]
        lines.append(hardware.cuda_state()["detail"])

        lines.append("\n== capture ==")
        lines.append(f"command  {self.cfg.capture.record_cmd}")
        for source in audio.list_sources():
            role = ""
            if source.target == self.cfg.capture.mic_source:
                role = f" -> {self.cfg.capture.mic_label}"
            elif source.target == self.cfg.capture.system_source:
                role = f" -> {self.cfg.capture.system_label}"
            lines.append(f"[{source.kind:6}] {source.label}{role}")
            lines.append(f"          {source.name} (target {source.target})")
        if not audio.list_sources():
            lines.append("no sources found")

        lines.append("\n== language ==")
        lines.append(f"mode {self.cfg.asr.language_mode}, languages {self.cfg.asr.languages}")

        lines.append("\n== output ==")
        lines.append(str(self.cfg.root))
        self.diagnostics.setPlainText("\n".join(lines))

    def copy_diagnostics(self):
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.diagnostics.toPlainText())

    def refresh_gpu(self):
        state = hardware.cuda_state()
        names = ", ".join(g["name"] for g in state["gpus"])
        self.gpu_status.setText(f"{state['detail']}" + (f"\n{names}" if names else ""))
        self.gpu_button.setVisible(state["installable"])
        self.gpu_button.setText("Install GPU support (about 1.4 GB)")

    def _append_gpu_log(self, line: str):
        self.gpu_output.show()
        self.gpu_output.appendPlainText(line)
        if line.startswith("installed"):
            self.refresh_gpu()

    def install_gpu(self):
        confirm = QMessageBox.question(
            self,
            "Install GPU support",
            "Download the CUDA libraries (about 1.4 GB) into this project?\n\n"
            "meetnotes needs a restart afterwards to use them.",
        )
        if confirm != QMessageBox.Yes:
            return
        self.gpu_button.setEnabled(False)
        self.gpu_output.clear()
        self.gpu_output.show()

        def run():
            ok = hardware.install_cuda(log=self.gpu_log.emit)
            if not ok:
                self.gpu_log.emit("install failed")
            self.gpu_log.emit("")

        threading.Thread(target=run, daemon=True, name="cuda-install").start()

    def reset_prompts(self):
        self.summary_prompt.setPlainText(prompts.SUMMARY)
        self.actions_prompt.setPlainText(prompts.ACTIONS)

    def save(self):
        cfg = self.cfg
        cfg.data_dir = self.data_dir.text().strip()
        cfg.auto_process = self.auto.isChecked()
        cfg.start_in_tray = self.start_in_tray.isChecked()
        cfg.minimize_on_quit = self.minimize_on_quit.isChecked()
        cfg.asr.profile = self.profile.currentText()
        mode, codes = self.language.currentData()
        cfg.asr.language_mode = mode
        cfg.asr.languages = list(codes)
        cfg.asr.language = codes[0] if codes else ""
        cfg.asr.multilingual = self.multilingual.isChecked()
        cfg.llm.temperature = self.temperature.value()
        cfg.llm.keep_asr_loaded = self.keep_loaded.isChecked()
        cfg.llm.summary_prompt = self.summary_prompt.toPlainText().strip()
        cfg.llm.actions_prompt = self.actions_prompt.toPlainText().strip()
        cfg.save()
        self.window.reload()
        resolved = hardware.plan(cfg)
        self.status.setText(
            f"Saved. Profile {resolved['profile']}, live {resolved['live_model']}, "
            f"final {resolved['final_model'] or '(skipped)'} on {resolved['device']}."
        )


class ModelsScreen(QWidget):
    """One model choice per step, offering only what fits that step."""

    def __init__(self, cfg, session, window):
        super().__init__()
        self.cfg = cfg
        self.window = window

        self.gpu = QLabel("")
        bold = self.gpu.font()
        bold.setBold(True)
        self.gpu.setFont(bold)

        self.live = QComboBox()
        self.final = QComboBox()
        self.precision = QComboBox()
        self.summary = QComboBox()
        self.summary.setEditable(True)

        self.live_note = self._note()
        self.final_note = self._note()
        self.summary_note = self._note()
        self.language_note = self._note()
        self.hardware_note = self._note()
        self._device = "cpu"
        self._free_mb = 0
        self.show_all = QCheckBox("Show models this machine or language setting rules out")
        self.show_all.toggled.connect(self.reload)
        self.vocabulary = QPlainTextEdit("\n".join(cfg.asr.vocabulary))
        self.vocabulary.setPlaceholderText("Chloé Gagnon\nCatena\nPortainer")
        self.vocabulary.setMaximumHeight(110)
        theme.comfortable(self.live, self.final, self.precision, self.summary)

        speech = QWidget()
        speech_form = QFormLayout(speech)
        speech_form.addRow("", self.hardware_note)
        speech_form.addRow("", self.language_note)
        speech_form.addRow("", self.show_all)
        speech_form.addRow("Live transcription", self.live)
        speech_form.addRow("", self.live_note)
        speech_form.addRow("Final pass", self.final)
        speech_form.addRow("", self.final_note)
        speech_form.addRow("Precision", self.precision)
        speech_form.addRow("Expected names", self.vocabulary)
        speech_form.addRow("", self._note(
            "One per line: people, companies, bands, products, jargon. These are "
            "given to the recogniser as hints, which is what stops proper nouns "
            "coming back garbled. Whisper cannot be calibrated to a voice; this "
            "is the lever it does have."
        ))
        speech_form.addRow("", self._note(
            "int8 roughly halves the memory of float16 with little quality cost. "
            "CTranslate2 accepts several more compute types, but they differ only "
            "in the precision of the layers that are not quantized, which is not "
            "a difference you can hear on a speech model."
        ))

        self.provider = QComboBox()
        self.provider.addItem("Custom", "")
        for preset in llm.PRESETS:
            self.provider.addItem(preset["name"], preset["name"])
        self.provider.activated.connect(self.apply_preset)
        self.base_url = QLineEdit(cfg.llm.base_url)
        self.api_key = QLineEdit(cfg.llm.api_key)
        self.ttl = QSpinBox()
        self.ttl.setRange(0, 86400)
        self.ttl.setSuffix(" s")
        self.ttl.setValue(cfg.llm.ttl_seconds)
        self.free_vram = QCheckBox("Unload language models before recording starts")
        self.free_vram.setChecked(cfg.llm.free_vram_before_recording)
        theme.comfortable(self.provider, self.base_url, self.api_key, self.ttl)
        fetch = QPushButton("Fetch models")
        fetch.clicked.connect(self.fetch)
        unload = QPushButton("Unload now")
        unload.clicked.connect(self.unload_now)
        server_row = QHBoxLayout()
        server_row.addWidget(fetch)
        server_row.addWidget(unload)
        server_row.addStretch()

        summarize = QWidget()
        summarize_form = QFormLayout(summarize)
        summarize_form.addRow("Provider", self.provider)
        summarize_form.addRow("Base URL", self.base_url)
        summarize_form.addRow("API key", self.api_key)
        summarize_form.addRow("Summary and actions", self.summary)
        summarize_form.addRow("", self.summary_note)
        summarize_form.addRow("Idle unload", self.ttl)
        summarize_form.addRow("", self._note(
            "LM Studio unloads a model after this long idle. 0 disables it; other "
            "servers ignore the field."
        ))
        summarize_form.addRow("", self.free_vram)
        summarize_form.addRow("", server_row)

        save = QPushButton("Save")
        save.clicked.connect(self.save)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        auto = QPushButton("Choose for this machine")
        auto.setToolTip(
            "Measure free VRAM and ask LM Studio what each model would cost, "
            "then pick the largest that fits with a usable context."
        )
        auto.clicked.connect(self.autotune)
        gguf = QPushButton("Why not the GGUF whisper models?")
        gguf.clicked.connect(
            lambda: QMessageBox.information(self, "GGUF speech models", models.gguf_note())
        )
        buttons = QHBoxLayout()
        buttons.addWidget(save)
        buttons.addWidget(refresh)
        buttons.addWidget(auto)
        buttons.addWidget(gguf)
        buttons.addStretch()
        self.status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(self.gpu)
        layout.addWidget(QLabel("Speech"))
        layout.addWidget(speech)
        layout.addWidget(QLabel("Summarization"))
        layout.addWidget(summarize)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addStretch()
        self.reload()

    def _note(self, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        theme.muted(label)
        return label

    def _fill_whisper(self, combo: QComboBox, current: str, allow_skip: bool):
        combo.blockSignals(True)
        combo.clear()
        if allow_skip:
            combo.addItem("Skip the final pass, keep the live transcript", "")
        # The config drops English-only models unless the configured languages
        # are English and nothing else; the device and free VRAM drop the small
        # models on a card with room for Turbo, which is as fast as they are.
        for choice in models.whisper_choices(
            self.cfg, all_models=self.show_all.isChecked(),
            device=self._device, free_mb=self._free_mb,
        ):
            size = f"{choice['params_m']} M" if choice["params_m"] else ""
            note = f"  -  {choice['note']}" if choice["note"] else ""
            combo.addItem(f"{choice['label']}   {size}{note}", choice["alias"])
            # The repository is what actually gets downloaded, so it stays
            # reachable without cluttering the line.
            combo.setItemData(combo.count() - 1, choice["repo"], Qt.ToolTipRole)
        index = combo.findData(current)
        if index < 0 and current:
            # A model chosen before, or hand-edited into the config, that the
            # curated list no longer offers. Keep it rather than silently
            # switching what runs.
            combo.addItem(f"{models.label(current)}   (not in the short list)", current)
            index = combo.count() - 1
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def reload(self):
        # The offered models depend on the language setting, which lives on
        # another screen, so say why the list looks the way it does.
        if models.english_only_setup(self.cfg):
            self.language_note.setText(
                "Expecting English only, so the .en builds are offered in place of "
                "their multilingual twins: same size, better at English, and no "
                "other language. Change Language in Settings to get them back."
            )
        else:
            self.language_note.setText(
                "English-only (.en) models are hidden because more than English is "
                "expected. They cannot transcribe anything else at all."
            )
        plan = hardware.plan(self.cfg)
        gpus = hardware.nvidia()
        if gpus:
            self.gpu.setText(
                f"{gpus[0]['name']} - {gpus[0]['vram_mb']} MB, "
                f"{gpus[0].get('used_mb', 0)} MB in use, running on {plan['device']}"
            )
        else:
            self.gpu.setText(f"No GPU detected, running on {plan['device']}")

        # Read once per reload: _fill_whisper is called twice and nvidia-smi is
        # a subprocess, so asking it per combo box would double the cost and
        # could hand the two lists different numbers.
        self._device = plan["device"]
        self._free_mb = gpus[0].get("free_mb", 0) if gpus else 0
        pool = [c["alias"] for c in models.whisper_choices(self.cfg)]
        _, reason = models.hardware_pool(self._device, self._free_mb, pool)
        self.hardware_note.setText(reason)
        self.hardware_note.setVisible(bool(reason) and not self.show_all.isChecked())

        self._fill_whisper(self.live, self.cfg.asr.live_model or plan["live_model"], False)
        self._fill_whisper(
            self.final,
            "" if not self.cfg.asr.final_pass else (self.cfg.asr.final_model or plan["final_model"]),
            True,
        )
        self.live_note.setText(
            "Runs on short windows while the meeting happens. Speed matters more than accuracy."
        )
        self.final_note.setText(
            "Runs once over the whole recording afterwards. This produces the saved transcript."
        )

        self.precision.blockSignals(True)
        self.precision.clear()
        for choice in models.precision_choices(plan["device"]):
            note = f"  -  {choice['note']}" if choice["note"] else ""
            self.precision.addItem(f"{choice['label']}{note}", choice["alias"])
            self.precision.setItemData(
                self.precision.count() - 1, choice["alias"], Qt.ToolTipRole
            )
        index = self.precision.findData(self.cfg.asr.compute_type)
        if index < 0 and self.cfg.asr.compute_type not in ("", "auto"):
            # Set before, or hand-edited, and not one of the two offered here.
            # Keep it rather than silently changing what runs.
            current = self.cfg.asr.compute_type
            label = models.PRECISION.get(current, (current, ""))[0]
            self.precision.addItem(f"{label}  -  {current}", current)
            index = self.precision.count() - 1
        self.precision.setCurrentIndex(index if index >= 0 else 0)
        self.precision.blockSignals(False)

        self.fetch(quiet=True)

    def apply_preset(self, index: int):
        name = self.provider.itemData(index)
        if not name or not llm.apply_preset(self.cfg, name):
            return
        self.base_url.setText(self.cfg.llm.base_url)
        self.api_key.setText(self.cfg.llm.api_key)
        self.ttl.setValue(self.cfg.llm.ttl_seconds)
        note = next(p["note"] for p in llm.PRESETS if p["name"] == name)
        self.summary_note.setText(note)
        self.fetch(quiet=True)

    def fetch(self, quiet: bool = False):
        self.cfg.llm.base_url = self.base_url.text().strip()
        self.cfg.llm.api_key = self.api_key.text().strip()
        try:
            entries = [e for e in llm.catalog(self.cfg) if models.is_language_model(e)]
        except llm.LlmError as exc:
            if not quiet:
                QMessageBox.warning(self, "Cannot reach the model server", str(exc))
            self.summary_note.setText(f"Not reachable at {self.cfg.llm.base_url}")
            return

        current = self.cfg.llm.model or self.summary.currentText()
        self.summary.blockSignals(True)
        self.summary.clear()
        for entry in entries:
            label = entry["id"]
            extras = [part for part in (entry.get("quantization"), entry.get("state")) if part]
            if extras:
                label += "  -  " + ", ".join(extras)
            self.summary.addItem(label, entry["id"])
        index = self.summary.findData(current)
        if index >= 0:
            self.summary.setCurrentIndex(index)
        elif current:
            self.summary.setCurrentText(current)
        self.summary.blockSignals(False)
        if entries:
            self.summary_note.setText(f"{len(entries)} models offered by this server")

    def unload_now(self):
        freed, detail = llm.unload_all()
        self.status.setText(detail if freed else f"could not unload: {detail}")

    def autotune(self):
        """Size the models to what this machine has free, and say what changed.

        No sample recording here, so speech models are sized rather than timed;
        `meetnotes tune --record 30` does the measured version. The part that
        matters in the UI is the language model, where free VRAM decides
        whether a usable context is available at all.
        """
        from . import tuning

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            plan = tuning.tune(self.cfg)
        except Exception as exc:
            QMessageBox.warning(self, "Could not measure this machine", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        lines = [
            f"Live speech model:  {models.label(plan.live)}",
            f"Final speech model: {models.label(plan.final)}",
            f"Device:             {plan.device} ({plan.compute_type})",
        ]
        if plan.summary_model:
            lines.append(
                f"Summary model:      {plan.summary_model}"
                f" at {plan.summary_context} tokens of context"
            )
        else:
            lines.append("Summary model:      could not be chosen")
        lines += [f"\n{note}" for note in plan.notes]

        answer = QMessageBox.question(
            self, "Chosen for this machine", "\n".join(lines) + "\n\nApply these?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        tuning.apply(plan, self.cfg)
        self.reload()
        self.status.setText("Models chosen for this machine and saved.")

    def save(self):
        cfg = self.cfg
        cfg.asr.live_model = self.live.currentData() or ""
        final = self.final.currentData()
        cfg.asr.final_pass = bool(final)
        cfg.asr.final_model = final or ""
        cfg.asr.compute_type = self.precision.currentData() or "auto"
        cfg.asr.vocabulary = [
            line.strip() for line in self.vocabulary.toPlainText().splitlines() if line.strip()
        ]
        cfg.llm.base_url = self.base_url.text().strip()
        cfg.llm.api_key = self.api_key.text().strip()
        cfg.llm.model = self.summary.currentData() or self.summary.currentText().strip()
        cfg.llm.ttl_seconds = self.ttl.value()
        cfg.llm.free_vram_before_recording = self.free_vram.isChecked()
        cfg.save()
        self.window.reload()
        self.status.setText("Saved.")


class MainWindow(QWidget):
    def __init__(self, cfg, session):
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.setWindowTitle("meetnotes")
        self.resize(1000, 640)

        self.record = RecordScreen(cfg, session, self)
        self.library = LibraryScreen(cfg, session, self)
        self.models = ModelsScreen(cfg, session, self)
        self.settings = SettingsScreen(cfg, session, self)

        self.stack = QTabWidget()
        self.stack.addTab(self.record, "Record")
        self.stack.addTab(self.library, "Library")
        self.stack.addTab(self.models, "Models")
        self.stack.addTab(self.settings, "Settings")
        self.stack.currentChanged.connect(self._on_tab)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setContentsMargins(8, 6, 8, 6)
        theme.notice(self.banner)
        self.banner.hide()

        self.status = QLabel("Ready")
        theme.muted(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(140)
        self.progress.setMaximumHeight(12)
        self.progress.hide()

        corner = QWidget()
        corner_row = QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 0, 6, 0)
        corner_row.addWidget(self.status)
        corner_row.addWidget(self.progress)
        self.stack.setCornerWidget(corner)

        layout = QVBoxLayout(self)
        layout.addWidget(self.banner)
        layout.addWidget(self.stack)

    def _on_tab(self, index: int):
        widget = self.stack.widget(index)
        if widget is self.library:
            self.library.reload()
        elif widget is self.models:
            self.models.reload()

    def show_banner(self, text: str):
        self.banner.setText(text)
        self.banner.show()

    def show_screen(self, index: int):
        self.stack.setCurrentIndex(index)

    def closeEvent(self, event):
        """Closing hides to the tray unless the user asked for a real quit."""
        from PySide6.QtWidgets import QApplication, QSystemTrayIcon

        if self.cfg.minimize_on_quit and QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
            return
        event.accept()
        QApplication.quit()

    def reload(self):
        self.record.reload()
        self.record.refresh_engine()
        self.library.reload()
        self.models.reload()

    def surface(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def start_from_tray(self):
        if self.session.recording:
            return
        try:
            self.session.start(
                "",
                mic_label=self.cfg.capture.mic_label,
                system_label=self.cfg.capture.system_label,
            )
        except Exception as exc:
            self.surface()
            QMessageBox.critical(self, "Cannot record", str(exc))
            return
        self.record.timer.start()
        self.record.button.setText("Stop")

    def on_state(self, state: str, detail: str):
        self.status.setText(f"{state}: {detail}" if detail else state)
        fractions = list(self.session.fraction.values())
        if state == "processing" and fractions:
            self.progress.setValue(int(min(fractions) * 1000))
            self.progress.show()
        elif state == "processing":
            # Working, but nothing has reported a fraction yet.
            self.progress.setRange(0, 0)
            self.progress.show()
        else:
            self.progress.setRange(0, 1000)
            self.progress.hide()
        # Every transition, not just the terminal ones: the Library is the only
        # place that shows a job is still running.
        self.library.reload()
        if state in ("done", "failed", "idle"):
            self.record.button.setText("Start recording")
            self.record.timer.stop()
