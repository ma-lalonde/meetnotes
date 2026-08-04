from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QFileDialog, QMenu, QSystemTrayIcon

from . import audio, icons, outputs


class Tray(QSystemTrayIcon):
    def __init__(self, cfg, session, window, parent=None):
        super().__init__(icons.state_icon("idle"), parent)
        self.cfg = cfg
        self.session = session
        self.window = window
        self.state = "idle"

        self.menu = QMenu()
        self.action_record = QAction("Record", self.menu)
        self.action_record.triggered.connect(self.toggle)

        self.action_auto = QAction("Auto post-process on stop", self.menu)
        self.action_auto.setCheckable(True)
        self.action_auto.setChecked(cfg.auto_process)
        self.action_auto.toggled.connect(self.set_auto)

        self.action_folder = QAction("Output folder...", self.menu)
        self.action_folder.triggered.connect(self.pick_folder)

        self.menu_mic = QMenu("Microphone", self.menu)
        self.menu_system = QMenu("System audio", self.menu)

        action_window = QAction("Open window", self.menu)
        action_window.triggered.connect(self.window.surface)
        action_open = QAction("Open output folder", self.menu)
        action_open.triggered.connect(self.open_folder)
        action_quit = QAction("Quit", self.menu)
        action_quit.triggered.connect(self.quit)

        self.menu.addAction(self.action_record)
        self.menu.addSeparator()
        self.menu.addAction(self.action_auto)
        self.menu.addSeparator()
        self.menu.addAction(self.action_folder)
        self.menu.addMenu(self.menu_mic)
        self.menu.addMenu(self.menu_system)
        self.menu.addSeparator()
        self.menu.addAction(action_window)
        self.menu.addAction(action_open)
        self.menu.addAction(action_quit)

        self.menu.aboutToShow.connect(self.refresh_sources)
        self.setContextMenu(self.menu)
        self.activated.connect(self.on_activated)
        self.refresh_sources()
        self.set_state("idle", "")

    def on_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.toggle()

    def toggle(self):
        if self.session.recording:
            self.session.stop()
        else:
            self.window.start_from_tray()

    def set_auto(self, enabled: bool):
        self.cfg.auto_process = enabled
        self.cfg.save()

    def pick_folder(self):
        chosen = QFileDialog.getExistingDirectory(None, "Output folder", str(self.cfg.root))
        if chosen:
            self.cfg.data_dir = chosen
            self.cfg.save()
            self.window.reload()

    def open_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        self.cfg.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.cfg.root)))

    def refresh_sources(self):
        sources = audio.list_sources()
        self._fill(self.menu_mic, [s for s in sources if s.kind == "mic"], "mic_source")
        self._fill(self.menu_system, [s for s in sources if s.kind == "system"], "system_source")
        if self.session.recording:
            self.action_record.setText(f"Stop - {outputs.clock(self.session.elapsed())}")
        else:
            self.action_record.setText("Record")

    def _fill(self, menu: QMenu, sources, key: str):
        menu.clear()
        group = QActionGroup(menu)
        group.setExclusive(True)
        current = getattr(self.cfg.capture, key)
        if not sources:
            empty = QAction("(none found)", menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        for source in sources:
            action = QAction(source.label, menu)
            action.setCheckable(True)
            action.setChecked(source.target == current)
            action.triggered.connect(lambda _=False, n=source.target, k=key: self.pick_source(k, n))
            group.addAction(action)
            menu.addAction(action)

    def pick_source(self, key: str, name: str):
        setattr(self.cfg.capture, key, name)
        self.cfg.save()
        self.window.reload()

    def set_state(self, state: str, detail: str):
        self.state = state
        self.setIcon(icons.state_icon(state))
        label = {"idle": "Ready", "recording": "Recording", "processing": "Processing",
                 "done": "Done", "failed": "Failed"}.get(state, state)
        self.setToolTip(f"meetnotes - {label}" + (f"\n{detail}" if detail else ""))
        if state in ("done", "failed") and detail:
            self.showMessage("meetnotes", detail, icons.state_icon(state), 5000)

    def quit(self):
        from PySide6.QtWidgets import QApplication

        if self.session.recording:
            self.session.stop()
        QApplication.quit()
