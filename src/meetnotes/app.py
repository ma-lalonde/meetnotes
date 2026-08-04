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

import os
import sys

MISSING_LIBS = """meetnotes could not load Qt: {error}

PySide6 ships Python wheels but relies on system graphics libraries.
On Debian or Ubuntu:

  sudo apt install libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \\
                   libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libdbus-1-3

Everything except the window works without them:

  uv run meetnotes doctor
  uv run meetnotes record --title test
"""

NO_TRAY = (
    "No system tray is available. GNOME needs the AppIndicator extension; "
    "KDE and most other desktops work out of the box. Everything still works "
    "from this window."
)


def run(cfg, check: bool = False, platform: str = "") -> int:
    """Launch the window and tray.

    check builds every screen and exits, which is enough to catch import and
    construction errors on a machine with no display.
    """
    if platform:
        os.environ["QT_QPA_PLATFORM"] = platform
    elif check:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import QApplication, QSystemTrayIcon
    except ImportError as exc:
        print(MISSING_LIBS.format(error=exc))
        return 1

    from . import audio, store
    from .session import Session
    from .tray import Tray
    from .window import MainWindow

    class Bridge(QObject):
        """Marshals worker-thread callbacks onto the GUI thread.

        Qt queues signals across threads automatically, so nothing else in the
        codebase needs to know which thread it is on.
        """

        segment = Signal(dict)
        note = Signal(dict)
        state = Signal(str, str)

    app = QApplication(sys.argv)
    app.setApplicationName("meetnotes")
    app.setQuitOnLastWindowClosed(False)

    store.recover(cfg.root)

    bridge = Bridge()
    session = Session(
        cfg,
        on_segment=bridge.segment.emit,
        on_note=bridge.note.emit,
        on_state=bridge.state.emit,
    )

    window = MainWindow(cfg, session)
    bridge.segment.connect(window.record.on_segment)
    bridge.note.connect(window.record.on_note)
    bridge.state.connect(window.on_state)

    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = Tray(cfg, session, window)
        bridge.state.connect(tray.set_state)
        tray.show()
    else:
        window.show_banner(NO_TRAY)

    # The window opens on launch. The tray is for reaching the app again after
    # the window is closed, not a replacement for showing it in the first place.
    if not cfg.start_in_tray or tray is None:
        window.surface()

    if check:
        for index in range(window.stack.count()):
            window.show_screen(index)
        window.show_screen(0)
        if tray is not None:
            tray.refresh_sources()
        print(f"built {window.stack.count()} screens, tray={'yes' if tray else 'no'}")
        return 0

    if not audio.list_sources():
        window.show_banner(
            "No audio sources found. Install pipewire and pulseaudio-utils, "
            "then reopen this window."
        )
        window.show()
    elif not (cfg.capture.mic_source or cfg.capture.system_source):
        window.show_banner("Pick a microphone and a system audio source to start recording.")
        window.show()

    return app.exec()
