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

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap

COLORS = {
    "idle": "#7a7a7a",
    "recording": "#d64545",
    "processing": "#d6a545",
    "done": "#45a05a",
    "failed": "#b03030",
}


def state_icon(state: str, size: int = 32) -> QIcon:
    """Tray icon drawn at runtime so no image assets need shipping."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QBrush(QColor(COLORS.get(state, COLORS["idle"]))))
    painter.setPen(Qt.NoPen)

    inset = size // 8
    box = QRect(inset, inset, size - 2 * inset, size - 2 * inset)
    if state == "recording":
        painter.drawEllipse(box)
    elif state == "processing":
        painter.drawEllipse(box)
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
        painter.setPen(QColor("#ffffff"))
        painter.drawArc(box, 0, 180 * 16)
    else:
        painter.drawRoundedRect(box, size // 6, size // 6)
    painter.end()
    return QIcon(pixmap)
