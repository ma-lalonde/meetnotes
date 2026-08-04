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
