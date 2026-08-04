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

"""Theme-aware colours and sizing.

Nothing here hardcodes a colour that has to work against an unknown
background. Qt's "mid" role in particular is a frame shadow, not a text
colour, and using it for text produces black-on-grey on many themes.
"""

from PySide6.QtGui import QColor, QPalette

# Chosen for contrast against both light and dark backgrounds.
SPEAKER_COLORS_LIGHT = ["#1f5fa9", "#a35200"]
SPEAKER_COLORS_DARK = ["#7ab6f5", "#f0a860"]


def is_dark(widget) -> bool:
    return widget.palette().color(QPalette.Window).lightness() < 128


def blend(front: QColor, back: QColor, weight: float) -> QColor:
    weight = max(0.0, min(1.0, weight))
    return QColor(
        round(front.red() * weight + back.red() * (1 - weight)),
        round(front.green() * weight + back.green() * (1 - weight)),
        round(front.blue() * weight + back.blue() * (1 - weight)),
    )


def muted(widget, weight: float = 0.65):
    """Dim text that stays legible, derived from the active palette."""
    palette = widget.palette()
    dimmed = blend(
        palette.color(QPalette.WindowText), palette.color(QPalette.Window), weight
    )
    for role in (QPalette.WindowText, QPalette.Text):
        palette.setColor(role, dimmed)
    widget.setPalette(palette)
    return widget


def notice(widget):
    """A banner that reads clearly on either theme."""
    palette = widget.palette()
    window = palette.color(QPalette.Window)
    accent = QColor("#c08a2e")
    palette.setColor(QPalette.Window, blend(accent, window, 0.28))
    palette.setColor(QPalette.WindowText, palette.color(QPalette.WindowText))
    widget.setAutoFillBackground(True)
    widget.setPalette(palette)
    return widget


def speaker_colors(widget) -> list[str]:
    return SPEAKER_COLORS_DARK if is_dark(widget) else SPEAKER_COLORS_LIGHT


def comfortable(*widgets):
    """Stop combo boxes and fields clipping descenders.

    Qt sizes these from the style's own metrics, which several themes get
    wrong for the fonts people actually use.
    """
    for widget in widgets:
        height = widget.fontMetrics().height()
        widget.setMinimumHeight(height + 14)
    return widgets[0] if len(widgets) == 1 else widgets
