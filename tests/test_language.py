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

import pytest

from meetnotes import asr
from meetnotes.config import Config


class FakeModel:
    def __init__(self, probs):
        self.probs = probs
        self.calls = []

    def detect_language(self, **kwargs):
        self.calls.append(kwargs)
        best = max(self.probs, key=lambda p: p[1])
        return best[0], best[1], self.probs


def restrict_cfg(codes):
    cfg = Config()
    cfg.asr.language_mode = "restrict"
    cfg.asr.languages = list(codes)
    return cfg


def test_primary_mode_pins_one_language():
    cfg = Config()
    cfg.asr.language_mode = "primary"
    cfg.asr.language = "fr"
    assert asr.language_args(cfg) == {"language": "fr"}


def test_auto_mode_lets_whisper_detect_per_segment():
    cfg = Config()
    cfg.asr.language_mode = "auto"
    cfg.asr.multilingual = True
    assert asr.language_args(cfg) == {"language": None, "multilingual": True}


def test_restrict_mode_uses_the_detected_language():
    assert asr.language_args(restrict_cfg(["fr", "en"]), "en") == {"language": "en"}


def test_restrict_picks_best_allowed_language_not_global_argmax():
    # Whisper is most confident about Welsh, which cannot occur in this meeting.
    model = FakeModel([("cy", 0.61), ("fr", 0.24), ("en", 0.10), ("de", 0.05)])
    assert asr.detect_restricted(model, None, restrict_cfg(["fr", "en"])) == "fr"


def test_restrict_prefers_english_when_english_leads_the_allowed_set():
    model = FakeModel([("cy", 0.5), ("en", 0.31), ("fr", 0.12)])
    assert asr.detect_restricted(model, None, restrict_cfg(["fr", "en"])) == "en"


def test_single_allowed_language_skips_detection_entirely():
    model = FakeModel([("en", 0.9)])
    assert asr.detect_restricted(model, None, restrict_cfg(["fr"])) == "fr"
    assert model.calls == []


def test_unlisted_language_gets_zero_weight():
    model = FakeModel([("de", 0.8)])
    assert asr.detect_restricted(model, None, restrict_cfg(["fr", "en"])) in ("fr", "en")


def test_detection_failure_falls_back_to_first_allowed():
    class Broken:
        def detect_language(self, **kwargs):
            raise RuntimeError("no features")

    assert asr.detect_restricted(Broken(), None, restrict_cfg(["fr", "en"])) == "fr"


def test_empty_allowed_set_returns_nothing():
    assert asr.detect_restricted(FakeModel([("fr", 1.0)]), None, restrict_cfg([])) == ""
