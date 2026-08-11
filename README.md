# meetnotes

Local meeting recorder. Captures your microphone and your computer's audio as
two separate tracks, transcribes both live, lets you type notes anchored to a
point in the recording, then writes a transcript, a summary, and a follow-up
list into a folder you own.

Nothing leaves the machine. Speech recognition runs in-process; summaries go to
a local model server.

Linux only for now. The capture layer is one module, so other platforms are
additive, but macOS has no monitor device and would need ScreenCaptureKit.

## Setup

If [uv](https://docs.astral.sh/uv/getting-started/installation/) is not already
installed:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Then:

```bash
sudo apt install git pipewire-bin pulseaudio-utils libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0
git clone https://github.com/ma-lalonde/meetnotes.git
cd meetnotes
chmod +x meetnotes
./meetnotes &
```

The first launch downloads dependencies and a speech model, so give it a few
minutes. After that it starts immediately.

No Python environment to create, no packages to install by hand.

### Summaries (optional)

Transcripts work with no extra setup. Summaries and action items need a local
model server:

1. Install [LM Studio](https://lmstudio.ai/)
2. Download a model from [lmstudio.ai/models](https://lmstudio.ai/models) or the
   app's own search. Anything from about 7B upward; smaller models fail at the
   structured output that action items rely on.
3. Start its server: **Developer > Start Server**, or `lms server start`
4. In meetnotes: **Models** tab, pick the LM Studio preset, **Fetch models**,
   choose one, **Save**

[Ollama](https://ollama.com/), Open WebUI, llama.cpp, and vLLM presets are
there too. Without a server, meetnotes still writes every transcript and simply
records that the summary was skipped.

## Using it

Pick a microphone and a system audio source on the **Record** tab; the meters
next to them move when audio is arriving. Start recording. Type notes as the
meeting goes; each is stamped against the moment you pressed Enter. Stop.

Everything else is automatic. Closing the window keeps it running in the tray.

One folder per meeting:

```
transcription_cleaned_with_notes.md
                          cleaned transcript, notes interleaved in time
summary.md                Context / Key points / Decisions / Open questions
actions.md                readable follow-ups
actions.json              structured, drives the calendar files
calendar/*.ics            one VTODO or VEVENT per dated action

raw_output/
  transcription.md        verbatim, timecoded, one heading per speaker
  transcription_cleaned.md
                          fillers removed, nothing else changed
  notes.md                your notes with timecodes

audio/*.wav               the source recordings
meeting.json              state, notes, segments, artifact fingerprints
```

## If something is wrong

```bash
./meetnotes doctor    # hardware, models, missing tools, environment
./meetnotes gpu       # every check that decides GPU acceleration
./meetnotes probe     # which capture backend actually records audio
```

Settings > Diagnostics shows all of it in one pasteable block.

### No audio on the participants track

Play something, then:

```bash
./meetnotes probe --save
```

It tries `pw-record`, `pw-record --capture-sink`, `parec`, and `ffmpeg -f pulse`
against your sources and keeps whichever produced signal.

System audio comes from **sinks**, not from a PulseAudio `.monitor` source.
PulseAudio invents a `<sink>.monitor` source that has no matching PipeWire node,
so targeting it silently falls back to the default input - your microphone lands
on the participants track while your own stays empty.

### Slow or inaccurate transcription

`./meetnotes gpu`. A GPU that shows up in `nvidia-smi` but reports `0` CUDA
devices means the CUDA libraries are missing: `./meetnotes gpu --install`, or the
button in the Models tab.

Without a GPU the live model drops to `small`, which is slow and weak. The saved
transcript still comes from the full-file pass and is far better than what
scrolls by live.

### Bilingual meetings

Whisper decodes one language per call and picks among all 99, so a mumbled
passage can come back as Welsh. Models tab, or:

```bash
./meetnotes language fr+en   # only these two, chosen per passage (default)
./meetnotes language fr      # mainly French, English words left in place
./meetnotes language auto    # anything
```

### uv reinstalls everything on every launch

Use `./meetnotes`, not `uv run meetnotes`. uv keeps its environment in `./.venv`
and validates it by resolving `.venv/bin/python`. On exFAT or NTFS - most
VeraCrypt containers and many external drives - that symlink cannot exist, so uv
deletes and reinstalls all 33 packages every time, PySide6 included. The
launcher puts the environment on a normal filesystem first.

To keep using `uv run` directly, export the same things yourself:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/meetnotes/venv"
export UV_LINK_MODE=copy
```

## Notes on how it works

**Two tracks, no diarization.** Recording microphone and system output
separately means the speaker is known from which file the audio came from. For
a one-on-one call that is exact and free. Diarization is a documented seam in
`asr.py`, needed only when several people share one stream.

**Two passes.** The live pass works on 6-second windows with no cross-window
context, which is enough to read along and anchor notes. The final pass re-runs
the whole recording afterwards and produces the transcript that gets saved. On a
GPU both use the same `large-v3-turbo` weights, loaded once.

**Notes are the best summary input.** `transcription_cleaned_with_notes.md` is
what the summary reads. A note only means something next to the speech that
prompted it; appended at the end, the model has to re-align timecodes itself and
does it badly. The prompt states that NOTE lines were never spoken aloud.

A note typed mid-sentence **splits that sentence**, so it sits between what
prompted it and what followed rather than after the whole utterance. The final
pass records word timestamps, which makes two rules possible:

- **Cut at a clause end.** A comma or full stop near the note is preferred over
  whatever word happened to be finishing. The search looks up to 2.5s back but
  only 1s forward, so a note is never placed ahead of words that had not been
  spoken when it was typed.
- **Do not cut off scraps.** If either side would be under a quarter of the
  sentence, no cut is made. The sentence stays whole and the note goes on the
  side where most of it was said - before it if the note came early, after it if
  late. A one-word fragment reads worse than a sentence slightly out of place.

Where a cut does happen, the remainder is prefixed with `...`.

**Re-running is safe.** Each artifact records a fingerprint of its inputs and a
hash of what was written. Unchanged inputs are skipped, changed inputs
regenerate, and **a file you edited by hand is never overwritten** without
`--force`. Writes are atomic and a per-meeting lock stops the automatic run and
a manual one colliding.

**Raw inputs are kept apart.** The three files under `raw_output/` are the
unmerged pieces; the four at the top level are what you normally read. Meetings
recorded before this layout are moved on the next run, hand edits included.

**Cleaning is deterministic.** `raw_output/transcription_cleaned.md` comes from a filler-word
rule set, never from a language model. The verbatim transcript has to stay
trustworthy. An LLM rewrite is available per meeting, never automatic.

**GPU memory.** Speech recognition runs in a child process, because CTranslate2
allocates through a caching allocator that never returns VRAM to the driver
while the process lives; its exit is the only release. Language models are
unloaded before a load and before recording starts (`lms unload --all`), the
speech model is released before summarizing, and requests carry a `ttl` so LM
Studio drops the model when idle.

**One model at a time, with no way to ask for two.** The passes never need the
card together, and holding one while the other loads is the entire reason
summarizing ran out of memory. There is no setting to keep either resident,
because the only thing such a setting can do is reproduce that failure.

**Choosing models.** Free VRAM decides more than it should: freeing a gigabyte
by switching NVIDIA PRIME to on-demand can be the difference between a 4k
context and a usable one. Rather than leave that to be discovered as `failed to
load model`:

```
meetnotes tune                # size models against free VRAM
meetnotes tune --record 30    # record a sample and time each model on it
meetnotes tune --apply        # keep the result
```

Speech models are **timed**, because whether one keeps up with live speech
depends on the machine and not on any published number. Language models are
**estimated**, via `lms load --estimate-only`, which reports what a load would
cost without loading it. The Models tab has the same thing as **Choose for this
machine**.

**Retired: never offered, at any setting.** A model another one beats without
costing meaningfully more is not the right answer on any hardware or in any
language, so nothing brings it back:

| Retired | Why |
| --- | --- |
| `large-v1`, `large-v2` | Turbo matches large-v2 at half the size |
| `medium` | Turbo is 5% larger and a whole tier better |
| `medium.en` | `distil-large-v3.5` is smaller and better |
| `distil-large-v2`, `distil-large-v3` | superseded by `v3.5` |

A config that still names one keeps working; it just is not offered.

Equal accuracy only displaces at strictly lower cost; a genuine accuracy gain
displaces within a 10% size margin. The frontier is computed from size and
accuracy rank rather than hand-listed, so adding a model cannot leave a stale
recommendation behind, and a test asserts the surviving models stay more than
that margin apart so the rule cannot cascade.

**The list also follows the hardware.** On a GPU, speed is never the binding
constraint. Whisper's own figures put Turbo at ~8x against Tiny's ~10x, so a
card with room for Turbo gains a quarter of the time from the small models and
loses several tiers of accuracy for it. Memory is the only real limit:

| Card | Offered |
| --- | --- |
| 8 GB | Large v3, Turbo |
| 4 GB | Turbo |
| 2 GB | Small, Base, Tiny |
| CPU | Large v3, Turbo, Small, Base, Tiny |

Sized against **the card**, not what is free at that moment. Recognition runs in
its own process, after the language model has been unloaded, so an 8 GB card is
an 8 GB card whether or not LM Studio is holding six of them. A list that
changed because another application was resident when the tab opened would be
the same friction in a new place.

On CPU the limits are absolute rather than relative, and the two passes differ:

| Pass | Offered on CPU |
| --- | --- |
| Live | Base, Tiny |
| Final | Turbo, Small, Base, Tiny |

Only the two smallest reliably transcribe faster than speech arrives, and a
live pass that falls behind never catches up. Large v3 is not offered for the
final pass either: at 1x relative speed an hour of audio takes longer than the
meeting did. `meetnotes tune --record 30` settles the rest by measurement, and
still respects both limits, since one quiet sample can time well on a model
that falls behind in a real meeting.

The hardware and language rules are **situational** - right on one machine,
wrong on another - so the Models tab can waive both with *Show models this
machine or language setting rules out*. Retirement is not situational and is
never waived.

**The list follows the language setting.** English-only (`.en`) models have the
same parameter count as the multilingual models they were cut from and are
better at English for it - Whisper's README: they "tend to perform better,
especially for the `tiny.en` and `base.en` models". They also cannot transcribe
anything else. So:

| Language setting | Offered |
| --- | --- |
| French and English | Large v3, Turbo, Medium, Small, Base, Tiny |
| English only | Large v3, Turbo, Distil v3.5 EN, Small EN, Base EN, Tiny EN |

On an English-only setup the `.en` builds *replace* their twins rather than
sitting beside them, since at equal size the twin is simply worse at the only
language being spoken. Turbo and Large v3 have no `.en` build and stay either
way. Only those exact same-size pairs are compared across the two classes;
relative English accuracy between, say, Distil v3.5 and Turbo is not published
anywhere, so neither displaces the other.

**Precision is two choices, not seven.** CTranslate2 accepts `float32`,
`float16`, `bfloat16`, `int8`, `int8_float16`, `int8_bfloat16` and
`int8_float32`, but the `int8_*` variants differ only in the precision of the
layers that are *not* quantized, which is not audible on a speech model. What is
offered:

| Device | Offered |
| --- | --- |
| GPU | Balanced (`float16`), Half memory (`int8_float16`) |
| CPU | Smallest (`int8`), Exact (`float32`) |

`float16` and `bfloat16` are **not offered on CPU**: CTranslate2 converts them
to `float32` there, so picking one costs `float32` memory and gains nothing.
Nothing unsupported on the current device is listed either, since it would
silently fall back to something else. A type already set in the config is kept
even when it is outside this list.

**Why not the 500 MB GGUF whisper builds.** CTranslate2, the engine behind
faster-whisper, converts from Fairseq, Marian, OpenNMT and Transformers into its
own `model.bin`; GGUF is llama.cpp's format and is not among them. Separately,
LM Studio has no `/v1/audio/transcriptions` endpoint, so it cannot serve a
speech model at all. The size saving is available here as **Precision**: `int8`
roughly halves `float16` at little cost on turbo.

## Limitations

- **Wear headphones.** No echo cancellation. On speakers the microphone picks up
  the far end and both tracks transcribe the same speech.
- GNOME needs the AppIndicator extension for a tray icon. The window works
  regardless; a banner explains it.
- Group calls where several people share one stream would need diarization,
  which is not wired up.

## Development

```bash
./meetnotes ui --check      # build every screen offscreen, then exit
uv run pytest tests/
```

## License

Copyright (C) 2026 Marc-Antoine Lalonde

GNU General Public License v3.0 or later. See [LICENSE](LICENSE), or run
`./meetnotes version`.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
