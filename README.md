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

Two commands.

```bash
sudo apt install pipewire-bin pulseaudio-utils libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0
./meetnotes
```

The first launch downloads dependencies and a speech model, so give it a few
minutes. After that it starts immediately.

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/). Nothing
else - no Python environment to create, no packages to install by hand.

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
transcription.md          verbatim, timecoded, one heading per speaker
transcription_cleaned.md  fillers removed, nothing else changed
transcription_cleaned_with_notes.md
                          the above with your notes interleaved in time
notes.md                  your notes with timecodes
summary.md                Context / Key points / Decisions / Open questions
actions.md                readable follow-ups
actions.json              structured, drives the calendar files
calendar/*.ics            one VTODO or VEVENT per dated action
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

**Re-running is safe.** Each artifact records a fingerprint of its inputs and a
hash of what was written. Unchanged inputs are skipped, changed inputs
regenerate, and **a file you edited by hand is never overwritten** without
`--force`. Writes are atomic and a per-meeting lock stops the automatic run and
a manual one colliding.

**Cleaning is deterministic.** `transcription_cleaned.md` comes from a filler-word
rule set, never from a language model. The verbatim transcript has to stay
trustworthy. An LLM rewrite is available per meeting, never automatic.

**GPU memory.** The speech model is unloaded before the language model is
called, language models are unloaded before recording starts (`lms unload --all`),
and requests carry a `ttl` so LM Studio drops the model when idle. At most one
model is resident at a time.

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
