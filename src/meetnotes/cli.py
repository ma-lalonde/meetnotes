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

import argparse
import sys
import time
from pathlib import Path

from . import audio, hardware, pipeline, store
from .config import CONFIG_PATH, Config


def cmd_doctor(cfg, args) -> int:
    from . import models

    info = hardware.report(cfg)
    venv = hardware.venv_health()
    extra = [
        ("config", CONFIG_PATH),
        ("tray", hardware.tray_available()),
        # Named because a huggingface_hub warning during the first
        # transcription is otherwise the first anyone hears of where the
        # weights come from.
        ("speech models", f"huggingface.co -> {models.cache_dir()}"),
        ("venv", venv["venv"]),
        ("symlinks", venv["symlinks_supported"]),
    ]
    width = max(len(k) for k in list(info) + [k for k, _ in extra])
    for key, value in info.items():
        print(f"{key.ljust(width)}  {value}")
    for key, value in extra:
        print(f"{key.ljust(width)}  {value}")

    if venv["rebuilds_every_launch"]:
        print(
            "\nThis filesystem does not support symlinks, and the environment lives\n"
            "inside the project, so uv deletes and reinstalls every package on each\n"
            "launch. That is the startup freeze. Launch with ./meetnotes instead of\n"
            "'uv run meetnotes', or export:\n"
            "  UV_PROJECT_ENVIRONMENT=\"$HOME/.local/share/meetnotes/venv\"\n"
            "  UV_LINK_MODE=copy"
        )

    if not info["pw_record"]:
        print("\nmissing pw-record: install pipewire-utils (Debian: pipewire-bin)")
    if not info["pactl"]:
        print("missing pactl: install pulseaudio-utils")
    state = hardware.cuda_state()
    print(f"\n{state['detail']}")
    if state["installable"]:
        print("Install it with:  ./meetnotes gpu --install   (or the button in Settings)")
    elif not state["gpus"]:
        plan = hardware.plan(cfg)
        print(
            f"On CPU the live model is '{plan['live_model']}', which is fast enough to "
            f"keep up\nand noticeably less accurate. The final pass uses "
            f"'{plan['final_model']}', so the\nsaved transcript is much better than the "
            f"live one."
        )
    return 0


def cmd_gpu(cfg, args) -> int:
    from . import asr

    rows = hardware.cuda_diagnostics()
    plan = hardware.plan(cfg)
    rows.append(("recognition device", plan["device"]))
    rows.append((
        "recognition process",
        "child process, VRAM freed on exit" if asr.isolated(cfg, plan)
        else "in this process",
    ))
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key.ljust(width)}  {value}")

    procs = hardware.gpu_processes()
    if procs:
        print("\nholding VRAM now:")
        for proc in procs:
            print(f"  {proc['used_mb']:>6} MB  {proc['name']} (pid {proc['pid']})")
    state = hardware.cuda_state()
    print(f"\n{state['detail']}")
    if not args.install:
        if state["installable"]:
            print("\nrun with --install to add CUDA support (about 1.4 GB)")
        return 0
    if not state["installable"]:
        # Say which of the three reasons it is, rather than leaving the user to
        # guess why an install they were told to run declined to do anything.
        if not state["gpus"]:
            print("\nNothing to install: no NVIDIA GPU is present.")
        elif state["usable"]:
            print("\nNothing to install: GPU acceleration is already working.")
        else:
            print(
                "\nNothing to install: cuBLAS loads, so the libraries are there.\n"
                "The driver is what does not match. Check `nvidia-smi` against the\n"
                "CUDA version ctranslate2 was built for."
            )
        return 0
    print("\ninstalling the CUDA libraries (about 1.4 GB)...")
    if not hardware.install_cuda(log=print):
        return 1
    # The check is cached, and it ran before the libraries existed.
    hardware.cuda_runtime_ok(refresh=True)
    print(f"\n{hardware.cuda_state()['detail']}")
    return 0


def cmd_sources(cfg, args) -> int:
    sources = audio.list_sources()
    if not sources:
        print("no sources found (is pactl installed and a session running?)")
        return 1

    changed = False
    if args.auto:
        mic, system = audio.default_sources()
        cfg.capture.mic_source, cfg.capture.system_source = mic, system
        changed = True
    for value, kind, key in (
        (args.mic, "mic", "mic_source"),
        (args.system, "system", "system_source"),
    ):
        if value is None:
            continue
        try:
            setattr(cfg.capture, key, audio.resolve(value, kind))
        except ValueError as exc:
            print(exc)
            return 1
        changed = True

    if changed:
        cfg.save()

    for source in sources:
        selected = source.target in (cfg.capture.mic_source, cfg.capture.system_source)
        role = "-> Me" if source.target == cfg.capture.mic_source else (
            "-> Participants" if source.target == cfg.capture.system_source else ""
        )
        print(f"{'*' if selected else ' '} [{source.kind:6}] {source.label}  {role}")
        print(f"             {source.name}  (target {source.target})")
    if changed:
        print(f"\nsaved to {CONFIG_PATH}")
    elif not (cfg.capture.mic_source or cfg.capture.system_source):
        print("\nnothing selected yet. Pick sources with:")
        print("  ./meetnotes sources --auto")
        print("  ./meetnotes sources --mic <text> --system <text>")
    return 0


def _bar(dbfs: float, width: int = 34) -> str:
    filled = 0 if dbfs <= -60 else min(width, int((dbfs + 60) / 60 * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {dbfs:6.1f} dBFS"


def cmd_probe(cfg, args) -> int:
    """Measure which capture backend actually produces audio for each source."""
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="meetnotes-probe-"))
    chosen = {}
    for kind, key in (("mic", "mic_source"), ("system", "system_source")):
        target = getattr(cfg.capture, key)
        source = next((s for s in audio.list_sources() if s.target == target), None)
        if source is None:
            print(f"{kind}: nothing selected, run './meetnotes sources --auto' first\n")
            continue

        print(f"{kind}: {source.label}")
        if kind == "system":
            print("  play some audio now")
        best = None
        for name in audio.candidate_backends(kind):
            template = audio.RECORD_BACKENDS[name]
            cmd = audio.build_command(
                template, source.target, source.pulse_name, cfg.capture.sample_rate,
                work / f"{kind}-{name}.wav",
            )
            result = audio.measure(cmd, work / f"{kind}-{name}.wav", args.seconds)
            verdict = "silent"
            if not result["ok"]:
                verdict = f"failed: {result['error'][:60]}"
            elif result["peak"] > -50:
                verdict = "AUDIO"
                if best is None:
                    best = (name, template)
            print(f"  {name:16} {verdict:12} peak {result['peak']:6.1f} dBFS")
        if best:
            chosen[kind] = best
            print(f"  -> {best[0]} works\n")
        else:
            print("  -> nothing captured audio\n")

    if not chosen:
        print("No backend captured audio. Was anything playing during the system test?")
        return 1
    pick = chosen.get("system") or chosen.get("mic")
    if args.save:
        cfg.capture.record_cmd = pick[1]
        cfg.save()
        print(f"saved record_cmd = {pick[1]}")
    else:
        print(f"re-run with --save to store:\n  {pick[1]}")
    return 0


def cmd_levels(cfg, args) -> int:
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="meetnotes-levels-"))
    sources = {s.target: s for s in audio.list_sources()}
    meters = {}
    levels = {}
    for label, key in ((cfg.capture.mic_label, "mic_source"),
                       (cfg.capture.system_label, "system_source")):
        source = sources.get(getattr(cfg.capture, key))
        if source is None:
            continue
        levels[label] = -120.0
        meters[label] = audio.Meter(
            cfg.capture.record_cmd, cfg.capture.sample_rate, source, work,
            lambda db, name=label: levels.__setitem__(name, db),
        )
    if not meters:
        print("no sources selected: ./meetnotes sources --auto")
        return 1

    for meter in meters.values():
        meter.start()
    print("ctrl-c to stop\n")
    try:
        while True:
            line = "   ".join(f"{name} {_bar(levels[name])}" for name in meters)
            print("\r" + line, end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print()
    finally:
        for meter in meters.values():
            meter.stop()
        for label, meter in meters.items():
            meter.join(timeout=3)
            if meter.error:
                print(f"{label}: {meter.error}")
    return 0


NOTICE = """meetnotes {version}
Copyright (C) 2026 Marc-Antoine Lalonde

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under the terms
of the GNU General Public License version 3 or later. See the LICENSE file, or
<https://www.gnu.org/licenses/gpl-3.0.html>."""


def cmd_version(cfg, args) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("meetnotes")
    except PackageNotFoundError:
        release = "development"
    print(NOTICE.format(version=release))
    return 0


def cmd_ui(cfg, args) -> int:
    from .app import run

    return run(cfg, check=args.check, platform=args.platform)


def cmd_prompt(cfg, args) -> int:
    """Show exactly what would be sent to the language model."""
    from . import outputs, prompts

    path = Path(args.meeting).expanduser().resolve()
    if not store.is_meeting(path):
        print(f"not a meeting folder: {path}")
        return 1

    meta = store.read_meta(path)
    segments = meta.get("segments", [])
    notes = meta.get("notes", [])
    spoken = [s for s in segments if outputs.clean_text(s.get("text", ""))]

    print(f"segments stored      {len(segments)}")
    print(f"segments with speech {len(spoken)}")
    print(f"notes                {len(notes)}")
    print(f"tracks               {meta.get('tracks', {})}")
    print(f"state                {meta.get('state')}  {meta.get('error', '')}")
    print(f"model                {cfg.llm.model or '(none selected)'} at {cfg.llm.base_url}")

    system = cfg.llm.actions_prompt if args.which == "actions" else cfg.llm.summary_prompt
    user = outputs.transcript_for_llm(meta, segments)
    # Roughly four characters per token for English and French.
    estimate = (len(system) + len(user)) // 4
    print(f"\nsystem prompt        {len(system)} characters")
    print(f"transcript sent      {len(user)} characters")
    print(f"estimated tokens     ~{estimate}")

    from . import llm

    resident, maximum = _model_context(cfg)
    wanted = llm.required_context(len(system) + len(user))
    print(f"context needed       {wanted}")
    if maximum:
        print(f"model context limit  {maximum}")
    else:
        print("model context limit  unknown (server does not report it)")
    if resident:
        print(f"loaded right now     {resident}")
    # Only the loaded size constrains the request; the maximum is what it could
    # be reloaded at.
    effective = resident or maximum
    if effective and estimate > effective * 0.75:
        print(
            f"\nTOO LONG FOR THE CONTEXT WINDOW: ~{estimate} tokens of input against a\n"
            f"{effective}-token limit. LM Studio truncates the prompt, leaving the model\n"
            "no room to answer, which comes back as an empty summary."
        )
        if cfg.llm.auto_context:
            print("Post-processing will reload the model at a larger size automatically.")
        else:
            print("Turn on auto_context, or reload the model with a larger context.")

    gpus = hardware.nvidia()
    if gpus:
        gpu = gpus[0]
        print(f"gpu free             {gpu.get('free_mb', 0)} MB of {gpu['vram_mb']} MB")
    if not spoken:
        print("\nNOTHING TO SUMMARIZE: no transcript segments carry any speech.")
        print("The language model is not the problem; transcription produced nothing.")

    if args.full:
        print("\n" + "=" * 60 + "\nSYSTEM\n" + "=" * 60)
        print(system)
        print("\n" + "=" * 60 + "\nUSER\n" + "=" * 60)
        print(user)
    else:
        print("\nfirst 800 characters of what would be sent:\n")
        print(user[:800] or "(empty)")
        print("\nre-run with --full to see all of it")
    return 0


def _model_context(cfg) -> tuple[int, int]:
    """(context the instance is loaded with, the model's maximum).

    The two differ, and only the first one constrains a request. Reporting the
    maximum alone is how a model loaded at 4096 looks like it has 131072.
    """
    from . import llm

    if not cfg.llm.model:
        return 0, 0
    try:
        for entry in llm.catalog(cfg):
            if entry.get("id") == cfg.llm.model:
                resident = int(entry.get("loaded_context") or 0)
                if entry.get("state") != "loaded":
                    resident = 0
                return resident, int(entry.get("context") or 0)
    except Exception:
        pass
    return 0, 0


def cmd_llm_check(cfg, args) -> int:
    """Exercise the language model server and show exactly what it returns."""
    import json as jsonlib

    import httpx

    from . import llm

    base = cfg.llm.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.llm.api_key or 'none'}"}
    print(f"base url   {base}")
    print(f"model      {cfg.llm.model or '(none selected)'}")

    try:
        names = llm.list_models(cfg)
        print(f"models     {len(names)} available")
        if cfg.llm.model and cfg.llm.model not in names:
            print(f"  WARNING: '{cfg.llm.model}' is not in the list. Available:")
            for name in names[:15]:
                print(f"    {name}")
    except llm.LlmError as exc:
        print(f"models     FAILED: {exc}")
        print(
            "\nLM Studio checklist:\n"
            "  1. Developer tab > Status: the server must be Running\n"
            "  2. the port must match the base URL above (default 1234)\n"
            "  3. 'Serve on Local Network' is not required for localhost"
        )
        return 1

    if not cfg.llm.model:
        print("\nNo model selected. Models tab, or edit config.json.")
        return 1

    resident, maximum = _model_context(cfg)
    if maximum:
        print(f"context    {maximum} tokens maximum")
    if resident:
        print(f"           {resident} tokens as currently loaded")
    if resident and resident < 8192:
        print(
            "  A meeting transcript rarely fits in this. Load the model with a\n"
            "  larger context length in LM Studio, 16384 or more, or let\n"
            "  auto_context reload it for you."
        )
    gpus = hardware.nvidia()
    if gpus:
        free = gpus[0].get("free_mb", 0)
        print(f"gpu        {free} MB free of {gpus[0]['vram_mb']} MB")
        procs = hardware.gpu_processes()
        if procs:
            print("holding    " + f"{procs[0]['used_mb']} MB  {procs[0]['name']} (pid {procs[0]['pid']})")
            for proc in procs[1:]:
                print(f"           {proc['used_mb']} MB  {proc['name']} (pid {proc['pid']})")
        if free < 2000:
            print(
                "\n  NOT ENOUGH FREE VRAM to load a language model. Whatever is listed\n"
                "  above is holding the card. LM Studio keeps several models resident\n"
                "  at once: `lms ps` lists them, `lms unload --all` evicts them."
            )
    lms = llm.lms_binary()
    print(f"lms CLI    {lms or 'NOT FOUND'}")
    if lms:
        for line in llm.loaded()[:12]:
            print(f"  lms ps   {line}")
        free = llm.free_vram_mb()
        print("\ncost of this model at each context size, without loading it:")
        for size in llm.CONTEXT_STEPS:
            cost, note = llm.estimate_load(cfg.llm.model, size, cfg.llm.gpu_offload)
            if not cost:
                print(f"  {size:>6}  no estimate: {note.splitlines()[0][:80] if note else 'none'}")
                break
            verdict = "fits" if not free or cost <= free else f"NEEDS {cost - free} MB MORE"
            print(f"  {size:>6}  {cost:>6} MB   {verdict}")
    else:
        print(
            "\n  Without it meetnotes cannot set the context size or unload a\n"
            "  model, so LM Studio loads at its own default (often 4096) and a\n"
            "  second copy stacks on the first. lms ships inside LM Studio; run\n"
            "  ~/.lmstudio/bin/lms bootstrap once, then restart meetnotes."
        )

    payload = {
        "model": cfg.llm.model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 2000,
    }

    print("\n--- plain request ---")
    try:
        with httpx.Client(timeout=cfg.llm.timeout) as client:
            resp = client.post(base + "/chat/completions", json=payload, headers=headers)
        print(f"status     {resp.status_code}")
        body = resp.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        print(f"content    {message.get('content')!r}")
        if not (message.get("content") or "").strip():
            for key in ("reasoning_content", "reasoning"):
                if message.get(key):
                    print(
                        f"\n  The answer came back under '{key}' with content empty.\n"
                        "  This model spends its budget reasoning and never emits an answer.\n"
                        "  Pick a non-reasoning model, or raise its context length."
                    )
            print(f"\n  finish_reason: {choice.get('finish_reason')}")
            print(f"  usage: {body.get('usage')}")
            print(f"\n  raw first choice:\n{jsonlib.dumps(choice, indent=2)[:900]}")
    except (httpx.HTTPError, ValueError) as exc:
        print(f"           FAILED: {exc}")
        return 1

    print("\n--- streaming request ---")
    try:
        with httpx.Client(timeout=cfg.llm.timeout) as client:
            with client.stream(
                "POST", base + "/chat/completions",
                json={**payload, "stream": True}, headers=headers,
            ) as resp:
                print(f"status     {resp.status_code}")
                shown = 0
                for line in resp.iter_lines():
                    if line.strip() and shown < 6:
                        print(f"  {line[:160]}")
                        shown += 1
                print(f"  ...{shown} lines shown")
    except httpx.HTTPError as exc:
        print(f"           FAILED: {exc}")
    return 0


def cmd_unload(cfg, args) -> int:
    """Try every route to free the GPU, and report what each one did."""
    import json as jsonlib

    import httpx

    from . import llm

    base = cfg.llm.base_url.rstrip("/")
    print(f"base url   {base}")
    print(f"lms CLI    {llm.lms_binary() or 'NOT FOUND'}")

    for label, url in (
        ("/api/v1/models", llm._v1(cfg, "models")),
        ("/api/v0/models", base[: -len('/v1')] + "/api/v0/models"
         if base.endswith("/v1") else base + "/models"),
    ):
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, headers={"Authorization": f"Bearer {cfg.llm.api_key or 'none'}"})
            print(f"\n{label}  ->  HTTP {resp.status_code}")
            if resp.status_code == 200 and args.full:
                print(jsonlib.dumps(resp.json(), indent=2)[:1500])
        except httpx.HTTPError as exc:
            print(f"\n{label}  ->  FAILED: {exc}")

    print("\nloaded instances seen over REST:")
    instances = llm.rest_instances(cfg)
    for instance in instances:
        print(f"  {instance['id']}  context {instance['context']}  "
              f"{int(instance['size_bytes'] / (1024 * 1024))} MB")
        for key, value in sorted((instance.get("config") or {}).items()):
            if key != "context_length":
                print(f"      {key}: {value}")
    if not instances:
        print("  none reported")

    for line in llm.loaded()[:12]:
        print(f"  lms ps   {line}")

    before = llm.free_vram_mb()
    print(f"\nfree VRAM before  {before} MB")
    cleared, detail = llm.unload_everything(cfg, log=lambda line: print(f"  {line}"))
    print(f"\nresult     {'cleared' if cleared else 'NOT CLEARED'}")
    print(f"           {detail}")
    print(f"free VRAM after   {llm.free_vram_mb()} MB")

    procs = hardware.gpu_processes()
    if procs:
        print("\nstill holding VRAM:")
        for proc in procs:
            print(f"  {proc['used_mb']:>6} MB  {proc['name']} (pid {proc['pid']})")
    return 0 if cleared else 1


def cmd_tune(cfg, args) -> int:
    """Measure what this machine can actually run, and optionally keep it."""
    import tempfile

    from . import models, tuning

    sample = Path(args.sample).expanduser() if args.sample else None
    if args.record:
        source = audio.resolve(cfg.capture.mic_source, "mic")
        if source is None:
            print("No microphone selected. Run: ./meetnotes sources --auto")
            return 1
        work = Path(tempfile.mkdtemp(prefix="meetnotes-tune-"))
        sample = work / "sample.wav"
        print(f"Recording {args.record:.0f}s from {source.label}. Speak normally.")
        recorder = audio.Recorder(
            cfg.capture.record_cmd, cfg.capture.sample_rate, {"sample": source}, work
        )
        recorder.start()
        time.sleep(args.record)
        recorder.stop()
        sample = recorder.paths["sample"]
        print(f"Sample written to {sample}\n")

    if sample and not sample.exists():
        print(f"No such sample: {sample}")
        return 1
    if not sample:
        print(
            "No sample given, so speech models are sized rather than timed.\n"
            "For a real measurement: ./meetnotes tune --record 30\n"
        )

    plan = tuning.tune(cfg, sample, log=lambda line: print(f"  {line}"))

    print()
    if plan.measurements:
        print("speech models, timed on this machine:")
        for found in plan.measurements:
            if found.error:
                print(f"  {found.label:<12} FAILED  {found.error[:70]}")
                continue
            verdict = "keeps up live" if found.keeps_up else "final pass only"
            print(
                f"  {found.label:<12} {found.realtime:>6.2f}x realtime  "
                f"~{found.vram_mb} MB  {verdict}"
            )
        print()

    print(f"live model     {models.label(plan.live)} ({plan.live})")
    print(f"final model    {models.label(plan.final)} ({plan.final})")
    print(f"device         {plan.device} ({plan.compute_type})")
    if plan.summary_model:
        print(f"summary model  {plan.summary_model} at {plan.summary_context} tokens")
    else:
        print("summary model  not chosen")
    for note in plan.notes:
        print(f"  note: {note}")

    if args.apply:
        tuning.apply(plan, cfg)
        print(f"\nSaved to {CONFIG_PATH}")
    else:
        print("\nRe-run with --apply to keep this.")
    return 0


def cmd_vocabulary(cfg, args) -> int:
    terms = list(cfg.asr.vocabulary)
    if args.clear:
        terms = []
    for term in args.remove:
        terms = [t for t in terms if t.casefold() != term.casefold()]
    for term in args.add:
        if term.strip() and term.casefold() not in {t.casefold() for t in terms}:
            terms.append(term.strip())

    if args.add or args.remove or args.clear:
        cfg.asr.vocabulary = terms
        cfg.save()

    if terms:
        for term in terms:
            print(f"  {term}")
    else:
        print("no expected names set")
    print(
        "\nThese are given to the recogniser as hints, which is what stops proper\n"
        "nouns coming back garbled. Add with:\n"
        '  meetnotes vocabulary --add "Chloe Gagnon" Catena Portainer'
    )
    return 0


def cmd_language(cfg, args) -> int:
    if args.spec:
        spec = args.spec.strip().lower()
        if spec == "auto":
            cfg.asr.language_mode, cfg.asr.languages, cfg.asr.language = "auto", [], ""
        elif "+" in spec:
            codes = [c for c in spec.split("+") if c]
            cfg.asr.language_mode, cfg.asr.languages, cfg.asr.language = "restrict", codes, codes[0]
        else:
            cfg.asr.language_mode, cfg.asr.languages, cfg.asr.language = "primary", [spec], spec
        cfg.save()

    mode = cfg.asr.language_mode
    if mode == "primary":
        print(f"mainly {cfg.asr.language}: pinned, foreign words transcribed where they fall")
    elif mode == "restrict":
        print(f"restricted to {', '.join(cfg.asr.languages)}: detected per passage, nothing else")
    else:
        print("automatic: any of Whisper's languages, detected per segment")
    print("\nset with:  meetnotes language fr+en | fr | en | auto")
    return 0


def cmd_record(cfg, args) -> int:
    from .session import Session

    session = Session(
        cfg,
        on_segment=lambda s: print(f"[{s['start']:7.1f}] {s['speaker']}: {s['text']}", flush=True),
        on_state=lambda state, detail: print(f"-- {state}: {detail}", flush=True),
    )
    path = session.start(args.title)
    print(f"recording to {path}")
    try:
        if args.seconds:
            time.sleep(args.seconds)
        else:
            print("press ctrl-c to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print()
    path = session.stop(post_process=False)
    if args.no_process:
        print(f"\nno markdown written (--no-process). Generate it with:\n  ./meetnotes process {path}")
        return 0
    report = pipeline.process(path, cfg, progress=_print_step)
    _print_report(path, report)
    return 0


def _print_step(step: str, fraction: float | None = None) -> None:
    percent = f"{fraction * 100:3.0f}%" if fraction is not None else "  --"
    print(f"\r{percent}  {step:<48}", end="", flush=True)
    if step == "done":
        print()


def _print_report(path, report: dict) -> None:
    run = store.read_meta(path).get("run", {})
    if run:
        print(
            f"\nlive={run.get('live_model')} final={run.get('final_model')} "
            f"device={run.get('device')} compute={run.get('compute_type')} "
            f"gpu={run.get('gpu')}"
        )
    for name, action in report.items():
        print(f"{action.ljust(12)} {name}")
    print(f"\nfiles in {path}")


def cmd_process(cfg, args) -> int:
    path = Path(args.meeting).expanduser().resolve()
    if not store.is_meeting(path):
        print(f"not a meeting folder: {path}")
        return 1
    try:
        report = pipeline.process(
            path, cfg, force=args.force, with_llm=not args.no_llm,
            progress=_print_step,
        )
    except store.Busy as exc:
        print(f"busy: {exc}")
        return 1
    _print_report(path, report)
    return 0


def cmd_list(cfg, args) -> int:
    meetings = store.list_meetings(cfg.root)
    if not meetings:
        print(f"no meetings in {cfg.root}")
        return 0
    for meta in meetings:
        print(f"{meta['state'].ljust(12)} {meta['id']}  ({meta.get('duration', 0):.0f}s)")
    return 0


def cmd_recover(cfg, args) -> int:
    reset = store.recover(cfg.root)
    print(f"reset {len(reset)} interrupted meeting(s)" + (": " + ", ".join(reset) if reset else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="meetnotes", description=__doc__)
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("version", help="show the version and licence notice")
    subs.add_parser("doctor", help="report hardware, models, and missing tools")
    subs.add_parser("list", help="list meetings")
    subs.add_parser("recover", help="reset meetings interrupted by a crash")

    probe = subs.add_parser("probe", help="find which capture backend actually records audio")
    probe.add_argument("--seconds", type=float, default=2.5)
    probe.add_argument("--save", action="store_true", help="store the working command")

    lev = subs.add_parser("levels", help="live level meter for the selected sources")
    lev.add_argument("--seconds", type=float, default=0.0)

    ui = subs.add_parser("ui", help="launch the window (same as no subcommand)")
    ui.add_argument("--check", action="store_true", help="build every screen offscreen and exit")
    ui.add_argument("--platform", default="", help="override QT_QPA_PLATFORM, e.g. xcb or wayland")

    subs.add_parser("llm-check", help="test the language model server and show its reply")

    shown = subs.add_parser("prompt", help="show what would be sent to the language model")
    shown.add_argument("meeting")
    shown.add_argument("--which", choices=["summary", "actions"], default="summary")
    shown.add_argument("--full", action="store_true")

    vocab = subs.add_parser("vocabulary", help="names and jargon to expect in speech")
    vocab.add_argument("--add", nargs="+", default=[], help="terms to add")
    vocab.add_argument("--remove", nargs="+", default=[], help="terms to remove")
    vocab.add_argument("--clear", action="store_true")

    lang = subs.add_parser("language", help="show or set expected languages")
    lang.add_argument(
        "spec", nargs="?",
        help="fr+en (restrict), fr (mainly French), en (mainly English), auto",
    )

    gpu = subs.add_parser("gpu", help="report or install GPU acceleration")
    gpu.add_argument("--install", action="store_true", help="install the CUDA libraries")

    src = subs.add_parser("sources", help="list or select audio sources")
    src.add_argument("--mic", help="select a microphone by name or substring")
    src.add_argument("--system", help="select a system audio (.monitor) source")
    src.add_argument("--auto", action="store_true", help="pick the first of each")

    rec = subs.add_parser("record", help="record from the terminal")
    rec.add_argument("--title", default="meeting")
    rec.add_argument("--seconds", type=float, default=0.0)
    rec.add_argument("--no-process", action="store_true")

    proc = subs.add_parser("process", help="post-process a meeting folder")
    proc.add_argument("meeting")
    proc.add_argument("--force", action="store_true", help="ignore fingerprints and overwrite")
    proc.add_argument("--no-llm", action="store_true", help="transcripts only")

    unl = subs.add_parser("unload", help="free the GPU, showing every step")
    unl.add_argument("--full", action="store_true", help="print the raw server responses")

    tune = subs.add_parser("tune", help="measure this machine and choose models")
    tune.add_argument("--sample", help="a recording to time the speech models on")
    tune.add_argument(
        "--record", type=float, metavar="SECONDS", default=0.0,
        help="record a fresh sample from the microphone first",
    )
    tune.add_argument("--apply", action="store_true", help="save the result to the config")
    tune.add_argument(
        "--context", type=int, default=32768,
        help="context to aim for when choosing a language model (default 32768)",
    )

    args = parser.parse_args(argv)
    cfg = Config.load()

    if args.command is None:
        from .app import run

        return run(cfg)

    handlers = {
        "version": cmd_version,
        "doctor": cmd_doctor,
        "gpu": cmd_gpu,
        "probe": cmd_probe,
        "levels": cmd_levels,
        "language": cmd_language,
        "vocabulary": cmd_vocabulary,
        "prompt": cmd_prompt,
        "llm-check": cmd_llm_check,
        "ui": cmd_ui,
        "sources": cmd_sources,
        "record": cmd_record,
        "process": cmd_process,
        "unload": cmd_unload,
        "tune": cmd_tune,
        "list": cmd_list,
        "recover": cmd_recover,
    }
    return handlers[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
