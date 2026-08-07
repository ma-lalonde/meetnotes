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

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx


class LlmError(RuntimeError):
    pass


class LoadFailed(LlmError):
    """The model could not be loaded at any context size."""


class Truncated(LlmError):
    """The answer stopped because it ran out of room, not because it ended."""


class ContextTooSmall(LlmError):
    """The largest context that loads cannot hold this transcript.

    Sending it anyway is worse than failing: the server drops the front of the
    prompt without saying so, and the result reads like a summary of the whole
    meeting while covering only the end of it.
    """


def _server_error(resp) -> str:
    """The server's own message, which raise_for_status throws away.

    LM Studio reports an out-of-memory condition in the response body; the
    HTTP status alone says only "400 Bad Request".
    """
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:300]
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        error = error.get("message") or json.dumps(error)
    return str(error).strip()[:300]


def vram_note() -> str:
    from . import hardware

    gpus = hardware.nvidia()
    if not gpus:
        return ""
    gpu = gpus[0]
    return f" GPU: {gpu['free_mb']} MB free of {gpu['vram_mb']} MB."


# Defaults for the servers people actually run. ttl is LM Studio's idle-unload
# field; the others ignore or reject it, so it is left at zero for them.
PRESETS = [
    {
        "name": "LM Studio",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "ttl_seconds": 300,
        "note": "Reports quantization and load state. Idle TTL unloads the model for you.",
    },
    {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "ttl_seconds": 0,
        "note": "OpenAI-compatible endpoint. Unloads on its own schedule.",
    },
    {
        "name": "Open WebUI",
        "base_url": "http://localhost:3000/api",
        "api_key": "",
        "ttl_seconds": 0,
        "note": "Needs an API key from Settings > Account in Open WebUI.",
    },
    {
        "name": "llama.cpp server",
        "base_url": "http://localhost:8080/v1",
        "api_key": "",
        "ttl_seconds": 0,
        "note": "Serves whichever single model it was started with.",
    },
    {
        "name": "vLLM",
        "base_url": "http://localhost:8000/v1",
        "api_key": "",
        "ttl_seconds": 0,
        "note": "Serves whichever single model it was started with.",
    },
]


def apply_preset(cfg, name: str) -> bool:
    for preset in PRESETS:
        if preset["name"] == name:
            cfg.llm.base_url = preset["base_url"]
            cfg.llm.api_key = preset["api_key"]
            cfg.llm.ttl_seconds = preset["ttl_seconds"]
            return True
    return False


def _headers(cfg) -> dict:
    return {"Authorization": f"Bearer {cfg.llm.api_key or 'none'}"}


# lms ships inside LM Studio and only reaches PATH when `lms bootstrap` edits a
# shell profile. An application launched from a desktop entry or a tray icon
# never reads those, so PATH alone finds nothing and every load and unload turns
# into a silent no-op. These are the documented install locations.
LMS_PATHS = (
    "~/.lmstudio/bin/lms",
    "~/.cache/lm-studio/bin/lms",
    "/opt/LM Studio/resources/app/.webpack/lms",
    "~/Library/Application Support/LM Studio/bin/lms",
    "~/AppData/Local/LM-Studio/lms.exe",
)


def lms_binary() -> str:
    """The lms CLI, found on PATH or where LM Studio installs it."""
    found = shutil.which("lms")
    if found:
        return found
    for candidate in LMS_PATHS:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return ""


def _v1(cfg, path: str) -> str:
    """A native LM Studio endpoint, alongside the OpenAI-compatible one."""
    base = cfg.llm.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/api/v1/{path.lstrip('/')}"


def rest_instances(cfg) -> list[dict]:
    """Loaded model instances, from LM Studio's native model list.

    Instances rather than models: the same model can be loaded more than once,
    which is exactly how a card fills up without anything looking wrong.
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(_v1(cfg, "models"), headers=_headers(cfg))
            if resp.status_code != 200:
                return []
            found = []
            for entry in resp.json().get("models", []):
                for instance in entry.get("loaded_instances") or []:
                    config = instance.get("config") or {}
                    found.append({
                        "id": instance.get("id") or entry.get("key") or "",
                        "model": entry.get("key") or "",
                        "context": config.get("context_length", 0) or 0,
                        "size_bytes": entry.get("size_bytes") or 0,
                        # Whatever else the server reports. Concurrency in
                        # particular multiplies the KV cache and is not
                        # something meetnotes can set.
                        "config": config,
                    })
            return found
    except (httpx.HTTPError, ValueError, AttributeError):
        return []


def rest_unload(cfg, instance_id: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                _v1(cfg, "models/unload"),
                json={"instance_id": instance_id}, headers=_headers(cfg),
            )
            if resp.status_code >= 400:
                return False, f"{resp.status_code}: {_server_error(resp)}"
            return True, "unloaded"
    except httpx.HTTPError as exc:
        return False, str(exc)


def rest_load(cfg, model: str, context: int) -> tuple[bool, int, str]:
    """Load over REST. Returns (ok, the context actually applied, detail).

    echo_load_config makes the server report what it settled on, so a request
    that was accepted but clamped is visible rather than discovered later as a
    truncated summary.
    """
    payload = {
        "model": model,
        "context_length": context,
        # Documented as decreasing memory use and improving generation speed.
        # The KV cache is what makes a long transcript expensive, so this is
        # the one load option that reliably helps here.
        "flash_attention": True,
        "offload_kv_cache_to_gpu": cfg.llm.kv_cache_on_gpu,
        "echo_load_config": True,
    }
    try:
        with httpx.Client(timeout=cfg.llm.timeout) as client:
            resp = client.post(_v1(cfg, "models/load"), json=payload, headers=_headers(cfg))
            if resp.status_code >= 400:
                return False, 0, _server_error(resp)
            body = resp.json()
            applied = int((body.get("load_config") or {}).get("context_length") or context)
            return True, applied, f"loaded {model} with {applied} tokens of context"
    except (httpx.HTTPError, ValueError) as exc:
        return False, 0, str(exc)


def unload(model: str = "") -> tuple[bool, str]:
    """Evict one model, or every loaded model when no name is given.

    Kept for the CLI path; callers that have a config use unload_everything,
    which prefers REST and does not need the CLI to exist at all.
    """
    lms = lms_binary()
    if not lms:
        return False, "the lms CLI was not found"
    args = [lms, "unload", model] if model else [lms, "unload", "--all"]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip()[:200]
    return True, (done.stdout or "unloaded").strip()[:200]


def unload_everything(cfg, log=None) -> tuple[bool, str]:
    """Evict every loaded instance, and verify that it actually happened.

    Both routes are tried, not one or the other. REST first because lms only
    reaches PATH when `lms bootstrap` has edited a shell profile, which an
    application launched from a desktop entry never reads; the CLI after,
    because an older LM Studio has no unload endpoint. Whichever ran, the
    result is checked by asking again: an accepted unload is not evidence that
    the memory came back, and proceeding on that assumption is how the load
    that follows runs out of memory.
    """
    def note(line):
        if log:
            log(line)

    tried = []
    instances = rest_instances(cfg)
    note(f"loaded instances before: {[i['id'] for i in instances] or 'none reported'}")

    for instance in instances:
        ok, detail = rest_unload(cfg, instance["id"])
        tried.append(f"REST {instance['id']}: {'ok' if ok else detail}")
        note(tried[-1])

    if lms_binary():
        ok, detail = unload()
        tried.append(f"lms unload --all: {detail}")
        note(tried[-1])
    elif not instances:
        note("no REST instances and no lms CLI: nothing to act on")

    # The only answer that counts.
    left = rest_instances(cfg) or [{"id": line} for line in loaded() if line]
    if left:
        names = ", ".join(i["id"] for i in left[:4])
        return False, f"still loaded after unloading: {names}. Tried: {'; '.join(tried) or 'nothing'}"
    return True, "nothing is loaded" + (f" ({'; '.join(tried)})" if tried else "")


def unload_all() -> tuple[bool, str]:
    """Free the GPU by evicting every loaded model, via the CLI.

    Retained for callers without a config to hand; prefer unload_everything.
    """
    return unload()


CONTEXT_STEPS = (4096, 8192, 16384, 32768, 65536, 131072)


# Room for the answer, on top of the prompt. A reasoning model spends most of
# this thinking before it emits a word, and running out mid-answer is what a
# truncated or empty summary actually is.
ANSWER_RESERVE = 4096

# Never load below this, however short the transcript. Sizing the window to a
# five-minute recording produced a 4096-token load with no room to work in.
MIN_CONTEXT = 8192


# Characters per token. Around 4 for English; French runs shorter because
# accented characters and morphology split more often, so this is deliberately
# below the English figure and the 20% slack below covers the rest.
CHARS_PER_TOKEN = 3.5


def required_context(prompt_chars: int, reserve: int = ANSWER_RESERVE) -> int:
    """Context needed for a prompt of this size.

    Not rounded to a power of two. Nothing requires one: LM Studio's
    context_length is simply a token count, and rounding 20000 up to 32768
    reserves KV cache for 12768 tokens that will never be used, which on a card
    that is already tight is the difference between loading and not.
    """
    estimate = int(prompt_chars / CHARS_PER_TOKEN * 1.2) + reserve
    return max(estimate, MIN_CONTEXT)


_ESTIMATE = re.compile(r"Estimated GPU Memory:\s*([\d.]+)\s*(GB|MB)", re.I)


def estimate_load(model: str, context: int, gpu: str = "max") -> tuple[int, str]:
    """GPU memory this model would need at this context, in MB. Zero if unknown.

    `lms load --estimate-only` reports it without loading anything, and honours
    --context-length and --gpu. It is the difference between "failed to load
    model" and "needs 9.2 GB, the card has 8.0". LM Studio's resource guardrails
    also refuse loads they predict will not fit, and this is where that shows up.
    """
    lms = lms_binary()
    if not lms:
        return 0, "the lms CLI was not found"
    try:
        done = subprocess.run(
            [lms, "load", model, "--estimate-only",
             f"--context-length={context}", f"--gpu={gpu}"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 0, str(exc)
    text = ((done.stdout or "") + (done.stderr or "")).strip()
    found = _ESTIMATE.search(text)
    if not found:
        return 0, text[:300]
    size = float(found.group(1))
    return int(size * 1024 if found.group(2).upper() == "GB" else size), text[:300]


def free_vram_mb() -> int:
    from . import hardware

    gpus = hardware.nvidia()
    return gpus[0].get("free_mb", 0) if gpus else 0


def model_size_mb(cfg, model: str) -> int:
    """The model's own size on disk, from LM Studio. Zero if not reported.

    Weights dominate what a load costs, and this is a real number rather than
    an estimate, so it works without the lms CLI. It excludes the KV cache, so
    it is a floor: if the weights alone do not fit, nothing else matters.
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(_v1(cfg, "models"), headers=_headers(cfg))
            if resp.status_code != 200:
                return 0
            for entry in resp.json().get("models", []):
                if entry.get("key") == model:
                    return int((entry.get("size_bytes") or 0) / (1024 * 1024))
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return 0


def load_model(model: str, context: int, gpu: str = "max") -> tuple[bool, str]:
    """Load a model at a given context length via the lms CLI.

    The OpenAI-compatible API has no context parameter: the size is fixed when
    the model is loaded, so changing it means reloading.
    """
    lms = lms_binary()
    if not lms:
        return False, "the lms CLI was not found"
    try:
        done = subprocess.run(
            # No confirmation flag: lms load takes [path], --ttl, --gpu,
            # --context-length, --identifier, --estimate-only and --host, and
            # rejects anything else.
            [lms, "load", model, f"--context-length={context}", f"--gpu={gpu}"],
            capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip()[:300]
    return True, f"loaded {model} with {context} tokens of context"


def _resident_context(cfg) -> int:
    """What the server says the model is loaded with, zero if it does not say."""
    try:
        for entry in catalog(cfg):
            if entry.get("id") == cfg.llm.model and entry.get("state") == "loaded":
                return int(entry.get("loaded_context") or 0)
    except LlmError:
        pass
    return 0


def _check_room(cfg, achieved: int, needed: int) -> None:
    if achieved >= needed:
        return
    raise ContextTooSmall(
        f"{cfg.llm.model} is loaded with {achieved} tokens of context but this "
        f"transcript needs about {needed}. The server would silently drop the "
        f"start of it and summarize only what was left. Use a smaller model so "
        f"the context fits in VRAM, raise max_context if it is capping this, or "
        f"summarize a shorter recording.{vram_note()}"
    )


def fit_context(cfg, prompt_chars: int, log=None) -> tuple[bool, str]:
    """Unload everything, then load the model with the context this needs.

    Unconditional, and deliberately so. Deciding whether the resident instance
    could be reused meant trusting the state the server reports, and when that
    is missing or stale nothing gets unloaded and `lms load` stacks a second
    copy of the same weights on the card. Always evicting first cannot fail
    that way, and costs one reload.

    One attempt, at the size the transcript needs. No stepping down: a smaller
    context that loads is not a success, it is a summary of part of the meeting
    presented as a summary of all of it, and the server truncates silently.
    Failing here leaves the transcripts intact and says what to change.
    """
    needed = required_context(prompt_chars, cfg.llm.answer_reserve or ANSWER_RESERVE)
    try:
        entries = catalog(cfg)
    except LlmError as exc:
        # The server is not reachable, so there is nothing to unload, size or
        # load. Not fatal: the caller records it, and the request itself may
        # still be going somewhere else entirely.
        return False, f"cannot reach {cfg.llm.base_url} to size the context: {exc}"

    ceiling = 0
    for entry in entries:
        if entry.get("id") == cfg.llm.model:
            ceiling = int(entry.get("context") or 0)
            break

    wanted = needed
    if ceiling and ceiling < wanted:
        raise ContextTooSmall(
            f"this transcript needs about {wanted} tokens of context and "
            f"{cfg.llm.model} supports at most {ceiling}. Use a model with a "
            f"larger context window, or summarize a shorter recording."
        )
    if cfg.llm.max_context and cfg.llm.max_context < wanted:
        raise ContextTooSmall(
            f"this transcript needs about {wanted} tokens of context and "
            f"llm.max_context caps it at {cfg.llm.max_context}. Raise the cap "
            f"or summarize a shorter recording."
        )

    before = free_vram_mb()
    cleared, detail = unload_everything(cfg, log=log)
    after = free_vram_mb()
    if log and (before or after):
        log(f"free VRAM {before} MB before the unload, {after} MB after")
    if not cleared:
        # Loading on top of a model that refused to go is precisely the
        # out-of-memory this exists to prevent, and doing it anyway replaces a
        # specific reason with a generic one.
        raise LoadFailed(
            f"could not free the GPU before loading {cfg.llm.model}: {detail}."
            f"{vram_note()} Unload it from LM Studio's own interface, or check "
            f"whether another application is using the model."
        )

    # Weights alone, from the server's own size_bytes. Deliberately not
    # estimate_load: that runs `lms load <model> --estimate-only`, and a build
    # that does not know the flag runs `lms load <model>` instead, which loads
    # the model. Putting a copy on the card between the unload and the load is
    # exactly the failure this whole path exists to avoid, and the estimate it
    # bought is no longer needed now that size_bytes is available over REST.
    weights = model_size_mb(cfg, cfg.llm.model)
    if log:
        log(f"{cfg.llm.model} weighs {weights or '?'} MB, {after or '?'} MB free")
    if weights and after and weights > after:
        raise LoadFailed(
            f"{cfg.llm.model} is {weights} MB and only {after} MB is free, before "
            f"any context is allocated.{vram_note()} Use a smaller quantization "
            f"or a smaller model, or set llm.gpu_offload below max so some layers "
            f"run on the CPU."
        )

    if log:
        log(f"loading {cfg.llm.model} with {wanted} tokens of context")
    ok, applied, detail = _load(cfg, cfg.llm.model, wanted)
    if not ok:
        raise LoadFailed(
            f"could not load {cfg.llm.model} with {wanted} tokens of context."
            f"{vram_note()} {detail or 'no reason reported'}\n"
            f"The weights are {weights or 'an unreported number of'} MB, so the "
            f"rest is the KV cache. Two things multiply it, and neither can be "
            f"set from here: Max Concurrent Predictions in LM Studio allocates "
            f"the cache once per parallel slot, so 4 costs four times 1; and "
            f"llm.kv_cache_on_gpu = false moves the cache to system RAM, which "
            f"is slower but loads."
        )
    # Accepted is not the same as applied: the server may clamp the request.
    achieved = applied or _resident_context(cfg) or wanted
    _check_room(cfg, achieved, needed)
    return True, f"loaded {cfg.llm.model} with {achieved} tokens of context"


def _load(cfg, model: str, context: int) -> tuple[bool, int, str]:
    """REST first, then the CLI, so a missing lms is not a dead end."""
    ok, applied, detail = rest_load(cfg, model, context)
    if ok:
        return True, applied, detail
    if not lms_binary():
        return False, 0, detail or "no REST load endpoint and no lms CLI"
    cli_ok, cli_detail = load_model(model, context, cfg.llm.gpu_offload)
    return cli_ok, 0, cli_detail


def loaded() -> list[str]:
    lms = lms_binary()
    if not lms:
        return []
    try:
        done = subprocess.run([lms, "ps"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def catalog(cfg) -> list[dict]:
    """Model list with quantization and load state where the server exposes it.

    LM Studio's /api/v0/models adds quantization, architecture, context length,
    and whether a model is loaded. It reports no size, so memory has to be
    estimated. Servers without that endpoint fall back to plain /v1/models.
    """
    base = cfg.llm.base_url.rstrip("/")
    v0 = base[: -len("/v1")] + "/api/v0/models" if base.endswith("/v1") else base + "/models"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(v0, headers=_headers(cfg))
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                # Any entry identifies the schema. Testing only the first meant
                # one unusual entry at the front dropped the whole response back
                # to /v1/models, losing the load state that decides whether a
                # model needs loading at all.
                if any("quantization" in item or "state" in item for item in data):
                    return [
                        {
                            "id": item.get("id", ""),
                            "type": item.get("type", ""),
                            "arch": item.get("arch", ""),
                            "quantization": item.get("quantization", ""),
                            "state": item.get("state", ""),
                            "context": item.get("max_context_length", 0) or 0,
                            # Present only while a model is loaded, and absent
                            # from the published example, so read it defensively:
                            # zero means "loaded, size unknown", which is a
                            # reason to reload rather than to trust it.
                            "loaded_context": item.get("loaded_context_length", 0) or 0,
                        }
                        for item in data
                    ]
    except (httpx.HTTPError, ValueError):
        pass
    return [{"id": name, "type": "", "arch": "", "quantization": "", "state": "",
             "context": 0, "loaded_context": 0}
            for name in list_models(cfg)]


def list_models(cfg) -> list[str]:
    url = cfg.llm.base_url.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=_headers(cfg))
            resp.raise_for_status()
            return sorted(item["id"] for item in resp.json().get("data", []))
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise LlmError(f"cannot reach {url}: {exc}") from exc


def _stream(cfg, url: str, payload: dict, on_token) -> tuple[str, str, int]:
    """Server-sent events. Returns (text, finish_reason, reasoning tokens).

    A single blocking request reports nothing until it finishes, which for a
    long summary looks indistinguishable from a hang. Streaming gives a token
    count to show instead.

    finish_reason matters as much as the text: "length" means the answer was
    cut off mid-sentence, and a truncated summary saved as a complete one is
    worse than no summary. Reasoning tokens are counted but never appended,
    because they are the model's scratchpad; counting them is what makes a
    reasoning model's long silence legible rather than looking like a stall.
    """
    payload = {**payload, "stream": True}
    pieces = []
    thinking = 0
    finish = ""
    with httpx.Client(timeout=cfg.llm.timeout) as client:
        with client.stream("POST", url, json=payload, headers=_headers(cfg)) as resp:
            if resp.status_code >= 400:
                resp.read()
                if resp.status_code in (400, 422) and "ttl" in payload:
                    payload.pop("ttl")
                    return _stream(cfg, url, payload, on_token)
                raise LlmError(f"{resp.status_code} from {url}: {_server_error(resp)}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                message = choices[0].get("message") or {}
                finish = choices[0].get("finish_reason") or finish
                # Some servers only populate message on the final chunk, and
                # reasoning models put their scratchpad in a separate field
                # that must not end up in the summary.
                piece = delta.get("content") or message.get("content")
                if piece:
                    pieces.append(piece)
                    on_token(len(pieces) + thinking)
                    continue
                for key in ("reasoning_content", "reasoning"):
                    if delta.get(key) or message.get(key):
                        thinking += 1
                        on_token(len(pieces) + thinking)
                        break
    return "".join(pieces), finish, thinking


def chat(cfg, system: str, user: str, schema: dict | None = None, schema_name: str = "result",
         on_token=None):
    if not cfg.llm.model:
        raise LlmError("no LLM model selected in Settings")
    url = cfg.llm.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
    # LM Studio unloads a model this many idle seconds after the request, which
    # is how the GPU gets released without meetnotes having to manage it.
    # Servers that reject the field are retried without it.
    if cfg.llm.ttl_seconds > 0:
        payload["ttl"] = cfg.llm.ttl_seconds

    finish = ""
    thinking = 0
    try:
        if on_token is not None:
            content, finish, thinking = _stream(cfg, url, payload, on_token)
            if not (content or "").strip():
                # A stream that yielded nothing must not become an empty file.
                # Fall back to a plain request before giving up.
                with httpx.Client(timeout=cfg.llm.timeout) as client:
                    resp = client.post(url, json=payload, headers=_headers(cfg))
                    resp.raise_for_status()
                    choice = resp.json()["choices"][0]
                    content = choice["message"]["content"]
                    finish = choice.get("finish_reason") or finish
        else:
            with httpx.Client(timeout=cfg.llm.timeout) as client:
                resp = client.post(url, json=payload, headers=_headers(cfg))
                if resp.status_code in (400, 422) and "ttl" in payload:
                    payload.pop("ttl")
                    resp = client.post(url, json=payload, headers=_headers(cfg))
                if resp.status_code >= 400:
                    raise LlmError(
                        f"{resp.status_code} from {url}: {_server_error(resp)}"
                    )
                choice = resp.json()["choices"][0]
                content = choice["message"]["content"]
                finish = choice.get("finish_reason") or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise LlmError(f"{cfg.llm.model} at {url}: {exc}") from exc

    if finish == "length":
        # The answer stopped because it ran out of room, not because it was
        # finished. Saving it would put a summary that ends mid-sentence in the
        # meeting folder with nothing to say it is incomplete.
        spent = f", after {thinking} tokens of reasoning" if thinking else ""
        raise Truncated(
            f"{cfg.llm.model} ran out of context before finishing its answer"
            f"{spent}. It wrote {len(content)} characters and stopped mid-way.\n"
            f"The window has to hold the transcript and the whole answer. A "
            f"reasoning model spends most of it thinking before writing a word, "
            f"so raise llm.answer_reserve (currently "
            f"{cfg.llm.answer_reserve or ANSWER_RESERVE}) and run it again, or "
            f"pick a model that does not reason. Raising llm.max_context does "
            f"nothing: that is a cap, not a target."
        )

    content = (content or "").strip()
    if not content:
        raise LlmError(
            f"{cfg.llm.model} returned an empty response. Reasoning models can spend "
            "their whole budget before emitting any answer; try raising the context "
            "length or picking a different model."
        )
    if not schema:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise LlmError(
            f"{cfg.llm.model} did not return valid JSON for structured output "
            f"(models under ~7B commonly fail this): {exc}"
        ) from exc
