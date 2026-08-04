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
import shutil
import subprocess

import httpx


class LlmError(RuntimeError):
    pass


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


def unload_all() -> tuple[bool, str]:
    """Free the GPU by evicting every loaded model.

    LM Studio's REST API has no unload endpoint, but the lms CLI does. Called
    before recording so the language model is not squatting on VRAM the speech
    model is about to need.
    """
    if not shutil.which("lms"):
        return False, "the lms CLI is not on PATH"
    try:
        done = subprocess.run(
            ["lms", "unload", "--all"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip()[:200]
    return True, (done.stdout or "unloaded").strip()[:200]


def loaded() -> list[str]:
    if not shutil.which("lms"):
        return []
    try:
        done = subprocess.run(["lms", "ps"], capture_output=True, text=True, timeout=15)
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
                if data and "quantization" in data[0]:
                    return [
                        {
                            "id": item.get("id", ""),
                            "type": item.get("type", ""),
                            "arch": item.get("arch", ""),
                            "quantization": item.get("quantization", ""),
                            "state": item.get("state", ""),
                            "context": item.get("max_context_length", 0) or 0,
                        }
                        for item in data
                    ]
    except (httpx.HTTPError, ValueError):
        pass
    return [{"id": name, "type": "", "arch": "", "quantization": "", "state": "", "context": 0}
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


def _stream(cfg, url: str, payload: dict, on_token) -> str:
    """Server-sent events from an OpenAI-compatible endpoint.

    A single blocking request reports nothing until it finishes, which for a
    long summary looks indistinguishable from a hang. Streaming gives a token
    count to show instead.
    """
    payload = {**payload, "stream": True}
    pieces = []
    with httpx.Client(timeout=cfg.llm.timeout) as client:
        with client.stream("POST", url, json=payload, headers=_headers(cfg)) as resp:
            if resp.status_code >= 400:
                resp.read()
                if resp.status_code in (400, 422) and "ttl" in payload:
                    payload.pop("ttl")
                    return _stream(cfg, url, payload, on_token)
                resp.raise_for_status()
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
                # Some servers only populate message on the final chunk, and
                # reasoning models put their scratchpad in a separate field
                # that must not end up in the summary.
                piece = delta.get("content") or (choices[0].get("message") or {}).get("content")
                if piece:
                    pieces.append(piece)
                    on_token(len(pieces))
    return "".join(pieces)


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

    try:
        if on_token is not None:
            content = _stream(cfg, url, payload, on_token)
            if not (content or "").strip():
                # A stream that yielded nothing must not become an empty file.
                # Fall back to a plain request before giving up.
                with httpx.Client(timeout=cfg.llm.timeout) as client:
                    resp = client.post(url, json=payload, headers=_headers(cfg))
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
        else:
            with httpx.Client(timeout=cfg.llm.timeout) as client:
                resp = client.post(url, json=payload, headers=_headers(cfg))
                if resp.status_code in (400, 422) and "ttl" in payload:
                    payload.pop("ttl")
                    resp = client.post(url, json=payload, headers=_headers(cfg))
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise LlmError(f"{cfg.llm.model} at {url}: {exc}") from exc

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
