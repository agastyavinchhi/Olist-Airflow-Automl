"""Minimal OpenRouter JSON client for the Olist agents (no dependency on the class template).

Every agent asks for ONE JSON object and validates it itself; on any failure the agent
falls back to its own coded heuristic, so the loop always completes.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
NO_LLM = "none"            # UI option "Heuristics only" → agents use their coded fallbacks
_quota_blocked_until = 0.0  # circuit breaker: daily free quota exhausted → stop calling for 1 h


def configured(model: str | None = None) -> bool:
    """True when an OpenRouter key is set and the model is an OpenRouter id."""
    m = (model or DEFAULT_MODEL).strip()
    return (bool(os.getenv("OPENROUTER_API_KEY", "").strip())
            and m != NO_LLM and not m.startswith("cursor"))


def _first_json_object(text: str) -> dict | None:
    """Pull the first balanced {...} out of a reply (handles ``` fences and prose)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def chat_json(system: str, user: str, *, model: str | None = None, max_tokens: int = 600,
              temperature: float = 0.2, retries: int = 3) -> tuple[dict | None, str]:
    """Return (parsed JSON or None, short status note). Retries 429 / 5xx with backoff."""
    global _quota_blocked_until
    model = (model or DEFAULT_MODEL).strip()
    if model == NO_LLM:
        return None, "heuristics-only run"
    if not configured(model):
        return None, "no LLM key → heuristic"
    if time.time() < _quota_blocked_until:
        return None, "LLM daily quota exhausted → heuristic"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system + "\nReply with ONE JSON object only."},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        # Free reasoning models otherwise spend the whole budget "thinking" and return no content.
        "reasoning": {"enabled": False},
    }).encode()
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}",
               "Content-Type": "application/json", "X-Title": "olist-bad-review-loop"}
    last = "llm error"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or msg.get("reasoning") or ""
            parsed = _first_json_object(content)
            if parsed is not None:
                return parsed, f"llm · {model}" + (f" (retry {attempt})" if attempt else "")
            last = "unparsable reply"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
            if exc.code == 429:
                detail = exc.read().decode("utf-8", "replace")
                if "per-day" in detail or "per day" in detail:  # daily cap: retrying won't help
                    _quota_blocked_until = time.time() + 3600
                    return None, "LLM daily quota exhausted → heuristic"
                time.sleep(5 * (attempt + 1))  # short-term rate limit: back off
                continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    return None, f"{last} → heuristic"
