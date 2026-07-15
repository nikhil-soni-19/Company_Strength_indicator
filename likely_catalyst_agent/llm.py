"""
llm.py — two-tier LLM provider router.

Honors the .env schema the user defined:

  LLM_LIGHT_PROVIDER   / LLM_LIGHT_MODEL   / OPENAI_MODEL      (cheap, structured)
  LLM_THINKING_PROVIDER/ LLM_THINKING_MODEL/ ANTHROPIC_MODEL   (high-quality synthesis)
  LLM_LIGHT_MAX_TOKENS / LLM_THINKING_MAX_TOKENS / LLM_TEMPERATURE
  OPENAI_API_KEY / ANTHROPIC_API_KEY

Two tiers:
  • complete_light()    — small/fast model for query parsing (default: OpenAI)
  • complete_thinking() — strong model for catalyst + reasoning synthesis
                          (default: Anthropic claude-sonnet-4-6)

OpenAI calls go through the official ``openai`` SDK; Claude calls through the
official ``anthropic`` SDK. The two are never mixed and there is no
OpenAI-compatible shim for Claude. Env is read at call time so values loaded
by python-dotenv after import are still picked up. Raises a clear RuntimeError
on failure; callers fall back to keyword/template paths.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v not in (None, ""):
            return v
    return default


# ── Config (resolved per call so dotenv-loaded values are honored) ───────────

def _light_provider() -> str:
    return (_env("LLM_LIGHT_PROVIDER", default="openai") or "openai").lower()


def _thinking_provider() -> str:
    return (_env("LLM_THINKING_PROVIDER", default="anthropic") or "anthropic").lower()


def _light_model() -> str:
    return _env("LLM_LIGHT_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")


def _thinking_model() -> str:
    return _env("LLM_THINKING_MODEL", "ANTHROPIC_MODEL", default="claude-sonnet-4-6")


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, default=str(default)))
    except (TypeError, ValueError):
        return default


def _temperature() -> float:
    try:
        return float(_env("LLM_TEMPERATURE", default="0.2"))
    except (TypeError, ValueError):
        return 0.2


def _provider_available(provider: str) -> bool:
    if provider == "openai":
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def light_available() -> bool:
    """True if the configured light-tier provider can actually be called."""
    return _provider_available(_light_provider())


def thinking_available() -> bool:
    """True if the configured thinking-tier provider can actually be called."""
    return _provider_available(_thinking_provider())


# ── Provider backends ────────────────────────────────────────────────────────

def _openai_complete(
    model: str, system: str, user: str, *, max_tokens: int, json_mode: bool
) -> str:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": _temperature(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _anthropic_complete(
    model: str, system: str, user: str, *, max_tokens: int, json_mode: bool
) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        # Stable instruction prefix is prompt-cached; volatile evidence goes
        # in the user message (after the cached prefix).
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user}],
    }
    # Sonnet 4.6 / Opus 4.6: adaptive thinking (budget_tokens is deprecated on
    # these models). Temperature must NOT be sent alongside thinking.
    if model.startswith(("claude-sonnet-4-6", "claude-opus-4-6")):
        kwargs["thinking"] = {"type": "adaptive"}
    else:
        kwargs["temperature"] = _temperature()

    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")


def _dispatch(
    provider: str, model: str, system: str, user: str,
    *, max_tokens: int, json_mode: bool,
) -> str:
    if provider == "openai":
        return _openai_complete(
            model, system, user, max_tokens=max_tokens, json_mode=json_mode
        )
    if provider == "anthropic":
        return _anthropic_complete(
            model, system, user, max_tokens=max_tokens, json_mode=json_mode
        )
    raise RuntimeError(f"Unknown LLM provider {provider!r}")


# ── Public API ───────────────────────────────────────────────────────────────

def complete_light(system: str, user: str, *, json_mode: bool = False) -> str:
    """Cheap/fast tier — query parsing. Raises on failure (caller falls back)."""
    provider, model = _light_provider(), _light_model()
    try:
        return _dispatch(
            provider, model, system, user,
            max_tokens=_int_env("LLM_LIGHT_MAX_TOKENS", 512),
            json_mode=json_mode,
        )
    except Exception as e:
        logger.warning(f"light LLM ({provider}/{model}) failed: {e}")
        raise


def complete_thinking(system: str, user: str, *, json_mode: bool = False) -> str:
    """Strong tier — catalyst + reasoning synthesis. Raises on failure."""
    provider, model = _thinking_provider(), _thinking_model()
    try:
        return _dispatch(
            provider, model, system, user,
            max_tokens=_int_env("LLM_THINKING_MAX_TOKENS", 8000),
            json_mode=json_mode,
        )
    except Exception as e:
        logger.warning(f"thinking LLM ({provider}/{model}) failed: {e}")
        raise


def parse_json(raw: str) -> dict:
    """Tolerant JSON parse — strips markdown fences a model may add."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(
            ln for ln in raw.splitlines() if not ln.strip().startswith("```")
        )
    return json.loads(raw)


def describe() -> str:
    """One-line summary of the active routing (for logs / report metadata)."""
    return (
        f"light={_light_provider()}:{_light_model()}"
        f"(avail={light_available()})  "
        f"thinking={_thinking_provider()}:{_thinking_model()}"
        f"(avail={thinking_available()})"
    )
