"""Application composition root — registers model profiles and (later)
subagent types.

This module is imported (for its registration side-effects) by every
entry point: the CLI today, a future API server, batch scripts, tests.
That's the Pythonic "declarative file" pattern — same as pytest's
conftest.py or Django's apps.py.

────────────────────────────────────────────────────────────────────
Profile names are arbitrary strings. The framework treats them as
opaque keys; no name is special.

Pick a naming convention that makes sense for YOUR application. Some
real-world conventions (all equally valid):

  by role:                "default", "fast", "advanced", "vision"
  by provider + model:    "anthropic_sonnet", "anthropic_haiku", "openai_gpt4"
  by workflow purpose:    "main_chat", "background_research", "code_review"
  by environment:         "production", "dev", "experimental"

Whatever name you pick, subagent definitions reference it as a string
(SubagentDefinition(model="your_name")). The framework just does a dict
lookup.
────────────────────────────────────────────────────────────────────

Environment variables (optional, used by the example registration below):
    AGENT_MODEL       — model name passed to ChatAnthropic
                        (default "claude-sonnet-4-5")
    AGENT_MAX_TOKENS  — max output tokens (default 4096)
"""
from __future__ import annotations

import os
from pathlib import Path

# Load a project-local `.env` BEFORE anything else reads os.environ.
# Real shell-exported variables win over .env values (override=False is
# the default) — matches every other Python app's expectation and keeps
# CI/CD behavior predictable when secrets are injected as real env vars.
#
# We pass the path EXPLICITLY (rather than relying on dotenv's frame-
# walking find_dotenv()) because the auto-discovery is unreliable when
# bootstrap is imported from various contexts — interactive `python -c`,
# `python -m cli.main`, test harnesses, etc. — and especially when the
# project path contains spaces. Explicit path = same answer every time.
#
# Optional dependency: if python-dotenv isn't installed we silently skip,
# falling back to whatever the shell already provided.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from langchain_anthropic import ChatAnthropic

from agent_runtime import register_model, register_subagent
from agents import (
    EXPLORE_AGENT,
    GENERAL_PURPOSE_AGENT,
    PLAN_AGENT,
    SECURITY_REVIEW_AGENT,
)


# ── Register model profiles ─────────────────────────────────────────────
# To add a profile: import its chat class, call register_model(name, instance).
# Mix providers freely — register_model accepts any BaseChatModel.

register_model(
    "default",
    ChatAnthropic(
        model_name=os.environ.get("AGENT_MODEL", "claude-sonnet-4-5"),
        max_tokens_to_sample=int(os.environ.get("AGENT_MAX_TOKENS", "4096")),
        timeout=None,
        stop=None,
    ),
)

# Examples of how to add more profiles (uncomment and adjust):
#
# # Faster, cheaper profile for background subagents:
# register_model(
#     "fast",
#     ChatAnthropic(model_name="claude-haiku-4-5", max_tokens_to_sample=2048,
#                   timeout=None, stop=None),
# )
#
# # OpenAI profile for tasks that benefit from a different model family:
# from langchain_openai import ChatOpenAI
# register_model("smart", ChatOpenAI(model="gpt-4-turbo", max_tokens=8192))
#
# # Local Ollama model:
# from langchain_ollama import ChatOllama
# register_model("local", ChatOllama(model="llama3.2"))


# ── Subagent types ──────────────────────────────────────────────────────
# Register the built-in subagent kinds. The first three mirror Claude Code's
# core built-in agents (same prompts, same tool restrictions); security-review
# is ported from Claude Code's /security-review command and orchestrates its
# own finder + verifier fan-out. Definitions live under `agents/` so the
# catalog is its own browseable module.

register_subagent(GENERAL_PURPOSE_AGENT)
register_subagent(EXPLORE_AGENT)
register_subagent(PLAN_AGENT)
register_subagent(SECURITY_REVIEW_AGENT)

# To add a custom subagent type, define a SubagentDefinition (anywhere)
# and call register_subagent() here. Re-registration with the same name
# overwrites, so application code can shadow built-ins if needed.
