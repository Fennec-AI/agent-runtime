"""Model profile registry — named LLM instances any subagent can request.

A "profile" is a name → BaseChatModel mapping. Names are **arbitrary
strings**; the registry treats them as opaque keys. Pick whatever naming
convention fits your application:

  by role:               "default", "fast", "advanced", "vision"
  by provider + model:   "anthropic_sonnet", "openai_gpt4"
  by purpose:            "main_chat", "code_review", "background_research"
  by environment:        "production", "staging", "experimental"

The framework imposes no convention. Subagent definitions reference
profiles by whatever string the application picked.

Usage:
    from langchain_anthropic import ChatAnthropic
    from agent_runtime import register_model, get_model

    register_model("anything_you_want", ChatAnthropic(...))
    # ...
    llm = get_model("anything_you_want")

Lookups for unregistered names raise ModelProfileNotFoundError (a
LookupError subclass) with a clear message listing available profiles.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


# Module-level singleton. Use register_model() / get_model() to manipulate.
MODEL_REGISTRY: dict[str, BaseChatModel] = {}


class ModelProfileNotFoundError(LookupError):
    """Raised by get_model() when no profile is registered under the
    requested name.

    Carries the requested name and the list of available names so
    callers can format error messages without re-querying the registry.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = list(available)
        avail = ", ".join(self.available) or "(none registered)"
        super().__init__(
            f"Model profile {name!r} not found. Available: {avail}"
        )


def register_model(name: str, model: BaseChatModel) -> None:
    """Register a model under a profile name. Re-registering overwrites."""
    if name in MODEL_REGISTRY:
        logger.debug("re-registering model profile %r", name)
    MODEL_REGISTRY[name] = model


def get_model(name: str) -> BaseChatModel:
    """Look up a model profile by name.

    Raises ModelProfileNotFoundError (a LookupError) if no profile is
    registered under the given name. The error message lists the
    available profile names.
    """
    if name not in MODEL_REGISTRY:
        raise ModelProfileNotFoundError(name, list(MODEL_REGISTRY.keys()))
    return MODEL_REGISTRY[name]


def has_model(name: str) -> bool:
    """Whether a profile is registered under this name."""
    return name in MODEL_REGISTRY


def list_model_names() -> list[str]:
    """Names of all registered profiles."""
    return list(MODEL_REGISTRY.keys())
