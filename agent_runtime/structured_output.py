"""StructuredOutput tool — faithful port of Claude Code's SyntheticOutputTool.

Claude Code produces a typed final answer NOT by parsing free text, but by
giving the model a tool whose input schema IS the requested output shape.
The model calls it once at the end; the tool echoes its (already-validated)
input back as the structured result. Single pass, no separate extraction
call.

This module builds that tool for a given Pydantic schema. Session.run_structured
adds it to the (main-agent-only) tool list, drives the loop, and captures the
validated instance the moment the model calls it.

Verbatim strings are from tools/SyntheticOutputTool/SyntheticOutputTool.ts.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel


# The tool name the model sees — must match exactly for detection.
STRUCTURED_OUTPUT_TOOL_NAME = "StructuredOutput"

# Verbatim from CC's SyntheticOutputTool.prompt().
STRUCTURED_OUTPUT_PROMPT = (
    "Use this tool to return your final response in the requested structured "
    "format. You MUST call this tool exactly once at the end of your response "
    "to provide the structured output."
)

# Injected when a turn ends without the call (CC's stop-hook nudge).
STRUCTURED_OUTPUT_NUDGE = (
    "You MUST call the StructuredOutput tool to complete this request. "
    "Call this tool now."
)

# CC's MAX_STRUCTURED_OUTPUT_RETRIES default.
MAX_STRUCTURED_OUTPUT_RETRIES = 5


def make_structured_output_tool(
    output_schema: type[BaseModel],
    capture: dict[str, Any],
) -> BaseTool:
    """Build a StructuredOutput tool whose args-schema IS `output_schema`.

    When the model calls it, LangChain validates the input against
    `output_schema` first (a bad call raises and becomes an error
    ToolMessage the model can retry from). On a valid call, the validated
    instance is stored in `capture["value"]` and a success string is
    returned — exactly CC's `{structured_output: input}` echo.
    """

    class StructuredOutput(BaseTool):
        name: str = STRUCTURED_OUTPUT_TOOL_NAME
        description: str = STRUCTURED_OUTPUT_PROMPT
        args_schema: type[BaseModel] = output_schema
        # Read-only echo — safe to run alongside other safe tools.
        metadata: dict[str, Any] = {"concurrency_safe": True}

        async def _arun(self, **kwargs: Any) -> str:
            # LangChain already validated kwargs against output_schema;
            # reconstruct the instance and capture it.
            capture["value"] = output_schema(**kwargs)
            return "Structured output provided successfully"

        def _run(self, **kwargs: Any) -> str:
            raise NotImplementedError("StructuredOutput is async-only.")

    return StructuredOutput()
