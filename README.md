# agent_runtime

A Python port of **Claude Code's agent-spawning architecture**, built on LangChain.

It reproduces the part of Claude Code that's genuinely interesting: how a main
agent spawns subagents, coordinates background work, and cancels a whole tree
cleanly — not the UI, not the product surface. ~5k lines, one shared agent loop,
13 tools, a real test suite.

> **Scope:** this is the *agent runtime* slice — the loop, the spawn hierarchy,
> cancellation, tool dispatch. It is a study/port, not a drop-in replacement for
> the Claude Code CLI.

---

## What it does

- **One agent loop, used at every level.** `query()` runs the main agent *and*
  every subagent. The only thing that differs between levels is the `RunContext`
  passed in. Recursion through the loop *is* the agent tree.
- **Subagent spawning** — sync (parent blocks) or background (`run_in_background=True`,
  parent gets an id back and a notification later).
- **Background coordination** — `TaskOutput` (poll/wait), `TaskStop` (kill),
  `SendMessage` (push a message into a running agent's inbox).
- **Persistent background shells** — `Bash(run_in_background=True)` →
  `BashOutput` (cursor-based polling) → `KillShell`.
- **Cooperative cancellation** — one `Session.cancel()` propagates through the
  `AbortController` hierarchy to child agents, in-flight subprocesses, and
  background shells. Subprocess kills use process groups, so `sh -c "sleep 30"`
  dies instantly instead of orphaning `sleep`.
- **Context compaction** — a faithful port of Claude Code's `services/compact`:
  cheap mechanical microcompaction first, then threshold-triggered LLM
  auto-compaction (summarize-and-replace) when the window fills, with token
  budgeting. Keeps long sessions under the model's context limit.
- **Typed structured output** — `Session.run_structured(prompt, PydanticModel)`
  returns a validated instance. Port of Claude Code's `SyntheticOutputTool`: the
  model produces its typed answer by *calling* a `StructuredOutput` tool whose
  schema is your model — single pass, no separate extraction call.
- **Permission gating** — four modes (`default`, `acceptEdits`, `plan`,
  `bypassPermissions`): every tool call is evaluated before it runs and is
  allowed, denied, or routed to an approval callback
  (`Session(..., permission_mode=…, ask_callback=…)`).
- **Hooks pipeline** — `pre_tool` / `post_tool` / `on_error` / `on_spawn` /
  `on_terminate` / `permission_denied` events for observation and extension.

**Built on LangChain.** Messages (`HumanMessage`, `AIMessage`, `ToolMessage`),
the chat model (`BaseChatModel`), and tools (`BaseTool`) all come from
`langchain_core`. The default provider seam is `langchain_anthropic.ChatAnthropic`;
register a different `BaseChatModel` (OpenAI, Bedrock, Ollama, …) without
touching agent code.

---

## Architecture

```
Session                      ← main agent: multi-turn state, the public API
  └─ query(ctx)              ← THE LOOP (shared by main + every subagent)
       ├─ drain inbox        ← notifications from background children
       ├─ compaction         ← microcompact + threshold auto-compaction
       ├─ stream the model   ← llm.bind_tools(tools).astream(...)
       └─ run_tools(...)     ← sequential-batch dispatcher
            └─ SpawnAgent ───→ query(child_ctx)   ← a subagent: same loop, one level down
```

- **`query.py`** — the loop. Streams the model, dispatches tools, drains the
  inbox, idle-waits for background children, and runs compaction
  (microcompaction + threshold auto-compaction) at its compaction step. Has a
  wired-but-empty extension point for stop-hooks.
- **`registry.py`** — `AGENT_REGISTRY` (every live agent) + per-agent inbox
  (`pending_messages`) for push/drain notifications. Final transcripts are
  written to disk so `TaskOutput` can read a finished agent's result after its
  registry entry is gone.
- **`tool_executor.py`** — preserves the model's declared tool order, but runs
  *consecutive* `concurrency_safe` tools in parallel under a `Semaphore`.
- **`cancellation.py`** — `AbortController` (a latched, awaitable flag) +
  `create_child_controller` (GC-safe hierarchical propagation via weakref).
- **`cleanup_registry.py`** — shutdown hooks that kill orphaned background work
  on `Session.aclose()`.
- **`session.py`** — the `Session` facade (`submit_message`, `run_structured`,
  `cancel`, `aclose`), plus `permission_mode` / `ask_callback`.
- **`permissions.py`** — per-tool-call gating with four modes
  (`default` / `acceptEdits` / `plan` / `bypassPermissions`) and an
  approval-callback seam.

---

## Tools

13 built-in tools (in `cli.config.default_tools()`):

| Tool | Purpose | Concurrency-safe |
|------|---------|:---:|
| `Read` | Read a file | ✅ |
| `Glob` | File pattern match | ✅ |
| `Grep` | Content search | ✅ |
| `Bash` | Run a shell command (sync, or `run_in_background`) | ❌ |
| `BashOutput` | Poll a background shell's new output (cursor-based) | ✅ |
| `KillShell` | Kill a background shell (process-group SIGTERM→SIGKILL) | ✅ |
| `Edit` | Find/replace in a file | ❌ |
| `Write` | Create/overwrite a file | ❌ |
| `TodoWrite` | Maintain the session's task checklist (replace-whole-list) | ❌ |
| `Agent` | **Spawn a subagent** (sync or background) | ✅ |
| `TaskOutput` | Read a background agent's status/result by id | ✅ |
| `TaskStop` | Kill a background agent by id | ✅ |
| `SendMessage` | Push a message into a running agent's inbox | ✅ |

Three subagent types ship in `bootstrap.py`: `general-purpose`, `Explore`
(read-only), `Plan` (read-only, returns a plan).

---

## Install

Requires **Python 3.11+**.

```bash
git clone https://github.com/NabilAziz99/agent-runtime.git
cd agent-runtime/python_version

pip install -e .                # core
pip install -e ".[test]"        # + pytest
pip install -e ".[otel]"        # + OpenTelemetry (see Observability)

export ANTHROPIC_API_KEY=sk-ant-...
# optional: export AGENT_MODEL=claude-sonnet-4-5   (the default)
```

---

## Quickstart

### CLI

```bash
python -m cli
```

A streaming REPL. Ctrl+C cancels the current turn (and any background agents/
shells it spawned); Ctrl+D exits. Multi-line prompts work when stdin is piped.

### Library — streaming text

```python
import asyncio
import bootstrap                          # registers models + subagent types
from agent_runtime import Session, get_model
from cli.config import default_tools
from langchain_core.messages import AIMessageChunk

async def main():
    session = Session(llm=get_model("default"), tools=default_tools())
    try:
        async for event in session.submit_message("List the Python files here."):
            if isinstance(event, AIMessageChunk):
                print(event.content, end="", flush=True)
    finally:
        await session.aclose()            # kills any orphaned background work

asyncio.run(main())
```

### Library — typed structured output

```python
from pydantic import BaseModel

class RepoSummary(BaseModel):
    file_count: int
    languages: list[str]

# The agent runs its normal loop and finishes by CALLING a StructuredOutput
# tool whose schema is RepoSummary; the validated instance is captured the
# moment it's called. If a turn ends without the call, it's nudged and retried.
result = await session.run_structured(
    "Survey this repo and summarize it.", RepoSummary,
)
if result:                                    # None if aborted / no valid output
    print(result.file_count, result.languages)   # typed: mypy sees RepoSummary
```

`run_structured` is **main-agent-only by construction** — the `StructuredOutput`
tool is added only to *this* session's tool list (and removed in `finally`).
Subagents build their own tool lists from their `SubagentDefinition` and never
see it, so spawned children are completely unaffected. Returns `None` if the
turn was aborted before the model produced valid structured output.

---

## Observability (optional)

OpenTelemetry spans + metrics, wired as built-in hooks, live on the
**`observability` branch** (kept off `main` to keep the core dependency-light):

```bash
git checkout observability
pip install -e ".[otel]"
```

```python
from agent_runtime.observability import enable_otel
enable_otel(exporter="otlp_http", endpoint="http://localhost:4318")  # e.g. Jaeger
```

Emits `tool.<name>` and `agent.<type>` spans plus tool/subagent counters and
duration histograms — ingestible by any OTLP backend (Jaeger, Tempo, Honeycomb,
Datadog, Langfuse). On `main`, the hooks pipeline is still present (no
subscribers), so you can attach your own observability without that branch.

---

## Testing

```bash
pip install -e ".[test]"
pytest
```

67 tests (`tests/`): cancellation, the hooks pipeline, permission gating, the
Bash sync abort paths (with timing assertions — cancels must be *fast*),
background shells + process-group kills, `Write`, `TodoWrite`, `TaskOutput`
disk fallback, compaction, and structured-output scoping. Runs in ~5s.

---

## What's not here yet

Honest gaps vs. the real Claude Code, roughly by impact:

- **Stop hooks** — the loop can't be re-engaged after a natural stop.
- **MCP** — no external tool-server support.
- **`WebFetch`/`WebSearch`**, the `EnterPlanMode`/`ExitPlanMode` plan-approval
  UX (a `plan` *permission* mode exists, but not the enter/exit-plan tools),
  persistent transcripts / resume, `CLAUDE.md` memory loading, multi-model
  subagent routing (plumbed via `SubagentDefinition.model`, but unused).

---

## Project layout

```
python_version/
├── agent_runtime/
│   ├── query.py            # the shared agent loop
│   ├── session.py          # Session facade (submit_message, run_structured)
│   ├── structured_output.py # StructuredOutput tool (port of SyntheticOutputTool)
│   ├── compaction/         # context compaction (port of services/compact)
│   ├── registry.py         # AGENT_REGISTRY + inbox
│   ├── tool_executor.py    # sequential-batch tool dispatcher
│   ├── cancellation.py     # AbortController + hierarchy
│   ├── cleanup_registry.py # shutdown hooks
│   ├── permissions.py      # per-tool-call permission gating (4 modes)
│   ├── bash_shells.py      # persistent background-shell registry
│   ├── hooks.py            # pre/post-tool + lifecycle event bus
│   ├── models.py           # model-profile registry
│   ├── subagents.py        # subagent-type registry
│   ├── config.py           # runtime config / settings
│   ├── types.py            # RunContext, AgentId, ToolUseId
│   └── tools/              # the 13 built-in tools
├── agents/                 # general-purpose, Explore, Plan definitions
├── cli/                    # streaming terminal REPL
├── rendering/              # ANSI renderer for the CLI
├── tests/                  # pytest suite (67 tests)
├── bootstrap.py            # composition root: registers models + subagents
└── pyproject.toml
```
