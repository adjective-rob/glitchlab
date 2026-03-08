"""
Semantic context compression for GLITCHLAB agent tool-loops.

Provides three mechanisms to keep prompt token usage bounded while
preserving structural meaning:

1. ``compress_tool_result`` — semantic compression of individual tool
   outputs at insertion time (before they enter the message history).
   Preserves file names, symbol names, line references, and tool actions.

2. ``compress_write_file_args`` — compresses large ``write_file`` payloads
   in assistant tool-call messages.

3. ``prune_message_history`` — sliding-window pruner that drops old
   messages beyond a configurable window, substituting a lightweight
   structured checkpoint so the agent retains awareness of prior work.

Helper functions ``build_tool_message`` and ``build_assistant_message``
centralise message construction so agents remain simple.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Tools whose results should never be compressed (they're already short
# or the agent needs the exact content for correctness).
_SKIP_COMPRESSION = frozenset({"think", "done", "submit_report"})

# Prefixes that identify symbol definitions for read_file extraction.
_SYMBOL_PREFIXES = (
    "def ", "class ", "async def ", "pub fn ", "fn ",
    "struct ", "type ", "export ", "interface ", "enum ",
    "const ", "module ", "impl ",
)

# Number of head/tail lines to keep for read_file compression.
_READ_HEAD_LINES = 10
_READ_TAIL_LINES = 10
_MAX_SYMBOLS = 30

# Max characters for run_check / get_error truncation.
_CHECK_MAX_CHARS = 500

# Max grep references to keep.
_MAX_GREP_REFS = 30

# ── Semantic tool-result compression ─────────────────────────────────


def _compress_read_file(content: str) -> str:
    """Semantic compression for read_file results.

    Strategy:
    - Keep first ~10 lines (file header / imports).
    - Keep last ~10 lines (file tail / exports).
    - Extract key symbols (def, class, struct, etc.) from the middle.
    - Summarise the middle using the extracted symbols.
    """
    lines = content.splitlines()
    total = len(lines)

    if total <= (_READ_HEAD_LINES + _READ_TAIL_LINES + 5):
        return content  # Too short to compress meaningfully.

    head = lines[:_READ_HEAD_LINES]
    tail = lines[-_READ_TAIL_LINES:]
    middle = lines[_READ_HEAD_LINES:-_READ_TAIL_LINES]

    # Extract symbol definitions from the middle section.
    symbols: list[str] = []
    for line in middle:
        stripped = line.strip()
        if stripped.startswith(_SYMBOL_PREFIXES):
            symbols.append(stripped)
            if len(symbols) >= _MAX_SYMBOLS:
                break

    # Build the summary of the omitted middle section.
    omitted = len(middle)
    sym_block = "\n".join(f"  {s}" for s in symbols)
    if symbols:
        middle_summary = (
            f"... [{omitted} lines omitted. {len(symbols)} key symbol(s) extracted:]\n"
            f"{sym_block}"
        )
    else:
        middle_summary = f"... [{omitted} lines omitted — no symbol definitions found]"

    return "\n".join(head) + "\n\n" + middle_summary + "\n\n" + "\n".join(tail)


def _compress_search_grep(content: str) -> str:
    """Semantic compression for search_grep results.

    Strategy:
    - Convert each match to a compact file:line reference.
    - Drop the matched code text to save tokens.
    """
    lines = content.splitlines()
    refs: list[str] = []
    for line in lines:
        # grep -rn output: ./path/to/file:42:matched text
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].strip().isdigit():
            refs.append(f"{parts[0]}:{parts[1]}")
        elif len(parts) >= 2:
            # Fallback: keep file + first token
            refs.append(f"{parts[0]}:{parts[1]}")

    if refs:
        ref_block = "\n".join(refs[:_MAX_GREP_REFS])
        suffix = ""
        if len(refs) > _MAX_GREP_REFS:
            suffix = f"\n... [{len(refs) - _MAX_GREP_REFS} more references omitted]"
        return ref_block + suffix + "\n... [search results compressed to file:line references]"

    # Fallback if we couldn't parse references.
    return content[:_CHECK_MAX_CHARS] + "\n... [search results compressed]"


def _compress_check_output(content: str) -> str:
    """Semantic compression for run_check / get_error results.

    Strategy:
    - Keep first ~500 characters (error messages are usually at the top).
    - Append truncation marker.
    """
    return content[:_CHECK_MAX_CHARS] + "\n... [output truncated]"


def _compress_write_file(content: str) -> str:
    """Semantic compression for write_file results in assistant args.

    Strategy:
    - Replace large payloads with a metadata summary.
    """
    lines_written = len(content.splitlines())
    return f"... [Content compressed: wrote {lines_written} lines to file]"


def compress_tool_result(tool_name: str, content: str, *, max_chars: int = 500) -> str:
    """Compress a single tool result based on tool type.

    Called at insertion time — *before* the message is appended to the
    history — so oversized payloads never enter the context window.

    Preserves: file names, symbol names, line references, tool actions.

    Returns the (possibly compressed) content string.
    """
    if tool_name in _SKIP_COMPRESSION:
        return content

    # Already compressed in a previous pass (defensive).
    if "... [Content compressed" in content or "... [search results compressed" in content:
        return content

    # ── read_file: head + symbols + tail ─────────────────────────────
    if tool_name == "read_file" and len(content) > 1000:
        return _compress_read_file(content)

    # ── find_references: keep first 30 lines ─────────────────────────
    if tool_name == "find_references" and len(content) > 500:
        lines = content.splitlines()
        return "\n".join(lines[:30]) + "\n... [references compressed]"

    # ── get_function: head only ──────────────────────────────────────
    if tool_name == "get_function" and len(content) > 1000:
        lines = content.splitlines()
        return "\n".join(lines[:20]) + "\n... [function body compressed]"

    # ── query_project_context: truncate ──────────────────────────────
    if tool_name == "query_project_context" and len(content) > 500:
        return content[:500] + "\n... [context compressed]"

    # ── search_grep: extract file:line references ────────────────────
    if tool_name == "search_grep" and len(content) > 500:
        return _compress_search_grep(content)

    # ── run_check / get_error: head truncation ───────────────────────
    if tool_name in ("run_check", "get_error") and len(content) > _CHECK_MAX_CHARS:
        return _compress_check_output(content)

    # ── Fallback: generic truncation for any unknown large result ────
    if len(content) > max_chars * 3:
        return content[:max_chars] + "\n... [output truncated]"

    return content


def compress_write_file_args(tool_calls: list[dict]) -> None:
    """Compress ``write_file`` content in assistant tool-call arguments.

    Mutates *tool_calls* in place.  Safe to call on every assistant
    message — it only touches ``write_file`` calls with large content.
    """
    for tc in tool_calls:
        if tc.get("function", {}).get("name") != "write_file":
            continue
        try:
            args = json.loads(tc["function"]["arguments"])
            raw = str(args.get("content", ""))
            if len(raw) > 200:
                lines_written = len(raw.splitlines())
                path = args.get("path", "unknown")
                args["content"] = (
                    f"... [Content compressed: wrote {lines_written} lines to {path}]"
                )
                tc["function"]["arguments"] = json.dumps(args)
        except Exception:
            pass


# ── Reusable message-building helpers ────────────────────────────────


def build_tool_message(
    tool_call_id: str,
    name: str,
    content: str,
) -> dict[str, str]:
    """Build a compressed tool-result message ready for the history.

    Applies ``compress_tool_result`` automatically so agents don't need
    to call it themselves.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": compress_tool_result(name, content),
    }


def build_assistant_message(response: Any) -> dict[str, Any]:
    """Build an assistant message from a router response.

    Applies ``compress_write_file_args`` automatically so agents don't
    need to call it themselves.

    *response* must have ``.content`` (str | None) and ``.tool_calls``
    (list | None) attributes (the ``RouterResponse`` protocol).
    """
    msg: dict[str, Any] = {"role": "assistant"}
    if response.content:
        msg["content"] = response.content
    if response.tool_calls:
        msg["tool_calls"] = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in response.tool_calls
        ]
        compress_write_file_args(msg["tool_calls"])
    return msg


# ── Sliding-window message pruning ───────────────────────────────────


def prune_message_history(
    messages: list[dict[str, Any]],
    *,
    keep_last_n: int = 8,
    checkpoint: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a pruned copy of *messages* using a sliding window.

    Keeps:
    - All leading ``system`` messages (positions 0…)
    - The first ``user`` message (the task description)
    - An injected checkpoint summary (if provided)
    - The last *keep_last_n* messages

    Everything in between is dropped.  The checkpoint is a plain dict
    rendered as YAML — no extra LLM call is required.
    """
    if len(messages) <= keep_last_n + 4:
        return messages  # Nothing worth pruning.

    # 1. Collect leading system messages.
    prefix: list[dict] = []
    first_user_idx: int | None = None
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system":
            prefix.append(msg)
        else:
            first_user_idx = idx
            break

    # 2. Keep the first user message (task description).
    if first_user_idx is not None:
        prefix.append(messages[first_user_idx])

    # 3. Inject checkpoint if provided.
    if checkpoint:
        ckpt_yaml = yaml.dump(
            checkpoint,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).rstrip("\n")
        prefix.append({
            "role": "user",
            "content": (
                "[CHECKPOINT] Summary of your work so far:\n"
                f"```yaml\n{ckpt_yaml}\n```\n"
                "Continue from where you left off."
            ),
        })

    # 4. Keep the tail — must start at a valid message boundary.
    # The Anthropic API requires every tool_result to have a corresponding
    # tool_use in the previous message.  If messages[-keep_last_n] is a
    # "tool" role message, walk backwards to include the preceding
    # "assistant" message that contains the tool_calls.
    tail_start = len(messages) - keep_last_n
    min_start = len(prefix)
    while tail_start > min_start and messages[tail_start].get("role") == "tool":
        tail_start -= 1
    tail = messages[tail_start:]

    pruned = prefix + tail
    dropped = len(messages) - len(pruned)
    if dropped > 0:
        logger.debug(
            f"[CONTEXT] Pruned {dropped} messages "
            f"(kept {len(prefix)} prefix + {len(tail)} tail)"
        )
    return pruned
