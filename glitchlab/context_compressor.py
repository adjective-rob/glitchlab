"""
Shared context compression utilities for GLITCHLAB agent tool-loops.

Provides two mechanisms to keep prompt token usage bounded:

1. ``compress_tool_result`` — immediate compression of individual tool
   outputs at insertion time (before they enter the message history).

2. ``prune_message_history`` — sliding-window pruner that drops old
   messages beyond a configurable window, substituting a lightweight
   structured checkpoint so the agent retains awareness of prior work.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Tool-result compression ──────────────────────────────────────────

# Tools whose results should never be compressed (they're already short
# or the agent needs the exact content for correctness).
_SKIP_COMPRESSION = frozenset({"think", "done", "submit_report"})

# Prefixes that identify symbol definitions for read_file extraction.
_SYMBOL_PREFIXES = (
    "def ", "class ", "async def ", "pub fn ", "struct ", "type ", "export ",
)


def compress_tool_result(tool_name: str, content: str, *, max_chars: int = 500) -> str:
    """Compress a single tool result based on tool type.

    Called at insertion time — *before* the message is appended to the
    history — so oversized payloads never enter the context window.

    Returns the (possibly compressed) content string.
    """
    if tool_name in _SKIP_COMPRESSION:
        return content

    # Already compressed in a previous pass (defensive).
    if "... [Content compressed" in content or "... [Search results compressed" in content:
        return content

    # ── read_file: head + symbols + tail ─────────────────────────────
    if tool_name == "read_file" and len(content) > 1000:
        lines = content.splitlines()
        head = "\n".join(lines[:10])
        tail = "\n".join(lines[-10:])
        symbols = [
            l.strip() for l in lines
            if l.strip().startswith(_SYMBOL_PREFIXES)
        ]
        sym_str = "\n".join(symbols[:20])
        return (
            f"{head}\n\n... [Content compressed. Key symbols:]\n"
            f"{sym_str}\n...\n{tail}"
        )

    # ── find_references: keep first 30 lines ─────────────────────────
    if tool_name == "find_references" and len(content) > 500:
        lines = content.splitlines()
        return "\n".join(lines[:30]) + "\n... [References compressed]"

    # ── get_function: head only ──────────────────────────────────────
    if tool_name == "get_function" and len(content) > 1000:
        lines = content.splitlines()
        return "\n".join(lines[:20]) + "\n... [Function body compressed]"

    # ── query_project_context: truncate ──────────────────────────────
    if tool_name == "query_project_context" and len(content) > 500:
        return content[:500] + "\n... [Context compressed]"

    # ── search_grep: extract file:line references ────────────────────
    if tool_name == "search_grep" and len(content) > 500:
        lines = content.splitlines()
        refs: list[str] = []
        for l in lines:
            parts = l.split(":")
            if len(parts) >= 2:
                refs.append(f"{parts[0]}:{parts[1]}")
        if refs:
            return (
                "\n".join(refs[:30])
                + "\n... [Search results compressed to references only]"
            )
        return content[:500] + "\n... [Search results compressed]"

    # ── run_check / get_error: head truncation ───────────────────────
    if tool_name in ("run_check", "get_error") and len(content) > 500:
        return (
            content[:500]
            + "\n... [Content compressed to save budget. Use tool again if needed]"
        )

    # ── Fallback: generic truncation for any unknown large result ────
    if len(content) > max_chars * 3:
        return content[:max_chars] + "\n... [Content compressed]"

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

    # 4. Keep the tail.
    tail = messages[-keep_last_n:]

    pruned = prefix + tail
    dropped = len(messages) - len(pruned)
    if dropped > 0:
        logger.debug(
            f"[CONTEXT] Pruned {dropped} messages "
            f"(kept {len(prefix)} prefix + {len(tail)} tail)"
        )
    return pruned
