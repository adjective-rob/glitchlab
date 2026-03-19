"""Shared context compression for agentic tool loops."""

from __future__ import annotations

import json

from loguru import logger


def compress_stale_messages(messages: list[dict]) -> None:
    """
    Mutate messages in-place to compress stale tool outputs and inputs.

    Called at the top of each agent loop iteration. Compresses tool results
    that have already been consumed by a subsequent assistant message, and
    compresses large write_file arguments in consumed assistant messages.
    """
    for i in range(len(messages)):
        # Compress tool outputs after they've been consumed by the assistant
        if messages[i].get("role") == "tool":
            consumed = any(m.get("role") == "assistant" for m in messages[i + 1 :])
            if consumed:
                content = str(messages[i].get("content", ""))
                if (
                    "... [Content compressed" in content
                    or "... [Search results compressed" in content
                ):
                    continue  # Already compressed

                tname = messages[i].get("name")

                # Smart symbol extraction for read_file
                if tname == "read_file" and len(content) > 1000:
                    lines = content.splitlines()
                    head = "\n".join(lines[:10])
                    tail = "\n".join(lines[-10:])
                    symbols = [
                        line.strip()
                        for line in lines
                        if line.strip().startswith(
                            (
                                "def ",
                                "class ",
                                "async def ",
                                "pub fn ",
                                "struct ",
                                "type ",
                                "export ",
                            )
                        )
                    ]
                    sym_str = "\n".join(symbols[:20])
                    messages[i]["content"] = (
                        f"{head}\n\n... [Content compressed. Key symbols:]\n"
                        f"{sym_str}\n...\n{tail}"
                    )

                # Reference-only extraction for search_grep
                elif tname == "search_grep" and len(content) > 500:
                    lines = content.splitlines()
                    refs = []
                    for line in lines:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            refs.append(f"{parts[0]}:{parts[1]}")
                    if refs:
                        messages[i]["content"] = (
                            "\n".join(refs[:30])
                            + "\n... [Search results compressed to references only]"
                        )
                    else:
                        messages[i]["content"] = (
                            content[:500] + "\n... [Search results compressed]"
                        )

                # Compress verbose run_check/get_error output
                elif tname in ("run_check", "get_error") and len(content) > 800:
                    lines = content.splitlines()
                    exit_line = lines[0] if lines else ""
                    tail = "\n".join(lines[-20:])
                    messages[i]["content"] = (
                        f"{exit_line}\n... [Content compressed to last 20 lines]\n"
                        f"{tail}"
                    )

                # Compress write_file/replace_in_file confirmations
                elif (
                    tname in ("write_file", "replace_in_file") and len(content) > 200
                ):
                    messages[i]["content"] = (
                        content[:200] + "\n... [Content compressed]"
                    )

        # Compress tool inputs (e.g. massive write_file contents) after consumption
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            consumed = any(m.get("role") == "tool" for m in messages[i + 1 :])
            if consumed:
                for tc in messages[i]["tool_calls"]:
                    if tc.get("function", {}).get("name") == "write_file":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            if "content" in args and len(str(args["content"])) > 200:
                                lines_written = len(
                                    str(args["content"]).splitlines()
                                )
                                path = args.get("path", "unknown")
                                args["content"] = (
                                    f"... [Content compressed: wrote "
                                    f"{lines_written} lines to {path}]"
                                )
                                tc["function"]["arguments"] = json.dumps(args)
                        except Exception:
                            pass


# --------------------------------------------------------------------------- #
# Hard compaction: drop consumed message pairs when list grows too long.
# compress_stale_messages shrinks content; this function removes messages.
# --------------------------------------------------------------------------- #

_HARD_COMPACT_THRESHOLD = 40  # Trigger compaction above this many messages
_HARD_COMPACT_KEEP_RECENT = 12  # Always preserve the N most recent messages


def hard_compact_messages(messages: list[dict]) -> None:
    """
    Remove old consumed assistant+tool message pairs when the list is too long.

    Preserves:
    - messages[0] (system) and messages[1] (user) — always sacred
    - The most recent _HARD_COMPACT_KEEP_RECENT messages — the agent's working memory
    - Any assistant message containing write_file or replace_in_file tool calls — edit history matters

    Replaces all dropped messages with a single user-role summary message listing
    files read, edits made, and check results so the agent retains orientation.
    """
    if len(messages) <= _HARD_COMPACT_THRESHOLD:
        return

    # Sacred prefix: system + initial user message
    prefix = messages[:2]

    # Find a safe cut point that doesn't split assistant+tool pairs.
    # Start at the target and walk backward until we hit an assistant message,
    # which means its tool_results are all after the cut (in the suffix).
    cut = len(messages) - _HARD_COMPACT_KEEP_RECENT
    while cut > 2 and messages[cut].get("role") != "assistant":
        cut -= 1

    # If we walked all the way back to prefix, skip compaction this round
    if cut <= 2:
        return

    suffix = messages[cut:]
    middle = messages[2:cut]

    if not middle:
        return

    # Extract summary context from the middle before dropping it
    files_read = set()
    files_edited = set()
    checks_run = []
    searches_done = []

    for msg in middle:
        role = msg.get("role")

        if role == "tool":
            name = msg.get("name", "")
            content = str(msg.get("content", ""))

            if name == "read_file":
                # Extract filename from "Read {path}" or "Read N characters from {path}"
                for token in content.split():
                    if "/" in token or "." in token:
                        files_read.add(token.rstrip(":").rstrip(","))
                        break

            elif name in ("write_file", "replace_in_file"):
                for token in content.split():
                    if "/" in token or "." in token:
                        files_edited.add(token.rstrip(":").rstrip(","))
                        break

            elif name == "run_check":
                # Keep first line (exit code) as summary
                first_line = content.split("\n")[0] if content else ""
                if first_line and len(checks_run) < 5:
                    checks_run.append(first_line[:100])

            elif name == "search_grep":
                if len(searches_done) < 3:
                    searches_done.append(content[:80])

        elif role == "assistant" and msg.get("tool_calls"):
            # Check if this assistant message contains edits — if so, keep it
            for tc in msg.get("tool_calls", []):
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name in ("write_file", "replace_in_file"):
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        path = args.get("path", "unknown")
                        files_edited.add(path)
                    except Exception:
                        pass

    # Build compact summary
    summary_parts = ["[Context compacted — older messages removed to save tokens]"]
    if files_read:
        summary_parts.append(f"Files read: {', '.join(sorted(files_read))}")
    if files_edited:
        summary_parts.append(f"Files edited: {', '.join(sorted(files_edited))}")
    if checks_run:
        summary_parts.append(f"Checks run: {'; '.join(checks_run)}")
    if searches_done:
        summary_parts.append(f"Searches: {'; '.join(searches_done)}")

    summary_msg = {
        "role": "user",
        "content": "\n".join(summary_parts)
    }

    # Rebuild the message list in place
    messages.clear()
    messages.extend(prefix)
    messages.append(summary_msg)
    messages.extend(suffix)

    logger.info(
        f"[COMPACT] Hard compaction: {len(middle)} messages → 1 summary. "
        f"List now {len(messages)} messages."
    )
