"""Shared context compression for agentic tool loops."""

from __future__ import annotations

import json


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
