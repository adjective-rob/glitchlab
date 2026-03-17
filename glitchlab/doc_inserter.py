import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


def insert_doc_comments(file_path: Path, router: Any) -> bool:
    """
    Surgically insert /// doc comments above public functions that lack one.
    Asks the model only for comment text, does file manipulation in Python.
    """
    lines = file_path.read_text().splitlines()
    functions_needing_docs = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("pub fn ") or stripped.startswith("pub async fn "):
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j < 0 or not lines[j].strip().startswith("///"):
                functions_needing_docs.append((i, line))

    if not functions_needing_docs:
        logger.info("[DOC] No public functions missing doc comments.")
        return False

    fn_list = "\n".join(
        f"Line {i+1}: {line.strip()}" for i, line in functions_needing_docs
    )
    prompt = f"""For each of the following Rust public functions, write a single concise /// doc comment (one line only).
Return a JSON array where each item has "line" (the line number) and "comment" (the full /// comment string).
Example: [{{"line": 42, "comment": "/// Initializes the vault and loads existing keys."}}]

Functions:
{fn_list}
"""
    response = router.complete(
        role="implementer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    content = response.content.strip()
    if content.startswith("```"):
        content = "\n".join(
            line for line in content.split("\n") if not line.strip().startswith("```")
        )

    try:
        comments = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                comments = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("[DOC] Failed to parse doc comments from model response")
                return False
        else:
            logger.error("[DOC] Failed to parse doc comments from model response")
            return False

    comment_map = {item["line"]: item["comment"] for item in comments}

    for i, line in reversed(functions_needing_docs):
        line_num = i + 1
        comment = comment_map.get(line_num)
        if comment:
            indent = len(line) - len(line.lstrip())
            comment_text = comment.strip()
            if not comment_text.startswith("///"):
                comment_text = "/// " + comment_text
            comment_line = " " * indent + comment_text
            lines.insert(i, comment_line)

    file_path.write_text("\n".join(lines) + "\n")
    logger.info(f"[DOC] Inserted {len(comment_map)} doc comments into {file_path.name}")
    return True
