"""Tests for the semantic context compression system."""

from __future__ import annotations

import json

from glitchlab.context_compressor import (
    build_assistant_message,
    build_tool_message,
    compress_tool_result,
    compress_write_file_args,
    prune_message_history,
)


# ── compress_tool_result ─────────────────────────────────────────────


class TestCompressToolResult:
    """Tests for semantic compression of tool results."""

    def test_skip_compression_for_think(self):
        content = "x" * 5000
        assert compress_tool_result("think", content) == content

    def test_skip_compression_for_done(self):
        content = "x" * 5000
        assert compress_tool_result("done", content) == content

    def test_skip_compression_for_submit_report(self):
        content = "x" * 5000
        assert compress_tool_result("submit_report", content) == content

    def test_already_compressed_passthrough(self):
        content = "some text\n... [Content compressed: wrote 10 lines to foo.py]"
        assert compress_tool_result("read_file", content) == content

    # ── read_file semantic compression ───────────────────────────────

    def test_read_file_short_content_unchanged(self):
        content = "line1\nline2\nline3"
        assert compress_tool_result("read_file", content) == content

    def test_read_file_preserves_head_and_tail(self):
        lines = [f"line_{i}" for i in range(100)]
        content = "\n".join(lines)
        result = compress_tool_result("read_file", content)
        # Head lines preserved
        for i in range(10):
            assert f"line_{i}" in result
        # Tail lines preserved
        for i in range(90, 100):
            assert f"line_{i}" in result

    def test_read_file_extracts_symbols(self):
        # Content must exceed 1000 chars to trigger compression
        head = ["import os  # " + "x" * 40] * 10
        middle = [
            "def hello():  # " + "x" * 40,
            "    pass",
            "class Foo:  # " + "x" * 40,
            "    pass",
            "async def bar():  # " + "x" * 40,
            "    pass",
            "struct Point:  # " + "x" * 40,
            "    x: int",
        ] * 5
        tail = ["# end  " + "x" * 40] * 10
        content = "\n".join(head + middle + tail)
        assert len(content) > 1000
        result = compress_tool_result("read_file", content)
        assert "def hello():" in result
        assert "class Foo:" in result
        assert "async def bar():" in result
        assert "key symbol(s) extracted" in result

    def test_read_file_reports_omitted_lines(self):
        # Each line must be long enough so total > 1000 chars
        lines = [f"line_{i}  " + "x" * 30 for i in range(100)]
        content = "\n".join(lines)
        assert len(content) > 1000
        result = compress_tool_result("read_file", content)
        assert "lines omitted" in result

    # ── search_grep semantic compression ─────────────────────────────

    def test_search_grep_short_content_unchanged(self):
        content = "./foo.py:10:hello world"
        assert compress_tool_result("search_grep", content) == content

    def test_search_grep_extracts_file_line_refs(self):
        lines = [f"./src/file_{i}.py:{i}:some matched text here" for i in range(40)]
        content = "\n".join(lines)
        result = compress_tool_result("search_grep", content)
        # Should have file:line references
        assert "./src/file_0.py:0" in result
        assert "./src/file_29.py:29" in result
        # Should NOT have the matched code text
        assert "some matched text here" not in result
        # Should indicate compression
        assert "file:line references" in result

    def test_search_grep_limits_references(self):
        lines = [f"./src/file_{i}.py:{i}:text" for i in range(50)]
        content = "\n".join(lines)
        result = compress_tool_result("search_grep", content)
        assert "more references omitted" in result

    # ── run_check / get_error compression ────────────────────────────

    def test_run_check_short_content_unchanged(self):
        content = "All tests passed."
        assert compress_tool_result("run_check", content) == content

    def test_run_check_truncates_long_output(self):
        content = "x" * 1000
        result = compress_tool_result("run_check", content)
        assert len(result) < 1000
        assert "... [output truncated]" in result

    def test_get_error_truncates_long_output(self):
        content = "Error: " + "x" * 1000
        result = compress_tool_result("get_error", content)
        assert len(result) < 1000
        assert "... [output truncated]" in result

    def test_run_check_keeps_first_500_chars(self):
        content = "E" * 500 + "X" * 500
        result = compress_tool_result("run_check", content)
        assert result.startswith("E" * 500)
        assert "X" not in result.split("...")[0]

    # ── Fallback compression ─────────────────────────────────────────

    def test_fallback_truncates_unknown_large_content(self):
        content = "x" * 2000
        result = compress_tool_result("unknown_tool", content)
        assert "... [output truncated]" in result
        assert len(result) < 2000


# ── compress_write_file_args ─────────────────────────────────────────


class TestCompressWriteFileArgs:
    def test_compresses_large_write_file(self):
        tc = {
            "function": {
                "name": "write_file",
                "arguments": json.dumps({
                    "path": "src/main.py",
                    "content": "x\n" * 200,
                }),
            }
        }
        compress_write_file_args([tc])
        args = json.loads(tc["function"]["arguments"])
        assert "Content compressed" in args["content"]
        assert "200 lines" in args["content"]
        assert "src/main.py" in args["content"]

    def test_ignores_small_write_file(self):
        original_content = "hello"
        tc = {
            "function": {
                "name": "write_file",
                "arguments": json.dumps({
                    "path": "small.txt",
                    "content": original_content,
                }),
            }
        }
        compress_write_file_args([tc])
        args = json.loads(tc["function"]["arguments"])
        assert args["content"] == original_content

    def test_ignores_non_write_file_tools(self):
        tc = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "foo.py"}),
            }
        }
        original = json.dumps({"path": "foo.py"})
        compress_write_file_args([tc])
        assert tc["function"]["arguments"] == original


# ── build_tool_message ───────────────────────────────────────────────


class TestBuildToolMessage:
    def test_returns_correct_structure(self):
        msg = build_tool_message("call_123", "read_file", "short content")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["name"] == "read_file"
        assert msg["content"] == "short content"

    def test_applies_compression_automatically(self):
        long_content = "\n".join([f"line {i}" for i in range(200)])
        msg = build_tool_message("call_456", "read_file", long_content)
        assert "lines omitted" in msg["content"] or "key symbol" in msg["content"]


# ── build_assistant_message ──────────────────────────────────────────


class TestBuildAssistantMessage:
    def test_content_only(self):
        class FakeResponse:
            content = "Hello"
            tool_calls = None

        msg = build_assistant_message(FakeResponse())
        assert msg == {"role": "assistant", "content": "Hello"}

    def test_tool_calls_compressed(self):
        class FakeToolCall:
            def model_dump(self):
                return {
                    "id": "tc_1",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "big.py",
                            "content": "x\n" * 300,
                        }),
                    },
                }

        class FakeResponse:
            content = None
            tool_calls = [FakeToolCall()]

        msg = build_assistant_message(FakeResponse())
        assert msg["role"] == "assistant"
        assert "content" not in msg  # No text content
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert "Content compressed" in args["content"]


# ── prune_message_history ────────────────────────────────────────────


class TestPruneMessageHistory:
    def _make_messages(self, n: int) -> list[dict]:
        msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Do the task."},
        ]
        for i in range(n):
            msgs.append({"role": "assistant", "content": f"Step {i}"})
            msgs.append({"role": "user", "content": f"OK {i}"})
        return msgs

    def test_short_history_unchanged(self):
        msgs = self._make_messages(3)
        result = prune_message_history(msgs, keep_last_n=8)
        assert result is msgs  # Same object, no pruning needed.

    def test_long_history_pruned(self):
        msgs = self._make_messages(20)
        result = prune_message_history(msgs, keep_last_n=8)
        assert len(result) < len(msgs)
        # System and first user message preserved
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Do the task."

    def test_checkpoint_injected(self):
        msgs = self._make_messages(20)
        result = prune_message_history(
            msgs, keep_last_n=8, checkpoint={"step": 5, "files_modified": ["a.py"]}
        )
        checkpoint_msgs = [m for m in result if "[CHECKPOINT]" in m.get("content", "")]
        assert len(checkpoint_msgs) == 1
        assert "step" in checkpoint_msgs[0]["content"]

    def test_tool_result_boundary_safety(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        # Add many messages, ending with tool results
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"thinking {i}", "tool_calls": [{"id": f"tc_{i}"}]})
            msgs.append({"role": "tool", "tool_call_id": f"tc_{i}", "name": "read_file", "content": f"result {i}"})
        result = prune_message_history(msgs, keep_last_n=4)
        # Verify that a tool message is never the first in the tail
        tail_start = None
        for i, m in enumerate(result):
            if "[CHECKPOINT]" in m.get("content", ""):
                tail_start = i + 1
                break
        if tail_start and tail_start < len(result):
            assert result[tail_start]["role"] != "tool"
