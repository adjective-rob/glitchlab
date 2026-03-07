"""Tests for semantic context snipping in the router's ContextMonitor."""

from glitchlab.router import ContextMonitor, _extract_file_refs, _estimate_tokens


class TestExtractFileRefs:
    def test_python_file_paths(self):
        text = "Modified glitchlab/router.py and glitchlab/controller.py"
        refs = _extract_file_refs(text)
        assert "glitchlab/router.py" in refs
        assert "glitchlab/controller.py" in refs

    def test_nested_paths(self):
        text = 'read_file("src/agents/implementer.py")'
        refs = _extract_file_refs(text)
        assert any("implementer.py" in r for r in refs)

    def test_no_false_positives_on_plain_text(self):
        text = "The function works correctly and returns true."
        refs = _extract_file_refs(text)
        assert len(refs) == 0

    def test_config_files(self):
        text = "Updated pyproject.toml and package.json"
        refs = _extract_file_refs(text)
        assert any("pyproject.toml" in r for r in refs)
        assert any("package.json" in r for r in refs)

    def test_leading_dot_slash_normalized(self):
        text = "File at ./src/main.py"
        refs = _extract_file_refs(text)
        assert any("src/main.py" in r for r in refs)


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert _estimate_tokens("hello world") == 2  # 11 chars // 4

    def test_empty_string(self):
        assert _estimate_tokens("") == 0


class TestContextMonitorUpdateState:
    def test_update_active_files(self):
        cm = ContextMonitor()
        cm.update_file_state(active_files=["src/main.py", "./lib/utils.py"])
        assert "src/main.py" in cm._active_files
        assert "lib/utils.py" in cm._active_files  # normalized

    def test_update_committed_files(self):
        cm = ContextMonitor()
        cm.update_file_state(committed_files=["old/done.py"])
        assert "old/done.py" in cm._committed_files

    def test_update_symbols(self):
        cm = ContextMonitor()
        cm.update_file_state(symbol_names=["Router", "BudgetTracker"])
        assert "Router" in cm._symbol_names


class TestScoreMessage:
    def setup_method(self):
        self.cm = ContextMonitor()
        self.cm.update_file_state(
            active_files=["glitchlab/router.py"],
            committed_files=["glitchlab/old_module.py"],
            symbol_names=["ContextMonitor"],
        )

    def test_active_file_reference_scores_high(self):
        msg = {"role": "user", "content": "I modified glitchlab/router.py to fix the bug"}
        score = self.cm._score_message(msg, position=5, total=10)
        assert score > 10  # Active file ref (+10) + recency

    def test_committed_file_reference_scores_low(self):
        msg = {"role": "user", "content": "Previously changed glitchlab/old_module.py"}
        score = self.cm._score_message(msg, position=1, total=10)
        # Committed file ref (-5) should make this low
        assert score < 5

    def test_symbol_reference_boosts_score(self):
        msg = {"role": "user", "content": "The ContextMonitor class handles truncation"}
        score_with_sym = self.cm._score_message(msg, position=5, total=10)

        msg_no_sym = {"role": "user", "content": "This class handles truncation"}
        score_no_sym = self.cm._score_message(msg_no_sym, position=5, total=10)

        assert score_with_sym > score_no_sym

    def test_error_messages_score_high(self):
        msg = {"role": "tool", "content": "Traceback: error in line 42"}
        score = self.cm._score_message(msg, position=3, total=10)
        assert score >= 10  # tool bonus + error bonus

    def test_recency_bonus(self):
        msg = {"role": "user", "content": "hello"}
        old_score = self.cm._score_message(msg, position=1, total=100)
        new_score = self.cm._score_message(msg, position=99, total=100)
        assert new_score > old_score


class TestSemanticSnip:
    def _make_messages(self, count, active_file=None, committed_file=None):
        """Helper to build a message list with known file references."""
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(count):
            if committed_file and i < count // 2:
                content = f"Step {i}: working on {committed_file} " + "x" * 500
            elif active_file and i >= count // 2:
                content = f"Step {i}: working on {active_file} " + "x" * 500
            else:
                content = f"Step {i}: generic work " + "x" * 500
            msgs.append({"role": "user", "content": content})
            msgs.append({"role": "assistant", "content": f"Done with step {i}."})
        return msgs

    def test_semantic_snip_preserves_active_file_messages(self):
        cm = ContextMonitor()
        cm.update_file_state(
            active_files=["src/active.py"],
            committed_files=["src/done.py"],
        )

        messages = self._make_messages(6, active_file="src/active.py", committed_file="src/done.py")
        current_tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
        input_limit = current_tokens // 2  # Force snipping ~50%

        result = cm._semantic_snip(messages, current_tokens, input_limit)

        # System message always preserved
        assert result[0]["role"] == "system"

        # Messages about active file should be less truncated than committed file messages
        active_content = ""
        committed_content = ""
        for msg in result:
            c = msg.get("content", "")
            if "src/active.py" in c:
                active_content += c
            if "SNIPPED" in c and "committed" in c:
                committed_content += c

        # Active file messages should have more content preserved
        assert len(active_content) > len(committed_content)

    def test_semantic_snip_preserves_last_messages(self):
        cm = ContextMonitor()
        cm.update_file_state(active_files=["src/main.py"])

        messages = self._make_messages(10, active_file="src/main.py")
        current_tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
        input_limit = current_tokens // 3

        result = cm._semantic_snip(messages, current_tokens, input_limit)

        # Last 2 non-system messages should be intact (protected tail)
        original_last_two = [m for m in messages if m.get("role") != "system"][-2:]
        result_last_two = [m for m in result if m.get("role") != "system"][-2:]
        assert result_last_two[0]["content"] == original_last_two[0]["content"]
        assert result_last_two[1]["content"] == original_last_two[1]["content"]

    def test_tool_messages_preserve_tool_call_id(self):
        cm = ContextMonitor()
        cm.update_file_state(committed_files=["old.py"])

        messages = [
            {"role": "system", "content": "system"},
            {"role": "tool", "tool_call_id": "call_123", "name": "read_file",
             "content": "Content of old.py " + "x" * 1000},
            {"role": "user", "content": "recent message"},
        ]
        current_tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
        input_limit = current_tokens // 2

        result = cm._semantic_snip(messages, current_tokens, input_limit)

        tool_msgs = [m for m in result if m.get("role") == "tool"]
        for tm in tool_msgs:
            assert "tool_call_id" in tm


class TestChronologicalSnipFallback:
    def test_falls_back_without_semantic_state(self):
        """When no file state is set, uses chronological snipping."""
        cm = ContextMonitor()
        # Don't set any semantic state

        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
        ]

        result = cm._chronological_snip(messages, current_tokens=1500, input_limit=750)

        # System preserved
        assert result[0]["content"] == "system prompt"
        # Others truncated
        assert "TRUNCATED BY CONTEXT MONITOR" in result[1]["content"]
