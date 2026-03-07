"""
GLITCHLAB Task History — Append-Only Run Log

Every GLITCHLAB run is recorded in .glitchlab/logs/history.json.
This provides:
  - A record of what was attempted and what succeeded/failed
  - Failure patterns that can inform future planning
  - Cost tracking over time
  - A foundation for adaptive agent routing

The log is append-only. Each entry is one JSON object per line (JSONL).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class TaskHistory:
    """
    Manages the append-only task history log.

    Storage: .glitchlab/logs/history.jsonl (one JSON object per line)
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        self.log_dir = self.repo_path / ".glitchlab" / "logs"
        self.history_file = self.log_dir / "history.jsonl"

    def record(self, result: dict[str, Any]) -> None:
        """
        Append a completed task run to the history log.

        Args:
            result: The full result dict from Controller.run()
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": result.get("task_id", "unknown"),
            "status": result.get("status", "unknown"),
            "pr_url": result.get("pr_url"),
            "branch": result.get("branch"),
            "error": result.get("error"),
            "budget": result.get("budget", {}),
            "quality_score": result.get("quality_score", {}),  # NEW: Persist the score
            "events_summary": self._summarize_events(result.get("events", [])),
        }

        try:
            with open(self.history_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.debug(f"[HISTORY] Recorded: {entry['task_id']} → {entry['status']}")
        except Exception as e:
            logger.warning(f"[HISTORY] Failed to write history: {e}")

    def get_recent(self, n: int = 10) -> list[dict]:
        """Read the most recent N history entries."""
        if not self.history_file.exists():
            return []

        try:
            lines = self.history_file.read_text().strip().splitlines()
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception as e:
            logger.warning(f"[HISTORY] Failed to read history: {e}")
            return []

    def get_failures(self, n: int = 20) -> list[dict]:
        """Read recent failures for pattern analysis."""
        recent = self.get_recent(n * 2)  # read more to filter
        return [
            e for e in recent
            if e.get("status") not in ("pr_created", "committed")
        ][:n]

    def get_stats(self) -> dict[str, Any]:
        """Compute summary statistics across all runs."""
        entries = self.get_all()
        if not entries:
            return {"total_runs": 0}

        total = len(entries)
        statuses = {}
        total_cost = 0.0
        total_tokens = 0
        total_quality = 0    # NEW
        scored_runs = 0      # NEW

        for entry in entries:
            s = entry.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1

            budget = entry.get("budget", {})
            total_cost += budget.get("estimated_cost", 0)
            total_tokens += budget.get("total_tokens", 0)

            # --- NEW: Aggregate quality ---
            q_score = entry.get("quality_score", {}).get("score")
            if q_score is not None:
                total_quality += q_score
                scored_runs += 1

        return {
            "total_runs": total,
            "statuses": statuses,
            "success_rate": round(
                statuses.get("pr_created", 0) / total * 100, 1
            ) if total > 0 else 0,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_cost_per_run": round(total_cost / total, 4) if total > 0 else 0,
            "avg_quality_score": round(total_quality / scored_runs, 1) if scored_runs > 0 else 0, # NEW
        }

    def get_all(self) -> list[dict]:
        """Read all history entries."""
        if not self.history_file.exists():
            return []
        try:
            lines = self.history_file.read_text().strip().splitlines()
            entries = []
            for line in lines:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception:
            return []

    def build_failure_context(self, max_entries: int = 5) -> str:
        """
        Build a context string from recent failures to inject into planner.
        Helps agents avoid repeating the same mistakes.
        """
        failures = self.get_failures(max_entries)
        if not failures:
            return ""

        parts = ["=== RECENT FAILURES (learn from these) ===\n"]
        for entry in failures:
            parts.append(
                f"- Task: {entry.get('task_id')} | "
                f"Status: {entry.get('status')} | "
                f"Error: {entry.get('error', 'N/A')}"
            )
            events = entry.get("events_summary", {})
            if events.get("security_verdict"):
                parts.append(f"  Security: {events['security_verdict']}")
            if events.get("fix_attempts"):
                parts.append(f"  Fix attempts: {events['fix_attempts']}")

        return "\n".join(parts)

    @staticmethod
    def _summarize_events(events: list[dict]) -> dict[str, Any]:
        """Extract key facts from the event log."""
        summary: dict[str, Any] = {}
        fix_attempts = 0

        for event in events:
            etype = event.get("type", "")
            data = event.get("data", {})

            if etype == "plan_created":
                summary["plan_steps"] = data.get("steps", 0)
                summary["plan_risk"] = data.get("risk")
            elif etype == "tests_failed":
                fix_attempts += 1
            elif etype == "tests_passed":
                summary["tests_passed_on_attempt"] = data.get("attempt", 1)
            elif etype == "security_review":
                summary["security_verdict"] = data.get("verdict")
            elif etype == "release_assessment":
                summary["version_bump"] = data.get("bump")

        summary["fix_attempts"] = fix_attempts
        return summary
    
    def record_patterns(self, task_id: str, patterns: list[dict]) -> None:
        """Append extracted tool-loop patterns to patterns.jsonl."""
        if not patterns:
            return
            
        patterns_file = self.log_dir / "patterns.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(patterns_file, "a", encoding="utf-8") as f:
                for p in patterns:
                    p["task_id"] = task_id
                    p["timestamp"] = datetime.now(timezone.utc).isoformat()
                    p["type"] = "discovery_pattern"
                    f.write(json.dumps(p) + "\n")
            
            # Cap at ~500 entries to prevent unbounded growth
            self._rotate_file_if_needed(patterns_file, max_lines=500)
        except Exception as e:
            logger.warning(f"[HISTORY] Failed to record patterns: {e}")

    def record_failure_detail(self, task_id: str, file_modified: str, error_type: str, resolution: str) -> None:
        """Capture specific debug loop failures and how they were resolved."""
        patterns_file = self.log_dir / "patterns.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "failure_resolution",
            "file_modified": file_modified,
            "error_type": error_type,
            "resolution": resolution
        }
        
        try:
            with open(patterns_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[HISTORY] Failed to record failure detail: {e}")

    def build_heuristics(self, files_in_scope: list[str]) -> str:
        """Filter and rank historical patterns relevant to the current files."""
        patterns_file = self.log_dir / "patterns.jsonl"
        if not patterns_file.exists() or not files_in_scope:
            return ""
            
        file_stats = {}
        failure_examples = []
        
        try:
            lines = patterns_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                target_file = data.get("file_modified")
                if target_file not in files_in_scope:
                    continue
                    
                ptype = data.get("type")
                if ptype == "discovery_pattern" and data.get("outcome") == "pass":
                    reads = data.get("files_read_first", [])
                    if reads:
                        if target_file not in file_stats:
                            file_stats[target_file] = {"runs": 0, "reads": {}}
                        
                        file_stats[target_file]["runs"] += 1
                        for r in reads:
                            file_stats[target_file]["reads"][r] = file_stats[target_file]["reads"].get(r, 0) + 1
                            
                elif ptype == "failure_resolution":
                    failure_examples.append(
                        f"- Avoid modifying {target_file} without addressing: {data.get('error_type')}. "
                        f"Previous resolution: {data.get('resolution')}"
                    )
        except Exception as e:
            logger.warning(f"[HISTORY] Error building heuristics: {e}")
            return ""

        # Format the output string
        parts = []
        for target, stats in file_stats.items():
            runs = stats["runs"]
            # Get the top 2 most common reads for this file
            top_reads = sorted(stats["reads"].items(), key=lambda x: -x[1])[:2]
            if top_reads:
                read_strs = [f"{f} ({count}/{runs} runs)" for f, count in top_reads]
                parts.append(f"- {target}: Usually requires reading {', '.join(read_strs)} first.")

        if not parts and not failure_examples:
            return ""

        result = "Known patterns from previous runs:\n"
        if parts:
            result += "\n".join(parts[:10]) + "\n" # Cap at 10
        if failure_examples:
            result += "\nFailure Contexts to Avoid:\n" + "\n".join(failure_examples[-3:]) # Cap at 3 recent failures
            
        return result

    def _rotate_file_if_needed(self, filepath: Path, max_lines: int) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > max_lines:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(lines[-int(max_lines * 0.8):]) # Keep the most recent 80%
        except Exception:
            pass

def extract_patterns_from_messages(messages: list[dict], outcome: str) -> list[dict]:
    """Pure function to extract discovery patterns and score tool sequences."""
    patterns = []
    files_read = set()
    tools_used = []
    
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                try:
                    name = tc.get("function", {}).get("name", "")
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    tools_used.append(name)
                    
                    if name in ("read_file", "search_grep", "find_references", "get_function"):
                        path = args.get("path") or args.get("pattern") or args.get("symbol")
                        if path:
                            files_read.add(path)
                            
                    elif name in ("write_file", "replace_in_file"):
                        file_modified = args.get("path")
                        if file_modified:
                            # --- NEW: Evaluate the "Right or Wrong" Sequence ---
                            seq_score = 100
                            
                            if not tools_used or tools_used[0] != "think":
                                seq_score -= 30  # Heavy penalty for acting without planning
                            
                            if not files_read:
                                seq_score -= 40  # Massive penalty for blind writes
                                
                            if "run_check" in tools_used:
                                seq_score += 10  # Reward for verifying changes
                                
                            patterns.append({
                                "file_modified": file_modified,
                                "files_read_first": list(files_read),
                                "tools_used": list(tools_used),
                                "outcome": outcome,
                                "sequence_score": max(0, min(100, seq_score)) # Constrain 0-100
                            })
                            # Reset discovery state after a write
                            files_read.clear()
                            tools_used.clear()
                except Exception:
                    continue
                    
    return patterns