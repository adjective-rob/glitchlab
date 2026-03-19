"""
Scope Resolver (Layer 1) — Computed Context, Not Guessed

Extracted from controller.py. Computes precise file context for agents
based on actual dependency analysis rather than planner guesses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scope Resolver (Layer 1) — Computed Context, Not Guessed
# ---------------------------------------------------------------------------

class ScopeResolver:
    """
    Computes precise file context for agents based on actual
    dependency analysis rather than planner guesses.

    Replaces the old `gather_file_context` pattern that blindly
    read whatever the planner listed.
    """

    def __init__(self, working_dir: Path, repo_index: Any = None):
        self.working_dir = working_dir
        self.repo_index = repo_index

    def resolve_for_files(
        self,
        target_files: list[str],
        max_lines: int = 2000,
        include_deps: bool = True,
        signatures_only: bool = False,
    ) -> dict[str, str]:
        """
        Read target files + optionally resolve their local imports
        to provide dependency signatures.
        """
        context = {}

        for fpath in target_files:
            full = self.working_dir / fpath
            if not full.exists() or not full.is_file():
                continue

            try:
                if signatures_only:
                    line_count = len(full.read_text().splitlines())
                    sigs = self._extract_signatures(full)
                    if sigs:
                        context[fpath] = (
                            f"({line_count} lines)\n\n{sigs}\n\n"
                            "Use read_file or get_function for full content."
                        )
                    else:
                        context[fpath] = (
                            f"({line_count} lines) — no signatures extracted. "
                            "Use read_file for content."
                        )
                else:
                    lines = full.read_text().splitlines()
                    if len(lines) > max_lines:
                        content = "\n".join(lines[:max_lines]) + f"\n\n... truncated ({len(lines)} lines total)"
                    else:
                        content = "\n".join(lines)
                    context[fpath] = content
            except Exception as e:
                context[fpath] = f"(could not read: {e})"

            # Resolve local dependencies and include signatures only
            if include_deps:
                deps = self._resolve_imports(full)
                for dep_path, signatures in deps.items():
                    if dep_path not in context:
                        context[f"[dep] {dep_path}"] = signatures

        return context

    def _resolve_imports(self, file_path: Path) -> dict[str, str]:
        """
        Parse imports from a file and return signature summaries
        of local dependencies (not full file contents).
        """
        deps = {}
        try:
            content = file_path.read_text()
        except Exception:
            return deps

        suffix = file_path.suffix

        if suffix == ".py":
            deps = self._resolve_python_imports(content, file_path)
        elif suffix == ".rs":
            deps = self._resolve_rust_imports(content, file_path)
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            deps = self._resolve_js_imports(content, file_path)

        return deps

    def _extract_signatures(self, file_path: Path) -> str:
        """Extract signatures from a file based on its extension."""
        suffix = file_path.suffix
        if suffix == ".py":
            return self._extract_python_signatures(file_path)
        elif suffix == ".rs":
            return self._extract_rust_signatures(file_path)
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            return self._extract_js_signatures(file_path)
        return ""

    def _resolve_python_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local Python imports and return their signatures."""
        deps = {}
        for line in content.splitlines():
            line = line.strip()

            # Match: from glitchlab.foo import Bar
            match = re.match(r'^from\s+(glitchlab\.\S+)\s+import', line)
            if match:
                module = match.group(1).replace(".", "/") + ".py"
                dep_path = self.working_dir / module
                if dep_path.exists():
                    sigs = self._extract_python_signatures(dep_path)
                    if sigs:
                        deps[module] = sigs

        return deps

    def _resolve_rust_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local Rust use statements and return signatures."""
        deps = {}
        for line in content.splitlines():
            match = re.match(r'^use\s+crate::(\S+)', line.strip())
            if match:
                mod_path = match.group(1).replace("::", "/")
                # Try both mod.rs and direct .rs
                for candidate in [
                    self.working_dir / "src" / f"{mod_path}.rs",
                    self.working_dir / "src" / mod_path / "mod.rs",
                ]:
                    if candidate.exists():
                        sigs = self._extract_rust_signatures(candidate)
                        if sigs:
                            deps[str(candidate.relative_to(self.working_dir))] = sigs
                        break
        return deps

    def _resolve_js_imports(self, content: str, source: Path) -> dict[str, str]:
        """Extract local JS/TS imports and return signatures."""
        deps = {}
        for line in content.splitlines():
            match = re.match(r'''(?:import|from)\s+[^'"]*['"](\./[^'"]+)['"]''', line.strip())
            if match:
                rel = match.group(1)
                for ext in ["", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"]:
                    candidate = (source.parent / (rel + ext)).resolve()
                    if candidate.exists() and candidate.is_file():
                        sigs = self._extract_js_signatures(candidate)
                        if sigs:
                            deps[str(candidate.relative_to(self.working_dir))] = sigs
                        break
        return deps

    @staticmethod
    def _extract_python_signatures(path: Path) -> str:
        """Extract class/function signatures from a Python file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("class ", "def ", "async def ")):
                    lines.append(stripped.split(":")[0] + ":")
                elif stripped.startswith('"""') and lines:
                    lines.append(f"    {stripped}")
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _extract_rust_signatures(path: Path) -> str:
        """Extract pub fn/struct/enum signatures from a Rust file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("pub fn ", "pub async fn ", "pub struct ", "pub enum ", "pub trait ")):
                    lines.append(stripped.rstrip("{").strip())
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _extract_js_signatures(path: Path) -> str:
        """Extract export signatures from a JS/TS file."""
        lines = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("export ", "export default ")):
                    # Grab just the signature line
                    sig = stripped.split("{")[0].strip()
                    if sig:
                        lines.append(sig)
        except Exception:
            pass
        return "\n".join(lines) if lines else ""
