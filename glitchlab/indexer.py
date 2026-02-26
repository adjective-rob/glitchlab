"""
GLITCHLAB Repo Indexer — The Navigator

Philosophy: Context Routing, not Context Hoarding.
Walks the repository to build a "Route Map" of symbols and dependencies.
Agents use this to surgically request code instead of receiving repo dumps.

v2.0 Improvements:
  - Surgical Symbol Extraction (Regex-based for speed)
  - Import Discovery (Finds local dependencies)
  - Routing-First Agent Context (Encourages tool use)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".git", ".glitchlab", ".context", ".venv", "venv", "env",
    "node_modules", "target", "dist", "build", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", ".nuxt", "coverage", ".cargo", "vendor",
}

CODE_EXTENSIONS = {
    ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java",
}

KEY_FILES = {
    "Cargo.toml", "package.json", "pyproject.toml", "setup.py",
    "go.mod", "Makefile", "Dockerfile", "README.md",
}

# Regex patterns for surgical symbol extraction (no heavy AST required for indexing)
SYMBOL_PATTERNS = {
    ".py": [
        re.compile(r"^(?:class|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
    ],
    ".rs": [
        re.compile(r"^(?:pub\s+)?(?:struct|enum|trait|union|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
        re.compile(r"^(?:pub\s+)?type\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
    ],
    ".ts": [
        re.compile(r"^(?:export\s+)?(?:class|interface|type|function|const|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
    ],
}

# Patterns to find local imports (for Layer 3 Dependency Graphing)
IMPORT_PATTERNS = {
    ".py": re.compile(r"^(?:from|import)\s+([a-zA-Z0-9_\.]+)"),
    ".rs": re.compile(r"^(?:use)\s+([a-zA-Z0-9_:]+)"),
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SymbolEntry:
    name: str
    kind: str  # class, fn, struct, etc.
    line: int

@dataclass
class FileEntry:
    path: str
    extension: str
    symbols: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    is_test: bool = False
    is_key: bool = False

@dataclass
class RepoIndex:
    """A deterministic map of the repository structure and logic."""
    root: str
    files: Dict[str, FileEntry] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0

    def to_agent_context(self, max_files: int = 200) -> str:
        """
        Formats the index as a "Routing Map".
        Encourages the agent to use `read_file` or `get_symbol` tools.
        """
        parts = ["=== REPO ROUTE MAP ===", "Use this to identify where to surgicaly inject context.\n"]

        # 1. Summary
        lang_summary = ", ".join(f"{k}({v})" for k, v in sorted(self.languages.items(), key=lambda x: -x[1]))
        parts.append(f"Project Stats: {self.total_files} code files | Languages: {lang_summary}")

        # 2. Key Entry Points
        keys = [f.path for f in self.files.values() if f.is_key]
        if keys:
            parts.append(f"Entry Points: {', '.join(keys)}")

        # 3. Symbol Registry (The "Router" part)
        # We show symbols for the most important files to help the Planner point correctly
        parts.append("\nLogic Registry (Top Files & Symbols):")
        
        # Sort files by importance (Key files first, then by number of symbols)
        sorted_files = sorted(
            self.files.values(), 
            key=lambda x: (x.is_key, len(x.symbols)), 
            reverse=True
        )

        for entry in sorted_files[:max_files]:
            symbol_str = f" symbols: [{', '.join(entry.symbols[:10])}]" if entry.symbols else ""
            test_marker = " [TEST]" if entry.is_test else ""
            parts.append(f"  - {entry.path}{test_marker}{symbol_str}")

        if len(self.files) > max_files:
            parts.append(f"\n... and {len(self.files) - max_files} more files.")

        parts.append("\nINSTRUCTION: Do not guess code. Use `get_file(path)` to see full content.")
        return "\n".join(parts)

# ---------------------------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------------------------

def _harvest_metadata(file_path: Path, rel_path: str) -> tuple[List[str], List[str]]:
    """Surgically extract symbols and imports without full AST overhead."""
    symbols: List[str] = []
    imports: List[str] = []
    ext = file_path.suffix

    if ext not in SYMBOL_PATTERNS and ext not in IMPORT_PATTERNS:
        return [], []

    try:
        # Read only first 500 lines for indexing purposes to keep it fast
        content = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(500):
                line = f.readline()
                if not line: break
                content.append(line)
        
        full_text = "".join(content)

        # Extract Symbols
        for pattern in SYMBOL_PATTERNS.get(ext, []):
            matches = pattern.findall(full_text)
            symbols.extend(matches)

        # Extract Imports (to help build the Layer 3 dependency graph later)
        if ext in IMPORT_PATTERNS:
            import_matches = IMPORT_PATTERNS[ext].findall(full_text)
            imports.extend(import_matches)

    except Exception as e:
        logger.debug(f"[INDEX] Could not harvest {rel_path}: {e}")

    return sorted(list(set(symbols))), sorted(list(set(imports)))

def _is_test(path: str) -> bool:
    lower = path.lower()
    return any(x in lower for x in ["test", "spec", "__tests__"])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(repo_path: Path) -> RepoIndex:
    """
    Builds the GlitchLab v2 Navigator Index.
    Uses git ls-files for speed and respects .gitignore automatically.
    """
    repo_path = repo_path.resolve()
    index = RepoIndex(root=str(repo_path))

    # 1. Discovery (Deterministic via Git)
    try:
        raw_files = subprocess.check_output(
            ["git", "ls-files"], cwd=repo_path, text=True
        ).splitlines()
    except subprocess.CalledProcessError:
        logger.warning("[INDEX] Git not found, falling back to manual walk.")
        raw_files = [str(p.relative_to(repo_path)) for p in repo_path.rglob("*") if p.is_file()]

    # 2. Processing
    for rel_path in raw_files:
        if any(d in rel_path for d in SKIP_DIRS):
            continue

        full_path = repo_path / rel_path
        ext = full_path.suffix

        if ext in CODE_EXTENSIONS or rel_path in KEY_FILES:
            symbols, imports = [], []
            if ext in CODE_EXTENSIONS:
                symbols, imports = _harvest_metadata(full_path, rel_path)
                index.languages[ext] = index.languages.get(ext, 0) + 1
            
            index.files[rel_path] = FileEntry(
                path=rel_path,
                extension=ext,
                symbols=symbols,
                imports=imports,
                is_test=_is_test(rel_path),
                is_key=rel_path in KEY_FILES
            )

    index.total_files = len(index.files)
    logger.info(f"[INDEX] Navigator built: {index.total_files} files mapped.")
    return index