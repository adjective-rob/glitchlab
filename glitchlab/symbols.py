"""
GLITCHLAB AST Layer — Symbol Index

Provides AST-aware search and extraction for the agents using tree-sitter.
Allows precise "find references" and "get function body" without blind grepping.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

LANGUAGE_MAP = {
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}

SKIP_DIRS = {
    ".git", ".glitchlab", ".context", ".venv", "venv", "env",
    "node_modules", "target", "dist", "build", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


class SymbolIndex:
    """Lazily builds and queries a local AST map of the workspace."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self._cache = {}  # rel_path -> (tree, source_bytes, lang)
        self._scanned = False

    def _scan_workspace(self) -> None:
        """Lazily parse all supported code files on first query."""
        if self._scanned or not TREE_SITTER_AVAILABLE:
            return
            
        logger.debug("[SYMBOLS] Building lazy AST index...")
        for f in self.workspace_dir.rglob("*"):
            if f.is_file() and f.suffix in LANGUAGE_MAP:
                if not any(part in SKIP_DIRS for part in f.parts):
                    self._load_file(f)
        self._scanned = True

    def _load_file(self, path: Path) -> tuple | None:
        lang_name = LANGUAGE_MAP.get(path.suffix)
        if not lang_name or not TREE_SITTER_AVAILABLE:
            return None
            
        try:
            parser = get_parser(lang_name)
            source = path.read_bytes()
            tree = parser.parse(source)
            rel_path = str(path.relative_to(self.workspace_dir))
            self._cache[rel_path] = (tree, source, lang_name)
            return self._cache[rel_path]
        except Exception as e:
            logger.debug(f"[SYMBOLS] Failed to parse {path.name}: {e}")
            return None

    def invalidate(self, path: str) -> None:
        """Clear a file from the cache so it gets re-parsed on next query."""
        if path in self._cache:
            del self._cache[path]
            logger.debug(f"[SYMBOLS] Invalidated AST cache for {path}")
            
        # Re-parse immediately if we've already scanned
        if self._scanned:
            full_path = self.workspace_dir / path
            if full_path.exists():
                self._load_file(full_path)

    def find_references(self, symbol: str, language: str | None = None) -> list[dict]:
        """Find structural references of a symbol (ignores comments/strings)."""
        self._scan_workspace()
        results = []
        if not TREE_SITTER_AVAILABLE:
            return results

        target_bytes = symbol.encode('utf8')

        for path, (tree, source, lang) in self._cache.items():
            if language and lang != language:
                continue

            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                
                # Identify identifier nodes
                if node.type in ('identifier', 'name', 'type_identifier', 'property_identifier') and node.text == target_bytes:
                    parent = node.parent
                    ptype = parent.type if parent else ""
                    
                    # Deduce context kind
                    kind = "reference"
                    if "call" in ptype:
                        kind = "call"
                    elif "import" in ptype or "use" in ptype:
                        kind = "import"
                    elif any(x in ptype for x in ["def", "decl", "class", "struct", "function", "method"]):
                        kind = "definition"

                    line_idx = node.start_point[0]
                    lines = source.split(b'\n')
                    context_str = lines[line_idx].decode('utf8', errors='ignore').strip()

                    results.append({
                        "file": path,
                        "line": line_idx + 1,
                        "context": context_str,
                        "kind": kind
                    })

                stack.extend(node.children)

        return results

    def get_function_body(self, symbol: str, file: str | None = None) -> dict | None:
        """Extract the full block of a function or method by name."""
        self._scan_workspace()
        if not TREE_SITTER_AVAILABLE:
            return None
            
        target_bytes = symbol.encode('utf8')

        for path, (tree, source, lang) in self._cache.items():
            if file and path != file:
                continue

            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                
                # Look for function/method definitions
                if "function" in node.type or "method" in node.type:
                    # Check if any immediate child is the identifier we want
                    for child in node.children:
                        if child.type in ('identifier', 'name', 'property_identifier') and child.text == target_bytes:
                            lines = source.split(b'\n')
                            start_line = node.start_point[0]
                            end_line = node.end_point[0]
                            body = b'\n'.join(lines[start_line:end_line+1]).decode('utf8', errors='ignore')
                            
                            return {
                                "file": path,
                                "line_start": start_line + 1,
                                "line_end": end_line + 1,
                                "signature": body.split('\n')[0].strip(),
                                "body": body,
                                "decorators": []
                            }
                            
                stack.extend(node.children)
                
        return None

    def get_class_outline(self, class_name: str, file: str | None = None) -> dict | None:
        """Extract a class body with method signatures shown but bodies collapsed."""
        self._scan_workspace()
        if not TREE_SITTER_AVAILABLE:
            return None

        target_bytes = class_name.encode('utf8')

        for path, (tree, source, lang) in self._cache.items():
            if file and path != file:
                continue

            stack = [tree.root_node]
            while stack:
                node = stack.pop()

                if "class" in node.type and "definition" in node.type:
                    # Check if any immediate child is the identifier we want
                    for child in node.children:
                        if child.type in ('identifier', 'name') and child.text == target_bytes:
                            lines = source.split(b'\n')
                            start_line = node.start_point[0]
                            end_line = node.end_point[0]

                            # Build outline: show full class signature + method signatures with bodies collapsed
                            outline_lines = []
                            in_method = False
                            method_indent = 0

                            for i in range(start_line, end_line + 1):
                                line = lines[i].decode('utf8', errors='ignore')
                                stripped = line.strip()

                                # Detect method/function definitions inside the class
                                if stripped.startswith(('def ', 'async def ')):
                                    if in_method:
                                        outline_lines.append(f"{' ' * method_indent}    ...")
                                    outline_lines.append(f"{i + 1}: {line}")
                                    in_method = True
                                    method_indent = len(line) - len(line.lstrip())
                                elif not in_method:
                                    # Class-level code (decorators, class vars, docstrings before first method)
                                    outline_lines.append(f"{i + 1}: {line}")

                            if in_method:
                                outline_lines.append(f"{' ' * method_indent}    ...")

                            return {
                                "file": path,
                                "line_start": start_line + 1,
                                "line_end": end_line + 1,
                                "outline": "\n".join(outline_lines),
                            }

                stack.extend(node.children)

        return None