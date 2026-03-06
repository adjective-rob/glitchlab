"""
GLITCHLAB Auditor — Scanner (v2: Scout Edition)

Multi-layer codebase analysis engine.

Layer 1: Static Analysis (tree-sitter + regex) — fast, deterministic, no API calls
  - Missing docs, TODOs, complex functions (original)
  - Dead code detection (unused imports, unreachable code)
  - Error handling gaps (bare except, unwrap chains)
  - Code smells (deeply nested logic, magic numbers, duplicate blocks)
  - API inconsistencies (mixed return patterns, naming violations)

Layer 2: Dependency Vulnerability Scanning — OSV.dev integration
  - Reads lockfiles (Cargo.lock, package-lock.json, requirements.txt, go.sum)
  - Queries https://api.osv.dev/v1/query for known CVEs
  - Zero config, no API key needed

Layer 3: LLM-Powered Creative Analysis (Scout Brain) — requires API call
  - Reads repo structure + key files holistically
  - Proposes new features, architectural improvements, DX wins
  - Identifies implicit bugs from code patterns
  - Generates "opportunity" findings that humans wouldn't grep for

All findings are structured and sized for GLITCHLAB tasks.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Finding Types
# ---------------------------------------------------------------------------

FINDING_KINDS = [
    "missing_doc",
    "todo",
    "complex_function",
    "dead_code",
    "error_handling",
    "code_smell",
    "api_inconsistency",
    "test_gap",
    "dependency_vuln",
    "feature_opportunity",
    "bug_risk",
    "performance",
]


@dataclass
class Finding:
    """A single actionable finding in the codebase."""
    kind: str                    # One of FINDING_KINDS
    file: str                    # relative path from repo root
    line: int                    # 1-indexed line number
    symbol: str                  # function/struct name or relevant identifier
    description: str             # human-readable description
    severity: str = "low"        # "low" | "medium" | "high"
    context: str = ""            # surrounding code snippet for model context
    category: str = "cleanup"    # "bug", "security", "feature", "refactor", "cleanup", "docs", "test"
    effort: str = "small"        # "small" | "medium" | "large" — estimated task size


@dataclass
class ScanResult:
    """Results of a full repository scan."""
    repo_path: Path
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    languages_found: set[str] = field(default_factory=set)
    dependency_count: int = 0

    def by_file(self) -> dict[str, list[Finding]]:
        grouped: dict[str, list[Finding]] = {}
        for f in self.findings:
            grouped.setdefault(f.file, []).append(f)
        return grouped

    def by_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        categories: dict[str, int] = {}
        severities: dict[str, int] = {}
        for f in self.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
            categories[f.category] = categories.get(f.category, 0) + 1
            severities[f.severity] = severities.get(f.severity, 0) + 1
        return {
            "total": len(self.findings),
            "files_scanned": self.files_scanned,
            "by_kind": kinds,
            "by_category": categories,
            "by_severity": severities,
            "dependency_count": self.dependency_count,
        }


# ---------------------------------------------------------------------------
# Language Config
# ---------------------------------------------------------------------------

LANGUAGE_MAP = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".go": "go",
}


# ---------------------------------------------------------------------------
# Scanner — Layer 1: Static Analysis
# ---------------------------------------------------------------------------

class Scanner:
    """
    Multi-layer codebase scanner.

    Layer 1 (static): Always runs, no API calls.
    Layer 2 (deps): OSV.dev vulnerability scan, no API key needed.
    Layer 3 (scout brain): LLM-powered creative analysis, requires router.
    """

    def __init__(self, repo_path: Path, exclude_dirs: list[str] | None = None):
        self.repo_path = repo_path.resolve()
        self.exclude_dirs = set(exclude_dirs or [
            ".git", "target", "node_modules", ".glitchlab",
            ".context", "dist", "build", "__pycache__", "venv",
            "mcp", ".venv", "site-packages", ".tox", ".mypy_cache",
            ".pytest_cache", ".ruff_cache", "coverage", ".next",
        ])
        # Collected during scan for cross-file analysis
        self._all_exports: dict[str, set[str]] = {}  # file -> set of exported symbols
        self._all_imports: dict[str, set[str]] = {}  # file -> set of imported symbols
        self._public_functions: dict[str, list[str]] = {}  # file -> list of public fn names
        self._test_files: set[str] = set()

    def scan(
        self,
        layers: list[str] | None = None,
    ) -> ScanResult:
        """
        Run analysis and return aggregated findings.

        Args:
            layers: Which layers to run. Default: ["static", "deps"].
                    Add "scout" for LLM-powered creative analysis (requires router).
        """
        active_layers = set(layers or ["static", "deps"])
        result = ScanResult(repo_path=self.repo_path)

        # Pre-scan: identify test files for cross-referencing
        self._identify_test_files()

        for file_path in self._iter_source_files():
            rel = str(file_path.relative_to(self.repo_path))
            ext = file_path.suffix
            lang = LANGUAGE_MAP.get(ext)

            if lang:
                result.languages_found.add(lang)

            try:
                source = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            result.files_scanned += 1

            if "static" in active_layers:
                # Original checks
                result.findings.extend(self._check_missing_docs(file_path, rel, source, lang))
                result.findings.extend(self._check_todos(file_path, rel, source))
                result.findings.extend(self._check_complex_functions(file_path, rel, source, lang))

                # New v2 checks
                result.findings.extend(self._check_error_handling(rel, source, lang))
                result.findings.extend(self._check_dead_code(rel, source, lang))
                result.findings.extend(self._check_code_smells(rel, source, lang))
                result.findings.extend(self._check_test_gaps(rel, source, lang))

        if "deps" in active_layers:
            dep_findings, dep_count = self._scan_dependencies()
            result.findings.extend(dep_findings)
            result.dependency_count = dep_count

        return result

    def _identify_test_files(self) -> None:
        """Pre-scan to identify test files for coverage gap analysis."""
        for path in self._iter_source_files():
            rel = str(path.relative_to(self.repo_path))
            name = path.name.lower()
            parts = [p.lower() for p in path.parts]
            if (
                name.startswith("test_") or name.endswith("_test.py")
                or name.endswith(".test.ts") or name.endswith(".test.js")
                or name.endswith("_test.rs") or name.endswith("_test.go")
                or "tests" in parts or "test" in parts
                or "__tests__" in parts or "spec" in parts
            ):
                self._test_files.add(rel)
                # Read test file to find what symbols it references
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    # Collect function names being tested (heuristic)
                    for m in re.finditer(r'(?:test_|it\(|describe\(|fn\s+test_)(\w+)', content):
                        self._all_imports.setdefault(rel, set()).add(m.group(1))
                except Exception:
                    pass

    def _iter_source_files(self) -> Iterator[Path]:
        """Yield all source files, respecting exclusions."""
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if any(exc in path.parts for exc in self.exclude_dirs):
                continue
            if path.suffix in LANGUAGE_MAP:
                yield path

    # -----------------------------------------------------------------------
    # Check: Missing Doc Comments (original, preserved)
    # -----------------------------------------------------------------------

    def _check_missing_docs(
        self, file_path: Path, rel: str, source: str, lang: str | None
    ) -> list[Finding]:
        findings = []

        if lang == "rust":
            findings.extend(self._check_missing_docs_rust(rel, source))
        elif lang == "python":
            findings.extend(self._check_missing_docs_python(rel, source))

        return findings

    def _check_missing_docs_rust(self, rel: str, source: str) -> list[Finding]:
        findings = []
        lines = source.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not (stripped.startswith("pub fn ") or stripped.startswith("pub async fn ")):
                continue

            match = re.search(r"pub\s+(?:async\s+)?fn\s+(\w+)", stripped)
            if not match:
                continue
            fn_name = match.group(1)

            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1

            prev = lines[j].strip() if j >= 0 else ""
            if not prev.startswith("///"):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = "\n".join(lines[start:end])

                findings.append(Finding(
                    kind="missing_doc",
                    file=rel,
                    line=i + 1,
                    symbol=fn_name,
                    description=f"Public function `{fn_name}` is missing a /// doc comment",
                    severity="low",
                    context=context,
                    category="docs",
                    effort="small",
                ))

        return findings

    def _check_missing_docs_python(self, rel: str, source: str) -> list[Finding]:
        findings = []
        lines = source.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("def "):
                continue

            match = re.search(r"def\s+(\w+)", stripped)
            if not match:
                continue
            fn_name = match.group(1)
            if fn_name.startswith("_"):
                continue

            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            next_line = lines[j].strip() if j < len(lines) else ""
            if not (next_line.startswith('"""') or next_line.startswith("'''")):
                start = max(0, i - 1)
                end = min(len(lines), i + 4)
                context = "\n".join(lines[start:end])

                findings.append(Finding(
                    kind="missing_doc",
                    file=rel,
                    line=i + 1,
                    symbol=fn_name,
                    description=f"Public function `{fn_name}` is missing a docstring",
                    severity="low",
                    context=context,
                    category="docs",
                    effort="small",
                ))

        return findings

    # -----------------------------------------------------------------------
    # Check: TODO / FIXME Comments (original, preserved)
    # -----------------------------------------------------------------------

    def _check_todos(self, file_path: Path, rel: str, source: str) -> list[Finding]:
        findings = []
        lines = source.splitlines()
        pattern = re.compile(r"(TODO|FIXME|HACK|XXX)\s*[:\-]?\s*(.*)", re.IGNORECASE)

        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                kind_label = match.group(1).upper()
                message = match.group(2).strip()
                findings.append(Finding(
                    kind="todo",
                    file=rel,
                    line=i + 1,
                    symbol=kind_label,
                    description=f"{kind_label}: {message}" if message else f"{kind_label} comment at line {i+1}",
                    severity="medium" if kind_label == "FIXME" else "low",
                    context=line.strip(),
                    category="bug" if kind_label == "FIXME" else "cleanup",
                    effort="small",
                ))

        return findings

    # -----------------------------------------------------------------------
    # Check: Complex Functions (original, preserved)
    # -----------------------------------------------------------------------

    def _check_complex_functions(
        self, file_path: Path, rel: str, source: str, lang: str | None
    ) -> list[Finding]:
        findings = []

        if lang not in ("rust", "python", "go", "typescript"):
            return findings

        lines = source.splitlines()
        threshold = 60

        if lang == "rust":
            fn_pattern = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")
        elif lang == "python":
            fn_pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
        else:
            fn_pattern = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?func(?:tion)?\s+(\w+)")

        fn_start = None
        fn_name = None
        brace_depth = 0

        for i, line in enumerate(lines):
            match = fn_pattern.match(line)
            if match and fn_start is None:
                fn_name = match.group(1)
                fn_start = i
                brace_depth = 0

            if fn_start is not None:
                brace_depth += line.count("{") - line.count("}")
                if lang == "rust" and brace_depth <= 0 and i > fn_start:
                    length = i - fn_start + 1
                    if length > threshold:
                        findings.append(Finding(
                            kind="complex_function",
                            file=rel,
                            line=fn_start + 1,
                            symbol=fn_name or "unknown",
                            description=f"Function `{fn_name}` is {length} lines long (>{threshold})",
                            severity="medium",
                            context=f"Function spans lines {fn_start+1}–{i+1}",
                            category="refactor",
                            effort="medium",
                        ))
                    fn_start = None
                    fn_name = None

        return findings

    # -----------------------------------------------------------------------
    # NEW Check: Error Handling Gaps
    # -----------------------------------------------------------------------

    def _check_error_handling(self, rel: str, source: str, lang: str | None) -> list[Finding]:
        """Detect dangerous error handling patterns."""
        findings = []
        lines = source.splitlines()

        if lang == "python":
            for i, line in enumerate(lines):
                stripped = line.strip()

                # Bare except
                if stripped == "except:" or stripped == "except Exception:":
                    if i + 1 < len(lines) and "pass" in lines[i + 1].strip():
                        context = "\n".join(lines[max(0, i - 1):min(len(lines), i + 3)])
                        findings.append(Finding(
                            kind="error_handling",
                            file=rel,
                            line=i + 1,
                            symbol="bare_except",
                            description="Bare except with pass — errors are silently swallowed",
                            severity="medium",
                            context=context,
                            category="bug",
                            effort="small",
                        ))

                # Catching and ignoring exceptions (except X as e: pass)
                if re.match(r"except\s+\w+.*:", stripped) and not stripped.endswith("pass"):
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    if j < len(lines) and lines[j].strip() == "pass":
                        context = "\n".join(lines[max(0, i):min(len(lines), j + 2)])
                        findings.append(Finding(
                            kind="error_handling",
                            file=rel,
                            line=i + 1,
                            symbol="swallowed_exception",
                            description="Exception caught but silently discarded",
                            severity="low",
                            context=context,
                            category="bug",
                            effort="small",
                        ))

        elif lang == "rust":
            for i, line in enumerate(lines):
                stripped = line.strip()

                # .unwrap() chains — panic risk
                unwrap_count = stripped.count(".unwrap()")
                if unwrap_count >= 2:
                    findings.append(Finding(
                        kind="error_handling",
                        file=rel,
                        line=i + 1,
                        symbol="unwrap_chain",
                        description=f"Line has {unwrap_count} .unwrap() calls — panic risk in chain",
                        severity="medium",
                        context=stripped,
                        category="bug",
                        effort="small",
                    ))

                # .expect("") with empty or generic message
                if re.search(r'\.expect\(\s*""\s*\)', stripped):
                    findings.append(Finding(
                        kind="error_handling",
                        file=rel,
                        line=i + 1,
                        symbol="empty_expect",
                        description="`.expect(\"\")` gives no useful panic message",
                        severity="low",
                        context=stripped,
                        category="cleanup",
                        effort="small",
                    ))

        elif lang in ("typescript", "javascript"):
            for i, line in enumerate(lines):
                stripped = line.strip()

                # Empty catch blocks
                if re.match(r"}\s*catch\s*\(\w*\)\s*{\s*}", stripped):
                    findings.append(Finding(
                        kind="error_handling",
                        file=rel,
                        line=i + 1,
                        symbol="empty_catch",
                        description="Empty catch block — errors silently swallowed",
                        severity="medium",
                        context=stripped,
                        category="bug",
                        effort="small",
                    ))

        return findings

    # -----------------------------------------------------------------------
    # NEW Check: Dead Code Detection
    # -----------------------------------------------------------------------

    def _check_dead_code(self, rel: str, source: str, lang: str | None) -> list[Finding]:
        """Detect potentially dead or unused code."""
        findings = []
        lines = source.splitlines()

        if lang == "python":
            # Unused imports (heuristic: import at top, symbol never used in rest of file)
            import_pattern = re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+(.+)")
            for i, line in enumerate(lines):
                if i > 30:  # Only check imports in the first 30 lines
                    break
                m = import_pattern.match(line)
                if not m:
                    continue

                imported_raw = m.group(1).strip()
                # Handle "import X as Y" and "from X import Y, Z"
                symbols = []
                for part in imported_raw.split(","):
                    part = part.strip()
                    if " as " in part:
                        symbols.append(part.split(" as ")[-1].strip())
                    else:
                        symbols.append(part.split(".")[-1].strip())

                rest = "\n".join(lines[i + 1:])
                for sym in symbols:
                    if not sym or sym == "*":
                        continue
                    # Check if symbol is used anywhere in the rest of the file
                    if not re.search(r'\b' + re.escape(sym) + r'\b', rest):
                        findings.append(Finding(
                            kind="dead_code",
                            file=rel,
                            line=i + 1,
                            symbol=sym,
                            description=f"Import `{sym}` appears unused in {rel}",
                            severity="low",
                            context=line.strip(),
                            category="cleanup",
                            effort="small",
                        ))

        elif lang == "rust":
            # Detect #[allow(dead_code)] annotations — someone already knows it's dead
            for i, line in enumerate(lines):
                if "#[allow(dead_code)]" in line:
                    # Get the next non-empty line (the dead item)
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    next_line = lines[j].strip() if j < len(lines) else ""
                    symbol = "unknown"
                    m = re.search(r"(?:fn|struct|enum|const|static|type)\s+(\w+)", next_line)
                    if m:
                        symbol = m.group(1)
                    findings.append(Finding(
                        kind="dead_code",
                        file=rel,
                        line=i + 1,
                        symbol=symbol,
                        description=f"`{symbol}` is marked #[allow(dead_code)] — confirm and remove if truly unused",
                        severity="low",
                        context="\n".join(lines[i:min(len(lines), j + 2)]),
                        category="cleanup",
                        effort="small",
                    ))

        return findings

    # -----------------------------------------------------------------------
    # NEW Check: Code Smells
    # -----------------------------------------------------------------------

    def _check_code_smells(self, rel: str, source: str, lang: str | None) -> list[Finding]:
        """Detect code patterns that suggest maintainability issues."""
        findings = []
        lines = source.splitlines()

        # Deep nesting detection (any language) — 5+ levels of indentation
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            # Count leading whitespace
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Normalize to levels (4 spaces or 1 tab = 1 level)
            if "\t" in line[:indent]:
                level = line[:indent].count("\t")
            else:
                level = indent // 4

            if level >= 5 and stripped and not stripped.startswith(("#", "//", "*", "/*")):
                findings.append(Finding(
                    kind="code_smell",
                    file=rel,
                    line=i + 1,
                    symbol="deep_nesting",
                    description=f"Code is nested {level} levels deep — consider extracting helper functions",
                    severity="low",
                    context=stripped[:120],
                    category="refactor",
                    effort="medium",
                ))

        # Magic numbers (numeric literals outside of obvious contexts)
        if lang in ("python", "rust", "go", "typescript", "javascript"):
            magic_pattern = re.compile(r'(?<!["\'])\b(\d{3,})\b(?!["\'])')
            obvious = {"1000", "1024", "2048", "4096", "8192", "16384", "65535", "100", "200", "404", "500"}
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip comments, imports, constants, test files
                if any(stripped.startswith(p) for p in ("#", "//", "*", "/*", "import", "from", "use")):
                    continue
                if rel in self._test_files:
                    continue
                for m in magic_pattern.finditer(stripped):
                    num = m.group(1)
                    if num not in obvious and not stripped.startswith("const ") and "= " + num not in stripped.upper():
                        # Only flag the first magic number per line
                        findings.append(Finding(
                            kind="code_smell",
                            file=rel,
                            line=i + 1,
                            symbol="magic_number",
                            description=f"Magic number `{num}` — consider extracting to a named constant",
                            severity="low",
                            context=stripped[:120],
                            category="refactor",
                            effort="small",
                        ))
                        break

        # Duplicate string literals (3+ occurrences of the same string)
        if lang in ("python", "typescript", "javascript"):
            string_pattern = re.compile(r'["\']([^"\']{8,})["\']')
            string_counts: dict[str, list[int]] = {}
            for i, line in enumerate(lines):
                for m in string_pattern.finditer(line):
                    s = m.group(1)
                    string_counts.setdefault(s, []).append(i + 1)
            for s, occurrences in string_counts.items():
                if len(occurrences) >= 3:
                    findings.append(Finding(
                        kind="code_smell",
                        file=rel,
                        line=occurrences[0],
                        symbol="duplicate_string",
                        description=f'String `"{s[:40]}"` repeated {len(occurrences)} times — extract to constant',
                        severity="low",
                        context=f"Lines: {', '.join(str(o) for o in occurrences[:5])}",
                        category="refactor",
                        effort="small",
                    ))

        return findings

    # -----------------------------------------------------------------------
    # NEW Check: Test Coverage Gaps
    # -----------------------------------------------------------------------

    def _check_test_gaps(self, rel: str, source: str, lang: str | None) -> list[Finding]:
        """Identify public modules/functions that have no corresponding test file."""
        findings = []

        # Skip if this IS a test file
        if rel in self._test_files:
            return findings

        if lang == "python":
            # Check if there's a corresponding test file
            path = Path(rel)
            possible_test_files = [
                f"test_{path.name}",
                f"{path.stem}_test.py",
                f"tests/test_{path.name}",
                f"tests/{path.stem}_test.py",
            ]
            has_test = any(
                str(p) in self._test_files or any(t.endswith(str(p)) for t in self._test_files)
                for p in possible_test_files
            )

            if not has_test and not path.name.startswith("_"):
                # Count public functions to gauge importance
                pub_funcs = re.findall(r"^def\s+([a-zA-Z]\w*)", source, re.MULTILINE)
                pub_funcs = [f for f in pub_funcs if not f.startswith("_")]
                if len(pub_funcs) >= 2:  # Only flag files with 2+ public functions
                    findings.append(Finding(
                        kind="test_gap",
                        file=rel,
                        line=1,
                        symbol=path.stem,
                        description=f"`{path.name}` has {len(pub_funcs)} public functions but no test file",
                        severity="medium",
                        context=f"Public functions: {', '.join(pub_funcs[:8])}",
                        category="test",
                        effort="medium",
                    ))

        elif lang == "rust":
            # Check for #[cfg(test)] mod tests block
            if "#[cfg(test)]" not in source and "mod tests" not in source:
                pub_funcs = re.findall(r"pub\s+(?:async\s+)?fn\s+(\w+)", source)
                if len(pub_funcs) >= 2:
                    path = Path(rel)
                    findings.append(Finding(
                        kind="test_gap",
                        file=rel,
                        line=1,
                        symbol=path.stem,
                        description=f"`{path.name}` has {len(pub_funcs)} pub functions but no #[cfg(test)] module",
                        severity="medium",
                        context=f"Public functions: {', '.join(pub_funcs[:8])}",
                        category="test",
                        effort="medium",
                    ))

        return findings

    # -----------------------------------------------------------------------
    # Layer 2: Dependency Vulnerability Scanning (OSV.dev)
    # -----------------------------------------------------------------------

    def _scan_dependencies(self) -> tuple[list[Finding], int]:
        """Query OSV.dev for known vulnerabilities in project dependencies."""
        findings: list[Finding] = []
        total_deps = 0

        # Detect and parse lockfiles
        lockfile_parsers = [
            ("requirements.txt", self._parse_requirements_txt),
            ("Pipfile.lock", self._parse_pipfile_lock),
            ("poetry.lock", self._parse_poetry_lock),
            ("package-lock.json", self._parse_package_lock),
            ("Cargo.lock", self._parse_cargo_lock),
            ("go.sum", self._parse_go_sum),
        ]

        for filename, parser in lockfile_parsers:
            lockfile = self.repo_path / filename
            if not lockfile.exists():
                continue

            try:
                packages = parser(lockfile)
                total_deps += len(packages)
                logger.info(f"[SCOUT] Found {len(packages)} packages in {filename}")

                for pkg in packages:
                    vulns = self._query_osv(pkg)
                    for vuln in vulns:
                        severity = "high" if any(
                            s.get("type") == "CVSS_V3" and float(s.get("score", 0)) >= 7.0
                            for s in vuln.get("severity", [])
                        ) else "medium"

                        aliases = vuln.get("aliases", [])
                        cve = next((a for a in aliases if a.startswith("CVE-")), vuln.get("id", "unknown"))
                        summary = vuln.get("summary", "No description available")

                        findings.append(Finding(
                            kind="dependency_vuln",
                            file=filename,
                            line=1,
                            symbol=f"{pkg['name']}@{pkg.get('version', '?')}",
                            description=f"{cve}: {summary[:200]}",
                            severity=severity,
                            context=f"Package: {pkg['name']} {pkg.get('version', '?')}\nAdvisory: {vuln.get('id', '?')}\nAffected: {json.dumps(vuln.get('affected', [{}])[0].get('ranges', []), default=str)[:200]}",
                            category="security",
                            effort="medium" if severity == "high" else "small",
                        ))
            except Exception as e:
                logger.warning(f"[SCOUT] Failed to scan {filename}: {e}")

        return findings, total_deps

    def _query_osv(self, package: dict[str, str]) -> list[dict]:
        """Query the OSV.dev API for vulnerabilities affecting a specific package."""
        payload: dict[str, Any] = {
            "package": {
                "name": package["name"],
                "ecosystem": package["ecosystem"],
            }
        }
        if "version" in package:
            payload["version"] = package["version"]

        try:
            req = urllib.request.Request(
                "https://api.osv.dev/v1/query",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("vulns", [])
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            logger.debug(f"[SCOUT] OSV query failed for {package['name']}: {e}")
            return []

    # --- Lockfile Parsers ---

    def _parse_requirements_txt(self, path: Path) -> list[dict[str, str]]:
        packages = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            m = re.match(r"([a-zA-Z0-9_.-]+)\s*==\s*([^\s;#]+)", line)
            if m:
                packages.append({"name": m.group(1), "version": m.group(2), "ecosystem": "PyPI"})
        return packages

    def _parse_pipfile_lock(self, path: Path) -> list[dict[str, str]]:
        packages = []
        try:
            data = json.loads(path.read_text())
            for section in ("default", "develop"):
                for name, info in data.get(section, {}).items():
                    version = info.get("version", "").lstrip("=")
                    if version:
                        packages.append({"name": name, "version": version, "ecosystem": "PyPI"})
        except Exception:
            pass
        return packages

    def _parse_poetry_lock(self, path: Path) -> list[dict[str, str]]:
        packages = []
        current_name = None
        current_version = None
        for line in path.read_text().splitlines():
            if line.strip().startswith("name = "):
                current_name = line.split("=", 1)[1].strip().strip('"')
            elif line.strip().startswith("version = ") and current_name:
                current_version = line.split("=", 1)[1].strip().strip('"')
                packages.append({"name": current_name, "version": current_version, "ecosystem": "PyPI"})
                current_name = None
        return packages

    def _parse_package_lock(self, path: Path) -> list[dict[str, str]]:
        packages = []
        try:
            data = json.loads(path.read_text())
            # npm v2+ lockfile format
            for pkg_path, info in data.get("packages", {}).items():
                if not pkg_path:  # root package
                    continue
                name = pkg_path.split("node_modules/")[-1]
                version = info.get("version", "")
                if name and version:
                    packages.append({"name": name, "version": version, "ecosystem": "npm"})
            # npm v1 fallback
            if not packages:
                for name, info in data.get("dependencies", {}).items():
                    version = info.get("version", "")
                    if version:
                        packages.append({"name": name, "version": version, "ecosystem": "npm"})
        except Exception:
            pass
        return packages

    def _parse_cargo_lock(self, path: Path) -> list[dict[str, str]]:
        packages = []
        current_name = None
        for line in path.read_text().splitlines():
            if line.startswith("name = "):
                current_name = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("version = ") and current_name:
                version = line.split("=", 1)[1].strip().strip('"')
                packages.append({"name": current_name, "version": version, "ecosystem": "crates.io"})
                current_name = None
        return packages

    def _parse_go_sum(self, path: Path) -> list[dict[str, str]]:
        packages = []
        seen = set()
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                version = parts[1].split("/")[0].lstrip("v")
                key = f"{name}@{version}"
                if key not in seen:
                    seen.add(key)
                    packages.append({"name": name, "version": version, "ecosystem": "Go"})
        return packages

    # -----------------------------------------------------------------------
    # Layer 3: Scout Brain (LLM-Powered Creative Analysis)
    # -----------------------------------------------------------------------

    def scout_brain(self, router: Any) -> list[Finding]:
        """
        LLM-powered creative codebase analysis.

        Reads the repo structure and key files, then asks the model to
        identify feature opportunities, implicit bugs, architectural
        improvements, and DX wins that static analysis can't find.

        Args:
            router: A GLITCHLAB Router instance for LLM calls.

        Returns:
            List of creative findings (kind="feature_opportunity" or "bug_risk").
        """
        findings: list[Finding] = []

        # Build a structural overview of the codebase
        overview = self._build_repo_overview()
        if not overview:
            return findings

        prompt = f"""You are Scout, an elite autonomous codebase analyst. You think like a senior
engineer reviewing a codebase for the first time — looking for what's MISSING,
what's FRAGILE, and what would make this project significantly better.

## Repository Overview
{overview}

## Your Mission
Analyze this codebase and identify ACTIONABLE findings in these categories:

1. **Feature Opportunities** — What features are clearly missing or half-built?
   Look for: incomplete implementations, obvious next steps, missing CLI flags,
   missing API endpoints, configuration gaps, missing integrations.

2. **Bug Risks** — What code patterns will likely cause bugs under real usage?
   Look for: race conditions, unhandled edge cases, missing input validation,
   state management issues, off-by-one risks, null/None propagation.

3. **Performance Issues** — What will be slow or wasteful at scale?
   Look for: O(n^2) in loops, unbounded collections, missing caching,
   redundant I/O, sync operations that should be async.

4. **Architectural Improvements** — What structural changes would pay dividends?
   Look for: modules that should be split, missing abstractions, coupling
   that prevents testing, missing error boundaries.

## Response Format
Return a JSON array of findings. Each finding:
```json
[
  {{
    "kind": "feature_opportunity" | "bug_risk" | "performance",
    "file": "path/to/relevant/file.py",
    "line": 1,
    "symbol": "descriptive_identifier",
    "description": "Clear, specific, actionable description (1-2 sentences)",
    "severity": "low" | "medium" | "high",
    "category": "feature" | "bug" | "refactor" | "security",
    "effort": "small" | "medium" | "large"
  }}
]
```

Rules:
- Be SPECIFIC. Reference actual files and functions you see in the overview.
- Each finding must be actionable — a developer should know exactly what to build/fix.
- Prefer many small findings over few large ones.
- Don't suggest things that are obviously already implemented.
- Maximum 20 findings.
- Return ONLY the JSON array, no other text.
"""

        try:
            response = router.complete(
                role="scout",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            content = response.content.strip()

            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                content = "\n".join(lines)

            raw = json.loads(content)

            # Handle both {"findings": [...]} and [...] formats
            if isinstance(raw, dict):
                items = raw.get("findings", raw.get("items", []))
            elif isinstance(raw, list):
                items = raw
            else:
                items = []

            for item in items[:20]:  # Hard cap
                kind = item.get("kind", "feature_opportunity")
                if kind not in FINDING_KINDS:
                    kind = "feature_opportunity"
                findings.append(Finding(
                    kind=kind,
                    file=item.get("file", "unknown"),
                    line=item.get("line", 1),
                    symbol=item.get("symbol", "scout_finding"),
                    description=item.get("description", "Scout-identified improvement"),
                    severity=item.get("severity", "medium") if item.get("severity") in ("low", "medium", "high") else "medium",
                    context=item.get("context", ""),
                    category=item.get("category", "feature") if item.get("category") in ("bug", "security", "feature", "refactor", "cleanup", "docs", "test") else "feature",
                    effort=item.get("effort", "medium") if item.get("effort") in ("small", "medium", "large") else "medium",
                ))

            logger.info(f"[SCOUT] Brain identified {len(findings)} creative findings")

        except Exception as e:
            logger.warning(f"[SCOUT] Brain analysis failed: {e}")

        return findings

    def _build_repo_overview(self) -> str:
        """Build a structural overview of the repo for the Scout brain."""
        parts = []

        # File tree (limited depth)
        tree_lines = []
        for path in sorted(self.repo_path.rglob("*")):
            if not path.is_file():
                continue
            if any(exc in path.parts for exc in self.exclude_dirs):
                continue
            rel = path.relative_to(self.repo_path)
            if len(rel.parts) > 4:  # Skip deeply nested files
                continue
            tree_lines.append(str(rel))

        if not tree_lines:
            return ""

        parts.append("### File Tree")
        parts.append("\n".join(tree_lines[:150]))  # Cap at 150 files

        # Key file summaries (first 80 lines of important files)
        key_patterns = [
            "README.md", "Cargo.toml", "pyproject.toml", "package.json",
            "go.mod", "Makefile", "Dockerfile",
        ]
        for pattern in key_patterns:
            fpath = self.repo_path / pattern
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()[:80]
                    parts.append(f"\n### {pattern}")
                    parts.append("\n".join(lines))
                except Exception:
                    pass

        # Source file signatures (public API surface)
        sig_count = 0
        for path in sorted(self.repo_path.rglob("*")):
            if sig_count >= 30:  # Cap signature extraction
                break
            if not path.is_file() or path.suffix not in LANGUAGE_MAP:
                continue
            if any(exc in path.parts for exc in self.exclude_dirs):
                continue

            rel = str(path.relative_to(self.repo_path))
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Extract function/class signatures
            sigs = []
            for line in source.splitlines():
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in (
                    "pub fn ", "pub async fn ", "pub struct ", "pub enum ",
                    "def ", "class ", "async def ",
                    "export function ", "export class ", "export const ",
                    "func ", "type ",
                )):
                    sigs.append(f"  {stripped[:120]}")

            if sigs:
                sig_count += 1
                parts.append(f"\n### {rel} (signatures)")
                parts.append("\n".join(sigs[:30]))

        return "\n".join(parts)
