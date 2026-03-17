"""Tests for ScopeResolver from glitchlab/controller.py."""

from glitchlab.controller import ScopeResolver


def test_resolve_for_files_reads_content(tmp_path):
    """resolve_for_files reads target files and returns their content."""
    (tmp_path / "main.py").write_text("print('hello')\n")

    resolver = ScopeResolver(working_dir=tmp_path)
    result = resolver.resolve_for_files(["main.py"])

    assert "main.py" in result
    assert "print('hello')" in result["main.py"]


def test_resolve_for_files_truncates_long_files(tmp_path):
    """resolve_for_files truncates files exceeding max_lines."""
    lines = [f"line {i}" for i in range(100)]
    (tmp_path / "big.py").write_text("\n".join(lines))

    resolver = ScopeResolver(working_dir=tmp_path)
    result = resolver.resolve_for_files(["big.py"], max_lines=10)

    assert "big.py" in result
    assert "truncated" in result["big.py"]
    assert "100 lines total" in result["big.py"]
    # Should contain the first 10 lines
    assert "line 0" in result["big.py"]
    assert "line 9" in result["big.py"]


def test_resolve_for_files_include_deps_follows_python_imports(tmp_path):
    """resolve_for_files with include_deps=True follows Python imports."""
    # Create the module being imported
    mod_dir = tmp_path / "glitchlab"
    mod_dir.mkdir()
    (mod_dir / "foo.py").write_text(
        "class Bar:\n    pass\n\ndef baz():\n    return 1\n"
    )

    # Create the file that imports it
    (tmp_path / "main.py").write_text("from glitchlab.foo import Bar\n")

    resolver = ScopeResolver(working_dir=tmp_path)
    result = resolver.resolve_for_files(["main.py"], include_deps=True)

    assert "main.py" in result
    # Should have a [dep] entry for the imported module
    dep_keys = [k for k in result if k.startswith("[dep]")]
    assert len(dep_keys) > 0
    dep_content = result[dep_keys[0]]
    assert "class Bar:" in dep_content
    assert "def baz():" in dep_content


def test_resolve_for_files_include_deps_false_skips_deps(tmp_path):
    """resolve_for_files with include_deps=False skips dependency resolution."""
    mod_dir = tmp_path / "glitchlab"
    mod_dir.mkdir()
    (mod_dir / "foo.py").write_text("class Bar:\n    pass\n")

    (tmp_path / "main.py").write_text("from glitchlab.foo import Bar\n")

    resolver = ScopeResolver(working_dir=tmp_path)
    result = resolver.resolve_for_files(["main.py"], include_deps=False)

    assert "main.py" in result
    dep_keys = [k for k in result if k.startswith("[dep]")]
    assert len(dep_keys) == 0


def test_missing_files_skipped(tmp_path):
    """Missing or nonexistent target files are skipped gracefully."""
    resolver = ScopeResolver(working_dir=tmp_path)
    result = resolver.resolve_for_files(["nonexistent.py", "also_missing.rs"])

    assert len(result) == 0


def test_extract_python_signatures(tmp_path):
    """_extract_python_signatures returns class/def lines only."""
    target = tmp_path / "module.py"
    target.write_text(
        "import os\n"
        "\n"
        "class MyClass:\n"
        '    """A docstring."""\n'
        "    x = 1\n"
        "\n"
        "def my_func(a, b):\n"
        "    return a + b\n"
        "\n"
        "async def async_func():\n"
        "    pass\n"
    )

    sigs = ScopeResolver._extract_python_signatures(target)

    assert "class MyClass:" in sigs
    assert "def my_func(a, b):" in sigs
    assert "async def async_func():" in sigs
    # Should not contain implementation lines
    assert "import os" not in sigs
    assert "x = 1" not in sigs
    assert "return a + b" not in sigs


def test_extract_rust_signatures(tmp_path):
    """_extract_rust_signatures returns pub fn/struct/enum lines only."""
    target = tmp_path / "lib.rs"
    target.write_text(
        "use std::io;\n"
        "\n"
        "pub fn process(data: &str) -> Result<(), Error> {\n"
        "    Ok(())\n"
        "}\n"
        "\n"
        "pub struct Config {\n"
        "    name: String,\n"
        "}\n"
        "\n"
        "pub enum Status {\n"
        "    Active,\n"
        "    Inactive,\n"
        "}\n"
        "\n"
        "fn private_helper() {\n"
        "}\n"
    )

    sigs = ScopeResolver._extract_rust_signatures(target)

    assert "pub fn process(data: &str) -> Result<(), Error>" in sigs
    assert "pub struct Config" in sigs
    assert "pub enum Status" in sigs
    # Private functions should not appear
    assert "private_helper" not in sigs
    assert "use std::io" not in sigs


def test_extract_js_signatures(tmp_path):
    """_extract_js_signatures returns export lines only."""
    target = tmp_path / "index.ts"
    target.write_text(
        "import { foo } from './foo';\n"
        "\n"
        "export function greet(name: string): string {\n"
        "    return `Hello ${name}`;\n"
        "}\n"
        "\n"
        "export default class App {\n"
        "    run() {}\n"
        "}\n"
        "\n"
        "const internal = 42;\n"
    )

    sigs = ScopeResolver._extract_js_signatures(target)

    assert "export function greet(name: string): string" in sigs
    assert "export default class App" in sigs
    assert "internal" not in sigs
    assert "import" not in sigs


def test_resolve_python_imports_finds_local_modules(tmp_path):
    """_resolve_python_imports finds local module files."""
    mod_dir = tmp_path / "glitchlab"
    mod_dir.mkdir()
    (mod_dir / "utils.py").write_text("def helper():\n    pass\n")

    content = "from glitchlab.utils import helper\nimport os\n"
    source = tmp_path / "main.py"
    source.write_text(content)

    resolver = ScopeResolver(working_dir=tmp_path)
    deps = resolver._resolve_python_imports(content, source)

    assert "glitchlab/utils.py" in deps
    assert "def helper():" in deps["glitchlab/utils.py"]


def test_unreadable_file_returns_error_entry(tmp_path):
    """Files that can't be read return a (could not read: ...) entry."""
    # Use a path that exists as a directory to trigger a read error,
    # since os.chmod(0o000) doesn't block root.
    target = tmp_path / "locked.py"
    target.mkdir()  # A directory, not a file — read_text() will fail

    resolver = ScopeResolver(working_dir=tmp_path)
    # The file exists check passes (it's a path that exists) but it's not a file,
    # so resolve_for_files skips it. Instead, test via a binary file that causes
    # a UnicodeDecodeError.
    bin_file = tmp_path / "binary.py"
    bin_file.write_bytes(b"\x80\x81\x82\x83\xff\xfe")

    result = resolver.resolve_for_files(["binary.py"])

    assert "binary.py" in result
    assert "(could not read:" in result["binary.py"]
