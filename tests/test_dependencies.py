import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path

def test_security_dependencies_patched():
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject_path, 'rb') as f:
        data = tomllib.load(f)
    
    dependencies = data.get('project', {}).get('dependencies', [])
    
    # Check for specific patched versions
    assert any('litellm>=1.82.0' in dep.replace(' ', '') for dep in dependencies), "litellm must be >= 1.82.0 to patch CVE-2024-5751 and CVE-2024-6587"
    assert any('gitpython>=3.1.46' in dep.lower().replace(' ', '') for dep in dependencies), "GitPython must be >= 3.1.46 to patch CVE-2024-22190"
    assert any('loguru>=0.7.3' in dep.replace(' ', '') for dep in dependencies), "loguru must be >= 0.7.3 to patch CVE-2022-0338"
