import os

def test_dependencies_updated():
    # Check requirements.txt
    with open('requirements.txt', 'r') as f:
        reqs = f.read()
    
    assert 'pydantic>=2.12.5' in reqs
    assert 'pyyaml>=6.0.3' in reqs
    assert 'httpx>=0.28.1' in reqs
    assert 'python-dotenv>=1.2.2' in reqs
    
    # Check pyproject.toml
    with open('pyproject.toml', 'r') as f:
        pyproject = f.read()
        
    assert 'pydantic>=2.12.5' in pyproject
    assert 'pyyaml>=6.0.3' in pyproject
    assert 'httpx>=0.28.1' in pyproject
    assert 'python-dotenv>=1.2.2' in pyproject
