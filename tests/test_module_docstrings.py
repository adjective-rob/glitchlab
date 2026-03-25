import glitchlab.display
import glitchlab.controller_utils
import glitchlab.doc_inserter


def test_module_docstrings_exist_and_contain_required_info():
    modules = [
        glitchlab.display,
        glitchlab.controller_utils,
        glitchlab.doc_inserter
    ]
    
    for mod in modules:
        doc = mod.__doc__
        assert doc is not None, f"Module {mod.__name__} is missing a docstring"
        
        doc_lower = doc.lower()
        assert "public api" in doc_lower, f"Module {mod.__name__} docstring missing 'Public API'"
        assert "controller decomposition" in doc_lower, f"Module {mod.__name__} docstring missing 'controller decomposition'"
        
        lines = [line for line in doc.strip().split('\n') if line.strip()]
        assert 1 <= len(lines) <= 5, f"Module {mod.__name__} docstring should be 3-5 lines, got {len(lines)}"
