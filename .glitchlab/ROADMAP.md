# Project Roadmap

## Now
- Auditor ouroboros: failure history injection, roadmap-aware prioritization
- Scanner noise reduction: skip missing_doc findings for test files (tests/ directory)
- Large file refactoring: break down cli.py, router.py, step_handlers.py, agent_runners.py
- Tier-based routing refactor: eliminate hardcoded model strings from config

## Next
- read_history tool for auditor agentic loop
- Forge mode: adversarial task decomposition engine
- ONNX complexity scorer integration

## Deferred
- Test file docstrings — low value, do not generate tasks for missing_doc in tests/
- Prelude v2 integration
