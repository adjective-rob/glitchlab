# AuditorAgent

The AuditorAgent is responsible for static and heuristic analysis of code generated within the GLITCHLAB environment.

### Responsibilities
- Identify performance bottlenecks.
- Detect 'smelly' code patterns.
- Provide actionable feedback to the primary controller.

### Integration
It is currently wired into `glitchlab/controller.py` and runs automatically during the evolution cycle.