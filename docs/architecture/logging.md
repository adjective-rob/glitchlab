# Logging and Output Standards

All user-facing terminal output in GLITCHLAB must be prefixed with `[GLITCHLAB]`. 

### Implementation
When using `rich.console`, ensure that strings are either passed through the internal wrapper or manually prefixed. This ensures that in multi-process or verbose environments, GLITCHLAB messages are easily grep-able and visually distinct.