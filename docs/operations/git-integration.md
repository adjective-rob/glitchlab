# Git Integration

GLITCHLAB interacts with Git to maintain environment parity.

## Pre-task Fetch
To prevent planning against stale code, the Controller performs a `git fetch origin main` at the start of every task. 

- **Soft Fail**: If the network is unavailable or the directory is not a git repository, the system logs a warning and proceeds with local state.
- **No Side Effects**: This operation only updates remote tracking branches (`origin/main`) and does not modify the working directory or the current branch head.