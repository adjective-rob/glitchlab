#!/bin/bash
set -e

TASK_DIR="tasks_for_claude_code"
TASKS=(CCGL_TASK_14 CCGL_TASK_15 CCGL_TASK_16 CCGL_TASK_17)

for task in "${TASKS[@]}"; do
  echo ""
  echo "========================================="
  echo "  Running: $task"
  echo "========================================="
  claude -p "$(cat ${TASK_DIR}/${task}.md)" --allowedTools Edit Bash

  git add -A
  git commit -m "glitchlab: ${task}" || echo "Nothing to commit"

  echo "✅ ${task} done"
done

echo ""
echo "========================================="
echo "  All tasks complete."
echo "========================================="
echo "  git push -u origin $(git branch --show-current)"