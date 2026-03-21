#!/bin/bash
set -e

TASK_DIR="tasks_for_claude_code"
TASKS=(CCGL_TASK_12C CCGL_TASK_12D CCGL_TASK_12E CCGL_TASK_13A CCGL_TASK_13B CCGL_TASK_13C)

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