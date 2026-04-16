#!/usr/bin/env bash
# UserPromptSubmit — surface a one-line hint telling Claude that a memory
# store exists for this project, so the memory-recall skill can be invoked.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true

# Only hint when the CLI is reachable AND there's at least one memory file.
if [ -z "${LETHE_CLI}" ]; then
  exit 0
fi
if ! ls "${LETHE_MEMORY_DIR}"/*.md >/dev/null 2>&1; then
  exit 0
fi

printf '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[lethe] Memory available — use the memory-recall skill if prior-session context could help."}}'
exit 0
