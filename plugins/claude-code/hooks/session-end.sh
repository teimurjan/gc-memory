#!/usr/bin/env bash
# SessionEnd — trigger a final reindex so suppression state and any last
# markdown edits land in SQLite before the session is torn down.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true

if [ -n "${LETHE_CLI}" ]; then
  ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
fi

exit 0
