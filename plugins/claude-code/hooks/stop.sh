#!/usr/bin/env bash
# Stop — summarize the last turn via `claude -p --model haiku`, append the
# bullets to today's markdown file with a progressive-disclosure anchor, and
# reindex `.lethe/memory` so the next retrieval sees the new content.

set -eu
set -o pipefail

# shellcheck source=common.sh
source "${CLAUDE_PLUGIN_ROOT}/hooks/common.sh"

_read_stdin_with_timeout || true

# Prevent recursive Stop when `claude -p` itself emits a Stop event.
STOP_ACTIVE="$(_json_val stop_hook_active || true)"
if [ "${STOP_ACTIVE}" = "True" ] || [ "${STOP_ACTIVE}" = "true" ]; then
  exit 0
fi

TRANSCRIPT="$(_json_val transcript_path || true)"
SESSION_ID="$(_json_val session_id || true)"

if [ -z "${TRANSCRIPT}" ] || [ ! -f "${TRANSCRIPT}" ]; then
  _log "stop: no transcript_path"
  exit 0
fi

if ! command -v claude >/dev/null 2>&1; then
  _log "stop: claude CLI missing — skip enrichment"
  exit 0
fi

TURN_TEXT="$("${CLAUDE_PLUGIN_ROOT}/hooks/parse-transcript.sh" "${TRANSCRIPT}" || true)"
if [ -z "${TURN_TEXT}" ]; then
  _log "stop: empty turn text"
  exit 0
fi

# Extract anchor metadata from parse-transcript output, then strip those
# header lines before sending the transcript to the summarizer.
TURN_ID="$(printf '%s\n' "${TURN_TEXT}" | awk -F': ' '/^TURN_ID:/ {print $2; exit}')"
PARSED_SESSION="$(printf '%s\n' "${TURN_TEXT}" | awk -F': ' '/^SESSION_ID:/ {print $2; exit}')"
[ -z "${SESSION_ID}" ] && SESSION_ID="${PARSED_SESSION}"
TURN_BODY="$(printf '%s\n' "${TURN_TEXT}" | awk 'skip{print; next} /^---$/ {skip=1}')"

SYSTEM_PROMPT='You are summarizing a Claude Code turn for a long-running memory store.

Produce 2-5 terse markdown bullets capturing what was done, what decisions were made, and any durable facts worth remembering (file paths, tool names, key numbers). Skip pleasantries and chain-of-thought. Output raw bullets only — no preamble, no heading, no closing remarks.'

SUMMARY="$(printf '%s' "${TURN_BODY}" \
  | claude -p --model haiku --append-system-prompt "${SYSTEM_PROMPT}" 2>/dev/null || true)"

if [ -z "${SUMMARY}" ]; then
  _log "stop: summarizer produced no output"
  exit 0
fi

TODAY="$(date +%Y-%m-%d)"
NOW="$(date +%H:%M)"
TODAY_FILE="${LETHE_MEMORY_DIR}/${TODAY}.md"
mkdir -p "${LETHE_MEMORY_DIR}"
[ -f "${TODAY_FILE}" ] || printf '# %s\n\n' "${TODAY}" >"${TODAY_FILE}"

{
  printf '\n### %s\n' "${NOW}"
  printf '<!-- session:%s turn:%s transcript:%s -->\n' \
    "${SESSION_ID}" "${TURN_ID}" "${TRANSCRIPT}"
  printf '%s\n' "${SUMMARY}"
} >>"${TODAY_FILE}"

# Reindex synchronously so the next SessionStart / search sees the new entry.
if [ -n "${LETHE_CLI}" ]; then
  ( cd "${LETHE_GIT_ROOT}" && ${LETHE_CLI} index >/dev/null 2>&1 ) || true
fi

exit 0
