#!/usr/bin/env bash
# parse-transcript.sh TRANSCRIPT_PATH
#
# Extract the last user turn and the following assistant response from
# a Claude Code JSONL transcript. Prints a plain-text blob to stdout
# suitable for piping into a downstream LLM enrichment prompt.
#
# Delegates to `lethe transcript` (Rust binary) so the plugin stays
# Python-free. The output format is the SESSION_ID/TURN_ID/USER/
# ASSISTANT block the rest of the hook pipeline parses.

set -eu
set -o pipefail

TRANSCRIPT="${1:-}"
if [ -z "${TRANSCRIPT}" ] || [ ! -f "${TRANSCRIPT}" ]; then
  exit 0
fi

if ! command -v lethe-claude-code >/dev/null 2>&1; then
  exit 0
fi

lethe-claude-code transcript "${TRANSCRIPT}"
