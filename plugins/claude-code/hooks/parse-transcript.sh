#!/usr/bin/env bash
# parse-transcript.sh TRANSCRIPT_PATH
#
# Extract the last user turn and the following assistant response from a
# Claude Code JSONL transcript. Prints a plain-text blob to stdout suitable
# for piping into a downstream LLM enrichment prompt.

set -eu
set -o pipefail

TRANSCRIPT="${1:-}"
if [ -z "${TRANSCRIPT}" ] || [ ! -f "${TRANSCRIPT}" ]; then
  exit 0
fi

python3 - "${TRANSCRIPT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

def message_text(msg):
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif btype == "tool_use":
                name = block.get("name", "?")
                parts.append(f"[tool_use: {name}]")
            elif btype == "tool_result":
                parts.append("[tool_result]")
        return "\n".join(parts)
    return ""


user_turn = None
assistant_turn = None
user_id = None
session_id = None

with path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message") or rec
        role = msg.get("role") or rec.get("type")
        if role == "user":
            text = message_text(msg)
            if text:
                user_turn = text
                user_id = rec.get("uuid") or rec.get("id")
                session_id = rec.get("sessionId") or session_id
                assistant_turn = None
        elif role == "assistant":
            text = message_text(msg)
            if text:
                assistant_turn = text
                session_id = rec.get("sessionId") or session_id

if not user_turn and not assistant_turn:
    sys.exit(0)

print(f"SESSION_ID: {session_id or ''}")
print(f"TURN_ID: {user_id or ''}")
print("---")
print("USER:")
print((user_turn or "").strip())
print("---")
print("ASSISTANT:")
print((assistant_turn or "").strip())
PY
