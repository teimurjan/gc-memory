# lethe — Codex CLI plugin

Persistent memory across [Codex CLI](https://developers.openai.com/codex) sessions. Markdown-first storage, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```bash
# 1. Install the lethe binary
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli

# 2. Wire the hooks into Codex
git clone https://github.com/teimurjan/lethe /tmp/lethe   # or use a release tarball
bash /tmp/lethe/plugins/codex/install.sh --auto-config
```

`install.sh` copies the hooks and skills to `~/.codex/lethe/` and (with `--auto-config`) appends a marked block to `~/.codex/config.toml`. Re-running the installer replaces the existing block, so updates are idempotent.

Without `--auto-config` the script prints the snippet for you to paste manually.

After install, a `.lethe/` directory will appear in each project's git root on first use:

```
.lethe/
├── memory/            # source of truth — daily markdown files (git-diffable)
├── index/             # rebuildable DuckDB artifacts (safe to delete)
└── hooks.log          # only when LETHE_DEBUG=1
```

## How it works

| Event | Behavior |
|-------|----------|
| `SessionStart` | Injects the last ~30 lines from the 2 most recent daily files via `systemMessage`. |
| `UserPromptSubmit` | On the first prompt of a session, appends a `## Session HH:MM` heading; emits a `[lethe] Memory available` hint so the agent invokes `recall` (this project) or `recall-global` (all projects). |
| `Stop` | Appends a turn marker with a progressive-disclosure anchor pointing at Codex's transcript file, then reindexes `.lethe/memory`. |

### Limitations vs. the Claude Code plugin

- **No turn-level summarization in v1.** Codex's transcript format isn't yet documented here, so the Stop hook only writes an anchor (timestamp + transcript path) and reindexes — it does not summarize the turn into bullets. Once the transcript schema is settled this will land in a follow-up.
- **No SessionEnd hook** in Codex CLI. The header sentinel is reused across the session and not flushed; harmless in practice.

## Requirements

- `lethe` binary on PATH (`brew tap teimurjan/lethe && brew install lethe` or `cargo install lethe-cli`).
- Codex CLI with hooks enabled (`[features].codex_hooks = true`, set automatically by `--auto-config`).

## Debugging

Set `LETHE_DEBUG=1` in your shell. Hook traces land in `<project>/.lethe/hooks.log`.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Codex hooks docs: <https://developers.openai.com/codex/hooks>
