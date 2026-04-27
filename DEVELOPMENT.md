# Development

## Setup

```bash
# Rust toolchain (1.94+)
rustup toolchain install stable

# Python venv for the legacy library + parity bench
uv venv --python 3.12
uv pip install -e 'legacy/[dev]'
```

The CLI is the Rust binary `lethe` (built from `crates/lethe-cli`). The Python package under `legacy/` is the original implementation, kept for the research trail and to back the parity bench. PyO3 bindings (`crates/lethe-py`) and napi-rs bindings (`crates/lethe-node`) are the supported language-binding paths going forward.

## Run tests

```bash
cargo test --workspace
cd legacy && uv run pytest tests/ -q
```

Rust: 71 unit tests, sub-second (plus the CLI smoke + cross-impl npz tests). Python: 148 production + 8 PyO3 parity = 156, ~3 minutes (the PyO3 set loads ONNX models). No network, no API keys required.

## Run the CLI locally

```bash
cargo run -p lethe-cli -- search "query"     # debug build, fast iteration
cargo install --path crates/lethe-cli        # install local build to ~/.cargo/bin
lethe                                         # opens TUI (if stdout is a terminal)
```

Common commands once installed:

```bash
lethe index                     # reindex .lethe/memory in the current repo
lethe search "query" --top-k 5
lethe search "query" --all      # cross-project via ~/.lethe/projects.json
lethe tui                       # explicit TUI (same as no-arg in a TTY)
lethe projects list
```

## Try the Claude Code plugin locally

Point Claude Code's marketplace at this checkout:

```
/plugin marketplace add /Users/you/path/to/lethe
/plugin install lethe
```

Hooks run `bash ${CLAUDE_PLUGIN_ROOT}/hooks/*.sh`; they invoke `lethe` from PATH. After `cargo install --path crates/lethe-cli`, the binary is on `~/.cargo/bin/lethe` and the hooks pick it up — no publish needed.

Turn on hook traces while iterating:

```bash
export LETHE_DEBUG=1   # writes to .lethe/hooks.log of the target repo
```

After editing `plugins/claude-code/` files (hooks, skills, manifest), run `/reload-plugins` in Claude Code.

## Building release artifacts

Everything is built in CI on native runners — ort's prebuilt ONNX
Runtime only links cleanly on the same platform it was compiled
for, and cross-compiling C++ from macOS hits libstdc++ / MSVC-runtime
ABI mismatches we don't want to fight. After the matrix builds, the
artifacts are committed to `release_artifacts/<tag>/` on main (via Git LFS so the
git history stays small) and attached to the GitHub Release.

### Supported targets

| Target | Friendly | Native runner |
|---|---|---|
| `aarch64-apple-darwin` | `macos-arm64` | `macos-14` |
| `x86_64-unknown-linux-gnu` | `linux-x64` | `ubuntu-latest` |
| `aarch64-unknown-linux-gnu` | `linux-arm64` | `ubuntu-24.04-arm` |
| `x86_64-pc-windows-msvc` | `windows-x64` | `windows-latest` |
| `aarch64-pc-windows-msvc` | `windows-arm64` | `windows-11-arm` |

**Intel Mac (`x86_64-apple-darwin`) is not supported.** Upstream `ort`
dropped Intel macOS in rc.11 and the minimum macOS was bumped to 13.4
([changelog](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.11));
there is no version going forward that ships prebuilt ONNX Runtime
for that target.

### Release flow end-to-end

The pipeline is built so that the registry-push workflows can't fire
before the binary artifacts exist. release-please creates the GitHub
Release as a **draft** (see `"draft": true` in
`release-please-config.json`); the only thing that flips it to
published is `release.yml` after every artifact is attached.

1. Land `feat:` / `fix:` commits on `main`. `release-please.yml`
   opens a PR bumping the workspace version everywhere.
2. Merge the release PR. `release-please.yml` runs again, tags the
   merge commit (`vX.Y.Z`), and creates a **draft** GitHub Release.
   No `release: published` event yet → registry-push workflows do
   not fire.
3. The tag push triggers `.github/workflows/release.yml`:
   - **build** matrix on five native runners. Each produces a
     tarball/zip, a napi `.node`, and a maturin wheel.
   - **commit** downloads every matrix artifact, places them under
     `release_artifacts/vX.Y.Z/`, and pushes a
     `chore(release): vX.Y.Z artifacts [skip ci]` commit to main.
     Git LFS handles the binary blobs (see `.gitattributes`).
   - **release** uploads the same artifacts to the draft Release,
     then runs `gh release edit --draft=false` — this is the moment
     the `release: published` event fires.
4. The published event finally fires `release-rust.yml`,
   `release-pypi.yml`, and `release-npm.yml`. They download the
   assets they need and push to crates.io / PyPI / npm / Homebrew.

If `release.yml` fails partway (e.g., a flaky linux-arm64 runner),
the GitHub Release stays in draft state and registry-push workflows
never run — you can re-trigger `release.yml` via `workflow_dispatch`
without rolling back the version bump.

### Local: the only build script you need

There's a `scripts/release/build.sh` that produces the macOS-arm64
tarball locally for sanity checks, but you do **not** need to run
it for a real release — CI does the same thing on a runner. The
script is useful when you want to test the binary without making a
release first:

```bash
scripts/release/build.sh           # binaries + tarball for macos-arm64
scripts/release/build.sh --napi    # plus the .node binding
scripts/release/build.sh --pypi    # plus the maturin wheel
```

Output lands directly in `release_artifacts/` (loose top-level files,
gitignored). The same directory holds CI-committed builds in
versioned subdirs `release_artifacts/vX.Y.Z/` (tracked via Git LFS).

### Git LFS

Once cloned, run `git lfs install` once per machine. Without it,
files in `release_artifacts/<tag>/` will fetch as small text pointers rather
than the real binaries. CI's `actions/checkout@v4` with `lfs: true`
handles it automatically on workflow runs.

## Commit conventions

Conventional commits. `release-please` only bumps on `feat:` / `fix:`. The workspace ships four artifacts on the same version (Rust binary via Homebrew/crates.io, `lethe-memory` wheel on PyPI, `lethe` on npm), so a `feat:` triggers releases everywhere — use it sparingly. Everything else (`chore:`, `docs:`, `refactor:`, `test:`) does not trigger a release. Breaking changes use `feat!:` or `fix!:`.
