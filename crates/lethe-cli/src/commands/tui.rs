//! `lethe-rs tui` — exec the `lethe-tui` binary so the CLI binary
//! stays small and we don't pay TUI deps on every invocation.

use anyhow::Result;
use std::process::Command;

pub fn run() -> Result<i32> {
    // Look up `lethe-tui` on PATH first; fall back to a sibling binary
    // alongside the running `lethe-rs` (cargo install drops both into
    // the same directory).
    let candidate = which_or_sibling("lethe-tui");
    let mut cmd = Command::new(candidate);
    let status = cmd.status()?;
    Ok(status.code().unwrap_or(0))
}

fn which_or_sibling(name: &str) -> String {
    if let Some(p) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&p) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(name);
            if sibling.is_file() {
                return sibling.to_string_lossy().into_owned();
            }
        }
    }
    name.to_owned()
}
