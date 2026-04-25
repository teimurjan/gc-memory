//! `lethe-rs` — Rust CLI counterpart to the Python `lethe` console-script.
//!
//! Phase 1 ships a `--version` flag only; subcommands land in Phase 6.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "lethe-rs",
    version,
    about = "Self-improving memory store for LLM agents (Rust port).",
    long_about = "Rust counterpart to the Python `lethe` CLI. \
                  Ships alongside, no replacement intended in v1."
)]
struct Cli {}

fn main() {
    let _ = Cli::parse();
    println!(
        "lethe-rs v{} (powered by lethe-core v{})",
        env!("CARGO_PKG_VERSION"),
        lethe_core::version(),
    );
}
