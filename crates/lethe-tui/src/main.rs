//! `lethe-tui` — ratatui-based browser. Mirrors the Textual TUI
//! (`src/lethe/tui.py`).
//!
//! Phase 1 ships a hello message; the TUI itself lands in Phase 7.

fn main() {
    println!(
        "lethe-tui v{} (Phase 7 brings ratatui UI)",
        lethe_core::version()
    );
}
