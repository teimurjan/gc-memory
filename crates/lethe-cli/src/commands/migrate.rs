//! `lethe migrate` — convert a legacy Python `embeddings.npz` into the
//! Rust-native `entry_embeddings` DuckDB table. One-shot per project.
//!
//! Behavior:
//!   * No-op (exit 0) when the npz is absent or already migrated.
//!   * Reads ids + f32 vectors from `<index>/embeddings.npz` and writes
//!     them in one transactional batch to the DuckDB store.
//!   * Sweeps every Python-era leftover from the index folder
//!     (`embeddings.npz`, `embeddings.npz.bak`, `query_embeddings.npz`,
//!     `faiss.index`) so the Rust runtime sees only Rust-native files.
//!   * `--all` iterates every registered project in `~/.lethe/projects.json`.

use std::path::Path;

use anyhow::{Context, Result};
use lethe_core::db::MemoryDb;
use lethe_core::npz;
use lethe_core::registry;
use rayon::prelude::*;
use serde::Serialize;

use crate::paths::resolve;

#[derive(Serialize, Default)]
struct Migrated {
    projects: Vec<ProjectResult>,
}

#[derive(Serialize)]
struct ProjectResult {
    root: String,
    status: String,
    entries: usize,
}

pub fn run(root: Option<&str>, all: bool, json_output: bool) -> Result<i32> {
    let mut out = Migrated::default();

    if all {
        let entries = registry::load();
        if entries.is_empty() {
            eprintln!("no projects registered");
            return Ok(1);
        }
        // Per-project migrations are independent (separate DuckDB +
        // npz files); fan out across cores.
        out.projects = entries.par_iter().map(|e| migrate_one(&e.root)).collect();
    } else {
        let paths = resolve(root);
        out.projects.push(migrate_one(&paths.root));
    }

    if json_output {
        println!("{}", serde_json::to_string(&out)?);
    } else {
        for p in &out.projects {
            println!("{} — {} ({} entries)", p.root, p.status, p.entries);
        }
    }
    let any_failed = out.projects.iter().any(|p| p.status == "error");
    Ok(i32::from(any_failed))
}

fn migrate_one(project_root: &Path) -> ProjectResult {
    let root_str = project_root.to_string_lossy().into_owned();
    match try_migrate(project_root) {
        Ok((status, entries)) => ProjectResult {
            root: root_str,
            status,
            entries,
        },
        Err(e) => {
            eprintln!("[lethe] {}: {e}", project_root.display());
            ProjectResult {
                root: root_str,
                status: "error".into(),
                entries: 0,
            }
        }
    }
}

/// Files in `.lethe/index/` that the legacy Python implementation
/// wrote but the Rust runtime never reads. Removed on successful
/// migrate so the index folder reflects only Rust-native artifacts.
const LEGACY_INDEX_LEFTOVERS: &[&str] = &[
    "embeddings.npz",
    "embeddings.npz.bak",
    "query_embeddings.npz",
    "faiss.index",
];

fn try_migrate(project_root: &Path) -> Result<(String, usize)> {
    let index = project_root.join(".lethe").join("index");
    let duckdb_path = index.join("lethe.duckdb");
    let npz_path = index.join("embeddings.npz");

    if !duckdb_path.exists() {
        return Ok(("no-store".into(), 0));
    }

    let db = MemoryDb::open(&duckdb_path).context("open lethe.duckdb")?;
    let already = !db.embeddings_empty()?;

    let count = if !already && npz_path.exists() {
        let map = npz::load_embeddings(&npz_path).context("read embeddings.npz")?;
        let n = map.len();
        let items: Vec<_> = map.into_iter().collect();
        db.save_embeddings_bulk(&items)
            .context("write entry_embeddings")?;
        n
    } else {
        0
    };

    // Sweep legacy artifacts the Rust runtime never opens. Idempotent.
    let mut removed: Vec<&str> = Vec::new();
    for name in LEGACY_INDEX_LEFTOVERS {
        let p = index.join(name);
        if p.exists() {
            std::fs::remove_file(&p).with_context(|| format!("remove leftover {}", p.display()))?;
            removed.push(name);
        }
    }

    let status = if count > 0 {
        "migrated"
    } else if already {
        "already-migrated"
    } else if removed.is_empty() {
        "no-npz"
    } else {
        "cleaned-up"
    };
    Ok((status.into(), count))
}
