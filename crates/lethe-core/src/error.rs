//! Crate-wide error type. Mirrors the (informal) error surfaces of the
//! Python implementation: missing files, schema mismatches, malformed
//! configuration, and underlying I/O / DB failures.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("toml: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("duckdb: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[error("regex: {0}")]
    Regex(#[from] regex::Error),

    #[error("missing file {path}")]
    MissingFile { path: PathBuf },

    #[error("legacy SQLite index detected at {path}; run `lethe reset` then `lethe index`")]
    LegacySqlite { path: PathBuf },

    #[error("invalid config key {key}: {reason}")]
    InvalidConfig { key: String, reason: String },

    #[error("encoder error: {0}")]
    Encoder(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("not initialized: {0}")]
    NotInitialized(&'static str),

    #[error("locked: {0}")]
    Locked(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
