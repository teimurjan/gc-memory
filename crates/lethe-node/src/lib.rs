//! Node.js bindings via napi-rs. The store + retrieve calls are
//! exposed as async functions so the Node event loop never blocks.
//!
//! Usage from TypeScript:
//!
//!     import { MemoryStore } from "@lethe/memory-rust";
//!     const store = await MemoryStore.open("./.lethe/index", {});
//!     await store.add("first entry");
//!     const hits = await store.retrieve("query", 5);

#![allow(unsafe_code)] // napi-derive expands to unsafe boilerplate.
#![allow(missing_debug_implementations)] // napi types are exposed to JS.

use std::sync::Arc;

use lethe_core::encoders::{BiEncoder as CoreBi, CrossEncoder as CoreCross};
use lethe_core::memory_store::{MemoryStore as CoreStore, StoreConfig};
use lethe_core::rif::RifConfig;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct StoreOptions {
    pub bi_encoder: Option<String>,
    pub cross_encoder: Option<String>,
    pub n_clusters: Option<u32>,
    pub use_rank_gap: Option<bool>,
    pub k_shallow: Option<u32>,
    pub k_deep: Option<u32>,
}

#[napi(object)]
pub struct Hit {
    pub id: String,
    pub content: String,
    pub score: f64,
}

#[napi]
pub struct MemoryStore {
    inner: Arc<CoreStore>,
}

fn map_err<E: std::fmt::Display>(e: E) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

#[napi]
impl MemoryStore {
    /// Open or create a store. `options` accepts encoder repo names
    /// and tuning knobs; defaults match the Python `MemoryStore`.
    #[napi(factory)]
    pub async fn open(path: String, options: Option<StoreOptions>) -> Result<Self> {
        let opts = options.unwrap_or(StoreOptions {
            bi_encoder: None,
            cross_encoder: None,
            n_clusters: None,
            use_rank_gap: None,
            k_shallow: None,
            k_deep: None,
        });
        let store = tokio::task::spawn_blocking(move || -> Result<CoreStore> {
            let bi_name = opts
                .bi_encoder
                .unwrap_or_else(|| "Xenova/all-MiniLM-L6-v2".to_owned());
            let cross_name = opts
                .cross_encoder
                .unwrap_or_else(|| "Xenova/ms-marco-MiniLM-L-6-v2".to_owned());
            let bi = Arc::new(CoreBi::from_repo(&bi_name).map_err(map_err)?);
            let cross = Arc::new(CoreCross::from_repo(&cross_name).map_err(map_err)?);
            let cfg = StoreConfig {
                dim: bi.dim(),
                k_shallow: opts.k_shallow.map_or(30, |v| v as usize),
                k_deep: opts.k_deep.map_or(100, |v| v as usize),
                rif: RifConfig {
                    n_clusters: opts.n_clusters.unwrap_or(0),
                    use_rank_gap: opts.use_rank_gap.unwrap_or(false),
                    ..RifConfig::default()
                },
                ..StoreConfig::default()
            };
            CoreStore::open(&path, Some(bi), Some(cross), cfg).map_err(map_err)
        })
        .await
        .map_err(map_err)??;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    #[napi]
    pub async fn add(&self, content: String) -> Result<Option<String>> {
        let store = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || store.add(&content, None, "", 0).map_err(map_err))
            .await
            .map_err(map_err)?
    }

    #[napi]
    pub async fn retrieve(&self, query: String, k: Option<u32>) -> Result<Vec<Hit>> {
        let store = Arc::clone(&self.inner);
        let k = k.unwrap_or(5) as usize;
        tokio::task::spawn_blocking(move || -> Result<Vec<Hit>> {
            let hits = store.retrieve(&query, k).map_err(map_err)?;
            Ok(hits
                .into_iter()
                .map(|h| Hit {
                    id: h.id,
                    content: h.content,
                    score: f64::from(h.score),
                })
                .collect())
        })
        .await
        .map_err(map_err)?
    }

    #[napi]
    pub async fn save(&self) -> Result<()> {
        let store = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || store.save().map_err(map_err))
            .await
            .map_err(map_err)?
    }

    #[napi]
    pub fn size(&self) -> u32 {
        self.inner.size() as u32
    }
}
