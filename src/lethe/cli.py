"""Command-line entry point for ``lethe``.

Thin dispatcher over :class:`~lethe.memory_store.MemoryStore` and
:class:`~lethe.markdown_store.MarkdownStore`. Designed to be called as a
subprocess from the Claude Code plugin (hooks + memory-recall skill), but
also usable directly:

    lethe index                 # reindex .lethe/memory
    lethe search "query"        # hybrid + RIF + xenc retrieval
    lethe expand <chunk-id>     # full markdown section for a hit
    lethe status                # diagnostic JSON
    lethe config get|set K [V]
    lethe reset --yes
    lethe enrich                # optional Haiku enrichment pass

Heavy imports (sentence-transformers / faiss) are lazy so `lethe --version`
stays snappy.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — Python <3.11 unsupported
    tomllib = None  # type: ignore[assignment]

__version__ = "0.1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "bi_encoder": "all-MiniLM-L6-v2",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,
    "n_clusters": 30,
    "use_rank_gap": True,
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / ".lethe"

    @property
    def memory(self) -> Path:
        return self.base / "memory"

    @property
    def index(self) -> Path:
        return self.base / "index"

    @property
    def enrichments(self) -> Path:
        return self.base / "enrichments.jsonl"

    @property
    def config(self) -> Path:
        return self.base / "config.toml"


def resolve_paths(root: str | Path | None = None) -> Paths:
    """Resolve the per-project lethe paths.

    If ``root`` is ``None``, walks up from CWD looking for a git root, then
    falls back to CWD.
    """
    if root is not None:
        return Paths(root=Path(root).resolve())

    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / ".git").exists():
            return Paths(root=candidate)
    return Paths(root=cwd)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(paths: Paths) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not paths.config.exists() or tomllib is None:
        return cfg
    try:
        with paths.config.open("rb") as f:
            parsed = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return cfg
    for k, v in parsed.items():
        cfg[k] = v
    return cfg


def _format_toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return json.dumps(v)


def save_config(paths: Paths, cfg: dict[str, Any]) -> None:
    paths.base.mkdir(parents=True, exist_ok=True)
    lines = ["# lethe config — edit and re-run `lethe index`"]
    for k in sorted(cfg):
        lines.append(f"{k} = {_format_toml_value(cfg[k])}")
    paths.config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coerce_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------

def _load_encoders(cfg: dict[str, Any]):
    """Import sentence-transformers lazily so `lethe --version` stays fast."""
    from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore[import-not-found]

    bi = SentenceTransformer(cfg["bi_encoder"])
    xenc = CrossEncoder(cfg["cross_encoder"])
    return bi, xenc


def _infer_stored_dim(paths: Paths, default: int) -> int:
    """Peek at embeddings.npz to match the dim used when the store was built.

    Prevents a FAISS dim mismatch when ``status`` runs with no encoders
    loaded but the store was previously indexed with a different-dim encoder
    (common in tests with a mock bi-encoder).
    """
    import numpy as np

    emb_path = paths.index / "embeddings.npz"
    if not emb_path.exists():
        return default
    try:
        with np.load(str(emb_path), allow_pickle=True) as data:
            embs = data["embeddings"]
            if embs.ndim == 2 and embs.shape[1] > 0:
                return int(embs.shape[1])
    except (OSError, KeyError, ValueError):
        pass
    return default


def _open_store(paths: Paths, cfg: dict[str, Any], *, need_encoders: bool):
    from lethe.memory_store import MemoryStore
    from lethe.rif import RIFConfig

    bi = xenc = None
    if need_encoders:
        bi, xenc = _load_encoders(cfg)
        dim = bi.get_sentence_embedding_dimension()
    else:
        dim = _infer_stored_dim(paths, default=384)

    rif = RIFConfig(
        n_clusters=int(cfg.get("n_clusters", 30)),
        use_rank_gap=bool(cfg.get("use_rank_gap", True)),
    )
    return MemoryStore(
        paths.index,
        bi_encoder=bi,
        cross_encoder=xenc,
        dim=dim,
        rif_config=rif,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    paths.memory.mkdir(parents=True, exist_ok=True)
    paths.index.mkdir(parents=True, exist_ok=True)
    cfg = load_config(paths)

    from lethe.markdown_store import MarkdownStore

    store = _open_store(paths, cfg, need_encoders=True)
    try:
        md = MarkdownStore(
            memory_dir=Path(args.dir) if args.dir else paths.memory,
            index_dir=paths.index,
        )
        counts = md.reindex(store)
        store.save()
    finally:
        store.close()

    if args.json_output:
        print(json.dumps(counts))
    else:
        print(
            f"indexed: added={counts['added']} removed={counts['removed']} "
            f"unchanged={counts['unchanged']} total={counts['total']}"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    cfg = load_config(paths)

    store = _open_store(paths, cfg, need_encoders=True)
    try:
        results = store.retrieve(args.query, k=args.top_k)
        store.save()
    finally:
        store.close()

    if args.json_output:
        print(json.dumps([
            {"id": eid, "content": content, "score": float(score)}
            for eid, content, score in results
        ]))
        return 0

    if not results:
        print("(no results)")
        return 0

    for eid, content, score in results:
        snippet = content.strip().splitlines()[0][:160]
        print(f"[{score:+.2f}] {eid}  {snippet}")
    return 0


def cmd_expand(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    from lethe.markdown_store import MarkdownStore

    md = MarkdownStore(memory_dir=paths.memory, index_dir=paths.index)
    content = md.get_chunk(args.chunk_id)
    if content is None:
        print(f"chunk {args.chunk_id!r} not found", file=sys.stderr)
        return 1
    print(content)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    cfg = load_config(paths)

    if not paths.index.exists():
        payload = {
            "root": str(paths.root),
            "initialized": False,
            "total_entries": 0,
        }
        print(json.dumps(payload, indent=2))
        return 0

    store = _open_store(paths, cfg, need_encoders=False)
    try:
        stats: dict[str, Any] = dict(store.stats())
    finally:
        store.close()

    stats["root"] = str(paths.root)
    stats["memory_dir"] = str(paths.memory)
    stats["initialized"] = True
    print(json.dumps(stats, indent=2))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    cfg = load_config(paths)

    if args.action == "get":
        if args.key is None:
            print(json.dumps(cfg, indent=2))
            return 0
        if args.key not in cfg:
            print(f"(unset) {args.key}")
            return 1
        print(cfg[args.key])
        return 0

    if args.action == "set":
        if args.key is None or args.value is None:
            print("usage: lethe config set KEY VALUE", file=sys.stderr)
            return 2
        cfg[args.key] = _coerce_scalar(args.value)
        save_config(paths, cfg)
        print(f"{args.key} = {cfg[args.key]!r}")
        return 0

    print(f"unknown config action: {args.action}", file=sys.stderr)
    return 2


def cmd_reset(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    if not args.yes:
        print(
            f"Would delete {paths.index} (markdown in {paths.memory} is preserved). "
            "Pass --yes to confirm.",
            file=sys.stderr,
        )
        return 1
    if paths.index.exists():
        shutil.rmtree(paths.index)
        print(f"removed {paths.index}")
    else:
        print("nothing to remove")
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    import asyncio

    from lethe.enrichment import enrich_dataset
    from lethe.markdown_store import MarkdownStore

    paths = resolve_paths(args.root)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — `lethe enrich` requires it.", file=sys.stderr)
        return 2

    md = MarkdownStore(
        memory_dir=Path(args.dir) if args.dir else paths.memory,
        index_dir=paths.index,
    )
    chunks = md.scan()
    pairs = [(c.id, c.content) for c in chunks]
    stats = asyncio.run(
        enrich_dataset(
            pairs,
            output_path=paths.enrichments,
            model=args.model,
            concurrency=args.concurrency,
        )
    )
    print(
        json.dumps(
            {
                "completed": stats.completed,
                "failed": stats.failed,
                "total": stats.total,
                "est_cost_usd": round(stats.est_cost_usd(args.model), 4),
                "output": str(paths.enrichments),
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lethe",
        description="Persistent memory store for LLM agents — hybrid retrieval, RIF, optional enrichment.",
    )
    p.add_argument("--version", action="version", version=f"lethe {__version__}")
    p.add_argument("--root", default=None, help="Project root. Default: git root of CWD.")

    sub = p.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Reindex markdown memory files.")
    p_index.add_argument("dir", nargs="?", default=None, help="Override memory directory.")
    p_index.add_argument("--json-output", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Retrieve top-k memories for a query.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--json-output", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_expand = sub.add_parser("expand", help="Print the full markdown section for a chunk id.")
    p_expand.add_argument("chunk_id")
    p_expand.set_defaults(func=cmd_expand)

    p_status = sub.add_parser("status", help="Print diagnostic JSON for the store.")
    p_status.set_defaults(func=cmd_status)

    p_cfg = sub.add_parser("config", help="Read or write config values.")
    p_cfg.add_argument("action", choices=["get", "set"])
    p_cfg.add_argument("key", nargs="?")
    p_cfg.add_argument("value", nargs="?")
    p_cfg.set_defaults(func=cmd_config)

    p_reset = sub.add_parser("reset", help="Delete .lethe/index/ (markdown preserved).")
    p_reset.add_argument("--yes", action="store_true")
    p_reset.set_defaults(func=cmd_reset)

    p_enrich = sub.add_parser("enrich", help="Run Haiku enrichment over scanned chunks.")
    p_enrich.add_argument("dir", nargs="?", default=None)
    p_enrich.add_argument("--model", default="claude-haiku-4-5")
    p_enrich.add_argument("--concurrency", type=int, default=5)
    p_enrich.set_defaults(func=cmd_enrich)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        return int(func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
