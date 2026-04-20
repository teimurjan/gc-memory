"""Driver for the replay demo collector.

Runs baseline and lethe passes as separate subprocesses (each with a
fresh Python memory space) over a multi-round schedule:

    Phase 1 (cold):  N_UNIQUE distinct queries, fixed order.
    Phase 2..N+1:    N_ROUNDS replay rounds, each playing the first
                     N_REPLAY queries in the same order. Tagged warm1,
                     warm2, ..., warmN.

Baseline is stateless (alpha=0, no clusters), so each warm round's NDCG
on the same qid equals the cold NDCG — a flat reference. Lethe builds
per-cluster RIF suppression over time; each successive warm round should
lift further above cold, showing the learning curve.

Merged into demo/public/run_replay.json.

Usage:
    uv run python demo/scripts/collect_replay.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parent
WORKER = HERE / "_pass_replay.py"
OUT = DEMO / "public" / "run_replay.json"
TMP = DEMO / ".pass_out"

N_UNIQUE = 100
N_REPLAY = 100
N_ROUNDS = 3


def run_pass(
    label: str,
    alpha: float,
    use_rank_gap: bool,
    n_clusters: int,
) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    out_json = TMP / f"{label}_replay.json"
    subprocess.check_call([
        sys.executable,
        str(WORKER),
        label,
        str(alpha),
        str(use_rank_gap),
        str(n_clusters),
        str(N_UNIQUE),
        str(N_REPLAY),
        str(N_ROUNDS),
        "0",  # n_warmup — see collect_replay_warm.py for the warmed variant
        str(out_json),
    ], cwd=str(REPO))
    return json.loads(out_json.read_text())


def _phase_mean(rows: list[dict], system: str, phase: str) -> float:
    vals = [r[system]["ndcg"] for r in rows if r["phase"] == phase]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    base = run_pass("baseline", alpha=0.0, use_rank_gap=False, n_clusters=0)
    leth = run_pass("lethe", alpha=0.3, use_rank_gap=True, n_clusters=30)

    assert base["qids"] == leth["qids"], "schedules diverged between passes"
    assert base["phases"] == leth["phases"], "phases diverged between passes"

    rows = []
    for i, qid in enumerate(leth["qids"]):
        rows.append({
            "idx": i,
            "qid": qid,
            "phase": leth["phases"][i],
            "baseline": {"ndcg": round(base["ndcgs"][i], 4)},
            "lethe": {"ndcg": round(leth["ndcgs"][i], 4)},
        })

    # Phase boundaries at the start of each warm round: N_UNIQUE,
    # N_UNIQUE + N_REPLAY, N_UNIQUE + 2*N_REPLAY, ...
    phase_boundaries = [N_UNIQUE + i * N_REPLAY for i in range(N_ROUNDS)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "fps": 30,
            "totalQueries": len(rows),
            "nUnique": N_UNIQUE,
            "nReplay": N_REPLAY,
            "nRounds": N_ROUNDS,
            "phaseBoundaries": phase_boundaries,
            "snapshotAt": [],
        },
        "queries": rows,
    }))

    phases = ["cold"] + [f"warm{r}" for r in range(1, N_ROUNDS + 1)]
    print(f"\nWrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("Mean NDCG@10 by phase:")
    for system in ("baseline", "lethe"):
        means = {p: _phase_mean(rows, system, p) for p in phases}
        cold = means["cold"]
        parts = [f"cold={cold:.3f}"]
        for r in range(1, N_ROUNDS + 1):
            tag = f"warm{r}"
            parts.append(f"{tag}={means[tag]:.3f} (Δ{means[tag] - cold:+.3f})")
        print(f"  {system:<9} " + "  ".join(parts))


if __name__ == "__main__":
    main()
