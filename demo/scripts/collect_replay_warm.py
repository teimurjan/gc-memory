"""Driver for the warmed replay demo.

Like collect_replay.py, but runs N_WARMUP silent queries before Phase 1
for the lethe pass. These seed the clustered-RIF centroids and
suppression state so "cold" is genuinely cold per-cluster (k-means
trained on enough points, RIF state populated before the first
recorded query).

Baseline is stateless — warmup is pure noise for it — so we skip
warmup on the baseline pass to save compute. Lethe runs warmup first,
then the same 100 cold + 3 replay rounds as collect_replay.py.

Warmup rows are dropped from public/run_replay_warm.json so the UI
only sees the cold + warm rounds. Per-phase means printed to stdout
include warmup for diagnostic purposes.

Usage:
    uv run python demo/scripts/collect_replay_warm.py
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
OUT = DEMO / "public" / "run_replay_warm.json"
TMP = DEMO / ".pass_out"

N_UNIQUE = 100
N_REPLAY = 100
N_ROUNDS = 3
N_WARMUP = 1000


def run_pass(
    label: str,
    alpha: float,
    use_rank_gap: bool,
    n_clusters: int,
    n_warmup: int,
) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    out_json = TMP / f"{label}_replay_warm.json"
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
        str(n_warmup),
        str(out_json),
    ], cwd=str(REPO))
    return json.loads(out_json.read_text())


def _phase_mean(rows: list[dict], system: str, phase: str) -> float:
    vals = [r[system]["ndcg"] for r in rows if r["phase"] == phase]
    return sum(vals) / len(vals) if vals else 0.0


def _raw_phase_mean(payload: dict, phase: str) -> float:
    vals = [
        n for n, p in zip(payload["ndcgs"], payload["phases"]) if p == phase
    ]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    # Baseline is stateless — no warmup needed, saves ~25 min of compute.
    base = run_pass(
        "baseline", alpha=0.0, use_rank_gap=False, n_clusters=0, n_warmup=0,
    )
    leth = run_pass(
        "lethe", alpha=0.3, use_rank_gap=True, n_clusters=30, n_warmup=N_WARMUP,
    )

    # Schedules don't match 1:1 because lethe has N_WARMUP extra rows at
    # the front. Align by phase: drop warmup from both, then the cold +
    # warm rounds must match qid-for-qid.
    def _filter_non_warmup(payload: dict) -> tuple[list[str], list[str], list[float]]:
        qids: list[str] = []
        phases: list[str] = []
        ndcgs: list[float] = []
        for q, p, n in zip(payload["qids"], payload["phases"], payload["ndcgs"]):
            if p == "warmup":
                continue
            qids.append(q)
            phases.append(p)
            ndcgs.append(n)
        return qids, phases, ndcgs

    b_qids, b_phases, b_ndcgs = _filter_non_warmup(base)
    l_qids, l_phases, l_ndcgs = _filter_non_warmup(leth)

    assert b_qids == l_qids, "cold+warm schedules diverged between passes"
    assert b_phases == l_phases, "cold+warm phases diverged between passes"

    rows = []
    for i, qid in enumerate(l_qids):
        rows.append({
            "idx": i,
            "qid": qid,
            "phase": l_phases[i],
            "baseline": {"ndcg": round(b_ndcgs[i], 4)},
            "lethe": {"ndcg": round(l_ndcgs[i], 4)},
        })

    phase_boundaries = [N_UNIQUE + i * N_REPLAY for i in range(N_ROUNDS)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "fps": 30,
            "totalQueries": len(rows),
            "nUnique": N_UNIQUE,
            "nReplay": N_REPLAY,
            "nRounds": N_ROUNDS,
            "nWarmup": N_WARMUP,  # diagnostic — not reflected in rows
            "phaseBoundaries": phase_boundaries,
            "snapshotAt": [],
        },
        "queries": rows,
    }))

    phases = ["cold"] + [f"warm{r}" for r in range(1, N_ROUNDS + 1)]
    print(f"\nWrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print(f"Silent warmup (lethe only): {N_WARMUP} queries, mean NDCG "
          f"{_raw_phase_mean(leth, 'warmup'):.3f}")
    print("Mean NDCG@10 by phase:")
    for system, payload in (("baseline", base), ("lethe", leth)):
        means = {p: _phase_mean(rows, system, p) for p in phases}
        cold = means["cold"]
        parts = [f"cold={cold:.3f}"]
        for r in range(1, N_ROUNDS + 1):
            tag = f"warm{r}"
            parts.append(f"{tag}={means[tag]:.3f} (Δ{means[tag] - cold:+.3f})")
        print(f"  {system:<9} " + "  ".join(parts))


if __name__ == "__main__":
    main()
