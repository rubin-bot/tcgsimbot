"""Master-study Workstream B2: clusters and ranks the decision-diff output
(runs/decision_diff/{expert,control}_diff.jsonl, from tools/decision_diff.py) by
(decision kind, game phase), with all 5 evidence fields the verification-gate skill and the
master-study task require: coverage, margin, control-filter, outcome-linkage (top clusters
only), and hand-traced examples.

Ranking (stated explicitly, not eyeballed): rank_score = n_disagreements_in_cluster *
(2.0 if control-correlated else 1.0 if ambiguous else 0.5 if anti-correlated). Absolute
disagreement COUNT (not just rate) is the primary driver -- a cluster with more real divergent
decisions is more actionable evidence regardless of how big the parent population is, but a
tiny-n cluster still can't out-rank a well-populated one just because its RATE happens to be
high. The control-filter multiplier separates "we differ from experts AND agree with the (weak)
control group" (rating-correlated, likely a real gap -- 2x) from "we differ from everyone
including control" (style, lower priority -- 0.5x); clusters with no control data at all stay
at the neutral 1.0x (ambiguous, not enough control signal either way).

Usage:
  .venv/Scripts/python tools/cluster_divergences.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ladder_episode_parser import find_our_seat  # noqa: E402
from measure_near_tie_hypothesis import active_decision_steps  # noqa: E402

EXPERT_DIFF_PATH = os.path.join(ROOT, "runs", "decision_diff", "expert_diff.jsonl")
CONTROL_DIFF_PATH = os.path.join(ROOT, "runs", "decision_diff", "control_diff.jsonl")
CORPUS_DIR = os.path.join(ROOT, "runs", "expert_corpus", "2026-07-22")
OUT_PATH = os.path.join(ROOT, "runs", "decision_diff", "clusters.json")
FORWARD_LOOK = 3
_TIE_EPS_REL = 1e-6  # matches agents/search_scorer.py's own tie-break epsilon
NEAR_TIE_REL_THRESHOLD = 0.05  # matches tools/loss_review.py's near-tie reporting threshold
MIN_CELL_N = 30  # per verification-gate skill / meta_report.py convention
N_TOP_CLUSTERS_FOR_OUTCOME_LINKAGE = 5
N_EXAMPLES_PER_TOP_CLUSTER = 3


def load_diff(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cluster_key(rec: dict) -> tuple[str, str]:
    return (rec["historical_kind"], rec["phase"])


def relative_margin(rec: dict) -> float | None:
    h, o = rec.get("historical_score"), rec.get("our_score")
    if h is None or o is None:
        return None
    scale = max(abs(h), abs(o), 1.0)
    return (o - h) / scale


def build_clusters(records: list[dict]) -> dict[tuple, dict]:
    by_cluster: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_cluster[cluster_key(r)].append(r)

    clusters = {}
    for key, recs in by_cluster.items():
        total = len(recs)
        disagreements = [r for r in recs if not r["agree"]]
        n_games = len({(r["team_name"], r["episode_id"]) for r in recs})
        n_disagree_games = len({(r["team_name"], r["episode_id"]) for r in disagreements})
        margins = [relative_margin(r) for r in disagreements]
        margins = [m for m in margins if m is not None]
        near_tie_count = sum(1 for m in margins if abs(m) <= NEAR_TIE_REL_THRESHOLD)
        clusters[key] = {
            "kind": key[0], "phase": key[1],
            "n_total_decisions": total, "n_disagreements": len(disagreements),
            "coverage_rate": len(disagreements) / total if total else 0.0,
            "n_games_touching_cluster": n_games,
            "coverage_per_game": len(disagreements) / n_games if n_games else 0.0,
            "n_near_tie_disagreements": near_tie_count,
            "n_large_margin_disagreements": len(margins) - near_tie_count,
            "mean_relative_margin": sum(margins) / len(margins) if margins else None,
            "disagreement_records": disagreements,
        }
    return clusters


def outcome_linkage_for_decision(episode_path: str, team_name: str,
                                  raw_step_index: int) -> float | None:
    """Adapts tools/analyze_near_tie_results.py::analyze_outcome_linkage's forward-tracing
    mechanism to an arbitrary team's episode. Critical detail matched exactly from the
    original (a real bug found and fixed here during this cycle: an earlier version of this
    function walked FORWARD_LOOK *raw* step indices ahead, which barely advances the game --
    most raw steps are stale INACTIVE carryovers, not new decisions -- and produced an
    all-zero, obviously-wrong delta for every single decision traced. The original walks
    forward through DECISION-LEVEL indices from active_decision_steps()'s own filtered list
    instead, i.e. "next FORWARD_LOOK real decisions by this player" (which can span many raw
    steps, including the opponent's whole intervening turn), not "next FORWARD_LOOK raw steps"."""
    with open(episode_path, encoding="utf-8") as f:
        raw = json.load(f)
    seat = find_our_seat(raw.get("info", {}), team_name)
    if seat is None:
        return None
    opp_seat = 1 - seat
    decision_steps = active_decision_steps(raw, seat)
    if raw_step_index not in decision_steps:
        return None
    decision_idx = decision_steps.index(raw_step_index)

    obs_dict = raw["steps"][raw_step_index][seat]["observation"]
    cur = obs_dict.get("current")
    if cur is None:
        return None
    our_prize_now = len(cur["players"][seat]["prize"])
    opp_prize_now = len(cur["players"][opp_seat]["prize"])

    future_idx = decision_idx + 1
    future_prize_delta = None
    for _ in range(FORWARD_LOOK):
        if future_idx >= len(decision_steps):
            break
        f_step = decision_steps[future_idx]
        f_obs = raw["steps"][f_step][seat]["observation"]
        f_cur = f_obs.get("current")
        if f_cur is None:
            future_idx += 1
            continue
        our_prize_then = len(f_cur["players"][seat]["prize"])
        opp_prize_then = len(f_cur["players"][opp_seat]["prize"])
        future_prize_delta = (our_prize_now - our_prize_then) - (opp_prize_now - opp_prize_then)
        future_idx += 1
    return future_prize_delta


def rank_clusters(expert_clusters: dict, control_clusters: dict) -> list[dict]:
    """Control-filter labels (computed inline, needs both expert and control rates together):
    "rating_correlated" (control agrees with US more often than experts do, on this SAME
    cluster -- i.e. weaker players make the mistake we'd also avoid, stronger players don't)
    -> 2.0x rank multiplier. "style_not_rating_correlated" (control ALSO disagrees at a similar
    or higher rate -- we differ from everyone, not specifically from skill) -> 0.5x.
    "no_control_data" (cluster has too few control-group decisions, n<5) -> 1.0x, neutral."""
    ranked = []
    for key, ec in expert_clusters.items():
        cc = control_clusters.get(key)
        if cc is None or cc["n_total_decisions"] < 5:
            label, mult = "no_control_data", 1.0
        else:
            expert_rate = ec["coverage_rate"]
            control_rate = cc["coverage_rate"]
            # Rating-correlated: control (weaker players) disagrees with us LESS than experts
            # do -- i.e. control's real play looks more like ours than the experts' does. That
            # means the divergence tracks skill, so it's a real, actionable gap.
            if control_rate < expert_rate * 0.7:
                label, mult = "rating_correlated", 2.0
            elif control_rate > expert_rate * 1.3:
                label, mult = "control_diverges_more", 1.0  # ambiguous, not evidence against us
            else:
                label, mult = "style_not_rating_correlated", 0.5
        rank_score = ec["n_disagreements"] * mult
        ranked.append({
            **{k: v for k, v in ec.items() if k != "disagreement_records"},
            "control_comparison": {
                "label": label, "multiplier": mult,
                "control_coverage_rate": control_clusters.get(key, {}).get("coverage_rate"),
                "control_n": control_clusters.get(key, {}).get("n_total_decisions", 0),
            },
            "rank_score": rank_score,
            "_disagreement_records": ec["disagreement_records"],
        })
    ranked.sort(key=lambda c: -c["rank_score"])
    return ranked


def add_outcome_linkage_and_examples(ranked: list[dict]) -> None:
    for cluster in ranked[:N_TOP_CLUSTERS_FOR_OUTCOME_LINKAGE]:
        recs = cluster["_disagreement_records"]
        deltas = []
        for r in recs:
            path = os.path.join(CORPUS_DIR, f"{r['episode_id']}.json")
            if not os.path.exists(path):
                continue
            delta = outcome_linkage_for_decision(path, r["team_name"], r["raw_step_index"])
            if delta is not None:
                deltas.append(delta)
        cluster["outcome_linkage"] = {
            "n_traced": len(deltas),
            "mean_prize_tempo_delta_next_3_turns": sum(deltas) / len(deltas) if deltas else None,
            "n_positive": sum(1 for d in deltas if d > 0),
            "n_negative": sum(1 for d in deltas if d < 0),
            "n_zero": sum(1 for d in deltas if d == 0),
        }
        cluster["hand_traced_examples"] = [
            {"team_name": r["team_name"], "episode_id": r["episode_id"],
             "raw_step_index": r["raw_step_index"], "turn": r["turn"], "phase": r["phase"],
             "historical_choice": r["historical_choice"], "historical_kind": r["historical_kind"],
             "historical_score": r["historical_score"], "our_choice": r["our_choice"],
             "our_kind": r["our_kind"], "our_score": r["our_score"], "outcome": r["outcome"]}
            for r in recs[:N_EXAMPLES_PER_TOP_CLUSTER]
        ]


def main() -> None:
    expert_records = load_diff(EXPERT_DIFF_PATH)
    control_records = load_diff(CONTROL_DIFF_PATH)
    print(f"expert records: {len(expert_records)}, control records: {len(control_records)}")

    expert_clusters = build_clusters(expert_records)
    control_clusters = build_clusters(control_records)
    print(f"{len(expert_clusters)} expert clusters found")

    ranked = rank_clusters(expert_clusters, control_clusters)
    add_outcome_linkage_and_examples(ranked)

    print("\n=== TOP 10 CLUSTERS BY RANK SCORE ===")
    for c in ranked[:10]:
        print(f"\n[{c['kind']}/{c['phase']}] rank_score={c['rank_score']:.1f} "
              f"n_disagree={c['n_disagreements']}/{c['n_total_decisions']} "
              f"(coverage_rate={c['coverage_rate']:.1%}, "
              f"coverage_per_game={c['coverage_per_game']:.2f}) "
              f"mean_margin={c['mean_relative_margin']}")
        print(f"  control: {c['control_comparison']}")
        if "outcome_linkage" in c:
            print(f"  outcome_linkage: {c['outcome_linkage']}")

    # strip internal-only field before writing
    for c in ranked:
        c.pop("_disagreement_records", None)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
