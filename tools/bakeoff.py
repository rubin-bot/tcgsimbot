"""Round-robin comparison across weight-set candidates before shipping v1 -- "ship whichever
wins on the whole local slate," not just whichever looks best on a single matchup.

Each candidate plays: vs. `baseline` (150 games), vs. `random` (100), and head-to-head
against every OTHER candidate (100 games each pair). Common seeds are reused across all
candidates for the SAME matchup (same noise-control principle as tools/tune_weights.py) so
differences reflect the weights, not shuffle/coin-flip luck. Reports a full table plus each
candidate's pooled win rate across its whole slate.

Usage:
  .venv/Scripts/python tools/bakeoff.py \\
      --candidate new_fix=runs/tune_run1/new_fix_weights.json \\
      --candidate pre_fix=runs/tune_run1/pretuning_weights.json \\
      --candidate tuned=runs/tune_run1/winner_weights.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from eval_arena import run_one_game, psutil_rss_monitoring_works  # noqa: E402

DECK = os.path.join(ROOT, "decks", "crustle_wall_deck.csv")


def _build_schedule(n: int, seed_base: int) -> list[tuple[int, int]]:
    return [(seed_base + i, i % 2) for i in range(n)]


def _run_matchup(name_a: str, weights_a: str | None, name_b: str, weights_b: str | None,
                  schedule: list[tuple[int, int]], workers: int, timeout: float,
                  rss_cap_mb: float, enforce_rss: bool) -> dict:
    """weights_a/weights_b: None means 'this side is the named non-search_scorer agent
    (baseline/random)'; a path means 'search_scorer with these weights'."""
    python_exe = sys.executable
    agent_a = "search_scorer" if weights_a else name_a
    agent_b = "search_scorer" if weights_b else name_b

    def _one(item):
        seed, a_seat = item
        return run_one_game(
            python_exe, agent_a, agent_b, DECK, DECK, a_seat, seed, timeout, rss_cap_mb,
            enforce_rss, None, seed, candidate_weights=weights_a, opponent_weights=weights_b,
        )

    wins = draws = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        for rec in ex.map(_one, schedule):
            if rec["outcome"] == "candidate_win":
                wins += 1
            elif rec["outcome"] == "draw":
                draws += 1
            elif rec["outcome"] == "crash" and rec["crashing_side"] == "opponent":
                wins += 1
    n = len(schedule)
    return {"wins": wins, "draws": draws, "n": n, "win_rate": (wins + 0.5 * draws) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                     help="NAME=weights.json, repeatable (2 or 3 candidates expected)")
    ap.add_argument("--games-vs-baseline", type=int, default=150)
    ap.add_argument("--games-vs-random", type=int, default=100)
    ap.add_argument("--games-pairwise", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--rss-cap-mb", type=float, default=2048.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates: dict[str, str] = {}
    for c in args.candidate:
        name, path = c.split("=", 1)
        candidates[name] = path

    enforce_rss = psutil_rss_monitoring_works()
    if not enforce_rss:
        print("WARNING: RSS cap self-test failed -- only --timeout is a real safety net.",
              file=sys.stderr)

    results: dict[str, dict] = {name: {"pooled_wins": 0.0, "pooled_n": 0, "matchups": {}}
                                 for name in candidates}

    baseline_schedule = _build_schedule(args.games_vs_baseline, args.seed + 1_000_000)
    random_schedule = _build_schedule(args.games_vs_random, args.seed + 2_000_000)

    for name, path in candidates.items():
        for opp_name, opp_n, schedule in (("baseline", args.games_vs_baseline, baseline_schedule),
                                           ("random", args.games_vs_random, random_schedule)):
            print(f"[bakeoff] {name} vs {opp_name} ({opp_n} games)...")
            r = _run_matchup(name, path, opp_name, None, schedule, args.workers, args.timeout,
                              args.rss_cap_mb, enforce_rss)
            results[name]["matchups"][opp_name] = r
            results[name]["pooled_wins"] += r["wins"] + 0.5 * r["draws"]
            results[name]["pooled_n"] += r["n"]
            print(f"    {name} win_rate={r['win_rate']:.3f} ({r['wins']}/{r['n']})")

    pairwise_schedule = _build_schedule(args.games_pairwise, args.seed + 3_000_000)
    for name_a, name_b in itertools.combinations(candidates, 2):
        print(f"[bakeoff] {name_a} vs {name_b} ({args.games_pairwise} games, head-to-head)...")
        r = _run_matchup(name_a, candidates[name_a], name_b, candidates[name_b],
                          pairwise_schedule, args.workers, args.timeout, args.rss_cap_mb,
                          enforce_rss)
        label = f"vs_{name_b}"
        results[name_a]["matchups"][label] = r
        results[name_a]["pooled_wins"] += r["wins"] + 0.5 * r["draws"]
        results[name_a]["pooled_n"] += r["n"]
        # mirror: name_b's perspective is the complement (draws split evenly, wins/losses flip)
        losses_b = r["n"] - r["wins"] - r["draws"]
        results[name_b]["matchups"][f"vs_{name_a}"] = {
            "wins": losses_b, "draws": r["draws"], "n": r["n"],
            "win_rate": (losses_b + 0.5 * r["draws"]) / r["n"],
        }
        results[name_b]["pooled_wins"] += losses_b + 0.5 * r["draws"]
        results[name_b]["pooled_n"] += r["n"]
        print(f"    {name_a} win_rate={r['win_rate']:.3f} ({r['wins']}/{r['n']})")

    print("\n" + "=" * 70)
    print("BAKE-OFF RESULTS")
    print("=" * 70)
    for name, data in sorted(results.items(), key=lambda kv: -kv[1]["pooled_wins"] / kv[1]["pooled_n"]):
        pooled = data["pooled_wins"] / data["pooled_n"]
        print(f"\n{name}: pooled win rate {pooled:.3f} ({data['pooled_wins']:.1f}/"
              f"{data['pooled_n']})")
        for opp, r in data["matchups"].items():
            print(f"    vs {opp:<12} {r['win_rate']:.3f} ({r['wins']}/{r['n']}, "
                  f"draws={r['draws']})")

    winner = max(results, key=lambda n: results[n]["pooled_wins"] / results[n]["pooled_n"])
    print(f"\nWinner (highest pooled win rate across the whole slate): {winner}")

    out_path = os.path.join(ROOT, "runs", "tune_run1", "bakeoff_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "winner": winner}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
