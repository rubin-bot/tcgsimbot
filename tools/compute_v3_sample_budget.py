"""v3 hard prerequisite (CLAUDE.md ARCHITECTURE DECISION / Stage 6): compute a safe N (number
of determinizations per decision for the multi-sample voting agent) from REAL data on disk,
not guesses. Reads:

  - runs/our_episodes/**/*.json          -- real Kaggle episode `configuration` blocks
                                             (actTimeout / runTimeout -- the actual Kaggle time
                                             budget, confirmed identical across every episode).
  - runs/near_tie_measurement/replays.jsonl -- real per-decision choose_action() wall-clock cost
                                             (elapsed_s), one determinization each.
  - runs/kernel_vs_baseline/*/vs_baseline_trace.jsonl -- real per-game decision counts from the
                                             400-game v1/v2 kernel gate (800 real games).

Computes the worst case: N * max_single_sample_cost * max_decisions_per_game, and requires it
stay under 50% of runTimeout. Per the task spec: if the budget makes N < 5, stop and report
before writing any v3 code.

Usage: PYTHONIOENCODING=utf-8 python tools/compute_v3_sample_budget.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAFETY_FRACTION = 0.5   # stay under this fraction of runTimeout
CHOSEN_N = 8             # see rationale printed below


def load_episode_configs() -> list[dict]:
    configs = []
    for path in glob.glob(os.path.join(REPO_ROOT, "runs", "our_episodes", "**", "*.json"),
                           recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cfg = d.get("configuration")
        if cfg:
            configs.append(cfg)
    return configs


def load_decision_costs() -> list[float]:
    path = os.path.join(REPO_ROOT, "runs", "near_tie_measurement", "replays.jsonl")
    costs = []
    if not os.path.exists(path):
        return costs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for replay in rec.get("replays", []):
                es = replay.get("elapsed_s")
                if es is not None:
                    costs.append(es)
    return costs


def load_decisions_per_game() -> list[int]:
    counts = []
    for path in glob.glob(os.path.join(REPO_ROOT, "runs", "kernel_vs_baseline", "*",
                                        "vs_baseline_trace.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                counts.append(len(g.get("decisions", [])))
    return counts


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def main() -> int:
    configs = load_episode_configs()
    if not configs:
        print("ERROR: no real episode configuration blocks found under runs/our_episodes/ -- "
              "cannot confirm the Kaggle time budget from real data. Run tools/measure.py "
              "first to download episodes.", file=sys.stderr)
        return 1

    act_timeouts = {c.get("actTimeout") for c in configs}
    run_timeouts = {c.get("runTimeout") for c in configs}
    print(f"Real episode configs scanned: {len(configs)}")
    print(f"  actTimeout values seen: {sorted(act_timeouts)}")
    print(f"  runTimeout values seen: {sorted(run_timeouts)}")
    if len(run_timeouts) != 1:
        print("WARNING: runTimeout is not constant across episodes -- using the minimum "
              "(most conservative) value.", file=sys.stderr)
    run_timeout_s = min(t for t in run_timeouts if t is not None)
    act_timeout = next(iter(act_timeouts))
    if act_timeout not in (0, None):
        print(f"WARNING: actTimeout is {act_timeout!r}, not 0/disabled as previously "
              f"documented -- re-check docs/ladder_attack_decline_diagnosis_2026-07-23.md "
              f"assumptions before trusting this budget.", file=sys.stderr)

    costs = load_decision_costs()
    if not costs:
        print("ERROR: no per-decision timing data found in "
              "runs/near_tie_measurement/replays.jsonl -- cannot compute real per-sample cost.",
              file=sys.stderr)
        return 1
    costs.sort()
    mean_cost = sum(costs) / len(costs)
    max_cost = costs[-1]
    print(f"\nReal single-determinization choose_action() cost, n={len(costs)} samples:")
    print(f"  mean={mean_cost*1000:.1f}ms median={percentile(costs, 0.5)*1000:.1f}ms "
          f"p90={percentile(costs, 0.9)*1000:.1f}ms max={max_cost*1000:.1f}ms")

    decisions_per_game = load_decisions_per_game()
    if not decisions_per_game:
        print("ERROR: no decision-count data found under "
              "runs/kernel_vs_baseline/*/vs_baseline_trace.jsonl -- cannot bound worst-case "
              "game length.", file=sys.stderr)
        return 1
    decisions_per_game.sort()
    max_decisions = decisions_per_game[-1]
    print(f"\nReal decisions/game, n={len(decisions_per_game)} games:")
    print(f"  median={percentile(decisions_per_game, 0.5)} "
          f"p90={percentile(decisions_per_game, 0.9)} max={max_decisions}")

    budget_s = SAFETY_FRACTION * run_timeout_s
    max_n = int(budget_s / (max_cost * max_decisions))
    print(f"\nWorst case: N * {max_cost*1000:.1f}ms(max) * {max_decisions}(max decisions) "
          f"<= {SAFETY_FRACTION:.0%} of runTimeout ({run_timeout_s}s) = {budget_s:.0f}s")
    print(f"  => N <= {max_n}")

    if max_n < 5:
        print(f"\nSTOP: computed ceiling N={max_n} is below 5. Do not write v3 code -- "
              f"report this to the user before proceeding.", file=sys.stderr)
        return 2

    print(f"\nCHOSEN N = {CHOSEN_N} "
          f"({max_n / CHOSEN_N:.1f}x safety margin below the ceiling of {max_n}).")
    print(f"  Per-decision cost at N={CHOSEN_N}: "
          f"p90 ~= {CHOSEN_N * percentile(costs, 0.9) * 1000:.0f}ms, "
          f"worst-case ~= {CHOSEN_N * max_cost * 1000:.0f}ms.")
    print(f"  Self-imposed anytime time-guard: 5s/decision (Kaggle imposes none -- "
          f"actTimeout={act_timeout} means disabled).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
