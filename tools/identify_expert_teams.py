"""Master-study Workstream A1: identifies Crustle-archetype teams from
runs/meta_mining/team_game_signatures.jsonl (tools/meta_miner.py's output), classified via
tools/meta_report.py's own team_pokemon_archetype() (reused, not reimplemented) so this stays
consistent with docs/meta_report_2026-07-22.md's own numbers.

Splits into an EXPERT group (all Crustle teams, ranked by game count) and a CONTROL group (the
lowest-activity Crustle teams in the observed population). Per user decision 2026-07-23: the
single scanned day's data has zero Crustle teams outside the top100/600+ bands, so there is no
genuinely low-rated control tier available -- the control group here is a *relative*,
small-sample stand-in (lowest game count within the population), explicitly not a true
low-rated tier. This script reports that gap; it doesn't paper over it.

Usage:
  .venv/Scripts/python tools/identify_expert_teams.py --archetype Crustle --control-n 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meta_report import (  # noqa: E402
    load_signatures, load_leaderboard_scores, band_for_score, team_pokemon_archetype,
)

OUT_PATH = os.path.join(ROOT, "runs", "expert_corpus", "expert_teams.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="Crustle")
    ap.add_argument("--control-n", type=int, default=5)
    args = ap.parse_args()

    sigs = load_signatures()
    by_team: dict[str, list[dict]] = defaultdict(list)
    for rec in sigs:
        if rec.get("team_name"):
            by_team[rec["team_name"]].append(rec)

    print("fetching leaderboard ...")
    scores = load_leaderboard_scores()
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    rank_by_team = {name: i + 1 for i, (name, _) in enumerate(ranked)}

    matches = []
    for team, records in by_team.items():
        arch = team_pokemon_archetype(records)
        if arch is None or arch[1] != args.archetype:
            continue
        band = band_for_score(scores.get(team), rank_by_team.get(team))
        matches.append({
            "team_name": team,
            "n_episodes": len(records),
            "band": band,
            "rank": rank_by_team.get(team),
            "score": scores.get(team),
            "episode_ids": sorted({r["episode_id"] for r in records}),
        })

    matches.sort(key=lambda m: -m["n_episodes"])

    bands_seen = {m["band"] for m in matches}
    low_rated_bands = {"400-600", "below400"}
    has_genuine_low_rated = bool(bands_seen & low_rated_bands)

    control = sorted(matches, key=lambda m: m["n_episodes"])[:args.control_n]
    control_names = {m["team_name"] for m in control}
    expert = [m for m in matches if m["team_name"] not in control_names]

    result = {
        "archetype": args.archetype,
        "n_teams_total": len(matches),
        "bands_observed": sorted(bands_seen),
        "has_genuine_low_rated_team": has_genuine_low_rated,
        "expert_teams": expert,
        "control_teams": control,
    }

    print(f"\n{args.archetype} teams found: {len(matches)}")
    print(f"bands observed: {sorted(bands_seen)}")
    if not has_genuine_low_rated:
        print("WARNING: no team in 400-600 or below400 band -- control group is a "
              "lowest-activity relative stand-in, NOT a genuinely low-rated tier. "
              "See master-study report methodology section.")
    print(f"\nexpert group ({len(expert)} teams):")
    for m in expert:
        print(f"  {m['team_name']}: {m['n_episodes']} episodes, band={m['band']}, "
              f"rank={m['rank']}, score={m['score']}")
    print(f"\ncontrol group ({len(control)} teams):")
    for m in control:
        print(f"  {m['team_name']}: {m['n_episodes']} episodes, band={m['band']}, "
              f"rank={m['rank']}, score={m['score']}")

    total_expert_episodes = sum(m["n_episodes"] for m in expert)
    total_control_episodes = sum(m["n_episodes"] for m in control)
    print(f"\ntotal expert team-episode records: {total_expert_episodes}")
    print(f"total control team-episode records: {total_control_episodes}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
