"""Master-study Workstream B1: runs the decision-diff -- for every real decision made by an
expert (or control) Crustle team in the fetched corpus (runs/expert_corpus/<date>/), reconstruct
the exact board state (tools/reconstruct_decision.py, B0-validated) and run OUR v1 agent
(same frozen snapshot + correctly-applied tuned weights B0 validated against) on it, recording
both choices and our own evaluate() scores for both.

Deck-list caveat (documented, not hidden -- see tools/reconstruct_decision.py's own docstring
and docs/master_study_<date>.md's methodology section): we pass OUR OWN deck
(decks/crustle_wall_deck.csv) as the acting player's deck_list, since we don't have the real
expert team's exact 60-card list (only a partial signature, see tools/meta_report.py). Both
teams run the SAME Crustle/Dwebble archetype, so this is a much closer approximation here than
it would be for a different archetype -- but it is still an approximation of their real deck,
not their actual list, and affects hidden-zone sampling for both the acting player's own unseen
cards and the opponent's.

Resumable via a done-keys set (team_name, episode_id, raw_step_index), matching the convention
tools/measure_near_tie_hypothesis.py already uses.

Usage:
  .venv/Scripts/python tools/decision_diff.py --group expert
  .venv/Scripts/python tools/decision_diff.py --group control
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from reconstruct_decision import reconstruct_episode_decisions  # noqa: E402
from baseline import read_deck_csv  # noqa: E402

EXPERT_TEAMS_PATH = os.path.join(ROOT, "runs", "expert_corpus", "expert_teams.json")
CORPUS_DIR_TEMPLATE = os.path.join(ROOT, "runs", "expert_corpus", "{date}")
OUT_DIR = os.path.join(ROOT, "runs", "decision_diff")
V1_SNAPSHOT_PATH = os.path.join(ROOT, "runs", "v2_tie_break", "search_scorer_v1_snapshot.py")
TUNED_WEIGHTS_PATH = os.path.join(ROOT, "runs", "tune_run1", "winner_weights.json")
DECK_PATH = os.path.join(ROOT, "decks", "crustle_wall_deck.csv")


def load_v1():
    spec = importlib.util.spec_from_file_location("search_scorer_v1_diff", V1_SNAPSHOT_PATH)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    with open(TUNED_WEIGHTS_PATH, encoding="utf-8") as f:
        override = json.load(f)
    weights = dict(v1.WEIGHTS)
    weights.update(override)
    return v1, weights


def load_teams(group: str) -> list[dict]:
    with open(EXPERT_TEAMS_PATH, encoding="utf-8") as f:
        d = json.load(f)
    key = "expert_teams" if group == "expert" else "control_teams"
    return d[key]


def phase_for_turn(turn: int) -> str:
    if turn <= 4:
        return "early"
    if turn <= 10:
        return "mid"
    return "late"


def load_done_keys(out_path: str) -> set[tuple]:
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["team_name"], rec["episode_id"], rec["raw_step_index"]))
    return done


def outcome_for_team(episode_path: str, team_name: str) -> str | None:
    with open(episode_path, encoding="utf-8") as f:
        raw = json.load(f)
    names = raw.get("info", {}).get("TeamNames", [])
    if team_name not in names:
        return None
    seat = names.index(team_name)
    rewards = raw.get("rewards") or [0, 0]
    r = rewards[seat] if seat < len(rewards) else 0
    if r > 0:
        return "win"
    if r < 0:
        return "loss"
    return "draw"


def run_diff(group: str, date: str, deck: list[int], v1, weights) -> dict:
    teams = load_teams(group)
    team_names = [t["team_name"] for t in teams]
    corpus_dir = CORPUS_DIR_TEMPLATE.format(date=date)
    episode_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))

    out_path = os.path.join(OUT_DIR, f"{group}_diff.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    done = load_done_keys(out_path)
    print(f"{group}: {len(team_names)} teams, {len(episode_paths)} episode files in corpus, "
          f"{len(done)} decisions already done")

    n_written = 0
    n_episodes_touched = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for path in episode_paths:
            episode_id = os.path.splitext(os.path.basename(path))[0]
            touched_this_episode = False
            for team_name in team_names:
                decisions = reconstruct_episode_decisions(path, team_name)
                if not decisions:
                    continue
                outcome = outcome_for_team(path, team_name)
                for dec in decisions:
                    key = (team_name, episode_id, dec["raw_step_index"])
                    if key in done:
                        continue
                    if dec["historical_chosen_index"] is None:
                        continue
                    touched_this_episode = True
                    captured = {}

                    def trace_fn(rec, _c=captured):
                        _c.update(rec)

                    result = v1.choose_action(dec["game_state"], dec["selection"],
                                               dec["obs_dict"], deck, None, trace_fn=trace_fn,
                                               weights=weights)
                    our_choice = result[0] if result else None
                    options_by_index = {o["index"]: o for o in (captured.get("options") or [])}
                    hist_opt = options_by_index.get(dec["historical_chosen_index"])
                    our_opt = options_by_index.get(our_choice)
                    rec = {
                        "group": group, "team_name": team_name, "episode_id": episode_id,
                        "raw_step_index": dec["raw_step_index"], "turn": dec["turn"],
                        "phase": phase_for_turn(dec["turn"]), "outcome": outcome,
                        "historical_choice": dec["historical_chosen_index"],
                        "historical_kind": hist_opt["kind"] if hist_opt else None,
                        "historical_score": hist_opt["score"] if hist_opt else None,
                        "our_choice": our_choice,
                        "our_kind": our_opt["kind"] if our_opt else None,
                        "our_score": our_opt["score"] if our_opt else None,
                        "agree": our_choice == dec["historical_chosen_index"],
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    n_written += 1
                    done.add(key)
            if touched_this_episode:
                n_episodes_touched += 1
            if n_written and n_written % 200 == 0:
                out_f.flush()
                print(f"  ...{n_written} decisions written so far")

    print(f"{group}: {n_written} new decisions written, {n_episodes_touched} episodes touched "
          f"this run, wrote {out_path}")
    return {"group": group, "n_written": n_written, "out_path": out_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("expert", "control"), required=True)
    ap.add_argument("--date", default="2026-07-22")
    args = ap.parse_args()

    deck = read_deck_csv(DECK_PATH)
    v1, weights = load_v1()
    run_diff(args.group, args.date, deck, v1, weights)


if __name__ == "__main__":
    main()
