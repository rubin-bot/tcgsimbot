"""Stage 6 DIAGNOSE: builds a ranked shortcomings report from SearchScorer's losses.

Reuses tools/loss_review.py's six analyses (attack-decline, attacker-starved,
energy-routing-detail, avoidable-KOs, evolve-misplay, near-ties) rather than duplicating them
-- see that module for what each one checks and why.

Two sources of losses:

  --source local: runs tools/eval_arena.py --candidate search_scorer --replay-out ... against
    every opponent in the local slate (baseline, random -- the only two that exist today) and
    feeds the resulting losses through loss_review.py's analyses directly.

  --source ladder --episodes DIR: parses real Kaggle episode JSON files (pulled down by
    tools/measure.py) into the same per-decision trace shape loss_review.py consumes, so the
    shared analyses run unmodified. Ladder replays carry no `score`/`features` fields (those
    come from our own trace_fn instrumentation, never recorded by Kaggle) -- the near-tie and
    energy-routing weight-imbalance/horizon-blind classification are LOCAL-ONLY and are skipped
    with an explicit note in ladder mode rather than emitting numbers that look real but aren't.
    Ladder mode adds checks local baseline/random structurally cannot: opponent archetype
    (Trainer-card usage, Pokemon lines seen) and win/loss broken out by archetype.

  --source auto (default): reads runs/measure_state.json (written by tools/measure.py) -- uses
    ladder mode if it found any of our episodes in the newest dump, else falls back to local.
    This is what a plain "iterate" invokes without the caller needing to know which mode
    applies on a given day.

Usage:
  .venv/Scripts/python tools/autopsy.py --source auto
  .venv/Scripts/python tools/autopsy.py --source local --games-per-opponent 60
  .venv/Scripts/python tools/autopsy.py --source ladder --episodes runs/our_episodes/
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import loss_review  # noqa: E402

LOCAL_OPPONENT_SLATE = ("baseline", "random")  # the only two local opponents that exist today
# CardType has no single TRAINER member -- Trainer cards split into these four (confirmed
# against cg.api.CardType: POKEMON/ITEM/TOOL/SUPPORTER/STADIUM/BASIC_ENERGY/SPECIAL_ENERGY).
TRAINER_CARD_TYPE_NAMES = {"ITEM", "TOOL", "SUPPORTER", "STADIUM"}
MEASURE_STATE_PATH = os.path.join(ROOT, "runs", "measure_state.json")
EVAL_ARENA = os.path.join(ROOT, "tools", "eval_arena.py")
DECK = os.path.join(ROOT, "decks", "crustle_wall_deck.csv")


# ---------------------------------------------------------------------------
# --source local: run fresh arena games, capture losses via --replay-out
# ---------------------------------------------------------------------------

def _run_local_losses(games_per_opponent: int, workers: int, out_dir: str) -> dict[str, list]:
    os.makedirs(out_dir, exist_ok=True)
    losses_by_opponent: dict[str, list] = {}
    for opponent in LOCAL_OPPONENT_SLATE:
        replay_path = os.path.join(out_dir, f"replay_vs_{opponent}.jsonl")
        results_path = os.path.join(out_dir, f"results_vs_{opponent}.jsonl")
        resume = os.path.exists(results_path) and os.path.getsize(results_path) > 0
        cmd = [
            sys.executable, EVAL_ARENA,
            "--candidate", "search_scorer", "--opponent", opponent,
            "--candidate-deck", DECK, "--opponent-deck", DECK,
            "--games", str(games_per_opponent), "--workers", str(workers),
            "--out", results_path, "--replay-out", replay_path,
        ]
        if resume:
            cmd.append("--resume")
        print(f"--- running {games_per_opponent} games vs {opponent} ---", file=sys.stderr)
        subprocess.run(cmd, check=True)
        losses_by_opponent[opponent] = loss_review.load_losses(replay_path)
    return losses_by_opponent


# ---------------------------------------------------------------------------
# --source ladder: parse Kaggle episode JSON into loss_review.py's decision-trace shape
# ---------------------------------------------------------------------------

def _load_ladder_games(episodes_dir: str) -> list[dict]:
    """Parses every episode JSON under episodes_dir into a loss_review.py-shaped record (any
    outcome -- callers filter for losses themselves; archetype win/loss breakdown needs wins
    too). Walks recursively since tools/measure.py now lays episodes out as
    runs/our_episodes/<date>/<episode_id>.json (one subdir per scanned day) rather than a
    single flat directory. Concrete field names are filled in against a real sample episode
    (see runs/measure_state.json / tools/measure.py for how episodes got here) -- this raises
    loudly rather than guessing at a schema if a file doesn't look like what we expect, so a
    format surprise shows up as an error, not a silently-empty report.

    Tags each record with submission_id/submission_label from
    kaggle_common.load_episode_submissions() -- episode JSON itself carries no submission id
    (only team name), so this is the only way to tell which of our active submissions produced
    a given game. See docs/ladder_attack_decline_diagnosis_2026-07-23.md."""
    from ladder_episode_parser import parse_episode_file  # noqa: E402 (local import, see below)
    from kaggle_common import load_episode_submissions, submission_label  # noqa: E402

    episode_submissions = load_episode_submissions()

    games = []
    for dirpath, _, filenames in os.walk(episodes_dir):
        for name in sorted(filenames):
            if not name.endswith(".json") or name == "episode_submissions.json":
                continue
            path = os.path.join(dirpath, name)
            rec = parse_episode_file(path)
            if rec is not None:
                submission_id = episode_submissions.get(rec["game"])
                rec["submission_id"] = submission_id
                rec["submission_label"] = submission_label(submission_id)
                games.append(rec)
    return games


# ---------------------------------------------------------------------------
# Ladder-only: opponent archetype detection (local baseline/random can't teach us this)
# ---------------------------------------------------------------------------

def analyze_opponent_archetypes(all_ladder_games: list[dict]) -> None:
    print("\n=== (ladder-only) Opponent archetypes seen, and win/loss by archetype ===")
    if not all_ladder_games:
        print("no ladder games available yet.")
        return
    try:
        from carddata import load_card_index
        cards = load_card_index()
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"could not load card index ({e!r}); showing raw card ids only.")
        cards = {}

    by_archetype = collections.Counter()
    wins_by_archetype = collections.Counter()
    trainer_using_opponents = 0
    for game in all_ladder_games:
        opp_card_ids = game.get("opponent_card_ids", [])
        names = sorted({cards[c].name for c in opp_card_ids if c in cards}) or ["unknown"]
        key = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
        by_archetype[key] += 1
        if game["outcome"] == "candidate_win":
            wins_by_archetype[key] += 1
        if any(cards.get(c) and cards[c].card_type.name in TRAINER_CARD_TYPE_NAMES
               for c in opp_card_ids):
            trainer_using_opponents += 1

    for key, n in by_archetype.most_common():
        w = wins_by_archetype[key]
        print(f"  {key}: {n} games, {w}/{n} won ({w / n:.1%})")
    print(f"opponents seen running Trainer cards: {trainer_using_opponents}/{len(all_ladder_games)}")


# ---------------------------------------------------------------------------

def _print_report(losses: list[dict], source_label: str, skip_local_only: bool) -> None:
    total_decisions = sum(len(g["decisions"]) for g in losses)
    print(f"\n############ {source_label}: {len(losses)} losses, {total_decisions} logged "
          f"decisions ############")
    if not losses:
        print("no losses to analyze.")
        return

    loss_review.analyze_attack_availability(losses)
    loss_review.analyze_attacker_starved(losses)
    if skip_local_only:
        print("\n=== Energy-routing detail / near-ties: SKIPPED in ladder mode -- these need "
              "evaluate()'s own score/features, which Kaggle never records (only our local "
              "trace_fn does). Run --source local for these two. ===")
    else:
        loss_review.analyze_energy_routing_detail(losses)
        loss_review.analyze_near_ties(losses)
    loss_review.analyze_avoidable_kos(losses)
    loss_review.analyze_evolve_misplay(losses)


def _resolve_auto_source() -> str:
    if not os.path.exists(MEASURE_STATE_PATH):
        print("no runs/measure_state.json found (run tools/measure.py first) -- "
              "defaulting to --source local.", file=sys.stderr)
        return "local"
    with open(MEASURE_STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("our_episode_count", 0) > 0 and state.get("our_episodes_dir"):
        return "ladder"
    print(f"measure_state.json shows 0 of our episodes as of {state.get('checked_at')} -- "
          f"falling back to --source local.", file=sys.stderr)
    return "local"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("auto", "local", "ladder"), default="auto")
    ap.add_argument("--episodes", default=None, help="ladder mode: dir of episode JSON files "
                     "(defaults to measure_state.json's our_episodes_dir)")
    ap.add_argument("--games-per-opponent", type=int, default=60,
                     help="local mode: games run against EACH opponent in the slate")
    ap.add_argument("--workers", type=int, default=2, help="local mode: max 2 per hardware "
                     "rules (each worker is a full engine subprocess)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "autopsy_local"))
    ap.add_argument("--split-by-submission", action="store_true",
                     help="ladder mode: run the report once per submission_label (e.g. "
                          "v1_search_scorer vs. net_checkpoint) instead of once for the pooled "
                          "set -- see docs/ladder_attack_decline_diagnosis_2026-07-23.md")
    args = ap.parse_args()

    source = _resolve_auto_source() if args.source == "auto" else args.source

    if source == "local":
        losses_by_opponent = _run_local_losses(args.games_per_opponent, args.workers,
                                                 args.out_dir)
        for opponent, losses in losses_by_opponent.items():
            _print_report(losses, f"LOCAL vs {opponent}", skip_local_only=False)
    else:
        episodes_dir = args.episodes
        if episodes_dir is None and os.path.exists(MEASURE_STATE_PATH):
            with open(MEASURE_STATE_PATH, encoding="utf-8") as f:
                episodes_dir = json.load(f).get("our_episodes_dir")
        if not episodes_dir or not os.path.isdir(episodes_dir):
            sys.exit(f"ladder mode needs a valid episodes dir; got {episodes_dir!r}")
        all_games = _load_ladder_games(episodes_dir)

        if args.split_by_submission:
            by_label: dict[str, list] = {}
            for g in all_games:
                by_label.setdefault(g["submission_label"], []).append(g)
            for label, games in sorted(by_label.items()):
                losses = [g for g in games if g["outcome"] == "opponent_win"]
                _print_report(losses, f"LADDER [{label}] ({episodes_dir})", skip_local_only=True)
                analyze_opponent_archetypes(games)
        else:
            losses = [g for g in all_games if g["outcome"] == "opponent_win"]
            _print_report(losses, f"LADDER ({episodes_dir})", skip_local_only=True)
            analyze_opponent_archetypes(all_games)


if __name__ == "__main__":
    main()
