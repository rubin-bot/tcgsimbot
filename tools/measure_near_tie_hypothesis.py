"""Stage 6 measurement cycle (2026-07-23): scales the near-tie/decline pilot from
tools/replay_ladder_decision.py to v1's full real-decision population, links declines to real
game outcomes, and cross-checks the local dry-run's "14/32 tied-and-lost" starvation pattern
against real ladder data for the first time. See
docs/near_tie_measurement_2026-07-23.md for the write-up this feeds.

Measurement only -- no agent or weight changes. Calls agents/search_scorer.py::choose_action()
directly, in-process (no cg battle session / subprocess needed, confirmed by the prior cycle's
pilot), so this is inherently light on the Hardware rules -- the concern here is surviving
interruption, not spawning simulator workers. Resumable: every completed decision's 20 replays
are appended to runs/near_tie_measurement/replays.jsonl immediately and flushed; a re-run skips
any (game, decision_index) already present in that file.

Usage:
  py -3.14 tools/measure_near_tie_hypothesis.py
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from ladder_episode_parser import parse_episode_file, find_our_seat  # noqa: E402
from kaggle_common import load_episode_submissions  # noqa: E402
from obs import parse_obs  # noqa: E402
from baseline import read_deck_csv  # noqa: E402
import search_scorer  # noqa: E402
import loss_review  # noqa: E402

V1_SUBMISSION_ID = "54909461"
N_REPLAYS = 20
OUR_DECK = read_deck_csv(os.path.join(ROOT, "decks", "crustle_wall_deck.csv"))
OUT_DIR = os.path.join(ROOT, "runs", "near_tie_measurement")
REPLAYS_PATH = os.path.join(OUT_DIR, "replays.jsonl")
EPISODES_ROOT = os.path.join(ROOT, "runs", "our_episodes")


# ---------------------------------------------------------------------------
# 1. Load v1's games, both via the parsed (decision-level) shape AND the raw episode dict
#    (needed for outcome-linkage's forward walk and for re-deriving each decision's raw
#    obs_dict/step index).
# ---------------------------------------------------------------------------

def find_episode_path(episode_id: str) -> str:
    for dirpath, _, filenames in os.walk(EPISODES_ROOT):
        if f"{episode_id}.json" in filenames:
            return os.path.join(dirpath, f"{episode_id}.json")
    raise FileNotFoundError(episode_id)


def load_v1_games() -> list[dict]:
    """Returns [{"episode_id":, "path":, "our_seat":, "raw":, "parsed":}, ...] for every
    episode currently attributed to v1 in runs/our_episodes/episode_submissions.json."""
    submissions = load_episode_submissions()
    v1_ids = sorted(eid for eid, sid in submissions.items() if sid == V1_SUBMISSION_ID)
    games = []
    for eid in v1_ids:
        path = find_episode_path(eid)
        parsed = parse_episode_file(path)
        if parsed is None:
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        our_seat = find_our_seat(raw["info"])
        games.append({"episode_id": eid, "path": path, "our_seat": our_seat, "raw": raw,
                       "parsed": parsed})
    return games


def active_decision_steps(raw: dict, our_seat: int) -> list[int]:
    """Raw step indices of every genuine (status=="ACTIVE") decision for our seat, in the same
    order/definition tools/ladder_episode_parser.py's fixed loop uses -- so index i here lines
    up 1:1 with parsed["decisions"][i]."""
    steps = raw["steps"]
    out = []
    for i in range(len(steps) - 1):
        if steps[i][our_seat].get("status") != "ACTIVE":
            continue
        obs = steps[i][our_seat]["observation"]
        if obs.get("select") is None or obs.get("current") is None:
            continue
        out.append(i)
    return out


# ---------------------------------------------------------------------------
# 2. Identify the DECLINE set (all v1 games) and STARVATION set (v1 LOSSES only, matching the
#    original local "14/32" analysis's own scope) -- see loss_review.py::analyze_attack_
#    availability / analyze_attacker_starved for the matching local criteria.
# ---------------------------------------------------------------------------

def chosen_option(dec: dict) -> dict | None:
    return next((o for o in dec["options"] if o["index"] == dec["chosen_index"]), None)


def find_decline_keys(games: list[dict]) -> set[tuple[str, int]]:
    keys = set()
    for g in games:
        for idx, dec in enumerate(g["parsed"]["decisions"]):
            kinds = [o["kind"] for o in dec["options"]]
            if "attack" not in kinds:
                continue
            chosen = chosen_option(dec)
            if chosen is None or chosen["kind"] != "attack":
                keys.add((g["episode_id"], idx))
    return keys


def our_crustles(dec: dict) -> list[dict]:
    mons = []
    if dec["you_active"] and dec["you_active"]["card_id"] == loss_review.CRUSTLE_ID:
        mons.append(dec["you_active"])
    for m in dec["you_bench"]:
        if m and m["card_id"] == loss_review.CRUSTLE_ID:
            mons.append(m)
    return mons


def find_starvation_keys(games: list[dict]) -> set[tuple[str, int]]:
    keys = set()
    for g in games:
        if g["parsed"]["outcome"] != "opponent_win":
            continue
        for idx, dec in enumerate(g["parsed"]["decisions"]):
            energy_opts = [o for o in dec["options"] if o["kind"] in ("energy", "attach")]
            if not energy_opts:
                continue
            unpowered = [m for m in our_crustles(dec) if len(m["energies"]) < 3]
            if not unpowered:
                continue
            unpowered_serials = {m["serial"] for m in unpowered}
            crustle_opts = [o for o in energy_opts if o["target_serial"] in unpowered_serials]
            if not crustle_opts:
                continue
            chosen = chosen_option(dec)
            if chosen is not None and chosen["target_serial"] in unpowered_serials:
                continue  # not starved -- we did power it
            keys.add((g["episode_id"], idx))
    return keys


# ---------------------------------------------------------------------------
# 3. Replay-and-vote: N_REPLAYS calls to choose_action() per decision, resumable via
#    REPLAYS_PATH.
# ---------------------------------------------------------------------------

def load_done_keys() -> set[tuple[str, int]]:
    if not os.path.exists(REPLAYS_PATH):
        return set()
    done = set()
    with open(REPLAYS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["game"], rec["decision_index"]))
    return done


def replay_decision(game: dict, decision_index: int, raw_step_index: int) -> dict:
    obs_dict = game["raw"]["steps"][raw_step_index][game["our_seat"]]["observation"]
    game_state, selection = parse_obs(obs_dict)
    replays = []
    for _ in range(N_REPLAYS):
        captured = {}

        def trace_fn(rec, _captured=captured):
            _captured.update(rec)

        t0 = time.perf_counter()
        result = search_scorer.choose_action(game_state, selection, obs_dict, OUR_DECK, None,
                                              trace_fn=trace_fn)
        elapsed = time.perf_counter() - t0
        # trace_fn's record shape (agents/search_scorer.py::_trace_options): score/features
        # live per-option inside "options", not as separate top-level dicts.
        opts = captured.get("options") or []
        scores = {o["index"]: o["score"] for o in opts if o["score"] is not None}
        features = {o["index"]: o["features"] for o in opts if o["features"] is not None}
        replays.append({
            "mode": captured.get("mode"),
            "chosen_index": result[0] if result else None,
            "scores": scores or None,
            "features_by_index": features or None,
            "elapsed_s": round(elapsed, 4),
        })
    dec = game["parsed"]["decisions"][decision_index]
    return {
        "game": game["episode_id"], "decision_index": decision_index,
        "raw_step_index": raw_step_index,
        "turn": dec["turn"], "turn_action_count": dec["turn_action_count"],
        "outcome": game["parsed"]["outcome"],
        "options": dec["options"], "ladder_chosen_index": dec["chosen_index"],
        "replays": replays,
    }


def run_replays(games: list[dict], keys: set[tuple[str, int]]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    done = load_done_keys()
    todo = sorted(k for k in keys if k not in done)
    print(f"{len(keys)} decisions to replay, {len(done & keys)} already done, "
          f"{len(todo)} remaining")
    games_by_id = {g["episode_id"]: g for g in games}
    with open(REPLAYS_PATH, "a", encoding="utf-8") as f:
        for n, (episode_id, decision_index) in enumerate(todo):
            game = games_by_id[episode_id]
            steps = active_decision_steps(game["raw"], game["our_seat"])
            raw_step_index = steps[decision_index]
            rec = replay_decision(game, decision_index, raw_step_index)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (n + 1) % 10 == 0 or n + 1 == len(todo):
                print(f"  {n + 1}/{len(todo)} decisions replayed ({N_REPLAYS} samples each)")


def main() -> None:
    print("loading v1's games ...")
    games = load_v1_games()
    print(f"  {len(games)} v1 games loaded")

    decline_keys = find_decline_keys(games)
    starvation_keys = find_starvation_keys(games)
    print(f"DECLINE set (all games): {len(decline_keys)} decisions")
    print(f"STARVATION set (losses only): {len(starvation_keys)} decisions")

    all_keys = decline_keys | starvation_keys
    run_replays(games, all_keys)
    print(f"\nwrote {REPLAYS_PATH}")


if __name__ == "__main__":
    main()
