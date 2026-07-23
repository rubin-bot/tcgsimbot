"""One-off diagnostic (2026-07-23, docs/ladder_attack_decline_diagnosis_2026-07-23.md, Q6): for
a handful of real v1 ladder decisions where an attack was legal but declined, replay the exact
recorded obs_dict through our own agents/search_scorer.py::agent() locally, 10x each (its
determinization step samples randomly and production never fixes an opponent-deck prior, so a
single replay isn't representative), and report what it chooses + how long each call takes.

obs_dict is proven byte-identical to what the live agent received on Kaggle (same schema
src/obs.py::parse_obs() decodes for the live agent -- see tools/ladder_episode_parser.py's
docstring), and agent() is fully self-contained (only needs obs_dict; deck_list/opp_deck_list
default exactly as production leaves them) -- no live battle session needs reconstructing.

Not permanent tooling -- a labeled one-off for this diagnosis cycle, not wired into iterate.
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

from ladder_episode_parser import parse_episode_file  # noqa: E402
from kaggle_common import load_episode_submissions  # noqa: E402
from baseline import read_deck_csv  # noqa: E402
import search_scorer  # noqa: E402

V1_SUBMISSION_ID = "54909461"
N_REPLAYS = 10
# On Kaggle, agent()'s bare read_deck_csv() resolves /kaggle_simulations/agent/deck.csv (the
# packaged deck). Locally there is no such path (CLAUDE.md: "No deck.csv exists at repo root
# yet; copy/rename this in at packaging time") -- read_deck_csv() would raise FileNotFoundError
# and get swallowed by agent()'s own except-Exception-fallback, producing a FALSE positive
# ("exception_to_baseline" on every call) that has nothing to do with the real ladder state.
# Passing the same 60 card ids explicitly sidesteps that path difference without changing what
# agent() actually does with them.
OUR_DECK = read_deck_csv(os.path.join(ROOT, "decks", "crustle_wall_deck.csv"))

# (episode_id, decision_index_within_parsed_decisions, human note) -- hand-picked for diversity
# of chosen kind (attach/evolve) and game/turn, from the 110 real v1 declined-attack decisions
# found post status-filter-fix (see docs/ladder_attack_decline_diagnosis_2026-07-23.md).
CASES = [
    ("87506107", 5, "turn 3.1, chose attach over attack"),
    ("87507277", 26, "turn 16.2, chose evolve over attack"),
    ("87508450", 26, "turn 12.1, chose attach over attack"),
    ("87509628", 15, "turn 6.2, chose evolve over attack"),
    ("87508450", 64, "turn 36.1, chose attach over attack (late game)"),
]


def find_episode_path(episode_id: str) -> str:
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, "runs", "our_episodes")):
        if f"{episode_id}.json" in filenames:
            return os.path.join(dirpath, f"{episode_id}.json")
    raise SystemExit(f"episode {episode_id} not found under runs/our_episodes/")


def find_our_seat(episode: dict) -> int:
    from kaggle_common import OUR_TEAM_NAME
    names = episode["info"]["TeamNames"]
    for i, n in enumerate(names):
        if n.strip().lower() == OUR_TEAM_NAME.strip().lower():
            return i
    raise SystemExit("our team not found in this episode")


def raw_obs_dict_for_decision(episode_id: str, decision_index: int) -> dict:
    """Re-derives which raw step index a parsed decision came from by replaying the same
    ACTIVE-status filter tools/ladder_episode_parser.py now uses, so the obs_dict handed to
    agent() here is exactly the one that decision's parsed record was built from."""
    path = find_episode_path(episode_id)
    with open(path, encoding="utf-8") as f:
        episode = json.load(f)
    our_seat = find_our_seat(episode)
    steps = episode["steps"]
    seen = 0
    for i in range(len(steps) - 1):
        if steps[i][our_seat].get("status") != "ACTIVE":
            continue
        obs_dict = steps[i][our_seat]["observation"]
        if obs_dict.get("select") is None or obs_dict.get("current") is None:
            continue
        if seen == decision_index:
            return obs_dict
        seen += 1
    raise SystemExit(f"decision index {decision_index} not found in {episode_id} "
                      f"(only {seen} ACTIVE decisions)")


def kind_of(obs_dict: dict, chosen_indices: list[int]) -> str | None:
    from obs import parse_obs
    _, selection = parse_obs(obs_dict)
    if not chosen_indices:
        return None
    idx = chosen_indices[0]
    for lo in selection.options:
        if lo.index == idx:
            return lo.kind
    return None


def main() -> None:
    submissions = load_episode_submissions()
    for episode_id, decision_index, note in CASES:
        sid = submissions.get(episode_id)
        assert sid == V1_SUBMISSION_ID, f"{episode_id} maps to {sid!r}, expected v1"
        obs_dict = raw_obs_dict_for_decision(episode_id, decision_index)
        print(f"\n=== episode {episode_id} decision #{decision_index}: {note} ===")

        kinds_seen = []
        times = []
        before_counts = dict(search_scorer._FALLBACK_COUNTS)
        for run in range(N_REPLAYS):
            t0 = time.perf_counter()
            result = search_scorer.agent(obs_dict, deck_list=OUR_DECK)
            elapsed = time.perf_counter() - t0
            kind = kind_of(obs_dict, result)
            kinds_seen.append(kind)
            times.append(elapsed)
        after_counts = dict(search_scorer._FALLBACK_COUNTS)
        deltas = {k: after_counts[k] - before_counts.get(k, 0) for k in after_counts
                  if after_counts[k] - before_counts.get(k, 0) != 0}

        from collections import Counter
        print(f"  chosen kinds across {N_REPLAYS} local replays: {dict(Counter(kinds_seen))}")
        print(f"  wall-clock per call: min={min(times):.3f}s max={max(times):.3f}s "
              f"mean={sum(times) / len(times):.3f}s")
        print(f"  fallback-tier deltas this decision: {deltas or '(none -- normal search path)'}")


if __name__ == "__main__":
    main()
