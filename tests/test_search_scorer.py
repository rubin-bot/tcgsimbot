"""Stage 1 verification for agents/search_scorer.py.

Plays SearchScorer (piloting decks/crustle_wall_deck.csv) against a random-action agent
(piloting decks/baseline_deck.csv, modeled on smoke_test.py's random_agent) for a full local
battle. Asserts every SearchScorer move is engine-legal and the game reaches a result within
a step cap -- i.e. a complete battle with zero crashes and zero illegal moves.

Run: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python tests/test_search_scorer.py
"""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from obs import parse_obs  # noqa: E402
from search_scorer import make_agent  # noqa: E402

OUR_DECK = [int(x) for x in open(os.path.join(ROOT, "decks", "crustle_wall_deck.csv"))
            .read().split("\n")[:60]]
OPP_DECK = [int(x) for x in open(os.path.join(ROOT, "decks", "baseline_deck.csv"))
            .read().split("\n")[:60]]

STEP_CAP = 4000


def random_agent(obs_dict: dict) -> list[int]:
    _, sel = parse_obs(obs_dict)
    if sel is None:
        return OPP_DECK
    n = len(sel.options)
    k = sel.max_count
    return random.sample(range(n), k)


def play_one_game(seed: int, search_scorer_seat: int):
    random.seed(seed)
    our_agent = make_agent(OUR_DECK)
    agents = {search_scorer_seat: our_agent, 1 - search_scorer_seat: random_agent}
    decks = {search_scorer_seat: OUR_DECK, 1 - search_scorer_seat: OPP_DECK}

    obs, start_data = battle_start(decks[0], decks[1])
    assert obs is not None, (start_data.errorPlayer, start_data.errorType)

    decisions = 0
    for _ in range(STEP_CAP):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            battle_finish()
            return st["result"], decisions

        yi = st["yourIndex"]
        _, sel = parse_obs(obs)
        assert sel is not None

        index_list = agents[yi](obs)

        if yi == search_scorer_seat:
            assert sel.min_count <= len(index_list) <= sel.max_count, (
                sel.min_count, len(index_list), sel.max_count)
            assert len(set(index_list)) == len(index_list), "duplicate indices"
            assert all(0 <= i < len(sel.options) for i in index_list), index_list
            decisions += 1

        obs = battle_select(index_list)
    battle_finish()
    raise AssertionError("game did not finish within step cap")


def main():
    total_decisions = 0
    for seed in range(2):
        for seat in (0, 1):
            result, decisions = play_one_game(seed=seed, search_scorer_seat=seat)
            total_decisions += decisions
            outcome = "draw" if result == 2 else ("won" if result == seat else "lost")
            print(f"seed={seed} seat={seat}: game finished result={result} "
                  f"(SearchScorer {outcome}) over {decisions} SearchScorer decisions")
    print(f"ALL SEARCH_SCORER TESTS PASSED: {total_decisions} legal SearchScorer moves "
          f"across full games vs. random agent; zero crashes, zero illegal moves")


if __name__ == "__main__":
    main()
