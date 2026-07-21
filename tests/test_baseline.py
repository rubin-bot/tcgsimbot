"""Baseline vs random win-rate check (plain-assert, no pytest). Verifies the rule-based
baseline agent (src/baseline.py) actually plays better than random -- a sanity check that
its heuristics fire, not a rigorous strength benchmark."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sdk_path import ensure_cg_importable

ensure_cg_importable()

import cg.game as game  # noqa: E402

from baseline import choose_action  # noqa: E402
from obs import parse_obs  # noqa: E402

_DECK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "decks", "baseline_deck.csv")


def _load_deck() -> list[int]:
    with open(_DECK_PATH) as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def _random_choice(obs: dict) -> list[int]:
    sel = obs["select"]
    max_count = sel["maxCount"]
    n_opts = len(sel["option"])
    return random.sample(range(n_opts), max_count) if max_count > 0 else []


def _play_one_game(deck: list[int], baseline_seat: int, seed: int) -> int:
    """Returns the winning player index (0/1), or 2 for a draw."""
    random.seed(seed)
    obs, start = game.battle_start(deck, deck)
    assert start.errorPlayer == -1, f"battle_start failed: {start}"

    steps = 0
    try:
        while obs["current"]["result"] == -1:
            acting_player = obs["current"]["yourIndex"]
            if acting_player == baseline_seat:
                game_state, selection = parse_obs(obs)
                choice = choose_action(game_state, selection)
            else:
                choice = _random_choice(obs)
            obs = game.battle_select(choice)
            steps += 1
            if steps > 1000:
                raise RuntimeError("game did not terminate")
    finally:
        game.battle_finish()
    return obs["current"]["result"]


def main():
    deck = _load_deck()
    n_games = 50
    baseline_wins = 0
    draws = 0
    for i in range(n_games):
        baseline_seat = i % 2  # alternate which seat the baseline plays to avoid first-turn bias
        result = _play_one_game(deck, baseline_seat, seed=i)
        if result == 2:
            draws += 1
        elif result == baseline_seat:
            baseline_wins += 1

    win_rate = baseline_wins / n_games
    print(f"baseline wins: {baseline_wins}/{n_games} (draws: {draws}), win_rate={win_rate:.2f}")
    assert win_rate > 0.5, f"baseline should clearly beat random, got win_rate={win_rate:.2f}"
    print("BASELINE WIN-RATE TEST PASSED")


if __name__ == "__main__":
    main()
