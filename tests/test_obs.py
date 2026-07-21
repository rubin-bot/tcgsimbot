"""Plain-assert test (no pytest dependency): exercises parse_obs/decode_options against
real games and checks the information-hiding invariant (opponent hand must never be
visible). Drives play using ONLY the wrapper's decoded LegalOption.index values, which
also proves those indices line up with what battle_select actually expects -- the same
mechanism Phase 3's baseline agent will rely on.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sdk_path import ensure_cg_importable

ensure_cg_importable()

import cg.game as game  # noqa: E402

from obs import parse_obs  # noqa: E402

_DECK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "pokemon-tcg-ai-battle",
    "sample_submission", "sample_submission", "deck.csv")


def _load_deck() -> list[int]:
    with open(_DECK_PATH) as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def _run_random_game_via_wrapper(deck: list[int], seed: int):
    random.seed(seed)
    obs, start = game.battle_start(deck, deck)
    assert start.errorPlayer == -1, f"battle_start failed: {start}"

    steps = 0
    kinds_seen = set()
    try:
        while obs["current"]["result"] == -1:
            game_state, selection = parse_obs(obs)
            assert game_state is not None and selection is not None

            # Information-hiding invariant.
            assert game_state.opponent.hand is None, "opponent hand leaked into GameState!"
            assert game_state.you.hand is not None
            assert len(game_state.you.hand) == game_state.you.hand_count

            for lo in selection.options:
                kinds_seen.add(lo.kind)
                assert not lo.kind.startswith("unknown_"), \
                    f"unclassified OptionType: {lo.raw.type}"

            assert selection.max_count == obs["select"]["maxCount"]
            assert selection.min_count == obs["select"]["minCount"]

            max_count = selection.max_count
            chosen = (random.sample([lo.index for lo in selection.options], max_count)
                      if max_count > 0 else [])
            obs = game.battle_select(chosen)
            steps += 1
            if steps > 500:
                break
    finally:
        game.battle_finish()

    return steps, obs["current"]["result"], kinds_seen


def main():
    deck = _load_deck()
    all_kinds = set()
    for seed in range(15):
        steps, result, kinds = _run_random_game_via_wrapper(deck, seed)
        all_kinds |= kinds
        assert result in (0, 1, 2), f"game did not finish cleanly: result={result}"
        print(f"seed={seed}: {steps} steps, result={result}")
    print("Option kinds observed across all games:", sorted(all_kinds))
    print("ALL OBS/DECODE TESTS PASSED")


if __name__ == "__main__":
    main()
