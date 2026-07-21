"""Phase 3 verification for src/mcts.py.

With a RANDOM (untrained) net we cannot test *strength* -- pure value-net MCTS only gets strong
after training. So this tests CORRECTNESS: every move MCTS emits is engine-legal (full games
complete with zero errors), the returned policy/index_list are well-formed, and the search never
sees hidden opponent information.

Run: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python tests/test_mcts.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from obs import parse_obs  # noqa: E402
from net import PVNet  # noqa: E402
from mcts import search  # noqa: E402

DECK = [int(x) for x in open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decks", "baseline_deck.csv"
)).read().split("\n")[:60]]

SIMS = 12


def play_one_game(net, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, sd = battle_start(DECK, DECK)
    assert obs is not None
    decisions = 0
    for step in range(4000):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            battle_finish()
            return st["result"], decisions
        gs, sel = parse_obs(obs)
        assert gs is not None and sel is not None

        # info-hiding invariant holds at every real decision fed to search
        assert gs.opponent.hand is None, "opponent hand visible to search!"

        out = search(obs, net, DECK, sims=SIMS, temperature=1.0, add_noise=True)
        assert out is not None, "search returned None mid-game"
        policy, choice, index_list, root_sel = out

        # policy well-formed
        assert policy.shape == (len(sel.options),), (policy.shape, len(sel.options))
        assert abs(policy.sum() - 1.0) < 1e-5, policy.sum()
        assert 0 <= choice < len(sel.options)

        # index_list is a legal engine selection
        assert sel.min_count <= len(index_list) <= sel.max_count, (
            sel.min_count, len(index_list), sel.max_count)
        assert len(set(index_list)) == len(index_list), "duplicate indices"
        assert all(0 <= i < len(sel.options) for i in index_list), index_list

        decisions += 1
        obs = battle_select(index_list)
    battle_finish()
    raise AssertionError("game did not finish within step cap")


def main():
    net = PVNet(hidden=64)  # small random net; strength irrelevant here
    total_decisions = 0
    for seed in range(2):
        result, decisions = play_one_game(net, seed)
        total_decisions += decisions
        print(f"seed {seed}: game finished result={result} over {decisions} MCTS decisions")
    print(f"ALL MCTS TESTS PASSED: {total_decisions} legal MCTS moves across full games "
          f"(sims={SIMS}); policy + index_list valid; info-hiding held")


if __name__ == "__main__":
    main()
