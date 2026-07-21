"""Phase 1 verification for src/encode.py: shapes, determinism, finiteness, and -- most
importantly -- that the state encoder leaks NO hidden opponent information.

Run: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python tests/test_encode.py
"""

import os
import random
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from obs import CardRef, parse_obs  # noqa: E402
from encode import STATE_DIM, OPTION_DIM, encode_state, encode_option  # noqa: E402
from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

DECK = [int(x) for x in open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decks", "baseline_deck.csv"
)).read().split("\n")[:60]]


def collect_decisions(seed, limit):
    """Yield (obs_dict, gs, selection) for real mid-game decisions."""
    random.seed(seed)
    obs, sd = battle_start(DECK, DECK)
    assert obs is not None
    out = []
    for _ in range(limit):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            break
        gs, sel = parse_obs(obs)
        if gs is not None:
            out.append((obs, gs, sel))
        s = obs["select"]
        obs = battle_select(random.sample(range(len(s["option"])), s["maxCount"]))
    battle_finish()
    return out


def main():
    decisions = []
    for seed in range(6):
        decisions += collect_decisions(seed, 400)
    assert decisions, "no decisions collected"
    print(f"collected {len(decisions)} decisions; STATE_DIM={STATE_DIM} OPTION_DIM={OPTION_DIM}")

    hand_sensitive_checked = False
    for obs, gs, sel in decisions:
        v = encode_state(gs)
        assert v.shape == (STATE_DIM,), v.shape
        assert v.dtype == np.float32
        assert np.all(np.isfinite(v)), "non-finite state feature"

        for lo in sel.options:
            o = encode_option(gs, lo)
            assert o.shape == (OPTION_DIM,), o.shape
            assert np.all(np.isfinite(o))

        # opponent hand is genuinely hidden by the wrapper
        assert gs.opponent.hand is None, "opponent hand should be None (engine-hidden)"

        # determinism: re-parsing the same raw obs encodes identically
        gs2, _ = parse_obs(obs)
        assert np.array_equal(v, encode_state(gs2)), "encoding not deterministic"

        # INFO-HIDING: injecting a fake opponent hand must NOT change the encoding
        spoofed = replace(gs, opponent=replace(gs.opponent, hand=[CardRef(1, 99999), CardRef(2, 99998)]))
        assert np.array_equal(v, encode_state(spoofed)), "encoder leaked opponent hand!"

        # sanity: OWN hand DOES influence the encoding (so the hand features aren't dead)
        if not hand_sensitive_checked and gs.you.hand:
            changed = replace(gs, you=replace(gs.you, hand=gs.you.hand + [CardRef(6, 88888)]))
            assert not np.array_equal(v, encode_state(changed)), "own-hand features are dead"
            hand_sensitive_checked = True

    assert hand_sensitive_checked, "never exercised own-hand sensitivity"
    print("ALL ENCODE TESTS PASSED (shapes, determinism, finiteness, info-hiding)")


if __name__ == "__main__":
    main()
