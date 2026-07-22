"""Evaluation ladder.

Plays a candidate agent against a FIXED reference set (the rule-based baseline + frozen
checkpoints) and reports win rates. Win rate against a fixed set -- not against the current
training opponent -- is the trustworthy signal of progress (the philosophy's requirement).

Seats are alternated across games so a first-player advantage can't skew the number.
"""

from __future__ import annotations

import numpy as np

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from mcts import search  # noqa: E402
from baseline import agent as baseline_agent  # noqa: E402


def make_net_agent(net, deck_list, sims=50):
    """Greedy (temperature≈0, no exploration noise) net+MCTS agent for evaluation/inference."""
    def _agent(obs_dict):
        out = search(obs_dict, net, deck_list, sims=sims, temperature=0.0, add_noise=False)
        if out is None:                     # deck-selection phase
            return list(deck_list)
        _, _, index_list, _ = out
        return index_list
    return _agent


def play_match(agent_a, agent_b, deck_a, deck_b, seed, a_is_player0=True):
    """Play one game; return +1 if A wins, -1 if B wins, 0 draw. `agent_a` sits in the seat
    given by a_is_player0 piloting `deck_a` (agent_b/deck_b likewise); dispatch each decision
    to whichever seat is to move. deck_a/deck_b may be the same list (mirror) or different
    (asymmetric matchup, e.g. sanity-checking against a non-mirror opponent)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    agents = {0: agent_a, 1: agent_b} if a_is_player0 else {0: agent_b, 1: agent_a}
    decks = {0: deck_a, 1: deck_b} if a_is_player0 else {0: deck_b, 1: deck_a}
    a_seat = 0 if a_is_player0 else 1
    obs, sd = battle_start(decks[0], decks[1])
    if obs is None:
        return 0
    for _ in range(6000):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            battle_finish()
            r = st["result"]
            if r == 2:
                return 0
            return 1 if r == a_seat else -1
        yi = st["yourIndex"]
        obs = battle_select(agents[yi](obs))
    battle_finish()
    return 0


def win_rate(candidate_agent, opponent_agent, candidate_deck, opponent_deck, n_games, base_seed=0):
    """Fraction of decisive games the candidate wins (draws count as 0.5), seats alternated.
    candidate_deck/opponent_deck may differ (asymmetric matchup)."""
    wins = draws = 0
    for g in range(n_games):
        r = play_match(candidate_agent, opponent_agent, candidate_deck, opponent_deck,
                       seed=base_seed + g, a_is_player0=(g % 2 == 0))
        if r > 0:
            wins += 1
        elif r == 0:
            draws += 1
    return (wins + 0.5 * draws) / max(n_games, 1)


def evaluate(candidate_net, opponents: dict, deck_list, n_games=40, sims=50, base_seed=0,
             opponent_decks: dict | None = None):
    """opponents: {name: agent_fn}. opponent_decks: optional {name: deck_list} for opponents
    that pilot a different deck than the candidate (default: mirror, same deck_list).
    Returns {name: candidate win rate}."""
    cand = make_net_agent(candidate_net, deck_list, sims=sims)
    opponent_decks = opponent_decks or {}
    results = {}
    for name, opp in opponents.items():
        opp_deck = opponent_decks.get(name, deck_list)
        results[name] = win_rate(cand, opp, deck_list, opp_deck, n_games, base_seed=base_seed)
    return results
