"""v3: multi-sample determinization voting (agents/search_scorer.py -- N_DETERMINIZATIONS,
_aggregate_votes, DECISION_TIME_GUARD_S). Two parts:

1. Aggregation logic (_aggregate_votes) is a pure function of (votes, score_sum, selection) --
   no search/engine access -- tested here with synthetic vote/score data covering: a clean
   majority winner, a vote-count tie broken by score-sum, and a residual exact tie (vote AND
   score-sum both tied) falling through to v2's target-aware _tie_break_key.
2. The time-guard (anytime behavior: return best-so-far from fully-completed world samples if a
   decision runs long) is tested against a REAL decision (reusing tests/test_tie_break.py's
   episode-fixture loader) with an injected per-world delay, so it exercises the real
   choose_action() code path rather than a synthetic stand-in.

Run: PYTHONIOENCODING=utf-8 py -3.14 tests/test_v3_voting.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from obs import LegalOption, PokemonView  # noqa: E402
import search_scorer  # noqa: E402
import test_tie_break  # noqa: E402  (reuses its real-episode fixture loader)


def _mon(card_id: int, energies: list) -> PokemonView:
    return PokemonView(card_id=card_id, serial=card_id * 100, hp=100, max_hp=100,
                        appear_this_turn=False, energies=energies, energy_cards=[], tools=[],
                        pre_evolution=[])


def _opt(index: int, kind: str, target: PokemonView | None = None) -> LegalOption:
    return LegalOption(index=index, kind=kind, raw=None, card=None, target=target)


def test_clear_majority_wins_regardless_of_score_sum() -> None:
    """Option 0 gets more votes than option 1 despite a lower total score -- majority vote is
    the primary signal, score-sum is only a tie-break among vote-tied options."""
    votes = Counter({0: 5, 1: 3})
    score_sum = {0: 10.0, 1: 12.0}
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1,
                                         options=[_opt(0, "attach"), _opt(1, "attach")])
    winner = search_scorer._aggregate_votes(votes, score_sum, selection)
    assert winner == 0, f"expected majority-vote winner 0, got {winner}"


def test_vote_tie_broken_by_score_sum() -> None:
    votes = Counter({0: 4, 1: 4})
    score_sum = {0: 10.0, 1: 12.0}
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1,
                                         options=[_opt(0, "attach"), _opt(1, "attach")])
    winner = search_scorer._aggregate_votes(votes, score_sum, selection)
    assert winner == 1, f"expected score-sum winner 1 (higher sum), got {winner}"


def test_double_tie_falls_through_to_v2_kind_priority() -> None:
    """Vote count AND score-sum both tied -- residual tie must fall through to
    _tie_break_key, which prefers "attack" (priority 0) over "attach" (priority 2)."""
    votes = Counter({0: 4, 1: 4})
    score_sum = {0: 10.0, 1: 10.0}
    selection = search_scorer.Selection(
        select_type=None, context=None, min_count=1, max_count=1,
        options=[_opt(0, "attach"), _opt(1, "attack")])
    winner = search_scorer._aggregate_votes(votes, score_sum, selection)
    assert winner == 1, f"expected attack (index 1, kind priority 0) to win over attach, got {winner}"


def test_double_tie_falls_through_to_v2_pipeline_proximity() -> None:
    """Same kind, both in the attacker pipeline, different energy deficits -- the option
    closer to powered (smaller deficit) must win, matching v2's _pipeline_energy_deficit."""
    dwebble_far = _mon(search_scorer.DWEBBLE_ID, energies=[])           # deficit 3
    crustle_close = _mon(search_scorer.CRUSTLE_ID, energies=["G", "C"])  # deficit 1
    votes = Counter({0: 4, 1: 4})
    score_sum = {0: 10.0, 1: 10.0}
    selection = search_scorer.Selection(
        select_type=None, context=None, min_count=1, max_count=1,
        options=[_opt(0, "attach", target=dwebble_far),
                 _opt(1, "attach", target=crustle_close)])
    winner = search_scorer._aggregate_votes(votes, score_sum, selection)
    assert winner == 1, f"expected the closer-to-powered target (index 1) to win, got {winner}"


def test_time_guard_returns_best_so_far_on_real_decision() -> None:
    """Injects an artificial per-world delay into a REAL decision (reusing test_tie_break's
    episode fixture) and shrinks the time guard so it fires well before N_DETERMINIZATIONS
    worlds complete. Asserts choose_action still returns a valid decision, fewer worlds were
    sampled than the configured N, and the trace records time_guard_fired=True."""
    episode_id, raw_step_index, _expected_serial, _desc = test_tie_break.CASES[0]
    obs_dict, game_state, selection = test_tie_break.load_decision(episode_id, raw_step_index)

    real_score_world = search_scorer._score_world

    def slow_score_world(*args, **kwargs):
        time.sleep(0.08)
        return real_score_world(*args, **kwargs)

    orig_guard = search_scorer.DECISION_TIME_GUARD_S
    orig_score_world = search_scorer._score_world
    search_scorer.DECISION_TIME_GUARD_S = 0.15  # trips after ~1-2 slowed worlds
    search_scorer._score_world = slow_score_world
    try:
        captured = {}

        def trace_fn(rec, _c=captured):
            _c.update(rec)

        result = search_scorer.choose_action(
            game_state, selection, obs_dict, test_tie_break.OUR_DECK, None, trace_fn=trace_fn)
    finally:
        search_scorer.DECISION_TIME_GUARD_S = orig_guard
        search_scorer._score_world = orig_score_world

    assert captured.get("mode") == "searched", f"expected searched mode, got {captured.get('mode')!r}"
    assert result, "expected a non-empty legal index list even under the time guard"
    n_completed = captured["n_samples_completed"]
    assert n_completed < search_scorer.N_DETERMINIZATIONS, (
        f"expected the time guard to cut sampling short of N_DETERMINIZATIONS="
        f"{search_scorer.N_DETERMINIZATIONS}, got n_samples_completed={n_completed}")
    assert captured["time_guard_fired"] is True, "expected time_guard_fired=True in the trace"
    print(f"    time guard fired after {n_completed}/{search_scorer.N_DETERMINIZATIONS} worlds, "
          f"decision still valid: {result}")


def main() -> None:
    tests = [
        test_clear_majority_wins_regardless_of_score_sum,
        test_vote_tie_broken_by_score_sum,
        test_double_tie_falls_through_to_v2_kind_priority,
        test_double_tie_falls_through_to_v2_pipeline_proximity,
        test_time_guard_returns_best_so_far_on_real_decision,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failures.append(t.__name__)
    if failures:
        raise SystemExit(f"{len(failures)}/{len(tests)} test(s) FAILED: {failures}")
    print(f"ALL {len(tests)} V3 VOTING TESTS PASSED")


if __name__ == "__main__":
    main()
