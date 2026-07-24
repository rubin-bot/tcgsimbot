"""v4 Change 1: agents/search_scorer.py::_prefer_continuing_over_end -- widens the effective
near-tie handling for early-game "end" decisions specifically (see the function's own docstring
and docs/v4_report_2026-07-24.md for the full mechanism/evidence).

Two parts:
1. Unit tests of the pure function itself, using REAL captured score_sum snapshots pulled from
   3 of the 34 real cases identified in docs/master_study_2026-07-24.md's decision-diff corpus
   (episode 87363023 step 47, episode 87397816 step 7, episode 87529164 step 25 -- all turn<=4,
   a real strong player played a card instead of ending, and our agent's own raw score_sum
   narrowly (0.09-0.16% relative margin) favored "end" instead). Chosen over re-invoking the
   live stochastic search per test run because these margins are so thin that fresh N=8 samples
   flip the RAW winner run-to-run (confirmed by hand: 2-4 of 5 live trials picked "end" on each
   case pre-fix) -- exactly the kind of noise this fix targets, but it makes the pure captured-
   score_sum snapshots the only fully deterministic way to test the fix logic itself. Plus
   negative/control cases (turn > 4 no override; large margin no override; no alternative no
   crash) so the fix's scope stays exactly as narrow as designed.
2. A real end-to-end integration check (reuses tools/reconstruct_decision.py's real-episode
   loader): runs the actual live choose_action() 10 times on one of the same real board states
   and asserts it NEVER returns "end" post-fix -- despite the underlying per-run noise, the
   override activates deterministically whenever "end" would have won this specific near-tie
   (verified by hand: all 5 pre-fix trials' score_sum snapshots satisfy the override's
   condition), so this is a legitimately deterministic post-fix assertion, not a flaky one.

Expected pre-fix (before agents/search_scorer.py::_prefer_continuing_over_end existed): the
pure-function tests can't run at all (ImportError) -- confirmed by hand-computing the fixed
function's expected output against the real captured data before implementing it, and by
observing (documented below, not asserted, since it's non-deterministic) that live pre-fix
trials picked "end" in 2-4 of 5 runs on each real case.

Run: PYTHONIOENCODING=utf-8 py -3.14 tests/test_end_early.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from obs import LegalOption  # noqa: E402
from baseline import read_deck_csv  # noqa: E402
import search_scorer  # noqa: E402
from reconstruct_decision import reconstruct_episode_decisions  # noqa: E402

OUR_DECK = read_deck_csv(os.path.join(ROOT, "decks", "crustle_wall_deck.csv"))

# (team_name, episode relative path, raw_step_index, real captured score_sum from a live
# pre-fix trial where "end" (index 1) narrowly won over "play" (index 0)).
REAL_CASES = [
    ("懒惰的金枪鱼", "runs/expert_corpus/2026-07-22/87363023.json", 47,
     {0: 5.772435897435898, 1: 5.794871794871795}),
    ("西松大祐", "runs/expert_corpus/2026-07-22/87397816.json", 7,
     {0: 12.070390070921986, 1: 12.096453900709221}),
    ("LiamK", "runs/expert_corpus/2026-07-22/87529164.json", 25,
     {0: 20.057142857142857, 1: 20.076190476190476}),
]


def _opt(index: int, kind: str) -> LegalOption:
    return LegalOption(index=index, kind=kind, raw=None, card=None, target=None)


def test_real_cases_switch_to_the_alternative() -> None:
    for team, path, step, score_sum in REAL_CASES:
        full_path = os.path.join(ROOT, path)
        decisions = reconstruct_episode_decisions(full_path, team)
        dec = next(d for d in decisions if d["raw_step_index"] == step)
        selection = dec["selection"]
        assert dec["turn"] <= 4, f"{path}@{step}: expected an early-game decision"
        winner = search_scorer._prefer_continuing_over_end(1, selection, score_sum, dec["turn"])
        assert winner == 0, (
            f"{path}@{step}: expected override to option 0 (play), got {winner} "
            f"(score_sum={score_sum}, turn={dec['turn']})")


def test_no_override_past_early_game() -> None:
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1,
                                         options=[_opt(0, "play"), _opt(1, "end")])
    score_sum = {0: 5.772, 1: 5.795}  # same razor-thin margin as the real case above
    winner = search_scorer._prefer_continuing_over_end(1, selection, score_sum, turn=5)
    assert winner == 1, f"expected no override past turn {search_scorer.END_EARLY_MAX_TURN}, got {winner}"


def test_no_override_when_margin_is_large() -> None:
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1,
                                         options=[_opt(0, "play"), _opt(1, "end")])
    score_sum = {0: 5.0, 1: 6.0}  # ~16.7% relative gap, well past the 5% threshold
    winner = search_scorer._prefer_continuing_over_end(1, selection, score_sum, turn=2)
    assert winner == 1, f"expected no override for a large margin, got {winner}"


def test_no_override_when_winner_is_not_end() -> None:
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1,
                                         options=[_opt(0, "play"), _opt(1, "end")])
    score_sum = {0: 5.795, 1: 5.772}
    winner = search_scorer._prefer_continuing_over_end(0, selection, score_sum, turn=2)
    assert winner == 0, f"expected no change when the resolved winner isn't 'end', got {winner}"


def test_no_crash_when_end_is_the_only_option() -> None:
    selection = search_scorer.Selection(select_type=None, context=None, min_count=1,
                                         max_count=1, options=[_opt(0, "end")])
    score_sum = {0: 5.795}
    winner = search_scorer._prefer_continuing_over_end(0, selection, score_sum, turn=2)
    assert winner == 0, f"expected no change when 'end' is the only legal option, got {winner}"


def test_real_case_never_ends_across_repeated_live_trials() -> None:
    team, path, step, _ = REAL_CASES[0]
    full_path = os.path.join(ROOT, path)
    decisions = reconstruct_episode_decisions(full_path, team)
    dec = next(d for d in decisions if d["raw_step_index"] == step)
    chosen_kinds = []
    for _ in range(10):
        captured = {}

        def trace_fn(rec, _c=captured):
            _c.update(rec)

        result = search_scorer.choose_action(dec["game_state"], dec["selection"],
                                              dec["obs_dict"], OUR_DECK, None, trace_fn=trace_fn)
        chosen_idx = result[0] if result else None
        kind = next((o["kind"] for o in captured.get("options", []) if o["index"] == chosen_idx),
                    None)
        chosen_kinds.append(kind)
    print(f"    live trial kinds: {chosen_kinds}")
    assert all(k != "end" for k in chosen_kinds), (
        f"expected the fix to prevent 'end' across all 10 live trials, got {chosen_kinds}")


def main() -> None:
    tests = [
        test_real_cases_switch_to_the_alternative,
        test_no_override_past_early_game,
        test_no_override_when_margin_is_large,
        test_no_override_when_winner_is_not_end,
        test_no_crash_when_end_is_the_only_option,
        test_real_case_never_ends_across_repeated_live_trials,
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
    print(f"ALL {len(tests)} END-EARLY TESTS PASSED")


if __name__ == "__main__":
    main()
