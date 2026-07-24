"""v4 Change 2: agents/search_scorer.py::_tie_break_key's new play/card-kind card-priority
discriminator (_PLAY_CARD_TIE_PRIORITY) -- generalizes v2's target-aware approach to play/card
ties, scoped ONLY to cards with strong real evidence (docs/master_study_2026-07-24.md's
tie-break research). v2's attach-kind tie-break (tests/test_tie_break.py) is untouched by this
file and this change; both are pure additions to the same key function.

Real repro cases (same convention as tests/test_tie_break.py: reconstruct the exact real board
state via tools/reconstruct_decision.py, call the tie-break directly on the tied subset -- not
the full live choose_action(), since a live run's full option list can include a DIFFERENT kind
(e.g. "retreat") that outranks "play" on the EXISTING base kind-priority, which would confound a
test of the NEW card-priority discriminator specifically; filtering to just the play-kind tied
options isolates it cleanly, exactly like test_tie_break.py already does for attach-kind ties):

1. episode 87363032 step 33: real tie among Switch(1123)/Mega Kangaskhan ex(756)/Pokegear 3.0
   (1122), all "play"-kind, engine list order Switch first. Expert historically chose Pokegear
   3.0. Pre-fix (arbitrary list order): Switch. Post-fix: Pokegear 3.0 (priority 0, lowest).
2. episode 87363039 step 135: real tie among Mega Kangaskhan ex(756)/Jumbo Ice Cream(1147)/
   Switch(1123), all "play"-kind (the live trace also included a tied "retreat" option --
   excluded here since it's outside the scope of this change and would confound the test, see
   above). Expert historically chose Jumbo Ice Cream. Pre-fix: Mega Kangaskhan ex (list order).
   Post-fix: Jumbo Ice Cream (priority 0).

Expected pre-fix (before _PLAY_CARD_TIE_PRIORITY existed): both cases resolve to the FIRST
option in engine list order (confirmed by hand below), which does NOT match the real expert
choice in either case.

Run: PYTHONIOENCODING=utf-8 py -3.14 tests/test_play_card_tiebreak.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

import search_scorer  # noqa: E402
from reconstruct_decision import reconstruct_episode_decisions  # noqa: E402

# (team_name, episode relative path, raw_step_index, expected winning card_id, description)
CASES = [
    ("懒惰的金枪鱼", "runs/expert_corpus/2026-07-22/87363032.json", 33, 1122,
     "turn 4: real tie among Switch(1123)/Mega Kangaskhan ex(756)/Pokegear 3.0(1122), all "
     "play-kind. Expert historically chose Pokegear 3.0 -- the highest-tier card by real "
     "expert-preference rate (81.7%, n=218). Engine list order alone would pick Switch."),
    ("Budew", "runs/expert_corpus/2026-07-22/87363039.json", 135, 1147,
     "turn 16: real tie among Mega Kangaskhan ex(756)/Jumbo Ice Cream(1147)/Switch(1123), all "
     "play-kind (a tied 'retreat' option in the live trace is excluded here, out of scope for "
     "this change). Expert historically chose Jumbo Ice Cream -- highest-tier (92.9%, n=85). "
     "Engine list order alone would pick Mega Kangaskhan ex."),
]


def test_case(team: str, path: str, raw_step_index: int, expected_card_id: int,
              description: str) -> None:
    full_path = os.path.join(ROOT, path)
    decisions = reconstruct_episode_decisions(full_path, team)
    dec = next(d for d in decisions if d["raw_step_index"] == raw_step_index)
    selection = dec["selection"]

    tied_options = [lo for lo in selection.options if lo.kind == "play"]
    assert len(tied_options) >= 2, (
        f"{path}@{raw_step_index}: expected a real multi-option play-kind tie, got "
        f"{len(tied_options)} -- fixture may be stale")
    card_ids = sorted(lo.card.card_id for lo in tied_options if lo.card)

    chosen = min(tied_options, key=search_scorer._tie_break_key)

    print(f"[{path}@{raw_step_index}] {description}")
    print(f"    tied cards: {card_ids}")
    print(f"    _tie_break_key chose card {chosen.card.card_id if chosen.card else None}, "
          f"expected {expected_card_id}")
    assert chosen.card is not None and chosen.card.card_id == expected_card_id, (
        f"{path}@{raw_step_index}: tie-break picked card {chosen.card.card_id if chosen.card else None}, "
        f"expected {expected_card_id}")


def main() -> None:
    failures = []
    for team, path, step, expected_card_id, description in CASES:
        try:
            test_case(team, path, step, expected_card_id, description)
            print("  PASS\n")
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            failures.append((path, step))
    if failures:
        raise SystemExit(f"{len(failures)}/{len(CASES)} case(s) FAILED: {failures}")
    print(f"ALL {len(CASES)} PLAY/CARD TIE-BREAK TESTS PASSED")


if __name__ == "__main__":
    main()
