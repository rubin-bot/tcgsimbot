"""Reproduces 3 real ladder board states where agents/search_scorer.py's tie-break picks the
wrong attach target among options evaluate() scored as an exact tie -- the mechanism behind
docs/near_tie_measurement_2026-07-23.md's "14/20 real starvation losses are tied-and-lost"
finding. Confirmed 2026-07-23 (see docs/tie_break_v2_2026-07-23.md) that the shipped tie-break
only recognizes an already-evolved, under-3-energy Crustle as "the attacker" -- it never
credits a pre-evolution Dwebble, and among multiple recognized Crustles it can't rank by which
one is closer to being powered, so it falls through to arbitrary engine option-list order in
both cases.

All 3 states are pulled from real downloaded episodes under runs/our_episodes/ (v1's own
ladder games), re-parsed with src/obs.py::parse_obs() exactly as the live agent would have seen
them, then scored via agents/search_scorer.py::choose_action()'s own trace_fn hook -- so the
"tied" scores asserted here are the real, reproducible (deterministic -- attaching energy to
our own Pokemon doesn't depend on the opponent's hidden hand, confirmed identical across 20
resamples during measurement) scores the agent actually computed, not synthetic numbers.

Run: PYTHONIOENCODING=utf-8 py -3.14 tests/test_tie_break.py
Expected pre-fix: all 3 cases FAIL (assert the wrong target is currently chosen).
Expected post-fix: all 3 cases PASS.
"""

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from ladder_episode_parser import find_our_seat  # noqa: E402
from obs import parse_obs  # noqa: E402
from baseline import read_deck_csv  # noqa: E402
import search_scorer  # noqa: E402

import json  # noqa: E402

OUR_DECK = read_deck_csv(os.path.join(ROOT, "decks", "crustle_wall_deck.csv"))

# (episode_id, raw_step_index, expected_target_serial, human description). raw_step_index is
# the exact `steps[i]` index in the downloaded episode JSON (status=="ACTIVE" for our seat,
# confirmed by hand during this cycle's investigation -- see docs/tie_break_v2_2026-07-23.md
# for how these 3 were found among 169 real candidates with multiple distinct pipeline
# targets).
CASES = [
    ("87507277", 51, 64,
     "turn 10.1: active Crustle(70) already at 3 energy (deficit 0) ties with 3 Dwebbles "
     "(64/63/66, all deficit 3) -- correct target is ANY still-building Dwebble, not the "
     "already-full Crustle. Asserts target 64 (first Dwebble in engine option order)."),
    ("87508450", 55, 69,
     "turn 8.3: two Crustles tie -- 67 (deficit 3, just evolved/no energy) and 69 (deficit 1, "
     "already has 2 energy). Correct target is 69 (closer to powered), not 67."),
    ("87510208", 48, 69,
     "turn 6.1: active Crustle 67 (deficit 3) ties with bench Crustle 69 (deficit 1) and bench "
     "Dwebble 63 (deficit 3). Correct target is 69 (closer to powered)."),
]


def find_episode_path(episode_id: str) -> str:
    matches = glob.glob(os.path.join(ROOT, "runs", "our_episodes", "*", f"{episode_id}.json"))
    if not matches:
        raise FileNotFoundError(
            f"{episode_id}.json not found under runs/our_episodes/ -- run "
            f"tools/measure.py first to (re-)download v1's real episodes.")
    return matches[0]


def load_decision(episode_id: str, raw_step_index: int):
    with open(find_episode_path(episode_id), encoding="utf-8") as f:
        episode = json.load(f)
    our_seat = find_our_seat(episode["info"])
    obs_dict = episode["steps"][raw_step_index][our_seat]["observation"]
    assert episode["steps"][raw_step_index][our_seat]["status"] == "ACTIVE", (
        "fixture step must be a genuine decision, not a stale INACTIVE replay step")
    game_state, selection = parse_obs(obs_dict)
    assert game_state is not None and selection is not None
    return obs_dict, game_state, selection


def scored_options(obs_dict, game_state, selection):
    """Real scores from search_scorer's own scoring path (trace_fn), not hardcoded numbers."""
    captured = {}

    def trace_fn(rec, _c=captured):
        _c.update(rec)

    search_scorer.choose_action(game_state, selection, obs_dict, OUR_DECK, None,
                                 trace_fn=trace_fn)
    assert captured.get("mode") == "searched", (
        f"expected the normal search path, got mode={captured.get('mode')!r}")
    return captured["options"]


def test_case(episode_id: str, raw_step_index: int, expected_target_serial: int,
              description: str) -> None:
    obs_dict, game_state, selection = load_decision(episode_id, raw_step_index)
    trace_options = scored_options(obs_dict, game_state, selection)

    attach_options = [o for o in trace_options if o["kind"] in ("attach", "energy")
                       and o["score"] is not None]
    assert attach_options, f"{episode_id}@{raw_step_index}: no scored attach options found"
    best_score = max(o["score"] for o in attach_options)
    eps = search_scorer._TIE_EPS_REL * max(abs(best_score), 1.0)
    tied_serials = {o["target_serial"] for o in attach_options
                     if abs(o["score"] - best_score) <= eps}
    assert len(tied_serials) >= 2, (
        f"{episode_id}@{raw_step_index}: expected a real multi-target tie, got "
        f"{tied_serials} -- fixture may be stale (re-verify against a fresh episode)")

    tied_options = [lo for lo in selection.options
                     if lo.kind in ("attach", "energy") and lo.target is not None
                     and lo.target.serial in tied_serials]
    scores = {lo.index: best_score for lo in tied_options}  # all tied by construction
    chosen = search_scorer._break_ties(selection, scores, best_score)

    print(f"[{episode_id}@{raw_step_index}] {description}")
    print(f"    tied targets (serial): {sorted(tied_serials)}")
    print(f"    _break_ties chose target serial {chosen.target.serial if chosen else None}, "
          f"expected {expected_target_serial}")
    assert chosen is not None and chosen.target is not None
    assert chosen.target.serial == expected_target_serial, (
        f"{episode_id}@{raw_step_index}: tie-break picked target serial "
        f"{chosen.target.serial} (card {chosen.target.card_id}, "
        f"energies={chosen.target.energies}), expected serial {expected_target_serial}")


def main() -> None:
    failures = []
    for episode_id, raw_step_index, expected_serial, description in CASES:
        try:
            test_case(episode_id, raw_step_index, expected_serial, description)
            print("  PASS\n")
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            failures.append((episode_id, raw_step_index))
    if failures:
        raise SystemExit(f"{len(failures)}/{len(CASES)} tie-break case(s) FAILED: {failures}")
    print(f"ALL {len(CASES)} TIE-BREAK TESTS PASSED")


if __name__ == "__main__":
    main()
