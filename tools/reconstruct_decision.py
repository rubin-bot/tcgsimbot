"""Master-study Workstream B0/B1: reconstructs real per-decision game state for an ARBITRARY
team's episode (not just ours), ready to feed into agents/search_scorer.py::choose_action.

Turned out to need very little new code: tools/ladder_episode_parser.py::find_our_seat and
::parse_episode_file already take an overridable `our_team_name` parameter (default
OUR_TEAM_NAME) -- the "our-team" naming is just the default, not a structural limitation. This
module is a thin wrapper that (1) resolves the ACTING team's seat via that same function, (2)
reuses tools/measure_near_tie_hypothesis.py::active_decision_steps (already seat-generic, no
name hardcoding) to find every genuine ACTIVE decision's raw step index, and (3) calls
src/obs.py::parse_obs on exactly that seat's own observation -- never a combined/God's-eye
view, same invariant tests/test_obs.py already asserts.

Usage (library, not a CLI): see tools/decision_diff.py and the B0 self-test in this file's
own __main__ block:
  .venv/Scripts/python tools/reconstruct_decision.py --self-test
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from ladder_episode_parser import find_our_seat  # noqa: E402
from measure_near_tie_hypothesis import active_decision_steps, V1_SUBMISSION_ID  # noqa: E402
from obs import parse_obs  # noqa: E402


class ReconstructionError(Exception):
    """A decision that should be reconstructable wasn't -- distinct from "this episode doesn't
    involve the requested team" (that's find_our_seat returning None, handled by the caller)."""


def reconstruct_episode_decisions(episode_path: str, acting_team_name: str) -> list[dict]:
    """Returns one dict per real ACTIVE decision for `acting_team_name`'s seat in this episode:
    {raw_step_index, obs_dict, game_state, selection, historical_chosen_index, turn,
    turn_action_count}. Empty list if the team isn't in this episode at all (not an error)."""
    with open(episode_path, encoding="utf-8") as f:
        raw = json.load(f)

    seat = find_our_seat(raw.get("info", {}), acting_team_name)
    if seat is None:
        return []

    steps = raw["steps"]
    out = []
    for step_idx in active_decision_steps(raw, seat):
        obs_dict = steps[step_idx][seat]["observation"]
        game_state, selection = parse_obs(obs_dict)
        if game_state is None or selection is None:
            continue
        action = steps[step_idx + 1][seat]["action"]
        historical_chosen_index = action[0] if action else None
        out.append({
            "raw_step_index": step_idx,
            "obs_dict": obs_dict,
            "game_state": game_state,
            "selection": selection,
            "historical_chosen_index": historical_chosen_index,
            "turn": game_state.turn,
            "turn_action_count": game_state.turn_action_count,
        })
    return out


# ---------------------------------------------------------------------------
# B0: reconstruction self-test -- run against OUR OWN real v1 games (known ground truth: the
# frozen v1 snapshot literally WAS the code that made these historical choices) before trusting
# this module on anyone else's games.
# ---------------------------------------------------------------------------

N_RESAMPLE = 20  # matches N_REPLAYS in tools/measure_near_tie_hypothesis.py, for a directly
                  # comparable reachability number against that cycle's own published
                  # self-consistency/flip-rate figures. For genuine (non-near-tie) disagreements:
                  # how many extra fresh replays to
                  # check whether the historical choice is REACHABLE at all under this
                  # reconstruction (v1 has no seed control -- sample_determinization() draws a
                  # fresh random hidden world every call, so single-sample variance alone can
                  # make one fresh replay disagree with history even when the reconstruction
                  # itself is fully correct; see docs/near_tie_measurement_2026-07-23.md /
                  # docs/v3_report_2026-07-23.md for the same phenomenon measured independently).


def run_self_test(episodes_root: str, deck_path: str, v1_snapshot_path: str,
                   our_team_name: str, v1_submission_id: str,
                   tuned_weights_path: str | None = None) -> dict:
    import glob
    import importlib.util

    from baseline import read_deck_csv
    from kaggle_common import load_episode_submissions

    deck = read_deck_csv(deck_path)

    spec = importlib.util.spec_from_file_location("search_scorer_v1_selftest", v1_snapshot_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)

    # CRITICAL: the real shipped v1 did NOT use agents/search_scorer.py's module-default
    # WEIGHTS -- scripts/build_submission.py --weights bakes a CMA-ES-tuned override in via
    # `WEIGHTS.update(...)` appended to the packaged source at package time (VERSIONS.md's v1
    # entry: "NOT the module's hand-set WEIGHTS default"). Reconstructing with the wrong
    # weights produces systematic, non-random divergence that (correctly) never resolves via
    # resampling -- this isn't noise, it's a different value function. Replicate the exact
    # packaging mechanism: start from the module's own default WEIGHTS, then .update() with the
    # tuned override (matching build_submission.py:114-115 exactly), rather than passing the
    # tuned dict alone (which is missing keys added to evaluate() after that tuning run, e.g.
    # turns_to_power/wasted_energy, and would leave them unweighted instead of at their real
    # shipped default).
    weights = None
    if tuned_weights_path:
        with open(tuned_weights_path, encoding="utf-8") as f:
            override = json.load(f)
        weights = dict(v1.WEIGHTS)
        weights.update(override)

    # Critical filter: runs/our_episodes/ blends TWO different submissions (v1's search_scorer
    # AND the older, architecturally-different deprecated net_checkpoint agent -- episode
    # parsing matches on team name only, not submission id, per CLAUDE.md/docs/submission_
    # ladder_audit_2026-07-23.md). Comparing "our v1 reconstruction" against a game the OTHER
    # agent actually played would show massive, genuine-looking disagreement that has nothing
    # to do with reconstruction correctness -- must filter to v1-attributed episodes only.
    submissions = load_episode_submissions()
    v1_episode_ids = {eid for eid, sid in submissions.items() if sid == v1_submission_id}
    all_paths = sorted(glob.glob(os.path.join(episodes_root, "**", "*.json"), recursive=True))
    episode_paths = [p for p in all_paths
                      if os.path.splitext(os.path.basename(p))[0] in v1_episode_ids]
    print(f"{len(all_paths)} episode files found under {episodes_root}, "
          f"{len(episode_paths)} attributed to v1 (submission {v1_submission_id})")
    n_agree = 0
    n_disagree = 0
    n_disagree_near_tie = 0
    n_disagree_genuine = 0
    n_genuine_reachable_via_resample = 0
    disagreements = []
    n_decisions = 0
    n_episodes_checked = 0

    for path in episode_paths:
        decisions = reconstruct_episode_decisions(path, our_team_name)
        if not decisions:
            continue
        n_episodes_checked += 1
        for dec in decisions:
            if dec["historical_chosen_index"] is None:
                continue
            n_decisions += 1
            captured = {}

            def trace_fn(rec, _c=captured):
                _c.update(rec)

            result = v1.choose_action(dec["game_state"], dec["selection"], dec["obs_dict"],
                                       deck, None, trace_fn=trace_fn, weights=weights)
            our_choice = result[0] if result else None
            if our_choice == dec["historical_chosen_index"]:
                n_agree += 1
                continue
            n_disagree += 1
            # scores live per-option inside "options", not as a separate top-level dict
            # (agents/search_scorer.py::_trace_options) -- same shape used throughout this repo.
            scores = {o["index"]: o["score"] for o in (captured.get("options") or [])
                      if o.get("score") is not None}
            hist_score = scores.get(dec["historical_chosen_index"])
            our_score = scores.get(our_choice)
            is_near_tie = False
            if hist_score is not None and our_score is not None:
                eps = v1._TIE_EPS_REL * max(abs(our_score), 1.0)
                is_near_tie = abs(our_score - hist_score) <= eps
            if is_near_tie:
                n_disagree_near_tie += 1
                reachable_via_resample = None  # not checked -- already explained
            else:
                n_disagree_genuine += 1
                # Diagnostic: is the historical choice reachable at all under this
                # reconstruction via ordinary single-sample determinization variance, or does it
                # never show up even across repeated fresh resamples? The latter is the real
                # signal of a reconstruction problem; the former is the already-documented
                # single-sample noise, not a bug.
                reachable_via_resample = False
                for _ in range(N_RESAMPLE):
                    resample_captured = {}

                    def resample_trace_fn(rec, _c=resample_captured):
                        _c.update(rec)

                    resample_result = v1.choose_action(
                        dec["game_state"], dec["selection"], dec["obs_dict"], deck, None,
                        trace_fn=resample_trace_fn, weights=weights)
                    resample_choice = resample_result[0] if resample_result else None
                    if resample_choice == dec["historical_chosen_index"]:
                        reachable_via_resample = True
                        break
                if reachable_via_resample:
                    n_genuine_reachable_via_resample += 1
            if len(disagreements) < 20:
                disagreements.append({
                    "episode": os.path.basename(path), "step": dec["raw_step_index"],
                    "turn": dec["turn"], "historical_choice": dec["historical_chosen_index"],
                    "our_choice": our_choice, "hist_score": hist_score, "our_score": our_score,
                    "near_tie": is_near_tie, "reachable_via_resample": reachable_via_resample,
                })

    agreement_rate = n_agree / n_decisions if n_decisions else 0.0
    # "Adjusted" agreement: near-tie disagreements + genuine disagreements that turned out to be
    # reachable via ordinary resampling are BOTH explained by single-sample determinization
    # variance, not reconstruction error -- this is the metric that actually answers "is the
    # reconstructor correct," separate from "does v1 behave deterministically" (it doesn't, by
    # design/lack of a seed -- a separate, already-documented fact).
    n_variance_explained = n_disagree_near_tie + n_genuine_reachable_via_resample
    adjusted_agreement_rate = (n_agree + n_variance_explained) / n_decisions if n_decisions \
        else 0.0
    return {
        "n_episodes_checked": n_episodes_checked, "n_decisions": n_decisions,
        "n_agree": n_agree, "n_disagree": n_disagree, "agreement_rate": agreement_rate,
        "n_disagree_near_tie": n_disagree_near_tie, "n_disagree_genuine": n_disagree_genuine,
        "n_genuine_reachable_via_resample": n_genuine_reachable_via_resample,
        "n_genuine_unreachable": n_disagree_genuine - n_genuine_reachable_via_resample,
        "adjusted_agreement_rate": adjusted_agreement_rate,
        "sample_disagreements": disagreements,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--snapshot",
                     default=os.path.join(ROOT, "runs", "v2_tie_break",
                                           "search_scorer_v1_snapshot.py"),
                     help="search_scorer code snapshot to self-test (default: frozen v1) -- "
                          "e.g. runs/v4_candidates/search_scorer_v4_snapshot.py to check a v4 "
                          "candidate for regression against the same real v1 ground truth.")
    ap.add_argument("--weights",
                     default=os.path.join(ROOT, "runs", "tune_run1", "winner_weights.json"),
                     help="tuned-weights JSON merged onto the snapshot's own module-default "
                          "WEIGHTS (default: the real shipped v1 tuning).")
    args = ap.parse_args()
    if args.self_test:
        from kaggle_common import OUR_TEAM_NAME
        result = run_self_test(
            os.path.join(ROOT, "runs", "our_episodes"),
            os.path.join(ROOT, "decks", "crustle_wall_deck.csv"),
            args.snapshot,
            OUR_TEAM_NAME,
            V1_SUBMISSION_ID,
            args.weights,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "sample_disagreements"},
                          indent=2))
        print(f"\nRAW AGREEMENT RATE: {result['agreement_rate']:.1%} "
              f"({result['n_agree']}/{result['n_decisions']})")
        print(f"disagreements: {result['n_disagree']} total, "
              f"{result['n_disagree_near_tie']} near-tie (expected noise), "
              f"{result['n_disagree_genuine']} genuine (non-near-tie)")
        print(f"  of the genuine disagreements: {result['n_genuine_reachable_via_resample']} "
              f"reachable via resampling (single-sample variance, not a bug), "
              f"{result['n_genuine_unreachable']} NEVER reachable across "
              f"{N_RESAMPLE} resamples (real signal)")
        print(f"\nADJUSTED AGREEMENT RATE (raw + near-tie + resample-reachable, i.e. "
              f"'is the reconstructor correct once single-sample variance is accounted for'): "
              f"{result['adjusted_agreement_rate']:.1%}")
        if result["sample_disagreements"]:
            print("\nsample disagreements:")
            for d in result["sample_disagreements"][:10]:
                print(f"  {d}")
