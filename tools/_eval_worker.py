"""Plays exactly one game between two named agents, then exits.

Spawned by tools/eval_arena.py as its own OS subprocess -- a native engine crash or a runaway
memory allocation here can't take down the parent or any other game. Communicates the result
back to the parent purely via stdout lines (never a shared file the parent has to lock):

  TURN <seat>          -- printed right before agents[seat] is asked to move. Lets the parent
                          attribute a hang/native crash to a side even if this process never
                          gets to print its own RESULT line.
  RESULT {json}         -- the final line on a clean exit (including self-caught exceptions).

Run standalone for debugging, e.g.:
  PYTHONPATH=src .venv/Scripts/python tools/_eval_worker.py --candidate search_scorer \
      --opponent random --candidate-deck decks/crustle_wall_deck.csv \
      --opponent-deck decks/crustle_wall_deck.csv --candidate-seat 0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

from obs import parse_obs  # noqa: E402

AGENT_NAMES = ("search_scorer", "baseline", "random")


def _load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def _make_random_agent(deck_list: list[int]):
    def _agent(obs_dict: dict) -> list[int]:
        _, sel = parse_obs(obs_dict)
        if sel is None:
            return deck_list
        return random.sample(range(len(sel.options)), sel.max_count)
    return _agent


def _build_agent(name: str, deck_list: list[int], opp_deck_list: list[int], trace_fn=None,
                  weights_path: str | None = None, snapshot_path: str | None = None,
                  snapshot_module_name: str = "search_scorer_snapshot"):
    if name == "random":
        return _make_random_agent(deck_list)
    if name == "baseline":
        from baseline import agent as baseline_agent
        return baseline_agent
    if name == "search_scorer":
        if snapshot_path:
            # Load a frozen code snapshot (e.g. runs/v2_tie_break/search_scorer_v1_snapshot.py)
            # instead of the live agents/search_scorer.py, so this process can pit two
            # DIFFERENT code versions against each other -- --candidate-weights/
            # --opponent-weights alone only vary numeric weights on identical code. Mirrors
            # tools/decision_diff.py's load_v1() merge: snapshot's own module-default WEIGHTS,
            # optionally overridden by a separate weights file on top.
            import importlib.util
            spec = importlib.util.spec_from_file_location(snapshot_module_name, snapshot_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            weights = dict(mod.WEIGHTS)
            if weights_path:
                with open(weights_path, encoding="utf-8") as f:
                    weights.update(json.load(f))
            return mod.make_agent(deck_list, trace_fn=trace_fn, weights=weights)
        from search_scorer import make_agent, load_weights
        weights = load_weights(weights_path) if weights_path else None
        return make_agent(deck_list, trace_fn=trace_fn, weights=weights)
    raise ValueError(f"unknown agent name: {name!r} (expected one of {AGENT_NAMES})")


def _emit_result(payload: dict) -> None:
    print(f"RESULT {json.dumps(payload)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=AGENT_NAMES)
    ap.add_argument("--opponent", required=True, choices=AGENT_NAMES)
    ap.add_argument("--candidate-deck", required=True)
    ap.add_argument("--opponent-deck", required=True)
    ap.add_argument("--candidate-seat", type=int, required=True, choices=(0, 1))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--replay-out", default=None,
                     help="if set and --candidate search_scorer, append one JSONL line with "
                          "this game's full decision trace (turn, options, evaluate() scores, "
                          "chosen index) -- see agents/search_scorer.py's trace_fn hook.")
    ap.add_argument("--game-index", type=int, default=None,
                     help="parent's global game index, carried into the replay record for "
                          "correlation; purely informational.")
    ap.add_argument("--candidate-weights", default=None,
                     help="JSON file of a WEIGHTS-shaped dict; only used when --candidate "
                          "search_scorer. Omit to use agents/search_scorer.py's module default.")
    ap.add_argument("--opponent-weights", default=None,
                     help="Same as --candidate-weights but for --opponent search_scorer -- "
                          "lets one process pit two DIFFERENTLY-weighted search_scorer "
                          "instances against each other (tools/tune_weights.py's self-relative "
                          "matchup).")
    ap.add_argument("--candidate-snapshot", default=None,
                     help="path to a frozen search_scorer.py code snapshot (e.g. "
                          "runs/v2_tie_break/search_scorer_v1_snapshot.py) to use for "
                          "--candidate search_scorer INSTEAD OF the live agents/search_scorer.py "
                          "-- lets this process pit two different CODE versions against each "
                          "other, not just different weights. --candidate-weights, if also "
                          "given, overrides on top of the snapshot's own module-default WEIGHTS.")
    ap.add_argument("--opponent-snapshot", default=None,
                     help="same as --candidate-snapshot but for --opponent search_scorer.")
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    candidate_deck = _load_deck(args.candidate_deck)
    opponent_deck = _load_deck(args.opponent_deck)

    track_fallbacks = "search_scorer" in (args.candidate, args.opponent)
    if track_fallbacks:
        from search_scorer import reset_fallback_counts
        reset_fallback_counts()

    decisions: list = []
    log_replay = args.replay_out is not None and args.candidate == "search_scorer"
    trace_fn = decisions.append if log_replay else None

    candidate_agent = _build_agent(args.candidate, candidate_deck, opponent_deck,
                                    trace_fn=trace_fn, weights_path=args.candidate_weights,
                                    snapshot_path=args.candidate_snapshot,
                                    snapshot_module_name="search_scorer_snapshot_candidate")
    opponent_agent = _build_agent(args.opponent, opponent_deck, candidate_deck,
                                   weights_path=args.opponent_weights,
                                   snapshot_path=args.opponent_snapshot,
                                   snapshot_module_name="search_scorer_snapshot_opponent")

    cseat = args.candidate_seat
    oseat = 1 - cseat
    agents = {cseat: candidate_agent, oseat: opponent_agent}
    decks = {cseat: candidate_deck, oseat: opponent_deck}

    def side_for_seat(seat: int) -> str:
        return "candidate" if seat == cseat else "opponent"

    def finish_payload(payload: dict) -> dict:
        if track_fallbacks:
            from search_scorer import get_fallback_counts
            payload["fallback_counts"] = get_fallback_counts()
        if log_replay:
            with open(args.replay_out, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "game": args.game_index, "seed": args.seed, "candidate_seat": cseat,
                    "outcome": payload["outcome"], "decisions": decisions,
                }) + "\n")
        return payload

    obs, start_data = battle_start(decks[0], decks[1])
    if obs is None:
        _emit_result(finish_payload({
            "outcome": "crash", "crashing_side": None,
            "reason": f"battle_start_failed errorPlayer={start_data.errorPlayer} "
                      f"errorType={start_data.errorType}",
        }))
        return

    STEP_CAP = 6000
    for _ in range(STEP_CAP):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            battle_finish()
            r = st["result"]
            if r == 2:
                outcome = "draw"
            else:
                outcome = "candidate_win" if r == cseat else "opponent_win"
            _emit_result(finish_payload({"outcome": outcome}))
            return

        yi = st["yourIndex"]
        print(f"TURN {yi}", flush=True)

        try:
            index_list = agents[yi](obs)
        except Exception as e:  # the agent's own code raised, uncaught
            _emit_result(finish_payload({
                "outcome": "crash", "crashing_side": side_for_seat(yi),
                "reason": f"agent_exception: {e!r}",
            }))
            try:
                battle_finish()
            except Exception:
                pass
            return

        try:
            obs = battle_select(index_list)
        except Exception as e:  # the agent returned an illegal selection
            _emit_result(finish_payload({
                "outcome": "crash", "crashing_side": side_for_seat(yi),
                "reason": f"illegal_select {index_list!r}: {e!r}",
            }))
            try:
                battle_finish()
            except Exception:
                pass
            return

    try:
        battle_finish()
    except Exception:
        pass
    _emit_result(finish_payload({
        "outcome": "crash", "crashing_side": None, "reason": f"step_cap_{STEP_CAP}_exceeded",
    }))


if __name__ == "__main__":
    main()
