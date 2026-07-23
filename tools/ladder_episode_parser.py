"""Parses a single Kaggle episode-replay JSON file into tools/loss_review.py's decision-trace
shape, so its analyses run unmodified over real ladder games.

Confirmed against a real downloaded episode (kaggle/pokemon-tcg-ai-battle-episodes-2026-07-21,
episode 87170443) that steps[i][seat]['observation'] is byte-for-byte the same obs_dict shape
src/obs.py::parse_obs() already decodes for the live agent -- it's the SDK's raw observation,
not a Kaggle-specific reshaping -- so this reuses parse_obs() directly instead of re-deriving
option classification. steps[i+1][seat]['action'] is that seat's response to the decision
observed at steps[i] (kaggle-environments' replay convention: an agent's `action` field is
what it submitted to produce that step, so it shows up one step later for the agent's own
record). Episode-level `visualize` at step 0 additionally exposes both players' full decklists
(post-game spectator data, not something the live agent itself ever sees) -- used only for the
ladder-only archetype checks in tools/autopsy.py.

score/features are always None here: those come from agents/search_scorer.py's own trace_fn
instrumentation, and Kaggle never records them -- see tools/autopsy.py's skip_local_only flag.
Recomputing real evaluate() scores retroactively from a replay isn't soundly possible either:
faithfully replaying an episode through the live cg engine to get a matching session would need
the ORIGINAL shuffle/draw RNG seed, which Kaggle never exposes -- a fresh battle_start()
reshuffles differently, so recorded action *indices* would no longer point at the same options.
So ladder-side "why" analysis stays qualitative (board-state fields only), never a fabricated
recomputed score.

Also surfaces terminal-state fields (final deck counts, prizes remaining, whether our side has
any Pokemon left, and the episode's own per-seat status) for tools/ladder_report.py's
win-condition taxonomy -- the real 3 PTCG loss conditions (deck-out, all-prizes-taken,
no-Pokemon-in-play) plus a timeout/illegal-action catch-all from `episode["statuses"]`.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from obs import parse_obs  # noqa: E402
from kaggle_common import OUR_TEAM_NAME  # noqa: E402


class LadderParseError(Exception):
    """Raised when a file that IS one of our extracted episodes doesn't match the expected
    schema -- distinct from parse_episode_file returning None, which means the file genuinely
    isn't one of our games (normal, not an error). Callers should catch this, count it, and
    report it rather than let a format surprise silently vanish."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def find_our_seat(info: dict, our_team_name: str = OUR_TEAM_NAME) -> int | None:
    names = info.get("TeamNames", [])
    for i, name in enumerate(names):
        if name.strip().lower() == our_team_name.strip().lower():
            return i
    return None


def _mon_dict(pv) -> dict | None:
    if pv is None:
        return None
    return {"card_id": pv.card_id, "serial": pv.serial, "hp": pv.hp, "max_hp": pv.max_hp,
            "energies": [int(e) for e in pv.energies]}


def _options_dict(selection) -> list[dict]:
    out = []
    for lo in selection.options:
        out.append({
            "index": lo.index, "kind": lo.kind,
            "card_id": lo.card.card_id if lo.card else None,
            "target_card_id": lo.target.card_id if lo.target else None,
            "target_serial": lo.target.serial if lo.target else None,
            "score": None, "features": None,
        })
    return out


def _opponent_deck_ids(episode: dict, our_seat: int) -> list[int]:
    opp_seat = 1 - our_seat
    try:
        players = episode["steps"][0][0]["visualize"][0]["current"]["players"]
        return [c["id"] for c in players[opp_seat].get("deck", [])]
    except (KeyError, IndexError, TypeError):
        return []


def _final_player_states(episode: dict, our_seat: int) -> tuple[dict | None, dict | None]:
    """Walks backward to the last step whose observation has a non-null `current` (the
    terminal board state) and returns (our_raw_player_dict, opp_raw_player_dict). deckCount
    and prize-array length are public information for both sides per PTCG rules (same
    invariant src/obs.py relies on), so these are genuinely visible, not inferred."""
    opp_seat = 1 - our_seat
    for step in reversed(episode["steps"]):
        obs = step[our_seat].get("observation") or {}
        cur = obs.get("current")
        if cur is not None:
            players = cur.get("players")
            if players and len(players) > max(our_seat, opp_seat):
                return players[our_seat], players[opp_seat]
    return None, None


def _terminal_state(episode: dict, our_seat: int) -> dict:
    our_final, opp_final = _final_player_states(episode, our_seat)
    statuses = episode.get("statuses") or []
    our_status = statuses[our_seat] if our_seat < len(statuses) else None

    result = {
        "our_status": our_status,
        "our_deck_count": None, "opp_deck_count": None,
        "our_prizes_remaining": None, "opp_prizes_remaining": None,
        "our_active_present": None, "our_bench_count": None,
    }
    if our_final is not None:
        result["our_deck_count"] = our_final.get("deckCount")
        result["our_prizes_remaining"] = len(our_final.get("prize") or [])
        result["our_active_present"] = bool(our_final.get("active"))
        result["our_bench_count"] = len(our_final.get("bench") or [])
    if opp_final is not None:
        result["opp_deck_count"] = opp_final.get("deckCount")
        result["opp_prizes_remaining"] = len(opp_final.get("prize") or [])
    return result


def _parse_matched_episode(episode: dict, our_seat: int, path: str) -> dict:
    rewards = episode.get("rewards") or [0, 0]
    our_reward = rewards[our_seat] if our_seat < len(rewards) else 0
    if our_reward > 0:
        outcome = "candidate_win"
    elif our_reward < 0:
        outcome = "opponent_win"
    else:
        outcome = "draw"

    steps = episode["steps"]
    decisions = []
    for i in range(len(steps) - 1):
        obs_dict = steps[i][our_seat]["observation"]
        if obs_dict.get("select") is None or obs_dict.get("current") is None:
            continue
        game_state, selection = parse_obs(obs_dict)
        if game_state is None or selection is None:
            continue
        action = steps[i + 1][our_seat]["action"]
        chosen_index = action[0] if action else None
        decisions.append({
            "turn": game_state.turn, "turn_action_count": game_state.turn_action_count,
            "mode": "ladder",
            "you_active": _mon_dict(game_state.you.active),
            "you_bench": [_mon_dict(m) for m in game_state.you.bench],
            "opp_active": _mon_dict(game_state.opponent.active),
            "options": _options_dict(selection),
            "chosen_index": chosen_index,
        })

    return {
        "game": os.path.splitext(os.path.basename(path))[0],
        "seed": None,
        "outcome": outcome,
        "decisions": decisions,
        "opponent_card_ids": _opponent_deck_ids(episode, our_seat),
        "opponent_team_name": episode.get("info", {}).get("TeamNames", [None, None])[1 - our_seat],
        "terminal": _terminal_state(episode, our_seat),
    }


def parse_episode_file(path: str, our_team_name: str = OUR_TEAM_NAME) -> dict | None:
    """Returns None if this episode genuinely doesn't involve our team (shouldn't happen if
    tools/measure.py filtered correctly before extracting episodes, but checked defensively
    rather than assumed) -- that is the normal, non-error path. Raises LadderParseError for
    anything that doesn't match the expected schema once we ARE looking at one of our matched
    episodes, so a format surprise is visible to the caller instead of silently vanishing."""
    try:
        with open(path, encoding="utf-8") as f:
            episode = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise LadderParseError(path, f"could not read/parse JSON: {e!r}") from e

    try:
        our_seat = find_our_seat(episode.get("info", {}), our_team_name)
    except Exception as e:
        raise LadderParseError(path, f"malformed 'info' block: {e!r}") from e

    if our_seat is None:
        return None

    try:
        return _parse_matched_episode(episode, our_seat, path)
    except LadderParseError:
        raise
    except Exception as e:
        raise LadderParseError(
            path, f"unexpected shape while parsing a matched episode: {e!r}") from e
