"""Analysis pass over runs/near_tie_measurement/replays.jsonl (written by
tools/measure_near_tie_hypothesis.py) -- computes everything
docs/near_tie_measurement_2026-07-23.md reports: flip rate + margin distribution (item 1),
outcome linkage (item 2), and the real-data tied-and-lost check (item 3). Read-only over
already-measured data + the raw episode JSON (for the outcome-linkage forward walk); does not
call choose_action() again, so it's fast and safe to re-run freely while iterating on the
report.

Usage:
  py -3.14 tools/analyze_near_tie_results.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from measure_near_tie_hypothesis import (  # noqa: E402
    load_v1_games, find_decline_keys, find_starvation_keys, active_decision_steps,
    REPLAYS_PATH,
)
from obs import parse_obs  # noqa: E402
import search_scorer  # noqa: E402
import loss_review  # noqa: E402

NEAR_TIE_REL_THRESHOLD = loss_review.NEAR_TIE_REL_THRESHOLD  # 0.05, reused for consistency
FORWARD_LOOK = 3  # turns-ahead window for opportunity/tempo tracking


def load_replays() -> dict[tuple[str, int], dict]:
    out = {}
    with open(REPLAYS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[(rec["game"], rec["decision_index"])] = rec
    return out


# ---------------------------------------------------------------------------
# Item 1: flip rate + margin distribution
# ---------------------------------------------------------------------------

def analyze_flip_and_margin(decline_keys: set, replays: dict) -> dict:
    pooled_margins = []
    pooled_rel_margins = []
    per_decision = []  # dicts: key, mean_margin, mean_rel_margin, attack_count/20
    for key in sorted(decline_keys):
        rec = replays[key]
        kind_by_index = {o["index"]: o["kind"] for o in rec["options"]}
        attack_indices = [idx for idx, k in kind_by_index.items() if k == "attack"]
        attack_count = 0
        margins = []
        rel_margins = []
        for rp in rec["replays"]:
            chosen_idx = rp["chosen_index"]
            if chosen_idx is not None and kind_by_index.get(chosen_idx) == "attack":
                attack_count += 1
            scores = rp.get("scores")
            if rp["mode"] != "searched" or not scores:
                continue
            best_attack = max((scores.get(str(i)) for i in attack_indices
                                if str(i) in scores), default=None)
            ladder_score = scores.get(str(rec["ladder_chosen_index"]))
            if best_attack is None or ladder_score is None:
                continue
            margin = best_attack - ladder_score
            # relative to the magnitude of the two scores being compared (not the margin
            # itself) -- the same normalization style search_scorer.py's own tie-break epsilon
            # uses, just with the more forgiving NEAR_TIE_REL_THRESHOLD (0.05) for reporting
            # "is this a near-tie" rather than the ultra-strict 1e-6 used for the exact
            # tied-and-lost classification in item 3.
            scale = max(abs(best_attack), abs(ladder_score), 1.0)
            rel_margin = margin / scale
            margins.append(margin)
            rel_margins.append(rel_margin)
            pooled_margins.append(margin)
            pooled_rel_margins.append(rel_margin)
        if margins:
            mean_margin = statistics.mean(margins)
            mean_rel_margin = statistics.mean(rel_margins)
        else:
            mean_margin = None
            mean_rel_margin = None
        per_decision.append({
            "key": key, "mean_margin": mean_margin, "mean_rel_margin": mean_rel_margin,
            "attack_count_of_20": attack_count,
        })

    n = len(decline_keys)
    flip_at_least_once = sum(1 for d in per_decision if d["attack_count_of_20"] >= 1)
    flip_majority = sum(1 for d in per_decision if d["attack_count_of_20"] >= 11)

    near_tie = confident_decline = declining_better = unscored = 0
    for d in per_decision:
        rm = d["mean_rel_margin"]
        if rm is None:
            unscored += 1
        elif abs(rm) <= NEAR_TIE_REL_THRESHOLD:
            near_tie += 1
        elif rm > 0:
            confident_decline += 1
        else:
            declining_better += 1

    return {
        "n_decisions": n, "flip_at_least_once": flip_at_least_once,
        "flip_majority": flip_majority, "pooled_margins": pooled_margins,
        "pooled_rel_margins": pooled_rel_margins, "per_decision": per_decision,
        "near_tie": near_tie, "confident_decline": confident_decline,
        "declining_better": declining_better, "unscored": unscored,
    }


# ---------------------------------------------------------------------------
# Item 2: outcome linkage
# ---------------------------------------------------------------------------

def lethal_available(game_state, selection) -> bool:
    opp = game_state.opponent.active
    if opp is None:
        return False
    for lo in selection.options:
        if lo.kind != "attack" or lo.raw.attackId is None:
            continue
        atk = search_scorer._ATTACK.get(lo.raw.attackId)
        if atk is not None and atk.damage >= opp.hp:
            return True
    return False


def analyze_outcome_linkage(games: list[dict], decline_keys: set) -> dict:
    by_game_decline = {}  # episode_id -> {"declines": n, "attack_legal": n, "outcome": str}
    for g in games:
        eid = g["episode_id"]
        outcome = g["parsed"]["outcome"]
        attack_legal = 0
        declines = 0
        for idx, dec in enumerate(g["parsed"]["decisions"]):
            kinds = [o["kind"] for o in dec["options"]]
            if "attack" not in kinds:
                continue
            attack_legal += 1
            if (eid, idx) in decline_keys:
                declines += 1
        by_game_decline[eid] = {"declines": declines, "attack_legal": attack_legal,
                                 "outcome": outcome}

    wins = [v for v in by_game_decline.values() if v["outcome"] == "candidate_win"]
    losses = [v for v in by_game_decline.values() if v["outcome"] == "opponent_win"]

    def rate(rows):
        legal = sum(r["attack_legal"] for r in rows)
        declined = sum(r["declines"] for r in rows)
        return declined, legal, (declined / legal if legal else None)

    win_declined, win_legal, win_rate = rate(wins)
    loss_declined, loss_legal, loss_rate = rate(losses)

    # 2b/2c: per-decision opportunity/tempo/lethal analysis
    games_by_id = {g["episode_id"]: g for g in games}
    buckets = Counter()
    opportunity_lost = opportunity_preserved = 0
    tempo_deltas = []
    examples = {"lethal_declined": [], "plausibly_correct_setup": [], "other": []}

    for eid, idx in sorted(decline_keys):
        game = games_by_id[eid]
        our_seat = game["our_seat"]
        steps = active_decision_steps(game["raw"], our_seat)
        step_idx = steps[idx]
        obs_dict = game["raw"]["steps"][step_idx][our_seat]["observation"]
        game_state, selection = parse_obs(obs_dict)
        dec = game["parsed"]["decisions"][idx]
        chosen = next((o for o in dec["options"] if o["index"] == dec["chosen_index"]), None)
        chosen_kind = chosen["kind"] if chosen else None

        lethal = lethal_available(game_state, selection)
        if lethal:
            bucket = "lethal_declined"
        elif chosen_kind in ("attach", "evolve"):
            bucket = "plausibly_correct_setup"
        else:
            bucket = "other"
        buckets[bucket] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append({
                "game": eid, "decision_index": idx, "turn": dec["turn"],
                "chosen_kind": chosen_kind,
                "opp_active_hp": game_state.opponent.active.hp if game_state.opponent.active
                else None,
            })

        # opportunity/tempo: look at our next up-to-FORWARD_LOOK real decisions
        opp_seat = 1 - our_seat
        target_serial = game_state.opponent.active.serial if game_state.opponent.active \
            else None
        target_hp = game_state.opponent.active.hp if game_state.opponent.active else None
        our_prize_now = len(obs_dict["current"]["players"][our_seat]["prize"])
        opp_prize_now = len(obs_dict["current"]["players"][opp_seat]["prize"])

        future_idx = idx + 1
        preserved = None
        future_prize_delta = None
        for _ in range(FORWARD_LOOK):
            if future_idx >= len(steps):
                break
            f_step = steps[future_idx]
            f_obs = game["raw"]["steps"][f_step][our_seat]["observation"]
            f_cur = f_obs.get("current")
            if f_cur is None:
                future_idx += 1
                continue
            f_opp_active = f_cur["players"][opp_seat].get("active") or [None]
            f_opp = f_opp_active[0]
            if preserved is None:
                if f_opp is not None and f_opp.get("serial") == target_serial \
                        and f_opp.get("hp", 0) <= (target_hp or 0):
                    preserved = True
                else:
                    preserved = False
            our_prize_then = len(f_cur["players"][our_seat]["prize"])
            opp_prize_then = len(f_cur["players"][opp_seat]["prize"])
            future_prize_delta = (our_prize_now - our_prize_then) - \
                (opp_prize_now - opp_prize_then)
            future_idx += 1

        if preserved is True:
            opportunity_preserved += 1
        elif preserved is False:
            opportunity_lost += 1
        if future_prize_delta is not None:
            tempo_deltas.append(future_prize_delta)

    return {
        "win_declined": win_declined, "win_legal": win_legal, "win_rate": win_rate,
        "loss_declined": loss_declined, "loss_legal": loss_legal, "loss_rate": loss_rate,
        "n_wins": len(wins), "n_losses": len(losses),
        "buckets": dict(buckets), "examples": examples,
        "opportunity_lost": opportunity_lost, "opportunity_preserved": opportunity_preserved,
        "tempo_deltas": tempo_deltas,
    }


# ---------------------------------------------------------------------------
# Item 3: ladder tied-and-lost check, reusing loss_review.py's own classification
# ---------------------------------------------------------------------------

def analyze_starvation_tied_and_lost(games: list[dict], starvation_keys: set,
                                      replays: dict) -> dict:
    games_by_id = {g["episode_id"]: g for g in games}
    # Rebuild game-shaped dicts loss_review.analyze_energy_routing_detail() expects, with
    # score/features backfilled from the MEAN across the 20 replays per option (reduces
    # per-call determinization noise before classifying -- also directly demonstrates
    # candidate (b)'s variance-reduction argument for the v2 ranking).
    by_game: dict[str, dict] = {}
    for eid, idx in sorted(starvation_keys):
        game = games_by_id[eid]
        dec = dict(game["parsed"]["decisions"][idx])  # shallow copy
        rec = replays[(eid, idx)]
        searched = [rp for rp in rec["replays"] if rp["mode"] == "searched" and rp["scores"]]
        mean_scores: dict[str, float] = {}
        mean_features: dict[str, dict] = {}
        if searched:
            all_indices = {i for rp in searched for i in rp["scores"]}
            for i in all_indices:
                vals = [rp["scores"][i] for rp in searched if i in rp["scores"]]
                if vals:
                    mean_scores[i] = statistics.mean(vals)
            all_feat_indices = {i for rp in searched for i in (rp["features_by_index"] or {})}
            for i in all_feat_indices:
                fdicts = [rp["features_by_index"][i] for rp in searched
                          if rp["features_by_index"] and i in rp["features_by_index"]]
                if fdicts:
                    keys = fdicts[0].keys()
                    mean_features[i] = {k: statistics.mean(fd[k] for fd in fdicts)
                                         for k in keys}
        new_options = []
        for o in dec["options"]:
            o = dict(o)
            o["score"] = mean_scores.get(str(o["index"]))
            o["features"] = mean_features.get(str(o["index"]))
            new_options.append(o)
        dec["options"] = new_options
        dec["mode"] = "searched" if searched else "ladder"
        by_game.setdefault(eid, {"game": eid, "seed": None, "decisions": []})["decisions"] \
            .append(dec)

    game_list = list(by_game.values())

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        loss_review.analyze_energy_routing_detail(game_list)
    return {"report_text": buf.getvalue(), "n_starvation_decisions": len(starvation_keys)}


# ---------------------------------------------------------------------------

def main() -> None:
    games = load_v1_games()
    decline_keys = find_decline_keys(games)
    starvation_keys = find_starvation_keys(games)
    replays = load_replays()

    missing = (decline_keys | starvation_keys) - set(replays)
    if missing:
        sys.exit(f"{len(missing)} decisions have no replay data -- run "
                  f"tools/measure_near_tie_hypothesis.py first. Missing: {sorted(missing)[:5]}")

    flip = analyze_flip_and_margin(decline_keys, replays)
    outcome = analyze_outcome_linkage(games, decline_keys)
    starvation = analyze_starvation_tied_and_lost(games, starvation_keys, replays)

    print("=== Item 1: flip rate + margin distribution ===")
    print(f"decisions: {flip['n_decisions']}")
    print(f"flip (attack chosen >=1/20 replays): {flip['flip_at_least_once']} "
          f"({flip['flip_at_least_once'] / flip['n_decisions']:.1%})")
    print(f"flip (attack chosen majority >=11/20 replays): {flip['flip_majority']} "
          f"({flip['flip_majority'] / flip['n_decisions']:.1%})")
    print(f"per-decision classification: near_tie={flip['near_tie']} "
          f"confident_decline={flip['confident_decline']} "
          f"declining_was_better={flip['declining_better']} unscored={flip['unscored']}")
    pm = flip["pooled_margins"]
    if pm:
        pm_sorted = sorted(pm)
        print(f"pooled margin (best_attack - chosen), n={len(pm)}: "
              f"mean={statistics.mean(pm):.3f} median={statistics.median(pm):.3f} "
              f"min={pm_sorted[0]:.3f} max={pm_sorted[-1]:.3f} "
              f"p10={pm_sorted[len(pm) // 10]:.3f} p90={pm_sorted[len(pm) * 9 // 10]:.3f}")

    print("\n=== Item 2: outcome linkage ===")
    print(f"2a: decline rate in WINS ({outcome['n_wins']} games): "
          f"{outcome['win_declined']}/{outcome['win_legal']} "
          f"({outcome['win_rate']:.1%})" if outcome['win_rate'] is not None else "n/a")
    print(f"2a: decline rate in LOSSES ({outcome['n_losses']} games): "
          f"{outcome['loss_declined']}/{outcome['loss_legal']} "
          f"({outcome['loss_rate']:.1%})" if outcome['loss_rate'] is not None else "n/a")
    print(f"2b: opportunity preserved next {FORWARD_LOOK} turns: "
          f"{outcome['opportunity_preserved']}, lost: {outcome['opportunity_lost']}")
    if outcome["tempo_deltas"]:
        td = outcome["tempo_deltas"]
        print(f"2b: prize-differential change over next {FORWARD_LOOK} turns: "
              f"mean={statistics.mean(td):.2f} median={statistics.median(td)}")
    print(f"2c buckets: {outcome['buckets']}")
    for label, exs in outcome["examples"].items():
        for ex in exs:
            print(f"    [{label}] {ex}")

    print("\n=== Item 3: ladder tied-and-lost (real data, n="
          f"{starvation['n_starvation_decisions']}) ===")
    print(starvation["report_text"])


if __name__ == "__main__":
    main()
