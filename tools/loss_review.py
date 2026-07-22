"""Analyzes SearchScorer's LOSSES from a tools/eval_arena.py --replay-out log and answers five
behavioral questions with counts + example traces, to drive a data-backed fix instead of a
guess. See the diagnosis plan for what prompted this (SearchScorer ~coin-flipping vs. the
rule-based baseline).

Reads only the replay file (each line is self-contained: game/seed/candidate_seat/outcome/
decisions) -- no need to cross-reference results.jsonl separately.

Usage:
  .venv/Scripts/python tools/loss_review.py replay_vs_baseline.jsonl
"""

from __future__ import annotations

import json
import sys

CRUSTLE_ID = 345
DWEBBLE_ID = 344
NEAR_TIE_REL_THRESHOLD = 0.05
MAX_EXAMPLES = 3
_TIE_EPS_REL = 1e-6  # matches agents/search_scorer.py's own tie-break epsilon exactly --
                     # "tied and lost" below means the agent's OWN tie-break should have caught
                     # this, not an arbitrary new threshold invented for diagnosis


def load_losses(path: str) -> list[dict]:
    losses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["outcome"] == "opponent_win":
                losses.append(rec)
    return losses


def _chosen_option(dec: dict) -> dict | None:
    return next((o for o in dec["options"] if o["index"] == dec["chosen_index"]), None)


def _fmt_mon(m: dict | None) -> str:
    if m is None:
        return "none"
    return f"card={m['card_id']} serial={m['serial']} hp={m['hp']}/{m['max_hp']} energies={m['energies']}"


def _fmt_option(o: dict) -> str:
    parts = [f"idx={o['index']}", o["kind"]]
    if o["card_id"] is not None:
        parts.append(f"card={o['card_id']}")
    if o["target_card_id"] is not None:
        parts.append(f"target={o['target_card_id']}(serial {o['target_serial']})")
    if o["score"] is not None:
        parts.append(f"score={o['score']:.4f}")
    return " ".join(parts)


def print_example(game: dict, dec: dict, note: str) -> None:
    chosen = _chosen_option(dec)
    print(f"    [game {game['game']} seed {game['seed']} turn {dec['turn']}."
          f"{dec['turn_action_count']}] {note}")
    print(f"      you_active: {_fmt_mon(dec['you_active'])}")
    print(f"      opp_active: {_fmt_mon(dec['opp_active'])}")
    for o in dec["options"]:
        marker = " <== CHOSEN" if o["index"] == dec["chosen_index"] else ""
        print(f"      - {_fmt_option(o)}{marker}")


# ---------------------------------------------------------------------------
# (a) attack available vs. taken
# ---------------------------------------------------------------------------

def analyze_attack_availability(losses: list[dict]) -> None:
    print("\n=== (a) Attack available vs. taken ===")
    total = 0
    declined = 0
    declined_kinds: dict[str, int] = {}
    examples = []
    for game in losses:
        for dec in game["decisions"]:
            attack_opts = [o for o in dec["options"] if o["kind"] == "attack"]
            if not attack_opts:
                continue
            total += 1
            chosen = _chosen_option(dec)
            if chosen is not None and chosen["kind"] == "attack":
                continue
            declined += 1
            kind = chosen["kind"] if chosen is not None else "unknown"
            declined_kinds[kind] = declined_kinds.get(kind, 0) + 1
            if len(examples) < MAX_EXAMPLES:
                examples.append((game, dec))
    print(f"decisions where an attack was legal: {total}")
    print(f"attack declined in favor of something else: {declined} "
          f"({declined / total:.1%} of those)" if total else "n/a")
    print(f"what was chosen instead: {declined_kinds}")
    for game, dec in examples:
        print_example(game, dec, "attack was legal but declined")


# ---------------------------------------------------------------------------
# (b) attacker (Crustle) starved of energy while it went elsewhere
# ---------------------------------------------------------------------------

def _our_crustles(dec: dict) -> list[dict]:
    mons = []
    if dec["you_active"] and dec["you_active"]["card_id"] == CRUSTLE_ID:
        mons.append(dec["you_active"])
    for m in dec["you_bench"]:
        if m and m["card_id"] == CRUSTLE_ID:
            mons.append(m)
    return mons


def analyze_attacker_starved(losses: list[dict]) -> None:
    print("\n=== (b) Main attacker (Crustle) starved while energy went elsewhere ===")
    total = 0
    starved = 0
    examples = []
    for game in losses:
        for dec in game["decisions"]:
            energy_opts = [o for o in dec["options"] if o["kind"] in ("energy", "attach")]
            if not energy_opts:
                continue
            unpowered = [m for m in _our_crustles(dec) if len(m["energies"]) < 3]
            if not unpowered:
                continue
            unpowered_serials = {m["serial"] for m in unpowered}
            targets_crustle = [o for o in energy_opts if o["target_serial"] in unpowered_serials]
            if not targets_crustle:
                continue  # no legal option even reaches our unpowered Crustle this decision
            total += 1
            chosen = _chosen_option(dec)
            if chosen is not None and chosen["target_serial"] in unpowered_serials:
                continue  # we did power it
            starved += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append((game, dec))
    print(f"decisions where powering our Crustle was a legal choice: {total}")
    print(f"energy sent elsewhere instead: {starved} "
          f"({starved / total:.1%} of those)" if total else "n/a")
    for game, dec in examples:
        print_example(game, dec, "could have powered Crustle, attached elsewhere instead")


# ---------------------------------------------------------------------------
# (b-detail) WHY energy got routed away from Crustle: outscored vs. tied-and-lost vs.
# horizon/feature-blind. Needs feature dicts in the trace (agents/search_scorer.py's
# collect_features path, active whenever trace_fn is set) -- requires a replay log captured
# after that instrumentation landed.
# ---------------------------------------------------------------------------

def analyze_energy_routing_detail(losses: list[dict]) -> None:
    print("\n=== Energy-routing detail: outscored vs. tied-and-lost vs. horizon-blind ===")
    weight_imbalance = 0
    horizon_blind = 0
    tied_and_lost = 0
    dwebble_target = 0
    other_target = 0
    skipped_no_features = 0
    examples: dict[str, list] = {"weight_imbalance": [], "horizon_blind": [], "tied_and_lost": []}

    for game in losses:
        for dec in game["decisions"]:
            if dec["mode"] != "searched":
                continue
            energy_opts = [o for o in dec["options"] if o["kind"] in ("energy", "attach")]
            if not energy_opts:
                continue
            unpowered = [m for m in _our_crustles(dec) if len(m["energies"]) < 3]
            if not unpowered:
                continue
            unpowered_serials = {m["serial"] for m in unpowered}
            crustle_opts = [o for o in energy_opts if o["target_serial"] in unpowered_serials]
            if not crustle_opts:
                continue
            chosen = _chosen_option(dec)
            if chosen is None or chosen["target_serial"] in unpowered_serials:
                continue  # not a starvation case

            scored_crustle_opts = [o for o in crustle_opts if o["score"] is not None]
            if not scored_crustle_opts or chosen["score"] is None:
                skipped_no_features += 1
                continue
            best_crustle_opt = max(scored_crustle_opts, key=lambda o: o["score"])
            best_score, chosen_score = best_crustle_opt["score"], chosen["score"]

            if chosen.get("target_card_id") == DWEBBLE_ID:
                dwebble_target += 1
            else:
                other_target += 1

            eps = _TIE_EPS_REL * max(abs(chosen_score), 1.0)
            if abs(best_score - chosen_score) <= eps:
                tied_and_lost += 1
                if len(examples["tied_and_lost"]) < MAX_EXAMPLES:
                    examples["tied_and_lost"].append((game, dec))
                continue

            best_feats = best_crustle_opt.get("features") or {}
            chosen_feats = chosen.get("features") or {}
            if not best_feats or not chosen_feats:
                skipped_no_features += 1
                continue
            energy_feat_keys = ("attacker_energy_progress", "best_bench_attacker_readiness")
            differs = any(abs(best_feats.get(k, 0.0) - chosen_feats.get(k, 0.0)) > 1e-9
                          for k in energy_feat_keys)
            if differs:
                weight_imbalance += 1
                if len(examples["weight_imbalance"]) < MAX_EXAMPLES:
                    examples["weight_imbalance"].append((game, dec))
            else:
                horizon_blind += 1
                if len(examples["horizon_blind"]) < MAX_EXAMPLES:
                    examples["horizon_blind"].append((game, dec))

    total = weight_imbalance + horizon_blind + tied_and_lost
    print(f"total classified starvation decisions: {total} "
          f"(skipped, no scores/features logged: {skipped_no_features})")
    print(f"  outscored, weight-imbalance (energy feature DOES differ, just outweighed): "
          f"{weight_imbalance}")
    print(f"  outscored, horizon/feature-blind (energy feature shows NO difference at all): "
          f"{horizon_blind}")
    print(f"  tied-and-lost (agent's own tie-break should have caught this): {tied_and_lost}")
    print(f"'elsewhere' target was Dwebble (pre-evolution, energy persists through evolve): "
          f"{dwebble_target}; other target: {other_target}")

    for label, exs in examples.items():
        for game, dec in exs:
            print_example(game, dec, f"[{label}]")
            best = max((o for o in dec["options"] if o["target_serial"] is not None
                        and o["score"] is not None), key=lambda o: o["score"], default=None)
            if best is not None and best.get("features") is not None:
                print(f"      best option's features: {best['features']}")
            chosen = _chosen_option(dec)
            if chosen is not None and chosen.get("features") is not None:
                print(f"      chosen option's features: {chosen['features']}")


# ---------------------------------------------------------------------------
# (c) KOs a retreat would have avoided (heuristic -- see caveat in printed output)
# ---------------------------------------------------------------------------

def analyze_avoidable_kos(losses: list[dict]) -> None:
    print("\n=== (c) KOs a retreat might have avoided (HEURISTIC, not ground truth -- we "
          "don't log the opponent's turns, only that our active's identity changed between "
          "our own consecutive decisions without us choosing to switch) ===")
    total_declines = 0
    probable_forced_swaps = 0
    examples = []
    for game in losses:
        decs = game["decisions"]
        for i in range(len(decs) - 1):
            dec, nxt = decs[i], decs[i + 1]
            retreat_opts = [o for o in dec["options"] if o["kind"] == "retreat"]
            if not retreat_opts:
                continue
            chosen = _chosen_option(dec)
            if chosen is not None and chosen["kind"] == "retreat":
                continue  # we retreated -- not a decline
            total_declines += 1
            active_i, active_next = dec["you_active"], nxt["you_active"]
            if active_i is None or active_next is None:
                continue
            if active_i["serial"] == active_next["serial"]:
                continue  # same mon, nothing forced
            nxt_chosen = _chosen_option(nxt)
            if nxt_chosen is not None and nxt_chosen["kind"] == "retreat":
                continue  # we voluntarily switched next decision -- not KO-forced
            probable_forced_swaps += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append((game, dec, nxt))
    print(f"decisions where retreat was legal and declined: {total_declines}")
    print(f"...followed by an unexplained active-identity change (probable forced KO swap): "
          f"{probable_forced_swaps}")
    for game, dec, nxt in examples:
        print_example(game, dec, "retreat declined here")
        print(f"      next decision's active: {_fmt_mon(nxt['you_active'])} "
              f"(was: {_fmt_mon(dec['you_active'])})")


# ---------------------------------------------------------------------------
# (d) evolve/bench misplay
# ---------------------------------------------------------------------------

def analyze_evolve_misplay(losses: list[dict]) -> None:
    print("\n=== (d) Evolve declined when legal ===")
    total = 0
    declined = 0
    examples = []
    for game in losses:
        for dec in game["decisions"]:
            evolve_opts = [o for o in dec["options"] if o["kind"] == "evolve"]
            if not evolve_opts:
                continue
            total += 1
            chosen = _chosen_option(dec)
            if chosen is not None and chosen["kind"] == "evolve":
                continue
            declined += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append((game, dec))
    print(f"decisions where evolve was legal: {total}")
    print(f"evolve declined: {declined}" + (f" ({declined / total:.1%})" if total else ""))
    for game, dec in examples:
        print_example(game, dec, "evolve was legal but declined")


# ---------------------------------------------------------------------------
# (e) near-identical scores (search couldn't tell options apart)
# ---------------------------------------------------------------------------

def analyze_near_ties(losses: list[dict]) -> None:
    print(f"\n=== (e) Near-ties in evaluate() scores (top-2 within "
          f"{NEAR_TIE_REL_THRESHOLD:.0%} relative) ===")
    total_searched = 0
    near_ties = 0
    examples = []
    for game in losses:
        for dec in game["decisions"]:
            if dec["mode"] != "searched":
                continue
            scored = [o for o in dec["options"] if o["score"] is not None]
            if len(scored) < 2:
                continue
            total_searched += 1
            ranked = sorted(scored, key=lambda o: o["score"], reverse=True)
            s1, s2 = ranked[0]["score"], ranked[1]["score"]
            denom = max(abs(s1), abs(s2), 1e-9)
            rel = abs(s1 - s2) / denom
            if rel <= NEAR_TIE_REL_THRESHOLD:
                near_ties += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append((game, dec))
    print(f"'searched' decisions with >=2 scored options: {total_searched}")
    print(f"near-ties: {near_ties}" +
          (f" ({near_ties / total_searched:.1%})" if total_searched else ""))
    for game, dec in examples:
        print_example(game, dec, "top-2 options scored within 5% of each other")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loss_review.py REPLAY.jsonl", file=sys.stderr)
        sys.exit(1)
    losses = load_losses(sys.argv[1])
    total_decisions = sum(len(g["decisions"]) for g in losses)
    print(f"Loaded {len(losses)} losses, {total_decisions} total logged decisions across them.")

    analyze_attack_availability(losses)
    analyze_attacker_starved(losses)
    analyze_energy_routing_detail(losses)
    analyze_avoidable_kos(losses)
    analyze_evolve_misplay(losses)
    analyze_near_ties(losses)


if __name__ == "__main__":
    main()
