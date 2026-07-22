"""SearchScorer: a lookahead agent for the Dwebble/Crustle wall deck
(decks/crustle_wall_deck.csv), built per the ARCHITECTURE DECISION in CLAUDE.md (no
self-play RL -- search over the engine's own search_begin/search_step API + a
hand-crafted evaluate()).

Per decision: determinize the hidden world once (src/determinize.py, same prior as the
deprecated MCTS), then for each of our legal root options run a fresh search_begin +
search_step branch, auto-advancing through forced/empty selections, greedily continuing
one further self-decision if the "2 ply" budget allows and it's still our turn, and
scoring the resulting state with evaluate(). We have no opponent model, so a branch that
lands on the opponent's decision (or a terminal) is scored immediately rather than
guessed further. This 2-ply lookahead legitimately makes many different first actions
converge to the same best-reachable position (e.g. "attach energy, then attack" ties
with "attack directly" when both can still reach the attack) -- see MAX_OUR_PLIES's
comment and _TIE_BREAK_PRIORITY for how real losses showed this needs a merit-based
tie-break, not a shallower search.

Robustness is the hard requirement: choose_action -> baseline agent -> raw legal slice
-> empty list, wrapped so this agent can never crash or return an illegal index.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()

from cg.api import (  # noqa: E402
    EnergyType, all_attack, all_card_data, search_begin, search_step, search_release,
    to_observation_class,
)

from obs import (  # noqa: E402
    GameState, PlayerView, PokemonView, Selection, parse_obs, parse_observation,
)
from determinize import sample_determinization  # noqa: E402
from baseline import choose_action as baseline_choose_action, read_deck_csv  # noqa: E402

_CARD = {c.cardId: c for c in all_card_data()}
_ATTACK = {a.attackId: a for a in all_attack()}

# Deck-specific card ids (decks/crustle_wall_deck.csv) -- see CLAUDE.md / plan for the read.
CRUSTLE_ID = 345
DWEBBLE_ID = 344
HEROS_CAPE_ID = 1159
FOREST_OF_VITALITY_ID = 1261
ATTACKER_ENERGY_COST = 3  # Superb Scissors: {G} + 2 colorless

MAX_ROOT_OPTIONS = 30  # beyond this, defer to the cheap baseline for that decision
# tools/loss_review.py on 114 real losses vs. baseline found 86.5% of searched decisions had
# their top-2 options tied within 5%: at MAX_OUR_PLIES=2, e.g. "play a Supporter, then attack"
# and "attack directly" both reach the same 2-ply-best position and score identically, so ties
# get resolved by option-list order (an engine-ordering artifact) rather than merit. First
# attempt at a fix (dropping this to 1, scoring each option's own immediate consequence
# instead of "best reachable within budget") was WORSE (10% vs. baseline over 200 games,
# confirmed, not noise) -- it made the search myopic, no longer crediting setup actions
# (attach energy, evolve) for the attack they set up next ply, which is fatal for a deck that
# must power up before it can do anything. Reverted to 2; the actual fix is the tie-break rule
# below, not the depth.
MAX_OUR_PLIES = 2
MAX_STEP_DEPTH = 80    # mirrors mcts.py's forced-step cap

# Tie-break priority for options within _TIE_EPS_REL of the best score (lower = more preferred).
# These near-ties are usually genuinely-equal-value positions under evaluate() (see above), so
# breaking them by option-list order is arbitrary; break them instead by the deck's own game
# plan -- attack now > evolve > power up the attacker > retreat > everything else > end.
_TIE_BREAK_PRIORITY = {
    "attack": 0,
    "evolve": 1,
    "attach": 2, "energy": 2,
    "retreat": 3,
    "play": 4, "ability": 4,
    "energy_card": 5, "tool_card": 5, "card": 5,
    "discard": 6, "special_condition": 6, "number": 6, "yes": 6, "no": 6, "skill": 6,
    "end": 7,
}
_TIE_EPS_REL = 1e-6  # near-exact float equality -- these are usually the SAME reachable state

WIN_SCORE = 1e6
LOSS_SCORE = -1e6

# Observability for tools/eval_arena.py: which fallback tier actually got used, if any.
# Incremented only once a tier's call *succeeds* (a tier that itself raises falls through to
# the next one uncounted), so counts reflect what really happened, not what was attempted.
_FALLBACK_COUNTS = {
    "too_many_options": 0,        # choose_action deferred to baseline -- root option count > cap
    "search_rejected": 0,         # every sampled world/branch was rejected -- deferred to baseline
    "exception_to_baseline": 0,   # choose_action raised; baseline_choose_action succeeded
    "exception_to_raw": 0,        # baseline_choose_action also raised; raw legal slice used
    "empty": 0,                   # everything failed; returned []
}


def reset_fallback_counts() -> None:
    for key in _FALLBACK_COUNTS:
        _FALLBACK_COUNTS[key] = 0


def get_fallback_counts() -> dict:
    return dict(_FALLBACK_COUNTS)

# Tunable weights -- single source of truth for later offline tuning (weights only, the
# feature functions below never need to change alongside a tuning pass).
WEIGHTS: dict[str, float] = {
    "prize_diff": 1.0,
    "hp_frac_diff": 2.0,
    "active_hp_frac_diff": 1.5,
    "attacker_energy_progress": 1.0,
    "attacker_ready_and_active": 1.5,
    "evolution_progress": 1.0,
    "hand_diff": 0.15,
    "cape_attached": 0.5,
    "stadium_active": 0.2,
    "ex_matchup_bonus": 0.75,
    "deck_out_risk": -1.0,
    # Threat / bench-attacker / tempo features -- added after the tie-collapse diagnosis
    # (86.5% of decisions near-tied; see MAX_OUR_PLIES's comment) showed the tie-break fix
    # alone (45% vs. baseline over 200 games) couldn't move win rate, because these states
    # were never actually distinguishable to evaluate() in the first place.
    "opp_can_ko_our_active": -3.0,       # large: losing our key piece usually costs more than
                                          # any single-turn tempo gain (existing features top
                                          # out around +-2-3 in traced examples)
    "we_threaten_ko": 1.5,               # real forward value, but a threat can still be
                                          # answered (they can retreat) -- less certain than
                                          # an immediate loss, so smaller than the defensive one
    "prize_race_delta": 0.75,            # same denomination as prize_diff (1.0), discounted
                                          # since it's predictive/next-turn, not realized
    "exposed_investment": -4.0,          # explicitly heavy: sized to dominate typical
                                          # non-terminal score gaps outright
    "best_bench_attacker_readiness": 1.0,   # mirrors attacker_energy_progress, bench slot
    "bench_attacker_advantage_bonus": 1.25,  # the actionable "retreat is worth it" signal
    "damage_dealt_this_turn": 0.5,       # small direct reward -- just enough to break the
                                          # literal "attack now vs. delay" tie
    # tools/loss_review.py's analyze_energy_routing_detail on 111 real "energy sent elsewhere"
    # losses found 74% were "outscored, weight-imbalance" (attacker_energy_progress/
    # best_bench_attacker_readiness DID increase for the right target, just got outweighed by
    # other features) vs. only 2.7% true horizon-blindness and ~23% tied-and-lost (mostly
    # DEFENSIBLE tie-break choices, e.g. preferring attack/evolve over a redundant attach when
    # genuinely tied -- not a bug). 29% of the "elsewhere" targets were a pre-evolution Dwebble,
    # which neither existing energy feature tracks (energy persists through evolution in real
    # PTCG rules) -- confirming a real, distinct gap. turns_to_power directly targets both
    # findings: bigger weight than the existing capped energy features, and covers Dwebble.
    "turns_to_power": 2.5,
    "wasted_energy": -1.0,               # smaller, complementary: redundant energy piling onto
                                          # an already-topped-off attacker while a teammate is
                                          # still short -- weaker evidence than turns_to_power's,
                                          # kept modest accordingly
}


def load_weights(path: str) -> dict[str, float]:
    """Load a candidate weights dict from a JSON file (tools/tune_weights.py's format),
    used by tools/_eval_worker.py's --candidate-weights/--opponent-weights so a single
    process can run two DIFFERENT search_scorer weight sets against each other (the
    self-relative tuning matchup, or a bake-off between versions from different sessions).

    Unknown keys (typos, stale feature names) are a hard error. Keys MISSING relative to the
    current WEIGHTS default to 0.0 -- this is what correctly reproduces an older snapshot
    taken before a feature existed (e.g. runs/tune_run1's pre-energy-routing-fix snapshots
    predate turns_to_power/wasted_energy): the feature literally didn't contribute to that
    version's score, which a 0 weight reproduces exactly, rather than silently pulling in
    today's default for a feature that version never had."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    extra = set(data) - set(WEIGHTS)
    if extra:
        raise ValueError(f"weights file {path!r} has unknown keys: {extra}")
    result = {k: float(v) for k, v in data.items()}
    for k in set(WEIGHTS) - set(data):
        result[k] = 0.0
    return result


def _in_play(pv: PlayerView) -> list:
    mons = list(pv.bench)
    if pv.active is not None:
        mons = [pv.active] + mons
    return mons


def _hp_fraction(pv: PlayerView) -> float:
    mons = _in_play(pv)
    total_max = sum(m.max_hp for m in mons)
    if total_max <= 0:
        return 0.0
    return sum(m.hp for m in mons) / total_max


def _active_hp_fraction(pv: PlayerView) -> float:
    if pv.active is None or pv.active.max_hp <= 0:
        return 0.0
    return pv.active.hp / pv.active.max_hp


def _mon_summary(mon) -> dict | None:
    """Compact, JSON-able summary of a PokemonView for replay traces (tools/loss_review.py)."""
    if mon is None:
        return None
    return {
        "card_id": mon.card_id, "serial": mon.serial, "hp": mon.hp, "max_hp": mon.max_hp,
        "energies": [int(e) for e in mon.energies],
    }


def _our_view(gs: GameState, root_your_index: int) -> tuple[PlayerView, PlayerView]:
    """gs.you/gs.opponent are relative to whichever side is CURRENTLY THE ACTOR, which flips
    once a search branch lands on the opponent's own decision (gs.your_index != root's) --
    same reason src/mcts.py's _backup flips sign on to_move. Never read gs.you/opponent
    directly when scoring a searched branch; always resolve through this."""
    if gs.your_index == root_your_index:
        return gs.you, gs.opponent
    return gs.opponent, gs.you


def _cost_satisfied(attached: Counter, cost: Counter) -> bool:
    """Can `attached` energy (Counter of int EnergyType) pay `cost`? Colorless cost entries
    (EnergyType.COLORLESS == 0) may be paid by any leftover energy of any type, after
    specific-type costs are met first -- standard PTCG cost matching."""
    remaining = Counter(attached)
    colorless_needed = 0
    for etype, count in cost.items():
        if etype == int(EnergyType.COLORLESS):
            colorless_needed += count
            continue
        if remaining[etype] < count:
            return False
        remaining[etype] -= count
    return sum(remaining.values()) >= colorless_needed


def _best_ready_attack_damage(mon: PokemonView | None) -> int:
    """Max damage among `mon`'s attacks whose energy cost is satisfied by its CURRENTLY
    attached energy -- a conservative "can KO right now, no further setup" signal computed
    purely from visible card/energy data (never hidden information)."""
    if mon is None:
        return 0
    card = _CARD.get(mon.card_id)
    if card is None:
        return 0
    attached = Counter(int(e) for e in mon.energies)
    best = 0
    for attack_id in card.attacks:
        atk = _ATTACK.get(attack_id)
        if atk is None:
            continue
        cost = Counter(int(e) for e in atk.energies)
        if _cost_satisfied(attached, cost):
            best = max(best, atk.damage)
    return best


def _prize_value(card_id: int | None) -> int:
    """Prizes the OPPONENT takes if this Pokemon is KO'd: 3 for Mega ex, 2 for ex, else 1."""
    card = _CARD.get(card_id) if card_id is not None else None
    if card is None:
        return 1
    if card.megaEx:
        return 3
    if card.ex:
        return 2
    return 1


def evaluate(gs: GameState, root_your_index: int,
             root_opp_active: PokemonView | None = None,
             weights: dict[str, float] | None = None,
             return_features: bool = False):
    """Score a (possibly searched) state from the ROOT decision-maker's perspective.
    `root_opp_active` is a snapshot of the opponent's active BEFORE this decision's simulated
    actions (only used by damage_dealt_this_turn, to notice damage that happened mid-line --
    evaluate() itself is otherwise purely a function of the current state). `weights` defaults
    to the module-level WEIGHTS; tools/tune_weights.py passes an explicit candidate dict so two
    differently-weighted search_scorer instances can play each other in the same process.
    `return_features`, if True, returns (score, raw_features_dict) instead of just score --
    used only by tools/loss_review.py's diagnostics (via choose_action's trace_fn path), never
    in normal play; default False keeps the plain-float contract everyone else relies on."""
    if weights is None:
        weights = WEIGHTS
    if gs.result != -1:
        if gs.result == 2:
            term_score = 0.0
        else:
            term_score = WIN_SCORE if gs.result == root_your_index else LOSS_SCORE
        return (term_score, {}) if return_features else term_score

    you, opp = _our_view(gs, root_your_index)

    active = you.active
    active_crustle = active if active is not None and active.card_id == CRUSTLE_ID else None

    energy_progress = 0.0
    ready_and_active = 0.0
    cape_attached = 0.0
    if active_crustle is not None:
        total_energy = len(active_crustle.energies)
        has_grass = EnergyType.GRASS in active_crustle.energies
        energy_progress = 0.5 * min(total_energy / ATTACKER_ENERGY_COST, 1.0) + \
            0.5 * (1.0 if has_grass else 0.0)
        if total_energy >= ATTACKER_ENERGY_COST and has_grass:
            ready_and_active = 1.0
        cape_attached = 1.0 if any(t.card_id == HEROS_CAPE_ID for t in active_crustle.tools) \
            else 0.0

    our_mons = _in_play(you)
    n_crustle = sum(1 for m in our_mons if m.card_id == CRUSTLE_ID)
    n_dwebble = sum(1 for m in our_mons if m.card_id == DWEBBLE_ID)
    evolution_progress = 1.0 * n_crustle + 0.4 * n_dwebble

    stadium_active = 1.0 if any(c.card_id == FOREST_OF_VITALITY_ID for c in gs.stadium) else 0.0

    opp_active_card = _CARD.get(opp.active.card_id) if opp.active is not None else None
    ex_matchup_bonus = 1.0 if (active_crustle is not None and opp_active_card is not None
                                and opp_active_card.ex) else 0.0

    deck_out_risk = 1.0 / (you.deck_count + 1)

    # --- threat features -------------------------------------------------------------
    opp_can_ko = (you.active is not None and you.active.hp > 0
                  and _best_ready_attack_damage(opp.active) >= you.active.hp)
    we_threaten = (opp.active is not None and opp.active.hp > 0
                   and _best_ready_attack_damage(you.active) >= opp.active.hp)
    prize_race_delta = ((_prize_value(opp.active.card_id) if we_threaten else 0)
                        - (_prize_value(you.active.card_id) if opp_can_ko else 0))

    exposed_investment = 0.0
    energized = [(m, len(m.energies)) for m in our_mons if len(m.energies) > 0]
    if energized:
        top_mon, top_energy = max(energized, key=lambda t: t[1])
        if (opp_can_ko and you.active is not None and top_mon.serial == you.active.serial):
            exposed_investment = top_energy / ATTACKER_ENERGY_COST

    # --- bench-attacker features ------------------------------------------------------
    bench_crustles = [m for m in you.bench if m.card_id == CRUSTLE_ID]
    best_bench_energy, best_bench_has_grass = 0, False
    for m in bench_crustles:
        total = len(m.energies)
        has_grass_m = EnergyType.GRASS in m.energies
        if total > best_bench_energy or (total == best_bench_energy and has_grass_m
                                          and not best_bench_has_grass):
            best_bench_energy, best_bench_has_grass = total, has_grass_m
    best_bench_attacker_readiness = 0.0
    if bench_crustles:
        best_bench_attacker_readiness = (
            0.5 * min(best_bench_energy / ATTACKER_ENERGY_COST, 1.0)
            + 0.5 * (1.0 if best_bench_has_grass else 0.0))
    bench_ready = best_bench_energy >= ATTACKER_ENERGY_COST and best_bench_has_grass
    bench_attacker_advantage_bonus = 1.0 if (bench_ready and ready_and_active == 0.0) else 0.0

    # --- energy-routing features (turns_to_power / wasted_energy) --------------------
    # Energy attached to a pre-evolution Dwebble persists through evolution into Crustle, so
    # it counts toward the SAME eventual attack cost -- neither attacker_energy_progress
    # (active-Crustle-only) nor best_bench_attacker_readiness (bench-Crustle-only) tracks that.
    pipeline_mons = [m for m in our_mons if m.card_id in (CRUSTLE_ID, DWEBBLE_ID)]
    deficits = [max(0, ATTACKER_ENERGY_COST - len(m.energies)) for m in pipeline_mons]
    turns_to_power = 1.0 - (min(deficits) / ATTACKER_ENERGY_COST) if deficits else 0.0
    wasted_energy = 0.0
    if deficits and min(deficits) > 0:  # someone in the pipeline is still short
        wasted_energy = sum(max(0, len(m.energies) - ATTACKER_ENERGY_COST)
                             for m in pipeline_mons) / ATTACKER_ENERGY_COST

    # --- tempo feature -------------------------------------------------------------
    damage_dealt_this_turn = 0.0
    if root_opp_active is not None:
        if opp.active is not None and opp.active.serial == root_opp_active.serial:
            damage_dealt_this_turn = (max(0, root_opp_active.hp - opp.active.hp)
                                      / max(root_opp_active.max_hp, 1))
        elif opp.active is None or opp.active.serial != root_opp_active.serial:
            # their original active is no longer in the active slot -- most likely we KO'd
            # it this line; credit the damage that KO represents (prize_diff separately
            # captures the prize-take itself)
            damage_dealt_this_turn = root_opp_active.hp / max(root_opp_active.max_hp, 1)

    features = {
        "prize_diff": float(len(opp.prize) - len(you.prize)),
        "hp_frac_diff": _hp_fraction(you) - _hp_fraction(opp),
        "active_hp_frac_diff": _active_hp_fraction(you) - _active_hp_fraction(opp),
        "attacker_energy_progress": energy_progress,
        "attacker_ready_and_active": ready_and_active,
        "evolution_progress": evolution_progress,
        "hand_diff": float(max(-5, min(5, you.hand_count - opp.hand_count))),
        "cape_attached": cape_attached,
        "stadium_active": stadium_active,
        "ex_matchup_bonus": ex_matchup_bonus,
        "deck_out_risk": deck_out_risk,
        "opp_can_ko_our_active": 1.0 if opp_can_ko else 0.0,
        "we_threaten_ko": 1.0 if we_threaten else 0.0,
        "prize_race_delta": float(prize_race_delta),
        "exposed_investment": exposed_investment,
        "best_bench_attacker_readiness": best_bench_attacker_readiness,
        "bench_attacker_advantage_bonus": bench_attacker_advantage_bonus,
        "damage_dealt_this_turn": damage_dealt_this_turn,
        "turns_to_power": turns_to_power,
        "wasted_energy": wasted_energy,
    }
    score = sum(weights[name] * value for name, value in features.items())
    return (score, features) if return_features else score


def _build_index_list(sel: Selection, live_index: int) -> list[int]:
    if sel.min_count <= 1:
        return [live_index]
    others = [lo.index for lo in sel.options if lo.index != live_index]
    return [live_index] + others[: sel.min_count - 1]


def _advance_to_branch(sid: int, cur, depth: int):
    """Step through forced/empty selections (max_count==0) until a real decision or terminal
    is reached. Returns (cur, gs, sel, result, depth)."""
    gs, sel = parse_observation(cur.observation)
    result = cur.observation.current.result if cur.observation.current is not None else -1
    while result == -1 and gs is not None and sel is not None and \
            (sel.max_count == 0 or not sel.options):
        cur = search_step(sid, [])
        gs, sel = parse_observation(cur.observation)
        result = cur.observation.current.result if cur.observation.current is not None else -1
        depth += 1
        if depth > MAX_STEP_DEPTH:
            break
    return cur, gs, sel, result, depth


def _score_branch(sid: int, cur, root_your_index: int, plies_used: int, depth: int,
                   root_opp_active: PokemonView | None,
                   weights: dict[str, float], collect_features: bool = False):
    try:
        cur, gs, sel, result, depth = _advance_to_branch(sid, cur, depth)
    except (RuntimeError, ValueError):
        return (0.0, {}) if collect_features else 0.0  # engine rejected a forced step here

    if result != -1 or gs is None or sel is None or depth > MAX_STEP_DEPTH:
        term_score = 0.0 if result in (-1, 2) else (
            WIN_SCORE if result == root_your_index else LOSS_SCORE)
        return (term_score, {}) if collect_features else term_score

    if gs.your_index != root_your_index or plies_used >= MAX_OUR_PLIES or not sel.options:
        return evaluate(gs, root_your_index, root_opp_active, weights,
                         return_features=collect_features)

    best = None
    best_features: dict = {}
    for lo in sel.options:
        try:
            nxt = search_step(sid, _build_index_list(sel, lo.index))
        except (RuntimeError, ValueError):
            continue
        if collect_features:
            score, feats = _score_branch(sid, nxt, root_your_index, plies_used + 1, depth + 1,
                                          root_opp_active, weights, collect_features=True)
        else:
            score = _score_branch(sid, nxt, root_your_index, plies_used + 1, depth + 1,
                                   root_opp_active, weights, collect_features=False)
        if best is None or score > best:
            best = score
            if collect_features:
                best_features = feats
    if best is None:
        return evaluate(gs, root_your_index, root_opp_active, weights,
                         return_features=collect_features)
    return (best, best_features) if collect_features else best


def _score_option(world: dict, root_observation, sel: Selection, lo, root_your_index: int,
                   root_opp_active: PokemonView | None, weights: dict[str, float],
                   collect_features: bool = False):
    """Returns a float score (or (score, features) if collect_features), or None if the engine
    rejected this sampled world/step."""
    try:
        ss = search_begin(
            root_observation, world["your_deck"], world["your_prize"],
            world["opponent_deck"], world["opponent_prize"],
            world["opponent_hand"], world["opponent_active"],
        )
    except (RuntimeError, ValueError):
        return None
    sid = ss.searchId
    try:
        try:
            cur = search_step(sid, _build_index_list(sel, lo.index))
        except (RuntimeError, ValueError):
            return None
        return _score_branch(sid, cur, root_your_index, plies_used=1, depth=0,
                              root_opp_active=root_opp_active, weights=weights,
                              collect_features=collect_features)
    finally:
        search_release(sid)


def _unpowered_crustle_serials(game_state: GameState) -> set[int]:
    mons = list(game_state.you.bench)
    if game_state.you.active is not None:
        mons.append(game_state.you.active)
    return {m.serial for m in mons
            if m.card_id == CRUSTLE_ID and len(m.energies) < ATTACKER_ENERGY_COST}


def _tie_break_key(lo, unpowered_serials: set[int]) -> tuple[int, int]:
    base = _TIE_BREAK_PRIORITY.get(lo.kind, 8)
    targets_unpowered_attacker = 0 if (lo.target is not None
                                        and lo.target.serial in unpowered_serials) else 1
    return (base, targets_unpowered_attacker)


def _break_ties(selection: Selection, scores: dict[int, float], best_score: float,
                 game_state: GameState):
    """Among options within _TIE_EPS_REL of best_score, prefer by _TIE_BREAK_PRIORITY instead
    of the arbitrary engine option-list order (see MAX_OUR_PLIES's comment for why this
    matters -- real losses showed the search legitimately ties many options)."""
    eps = _TIE_EPS_REL * max(abs(best_score), 1.0)
    tied = [lo for lo in selection.options
            if lo.index in scores and abs(scores[lo.index] - best_score) <= eps]
    if len(tied) <= 1:
        return tied[0] if tied else None
    unpowered = _unpowered_crustle_serials(game_state)
    return min(tied, key=lambda lo: _tie_break_key(lo, unpowered))


def _trace_options(selection: Selection, scores: dict[int, float] | None,
                    features_by_index: dict[int, dict] | None = None) -> list[dict]:
    out = []
    for lo in selection.options:
        out.append({
            "index": lo.index, "kind": lo.kind,
            "card_id": lo.card.card_id if lo.card else None,
            "target_card_id": lo.target.card_id if lo.target else None,
            "target_serial": lo.target.serial if lo.target else None,
            "score": (scores.get(lo.index) if scores else None),
            "features": (features_by_index.get(lo.index) if features_by_index else None),
        })
    return out


def choose_action(game_state: GameState, selection: Selection, obs_dict: dict,
                   deck_list: list[int], opp_deck_list: list[int] | None = None,
                   trace_fn=None, weights: dict[str, float] | None = None) -> list[int]:
    if weights is None:
        weights = WEIGHTS

    collect_features = trace_fn is not None

    def emit(mode: str, scores: dict[int, float] | None = None,
             chosen_index: int | None = None,
             features_by_index: dict[int, dict] | None = None) -> None:
        if trace_fn is None:
            return
        trace_fn({
            "turn": game_state.turn, "turn_action_count": game_state.turn_action_count,
            "mode": mode,
            "you_active": _mon_summary(game_state.you.active),
            "you_bench": [_mon_summary(m) for m in game_state.you.bench],
            "opp_active": _mon_summary(game_state.opponent.active),
            "options": _trace_options(selection, scores, features_by_index),
            "chosen_index": chosen_index,
        })

    if selection.max_count == 0:
        emit("no_choice")
        return []
    if len(selection.options) == 1:
        emit("single_option", chosen_index=selection.options[0].index)
        return [selection.options[0].index]
    if len(selection.options) > MAX_ROOT_OPTIONS:
        _FALLBACK_COUNTS["too_many_options"] += 1
        result = baseline_choose_action(game_state, selection)
        emit("too_many_options", chosen_index=result[0] if result else None)
        return result

    root_observation = to_observation_class(obs_dict)
    root_your_index = game_state.your_index
    root_opp_active = game_state.opponent.active  # snapshot before any simulated action
    world = sample_determinization(game_state, deck_list, opp_deck_list)

    best_lo = None
    best_score = None
    scores: dict[int, float] = {}
    features_by_index: dict[int, dict] = {}
    for lo in selection.options:
        result = _score_option(world, root_observation, selection, lo, root_your_index,
                                root_opp_active, weights, collect_features=collect_features)
        if result is None:
            continue
        if collect_features:
            score, feats = result
            features_by_index[lo.index] = feats
        else:
            score = result
        scores[lo.index] = score
        if best_score is None or score > best_score:
            best_score, best_lo = score, lo

    if best_lo is None:  # every candidate world/step rejected -- fall back
        _FALLBACK_COUNTS["search_rejected"] += 1
        result = baseline_choose_action(game_state, selection)
        emit("search_rejected", chosen_index=result[0] if result else None)
        return result

    tie_broken = _break_ties(selection, scores, best_score, game_state)
    if tie_broken is not None:
        best_lo = tie_broken

    emit("searched", scores=scores, chosen_index=best_lo.index,
         features_by_index=features_by_index)
    return _build_index_list(selection, best_lo.index)


def agent(obs_dict: dict, deck_list: list[int] | None = None,
          opp_deck_list: list[int] | None = None, trace_fn=None,
          weights: dict[str, float] | None = None) -> list[int]:
    """Top-level entry point. Never raises, never returns an illegal index: SearchScorer,
    falling back to the rule-based baseline, falling back to a raw legal-index slice.

    `trace_fn`, if given, is called once per decision with a JSON-able record (see
    choose_action.emit) -- used by tools/_eval_worker.py's --replay-out logging. It's purely
    additive/observational and never affects which action is chosen. `weights` defaults to the
    module-level WEIGHTS; see evaluate()'s docstring for why tools/tune_weights.py overrides it."""
    game_state = selection = None
    try:
        game_state, selection = parse_obs(obs_dict)
        if game_state is None:
            return read_deck_csv()
        deck = deck_list if deck_list is not None else read_deck_csv()
        return choose_action(game_state, selection, obs_dict, deck, opp_deck_list,
                              trace_fn=trace_fn, weights=weights)
    except Exception:
        pass

    try:
        if game_state is None or selection is None:
            game_state, selection = parse_obs(obs_dict)
        if game_state is not None and selection is not None:
            result = baseline_choose_action(game_state, selection)
            _FALLBACK_COUNTS["exception_to_baseline"] += 1
            return result
    except Exception:
        pass

    try:
        if selection is not None:
            result = [lo.index for lo in selection.options[:selection.max_count]]
            _FALLBACK_COUNTS["exception_to_raw"] += 1
            return result
    except Exception:
        pass

    _FALLBACK_COUNTS["empty"] += 1
    return []


def make_agent(deck_list: list[int], opp_deck_list: list[int] | None = None, trace_fn=None,
               weights: dict[str, float] | None = None):
    """Bind a deck (and optional opponent-deck prior, replay trace_fn, and candidate weights)
    into a plain obs_dict -> indices agent, e.g. for use in tests/evaluate.py's
    play_match/win_rate harness or tools/_eval_worker.py. `weights` defaults to the module
    WEIGHTS -- pass an explicit dict (tools.search_scorer.load_weights) to bind a specific
    tuning candidate, independent of any other search_scorer instance in the same process."""
    def _agent(obs_dict: dict) -> list[int]:
        return agent(obs_dict, deck_list, opp_deck_list, trace_fn=trace_fn, weights=weights)
    return _agent
