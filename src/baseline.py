"""Rule-based baseline agent: the sparring partner and submission fallback, NOT the
learned agent (see CLAUDE.md's agent philosophy -- no hand-coded strategy belongs in the
eventual network path; this module is a deliberately separate, disposable heuristic).

Reasons only over engine-offered legal options via src/obs.py -- it never invents
legality. Card knowledge (HP, attack damage/cost) comes straight from cg.api's own
all_card_data()/all_attack(), not the CSV -- keeping this on the cheap, CSV-free path
that a packaged submission actually runs.
"""

from __future__ import annotations

import os

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import all_attack, all_card_data  # noqa: E402

from obs import GameState, LegalOption, Selection, parse_obs  # noqa: E402

_CARD_BY_ID = {c.cardId: c for c in all_card_data()}
_ATTACK_BY_ID = {a.attackId: a for a in all_attack()}


def _pokemon_max_hp(card_id: int) -> int:
    card = _CARD_BY_ID.get(card_id)
    return card.hp if card else 0


def _attack_damage(attack_id: int) -> int:
    atk = _ATTACK_BY_ID.get(attack_id)
    return atk.damage if atk else 0


def _by_kind(selection: Selection, kind: str) -> list[LegalOption]:
    return [lo for lo in selection.options if lo.kind == kind]


def _pick_attack(selection: Selection, game_state: GameState) -> LegalOption | None:
    """Prefer a lethal attack (damage >= opponent active HP); else the highest-damage one."""
    attacks = _by_kind(selection, "attack")
    if not attacks:
        return None
    opp_active = game_state.opponent.active
    opp_hp = opp_active.hp if opp_active else None

    def damage_of(lo: LegalOption) -> int:
        return _attack_damage(lo.raw.attackId)

    if opp_hp is not None:
        lethal = [lo for lo in attacks if damage_of(lo) >= opp_hp > 0]
        if lethal:
            return min(lethal, key=damage_of)  # cheapest attack that still KOs
    return max(attacks, key=damage_of)


def _pick_retreat(selection: Selection, game_state: GameState) -> LegalOption | None:
    """Only retreat when active is badly hurt and a healthier bench Pokemon exists."""
    retreats = _by_kind(selection, "retreat")
    if not retreats:
        return None
    you = game_state.you
    if you.active is None or not you.bench:
        return None
    active_hp_frac = you.active.hp / max(_pokemon_max_hp(you.active.card_id), 1)
    if active_hp_frac > 0.3:
        return None
    healthier = [p for p in you.bench if p.hp > you.active.hp]
    if not healthier:
        return None
    return retreats[0]


def choose_action(game_state: GameState, selection: Selection) -> list[int]:
    """Priority: lethal attack > attach energy > evolve > best attack > retreat (if
    justified) > play a card > accept the least-bad remaining legal option / end."""
    if selection.max_count == 0:
        return []

    lethal_or_best_attack = _pick_attack(selection, game_state)
    if lethal_or_best_attack is not None:
        opp_active = game_state.opponent.active
        if opp_active is not None and _attack_damage(lethal_or_best_attack.raw.attackId) >= opp_active.hp > 0:
            return [lethal_or_best_attack.index]

    for kind in ("energy", "attach"):
        options = _by_kind(selection, kind)
        if options:
            return [options[0].index]

    for kind in ("evolve",):
        options = _by_kind(selection, kind)
        if options:
            return [options[0].index]

    if lethal_or_best_attack is not None:
        return [lethal_or_best_attack.index]

    retreat = _pick_retreat(selection, game_state)
    if retreat is not None:
        return [retreat.index]

    for kind in ("play", "ability"):
        options = _by_kind(selection, kind)
        if options:
            return [options[0].index]

    # Fallback: end the turn if legal, else take the first legal option(s).
    end = _by_kind(selection, "end")
    if end:
        return [end[0].index]

    return [lo.index for lo in selection.options[:selection.max_count]]


def read_deck_csv(path: str = "deck.csv") -> list[int]:
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + os.path.basename(path)
    with open(path) as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def agent(obs_dict: dict) -> list[int]:
    game_state, selection = parse_obs(obs_dict)
    if game_state is None:
        return read_deck_csv()
    return choose_action(game_state, selection)
