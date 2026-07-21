"""Observation/state wrapper: turns the raw engine obs dict into a typed GameState plus
classified legal options.

Own hand, own attachments, and every public zone (discard, stadium, bench/active Pokemon
and what's attached to them) are genuinely visible per PTCG rules, and the engine already
reflects that -- opponent hand contents come back None from cg itself; this module never
adds visibility the engine didn't already grant. tests/test_obs.py asserts that invariant
so it survives future edits (this matters beyond just this baseline: the future network
encoder reuses this same wrapper and must never see hidden opponent state either).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import (  # noqa: E402
    AreaType, Card, EnergyType, Observation, Option, OptionType, Pokemon, PlayerState,
    SelectContext, SelectData, SelectType, State, to_observation_class,
)


@dataclass
class CardRef:
    card_id: int
    serial: int


@dataclass
class PokemonView:
    card_id: int
    serial: int
    hp: int
    max_hp: int
    appear_this_turn: bool
    energies: list[EnergyType]
    energy_cards: list[CardRef]
    tools: list[CardRef]
    pre_evolution: list[CardRef]


@dataclass
class PlayerView:
    active: Optional[PokemonView]
    bench: list[PokemonView]
    bench_max: int
    deck_count: int
    discard: list[CardRef]
    # Face-down prize entries are None for BOTH players -- genuinely unknown until taken,
    # not a hidden-on-purpose redaction.
    prize: list[Optional[CardRef]]
    hand_count: int
    hand: Optional[list[CardRef]]  # None for the opponent, straight from the engine
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass
class GameState:
    turn: int
    turn_action_count: int
    your_index: int
    first_player: int
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    result: int
    stadium: list[CardRef]
    you: PlayerView
    opponent: PlayerView


@dataclass
class LegalOption:
    index: int  # position in select.option -- pass straight back to battle_select/search_step
    kind: str  # human label derived from OptionType, e.g. "attack", "attach", "end"
    raw: Option  # untouched engine Option, for any field this module doesn't surface
    card: Optional[CardRef] = None  # card being played/attached/evolved/discarded, if resolvable
    target: Optional[PokemonView] = None  # in-play Pokemon being targeted, if resolvable


@dataclass
class Selection:
    """Everything needed to build a valid battle_select/search_step call: pick between
    min_count and max_count option indices from `options`, no duplicates."""
    select_type: SelectType
    context: SelectContext
    min_count: int
    max_count: int
    options: list[LegalOption]


_KIND_BY_OPTION_TYPE = {
    OptionType.NUMBER: "number",
    OptionType.YES: "yes",
    OptionType.NO: "no",
    OptionType.CARD: "card",
    OptionType.TOOL_CARD: "tool_card",
    OptionType.ENERGY_CARD: "energy_card",
    OptionType.ENERGY: "energy",
    OptionType.PLAY: "play",
    OptionType.ATTACH: "attach",
    OptionType.EVOLVE: "evolve",
    OptionType.ABILITY: "ability",
    OptionType.DISCARD: "discard",
    OptionType.RETREAT: "retreat",
    OptionType.ATTACK: "attack",
    OptionType.END: "end",
    OptionType.SKILL: "skill",
    OptionType.SPECIAL_CONDITION: "special_condition",
}


def _card_ref(card: Optional[Card]) -> Optional[CardRef]:
    if card is None:
        return None
    return CardRef(card_id=card.id, serial=card.serial)


def _pokemon_view(p: Optional[Pokemon]) -> Optional[PokemonView]:
    if p is None:
        return None
    return PokemonView(
        card_id=p.id,
        serial=p.serial,
        hp=p.hp,
        max_hp=p.maxHp,
        appear_this_turn=p.appearThisTurn,
        energies=list(p.energies),
        energy_cards=[_card_ref(c) for c in p.energyCards],
        tools=[_card_ref(c) for c in p.tools],
        pre_evolution=[_card_ref(c) for c in p.preEvolution],
    )


def _player_view(ps: PlayerState) -> PlayerView:
    active = ps.active[0] if ps.active else None
    return PlayerView(
        active=_pokemon_view(active),
        bench=[_pokemon_view(p) for p in ps.bench],
        bench_max=ps.benchMax,
        deck_count=ps.deckCount,
        discard=[_card_ref(c) for c in ps.discard],
        prize=[_card_ref(c) for c in ps.prize],
        hand_count=ps.handCount,
        hand=None if ps.hand is None else [_card_ref(c) for c in ps.hand],
        poisoned=ps.poisoned,
        burned=ps.burned,
        asleep=ps.asleep,
        paralyzed=ps.paralyzed,
        confused=ps.confused,
    )


def parse_state(state: State) -> GameState:
    your_index = state.yourIndex
    opp_index = 1 - your_index
    return GameState(
        turn=state.turn,
        turn_action_count=state.turnActionCount,
        your_index=your_index,
        first_player=state.firstPlayer,
        supporter_played=state.supporterPlayed,
        stadium_played=state.stadiumPlayed,
        energy_attached=state.energyAttached,
        retreated=state.retreated,
        result=state.result,
        stadium=[_card_ref(c) for c in state.stadium],
        you=_player_view(state.players[your_index]),
        opponent=_player_view(state.players[opp_index]),
    )


def _player_view_by_index(game_state: GameState, player_index: int) -> PlayerView:
    return game_state.you if player_index == game_state.your_index else game_state.opponent


def _resolve_pokemon(game_state: GameState, area, index, player_index) -> Optional[PokemonView]:
    if area is None or index is None or player_index is None:
        return None
    pv = _player_view_by_index(game_state, player_index)
    if area == AreaType.ACTIVE:
        return pv.active
    if area == AreaType.BENCH and 0 <= index < len(pv.bench):
        return pv.bench[index]
    return None


def _resolve_card(game_state: GameState, area, index, player_index,
                   deck_hint: Optional[list] = None) -> Optional[CardRef]:
    if area is None or index is None:
        return None
    if area == AreaType.STADIUM:
        zone = game_state.stadium
    elif area == AreaType.DECK:
        zone = deck_hint
    else:
        pv = _player_view_by_index(
            game_state, player_index if player_index is not None else game_state.your_index)
        if area == AreaType.HAND:
            zone = pv.hand
        elif area == AreaType.DISCARD:
            zone = pv.discard
        elif area == AreaType.PRIZE:
            zone = pv.prize
        else:
            zone = None
    if zone is None or not (0 <= index < len(zone)):
        return None
    return zone[index]


def decode_selection(select: SelectData, game_state: GameState) -> Selection:
    """Classify select.option into readable, still-index-preserving LegalOptions.

    Never invents legality -- `index` always matches the option's position in
    select.option, so decode failures (unresolvable card/target) only affect the
    human-readable fields, never which indices are pickable, nor min_count/max_count.
    """
    deck_hint = [_card_ref(c) for c in select.deck] if select.deck else None
    options = []
    for i, opt in enumerate(select.option):
        kind = _KIND_BY_OPTION_TYPE.get(opt.type, f"unknown_{opt.type}")
        card = None
        target = None
        if opt.type in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD,
                        OptionType.DISCARD, OptionType.ABILITY):
            card = _resolve_card(game_state, opt.area, opt.index, opt.playerIndex, deck_hint)
        elif opt.type == OptionType.PLAY:
            card = _resolve_card(game_state, AreaType.HAND, opt.index, game_state.your_index,
                                  deck_hint)
        elif opt.type in (OptionType.ATTACH, OptionType.EVOLVE):
            card = _resolve_card(game_state, opt.area, opt.index, game_state.your_index,
                                  deck_hint)
            target = _resolve_pokemon(game_state, opt.inPlayArea, opt.inPlayIndex,
                                       game_state.your_index)
        elif opt.type == OptionType.ENERGY:
            target = _resolve_pokemon(game_state, opt.area, opt.index, opt.playerIndex)
        options.append(LegalOption(index=i, kind=kind, raw=opt, card=card, target=target))
    return Selection(
        select_type=select.type,
        context=select.context,
        min_count=select.minCount,
        max_count=select.maxCount,
        options=options,
    )


def parse_obs(obs_dict: dict):
    """Returns (game_state, selection). Both are None only at the very first call
    (obs.select is None), when the caller must return the 60-card deck instead."""
    observation: Observation = to_observation_class(obs_dict)
    if observation.select is None or observation.current is None:
        return None, None
    game_state = parse_state(observation.current)
    selection = decode_selection(observation.select, game_state)
    return game_state, selection
