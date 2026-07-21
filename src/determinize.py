"""Determinization: sample a plausible hidden world for MCTS search.

The agent cannot see the opponent's hand, either player's deck order, or the face-down prizes
(its own prizes are hidden from it too, per PTCG rules). Before every call to
`cg.api.search_begin` we must fill those hidden zones with a *guess*, and we must guess only
from information the agent legitimately has -- never from ground truth.

Prior (chosen with the user): sample from the CURRENT DECK DISTRIBUTION. In the fixed-deck
mirror slice both players run the same known 60-card list, so:
  * our own hidden cards (deck order + face-down prizes) = our 60-card list minus what we can
    already see of our own cards, partitioned into deck/prize by the right counts.
  * the opponent's hidden cards = that same 60-card list minus the opponent's visible cards,
    partitioned into deck/hand/prize by the right counts.

Only *counts* are strictly enforced by the engine, so we guarantee exact counts (padding from
the deck pool if our visible-card accounting leaves us short); the specific guessed cards are
plausible but not claimed to be exact -- that's the whole point of determinization. Because we
resample every simulation, search explores different plausible worlds run to run.
"""

from __future__ import annotations

import random
from collections import Counter

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import all_card_data  # noqa: E402

from obs import GameState, PlayerView  # noqa: E402

_CARD = {c.cardId: c for c in all_card_data()}


def _pokemon_card_ids(pv) -> list[int]:
    if pv is None:
        return []
    ids = [pv.card_id]
    ids += [c.card_id for c in pv.energy_cards]
    ids += [c.card_id for c in pv.tools]
    ids += [c.card_id for c in pv.pre_evolution]
    return ids


def _visible_ids(player: PlayerView, include_hand: bool) -> list[int]:
    """Card ids the deciding player can legitimately see for this player.
    Face-down prizes are None -> contribute nothing; opponent hand is None -> skipped."""
    ids: list[int] = []
    ids += _pokemon_card_ids(player.active)
    for b in player.bench:
        ids += _pokemon_card_ids(b)
    ids += [c.card_id for c in player.discard]
    ids += [c.card_id for c in player.prize if c is not None]  # only revealed prizes
    if include_hand and player.hand is not None:
        ids += [c.card_id for c in player.hand]
    return ids


def _unknown_pool(deck_list: list[int], visible: list[int], need: int) -> list[int]:
    """Deck multiset minus visible cards, shuffled; padded from the deck pool if short so the
    caller always has at least `need` cards to hand the engine (counts are what it enforces)."""
    remaining = Counter(deck_list) - Counter(visible)
    pool: list[int] = list(remaining.elements())
    random.shuffle(pool)
    while len(pool) < need:
        pool.append(random.choice(deck_list))
    return pool


def _basic_pokemon_id(candidates: list[int], deck_list: list[int]) -> int:
    for cid in candidates:
        c = _CARD.get(cid)
        if c and c.basic and c.hp > 0:  # a Basic Pokemon
            return cid
    for cid in deck_list:
        c = _CARD.get(cid)
        if c and c.basic and c.hp > 0:
            return cid
    return deck_list[0]


def sample_determinization(gs: GameState, your_deck_list: list[int],
                           opp_deck_list: list[int] | None = None) -> dict:
    """Return the six hidden-zone predictions search_begin expects, sampled from the deck
    distribution. `opp_deck_list` defaults to the same list (mirror-match assumption)."""
    if opp_deck_list is None:
        opp_deck_list = your_deck_list

    you, opp = gs.you, gs.opponent

    # --- our own hidden cards: deck order + face-down prizes ---
    your_prize_count = len(you.prize)
    your_need = you.deck_count + your_prize_count
    your_pool = _unknown_pool(your_deck_list, _visible_ids(you, include_hand=True), your_need)
    your_deck = your_pool[:you.deck_count]
    your_prize = your_pool[you.deck_count:you.deck_count + your_prize_count]

    # --- opponent hidden cards: deck order + hand + face-down prizes ---
    opp_prize_count = len(opp.prize)
    opp_need = opp.deck_count + opp.hand_count + opp_prize_count
    opp_pool = _unknown_pool(opp_deck_list, _visible_ids(opp, include_hand=False), opp_need)
    opp_deck = opp_pool[:opp.deck_count]
    opp_hand = opp_pool[opp.deck_count:opp.deck_count + opp.hand_count]
    opp_prize = opp_pool[opp.deck_count + opp.hand_count:opp_need]

    # --- opponent active only if it is face-down (unknown), e.g. during setup ---
    opp_active: list[int] = []
    if opp.active is None:
        opp_active = [_basic_pokemon_id(opp_deck, opp_deck_list)]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": opp_prize,
        "opponent_hand": opp_hand,
        "opponent_active": opp_active,
    }
