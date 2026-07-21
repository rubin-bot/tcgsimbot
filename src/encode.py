"""Feature encoders for the learned agent.

Turns the info-hidden `obs.GameState` into fixed-length float vectors the policy/value net
consumes. Two encoders:

  * `encode_state(gs)`   -> np.ndarray[STATE_DIM]   : whole-position features.
  * `encode_option(gs, lo)` -> np.ndarray[OPTION_DIM]: one legal option's features.

INFORMATION HIDING (first-class, tested in tests/test_encode.py): this module only ever reads
fields that are genuinely visible to the deciding player. `obs.parse_obs` already nulls the
opponent's hand (`opponent.hand is None`), and `encode_state` encodes the opponent with
`include_hand=False`, so no feature can derive from hidden opponent cards. The future MCTS/self-
play code reuses these encoders, so this invariant must survive edits.

Card identity is encoded via hand-crafted *stats* (HP, type, stage, ex, attack damage/cost),
not one-hot card ids -- this keeps the vector small, float-only (so the torch-free submission
forward can reuse it), and card-agnostic so the net generalizes rather than memorizing ids.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import AreaType, CardType, EnergyType, all_attack, all_card_data  # noqa: E402

from obs import GameState, LegalOption, PlayerView, PokemonView  # noqa: E402

_CARD_BY_ID = {c.cardId: c for c in all_card_data()}
_ATTACK_BY_ID = {a.attackId: a for a in all_attack()}

# --- normalization constants (rough maxima; keeps features ~O(1)) ---
_HP_NORM = 340.0        # max HP in the SV pool is ~330
_DMG_NORM = 340.0
_ENERGY_NORM = 6.0      # a single Pokemon rarely carries >6 energy
_COUNT_NORM = 60.0      # deck / discard / hand sizes
_TURN_NORM = 50.0

_NUM_ENERGY = 12        # EnergyType 0..11
_NUM_CARDTYPE = 7       # CardType 0..6
MAX_BENCH = 5

# Option `kind` labels (mirror obs._KIND_BY_OPTION_TYPE values); index = one-hot position.
_KINDS = [
    "number", "yes", "no", "card", "tool_card", "energy_card", "energy", "play", "attach",
    "evolve", "ability", "discard", "retreat", "attack", "end", "skill", "special_condition",
]
_KIND_INDEX = {k: i for i, k in enumerate(_KINDS)}

# --- layout dims (kept in sync with the builders below; asserted at import) ---
_SLOT_DIM = 26          # per Pokemon slot (active or one bench slot)
_HAND_DIM = _NUM_CARDTYPE + 1   # own-hand type histogram + hand_count (opp: all-zero hist)
_PLAYER_COUNTS = 3      # prize remaining, deck count, discard count
_PLAYER_DIM = (1 + MAX_BENCH) * _SLOT_DIM + _HAND_DIM + _PLAYER_COUNTS
_GLOBAL_DIM = 8
STATE_DIM = _GLOBAL_DIM + 2 * _PLAYER_DIM
OPTION_DIM = len(_KINDS) + 15


def _stage_flags(card_id: int) -> tuple[float, float, float]:
    c = _CARD_BY_ID.get(card_id)
    if c is None:
        return 0.0, 0.0, 0.0
    return float(c.basic), float(c.stage1), float(c.stage2)


def _is_ex(card_id: int) -> float:
    c = _CARD_BY_ID.get(card_id)
    return float(bool(c and c.ex))


def _slot_feats(pv: Optional[PokemonView], status: tuple[float, ...]) -> list[float]:
    """26 features for one Pokemon slot. `status` (len 5) only meaningful for the active slot;
    pass zeros for bench slots."""
    if pv is None:
        return [0.0] * _SLOT_DIM
    max_hp = max(pv.max_hp, 1)
    energy_hist = [0.0] * _NUM_ENERGY
    for e in pv.energies:
        idx = int(e)
        if 0 <= idx < _NUM_ENERGY:
            energy_hist[idx] += 1.0
    energy_hist = [x / _ENERGY_NORM for x in energy_hist]
    basic, s1, s2 = _stage_flags(pv.card_id)
    feats = [
        1.0,                                   # present
        pv.hp / max_hp,                        # hp fraction
        pv.max_hp / _HP_NORM,                  # max hp
        len(pv.energies) / _ENERGY_NORM,       # total energy
        *energy_hist,                          # 12: energy by type
        len(pv.tools) / 2.0,                   # tool count
        basic, s1, s2,                         # 3: evolution stage
        _is_ex(pv.card_id),                    # ex flag
        *status,                               # 5: status conditions
    ]
    assert len(feats) == _SLOT_DIM, (len(feats), _SLOT_DIM)
    return feats


def _hand_feats(player: PlayerView, include_hand: bool) -> list[float]:
    """Own hand: type histogram + count. Opponent: hand is hidden -> zeros + revealed count."""
    hist = [0.0] * _NUM_CARDTYPE
    if include_hand and player.hand is not None:
        for ref in player.hand:
            c = _CARD_BY_ID.get(ref.card_id)
            if c is not None and 0 <= int(c.cardType) < _NUM_CARDTYPE:
                hist[int(c.cardType)] += 1.0
        hist = [x / _COUNT_NORM for x in hist]
    return [*hist, player.hand_count / _COUNT_NORM]


def _player_feats(player: PlayerView, include_hand: bool) -> list[float]:
    status = (
        float(player.poisoned), float(player.burned), float(player.asleep),
        float(player.paralyzed), float(player.confused),
    )
    feats: list[float] = []
    feats += _slot_feats(player.active, status)          # active carries status
    for i in range(MAX_BENCH):
        bench = player.bench[i] if i < len(player.bench) else None
        feats += _slot_feats(bench, (0.0, 0.0, 0.0, 0.0, 0.0))
    feats += _hand_feats(player, include_hand)
    prize_remaining = sum(1 for _ in player.prize)
    feats += [
        prize_remaining / 6.0,
        player.deck_count / _COUNT_NORM,
        len(player.discard) / _COUNT_NORM,
    ]
    assert len(feats) == _PLAYER_DIM, (len(feats), _PLAYER_DIM)
    return feats


def encode_state(gs: GameState) -> np.ndarray:
    """Whole-position features from the perspective of the deciding player (`gs.you`).
    Opponent is encoded with include_hand=False so no hidden card ever enters the vector."""
    am_i_first = 1.0 if gs.first_player == gs.your_index else 0.0
    glob = [
        min(gs.turn, _TURN_NORM) / _TURN_NORM,
        min(gs.turn_action_count, 20) / 20.0,
        am_i_first,
        float(gs.supporter_played),
        float(gs.stadium_played),
        float(gs.energy_attached),
        float(gs.retreated),
        1.0 if gs.stadium else 0.0,
    ]
    assert len(glob) == _GLOBAL_DIM
    vec = glob + _player_feats(gs.you, include_hand=True) + _player_feats(gs.opponent, include_hand=False)
    return np.asarray(vec, dtype=np.float32)


def _best_attack_damage(card_id: int) -> int:
    c = _CARD_BY_ID.get(card_id)
    if not c or not c.attacks:
        return 0
    return max((_ATTACK_BY_ID[a].damage for a in c.attacks if a in _ATTACK_BY_ID), default=0)


def encode_option(gs: GameState, lo: LegalOption) -> np.ndarray:
    """Per-option features: kind one-hot + numerics. Includes the raw positional/target fields
    (which board slot, which source card, energy/tool index) so that options of the SAME kind --
    e.g. "attach energy to bench slot 0" vs "slot 1" -- get DISTINCT features; otherwise the
    policy head cannot express a preference between them. Gives the net raw signals (attack
    damage, target HP) rather than derived judgments like 'is lethal' -- no strategy baked in."""
    raw = lo.raw
    kind_oh = [0.0] * len(_KINDS)
    if lo.kind in _KIND_INDEX:
        kind_oh[_KIND_INDEX[lo.kind]] = 1.0

    atk_dmg = atk_cost = 0.0
    if lo.kind == "attack" and raw.attackId is not None:
        atk = _ATTACK_BY_ID.get(raw.attackId)
        if atk is not None:
            atk_dmg = atk.damage / _DMG_NORM
            atk_cost = len(atk.energies) / _ENERGY_NORM

    # target Pokemon (of an attach/evolve/energy option): identity by slot + state
    target_present = target_hp_frac = target_energy = target_slot = 0.0
    if lo.target is not None:
        target_present = 1.0
        target_hp_frac = lo.target.hp / max(lo.target.max_hp, 1)
        target_energy = len(lo.target.energies) / _ENERGY_NORM
    if raw.inPlayArea == AreaType.ACTIVE:
        target_slot = 0.0
    elif raw.inPlayArea == AreaType.BENCH and raw.inPlayIndex is not None:
        target_slot = (raw.inPlayIndex + 1) / 6.0

    card_hp = card_is_ex = card_stage = 0.0
    if lo.card is not None:
        c = _CARD_BY_ID.get(lo.card.card_id)
        if c is not None:
            card_hp = c.hp / _HP_NORM
            card_is_ex = float(bool(c.ex))
            basic, s1, s2 = _stage_flags(lo.card.card_id)
            card_stage = 0.5 * s1 + 1.0 * s2

    # raw engine fields that disambiguate otherwise-identical options
    src_index = (raw.index or 0) / 20.0
    inplay_idx = (raw.inPlayIndex or 0) / 6.0
    energy_idx = (raw.energyIndex or 0) / 6.0
    tool_idx = (raw.toolIndex or 0) / 2.0
    number = (raw.number or 0) / 20.0
    area_code = (int(raw.area) if raw.area is not None else 0) / 12.0

    numerics = [
        atk_dmg, atk_cost,
        target_present, target_hp_frac, target_energy, target_slot,
        card_hp, card_is_ex, card_stage,
        src_index, inplay_idx, energy_idx, tool_idx, number, area_code,
    ]
    vec = kind_oh + numerics
    assert len(vec) == OPTION_DIM, (len(vec), OPTION_DIM)
    return np.asarray(vec, dtype=np.float32)
