"""Card-data layer: joins the cg engine's CardData/Attack with EN_Card_Data.csv effect text.

Pure data access, no strategy. Development/analysis tool only — the running agent reasons
over the engine's own CardData/Attack objects directly and never parses this CSV at
inference time.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from sdk_path import ensure_cg_importable

ensure_cg_importable()

from cg.api import CardType, EnergyType, Skill, all_attack, all_card_data  # noqa: E402

_DEFAULT_CSV_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "pokemon-tcg-ai-battle-challenge-strategy", "EN_Card_Data.csv",
))


@dataclass
class CardAttack:
    attack_id: int
    name: str
    cost: list[EnergyType]
    damage: int
    text: str  # engine's short effect text
    effect_text: str  # EN_Card_Data.csv's fuller natural-language explanation


@dataclass
class Card:
    card_id: int
    name: str
    card_type: CardType
    hp: int
    weakness: EnergyType | None
    resistance: EnergyType | None
    retreat_cost: int
    energy_type: EnergyType
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    tera: bool
    ace_spec: bool
    evolves_from: str | None
    skills: list[Skill]
    attacks: list[CardAttack]

    @property
    def stage(self) -> str:
        if self.basic:
            return "basic"
        if self.stage1:
            return "stage1"
        if self.stage2:
            return "stage2"
        return "n/a"


def _load_csv_rows_by_card_id(csv_path: str) -> dict[int, list[dict]]:
    """Group CSV rows by Card ID; multi-attack cards share one Card ID across rows."""
    rows_by_id: dict[int, list[dict]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            card_id = int(row["Card ID"])
            rows_by_id.setdefault(card_id, []).append(row)
    return rows_by_id


def load_card_index(csv_path: str | None = None) -> dict[int, Card]:
    """Build a card_id -> Card index joining engine CardData/Attack with CSV effect text."""
    csv_path = csv_path or _DEFAULT_CSV_PATH
    attacks_by_id = {a.attackId: a for a in all_attack()}
    csv_rows_by_card = _load_csv_rows_by_card_id(csv_path)

    index: dict[int, Card] = {}
    for cd in all_card_data():
        csv_rows = csv_rows_by_card.get(cd.cardId, [])
        effect_by_move_name = {
            row["Move Name"]: row["Effect Explanation"]
            for row in csv_rows
            if row.get("Move Name")
        }

        card_attacks = []
        for attack_id in cd.attacks:
            eng = attacks_by_id.get(attack_id)
            if eng is None:
                continue
            card_attacks.append(CardAttack(
                attack_id=eng.attackId,
                name=eng.name,
                cost=list(eng.energies),
                damage=eng.damage,
                text=eng.text,
                effect_text=effect_by_move_name.get(eng.name, eng.text),
            ))

        index[cd.cardId] = Card(
            card_id=cd.cardId,
            name=cd.name,
            card_type=CardType(cd.cardType),
            hp=cd.hp,
            weakness=cd.weakness,
            resistance=cd.resistance,
            retreat_cost=cd.retreatCost,
            energy_type=cd.energyType,
            basic=cd.basic,
            stage1=cd.stage1,
            stage2=cd.stage2,
            ex=cd.ex,
            mega_ex=cd.megaEx,
            tera=cd.tera,
            ace_spec=cd.aceSpec,
            evolves_from=cd.evolvesFrom,
            skills=cd.skills,
            attacks=card_attacks,
        )
    return index
