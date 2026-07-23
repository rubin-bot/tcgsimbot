"""Master-study Workstream C1: reconstructs an APPROXIMATE 60-card decklist for a ladder
archetype (Alakazam, Munkidori) from runs/meta_mining/team_game_signatures.jsonl, for sparring
purposes only -- NOT a claim of competitive accuracy. See the printed/written assumptions doc
for exactly what's real data vs. inferred.

What's real: which cards appear, and how often, pooled across the top 3 real teams (by game
count) for that archetype -- confirmed distinct, real, frequently-played cards.
What's inferred (stated explicitly, not hidden): COPY COUNTS. The mined signature is only the
top-3-non-basic-energy-cards PER GAME (tools/meta_miner.py), so exact playset sizes are not
recoverable from this data source at all -- assigned here by a simple, documented frequency-tier
rule (relative to the pooled team-game count, not an arbitrary guess): cards seen in >=70% of
games -> 4 copies (near-universal staple), 40-70% -> 3, 15-40% -> 2, <15% -> 1. Basic energy
(never in the signature at all, filtered out by meta_miner.py itself) is filled in afterward,
by the archetype's own real energy_type(s) (weighted by each qualifying Pokemon's own frequency
when more than one energy type is needed), to reach exactly 60 cards.

Usage:
  .venv/Scripts/python tools/reconstruct_archetype_deck.py --archetype Alakazam --top-teams 3
  .venv/Scripts/python tools/reconstruct_archetype_deck.py --archetype Munkidori --top-teams 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sdk_path import ensure_cg_importable  # noqa: E402
ensure_cg_importable()
from cg.api import CardType, all_card_data  # noqa: E402
from meta_report import load_signatures, team_pokemon_archetype  # noqa: E402

_CARD_NAME = {c.cardId: c.name for c in all_card_data()}
_CARD_TYPE = {c.cardId: c.cardType for c in all_card_data()}
_CARD_ENERGY_TYPE = {c.cardId: c.energyType for c in all_card_data()}
BASIC_ENERGY_BY_TYPE = {c.energyType: c.cardId for c in all_card_data()
                         if c.cardType == CardType.BASIC_ENERGY}
DECK_SIZE = 60


def copies_for_frequency(frac: float) -> int:
    if frac >= 0.70:
        return 4
    if frac >= 0.40:
        return 3
    if frac >= 0.15:
        return 2
    return 1


def build_deck(archetype: str, top_teams: int) -> dict:
    sigs = load_signatures()
    by_team: dict[str, list[dict]] = defaultdict(list)
    for rec in sigs:
        if rec.get("team_name"):
            by_team[rec["team_name"]].append(rec)

    teams = []
    for team, recs in by_team.items():
        arch = team_pokemon_archetype(recs)
        if arch and arch[1] == archetype:
            teams.append((team, len(recs)))
    teams.sort(key=lambda x: -x[1])
    chosen_teams = teams[:top_teams]

    counts = Counter()
    total_games = 0
    for team, n in chosen_teams:
        total_games += n
        for rec in by_team[team]:
            for cid in rec["archetype"]:
                counts[cid] += 1

    cards = []
    for cid, n in counts.most_common():
        frac = n / total_games if total_games else 0
        copies = copies_for_frequency(frac)
        cards.append({"card_id": cid, "name": _CARD_NAME.get(cid, f"card_{cid}"),
                       "card_type": int(_CARD_TYPE.get(cid, -1)), "n_games_seen": n,
                       "frac_of_games": round(frac, 3), "assigned_copies": copies})

    # Cap total non-energy cards at 60 by dropping lowest-frequency entries if needed (rare with
    # only ~20-25 distinct cards recoverable from a top-3-per-game signature).
    total_non_energy = sum(c["assigned_copies"] for c in cards)
    dropped = []
    while total_non_energy > DECK_SIZE and cards:
        removed = cards.pop()  # lowest frequency (Counter.most_common is descending)
        dropped.append(removed)
        total_non_energy -= removed["assigned_copies"]

    energy_slots = DECK_SIZE - total_non_energy
    # Weight basic-energy-type mix by the summed frequency of Pokemon needing each energy type
    # among the cards actually included -- e.g. a mixed Munkidori(Psychic)/Grimmsnarl(Darkness)
    # deck gets a real, data-weighted split rather than an arbitrary 50/50 guess.
    energy_type_weight = Counter()
    for c in cards:
        if c["card_type"] == int(CardType.POKEMON):
            et = _CARD_ENERGY_TYPE.get(c["card_id"])
            if et in BASIC_ENERGY_BY_TYPE:
                energy_type_weight[et] += c["n_games_seen"]
    if not energy_type_weight:
        energy_type_weight = Counter({list(BASIC_ENERGY_BY_TYPE)[0]: 1})

    total_weight = sum(energy_type_weight.values())
    energy_cards = []
    allocated = 0
    energy_items = sorted(energy_type_weight.items(), key=lambda kv: -kv[1])
    for i, (et, w) in enumerate(energy_items):
        if i == len(energy_items) - 1:
            n = energy_slots - allocated  # remainder to the largest bucket-ordered-last fix
        else:
            n = round(energy_slots * w / total_weight)
        allocated += n
        cid = BASIC_ENERGY_BY_TYPE[et]
        energy_cards.append({"card_id": cid, "name": _CARD_NAME.get(cid, f"card_{cid}"),
                              "copies": n})

    deck_list = []
    for c in cards:
        deck_list.extend([c["card_id"]] * c["assigned_copies"])
    for e in energy_cards:
        deck_list.extend([e["card_id"]] * e["copies"])

    assert len(deck_list) == DECK_SIZE, f"deck size {len(deck_list)} != {DECK_SIZE}"

    return {
        "archetype": archetype,
        "source_teams": [{"team_name": t, "n_games": n} for t, n in chosen_teams],
        "total_games_pooled": total_games,
        "cards": cards,
        "dropped_low_frequency": dropped,
        "energy_cards": energy_cards,
        "deck_list": deck_list,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", required=True)
    ap.add_argument("--top-teams", type=int, default=3)
    args = ap.parse_args()

    result = build_deck(args.archetype, args.top_teams)

    slug = args.archetype.lower()
    deck_csv_path = os.path.join(ROOT, "decks", f"{slug}_sparring_deck.csv")
    with open(deck_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(cid) for cid in result["deck_list"]) + "\n")
    print(f"wrote {deck_csv_path} ({len(result['deck_list'])} cards)")

    doc_path = os.path.join(ROOT, "decks", f"{slug}_sparring_deck_ASSUMPTIONS.md")
    lines = [f"# {args.archetype} sparring deck -- reconstruction assumptions\n\n"]
    lines.append(
        "**NOT a real decklist.** Built for master-study sparring purposes only "
        "(docs/master_study_*.md, Workstream C). Card PRESENCE is real, mined data "
        "(runs/meta_mining/team_game_signatures.jsonl); COPY COUNTS are inferred via a simple "
        "frequency-tier rule since the mining signature only stores the top-3-per-game cards, "
        "never full decklists. See tools/reconstruct_archetype_deck.py's docstring for the "
        "exact rule.\n\n")
    team_summary = ", ".join(f"{t['team_name']} ({t['n_games']} games)"
                              for t in result["source_teams"])
    lines.append(f"Pooled from top {args.top_teams} real teams by game count: {team_summary} "
                 f"-- {result['total_games_pooled']} total games.\n\n")
    lines.append("## Cards (real presence, inferred copy count)\n\n")
    lines.append("| Card | seen in N games | frac of pooled games | assigned copies |\n")
    lines.append("|---|---|---|---|\n")
    for c in result["cards"]:
        lines.append(f"| {c['name']} | {c['n_games_seen']} | {c['frac_of_games']:.1%} | "
                      f"{c['assigned_copies']} |\n")
    if result["dropped_low_frequency"]:
        lines.append("\n**Dropped (lowest-frequency, to fit the 60-card cap):** " +
                      ", ".join(c["name"] for c in result["dropped_low_frequency"]) + "\n")
    lines.append("\n## Basic energy fill (never in the mined signature -- meta_miner.py "
                  "excludes it by design)\n\n")
    lines.append("Split by each energy-needing Pokemon's own real observed frequency, not "
                  "an arbitrary guess:\n\n")
    for e in result["energy_cards"]:
        lines.append(f"- {e['copies']}x {e['name']}\n")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {doc_path}")


if __name__ == "__main__":
    main()
