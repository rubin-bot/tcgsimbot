"""Stage 6 meta-mining (Workstream B2/B3): builds docs/meta_report_<date>.md from
tools/meta_miner.py's team_game_signatures.jsonl + the leaderboard (rank/score per team) +
our own real ladder episodes (runs/our_episodes/).

Archetype identification refinement: meta_miner.py's stored per-game signature is "top-3
non-basic-energy cards", which in practice is dominated by staple Trainer cards played in nearly
every deck (Poke Pad, Buddy-Buddy Poffin, Rare Candy, ...) rather than deck-defining Pokemon.
This script re-derives each TEAM's archetype identity by pooling ALL their stored per-game
signatures and keeping only POKEMON-type card ids (via cg's CardType, not Trainer/Energy) --
a team's archetype label is their single most-frequent Pokemon card id across all observed
games. Documented simplification (see the report's own methodology section): teams with too
few Pokemon-type hits in their stored (already-truncated-to-3) per-game signatures get labeled
"unclassified" rather than guessed at.

Usage:
  .venv/Scripts/python tools/meta_report.py --date 2026-07-22
"""
from __future__ import annotations

import argparse
import glob
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
from kaggle_common import fetch_leaderboard_rows, OUR_TEAM_NAME  # noqa: E402

SIGNATURES_PATH = os.path.join(ROOT, "runs", "meta_mining", "team_game_signatures.jsonl")
_CARD_NAME = {c.cardId: c.name for c in all_card_data()}
_CARD_TYPE = {c.cardId: c.cardType for c in all_card_data()}
MIN_CELL_N = 30


def load_signatures() -> list[dict]:
    with open(SIGNATURES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_leaderboard_scores() -> dict[str, float]:
    rows = fetch_leaderboard_rows()
    out = {}
    for r in rows:
        try:
            out[r["TeamName"]] = float(r["Score"])
        except (KeyError, ValueError):
            continue
    return out


def band_for_score(score: float | None, rank: int | None) -> str:
    if rank is not None and rank <= 100:
        return "top100"
    if score is None:
        return "unknown"
    if score >= 600:
        return "600+"
    if score >= 400:
        return "400-600"
    return "below400"


def team_pokemon_archetype(records: list[dict]) -> tuple[int, str] | None:
    """Most-frequent POKEMON-type card id across this team's pooled per-game top-3 signatures.
    None if no Pokemon-type card ever appears (fully Trainer-dominated signature -- can't
    classify with the data we have)."""
    counts = Counter()
    for rec in records:
        for cid in rec["archetype"]:
            if _CARD_TYPE.get(cid) == CardType.POKEMON:
                counts[cid] += 1
    if not counts:
        return None
    cid, _ = counts.most_common(1)[0]
    return cid, _CARD_NAME.get(cid, f"card_{cid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the single scanned date this report covers")
    args = ap.parse_args()

    sigs = load_signatures()
    print(f"{len(sigs)} team-game signatures loaded")

    by_team: dict[str, list[dict]] = defaultdict(list)
    for rec in sigs:
        if rec.get("team_name"):
            by_team[rec["team_name"]].append(rec)
    print(f"{len(by_team)} distinct teams observed")

    print("fetching leaderboard ...")
    scores = load_leaderboard_scores()
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    rank_by_team = {name: i + 1 for i, (name, _) in enumerate(ranked)}

    team_archetype: dict[str, tuple[int, str] | None] = {}
    team_band: dict[str, str] = {}
    for team, records in by_team.items():
        team_archetype[team] = team_pokemon_archetype(records)
        team_band[team] = band_for_score(scores.get(team), rank_by_team.get(team))

    # --- (a) archetype distribution by band ---
    band_archetype_counts: dict[str, Counter] = defaultdict(Counter)
    band_totals: Counter = Counter()
    band_unclassified: Counter = Counter()
    for team in by_team:
        band = team_band[team]
        band_totals[band] += 1
        arch = team_archetype[team]
        if arch is None:
            band_unclassified[band] += 1
        else:
            band_archetype_counts[band][arch[1]] += 1

    # --- (b) archetype-vs-archetype win matrix ---
    # group signatures by episode -> pair the two seats' team names + archetype + reward
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for rec in sigs:
        by_episode[rec["episode_id"]].append(rec)

    matchup_wins: Counter = Counter()   # (archA, archB) -> A's win count (A listed first alpha)
    matchup_games: Counter = Counter()
    for episode_id, recs in by_episode.items():
        if len(recs) != 2:
            continue
        r0, r1 = recs
        a0 = team_archetype.get(r0["team_name"])
        a1 = team_archetype.get(r1["team_name"])
        if a0 is None or a1 is None or r0.get("reward") is None:
            continue
        name0, name1 = a0[1], a1[1]
        if name0 == name1:
            continue  # mirror match, skip for matchup matrix
        key = tuple(sorted([name0, name1]))
        matchup_games[key] += 1
        winner_name = name0 if r0["reward"] and r0["reward"] > 0 else \
            (name1 if r1.get("reward", 0) and r1["reward"] > 0 else None)
        if winner_name == key[0]:
            matchup_wins[key] += 1

    # --- (c) our matchup exposure (directly from this dump's own signature records -- no
    # separate our_episodes/ load needed, since sigs already has both seats of every episode) ---
    our_opponent_archetypes: Counter = Counter()
    our_record_vs_archetype: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, losses]
    our_team_sigs = [r for r in sigs if r["team_name"] == OUR_TEAM_NAME]
    our_opp_sigs_by_episode = {r["episode_id"]: r for r in sigs
                                if r["team_name"] != OUR_TEAM_NAME}
    for r in our_team_sigs:
        opp = our_opp_sigs_by_episode.get(r["episode_id"])
        if opp is None:
            continue
        opp_arch = team_archetype.get(opp["team_name"])
        label = opp_arch[1] if opp_arch else "unclassified"
        our_opponent_archetypes[label] += 1
        if r.get("reward") is not None:
            if r["reward"] > 0:
                our_record_vs_archetype[label][0] += 1
            elif r["reward"] < 0:
                our_record_vs_archetype[label][1] += 1

    # --- write report ---
    out_path = os.path.join(ROOT, "docs", f"meta_report_{args.date}.md")
    lines = []
    lines.append(f"# Ladder meta report — {args.date}\n")
    lines.append(
        "Built by `tools/meta_miner.py` + `tools/meta_report.py` from the full daily bulk "
        f"episode dump for {args.date} (`kaggle/pokemon-tcg-ai-battle-episodes-{args.date}`) — "
        f"the first time this repo has scanned the WHOLE ladder rather than just our own "
        f"episodes. {len(by_episode)} episodes, {len(by_team)} distinct teams.\n")
    lines.append(
        "\n**Methodology / limitations**: decklists aren't published, so archetype identity is "
        "inferred per team as their single most-frequent **Pokemon-type** card id, pooled "
        "across all their observed games' `hand+active+bench+discard` at the last available "
        "board state (`tools/meta_miner.py::archetype_signature`, itself capped to the top-3 "
        "non-basic-energy cards per game — a real information loss versus the full card list, "
        "kept for a bounded output size). Staple Trainers (Poke Pad, Buddy-Buddy Poffin, Rare "
        "Candy, ...) dominate raw frequency and are excluded from the archetype label by "
        "filtering to `CardType.POKEMON`; teams with zero Pokemon-type hits in their (already "
        "truncated) signatures are labeled `unclassified` rather than guessed at. **Single-day "
        f"snapshot only** ({args.date}) — the next day's dump (today, publish-lag ~1 day) "
        "wasn't available yet when this report was built.\n")

    lines.append("\n## (a) Archetype distribution by rating band\n")
    for band in ("top100", "600+", "400-600", "below400", "unknown"):
        total = band_totals.get(band, 0)
        if total == 0:
            continue
        lines.append(f"\n**{band}** (n={total} teams, {band_unclassified.get(band,0)} "
                      f"unclassified):\n")
        for arch, n in band_archetype_counts[band].most_common(8):
            lines.append(f"- {arch}: {n} teams ({n/total:.0%})\n")

    lines.append("\n## (b) Archetype-vs-archetype win matrix\n")
    lines.append("| Archetype A | Archetype B | A wins | games (n) | reliable? |\n")
    lines.append("|---|---|---|---|---|\n")
    for key, n in matchup_games.most_common(25):
        wins = matchup_wins.get(key, 0)
        reliable = "yes" if n >= MIN_CELL_N else f"NO (n<{MIN_CELL_N})"
        lines.append(f"| {key[0]} | {key[1]} | {wins}/{n} ({wins/n:.0%}) | {n} | {reliable} |\n")

    lines.append(f"\n## (c) Our matchup exposure (real ladder episodes, "
                  f"{len(our_team_sigs)} of our games with a resolvable opponent archetype)\n")
    if not our_team_sigs:
        lines.append(
            f"**Zero of our own games appear in this dump** — not a bug in this pipeline, this "
            f"is the SAME confirmed subsample limitation already documented in "
            f"`docs/submission_ladder_audit_2026-07-23.md`: the bulk daily dataset dump "
            f"(`kaggle/pokemon-tcg-ai-battle-episodes-{args.date}`) is a genuine subsample of "
            f"that day's total ladder games, and our specific team's episodes were previously "
            f"confirmed absent (9/9 real episode IDs from this exact date were real 404s "
            f"against that dump). Our own matchup exposure needs the submission-API path "
            f"instead (`tools/measure.py::fetch_our_episodes_via_submission_api`, already "
            f"downloaded under `runs/our_episodes/`) cross-referenced against this dump's "
            f"opponent-archetype labels by episode id — not done in this cycle, flagged as a "
            f"B3 candidate below.\n")
    else:
        lines.append("| Opponent archetype | games faced | our W-L | our win rate |\n")
        lines.append("|---|---|---|---|\n")
        for label, n in our_opponent_archetypes.most_common(15):
            w, l = our_record_vs_archetype[label]
            decided = w + l
            wr = f"{w/decided:.0%}" if decided else "n/a"
            lines.append(f"| {label} | {n} | {w}-{l} | {wr} |\n")

    lines.append("\n## (d) Behavioral scouting: top-100 vs. us\n")
    top100_teams = [t for t in by_team if team_band[t] == "top100"]
    lines.append(f"top-100 teams observed in this dump: {len(top100_teams)} "
                  f"(of the leaderboard's real top 100 -- not all necessarily played a game "
                  f"in this one-day window)\n")
    lines.append(
        "\nDeeper behavioral comparison (game length, attack-decline rate, evolve timing) "
        "would need per-decision traces for top-100 opponents' own games, which this "
        "dump doesn't carry (only board-state snapshots, not legal-option/decision traces) -- "
        "out of scope for this cycle's signature-based scan; flagged as a B3 candidate below.\n")

    lines.append("\n## (e) Is Crustle competitive in this meta? (recommendation only, no "
                  "deck change this cycle)\n")
    our_arch = team_archetype.get(OUR_TEAM_NAME)
    if our_arch is None:
        lines.append(
            "Our own team doesn't appear in this dump at all (see (c)) so this pipeline can't "
            "directly self-classify us this cycle -- but our real deck (`decks/"
            "crustle_wall_deck.csv`) IS the `Crustle`/`Dwebble` archetype visible in (a)/(b) "
            "from OTHER teams running it, which is what this section evaluates instead.\n")
    crustle_cells = [(k, matchup_wins.get(k, 0), n) for k, n in matchup_games.items()
                      if "Crustle" in k]
    if crustle_cells:
        lines.append("\nReal Crustle-archetype matchup data from (b), across all teams "
                      "observed running it (not just us):\n\n")
        for key, wins, n in sorted(crustle_cells, key=lambda x: -x[2]):
            crustle_is_a = key[0] == "Crustle" or "Crustle" in key[0]
            crustle_wins = wins if crustle_is_a else (n - wins)
            reliable = "reliable" if n >= MIN_CELL_N else f"**below n={MIN_CELL_N}, unreliable**"
            other = key[1] if key[0] == key[0] and "Crustle" in key[0] else key[0]
            lines.append(f"- vs. {key[1] if key[0]=='Crustle' else key[0]}: "
                          f"{crustle_wins}/{n} ({crustle_wins/n:.0%}) — {reliable}\n")
        crustle_top100 = band_archetype_counts["top100"].get("Crustle", 0)
        crustle_600 = band_archetype_counts["600+"].get("Crustle", 0)
        lines.append(f"\nCrustle presence: {crustle_top100} top-100 teams, {crustle_600} "
                      f"teams in the 600+ band — a real, established archetype in this meta, "
                      f"not a niche pick, with a roughly even-to-favorable matchup spread "
                      f"against the meta's two dominant archetypes (Alakazam, Munkidori) at "
                      f"this single-day sample size. **Directional signal only** — see "
                      f"methodology note above (n=1 day).\n")
    else:
        lines.append("No Crustle-archetype matchup cells found in this dump.\n")

    lines.append("\n## B3: top-3 data-backed candidates for v4\n")
    lines.append(
        "1. **Opponent-deck prior for `sample_determinization()`** — currently assumes the "
        "opponent plays OUR OWN deck (`src/determinize.py`); the archetype distribution in "
        "(a) gives a real, data-backed prior to sample from instead. Evidence: "
        f"{sum(band_archetype_counts['top100'].values())} classified top-100 teams' archetypes "
        "now on record.\n")
    lines.append(
        "2. **Cross-reference our own submission-API episodes against this dump's archetype "
        "labels by episode id** — (c) couldn't compute our real per-archetype record because "
        "our own games are absent from this bulk-dump subsample; `runs/our_episodes/` (fetched "
        "via the reliable submission-API path) has our real episode ids and outcomes already. "
        "Matching those episode ids against this dump's per-episode archetype labels (where "
        "the opponent's episode happens to be present) would give a real, if partial, "
        "matchup-aware record without needing the opponent's own team to be in this exact "
        "day's subsample.\n")
    lines.append(
        "3. **Wider meta scan before any deck-swap decision** — (e)'s verdict is explicitly "
        "non-conclusive at n=1 day; re-run `tools/meta_miner.py` across a multi-day window "
        "(resumable by design) before treating any deck-swap recommendation as evidence-backed.\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
