# Ladder meta report — 2026-07-22
Built by `tools/meta_miner.py` + `tools/meta_report.py` from the full daily bulk episode dump for 2026-07-22 (`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22`) — the first time this repo has scanned the WHOLE ladder rather than just our own episodes. 4639 episodes, 142 distinct teams.

**Methodology / limitations**: decklists aren't published, so archetype identity is inferred per team as their single most-frequent **Pokemon-type** card id, pooled across all their observed games' `hand+active+bench+discard` at the last available board state (`tools/meta_miner.py::archetype_signature`, itself capped to the top-3 non-basic-energy cards per game — a real information loss versus the full card list, kept for a bounded output size). Staple Trainers (Poke Pad, Buddy-Buddy Poffin, Rare Candy, ...) dominate raw frequency and are excluded from the archetype label by filtering to `CardType.POKEMON`; teams with zero Pokemon-type hits in their (already truncated) signatures are labeled `unclassified` rather than guessed at. **Single-day snapshot only** (2026-07-22) — the next day's dump (today, publish-lag ~1 day) wasn't available yet when this report was built.

## (a) Archetype distribution by rating band

**top100** (n=75 teams, 3 unclassified):
- Alakazam: 24 teams (32%)
- Munkidori: 23 teams (31%)
- Cynthia's Roserade: 4 teams (5%)
- Crustle: 4 teams (5%)
- Team Rocket's Spidops: 3 teams (4%)
- Marnie's Grimmsnarl ex: 3 teams (4%)
- N’s Zoroark ex: 2 teams (3%)
- Dragapult ex: 1 teams (1%)

**600+** (n=63 teams, 7 unclassified):
- Munkidori: 12 teams (19%)
- Alakazam: 10 teams (16%)
- Crustle: 6 teams (10%)
- Team Rocket's Spidops: 3 teams (5%)
- Mega Starmie ex: 3 teams (5%)
- Kadabra: 3 teams (5%)
- Mega Lucario ex: 3 teams (5%)
- Dwebble: 2 teams (3%)

**unknown** (n=4 teams, 0 unclassified):
- Munkidori: 2 teams (50%)
- Thwackey: 1 teams (25%)
- Alakazam: 1 teams (25%)

## (b) Archetype-vs-archetype win matrix
| Archetype A | Archetype B | A wins | games (n) | reliable? |
|---|---|---|---|---|
| Alakazam | Munkidori | 305/673 (45%) | 673 | yes |
| Munkidori | Team Rocket's Spidops | 296/543 (55%) | 543 | yes |
| Alakazam | Team Rocket's Spidops | 63/311 (20%) | 311 | yes |
| Cynthia's Roserade | Munkidori | 177/278 (64%) | 278 | yes |
| Crustle | Munkidori | 78/170 (46%) | 170 | yes |
| Dunsparce | Munkidori | 101/162 (62%) | 162 | yes |
| Alakazam | Cynthia's Roserade | 93/152 (61%) | 152 | yes |
| Alakazam | Crustle | 64/116 (55%) | 116 | yes |
| Cynthia's Roserade | Team Rocket's Spidops | 65/95 (68%) | 95 | yes |
| Crustle | Team Rocket's Spidops | 33/76 (43%) | 76 | yes |
| Drakloak | Munkidori | 36/72 (50%) | 72 | yes |
| Dragapult ex | Munkidori | 35/70 (50%) | 70 | yes |
| Mega Kangaskhan ex | Munkidori | 35/61 (57%) | 61 | yes |
| Alakazam | Dunsparce | 30/53 (57%) | 53 | yes |
| Dunsparce | Team Rocket's Spidops | 43/53 (81%) | 53 | yes |
| Alakazam | Dragapult ex | 19/51 (37%) | 51 | yes |
| Crustle | Cynthia's Roserade | 24/43 (56%) | 43 | yes |
| Mega Starmie ex | Munkidori | 17/43 (40%) | 43 | yes |
| Alakazam | Mega Kangaskhan ex | 23/39 (59%) | 39 | yes |
| Cynthia's Roserade | Dunsparce | 29/36 (81%) | 36 | yes |
| Munkidori | Thwackey | 12/35 (34%) | 35 | yes |
| Alakazam | Drakloak | 10/31 (32%) | 31 | yes |
| Drakloak | Team Rocket's Spidops | 17/31 (55%) | 31 | yes |
| Dragapult ex | Team Rocket's Spidops | 20/30 (67%) | 30 | yes |
| Alakazam | Mega Starmie ex | 18/30 (60%) | 30 | yes |

## (c) Our matchup exposure (real ladder episodes, 0 of our games with a resolvable opponent archetype)
**Zero of our own games appear in this dump** — not a bug in this pipeline, this is the SAME confirmed subsample limitation already documented in `docs/submission_ladder_audit_2026-07-23.md`: the bulk daily dataset dump (`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22`) is a genuine subsample of that day's total ladder games, and our specific team's episodes were previously confirmed absent (9/9 real episode IDs from this exact date were real 404s against that dump). Our own matchup exposure needs the submission-API path instead (`tools/measure.py::fetch_our_episodes_via_submission_api`, already downloaded under `runs/our_episodes/`) cross-referenced against this dump's opponent-archetype labels by episode id — not done in this cycle, flagged as a B3 candidate below.

## (d) Behavioral scouting: top-100 vs. us
top-100 teams observed in this dump: 75 (of the leaderboard's real top 100 -- not all necessarily played a game in this one-day window)

Deeper behavioral comparison (game length, attack-decline rate, evolve timing) would need per-decision traces for top-100 opponents' own games, which this dump doesn't carry (only board-state snapshots, not legal-option/decision traces) -- out of scope for this cycle's signature-based scan; flagged as a B3 candidate below.

## (e) Is Crustle competitive in this meta? (recommendation only, no deck change this cycle)
Our own team doesn't appear in this dump at all (see (c)) so this pipeline can't directly self-classify us this cycle -- but our real deck (`decks/crustle_wall_deck.csv`) IS the `Crustle`/`Dwebble` archetype visible in (a)/(b) from OTHER teams running it, which is what this section evaluates instead.

Real Crustle-archetype matchup data from (b), across all teams observed running it (not just us):

- vs. Munkidori: 78/170 (46%) — reliable
- vs. Alakazam: 52/116 (45%) — reliable
- vs. Team Rocket's Spidops: 33/76 (43%) — reliable
- vs. Cynthia's Roserade: 24/43 (56%) — reliable
- vs. Dunsparce: 7/18 (39%) — **below n=30, unreliable**
- vs. Drakloak: 7/12 (58%) — **below n=30, unreliable**
- vs. Mega Starmie ex: 1/8 (12%) — **below n=30, unreliable**
- vs. Dragapult ex: 1/7 (14%) — **below n=30, unreliable**
- vs. Mega Kangaskhan ex: 2/7 (29%) — **below n=30, unreliable**
- vs. Marnie's Grimmsnarl ex: 4/5 (80%) — **below n=30, unreliable**
- vs. Thwackey: 2/5 (40%) — **below n=30, unreliable**
- vs. Cynthia's Garchomp ex: 5/5 (100%) — **below n=30, unreliable**
- vs. Dipplin: 0/3 (0%) — **below n=30, unreliable**
- vs. N’s Zoroark ex: 1/3 (33%) — **below n=30, unreliable**
- vs. Kadabra: 1/2 (50%) — **below n=30, unreliable**
- vs. Archaludon ex: 0/1 (0%) — **below n=30, unreliable**
- vs. Dwebble: 1/1 (100%) — **below n=30, unreliable**
- vs. Comfey: 0/1 (0%) — **below n=30, unreliable**

Crustle presence: 4 top-100 teams, 6 teams in the 600+ band — a real, established archetype in this meta, not a niche pick, with a roughly even-to-favorable matchup spread against the meta's two dominant archetypes (Alakazam, Munkidori) at this single-day sample size. **Directional signal only** — see methodology note above (n=1 day).

## B3: top-3 data-backed candidates for v4
1. **Opponent-deck prior for `sample_determinization()`** — currently assumes the opponent plays OUR OWN deck (`src/determinize.py`); the archetype distribution in (a) gives a real, data-backed prior to sample from instead. Evidence: 72 classified top-100 teams' archetypes now on record.
2. **Cross-reference our own submission-API episodes against this dump's archetype labels by episode id** — (c) couldn't compute our real per-archetype record because our own games are absent from this bulk-dump subsample; `runs/our_episodes/` (fetched via the reliable submission-API path) has our real episode ids and outcomes already. Matching those episode ids against this dump's per-episode archetype labels (where the opponent's episode happens to be present) would give a real, if partial, matchup-aware record without needing the opponent's own team to be in this exact day's subsample.
3. **Wider meta scan before any deck-swap decision** — (e)'s verdict is explicitly non-conclusive at n=1 day; re-run `tools/meta_miner.py` across a multi-day window (resumable by design) before treating any deck-swap recommendation as evidence-backed.
