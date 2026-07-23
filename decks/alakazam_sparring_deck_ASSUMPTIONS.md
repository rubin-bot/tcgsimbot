# Alakazam sparring deck -- reconstruction assumptions

**NOT a real decklist.** Built for master-study sparring purposes only (docs/master_study_*.md, Workstream C). Card PRESENCE is real, mined data (runs/meta_mining/team_game_signatures.jsonl); COPY COUNTS are inferred via a simple frequency-tier rule since the mining signature only stores the top-3-per-game cards, never full decklists. See tools/reconstruct_archetype_deck.py's docstring for the exact rule.

Pooled from top 3 real teams by game count: Yushin Ito (434 games), Majkel1337 (214 games), haggle (139 games) -- 787 total games.

## Cards (real presence, inferred copy count)

| Card | seen in N games | frac of pooled games | assigned copies |
|---|---|---|---|
| Alakazam | 323 | 41.0% | 3 |
| Hilda | 320 | 40.7% | 3 |
| Buddy-Buddy Poffin | 319 | 40.5% | 3 |
| Dawn | 305 | 38.8% | 2 |
| Poké Pad | 269 | 34.2% | 2 |
| Enhanced Hammer | 247 | 31.4% | 2 |
| Telepath Psychic Energy | 124 | 15.8% | 2 |
| Kadabra | 109 | 13.9% | 1 |
| Rare Candy | 64 | 8.1% | 1 |
| Abra | 47 | 6.0% | 1 |
| Xerosic’s Machinations | 47 | 6.0% | 1 |
| Boss’s Orders | 38 | 4.8% | 1 |
| Munkidori | 30 | 3.8% | 1 |
| Marnie's Grimmsnarl ex | 25 | 3.2% | 1 |
| Dunsparce | 17 | 2.2% | 1 |
| Lillie's Determination | 16 | 2.0% | 1 |
| Marnie's Impidimp | 15 | 1.9% | 1 |
| Team Rocket's Petrel | 14 | 1.8% | 1 |
| Nighttime Mine | 11 | 1.4% | 1 |
| Night Stretcher | 8 | 1.0% | 1 |
| Marnie's Morgrem | 7 | 0.9% | 1 |
| Lana’s Aid | 2 | 0.3% | 1 |
| Sacred Ash | 2 | 0.3% | 1 |
| Froslass | 2 | 0.3% | 1 |

## Basic energy fill (never in the mined signature -- meta_miner.py excludes it by design)

Split by each energy-needing Pokemon's own real observed frequency, not an arbitrary guess:

- 24x Basic {P} Energy
- 2x Basic {D} Energy
- 0x Basic {W} Energy
