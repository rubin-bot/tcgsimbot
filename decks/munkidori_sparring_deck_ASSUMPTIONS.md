# Munkidori sparring deck -- reconstruction assumptions

**NOT a real decklist.** Built for master-study sparring purposes only (docs/master_study_*.md, Workstream C). Card PRESENCE is real, mined data (runs/meta_mining/team_game_signatures.jsonl); COPY COUNTS are inferred via a simple frequency-tier rule since the mining signature only stores the top-3-per-game cards, never full decklists. See tools/reconstruct_archetype_deck.py's docstring for the exact rule.

Pooled from top 3 real teams by game count: Rmy (405 games), Luca (368 games), jiatu.l (323 games) -- 1096 total games.

## Cards (real presence, inferred copy count)

| Card | seen in N games | frac of pooled games | assigned copies |
|---|---|---|---|
| Munkidori | 720 | 65.7% | 3 |
| Marnie's Grimmsnarl ex | 462 | 42.2% | 3 |
| Poké Pad | 341 | 31.1% | 2 |
| Buddy-Buddy Poffin | 313 | 28.6% | 2 |
| Lillie's Determination | 291 | 26.6% | 2 |
| Marnie's Impidimp | 268 | 24.5% | 2 |
| Team Rocket's Petrel | 249 | 22.7% | 2 |
| Marnie's Morgrem | 166 | 15.1% | 2 |
| Rare Candy | 127 | 11.6% | 1 |
| Spikemuth Gym | 102 | 9.3% | 1 |
| Night Stretcher | 101 | 9.2% | 1 |
| Froslass | 47 | 4.3% | 1 |
| Boss’s Orders | 35 | 3.2% | 1 |
| Snorunt | 18 | 1.6% | 1 |
| Enhanced Hammer | 10 | 0.9% | 1 |
| Dawn | 10 | 0.9% | 1 |
| Alakazam | 8 | 0.7% | 1 |
| Hilda | 8 | 0.7% | 1 |
| Xerosic’s Machinations | 4 | 0.4% | 1 |
| Telepath Psychic Energy | 3 | 0.3% | 1 |
| Kadabra | 3 | 0.3% | 1 |
| Handheld Fan | 1 | 0.1% | 1 |
| Unfair Stamp | 1 | 0.1% | 1 |

## Basic energy fill (never in the mined signature -- meta_miner.py excludes it by design)

Split by each energy-needing Pokemon's own real observed frequency, not an arbitrary guess:

- 14x Basic {D} Energy
- 12x Basic {P} Energy
- 1x Basic {W} Energy
