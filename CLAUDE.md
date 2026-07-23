# Pokémon TCG AI Battle Challenge

Kaggle Featured Hackathon by The Pokémon Company. This repo is the workspace for competing
in **The Pokémon Company – PTCG AI Battle Challenge**. Entered under account
rubinsahota2009@gmail.com. **$240,000** prize pool (8 finalists × $30,000).

## Version control

- **GitHub repo:** https://github.com/rubin-bot/tcgsimbot (branch `main`)
- **Version-control all future changes here:** commit locally and `git push origin main`
  from this desktop clone. Push works from the desktop (`gh` authed as `rubin-bot`); it does
  NOT work from the cloud web sessions (their git gateway is scoped to other repos).
- Never commit Pokémon Elements — `.gitignore` keeps `data/` (card data, PDFs, the cg SDK &
  compiled libs) and the rules skill out of git. Keep it that way.

## The competition is two paired parts — we must enter both

| | **Simulation Category** (`pokemon-tcg-ai-battle`) | **Strategy Category** (`…-strategy`) |
|---|---|---|
| Build | The AI Training Agent + deck that plays battles | A written report about that agent & deck |
| Scored by | Automated TrueSkill-style ladder | Human judges |
| Deliverable | `.tar.gz` agent bundle | Kaggle Writeup (≤2000 words) + optional media |
| Prize | Knowledge (points/medals) only | $ flows here (8 × $30k) |

Team composition must be **identical** across both divisions (rule 2.1.c). Strategy requires
Simulation entry (not vice-versa). Strategy is a hackathon = **one submission per team**;
Simulation allows **5 submissions/day**. Max team size 5. Entered in **both** since 2026-07-22.

**Strategy scoring**: Model 70% (approach clarity/originality/soundness/consistency/robustness
+ Simulation performance), Deck 20% (concept/alignment/key-card use), Report 10%
(structure/writing/figures). Rank helps but doesn't guarantee winning.

**Key dates** (11:59 PM UTC, Simulation is the binding deadline): entry **2026-08-09**, final
submission **2026-08-16**, leaderboard runs to ~08-31. Strategy trails by ~1 month (final
2026-09-13, judged through 10-11) — build the agent first, write the report after.

## Architecture (do not revisit without strong new evidence)

**NO self-play RL, NO AlphaZero** — repeated training crashes plus public evidence that
search + heuristics beat RL/MCTS here. Current agent: `agents/search_scorer.py` — native
`cg.api.search_begin`/`search_step` lookahead (2-ply, `MAX_OUR_PLIES`) + hand-crafted
`evaluate()` + a merit-based tie-break + (as of v3, unshipped) N=8-sample determinization voting.
Full history of what was tried, what worked, and why: `VERSIONS.md`, `git log`, and the dated
reports under `docs/`. The **local arena is a pre-submission regression check only, not a ship
gate** — the live ladder + real loss autopsies (Stage 6, below) are what progress is judged
against; a fix ships only after clearing the `verification-gate` skill's protocol.

### Hardware rules (always enforce)

Max 2 local simulator processes at once; total RAM under ~6 GB; every game runs in its own
subprocess with a timeout + RSS cap; stream data to disk, not RAM; every long-running script
must be resumable and checkpoint frequently; print progress.

### Submission policy

**1 submission/day** so TrueSkill has time to converge. Version tags v1, v2, ... logged in
`VERSIONS.md` with date, changes, and μ once known. **Commit and push at the end of every
iterate cycle** — this repo is the project's memory across sessions. Stage explicit paths
(never `git add -A`); `runs/`, `submission/`, `*.tar.gz`, loose diagnostic `.jsonl`/`.log` at
the repo root are gitignored and never belong in a commit.

## The Simulation agent

- Battles run on the **`cabt` engine** (`kaggle-environments` v1.14.10). A local **SDK** with
  identical logic is provided for training/debug/search, at
  `data/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/` (gitignored; located at
  runtime via `src/sdk_path.py`). Import as `cg`.
- **Turn loop:** agent receives an observation and returns the index/indices of the chosen
  legal option(s) — the engine only ever offers legal moves, you pick, you don't generate
  legality. Opponent's hand is hidden.
- **Native lookahead:** `cg.api.search_begin`/`search_step`/`search_end`/`search_release`
  (see `docs/sdk_notes.md`). No per-move Kaggle timeout exists (`actTimeout: 0`, confirmed
  across 75+ real episodes) — only a whole-*episode* `runTimeout: 2000`.
- **Submission:** `.tar.gz` with `main.py` at top level + `deck.csv`. Limits: ≤197.7 MiB,
  2 vCPUs, 12.2 GiB RAM, 11.8 GiB HDD, **5 submissions/day, latest 2 active**.
- **Scoring:** N(μ, σ²), μ₀=600; win↑/loss↓/draw→mean; margin of victory ignored.
- **Deck:** `decks/crustle_wall_deck.csv` — current top meta pick (confirmed a real,
  established archetype on the ladder, not niche — see `docs/meta_report_2026-07-22.md`). No
  `deck.csv` exists at repo root — it's copied/renamed in only at packaging time
  (`scripts/build_submission.py`); local dev scripts read `decks/crustle_wall_deck.csv` directly.

## Repository layout

```
.
├── CLAUDE.md
├── data/                            # gitignored: engine SDK, C++ source, card CSVs/PDFs
├── decks/                           # crustle_wall_deck.csv is the active pick
├── agents/search_scorer.py          # the live agent
├── src/                             # active foundations + deprecated self-play (see below)
├── tools/                           # eval/measure/loss_review/ladder/meta-mining tooling
├── scripts/                         # build_submission / verify_submission / submit / kernels
├── tests/                           # plain-assert scripts, no pytest — run each directly
├── docs/                            # dated cycle reports (the project's detailed memory)
├── VERSIONS.md, ITERATE.md          # shipped-version log, Stage 6 loop reference
└── .claude/skills/                  # pokemon-tcg-rules, iterate, verification-gate,
                                      #   ladder-analysis
```

**Card data** (`EN_Card_Data.csv`, 17 columns: Card ID/Name/Expansion/Collection No./Stage or
Type/Rule/Category/Previous stage/HP/Type/Weakness/Resistance/Retreat/Move Name/Cost/Damage/
Effect Explanation) — a multi-attack card spans multiple rows sharing one Card ID. Scarlet &
Violet-era pool, 1267 unique cards.

## Tooling notes

Kaggle CLI authenticated via `KGAT_...` token at `~/.kaggle/access_token`; `kaggle.exe` isn't
on PATH — invoke as `python -m kaggle ...` or use `tools/kaggle_common.py`'s helpers. Platform
is **Windows**, Python 3.14; `chmod` is cosmetic on NTFS, use `icacls` for real file locking.
Read CSVs with `encoding='utf-8'` and set `PYTHONIOENCODING=utf-8` (console is cp1252; card
names/effects contain non-ASCII characters). Hardware rules above apply to every long-running
script, not just the deprecated trainer.

## Data-use constraints

Competition Data ("Pokémon Elements"): use only for this competition, **delete afterward**, no
redistribution. Winning code is MIT-licensed. Models trained on the data may not be used
commercially or to regenerate Pokémon Elements outside the competition.

## Active foundations vs. deprecated

**Active** — `src/carddata.py` (card index), `src/obs.py` (`parse_obs()`, information-hidden
state), `src/baseline.py` (rule-based fallback/sparring agent), `src/sdk_path.py`,
`src/determinize.py` (hidden-world sampling, used by `search_scorer.py`), `src/encode.py`
(feature encoders, dormant unless the numpy policy-net idea below gets revisited).

**Deprecated** (kept for reference, not deleted) — the AlphaZero self-play pipeline abandoned
per the Architecture decision above: `src/net.py`/`src/net_numpy.py` (PVNet), `src/mcts.py`
(IS-MCTS — its `search_begin`/`search_step` usage is what `search_scorer.py` was built from),
`src/selfplay.py`, `src/replay.py`, `src/train_step.py`, `src/train.py`, `src/evaluate.py`.
`runs/` training artifacts and `scripts/build_submission.py --mode net` belong here too. A
tiny numpy-only behavior-cloned policy net (`src/encode.py`) was never built — superseded by
the iterate-loop policy below; may revisit if the ladder autopsy loop calls for it.

## Iterate workflow (Stage 6)

Whenever the user says "iterate" (or `/iterate`), use the **`iterate`** skill — it points to
`ITERATE.md` (source of truth for the exact commands) and, for anything involving a fix that
needs shipping or real ladder data, the **`verification-gate`** and **`ladder-analysis`**
skills respectively. Shape: MEASURE (ladder μ + real episodes) → DIAGNOSE (ranked shortcomings)
→ FIX (one candidate, proven via the verification-gate protocol) → SHIP (build → verify →
submit → log → commit + push). Still 1 submission/day, still ≤2 local sim workers.

## Method lessons (cross-cutting — read before designing a new verification cycle)

- **A repro case being correct doesn't mean it's representative.** Always measure a fix's
  effect on the full real population before spending kernel time (`verification-gate` skill).
- **Seeds don't pin game-engine determinism** (`cg.game.battle_start()` takes none — confirmed
  same-code-same-seed runs agree at chance level, not near-100%). Treat every 400-game run as
  an independent sample; compare via CI overlap, never "paired by seed."
- **"Flip rate vs. historical choice" ≠ a stability metric** — a decisive, consistent
  disagreement with an old (possibly wrong) choice looks identical to noise under that metric.
  Use self-consistency (mode agreement across repeated replays) when the question is noise.
- **The bulk daily episode dataset dump is a confirmed subsample**, not a complete ladder
  record — reliable for ladder-*wide* characterization (meta-mining), not for finding any one
  specific team's (including our own) games; use the submission-API path for that instead.

## Current status

**v1 is the active Kaggle submission** (SearchScorer, shipped 2026-07-22). v2 (target-aware
tie-break) and v3 (N=8 determinization voting) are both implemented, tested, and committed but
**not shipped** — each cleared some but not all of the `verification-gate` protocol's criteria;
see `docs/tie_break_v2_2026-07-23.md` and `docs/v3_report_2026-07-23.md` for the full evidence
and reasoning. A ladder-wide meta-mining pipeline (`tools/meta_miner.py`/`meta_report.py`) is
new this cycle — see `docs/meta_report_2026-07-22.md` for the first real archetype/matchup
report and its top-3 data-backed candidates for the next fix cycle.

For anything more specific than this — exact μ/rank history, what a given cycle found, why a
past decision was made — read `VERSIONS.md`, `runs/mu_history.jsonl`, `git log`, and the dated
reports under `docs/` rather than expecting it maintained here.
