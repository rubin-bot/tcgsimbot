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
Simulation allows **5 submissions/day**. Max team size 5.

> ✅ As of 2026-07-22 the user is entered in **both Strategy and Simulation**
> (`userHasEntered: True` for both). First Simulation submission (baseline rule-based
> agent) made 2026-07-22.

## ARCHITECTURE DECISION (do not revisit)

**NO self-play RL, NO AlphaZero.** The self-play/MCTS/neural-net training loop (see
"Deprecated" section below) crashed the user's laptop repeatedly during training, and
public ladder results show search + a tuned heuristic scorer beating RL+MCTS in this
environment. We build instead:

1. A **SearchScorer** agent using the cg SDK's native `search_begin`/`search_step`
   lookahead API + a hand-crafted `evaluate()` heuristic. ✅ Built (`agents/search_scorer.py`).
2. A **crash-proof local eval harness** (see Hardware rules below). ✅ Built
   (`tools/eval_arena.py`, `tools/_eval_worker.py`).
3. **Offline weight tuning** (CMA-ES, `tools/tune_weights.py`) over the local arena. ✅ Run
   once (`runs/tune_run1`): converged but stalled at ~44% vs. `baseline` (95% CI overlapping
   every pre-tuning number) — confirmed the local win-rate plateau was a missing-feature
   problem, not a weight-calibration one. One more feature cycle (energy-routing: added
   `turns_to_power`/`wasted_energy`, diagnosed via `tools/loss_review.py`'s feature-level
   traces) landed before shipping v1, per the policy change below.
4. An **optional tiny numpy-only policy net** via behavior cloning (no RL, no torch at
   inference time). Not built — superseded by the policy change below; may revisit if the
   ladder autopsy loop calls for it.
5. **Packaging + Kaggle submission** (`scripts/build_submission.py --mode search_scorer`,
   `scripts/verify_submission.py`, `scripts/submit.py`). ✅ v1 shipped — see `VERSIONS.md`.
6. A **daily "iterate" loop**: measure ladder μ → autopsy losses (from Kaggle episode
   replays, not just local games) → one fix → verify locally (regression check only, not the
   bar) → resubmit. **This is the active mode from v1 onward** — see the policy change below.

### Local-arena gate retired as of v1 — the ladder is now the judge

The original plan gated shipping on **60%+ vs. `baseline` over 200-300 local games**. That
bar was never cleared (best local result: ~44%, CMA-ES-tuned, statistically indistinguishable
from every pre-tuning run) despite two full feature-diagnosis cycles finding and fixing real,
evidenced gaps each time (tie-collapse → tie-break fix; threat/bench/tempo blindness → new
features; energy-routing → `turns_to_power`/`wasted_energy`). Each cycle measurably changed
*behavior* (attack-decline 46.5%→~11%, for instance) without moving the win-rate number much,
which is itself informative: **`baseline` and `random` are two narrow, non-representative
opponents** — real ladder opponents (Trainer-card decks, different archetypes, ex/Mega
strategies) are a fundamentally richer test than anything two fixed local bots can provide,
and Kaggle episode replays (real losses, real opponent decks) are strictly more informative
than another synthetic local cycle chasing the same two opponents. Per the decision made
shipping v1: **stop gating on local win rate, ship, and let the live ladder + loss autopsies
(Stage 6) drive future iteration.** The local arena (`tools/eval_arena.py`) keeps a real job —
**pre-submission regression check** (did this change break something that used to work,
measured against `baseline`/`random`/prior versions before a resubmit) — but it is no longer
the thing progress is judged against.

### Hardware rules (always enforce)

Max 2 simulator processes at once; total RAM under ~6 GB; every game runs in its own
subprocess with a timeout + RSS cap; stream data to disk, not RAM; every long-running
script must be resumable and checkpoint frequently; print progress. (This is the same
discipline `src/selfplay.py` already used for the deprecated pipeline — carry it forward.)

### Submission policy

**1 submission/day** so TrueSkill has time to converge before the next one lands. Version
tags v1, v2, ... logged in `VERSIONS.md` (repo root) with date, changes, and μ once known.

**Commit and push at the end of every iterate cycle.** This repo is the project's memory
across sessions — uncommitted work is invisible to the next session. Stage explicit paths
(never `git add -A`); `runs/`, `submission/`, `*.tar.gz`, and loose diagnostic `.jsonl`/`.log`
at the repo root are gitignored and never belong in a commit (see `.gitignore`).

## The Simulation agent (the technical core / critical path)

- Battles run on the **`cabt` engine** (`kaggle-environments` v1.14.10). A local **SDK** with
  identical logic is provided for training/debug/search, at
  `data/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/` (gitignored;
  located at runtime via `src/sdk_path.py`). Import as `cg`.
- **Turn loop:** agent receives an observation `{game logs, board state, list of legal
  options}` and returns the **index/indices of the chosen option(s)**. The engine only ever
  offers **legal** moves — you pick, you don't generate legality. Opponent's hand is hidden.
- **Native lookahead API:** `cg.api.search_begin` / `search_step` / `search_end` /
  `search_release` — confirmed real SDK entry points (see `docs/sdk_notes.md`), already
  exercised by the (now-deprecated) `src/mcts.py`. The new SearchScorer agent drives these
  directly instead of layering a trained value net on top.
- **API docs:** https://matsuoinstitute.github.io/cabt/ (+ a page of simulator-vs-official
  rule differences).
- **Submission:** `.tar.gz` with **`main.py` at top level** + **`deck.csv`**
  (`tar -czvf submission.tar.gz *`). Runtime path `/kaggle_simulations/agent/`.
  Limits: ≤197.7 MiB, 2 vCPUs, 12.2 GiB RAM, 11.8 GiB HDD, **5 submissions/day, latest 2
  active**.
- **Scoring:** N(μ, σ²), μ₀=600; win↑ / loss↓ / draw→mean; **margin of victory ignored**.
  Latest 2 submissions active. Host warns pure rule-based agents won't rank high.
- **Deck:** `decks/crustle_wall_deck.csv` — the current top meta deck pick. (No `deck.csv`
  exists at repo root yet; copy/rename this in at packaging time.)

### SearchScorer's tie-break ordering (`agents/search_scorer.py::_break_ties`)

When multiple root options score within `_TIE_EPS_REL` (near-exact float equality — these are
usually genuinely-equal reachable positions under the 2-ply search, not noise), ties are broken
in this order, most-preferred first:

1. **Option kind** (`_TIE_BREAK_PRIORITY`): attack > evolve > attach/energy > retreat >
   play/ability > everything else > end.
2. **Within the same kind, attacker-pipeline membership**: prefer a target that's part of the
   attacker's evolution pipeline (`ATTACKER_PIPELINE_IDS`, walked from the deck's designated
   attacker card's `evolvesFrom` chain at import time — **not** a hardcoded per-deck tuple; for
   `decks/crustle_wall_deck.csv` this resolves to Crustle + its pre-evolution Dwebble) over any
   other target. A pre-evolution counts as the attacker it becomes — energy attached to it
   persists through evolution.
3. **Within the pipeline, proximity to `ATTACKER_ENERGY_COST`**: prefer whichever pipeline
   member still needs energy and needs the *least* of it (closest to attack-ready). A pipeline
   member that's *already* at cost is ranked **worse** than one still building — attaching more
   there doesn't help, and `evaluate()`'s own `wasted_energy` feature can't always catch this
   case itself (it's a `min`-across-pipeline feature, blind whenever any other pipeline member
   is already fully powered — see `docs/tie_break_v2_2026-07-23.md`).
4. Otherwise, the engine's own option-list order (last resort, unchanged from before).

Landed as v2 (2026-07-23) after the v1-era tie-break only recognized an *already-evolved*,
under-cost attacker (Crustle specifically) — it never credited an unpowered pre-evolution at
all, and among multiple recognized targets with different energy levels it couldn't rank by
proximity, only by engine list order. `evaluate()` itself is untouched by this fix; it only
changes which of several *already-equally-scored* options gets picked.

### Strategy Category scoring (this category's rubric)
- **Model Score 70%** — clarity of approach & rationale, originality, technical soundness,
  consistency across repeated matches, robustness (not reliant on lucky states/matchups),
  plus Simulation-track performance.
- **Deck Score 20%** — deck concept, alignment with strategy, key-card selection & use.
- **Report Score 10%** — structure, writing, figures/charts/tables.

High leaderboard rank helps but does not guarantee winning; strong original analysis can
outscore a top agent with a weak writeup.

### Key dates (11:59 PM UTC) — Simulation is the binding deadline
| Milestone | Simulation | Strategy |
|---|---|---|
| Entry / team-merger | **2026-08-09** | 2026-09-06 |
| Final submission | **2026-08-16** | 2026-09-13 |
| Leaderboard/judging | games run to ~08-31 | judged 09-14 → 10-11 |

The agent must be built against the **Aug 16** deadline; the Strategy writeup describes it after.

## Repository layout
```
.
├── CLAUDE.md
├── data/
│   ├── pokemon-tcg-ai-battle/                      # engine SDK + C++ source (gitignored)
│   │   └── sample_submission/sample_submission/cg/ # cg SDK (api.py, game.py, sim.py,
│   │                                                #   utils.py, cg.dll/.so/.dylib)
│   ├── pokemon-tcg-ai-battle-challenge-strategy/   # extracted competition data
│   │   ├── EN_Card_Data.csv        # English card DB (2022 rows, 1267 unique cards)
│   │   ├── JP_Card_Data.csv        # Japanese mirror, same schema
│   │   ├── Card_ID List_EN.pdf     # ~131 MB card-image scans keyed to Card ID
│   │   └── Card_ID List_JP.pdf     # ~174 MB
│   └── pokemon-tcg-ai-battle-challenge-strategy.zip # source archive (~299 MB)
├── decks/
│   ├── baseline_deck.csv           # original validated Water deck
│   └── crustle_wall_deck.csv       # current top meta pick — used for submission
├── src/                             # see "Active foundations" / "Deprecated" below
├── scripts/
│   └── build_submission.py         # assembles submission.tar.gz — needs rework for
│                                    #   the SearchScorer agent (currently loads the
│                                    #   deprecated torch PVNet checkpoint for --mode net)
├── .claude/
│   └── skills/pokemon-tcg-rules/   # PTCG rules skill (turn structure, deck building,
│                                   #   format legality, special conditions, tournaments)
└── pokemon-tcg-rules.skill         # original skill archive (redundant with extracted copy)
```

### Card data schema (17 columns)
`Card ID, Card Name, Expansion, Collection No., Stage (Pokémon)/Type (Energy and Trainer),
Rule, Category, Previous stage, HP, Type, Weakness, Resistance (Type), Retreat, Move Name,
Cost, Damage, Effect Explanation`

- A card with multiple attacks spans **multiple rows sharing one Card ID**.
- Effects are full natural-language text.
- Pool: Scarlet & Violet era (19 expansions). 270 Pokémon ex, 54 Mega ex, 29 ACE SPEC,
  plus Tera / Ancient / Future / Trainer's-Pokémon mechanics.

## Tooling notes
- **Kaggle CLI**: authenticated via `KGAT_...` token at `~/.kaggle/access_token`
  (Kaggle SDK 2.2.3 reads this format). `kaggle.exe` is **not on PATH** — invoke as
  `python -m kaggle ...`.
- Platform is **Windows**; Python 3.14. `chmod` is cosmetic on NTFS — use `icacls` for
  real file locking.
- Read CSVs with `encoding='utf-8'` and set `PYTHONIOENCODING=utf-8` (console is cp1252;
  card names/effects contain `é`, `×`, full-width `（）`, etc.).
- See **Hardware rules** above — they apply to every long-running script from here on,
  not just the deprecated self-play trainer.

## Data-use constraints (competition rules)
- Competition Data ("Pokémon Elements") may be used **only** for this competition and must
  be **deleted afterward**. Do not redistribute.
- Winning code is licensed **MIT**; open-source obligations apply to winning submissions.
- Models trained on the data may not be used commercially or to regenerate Pokémon Elements
  outside the competition.

## Active foundations (still in use)

Built and verified against the real engine; reused by the SearchScorer agent:
- `src/carddata.py` — `load_card_index()` joins engine `all_card_data()`/`all_attack()` with
  `EN_Card_Data.csv` effect text (join key = Card ID == engine cardId).
- `src/obs.py` — `parse_obs()` → typed, **information-hidden** GameState + classified legal
  options (opponent hand never exposed; invariant tested).
- `src/baseline.py` — priority-heuristic agent (lethal → attach energy → evolve → best attack
  → conditional retreat → fallback). **66–70% vs random.** Sparring partner / fallback,
  and the reference point the SearchScorer must beat.
- `src/sdk_path.py` — locates the gitignored cg SDK under `data/`.
- `src/encode.py` — visible-info-only state/option feature encoders (`STATE_DIM=342`,
  `OPTION_DIM=24`); reusable if the optional behavior-cloned policy net (item 4 above) gets
  built.

## Deprecated: AlphaZero self-play pipeline (kept for reference, not deleted)

Built and self-play-tested in an earlier phase, then abandoned per the **ARCHITECTURE
DECISION** above — repeated hardware crashes during training, plus evidence that
search + heuristic outperforms RL+MCTS here. Left in place under `src/` but **not** part of
the active build:

`src/net.py` (torch `PVNet`), `src/net_numpy.py` (torch-free NumPy mirror),
`src/determinize.py` (hidden-world sampling), `src/mcts.py` (determinized IS-MCTS —
NOTE: its use of `search_begin`/`search_step` is the one part worth mining for the new
SearchScorer agent), `src/selfplay.py`, `src/replay.py`, `src/train_step.py`,
`src/train.py`, `src/evaluate.py`.

`runs/` (training artifacts, gitignored) and any `--mode net` path through
`scripts/build_submission.py` belong to this deprecated pipeline too.

## Iterate workflow (Stage 6)

Whenever the user says "iterate" (or `/iterate`) in any session, run the full Stage 6 loop
documented in `ITERATE.md`: **MEASURE** (`tools/measure.py` — pull ladder μ into
`VERSIONS.md`, pull any of our games from the newest Kaggle episode-replay dataset) →
**DIAGNOSE** (`tools/autopsy.py --source auto` — ranked shortcomings report, ladder replays if
we have real losses, local `baseline`/`random` fallback otherwise, built on
`tools/loss_review.py`) → **FIX** (one candidate for the top failure mode, proven over 400+
games via a Kaggle kernel attached to the competition, `scripts/build_kernel_bakeoff.py` — no
proven win, no ship) → **SHIP** (`scripts/build_submission.py` →
`scripts/verify_submission.py` → `scripts/submit.py`, log in `VERSIONS.md`, commit + push).
`ITERATE.md` and `.claude/skills/iterate/SKILL.md` are the source of truth for the exact
commands; this is just the pointer. Still 1 submission/day, still ≤2 local sim workers.

## Status / next steps
- ✅ Both competitions understood; user entered in **both**. SDK verified on Win + Linux.
- ✅ Active foundations (carddata/obs/baseline/sdk_path/encode) built, tested, pushed to
  `rubin-bot/tcgsimbot`.
- ✅ Architecture pivot decided: dropping the AlphaZero self-play loop in favor of a
  SearchScorer agent (native `search_begin`/`search_step` + hand-crafted `evaluate()`).
- ✅ SearchScorer built, crash-proof local arena built, two feature-diagnosis cycles + one
  CMA-ES tuning pass run (see ARCHITECTURE DECISION above for the full history).
- ✅ **v1 shipped** — `scripts/build_submission.py --mode search_scorer`, verified via
  `scripts/verify_submission.py`'s fresh-extraction smoke test, submitted via
  `scripts/submit.py`. See `VERSIONS.md` for the exact weights/changes and μ once scored.
- ✅ **Stage 6 tooling built**: `tools/measure.py`, `tools/autopsy.py`,
  `scripts/build_kernel_bakeoff.py`, `ITERATE.md`, `.claude/skills/iterate/`. See the
  "Iterate workflow" section above.
- ✅ **First real ladder MEASURE+AUTOPSY cycle run (2026-07-23, AM)** — `tools/measure.py`
  extended with leaderboard rank fetch (fixed a UTF-8 BOM bug that silently broke the `Rank`
  column lookup), a persistent `runs/mu_history.jsonl` trajectory log, and multi-day episode
  scanning; new `tools/ladder_report.py` orchestrates PARSE+AUTOPSY+REPORT. That run's episode
  scan (the bulk daily-dataset dump) found 0 of our episodes and treated it as inconclusive.
- ✅ **Submission/ladder audit (2026-07-23, PM)** — verified the AM cycle's "0 episodes" via
  the Kaggle API directly instead of trusting the inconclusive read: both active submissions
  are `SubmissionStatus.COMPLETE`, no errors; **v1 has actually played 45 real ladder games**
  (net checkpoint: 30) that the bulk daily dataset simply didn't happen to include (confirmed
  a genuine subsample — 9/9 of our real episode IDs got real 404s from that dataset). Added
  `tools/measure.py::fetch_our_episodes_via_submission_api()` (the `kaggle competitions
  episodes <id>` + `replay <id>` API) as the new primary episode-discovery path — direct,
  complete, no 700MB+/day download; old bulk scan kept as an opt-in `--also-bulk-scan`
  fallback. Also fixed a latent `src/carddata.py` bug (`card_type` stored as raw int, not
  `CardType`) that the newly-unblocked ladder pipeline was the first thing to ever exercise.
  Full writeup: `docs/submission_ladder_audit_2026-07-23.md`.
  **Real record so far: 75 games, 34W/41L (45.3%)**, μ=568.6, rank≈3664/5552 — regenerated
  `docs/ladder_autopsy_2026-07-23.md` now has real numbers instead of the AM cycle's fallback
  section. Notable: **ladder attack-decline rate is 77.6% (554/714) vs. 9.5% locally** — a
  large real/local behavior gap worth prioritizing next. Caveat: the 75-game record blends
  both currently-active submissions (v1 + the older net checkpoint) since episode parsing
  matches on team name only, not submission ID — see the audit doc's Limitations section.
- ⏭️ **Stage 6 from here on**: real ladder data now exists for the first time — the
  attacker-starvation tie-break hypothesis from the AM cycle can finally be checked against it
  (`tools/autopsy.py --source ladder`, section (d)), and the large attack-decline gap above is
  worth its own look. Implement one fix, prove it on a Kaggle kernel
  (`scripts/build_kernel_bakeoff.py`, 400+ games), resubmit — 1/day per the Submission policy
  above. No agent/weights changes were made in the 2026-07-23 PM audit itself.
