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

> ⚠️ As of 2026-07-21 the user is entered in **Strategy but NOT Simulation**
> (`userHasEntered: False`). Must join Simulation before its entry deadline **2026-08-09**.

## The Simulation agent (the technical core / critical path)

- Battles run on the **`cabt` engine** (`kaggle-environments` v1.14.10). A local **SDK** with
  identical logic is provided for training/debug/RL.
- **Turn loop:** agent receives an observation `{game logs, board state, list of legal
  options}` and returns the **index/indices of the chosen option(s)**. The engine only ever
  offers **legal** moves — you pick, you don't generate legality. Opponent's hand is hidden.
- **API docs:** https://matsuoinstitute.github.io/cabt/ (+ a page of simulator-vs-official
  rule differences).
- **Submission:** `.tar.gz` with **`main.py` at top level** + **`deck.csv`**
  (`tar -czvf submission.tar.gz *`). Runtime path `/kaggle_simulations/agent/`.
  Limits: ≤197.7 MiB, 2 vCPUs, 12.2 GiB RAM, 11.8 GiB HDD.
- **Scoring:** N(μ, σ²), μ₀=600; win↑ / loss↓ / draw→mean; **margin of victory ignored**.
  Latest 2 submissions active. Host warns pure rule-based agents won't rank high.

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
│   ├── pokemon-tcg-ai-battle-challenge-strategy/   # extracted competition data
│   │   ├── EN_Card_Data.csv        # English card DB (2022 rows, 1267 unique cards)
│   │   ├── JP_Card_Data.csv        # Japanese mirror, same schema
│   │   ├── Card_ID List_EN.pdf     # ~131 MB card-image scans keyed to Card ID
│   │   └── Card_ID List_JP.pdf     # ~174 MB
│   └── pokemon-tcg-ai-battle-challenge-strategy.zip # source archive (~299 MB)
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

## Data-use constraints (competition rules)
- Competition Data ("Pokémon Elements") may be used **only** for this competition and must
  be **deleted afterward**. Do not redistribute.
- Winning code is licensed **MIT**; open-source obligations apply to winning submissions.
- Models trained on the data may not be used commercially or to regenerate Pokémon Elements
  outside the competition.

## The SDK (downloaded & verified running locally, 2026-07-21)

`data/pokemon-tcg-ai-battle/` (from `pokemon-tcg-ai-battle.zip`):
- `ptcg_engine/ptcgProgram 22/` — full **C++20 engine source** (38 `.h` + `Export.cpp`,
  VS2022 solution) = ground-truth game logic. Competition-use-only; delete after comp.
- `sample_submission/sample_submission/` — submission template:
  - `main.py` — `agent(obs_dict)->list[int]`; return the 60-card **deck** when
    `obs.select is None`, else option indices.
  - `deck.csv` — sample 60-card deck.
  - `cg/` — Python SDK (`api.py`, `game.py`, `sim.py`, `utils.py`) + compiled engine libs
    (`cg.dll`, `libcg*.so`, `libcg.dylib`). Import as `cg`.

**Local play API** (`cg.game`, no kaggle_environments needed):
`battle_start(deck0, deck1) -> (obs|None, StartData)`; `battle_select(list[int]) -> obs`;
`battle_finish()`. Loop until `obs["current"]["result"] != -1` (0/1 = winner, 2 = draw).
`cg.api.all_card_data()` → 1267 CardData; `all_attack()` → 1556 Attack.

Verified working on Windows / Python 3.14 (`cg.dll`). Smoke test:
`data/pokemon-tcg-ai-battle/sample_submission/sample_submission/smoke_test.py`
(run with `PYTHONIOENCODING=utf-8`).

## Foundations (built & verified in a cloud session on Linux, merged here 2026-07-21)

All under `src/` / `tests/`, verified against the real engine. See `docs/sdk_notes.md` for
ground-truth SDK findings captured on Linux.
- `src/carddata.py` — `load_card_index()` joins engine `all_card_data()`/`all_attack()` with
  `EN_Card_Data.csv` effect text (join key = Card ID == engine cardId).
- `src/obs.py` — `parse_obs()` → typed, **information-hidden** GameState + classified legal
  options (opponent hand never exposed; invariant tested).
- `src/baseline.py` — priority-heuristic agent (lethal → attach energy → evolve → best attack
  → conditional retreat → fallback). **66–70% vs random.** Sparring partner / fallback only —
  NOT the learned agent.
- `src/sdk_path.py` — locates the gitignored cg SDK under `data/`.
- `decks/baseline_deck.csv` — legal 60-card Water deck (Finizen/Palafin + Totodile line +
  Bruxish + energy), validated via `battle_start`.
- `scripts/build_submission.py` — assembles `submission.tar.gz` (~0.48 MiB, well under the
  197.7 MiB cap) from source; `submission/` and `*.tar.gz` are gitignored build output.

Note: the cloud built/tested against the Linux `libcg.so`; on this Windows desktop the code
runs against `data/…/cg/cg.dll`. Re-verify `src/` runs on Windows before relying on it here.

## AlphaZero self-play agent (the learned agent)

Training runs in the **Python 3.12 venv** (`.venv/`); torch is training-only. Run all
training-side code as `PYTHONPATH=src .venv/Scripts/python ...` (base 3.14 has no torch).

Pipeline modules (all in `src/`):
- `encode.py` — visible-info-only state vector (`STATE_DIM=342`) + per-option features
  (`OPTION_DIM=24`). Info-hiding invariant tested.
- `net.py` — tiny policy/value net (`PVNet`, ~78K params @ hidden=128). Policy scores each
  legal option; `evaluate_np` for inference, `forward_batch` for training, `export_numpy`
  for the torch-free submission forward.
- `determinize.py` — samples the hidden world (opp hand/deck/prizes + own deck order) from
  the deck distribution; engine-accepted.
- `mcts.py` — determinized IS-MCTS over `search_begin/step`; net-valued leaves; signature-keyed
  children; ~0.8 ms/sim, 0% truncation. `search(...)` returns the improved (visit) policy.
- `selfplay.py` — MCTS self-play games → shaped-return training records (`outcome +
  alpha*prize_diff`, alpha annealable); parallel across processes; npz storage.
- `replay.py` — sliding-window replay buffer with padded/masked batch collation.
- `train_step.py` — AlphaZero loss (policy CE vs visit dist + value MSE).
- `evaluate.py` — win-rate ladder vs fixed reference (baseline + frozen snapshots).
- `train.py` — **resumable orchestrator**. Artifacts under `runs/<name>/` (gitignored).

Run training (16 cores, ~4 min/iter train-only, ~2.3 h for 30 iters):
```
PYTHONPATH=src .venv/Scripts/python src/train.py --name run2 --iters 30 \
  --games-per-iter 32 --workers 16 --sims 32 --train-steps 300 --batch 256 \
  --eval-every 5 --eval-sims 32 --eval-games 12 --gate-games 12
```
`--eval-every N` runs the (expensive) net-vs-net eval+gate only every N iters;
every iter still self-plays + trains (non-eval iters print `(train-only)`, blank
win-rate cols). Resume/extend with the same `--name` and a higher `--iters`;
Ctrl-C is safe between iters. Watch eval rows in `runs/<name>/metrics.csv`
(`wr_baseline`→~0.60, `wr_pool` rising).

## Status / next steps
- ✅ Both competitions understood; user entered in **both**. SDK verified on Win + Linux.
- ✅ Foundations + full AlphaZero self-play loop built, tested, pushed to `rubin-bot/tcgsimbot`.
- ⏭️ Run enough iterations to hit the success bar (learned agent ≥~60% vs baseline + rising
  curve vs frozen checkpoints); then scale net/sims.
- ⏭️ **Phase B**: extend `scripts/build_submission.py` to bundle a trained checkpoint + a
  torch-free `net.export_numpy` forward (keep baseline fallback), then submit to
  `pokemon-tcg-ai-battle` before Aug 16. Also: deck co-optimization (outer loop).
