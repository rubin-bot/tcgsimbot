# Strategy snapshot — run2, iteration 40 (2026-07-22)

Honest account of what the learned agent is doing right now and how it got there. Written to
double as seed material for the Strategy-category writeup — numbers here are pulled straight
from `runs/run2/metrics.csv` and a live self-play sample, not rounded up for effect.

## 1. What the system is

An AlphaZero-style self-play loop over the `cabt` engine, not a hand-coded strategy:

- **State/option encoding** (`src/encode.py`) — the whole board position (both players' active +
  bench Pokémon, hand-type histogram, prizes, deck/discard counts, turn info) plus, separately,
  a feature vector for *each legal option* the engine currently offers (kind one-hot + raw
  target/cost/HP numerics). Strictly information-hidden: the opponent's hand and both players'
  face-down prizes never enter the vector (tested).
- **Net** (`src/net.py`) — a small 128-hidden-unit MLP (~78K params). A shared torso embeds the
  state; a value head scores the position; a policy head scores each legal option individually
  (so it handles a variable-size, context-dependent option list) and softmaxes over them.
- **Search** (`src/mcts.py` + `src/determinize.py`) — determinized information-set MCTS: every
  simulation samples a plausible hidden world (opponent hand/deck order/prizes, drawn from the
  known 60-card deck list) and runs a fresh internal rollout via the engine's own
  `search_begin`/`search_step` sandbox (separate from the live battle, so this is safe to run
  during an actual match). The net supplies move priors and a leaf value; no rollouts, no
  hand-coded heuristics — the "improved policy" is purely the resulting visit distribution.
- **Self-play → training loop** (`src/selfplay.py`, `src/replay.py`, `src/train_step.py`,
  `src/train.py`) — the current-best net plays itself under MCTS, games become training records
  (state, per-option visit distribution, shaped return), a learner trains on a sliding replay
  window, and is periodically evaluated against a **fixed** reference set (the rule-based
  baseline + frozen past snapshots). It's only promoted to be the new generator if it beats the
  current generator head-to-head by a clear margin (gate ≥ 0.55) — this gating is what prevents
  self-play from drifting into an exploitable cycle.

## 2. Training status as of now

- **40 iterations logged** (0–39), currently **paused** (background process was killed by the
  environment after cleanly finishing iteration 39 — not a crash; resumable from iteration 40).
- **7 promotions** so far, all clustered from iteration 30 onward (iters 30–36); pool holds 4
  frozen snapshots (max-pool cap).
- **Tuning change at iteration 37**: self-play sims/move 32→64 and games/iteration 32→48, aimed
  at sharper (less noisy) MCTS visit distributions. Iterations 37–39 ran clean on this setting
  with no engine faults (a crash-hardening fix landed the same session — see the training
  dashboard artifact for the full incident history).
- **Loss**: falling steadily and monotonically since the tuning change — 1.592 → 1.563 → 1.538
  (iters 37/38/39), continuing the downward trend from earlier iterations (1.879 at iter 0).
- **Win rate vs. the fixed rule-based baseline**: **oscillates, 0.08–0.50**, last measured 0.417
  at iteration 36 (the most recent evaluated iteration; 37–39 were train-only, no new eval).
  This has **not** yet turned into a clean, stable upward trend — the net is not a confirmed
  improvement over the baseline yet, just a promising-but-early trajectory. Say this plainly to
  anyone reading the ladder score.
- **Win rate vs. the frozen pool**: similarly noisy, 0.17–0.50, last measured 0.417 at iter 36.

## 3. Current behavioral profile — what it's actually doing

Measured over the freshest ~60 self-play games at the iteration-40 (sims=64) setting — the
*preferred* (highest-visit) move category per decision, across 6,209 decisions:

| Move category | Share of decisions |
|---|---|
| End turn | 44.6% |
| Attach energy | 15.1% |
| Play/choose a card | 12.7% |
| Attack | 7.7% |
| Play a Trainer card | 6.2% |
| Energy (type/target choice) | 4.4% |
| Evolve | 4.2% |
| Retreat | 4.0% |
| Confirm/decline/choose-number prompts | ~1.3% combined |
| Ability / Skill / Attach tool / Choose energy card / Discard / Special condition | **0.0% each** |

Other behavioral stats from the same sample:
- **Decisiveness** (average MCTS visit share captured by the top move): **~0.78–0.81** — the
  search is reasonably confident, not close to uniform/random, across a decision that on
  average has several legal options.
- **Draw rate**: ~0% in the most recent window (was ~1.7% a few iterations earlier).
- **Average game length**: ~98–106 decisions per game (both players combined).

### Interpretation (read as hypotheses, not conclusions)

- The bot currently reads as **development/attrition-oriented rather than aggressive**: across
  all decisions, it ends its turn or attaches energy/evolves far more often than it attacks.
  This could mean it has learned patient board-building is valuable in this matchup (a mirror
  match on the same 60-card Water deck) — or it could mean the value/policy signal for
  attacking just hasn't sharpened yet at 40 iterations. Too early to tell apart; the next few
  evaluated iterations' win-rate trend is the tell.
- **Retreating is rare (4%)** and **abilities/tools/special conditions never get preferred
  (0%)** — **resolved, not a net-learning gap**: `decks/baseline_deck.csv` is 40 Basic {W}
  Energy + 4x each of Totodile/Croconaw/Palafin/Finizen/Bruxish and contains **zero Trainer
  cards** (no Items, Supporters, Tools, or Stadiums) and no checked Ability text on the current
  five Pokémon. Those option kinds are essentially never *legal* in the first place, so their
  0% share reflects the deck, not the policy. This is a real deck-construction gap worth fixing
  independent of training — most competitive PTCG decks lean heavily on Trainer cards for
  consistency and tempo, and the Strategy category explicitly scores "key-card selection."
- **High decisiveness (~80%) with a still-noisy win rate** suggests the search has converged on
  *a* consistent policy, not necessarily a *good* one yet — confidence and correctness are
  different things at this stage of training.

## 4. Ladder status (Simulation category, `pokemon-tcg-ai-battle`) — updated post-scoring

Two submissions live as of 2026-07-22 (both count toward the "latest 2 active" scoring window),
checked directly on kaggle.com:

| Submission | Description | Status | Score |
|---|---|---|---|
| `submission.tar.gz` | Rule-based priority-heuristic baseline (lethal → attach energy → evolve → best attack → conditional retreat → fallback) | COMPLETE | **243.8** |
| `submission_net.tar.gz` | This net — iteration-36 checkpoint (last promoted), torch-free NumPy forward, determinized MCTS at 32 sims/move, falls back to the baseline agent on any exception | COMPLETE | **451.0** |

The net **nearly doubles the baseline's score** (451.0 vs. 243.8) — a real, ladder-measured
confirmation that the learned agent is meaningfully better than the hand-coded heuristic. Team
rank: **#4,556 of 5,491** (searched by team name "Rubin Sahota" on the public leaderboard). Both
scores sit below μ₀=600 (the starting mean) and well below the leaders (~1,050–1,180) — the net
is a clear improvement over our own baseline, but not yet competitive against the field.

### What the actual match replays show

Opened the Kaggle-hosted PTCG visualizer (`ptcgvis.heroz.jp`) for the net's 3 most recent
episodes at review time (vs. opponents "naoki", "nimous", "AibePC") — **all 3 were losses**.
Reading the logs and final board states directly (not inferred):

- **Every opponent deck seen runs Trainer cards; ours runs none.** naoki played Team Rocket's
  Archer/Giovanni/Proton/Spidops + a Stadium (Team Rocket's Factory) + a Supporter (Lillie's
  Determination) + Team Rocket's (special) Energy. nimous's active was a **260 HP Crustle**
  behind a Stadium (Battle Cage). AibePC had two Tool cards (Premium Power Pro) and an Item
  (Dusk Ball) in hand. Every single one of these card types is structurally absent from
  `decks/baseline_deck.csv` (confirmed zero Trainer cards, §3) — the agent is losing partly
  because **the deck it's holding is mechanically thinner than what real opponents bring**, not
  only because of play quality.
- **The live search's hidden-information sampling is silently wrong against real opponents.**
  `src/determinize.py`'s `sample_determinization` defaults `opp_deck_list` to **our own deck
  list** when the caller doesn't pass one — and `scripts/build_submission.py`'s net-mode
  `main.py` calls `mcts.search(obs_dict, net, _DECK, ...)` with only our deck. So every MCTS
  simulation during a live match guesses the opponent's hidden hand/deck/prizes by sampling
  from **our Water-deck card pool**, when the real opponent might be running Team Rocket's
  Giovanni or a 260 HP Crustle wall. The *visible* legal options the engine offers are always
  correct (the engine enforces those, not us) — but the simulated hidden-card guesses that
  drive the search's lookahead are built on a false premise against any non-mirror opponent.
  This is a concrete, fixable bug in the live-inference path (self-play training itself is
  unaffected, since self-play *is* a genuine mirror match).
- **One loss (vs. AibePC) ended with both sides at Prize 6/Deck 47/Discard 0** — i.e. essentially
  no cards had been played or prizes taken by either side when the game ended. That's not a
  "we got out-played" pattern, that's consistent with a stall or turn/step-limit ruling. Worth
  investigating directly (e.g. instrument `main.py`'s `agent()` for wall-clock per decision on
  the actual Kaggle sandbox) rather than assuming it's a strategy problem.

None of this contradicts §3's behavioral read (development/attrition-oriented, rarely retreats,
~80% decisive) — it adds *why that might be losing*: a patient, board-building style is a much
weaker plan when the opponent's deck has Trainer-card acceleration and bigger walls, and ours
has neither the cards nor (currently) an accurate read of what the opponent is actually holding.

## 5. Honest limitations / open questions

- 40 iterations is early for AlphaZero-style training; the loss curve is healthy but the win-rate
  signal vs. baseline hasn't crossed into a stable trend yet.
- The behavioral snapshot above is a **current-window sample only** — the replay buffer is a
  sliding window (old games are pruned every iteration), so there's no way to reconstruct how
  the move-mix looked at, say, iteration 10 vs. now. Only win-rate/loss have full history.
- Sims/move for the live submission (32) was chosen conservatively because no per-move time
  limit is documented for the competition; local smoke testing measured ~0.24s average /
  0.45s max per move on the dev machine, comfortably fast, but Kaggle's sandbox hardware is
  unverified against that number.
- Self-play is a mirror match (both seats play the same 60-card deck) — the learned policy has
  never seen an asymmetric matchup, which the real ladder will throw at it immediately.

## 6. Next steps

Reordered given the ladder evidence in §4 — the two concrete, diagnosed gaps first:

- **Fix `sample_determinization`'s mirror assumption for live play.** The submission's `main.py`
  should pass a *non-mirror* prior for the opponent once real opponent cards become visible
  (their played Pokémon/discard are genuinely observable — only their hand/deck order/prizes
  stay hidden), instead of silently defaulting to our own deck's distribution. At minimum,
  sampling the unseen opponent slots from a generic/uniform card pool would be more honest than
  assuming mirror; better still, build the guess from whatever of their cards are already
  visible on board.
- **Deck co-optimization**: the current deck has zero Trainer cards, which real opponents are
  clearly exploiting (§4). Build a deck with Items/Supporters/Tools once the pipeline is stable
  — likely higher-leverage than more training iterations on the current deck alone.
- Investigate the stalled/untouched-prizes loss pattern (§4) — confirm it's not a timing or
  exception issue in the live sandbox before assuming it's a strategy gap.
- Resume training from iteration 40 (paused, not broken) and watch whether the tuned
  sims=64/games=48 setting produces a cleaner upward win-rate trend over the next several
  evaluated iterations.
