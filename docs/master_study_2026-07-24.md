# Master-study analysis cycle — expert corpus, decision-diff, sparring pool — 2026-07-24

**Analysis only — no agent, weight, or deck changes were shipped this cycle.** This report
mines real games played by genuinely competitive Crustle-archetype ladder teams (not our own
games, not `baseline`) to find concrete decisions where our current agent (v1, the actual
active Kaggle submission) would choose differently than a real strong player did, ranks those
divergences by evidence strength, and separately calibrates a meta-representative local sparring
benchmark against the two real numbers already on record. Every finding below is evidence for a
*future* verification-gate cycle, not something shipped now.

## Workstream A — expert corpus

`tools/identify_expert_teams.py` reproduced `docs/meta_report_2026-07-22.md`'s own archetype
classification directly against `runs/meta_mining/team_game_signatures.jsonl` (1 scanned day,
2026-07-22, 9278 records, 142 teams): **10 distinct Crustle-archetype teams**, split 4 top-100 /
6 in the 600+ band. **No team in the 400-600 or below-400 band exists in this single day's
data** — a real corpus-composition gap, not something this cycle could paper over (per
user-confirmed decision 2026-07-23).

| Group | Teams (by game count) | Total team-episode records |
|---|---|---|
| Expert | LiamK (200), 懒惰的金枪鱼 (176), Budew (51), 西松大祐 (38), gecogeco (24) | 489 |
| Control (relative, NOT genuinely low-rated — 1-7 games each, small-sample) | denden12 (1), SantaClaws 🎅 (1), MPGaming (1), koala_bear "もりたにあん" (2), palsystem (7) | 12 |

`tools/fetch_expert_corpus.py` re-scanned the same already-probed bulk dump
(`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22`), this time saving **full episode JSON**
(not just archetype signatures) for every episode touching any of these 10 teams: **492 unique
episodes**. Clears the ~30-game viability floor by roughly 16×; no need to scan additional days
for volume this cycle.

**Correction to the literal task wording** (confirmed via research, not assumed): no other
team's submission ID is discoverable anywhere in this API (checked the leaderboard CSV, episode
JSON, and the `episodes` CLI response — none carry it). The working equivalent used instead:
`kaggle competitions replay <episode_id>` fetches any episode by ID regardless of submission
ownership, and the bulk dump already supplies real episode IDs for every team — no submission-ID
lookup was ever needed.

## Workstream B0 — reconstruction self-test (hard prerequisite)

`tools/reconstruct_decision.py` generalizes the proven `tools/replay_ladder_decision.py` /
`tools/measure_near_tie_hypothesis.py` primitives (both already accepted an overridable
`team_name` parameter — far less new code was needed than expected) to reconstruct an arbitrary
team's real decision from a downloaded episode, using only that seat's own observation (the
same info-hiding invariant `tests/test_obs.py` already asserts).

**Self-test procedure**: reconstruct every real ACTIVE decision from our own 45 v1-attributed
episodes (`runs/our_episodes/`, filtered by `V1_SUBMISSION_ID` — a real bug caught and fixed
mid-run: the raw 76-episode folder blends v1 with the older, architecturally-different
deprecated net-checkpoint agent, and an unfiltered first pass gave a misleadingly low 79.2%),
run the frozen v1 snapshot with its **actual shipped CMA-ES-tuned weights** (another real bug
caught and fixed: `runs/v2_tie_break/search_scorer_v1_snapshot.py` alone reflects the module's
*default* `WEIGHTS`, not what was really submitted — confirmed against `submission/
search_scorer.py`, the literal packaged file, which bakes in `runs/tune_run1/winner_weights.json`
via `WEIGHTS.update(...)`), and compare the reconstructed choice against history.

| | value |
|---|---|
| decisions checked | 1333 (45 real v1 episodes) |
| raw agreement | 81.7% (1089/1333) |
| near-tie disagreements (exact score tie, expected noise) | 54 |
| genuine (non-near-tie) disagreements | 190 |
| of those, reachable via 20 fresh resamples (single-sample-variance noise, not a bug) | 22 |
| never reachable across 20 resamples | 168 |
| **adjusted agreement rate** | **87.4%** |

Below the task's literal 90% target. Root-caused as far as mechanically possible: code (diffed
byte-identical to `submission/search_scorer.py`), deck (`decks/crustle_wall_deck.csv`, byte-
identical to `submission/deck.csv`), and weights (now correctly merged) are all confirmed
correct; a hand-traced example (episode 87507277, step 48 — historical choice: attach a 4th
energy to an already-3/3-energy attacker instead of attacking a 210HP opponent) showed no
indexing bug when traced against the raw JSON by hand. **User-confirmed decision (2026-07-23):
proceed to B1, treating 87.4% as clearing the intent of the bar** — v1 has no seed control
(`src/determinize.py::sample_determinization` draws fresh each call) and real, already-documented
run-to-run variance (`docs/near_tie_measurement_2026-07-23.md`'s own 93.6%-flip-rate finding);
the residual gap looks like this same property, not a reconstruction defect, though this is a
judgment call rather than a mechanically-closed proof.

## Workstream B1/B2 — decision-diff and ranked clusters

`tools/decision_diff.py` ran our v1 (same validated snapshot+weights) against every real
decision in the corpus: **26,311 expert decisions, 681 control decisions**. Deck-list caveat
(documented, not hidden): both sides pass **our own** deck as an approximation of the acting
player's real list (their exact decklist isn't recoverable — see Workstream C) — a closer
approximation here than for a different archetype, since experts run the same Crustle/Dwebble
line, but still an approximation, and it affects hidden-zone sampling for both players.

`tools/cluster_divergences.py` grouped by (decision kind, phase). Ranking (stated explicitly):
`rank_score = n_disagreements × control_multiplier` (2.0× if the divergence is
rating-correlated — control agrees with US more than experts do on the same cluster — 0.5× if
it's a style difference present in control too, 1.0× if there's not enough control data, n<5).
Real bug caught and fixed mid-cycle: an early version of the outcome-linkage forward-tracer
walked *raw* step indices ahead instead of *decision-level* indices (matching
`tools/analyze_near_tie_results.py::analyze_outcome_linkage`'s real convention) and produced an
all-zero delta for every single decision — corrected before trusting any of the numbers below.

### Top 5 ranked clusters (all 5 evidence fields)

| Rank | Cluster | n disagree / total | coverage/game | mean margin | near-tie : large-margin | control filter | outcome linkage (mean Δ prize-tempo, next 3 decisions) |
|---|---|---|---|---|---|---|---|
| 1 | `play`/mid | 2189/2568 (85.2%) | 4.63 | 0.19 | 1312 : 876 | style (0.5×) | −0.052 (135+/149−/1905=0) |
| 2 | `end`/early | 532/820 (64.9%) | 1.10 | 0.04 | 437 : 95 | **rating-correlated (2.0×)** | **−0.194 (0+/59−/472=0)** |
| 3 | `play`/late | 2126/2481 (85.7%) | 5.40 | 0.20 | 1302 : 821 | style (0.5×) | −0.007 (144+/118−) |
| 4 | `play`/early | 1606/2120 (75.8%) | 3.30 | 0.15 | 1142 : 464 | style (0.5×) | −0.044 (0+/38−) |
| 5 | `card`/mid | 758/2425 (31.3%) | 1.60 | 0.02 | 737 : 21 | ambiguous (1.0×) | −0.078 (92+/97−) |

**Reading the top cluster (`play`/mid, `card`/mid)**: a striking majority (1312/2189 ≈ 60% for
`play`/mid, 737/758 ≈ 97% for `card`/mid) are **exact score ties** (`historical_score ==
our_score` to full float precision) where our tie-break simply picked a *different, equally-
scored* Trainer-card option than the real expert did. This is the same shape of bug v2 fixed for
`attach` — same-kind exact ties falling through to arbitrary engine list-order — but showing up
for `play`/`card` kind decisions, which v2 never touched. Hand-traced example (episode
87362960, step 64, turn 6): options 0/1/2/4 all scored `5.2609...` identically; the real expert
chose index 1, our tie-break (`_TIE_BREAK_PRIORITY`'s catch-all ordering for `play`/`card` kinds
has no further discriminator) chose index 0.

**Reading `end`/early — the single most evidence-backed cluster**: this is the ONLY top-5
cluster that is **rating-correlated** (control's own coverage rate, 42.9%, is meaningfully
*lower* than experts', 64.9% — i.e. weaker control-tier players' real choices look more like
ours than strong experts' choices do) AND has a clean, one-directional outcome-linkage signal
(**all 59 traced non-zero cases are negative** — when the real expert chose to end their turn
early instead of continuing, their prize-tempo trended worse over the next few decisions). Hand-
traced example (episode 87362960, step 19, turn 2): our agent preferred `play` (score 1.537)
over the historical `end` (score 1.356) — a real, if modest, margin, not a near-tie.

Full 33-cluster ranked table with all fields: `runs/decision_diff/clusters.json`.

## Workstream C — meta-representative sparring pool

**C1** `tools/reconstruct_archetype_deck.py` pooled the real top-3 Alakazam teams (Yushin Ito
434 games, Majkel1337 214, haggle 139 — 787 total) and top-3 Munkidori teams (Rmy 405, Luca 368,
jiatu.l 323 — 1096 total) from `team_game_signatures.jsonl`. Card *presence* is real, mined data;
copy counts are inferred via a documented frequency-tier rule (meta_miner.py only ever stores
each game's top-3 cards, never full decklists — copy counts are not recoverable from this source
at all). Full assumptions: `decks/alakazam_sparring_deck_ASSUMPTIONS.md`,
`decks/munkidori_sparring_deck_ASSUMPTIONS.md`. **Not a competitive-accuracy claim.**

**C2**: `agents.search_scorer.make_agent()` piloting each reconstructed deck with default
weights — confirmed crash-free over 50 games each vs. `baseline` (Alakazam: 2W/48L, 4.0%;
Munkidori: 5W/45L, 10.0%). Both win rates are **very low**, consistent with the documented,
pre-flagged caveat: ~9 of 19 weighted `evaluate()` features (including the largest,
`turns_to_power`) are Crustle/Dwebble-specific module constants and become dead weight piloting
a different archetype. These sparring bots are much weaker than a genuine expert pilot of the
same deck.

**C3**: extended `tools/eval_arena.py` additively (`--opponent-pool`, a weighted per-game
opponent draw seeded off `--seed + game index`; existing single-opponent mode fully regression-
tested, unchanged). Ran v1 (the real submitted code+weights, confirmed via the same
`submission/search_scorer.py` used in B0/B1) vs. a pool weighted ~1/3 Alakazam-sparring / ~1/3
Munkidori-sparring / ~1/3 `baseline`, 400 games:

| | win rate | 95% CI | n |
|---|---|---|---|
| **Overall vs. meta pool** | **65.0%** | [60.3%, 69.7%] | 400 |
| vs. `baseline` component | 37.6% | [29.1%, 46.1%] | 125 |
| vs. Alakazam-sparring component | 89.0% | [83.7%, 94.2%] | 136 |
| vs. Munkidori-sparring component | 66.2% | [58.3%, 74.1%] | 139 |

**Which local benchmark better predicts the real ladder rate (53.3%, `docs/
ladder_attack_decline_diagnosis_2026-07-23.md`)?** Plain `baseline`-only (45.0%, 400 games,
`docs/v3_report_2026-07-23.md`) is **closer** to 53.3% (gap ≈ 8.3pp) than the new meta-pool
number (65.0%, gap ≈ 11.7pp, in the *opposite* direction — overestimating rather than
underestimating). The meta-pool's archetype-matching concept is sound, but in practice it's
undermined by C2's own documented weakness: our sparring bots are far below real expert-level
play, so beating them heavily doesn't mean much. **Honest finding: this cycle's meta-pool
implementation does not out-predict the plain baseline benchmark**, despite matching the real
archetype distribution — the sparring-bot quality gap dominates.

## Ranked v4 candidates (evidence × coverage, no implementation this cycle)

1. **Extend the tie-break to `play`/`card`-kind exact ties, generalizing v2's approach beyond
   `attach`.** Change: `agents/search_scorer.py::_TIE_BREAK_PRIORITY`/`_tie_break_key` gains a
   secondary discriminator for `play`/`card` options (currently falls straight to arbitrary
   engine list-order once kind-priority ties, same shape as the pre-v2 `attach` bug). Coverage:
   **2189 divergent decisions/game-cluster in `play`/mid alone** (4.63/game), plus 2126 in
   `play`/late and 1606 in `play`/early — by far the largest volume of any cluster, and
   **~60-97% of the disagreements in the top 5 clusters are exact ties**, meaning a tie-break
   fix (not an eval change) directly targets most of this evidence. Evidence strength: very
   high coverage, but **not** rating-correlated (0.5× control multiplier — control also
   diverges at a similar rate, so this reads as a style/preference gap rather than a proven
   skill gap) — the *volume* is the strongest evidence here, not the control-correlation.
   Predicted effect: per the v2 lesson (`verification-gate` skill), a tie-break fix's real
   population-level effect must be measured directly (coverage-estimate step) before trusting
   its size from repro cases alone — this candidate has that population-level evidence already
   in hand (unlike v2's original repro set), which is new information relative to prior cycles.

2. **Investigate `end`/early divergence specifically — the only rating-correlated top cluster.**
   Change: unclear yet whether this needs a tie-break change or an `evaluate()` feature change
   (needs the deeper dive this cycle didn't have scope for) — but the *signal* that something is
   wrong here is the strongest of any cluster found: real skilled players end their turn early
   far less often relative to us than weaker players do (rating-correlated, 2.0× multiplier),
   and their real games show a clean, one-directional cost to doing so (mean prize-tempo
   Δ = −0.194 over the next 3 decisions, 59/59 traced non-zero cases negative, zero positive).
   Coverage: 532 divergent decisions, 1.10/game — smaller volume than candidate 1, but the
   cleanest, most one-directional evidence of any cluster. Predicted effect: likely a real,
   if narrower, win-rate gain; needs its own repro-case + coverage-estimate cycle before gating.

3. **Re-attempt the meta-representative sparring pool with stronger opponent bots before
   trusting it as a benchmark.** Not an agent change — a *tooling* candidate. The pool concept
   (matching real archetype distribution) is sound and now has working infrastructure
   (`--opponent-pool`), but C3 showed it currently predicts the real ladder rate *worse* than
   the plain baseline. Before relying on it for future gates, either tune the sparring decks'
   own weights (CMA-ES per archetype) or source stronger opponent logic; until then, keep using
   `baseline`-vs-real-ladder as the primary local/real correspondence check.

## What's committed this cycle

Everything above (`tools/identify_expert_teams.py`, `tools/fetch_expert_corpus.py`,
`tools/reconstruct_decision.py`, `tools/decision_diff.py`, `tools/cluster_divergences.py`,
`tools/reconstruct_archetype_deck.py`, `decks/alakazam_sparring_deck.csv` +
`decks/munkidori_sparring_deck.csv` (+ assumptions docs), `tools/eval_arena.py`'s additive
`--opponent-pool` extension, and this report). **No changes to `agents/search_scorer.py`, no
weights file, no deck used in the real submission, no ship action.** v1 remains the active
Kaggle submission, unchanged.
