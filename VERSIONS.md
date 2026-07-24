# Submission versions

Per `CLAUDE.md`'s Submission policy: 1 submission/day, `latest 2` scored, μ logged here once
Kaggle scores it. Local win rates are the pre-submission regression check, not the ship gate,
from v1 onward (see `CLAUDE.md`'s "Architecture" section, and the `verification-gate` skill,
for why).

## v1 — 2026-07-22

**What shipped:** `agents/search_scorer.py` (search_begin/search_step lookahead + hand-crafted
`evaluate()` over the Dwebble/Crustle wall deck, `decks/crustle_wall_deck.csv`), replacing the
rule-based `src/baseline.py` as the active Simulation-category submission. Built via
`scripts/build_submission.py --mode search_scorer`, packaged as
`submission_search_scorer.tar.gz` (1.905 MiB).

**Weights:** NOT the module's hand-set `WEIGHTS` default. A local bake-off across 3 candidates
(this session's newly hand-fixed weights, the pre-fix hand-set weights, and the Stage 3
CMA-ES-tuned weights — 150 games vs. `baseline` + 100 vs. `random` + 100-game head-to-head per
pair, common seeds across candidates per matchup, `tools/bakeoff.py`) selected the **CMA-ES-
tuned weights** (`runs/tune_run1/winner_weights.json`, baked in via `WEIGHTS.update(...)`
appended to the packaged `search_scorer.py`) on pooled win rate: 0.598 vs. 0.587 (pre-fix) vs.
0.553 (this session's fix) across the whole 450-game slate per candidate. Notably, this
session's hand-designed `turns_to_power`/`wasted_energy` fix did not win the bake-off despite
directly targeting a diagnosed gap — see below.

**Path here (for the Strategy writeup and future autopsies):**
1. Stage 1-2: `SearchScorer` agent + crash-proof local arena (`tools/eval_arena.py`) built;
   ~coin-flip vs. `baseline` (43-51% across early runs).
2. Diagnosis: 86.5% of decisions near-tied (2-ply credit-sharing). Fix attempt (shallower
   search) regressed to 10%; reverted. Tie-break-by-merit fix landed (45%, no real change).
3. Stage 3: threat/bench-attacker/tempo features added (evaluate() could now genuinely
   discriminate — attack-decline dropped 46.5%→12.9%) but win rate stayed flat (~44.5%).
   CMA-ES tuning (`tools/tune_weights.py`, `runs/tune_run1/`) over the enriched feature set
   converged but stalled at ~44% vs. `baseline`, confirming the plateau was feature-
   completeness, not calibration.
4. Final feature cycle: `tools/loss_review.py`'s feature-level trace diagnosis on the
   remaining ~24% energy-starvation rate found 74% were "outscored, weight-imbalance" (not
   horizon-blindness — only 2.7%) and 29% involved energy invested in a pre-evolution Dwebble
   that neither existing feature tracked. Added `turns_to_power` (bigger-weighted, Dwebble-
   inclusive) + `wasted_energy`. Verification: win rate unchanged (44.0%, CI [0.373, 0.509]),
   attack-decline steady (~11%), starvation only modestly down (20.9%, target of "well under
   15%" not met).
5. Per the explicit decision to ship regardless of the 60%-local bar: bake-off across all
   three candidates (above), package + fresh-extraction verify
   (`scripts/verify_submission.py` — 10/10 smoke-test games completed using only the
   extracted tarball, zero illegal moves, 1.905 MiB), submit as v1.

**Local reference numbers for the shipped (tuned) weights** (from the bake-off, n=150/100):
vs. `baseline` 46.0% (69/150), vs. `random` 100% (100/100).

**μ (Kaggle ladder):** 568.6 (as of 2026-07-23 08:32Z, submitted 2026-07-22 17:47:52.847000 -- may still be settling; re-run tools/measure.py to refresh)

**Honest open question for Stage 6:** local win rate vs. two fixed opponents (`baseline`,
`random`) never moved much across three real, evidenced fixes, while measured *behavior*
changed substantially each time. This suggests `baseline`/`random` may be too narrow a test to
resolve further — real ladder opponents and episode replays are the next real signal.

## v4 — 2026-07-24

**What shipped:** `agents/search_scorer.py`, same shipped weights as v1
(`runs/tune_run1/winner_weights.json`, no new WEIGHTS features this cycle), plus two targeted
mechanism changes:
1. `_prefer_continuing_over_end` — widens the near-tie window (`END_NEAR_TIE_REL_THRESHOLD =
   0.05`) for turns ≤4 (`END_EARLY_MAX_TURN`) so the agent stops picking "end turn" over a
   near-equally-scored alternative this early, when ending early is very rarely correct.
2. `_PLAY_CARD_TIE_PRIORITY` — a play/card-kind tie-break table over 6 specific cards
   (Pokégear 3.0, Jumbo Ice Cream prefer-first; Mega Kangaskhan ex, Boss's Orders, Switch, Hand
   Trimmer prefer-last), derived from the expert-corpus screen's per-card disagreement
   direction.

**Path here — the verification-gate screen (26,311-decision expert+control corpus re-diff vs.
a frozen v4 snapshot, `tools/decision_diff.py`/`tools/cluster_divergences.py`/
`tools/reconstruct_decision.py`):**
1. **Targeted population (the causal signal):** the 34-case "our agent ends early when the
   human expert didn't, turn ≤4" population shrank to 9 cases (73.5% reduction) — a clean,
   unambiguous pass, and the strongest single piece of evidence behind this change.
2. **Tie-tier (6 named cards):** aggregate agreement with the expert corpus on these cards'
   ties rose 14.1%→18.0% (830/5897 → 903/5011), but not uniformly: Pokégear 3.0, Jumbo Ice
   Cream, and Mega Kangaskhan ex all improved; Boss's Orders was flat; **Switch and Hand
   Trimmer both got worse** (9.3%→7.1%, 11.8%→10.7%). Net positive, not a clean win.
3. **Untargeted cluster stability:** clusters neither change's mechanism should touch
   (`ability`/early, mid, late) moved substantially anyway — agreement roughly doubled
   (13.6-13.9%→23-25%, max delta 13.65pp) — larger than the targeted `play`/`card` clusters'
   own movement. Favorable direction, but unexplained by either change's stated mechanism;
   most likely hidden-world-sampling noise between separate run invocations (this codebase has
   no `battle_start()` seed — see `CLAUDE.md`'s method lessons), not confirmed.
4. **B0 self-test** (45 real v1 episodes replayed against the v4 snapshot): 86.6% adjusted
   agreement (1064 raw-agree + 77 near-tie + 14 resample-reachable of 1333 decisions),
   against a ≥87.4% bar (v1's own self-consistency baseline) — **misses by 0.8pp.**

Per the screen's own strict decision rule, criteria 2-4 above do not clear cleanly — only
criterion 1 is a clean pass. **The explicit decision this cycle: ship on criterion 1 plus a
faster, cheaper gate (below) rather than requiring the full screen to pass**, given deadline
pressure and criterion 1's strength as the causal signal. Criteria 2-4 are logged here honestly
as **unresolved pending a noise-floor measurement** (a same-code v1-vs-itself B0 re-run,
`runs/decision_diff/b0_selftest_ckpt_v1_noisefloor.json`, launched 2026-07-24, still running in
the background as of shipping — results TBD in a future update), not hidden or treated as a
pass.

**Smoke test** (`runs/v4_gate/smoke_test.jsonl`, 50 games vs. `baseline`): PASS — 0 crashes, 0
timeout-kills, 0 illegal selections, max single-game elapsed time 21.2s (well under the 120s
per-game timeout and the 2000s episode `runTimeout`). Win rate 42.0% (21/50) — not gate-relevant
here, this run only checks for breakage.

**Submission package:** built via `scripts/build_submission.py --mode search_scorer`
(`submission_search_scorer.tar.gz`, 1.909 MiB), verified via `scripts/verify_submission.py`
(10/10 fresh-extraction smoke-test games, every move legal) — same process as v1's own
packaging verification.

**Gate Leg A** (`runs/v4_gate/leg_a_v4_vs_baseline.jsonl`, v4 vs. `baseline`, 400 games,
local): **173/400 = 43.25%, 95% Wilson CI [38.5%, 48.1%]**, compared against v1's own fresh
number this cycle (45.0%, n=400, CI [40.1%, 49.9%]; not re-run). The two CIs overlap over most
of their range and the point estimates differ by <2pp — read as within this codebase's known
run-to-run noise (no `battle_start()` seed), not a regression, though the lower bound sits
~1.6pp under the plan's approximate "~40%" guardrail. The softest spot in this cycle's
evidence; see Leg B below for the more direct check.

**Gate Leg B** (`runs/v4_gate/leg_b_v4_vs_v1.jsonl`, v4 vs. v1, 400-game local head-to-head via
the new `--candidate-snapshot`/`--opponent-snapshot` mechanism added to `tools/eval_arena.py`/
`tools/_eval_worker.py` this cycle): **222/400 = 55.50%, 95% Wilson CI [50.6%, 60.3%]** — v4
clearly ahead of v1, not losing. The more decisive of the two legs since it's a direct,
confound-free comparison against the exact code being replaced.

**Mechanism metrics:**
- End/early rate in live play: **0 of 7,459** turn≤4 decisions (both legs combined) chose "end"
  when a non-end option was legally available — strong direct confirmation of Change 1's
  intended mechanism, consistent with the screen's 34→9 result.
- Loss-profile comparison (`tools/loss_review.py` on both legs' replay logs): attack-decline
  6.7%/7.3%, energy-starvation 20.9%/20.0%, evolve-decline 23.1%/26.4%, near-ties 87.1%/88.1%
  (Leg A/Leg B) — all in line with v1's own historical numbers (starvation 20.9%, attack-decline
  ~11%), no new pattern introduced.
- 20-loss spot-autopsy (sampled from Leg B, the harder opponent): all 20 read as ordinary game
  endings across a normal turn-length spread (5-23 turns); no hangs, loops, or degenerate
  always-same-move patterns.
- Time-budget: max single-game elapsed time 34.6s (Leg A) / 20.5s (Leg B), both ≪120s per-game
  timeout and ≪2000s episode `runTimeout` — no risk despite Change 1 lengthening early-game
  turns by design.

**Ship decision:** SHIP. Smoke test clean, Leg B (the decisive leg) clearly favorable, Leg A
within noise of v1 despite one soft CI bound, no new failure mode, no time-budget risk. Screen
criteria 2-4 stay logged above as open/unresolved rather than folded into this decision.
Submitted 2026-07-24 as Kaggle submission **54950362** (`SubmissionStatus.COMPLETE`). v1 stays
active on the ladder as the A/B control (both submissions' μ tracked separately in
`runs/mu_history.jsonl` going forward).

**μ (Kaggle ladder):** 600.0 (as of 2026-07-24 11:53Z, submitted 2026-07-24 11:51:42.633000 -- may still be settling; re-run tools/measure.py to refresh)
submission per CLAUDE.md's `N(μ, σ²), μ₀=600`, not yet real ladder signal; re-run
tools/measure.py to refresh once games accumulate)
