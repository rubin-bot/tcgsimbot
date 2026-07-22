# Submission versions

Per `CLAUDE.md`'s Submission policy: 1 submission/day, `latest 2` scored, μ logged here once
Kaggle scores it. Local win rates are the pre-submission regression check, not the ship gate,
from v1 onward (see `CLAUDE.md`'s "Local-arena gate retired as of v1" for why).

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

**μ (Kaggle ladder):** 500.9 (as of 2026-07-22 18:24Z, submitted 2026-07-22 17:47:52.847000 -- may still be settling; re-run tools/measure.py to refresh)

**Honest open question for Stage 6:** local win rate vs. two fixed opponents (`baseline`,
`random`) never moved much across three real, evidenced fixes, while measured *behavior*
changed substantially each time. This suggests `baseline`/`random` may be too narrow a test to
resolve further — real ladder opponents and episode replays are the next real signal.
