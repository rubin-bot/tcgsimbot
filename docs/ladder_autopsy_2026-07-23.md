# Ladder autopsy report

measure_state.json: checked_at=2026-07-23T07:02:13.188145+00:00, mu=573.3, rank=3616, submission_count=2
episode datasets scanned since 2026-07-22: ['2026-07-22'] (not yet published: ['2026-07-23'], failed: [])
episodes found per day: {'2026-07-22': 0}

## Parse summary
parsed OK: 0  |  not-ours (unexpected if >0): 0  |  failed to parse: 0

## Why zero ladder games, despite an active, scored submission
Our leaderboard score (573.3) has already moved away from the mu0=600 starting mean across several checks, which means SOME real rated games have been played and counted -- but none of them showed up in the one published day we could scan ({'2026-07-22': 0}). Two plausible explanations, not mutually exclusive: (1) each daily dataset appears to be a large sample of that day's games across ~5500+ teams, not a literal complete record of every rated game played -- with v1 only active for the last ~6 hours of 2026-07-22 (submitted 17:47), its share of that day's games was small and may simply not have been sampled into this particular dump; (2) there may be a publication lag between a game affecting the live score and that game's replay becoming available in a dataset. Re-running tools/measure.py after 2026-07-24's dataset publishes (covering the rest of 2026-07-22 and all of 2026-07-23) is the natural next check.

## Comparison vs. local dry-run (2026-07-22) -- ladder side not yet available
local losses analyzed: 42
local attack-decline rate: 14/147 (9.5%)
local evolve-decline rate: 12/47 (25.5%)
local starvation events: 32 (dwebble-targeted: 14)
local near-tie rate: 842/961 (87.6%)
ladder side: no games yet this cycle -- see 'Why zero ladder games' above.

## Ranked fix candidates (LOCAL evidence only -- ladder confirmation pending)
NOT implemented this cycle. Each would be verified via scripts/build_kernel_bakeoff.py's 400+-game kernel gate before shipping, and ideally re-confirmed against real ladder losses once they exist (this cycle couldn't -- 0 ladder games available).
1. **Tie-break-on-starvation fix** -- evidence: local dry-run found 14/32 (43.8%) attacker-starvation cases in the 'tied_and_lost' bucket (agent's own tie-break should have caught these), plus an 87.6% overall near-tie rate. Expected impact: HIGH if real -- this is the single largest local anomaly across two diagnosis cycles now. Verify: kernel bake-off of a revised tie-break rule against shipped v1, 400+ games.
2. **Evolve-decline reduction** -- evidence: 25.5% of legal evolve opportunities declined in local losses (12/47). Expected impact: MEDIUM -- delayed evolution compounds into the energy/tempo problems already diagnosed. Verify: same kernel gate, isolate this fix alone to avoid confounding with #1.
3. **Re-run this exact report once real ladder losses exist** -- not a code fix, but the highest-priority NEXT STEP: confirm whether local-only findings (#1, #2) actually hold against real opponents before spending a FIX cycle on them.
