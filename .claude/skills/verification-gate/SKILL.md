---
name: verification-gate
description: The Pokémon TCG AI Battle Challenge's fix-verification protocol — the 400-game gate, mechanism-metric requirement, coverage-estimate requirement, and ship/no-ship decision rule. Trigger whenever proposing, verifying, or shipping a change to agents/search_scorer.py (or any agent), or when the user asks to "prove", "verify", "gate", or "bake off" a fix before shipping.
---

# Verification gate

A candidate fix (tie-break change, sampling change, weight change, new eval feature) earns the
right to ship **only** by clearing this gate. "It looks correct in isolated repro cases" is
never sufficient on its own — see the coverage-estimate requirement below, added after v2's own
gate result.

## 1. Coverage-estimate requirement (do this BEFORE any kernel spend)

Isolated repro cases (a handful of real board states where the bug is clearly reproducible) only
prove the fix is *correct*, not that it's *impactful*. Before spending kernel time, check the
fix's effect on the **full real population** it targets, not just the repro cases:

- Reuse `tools/measure_near_tie_hypothesis.py` (replay-and-vote: N repeated calls to
  `choose_action()` per real decision) + `tools/analyze_near_tie_results.py` (flip rate, margin
  distribution, tied-and-lost classification) over the full decline/starvation decision set —
  not just the 3-5 hand-picked repro cases.
- **Metric care**: "flip rate vs. historical ladder choice" and "self-consistency across
  repeated replays" are DIFFERENT metrics that get conflated easily — see the method lesson
  below. Use self-consistency (mode agreement rate across repeated replays of the same decision)
  to test whether a fix reduces *noise*; only use vs-history flip rate to characterize how often
  the agent now disagrees with old (possibly-mistaken) choices, never as a stability metric.
- If the population-level effect doesn't move in the intended direction: **stop and debug before
  any kernel spend** — this is exactly the failure mode a fix can pass at the repro-case level
  while failing here (v3 cycle, 2026-07-23: N=8 voting was provably correct on all 3 repro cases
  but only produced a modest population-level effect).

## 2. The 400-game gate

- **Legs**: local (`tools/eval_arena.py`, ≤2 workers per the hardware rule) for anything that
  needs to be run twice with different code snapshots (e.g. "v_old vs baseline" and "v_new vs
  baseline" — swap `agents/search_scorer.py`'s content between runs, always via a safe
  freeze-then-restore procedure, never leaving the working tree mid-swap: freeze the current
  tree to `runs/<name>/search_scorer_snapshot.py` FIRST — this doubles as your restore point —
  then overwrite, run, then copy the snapshot back). Kernel
  (`scripts/build_kernel_vs_baseline.py`, or `scripts/build_kernel_head_to_head.py` for a direct
  agent-vs-agent match) for anything needing 4-worker parallelism or a genuine head-to-head.
- **400 games**, checkpointed every 20 (both local and kernel infra already do this).
- **95% CI** via normal approximation on each run's own win rate — report every run's own CI,
  compare via overlap. **Do NOT describe separate runs as "paired by seed"** — see the method
  lesson below; treat every run as an independent sample.
- **Mechanism (primary signal)**: reuse `tools/loss_review.py`'s analyses (`analyze_
  attack_availability`, `analyze_attacker_starved`/`analyze_energy_routing_detail`,
  `analyze_evolve_misplay`, `analyze_near_ties`) **unmodified** on each run's own replay/trace —
  this is the real test of whether a fix works, not the win-rate number alone. It must move
  substantially in the intended direction, not just be "up or flat."
- **Regressions**: attack-decline / evolve-decline / near-tie rates, plus a hand spot-autopsy of
  ~20 losses from the new candidate for any *new* failure mode (crashes, degenerate games,
  timeout/time-guard activations if the fix touches per-decision timing).

## 3. Decision rule

Ship only if **every** criterion holds: the targeted mechanism collapses (not just moves), win
rate is up or flat-within-CI, zero new failure modes, and (if a head-to-head leg was planned)
that leg actually completed. If a leg is unavailable (e.g. kernel infra failure) it counts as a
failed criterion, not a skipped one — do not ship on partial evidence. If any criterion fails:
**stop and report the real numbers** — do not ship a fix that helped the diagnosed mechanism but
not the outcome, or vice versa, without saying so plainly.

## 4. Ship checklist (only if the gate passes)

`scripts/build_submission.py --mode search_scorer` → `scripts/verify_submission.py`
(fresh-extraction smoke test) → `scripts/submit.py -m "..."` → new `## vN — <date>` entry in
`VERSIONS.md` → `tools/measure.py` to log μ for **all** active submissions into
`runs/mu_history.jsonl` (the old version stays active as a free ladder A/B control) → write the
full report to `docs/<name>_<date>.md` → commit + push.

## Method lessons (add to as new gates surface them)

- **Signature counts are correlational.** A repro case being real and correct doesn't mean it's
  representative — always run the coverage-estimate step above before kernel time.
- **Seeds don't pin game-engine determinism.** `cg.game.battle_start()` takes no seed argument;
  confirmed (2026-07-23) that same-code-same-seed local-vs-kernel and local-vs-local runs agree
  at chance level (30-51%), not near-100%. Treat every 400-game run as an independent sample —
  the CI-overlap comparison method this skill already prescribes is the correct treatment
  regardless, so this doesn't invalidate past conclusions, only the "paired" narrative.
- **"Flip rate vs. history" is not a stability metric.** A decisive, consistent preference that
  differs from an old (possibly wrong) historical choice looks identical to genuine noise under
  a vs-history metric. Use self-consistency (mode agreement across repeated replays) instead
  when the question is "did this fix reduce noise."
- **A kernel that hangs with zero output (not even a partial log) after real, proven-correct
  local testing is worth at most 2-3 retry attempts** (a fresh push, then one targeted
  structural fix) before treating it as an unresolved infra gap and proceeding on the evidence
  already available — don't burn unlimited kernel spend chasing a silent hang the CLI's
  `status`/`output` commands can't diagnose further.
