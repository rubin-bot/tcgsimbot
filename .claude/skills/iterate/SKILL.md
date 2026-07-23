---
name: iterate
description: Runs the Pokémon TCG AI Battle Challenge's Stage 6 daily iterate loop (measure ladder μ, diagnose losses, propose+prove one fix, ship). Trigger on the user saying "iterate" (bare word or /iterate) in this repo, or asking to check the ladder / run an autopsy / do the daily cycle.
---

# Iterate (Stage 6 daily loop)

Full procedure, tool commands, and file layout live in `ITERATE.md` at the repo root — read
it and follow it exactly. Do not re-derive the loop from scratch or improvise a different
shape; `ITERATE.md` is the source of truth and gets updated as the tooling evolves.

Quick shape (see `ITERATE.md` for the real commands and caveats):

1. **MEASURE** — `tools/measure.py`: current μ into `VERSIONS.md`, newest episode dataset for
   our games into `runs/our_episodes/`.
2. **DIAGNOSE** — `tools/autopsy.py --source auto`: ranked shortcomings report (ladder if we
   have real losses, local `baseline`/`random` fallback otherwise). Show it to the user.
3. **FIX** — one candidate fix for the top failure mode, proven over 400+ games via
   `scripts/build_kernel_bakeoff.py` (a Kaggle kernel attached to the competition — never
   re-upload the SDK/card data ourselves). No proven win, no ship; try the next candidate. Use
   the **`verification-gate`** skill for the full protocol (coverage-estimate requirement,
   mechanism-metric requirement, decision rule) — don't improvise a lighter version of it.
4. **SHIP** — `scripts/build_submission.py` → `scripts/verify_submission.py` →
   `scripts/submit.py`, log in `VERSIONS.md`, commit + push, tell the user to come back
   tomorrow.

Constraints that apply throughout (repeated in `ITERATE.md`, from `CLAUDE.md`): 1
submission/day, max 2 local simulator workers, everything resumable, stream to disk not RAM.

If `runs/measure_state.json` doesn't exist yet or looks stale, run MEASURE before DIAGNOSE —
`tools/autopsy.py --source auto` depends on it to pick ladder vs. local mode.

For anything involving real Kaggle episode data (fetching, parsing, replay-and-vote checks, or
ladder-wide meta-mining beyond our own games), use the **`ladder-analysis`** skill — it has the
parser gotchas (ACTIVE filter, BOM, bulk-dump subsample caveat) and the exact tool commands.
