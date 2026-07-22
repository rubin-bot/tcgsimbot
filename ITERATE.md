# The "iterate" loop (Stage 6)

Whenever the user says "iterate" (or runs `/iterate`) in any session, execute this loop.
Per `CLAUDE.md`'s Submission policy: **1 submission/day**, so don't run SHIP more than once
per calendar day even if MEASURE/DIAGNOSE/FIX get re-run. Enforce the Hardware rules on every
step: max 2 local simulator processes at once, everything resumable, stream to disk not RAM,
print progress.

## 1. MEASURE

```
.venv/Scripts/python tools/measure.py
```

- Pulls the current submissions list via the Kaggle API and updates the shipped version's
  **μ (Kaggle ladder)** line in `VERSIONS.md` in place, with an "as of <timestamp>" note (μ
  moves over time -- a bare number a few hours after submission is not a settled result).
- Looks for the newest daily episode-replay dataset
  (`kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`, published ~00:00 UTC the day *after*
  the games it contains -- so today's games are never in today's dump) and, if any of our
  games are in it, extracts just those episodes to `runs/our_episodes/<date>/` (gitignored).
- Writes `runs/measure_state.json` recording what it found, so `tools/autopsy.py --source
  auto` knows whether ladder data exists without re-hitting the API itself.
- If our team has zero episodes in the newest available dump (submission too recent, or ladder
  hasn't matched us into enough games yet), this is expected and not a bug -- DIAGNOSE just
  falls back to local mode.

## 2. DIAGNOSE

```
.venv/Scripts/python tools/autopsy.py --source auto
```

- Builds on `tools/loss_review.py`'s six analyses (attack-decline, attacker-starved,
  energy-routing-detail, avoidable-KOs, evolve-misplay, near-ties).
- Ladder mode (real losses, real opponent decks) additionally reports opponent archetypes
  seen and win/loss by archetype -- the thing local `baseline`/`random` structurally cannot
  teach us. Two of the six analyses (energy-routing-detail, near-ties) need our own
  `evaluate()` scores, which Kaggle never records, so they're LOCAL-ONLY and skipped (loudly)
  in ladder mode.
- Local-fallback mode runs `tools/eval_arena.py` fresh against the full local opponent slate
  (`baseline`, `random` -- there is no third local opponent today) with `--replay-out`, then
  runs the same six analyses on those losses.
- Show the printed report to the user. Rank failure modes by frequency across whatever
  sources were available.

## 3. FIX

- Pick the single highest-impact failure mode from the DIAGNOSE report; implement one fix.
- Prove it beats the currently-shipped version over **400+ games with common seeds**, run as
  a Kaggle notebook (not on this laptop) via `scripts/build_kernel_bakeoff.py` -- the kernel
  attaches to the competition itself (`competition_sources`) so it gets Kaggle's own mounted
  SDK/card data rather than us re-uploading a copy (the SDK/card data are competition
  "Pokémon Elements" and `CLAUDE.md`'s Data-use constraints forbid redistributing them).
- **No win at 400+ games = no submission.** Try the next candidate fix instead of shipping a
  fix that didn't prove out. A same-day local regression check against `baseline`/`random`
  (`tools/eval_arena.py`, per the existing pre-submission convention) still runs before
  packaging, but it is a regression check, not the bar -- the kernel bake-off is the bar.

## 4. SHIP

```
.venv/Scripts/python scripts/build_submission.py --mode search_scorer --weights <path>
.venv/Scripts/python scripts/verify_submission.py submission_search_scorer.tar.gz
.venv/Scripts/python scripts/submit.py submission_search_scorer.tar.gz -m "<version>: <what changed>"
```

- Log the new version in `VERSIONS.md` (date, what changed, why, the bake-off numbers that
  justified it) -- μ stays "pending" until the next MEASURE.
- Commit and push: this repo is the project's memory across sessions. Stage explicit paths
  (never `git add -A`); `runs/`, `submission/`, `*.tar.gz`, loose `.jsonl`/`.log` at the repo
  root stay gitignored.
- Tell the user to come back tomorrow -- 1 submission/day.

## Where things land

| What | Where |
|---|---|
| μ history | `VERSIONS.md` (tracked) |
| Autopsy report | console only (re-run to regenerate; not persisted as a file) |
| Downloaded ladder episodes | `runs/our_episodes/<date>/` (gitignored) |
| Measure/autopsy state | `runs/measure_state.json` (gitignored) |
| Local arena replay/results | `runs/autopsy_local/` (gitignored) |
| Kernel bake-off packages | `runs/kernel_bakeoff/<candidate>/` (gitignored) |
