# Recovery Audit — v4 "stateful hearth" crash (2026-07-24)

## Summary

The v4 cycle (implementing the two ranked candidates from
`docs/master_study_2026-07-24.md`: Change 1 — "prefer continuing over
`end`" near-tie widening, and Change 2 — play/card tie-break scoped to
a conservative high/low tier) was underway when the session died
silently overnight. **No machine crash occurred** — the OS uptime is
unbroken and no Windows crash/shutdown events exist in the window —
only the foreground process driving the background work stopped,
around **03:01:50 AM**, and nothing resumed it for the following
5h16m. **Nothing was committed, shipped, or submitted to Kaggle.**
HEAD is still `cf81b3d`. The uncommitted code changes in the working
tree are complete and internally coherent (not mid-edit), and the two
new test files pass when run. The verification-gate data run — the
required next step before any ship decision — is 18.8% complete on
its expert-corpus leg and safely resumable.

## Timeline

| Time (IST, 2026-07-24 unless noted) | Event |
|---|---|
| 2026-07-22 16:09:33 | Machine last booted — uptime unbroken through this entire audit, confirming no crash/reboot occurred |
| 07-23 21:12–22:36 | Three unrelated Kaggle kernel pushes for a v3-vs-v1 head-to-head gate (`...-2026-07-24`, `...-retry`, `...-spawnfix`) — predate this cycle, still `RUNNING` as of this audit, see Integrity Concerns |
| 00:58–01:00:07 | `docs/master_study_2026-07-24.md` finalized and committed as `cf81b3d` — explicitly analysis-only, "no ship action" |
| 01:00–01:42 | v4 code written: both changes in `agents/search_scorer.py`, tooling parameterized in `tools/decision_diff.py` / `tools/cluster_divergences.py` / `tools/reconstruct_decision.py`, two new test files added — all still uncommitted |
| 01:42:15 | v4 candidate snapshot frozen at `runs/v4_candidates/search_scorer_v4_snapshot.py` (confirmed byte-identical to the current working-tree `agents/search_scorer.py`) |
| 02:09:27 | `runs/decision_diff/clusters.json` re-touched — a no-args sanity check that parameterized `cluster_divergences.py` still reproduces v1's original output unchanged |
| ~02:12–02:27:51 | `decision_diff.py --group control --snapshot v4` run: completed fully, 681/681 records (matches v1 baseline's control count exactly) |
| ~02:28–**03:01:50** | `decision_diff.py --group expert --snapshot v4` run: reached 4,939/26,311 records (18.8%), then stopped — this is the last file write anywhere in the repo |
| 03:01:50 → 08:17:53 | **5h16m of total silence** — no further `runs/` writes, no running processes, no kernel pushes, no submissions |
| 08:17:53 | This recovery audit begins |

## Task-by-task status

The plan file (`~/.claude/plans/build-and-ship-v4-stateful-hearth.md`)
contains no literal checkboxes — it's prose organized into 8 `##`
sections. The table below decomposes it into 11 independently
verifiable work items and checks each against on-disk evidence, not
the plan's own framing.

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | Change 1 impl (`_prefer_continuing_over_end`) | **DONE** | `git diff agents/search_scorer.py` — matches plan design exactly: `END_NEAR_TIE_REL_THRESHOLD=0.05`, `END_EARLY_MAX_TURN=4`, applied after `_aggregate_votes` resolves a winner |
| 2 | Change 1 tests (`tests/test_end_early.py`) | **DONE** | 160 lines, 3 real captured episode/step cases (87363023/47, 87397816/7, 87529164/25) + negative controls; **this audit ran it read-only: 6/6 pass** |
| 3 | Change 2 impl (`_PLAY_CARD_TIE_PRIORITY`) | **DONE** | Matches plan's 6 named card IDs/tiers exactly (Pokégear 3.0, Jumbo Ice Cream → prefer-first; Mega Kangaskhan ex, Boss's Orders, Switch, Hand Trimmer → prefer-last); `_tie_break_key` return type correctly extended 3-tuple → 4-tuple |
| 4 | Change 2 tests (`tests/test_play_card_tiebreak.py`) | **DONE** | 95 lines, 2 real episode-traced cases (87363032/33, 87363039/135); **this audit ran it read-only: 2/2 pass** |
| 5 | Step 1 — reproduce first | **PARTIAL** | New tests pass post-fix (confirmed this audit); no artifact proves the crashed session itself ever executed them or checked pre-fix failure; existing suite 8/9 pass (see Test Suite below) |
| 6 | Step 2 — parameterize tooling + freeze v4 snapshot | **DONE** | All three tools files gained `--snapshot`/`--weights`/`--out-suffix`/`--in-suffix`, backward-compatible defaults preserve v1's original file paths; snapshot frozen and verified byte-identical to working tree |
| 7 | Step 2 — full-corpus decision-diff re-run (control + expert) | **PARTIAL — crash point** | Control: 681/681 (100%). Expert: 4,939/26,311 (**18.8%**), died mid-episode on team LiamK (only 35 of that team's 10,536 decisions processed) |
| 8 | Step 2 — regenerate `clusters_v4.json` | **NOT STARTED** | Only the original v1 `clusters.json` exists; input data (expert_diff_v4) is incomplete so this couldn't have run meaningfully |
| 9 | Step 2 — B0 self-test on v4 snapshot (≥87.4% agreement bar) | **NOT STARTED** | CLI gained `--snapshot`/`--weights` flags at 02:38:13 but no result artifact exists anywhere |
| 10 | Step 3(a)-(c) — 400-game local gates for v4 | **NOT STARTED** | No v4-equivalent gate directory exists; `runs/kernel_head_to_head/` untouched by this cycle |
| 11 | Step 3(d) — mechanism metrics / 20-loss autopsy / time-guard check | **NOT STARTED** | Depends on #10, which doesn't exist |
| — | Decision rule, shipping, `VERSIONS.md` v4 entry, `docs/v4_report_2026-07-24.md`, commit + push | **NOT REACHED** | Trivially blocked on the above |

**6 of 11 items fully done, 1 partial (the crash point), 4 not
started.** Nothing beyond item 7 was reached.

## Test suite (run read-only during this audit)

9 files in `tests/`, run directly (no pytest, per project convention):

| File | Result |
|---|---|
| `test_baseline.py` | **FAIL** — 20/50 wins (40%), assertion requires >50% |
| `test_encode.py` | PASS |
| `test_end_early.py` (untracked, new) | PASS — 6/6 sub-tests |
| `test_mcts.py` | PASS |
| `test_obs.py` | PASS |
| `test_play_card_tiebreak.py` (untracked, new) | PASS — 2/2 sub-tests |
| `test_search_scorer.py` | PASS |
| `test_tie_break.py` | PASS |
| `test_v3_voting.py` | PASS |

`test_baseline.py`'s failure is most likely the game engine's
documented non-determinism (CLAUDE.md: `cg.game.battle_start()` takes
no seed; same-code-same-seed runs agree only "at chance level") rather
than a v4 regression — none of the uncommitted v4 changes touch
`src/baseline.py` or its call path. Flagging as an open question, not
a confirmed regression; worth a re-run or two to check for consistency
before trusting either result.

## Integrity concerns

1. **Dangling doc reference.** `agents/search_scorer.py`'s new code
   comments cite `docs/v4_report_2026-07-24.md` as the evidence
   source for both changes — that file was never written. Anyone
   reading the diff in isolation will hit a broken reference.
2. **0-byte log despite complete data.** `runs/decision_diff/v4_control_run.log`
   is 0 bytes even though `control_diff_v4.jsonl` completed fully and
   correctly (681/681, matching the v1 baseline count). This is
   consistent with stdout being block-buffered and never flushed when
   the process died — not data corruption, since the JSONL writer
   itself flushes per-record.
3. **Orphaned Kaggle kernels (unrelated to the v4 crash, but currently
   live).** Three kernel pushes from the *prior* evening
   (`rubinsahota/v3-vs-v1-head-to-head-2026-07-24`, `...-retry-2026-07-24`,
   `...-spawnfix-2026-07-24`, pushed 21:12–22:36 IST on 07-23, for an
   unrelated v3-vs-v1 gate) all still report `RUNNING` via
   `kaggle kernels status` roughly 11 hours later and have never
   produced a result. The repeated retry/spawnfix naming suggests they
   were already failing before this crash. These are separate from
   the v4 work and warrant their own check/cancel decision.
4. **`test_baseline.py` failure** — see Test Suite above; not
   confirmed as a regression, needs a clean re-run to disambiguate
   from engine noise.

## Kaggle state — confirmed clean

Read-only `kaggle competitions submissions` check, live:

```
ref,fileName,date,status,publicScore
54909461,submission_search_scorer.tar.gz,2026-07-22 17:47:52,COMPLETE,593.2   <- v1, active
54898784,submission_net.tar.gz,        2026-07-22 09:00:48,COMPLETE,288.9    <- deprecated net checkpoint
54897157,submission.tar.gz,            2026-07-22 07:35:21,COMPLETE,238.2    <- original baseline
```

Only 3 submissions exist, ever. **Nothing new was submitted during the
crash window.** v1 remains the sole active Kaggle submission,
unchanged. No rollback is needed anywhere.

## Where to resume

The verification-gate protocol (per the `verification-gate` skill)
requires the full-corpus expert diff, the cluster regeneration, the B0
self-test, and the 400-game gates before any ship/no-ship decision —
none of the last three have happened, and the first is 81% incomplete.
To pick the cycle back up cleanly:

1. Resume the expert-corpus diff — the done-keys mechanism in
   `tools/decision_diff.py` will skip the 4,939 already-written
   records and continue from roughly episode 95/492:
   ```
   tools/decision_diff.py --group expert \
     --snapshot runs/v4_candidates/search_scorer_v4_snapshot.py \
     --out-suffix _v4
   ```
2. Regenerate the cluster table: `tools/cluster_divergences.py
   --in-suffix _v4 --out-suffix _v4`, compare to v1's 33-cluster
   baseline.
3. Run the B0 self-test regression on the v4 snapshot (≥87.4% bar).
4. Only then proceed to Step 3's 400-game local gates.
5. Write `docs/v4_report_2026-07-24.md` with the real findings before
   any ship action, and log a `VERSIONS.md` entry either way.

No code needs to be rewritten — the uncommitted implementation and
both test files are complete and passing. This is a data-generation
resume, not a re-implementation.
