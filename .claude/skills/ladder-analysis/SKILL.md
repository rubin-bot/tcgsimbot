---
name: ladder-analysis
description: How to fetch and analyze real Kaggle ladder data for the Pokémon TCG AI Battle Challenge — episode discovery, parser gotchas, replay-and-vote methodology, and ladder-wide meta-mining. Trigger when the user asks to check real ladder games, fetch episodes, diagnose losses from real data, or characterize the ladder's deck meta.
---

# Ladder analysis

## Episode discovery

- **Primary path**: `tools/measure.py`'s `fetch_our_episodes_via_submission_api()` —
  `kaggle competitions episodes <submission_id> --format json` + `kaggle competitions replay
  <episode_id> -p <dir>`. Direct, complete, no large download. Writes to
  `runs/our_episodes/<date>/<episode_id>.json`.
- **Bulk-dump fallback** (`tools/measure.py --also-bulk-scan`, off by default): downloads
  `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD` (700MB+/day) and byte-searches each entry's
  first 16KB for a team name. **Confirmed a genuine subsample, not a complete record** — 9/9 of
  our real episode IDs from one date were real 404s against that day's dump
  (`docs/submission_ladder_audit_2026-07-23.md`). Our own team's games specifically have been
  found absent from these dumps more than once (also confirmed 2026-07-22's dump in the
  meta-mining cycle, 2026-07-23). **Use the primary path for anything about OUR OWN games; the
  bulk dump is only useful for ladder-WIDE characterization** (see meta-mining below), where
  missing any one team (including us) doesn't matter.
- Each day's dump publishes with roughly a 1-day lag — "today's" dump usually isn't available
  yet.

## Parser gotchas

- **`status == "ACTIVE"` filter**: episode steps include INACTIVE replay steps for the seat not
  currently deciding — always filter to `ACTIVE` before treating a step as a real decision.
- **UTF-8 BOM**: the leaderboard CSV ships with a BOM — decode with `utf-8-sig`, not `utf-8`, or
  the first header key silently becomes `'﻿Rank'` instead of `'Rank'`
  (`tools/kaggle_common.py::fetch_leaderboard_rows` already handles this correctly).
- **Episode parsing matches on team name only**, not submission ID — if 2 submissions are
  simultaneously active (the "latest 2 active" rule), a episode scan by team name blends both;
  cross-reference `tools/kaggle_common.py::load_episode_submissions()`'s manifest
  (`runs/our_episodes/episode_submissions.json`) to disambiguate which submission actually
  played a given episode.

## Replay-and-vote methodology (real-decision-level causal checks)

`tools/measure_near_tie_hypothesis.py` re-runs `agents/search_scorer.py::choose_action()` N
times (default 20) against the SAME real recorded board state, capturing each replay's chosen
option/scores/features into `runs/near_tie_measurement/replays.jsonl` (resumable via
`(game, decision_index)` done-keys). `tools/analyze_near_tie_results.py` then computes flip
rate, margin distribution, outcome linkage, and the real-data tied-and-lost classification
(reusing `tools/loss_review.py::analyze_energy_routing_detail` unmodified). New
`tools/run_flip_rate_smoke_check.py` wraps both steps with a `--n`/`--label` override to compare
different `N_DETERMINIZATIONS` values side by side without needing to hand-edit
`agents/search_scorer.py` (monkeypatches the module constant).

**Metric care** (see `verification-gate` skill's method lessons): the built-in "flip rate"
compares a replay's choice against the historical LADDER choice, not against itself — it cannot
be used as a pure noise/stability metric. For that, compute self-consistency directly: group a
decision's 20 `chosen_index` values, take the mode's share of the 20.

## Meta-mining (ladder-wide, not just our own games)

`tools/meta_miner.py` scans a FULL daily bulk dump (every episode, every team — not filtered to
us) for archetype signatures: per game, the top-3 most-frequent non-basic-energy card ids seen
in a team's own hand/active/bench/discard at their last observed board state. Resumable at day
granularity (`runs/meta_mining/scanned_dates.json`); writes via a per-date temp file, only
committed into the real cross-day output once a full day's scan succeeds (a killed mid-day scan
restarts cleanly rather than duplicating records).

`tools/meta_report.py` builds `docs/meta_report_<date>.md`: archetype distribution by rating
band, an archetype-vs-archetype win matrix (flags any cell under n=30 as unreliable), and a
Crustle-specific competitiveness read. **Refines the raw per-game signature at analysis time**
by filtering to `CardType.POKEMON` (via `cg.api.CardType`) — the raw top-3 is dominated by
staple Trainers (Poké Pad, Rare Candy, ...) played in nearly every deck, which aren't archetype-
defining; a team's archetype label is their single most-frequent Pokémon-type card id pooled
across all their observed games.

Usage:
```
.venv/Scripts/python tools/meta_miner.py --since YYYY-MM-DD [--until YYYY-MM-DD] [--max-days N]
.venv/Scripts/python tools/meta_report.py --date YYYY-MM-DD
```
