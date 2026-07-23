# Submission/ladder audit — 2026-07-23

Triggered by a direct request to verify, via the Kaggle API rather than assumption, whether
our submissions are actually playing ladder games — the 2026-07-23 MEASURE+AUTOPSY cycle had
concluded "0 of our episodes found" and treated it as inconclusive-but-not-alarming. This audit
checked that conclusion directly against the API instead of re-running the same local dry-run
analysis again. **Tooling and documentation only — no agent/weights changes this cycle.**

## 1. Submission status

`kaggle competitions submissions -c pokemon-tcg-ai-battle --csv`:

| ref | file | submitted | status | publicScore |
|---|---|---|---|---|
| 54909461 | submission_search_scorer.tar.gz (v1) | 2026-07-22 17:47:52 | `SubmissionStatus.COMPLETE` | 568.6 |
| 54898784 | submission_net.tar.gz | 2026-07-22 09:00:48 | `SubmissionStatus.COMPLETE` | 309.3 |
| 54897157 | submission.tar.gz (baseline) | 2026-07-22 07:35:21 | `SubmissionStatus.COMPLETE` | 238.2 |

**No errors, no failed validation, on any submission.** Item 3 of the audit (reproduce a
failure locally) doesn't apply — there is no failure to reproduce.

**Why `SubmissionCount` is 2, not 3:** the leaderboard row for `TeamName="Rubin Sahota"`
(`TeamId=16579942`) reports `SubmissionCount=2`. This matches the competition rule that only
the **latest 2** submissions are active/scored at a time (`CLAUDE.md`'s Simulation section).
The two most recent by date — v1 (`54909461`) and the net checkpoint (`54898784`) — are active;
`baseline` (`54897157`), the third-most-recent, has aged out. Confirmed directly against the
leaderboard CSV, not inferred.

## 2. v1 IS playing real ladder games

`kaggle competitions episodes 54909461 --format json` — a CLI subcommand `tools/measure.py`
never called before this audit — returned **46 total episodes** for v1 (1
`EPISODE_TYPE_VALIDATION` self-play sanity check at submission time + **45 real ladder
games**), spanning 2026-07-22 17:47 through 2026-07-23 07:11. The net-checkpoint submission
(`54898784`) had **31 total (1 validation + 30 ladder)**. **75 real ladder episodes total**
across both active submissions as of this check. This flatly contradicts "0 episodes" as a
statement about whether the submissions are playing — that conclusion was only ever true of
the one data source `tools/measure.py` checked (see §3).

Overall record across all 75 (both submissions combined — see Limitations): **34 wins, 41
losses, 0 draws (45.3% win rate)**, consistent with a leaderboard score (568.6) sitting below
the μ₀=600 starting point.

## 3. Root cause of "0 episodes found": confirmed, not hypothesized

Downloaded one real replay directly (`kaggle competitions replay 87614196`, 3.68MB JSON). Its
schema matches `tools/ladder_episode_parser.py`'s assumptions exactly (`info.TeamNames`,
`steps`, `rewards`, `statuses`), and `"Rubin Sahota"` sits at **byte offset 217** — comfortably
inside `tools/measure.py`'s 16KB prefix-search window. **The byte-search matching logic that
existed before this audit was not broken.**

Then tried to fetch 9 of our real, submission-API-confirmed episode IDs from 2026-07-22
(`87505388`, `87506107`, `87507277`, `87513718`, `87519578`, `87524735`, `87531952`,
`87540452`, `87544839`) as named files (`<id>.json`) directly from the
`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22` dataset dump: **0/9 exist there** — real
404s from the Kaggle API. Cross-checked this wasn't a transient/auth issue by fetching a
filename actually listed in that dataset (`87362960.json`, unrelated to us) — it downloaded
fine (exit 0, 3.94MB). **The daily bulk dataset is a genuine subsample of that day's total
episodes across ~5500+ teams; our specific games simply weren't in the published sample.**
This was already flagged as hypothesis #1 in the same-day `docs/ladder_autopsy_2026-07-23.md`
autopsy but never confirmed until this audit.

**Conclusion:** item 2's original framing ("fix the matching logic") doesn't apply — nothing
was factually wrong with the string-match code. The real gap was that `tools/measure.py` only
ever tried the bulk-dump path, which is structurally unreliable (subsample) and expensive
(700MB+/day) for something the Kaggle API answers directly, cheaply, and completely by
submission ID.

## 4. measure.py's leaderboard row match: confirmed correct

Live query independently reproduced what `tools/measure.py::fetch_leaderboard_row` computes:
exact (case-insensitive) match on `TeamName == "Rubin Sahota"` returns `TeamId=16579942`,
`Rank=3664`-`3669` (moved slightly between checks the same morning — expected, mu is still
settling), `Score=568.6`, `SubmissionCount=2`. No neighboring-row mixup; the row is ours.

## 5. What changed

`tools/measure.py` gained a new **primary** episode-discovery path,
`fetch_our_episodes_via_submission_api()`: for each of our 2 active submissions, calls
`kaggle competitions episodes <id> --format json` then `kaggle competitions replay <id> -p
<dir>` for every non-validation episode, writing into the same
`runs/our_episodes/<date>/<episode_id>.json` layout `tools/autopsy.py` already walked. This
replaces the old bulk-dataset scan as the default. The old scan (`download_and_extract_our_
episodes` + the day-range probing machinery) is **kept, not deleted**, gated behind a new
`--also-bulk-scan` flag, in case the submission-API path is ever rate-limited or deprecated.

One incidental bug fix required to actually exercise the newly-unblocked ladder pipeline:
`src/carddata.py`'s `load_card_index()` stored `card_type=cd.cardType` as a raw int instead of
wrapping it in the `CardType` IntEnum, which crashed `tools/autopsy.py`'s (ladder-only)
opponent-archetype analysis the first time it ever ran against non-empty ladder data (this
code path had never been exercised with real games before). Fixed to `CardType(cd.cardType)` —
`CardType` is an `IntEnum`, so this is a no-op for every other existing int comparison
elsewhere in the codebase.

Two CLI-output quirks discovered and handled while wiring up the new path, for anyone touching
this later: `kaggle competitions episodes --format json` prints a JSON array followed by a
trailing plain-text usage line on the same stdout stream (parsed with
`json.JSONDecoder().raw_decode()`, not `json.loads()`); and its `type` field is prefixed with
the enum class name (`"EpisodeType.EPISODE_TYPE_VALIDATION"`, not the bare
`"EPISODE_TYPE_VALIDATION"` the Python API's `competition_list_episodes()` returns).

## Verification performed

- Ran the updated `tools/measure.py` for real: fetched 75 real episode replays (45 + 30) into
  `runs/our_episodes/2026-07-22/` and `runs/our_episodes/2026-07-23/`.
- Ran `tools/autopsy.py --source auto`: correctly resolved to `ladder` mode (not the local
  fallback) and produced a full real report (41 losses, 2762 logged decisions, opponent
  archetypes, etc.) instead of "no losses to analyze."
- Ran `tools/ladder_report.py` end to end: regenerated `docs/ladder_autopsy_2026-07-23.md`
  with real numbers (see below) instead of the "why zero ladder games" fallback section it had
  before this audit.

## Notable finding surfaced by finally having real data (flagged for a future iterate cycle,
not investigated further this cycle)

`docs/ladder_autopsy_2026-07-23.md`'s regenerated report shows the **ladder attack-decline
rate is 77.6% (554/714)**, vastly higher than the **9.5% (14/147)** seen in the same-day local
dry-run against `baseline`/`random`. This is a large, previously-invisible gap between local
and real-ladder agent behavior — direct evidence for `CLAUDE.md`'s own standing argument that
`baseline`/`random` are too narrow a test. Worth prioritizing in the next iterate cycle's
DIAGNOSE step; not diagnosed further here since this cycle was scoped to discovery/tooling
only.

## Limitations of this cycle's numbers

- The 75-game record and the loss-taxonomy breakdown in `docs/ladder_autopsy_2026-07-23.md`
  **mix games from both currently-active submissions** (v1 search_scorer and the older net
  checkpoint) — `tools/ladder_episode_parser.py` matches on team name only, which doesn't
  distinguish which of our two submissions played a given episode. The episode JSON's
  `info.Agents` isn't parsed for `submissionId` yet. These are two materially different agents
  (SearchScorer vs. an AlphaZero net checkpoint), so the combined 45.3% win rate and the
  attack-decline/evolve-decline rates are a blend, not a clean per-agent signal. Splitting by
  submission ID would need a small follow-up to `ladder_episode_parser.py`/`autopsy.py` — not
  done this cycle (scoped to discovery/tooling, no agent-analysis changes).
- `kaggle competitions episodes <id> --format json` doesn't include the `agents` field (which
  the Python API's `competition_list_episodes()` does return, including `teamName`/`reward`
  per seat) — only `id`/`createTime`/`endTime`/`state`/`type`. Not needed for this fix (the
  submission ID already scopes the query to us), but relevant if a future per-submission
  breakdown wants reward/opponent-identity without downloading the full replay.
