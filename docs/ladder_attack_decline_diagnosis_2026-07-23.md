# Diagnosis: the 77.6% ladder attack-decline rate — 2026-07-23

Triggered by `docs/submission_ladder_audit_2026-07-23.md`'s headline red flag: real ladder
data showed a 77.6% attack-decline rate vs. 9.5% locally. This report answers the 7 numbered
questions the user posed, in order, each with evidence. **Diagnosis only — no agent or weight
changes.** Tooling changes made to enable the diagnosis: `tools/ladder_episode_parser.py` (a
real parser bug fix), `tools/measure.py`/`tools/kaggle_common.py` (submission-attribution
manifest + per-submission μ logging), `tools/autopsy.py` (`--split-by-submission`), and a new
one-off `tools/replay_ladder_decision.py`.

## Headline result

**Most of the 77.6% was a parser bug, not agent behavior.** After fixing it, the real pooled
rate is **36.8%** (93/253, losses-scoped, matching `docs/ladder_autopsy_2026-07-23.md` after
regeneration) — still meaningfully above local's 9.5%, but nowhere near the original number.
The 5-decision smoking-gun replay (Q6) further shows most of *that* remaining gap traces to
genuine near-ties in `evaluate()`'s scoring interacting with per-call determinization sampling
variance, not a bug, timeout, or exception fallback.

---

## Q1 — Unblend the data by submission ID

Episode JSON carries no submission ID (`info.Agents` only has `Name`/`ThumbnailUrl`, confirmed
by direct inspection). Built a manifest instead: `tools/measure.py::fetch_our_episodes_via_
submission_api()` now also writes `runs/our_episodes/episode_submissions.json`
(`{episode_id: submission_id}`) from the per-submission `kaggle competitions episodes <id>`
index it already fetches — the only place this identity is knowable. `tools/autopsy.py::
_load_ladder_games()` tags every parsed record with `submission_id`/`submission_label`, and a
new `--split-by-submission` flag runs the existing report once per label.

**What submission `54898784` actually is**: confirmed via `git show 81b1e6e` ("Package the
trained net as a second Simulation submission (Phase B)") — it is **not** `search_scorer`. It's
the **deprecated AlphaZero net-checkpoint agent**: determinized MCTS at 32 sims/move over a
torch-free NumPy net (iter-36 checkpoint), which **falls back to `src/baseline.py`'s rule-based
agent on any exception**. A structurally different, much weaker agent (score 309.3 vs. v1's
568.6), per `CLAUDE.md`'s own "Deprecated" section.

**Recomputed, split by submission** (with the parser fix from Q5 applied):

| | v1 `search_scorer` | net checkpoint (deprecated) |
|---|---|---|
| games | 45 | 30 |
| losses | 21 | 20 |
| **wins** | **24 (53.3% WR)** | 10 (33.3% WR) |
| attack-legal decisions (losses) | 117 | 136 |
| attack declined | 41 (**35.0%**) | 52 (38.2%) |
| evolve-legal decisions (losses) | 33 | 25 |
| evolve declined | 13 (39.4%) | 12 (48.0%) |
| Crustle-starvation-legal decisions | 84 | 0 (no such decision point ever arose) |
| Crustle starved (energy sent elsewhere) | 20 (23.8%) | n/a |
| near-tie rate | N/A (Kaggle records no `evaluate()` scores) | N/A |

**Key question, answered directly**: v1's own attack-decline rate is **35.0%, not ~77%** —
close to the net checkpoint's 38.2%, and both far above local's 9.5%, but the alarming original
number was overwhelmingly a parsing artifact (Q5), not something specific to v1's own behavior.
Where the blend *did* meaningfully mislead: **win rate**. v1 alone won 53.3% of its real ladder
games; the pooled 45.3% figure was dragged down by the much weaker, deprecated net checkpoint
(33.3% WR) sharing the same 75-game sample.

## Q2 — Leaderboard attribution

`kaggle competitions submissions --csv`'s `publicScore` is genuinely per-submission (three
distinct values confirmed: v1=568.6, net=309.3, baseline=238.2). Independently queried the
team leaderboard row (`TeamName="Rubin Sahota"`) twice this investigation, both times its
`Score` field exactly equalled v1's own `publicScore` (568.6) — **the displayed ladder μ is
v1's own score**, not a blend of the two active submissions. `tools/measure.py::
append_mu_history()` now logs both submissions' own scores every run (new `"submissions"`
field alongside the existing `mu`/`rank`, which stay pointed at the newest/leaderboard-matching
submission for backward compatibility):
```json
{"checked_at": "...", "mu": "568.6", "rank": "3664",
 "submissions": [{"ref": "54909461", "fileName": "submission_search_scorer.tar.gz", "publicScore": "568.6"},
                  {"ref": "54898784", "fileName": "submission_net.tar.gz", "publicScore": "309.3"}]}
```

## Q3 — Time limits

Real episode `configuration` blocks (checked across 8+ real episodes, identical every time):
```json
{"actTimeout": 0, "episodeSteps": 10000000, "runTimeout": 2000, "seed": <per-episode>}
```
`specification.configuration.actTimeout.description`: *"Maximum runtime (seconds) to obtain an
action from an agent."* **Default/actual value is 0 — per-action timeout is explicitly
disabled** for this competition. `runTimeout: 2000` is a whole-*episode* budget (seconds), not
per-move. All 75 episodes' `statuses` end `["DONE","DONE"]` — no episode hit the episode-level
budget either.

**What happens on overrun**: not determinable from this competition's configuration, because
no overrun mechanism is configured to fire per-action in the first place.

**Per-step timing/timeout evidence in our episodes**: none exists. `step[i][seat]["info"]` is
`{}` in every sample checked; `status` is only ever `"ACTIVE"`/`"INACTIVE"`/`"DONE"`, never a
timeout indicator. **Count of declined attacks coinciding with timeout/overrun markers: not
computable — no such markers exist in the data.** Stating this plainly rather than fabricating
a number.

One open, unresolvable-from-here loophole: the native C++ engine has its own internal
`GameConfig.timeLimit`/chess-clock (`ptcg_engine/.../Game.h`: `remainingTime`,
`timerStart`/`timerStop`), but the exposed `BattleStart(int* cards)` ABI takes no config
parameter — so whatever value Kaggle's production harness uses internally (if any) isn't
visible from the SDK, the episode JSON, or anything else in this repo. Flagged as unverifiable,
not ruled out — but combined with `actTimeout: 0` and zero timeout evidence anywhere in 75 real
games, it's a low-probability explanation.

## Q4 — Fallback paths audit

`agents/search_scorer.py::agent()`/`choose_action()` has exactly **6 fallback tiers**, all
triggered by exceptions or option-count, **never by elapsed wall-clock time** (confirmed: no
`time.time()`, `signal`, or deadline logic exists anywhere in the file):

| Tier | Trigger | Behavior |
|---|---|---|
| `too_many_options` | `len(selection.options) > MAX_ROOT_OPTIONS` (30) | skips search entirely, calls `baseline_choose_action` |
| `search_rejected` | every sampled world/step rejected by the engine | calls `baseline_choose_action` |
| `exception_to_baseline` | any exception in `parse_obs`/`choose_action` | calls `baseline_choose_action` |
| `exception_to_raw` | exception even reaching baseline | raw `selection.options[:max_count]`, **engine order, no attack preference at all** |
| `empty` | everything above failed | returns `[]` |
| (normal path) | — | 2-ply search + merit-based tie-break, `"attack"` is the *most* preferred kind in ties |

`baseline_choose_action` itself only auto-takes an attack when it's **lethal**; non-lethal
attack is deprioritized below attach/evolve by design (not a bug) — any tier routing to
baseline inherits this intentional conservatism. Local harness (`tools/eval_arena.py`/
`_eval_worker.py`) has **zero per-move timeout of its own** (only a 120s whole-*game* timeout),
so even a hypothetical Kaggle-side per-move time budget would have no locally-observable
analogue regardless of whether one exists.

Confirmed the shipped tarball (`scripts/build_submission.py --mode search_scorer`) copies
`agents/search_scorer.py` byte-for-byte — packaging only appends `WEIGHTS.update(...)`, never
touches control flow, so this analysis applies to what's actually live on Kaggle.

## Q5 — Parser definition check (the actual root cause of the 77.6% headline number)

**Hand-traced 2 real games end-to-end** (`87505521`, `87436356`) by dumping raw
`status`/`action`/`select` fields per step. Found: `tools/ladder_episode_parser.py::
_parse_matched_episode()`'s decision loop filtered only on `select`/`current` being non-null —
**it never checked `steps[i][our_seat]["status"]`**. Kaggle logs an observation for *both*
seats at *every* step regardless of whose turn it is; when it's not our turn, our seat's
`status` is `"INACTIVE"` and its `observation.select` is a **stale carry-over** of our last
real decision. Concrete example, game `87436356`: step 21 (`status=ACTIVE`, `action=[0]`, a
real chosen attack) was followed by steps 22–30 (`status=INACTIVE`, `action=[]`, the identical
3-option select replayed **8 more times**) before the next real decision. The old parser logged
all 9 as separate decisions; the 8 phantom ones all had `chosen_index=None`, which
`loss_review.py::analyze_attack_availability` unconditionally counts as "declined."

**Quantified across all 75 episodes** (both submissions, all outcomes, before fixing):

| | attack-legal decisions | outcome |
|---|---|---|
| `status == "ACTIVE"` (genuine decisions) | 596 | 207 truly declined (34.7%) |
| `status != "ACTIVE"` (phantom/stale steps) | 835 | 835/835 (100%) had `chosen_index is None` → all miscounted as "declined" |

**Fix**: skip any step where `steps[i][our_seat]["status"] != "ACTIVE"`, added directly to
`_parse_matched_episode()`'s loop with an inline comment explaining why. Re-verified against
the same two hand-traced games post-fix: `87436356` now shows exactly one clean decision at
turn 5.2 (`chosen_idx=1`, kind `attack`) instead of 9 (matching the independently-confirmed
opponent-HP-drop evidence that an attack really did land there). Pooled effect after
regenerating `docs/ladder_autopsy_2026-07-23.md`: **attack-decline rate 77.6% → 36.8%**
(93/253, losses-scoped); evolve-decline barely moved (43.1% → 43.1%, this bug specifically hit
`"attack"`-containing selects, not evolve-only ones, apparently by chance of how those
game states repeat).

**Was the definition itself (what counts as "attack legal"/"declined") a problem, separate from
the status bug?** No — independently re-read `src/obs.py::decode_selection()` and confirmed it
performs **zero filtering**: it labels every entry the native engine's `select.option` array
already contains, one-to-one, with `kind == "attack"` firing only for the engine's own
`OptionType.ATTACK`. Per `CLAUDE.md`, the engine (not this Python code) is what enforces
first-turn/energy/asleep/paralyzed legality, and it's the same native binary for both local SDK
play and real `cabt` ladder games. This exact same `decode_selection()` function is imported
unmodified by both the local (`tools/_eval_worker.py` → `agents/search_scorer.py`) and ladder
(`tools/ladder_episode_parser.py`) paths — no reimplementation, no ladder-specific relabeling.
**The only local/ladder asymmetry was the status-filter bug above**, now fixed.

## Q6 — Smoking gun: replay 5 real v1 decisions through our own local agent

Picked 5 real, `status=="ACTIVE"` (post-fix), v1-only decisions where attack was legal but
declined, spanning different games/turns/chosen-kinds. For each, pulled
`episode["steps"][i][our_seat]["observation"]` verbatim (proven byte-identical to what the live
agent received — same schema `src/obs.py::parse_obs()` decodes, not a Kaggle-specific
reshaping) and called `agents.search_scorer.agent(obs_dict)` directly — fully self-contained,
confirmed by reading its signature. Ran **10x per decision** (not once — `choose_action()`'s
`sample_determinization()` samples the hidden world randomly, and production is never given
`opp_deck_list`, so a single replay isn't guaranteed representative).

One methodology correction worth recording: the first run of this test showed
`exception_to_baseline` firing 10/10 on every decision — a false positive from the test harness
itself, not a real finding. `agent()`'s bare `read_deck_csv()` resolves `deck.csv` relative to
CWD, then `/kaggle_simulations/agent/deck.csv` — neither exists when run locally from the repo
root (`deck.csv` is only created at packaging time), so it raised `FileNotFoundError`, silently
caught by `agent()`'s own except-Exception fallback. Fixed by passing the same 60 card IDs
explicitly via `deck_list=` (functionally identical to what the packaged path resolves to on
Kaggle). Re-ran with the fix:

| Decision | Ladder chose | 10x local replay | Fallback tier fired |
|---|---|---|---|
| `87506107` turn 3.1 | attach | **attack ×10/10** | none — normal search |
| `87507277` turn 16.2 | evolve | **attack ×10/10** | none — normal search |
| `87508450` turn 12.1 | attach | **attack ×10/10** | none — normal search |
| `87509628` turn 6.2 | evolve | **attack ×10/10** | none — normal search |
| `87508450` turn 36.1 (late game) | attach | attack ×3/10, attach ×7/10 | none — normal search |

**Verdict, per the user's own interpretive framework**: local-us attacked in 4/5 cases (and
leaned attack even in the 5th) where ladder-us declined, with **zero fallback-tier firings** in
any of the 50 replay calls (all normal-search-path). This is **not** a deterministic behavior
bug and **not** an exception/timeout fallback — those would either always reproduce the same
ladder choice, or show fallback-tier activity. Instead it points to **genuine near-ties in
`evaluate()`'s scoring** whose outcome is sensitive to which hidden world
`sample_determinization()` happens to sample on a given call — consistent with
`agents/search_scorer.py`'s own documented finding that 86.5% of decisions have their top-2
options tied within 5%. The single live Kaggle call for each of these decisions apparently drew
a hidden-world sample that (correctly, given that sample) favored the setup move; most
resamples favor attack instead. This is the first time this near-tie phenomenon has been
directly measured on real ladder states (`loss_review.py::analyze_near_ties` is N/A on ladder
data since Kaggle records no `evaluate()` scores) — this 10x-replay technique is a workaround
for that limitation, at small scale (n=5) this cycle.

## Q7 — Ranked diagnosis and recommendation

1. **Parser artifact (status-filter bug) — CONFIRMED, dominant driver of the original 77.6%
   headline number.** Fixed this cycle. Effect: 77.6% → 36.8% pooled attack-decline rate.
2. **Submission-blend contamination — CONFIRMED contributor, mainly to win-rate
   interpretation** (v1 alone 53.3% WR vs. pooled 45.3%), less so to the decline-rate itself
   (v1 35.0% vs. net 38.2% — both similar). Tooling now supports splitting by submission going
   forward (`--split-by-submission`, per-submission μ logging).
3. **Genuine near-ties + determinization-sampling variance — CONFIRMED via the Q6 smoking
   gun** (4/5 replays flipped to attack under resampling, 1/5 genuinely mixed; 0/50 fallback
   firings). This is the best-supported explanation for why the *corrected* ~35% rate still
   sits well above local's 9.5%: real ladder opponents create far more genuinely-close-call
   board states than the narrow local `baseline`/`random` pool ever does, and near-ties are
   inherently sensitive to which hidden world got sampled.
4. **Determinization-prior mismatch (opponent-deck assumption) — PLAUSIBLE, structurally
   confirmed to exist** (production `agent()` is never given `opp_deck_list`, so
   `sample_determinization()` always guesses the opponent plays *our own* deck; local dry-runs
   default to the same deck for both sides, masking this by coincidence) but not directly
   isolated as a magnitude this cycle. Likely compounds #3 by making the sampled world less
   representative against real (non-mirror) opponents specifically.
5. **`MAX_ROOT_OPTIONS`/option-count-triggered fallback — plausible in general, not observed
   in this cycle's sample** (0/50 smoking-gun replay calls hit any fallback tier).
6. **Kaggle-side per-action timeout — evidence AGAINST** (`actTimeout: 0` confirmed across 8+
   real episodes; zero timeout markers anywhere in 75 games' worth of replay data), not fully
   excludable only because of the unreachable native-engine chess-clock. Ranked lowest.
7. **Exception/crash fallback from novel real-opponent card mechanics — plausible in general,
   not observed** in the 5 sampled decisions (0/50 fallback fires post-methodology-fix); no
   evidence either way beyond this small sample.

### Recommended next step (not implemented this cycle)

The evidence points at **measurement, not a code fix, as the highest-leverage next step**:
extend the Q6 technique (10x local replay + vote) from this 5-decision pilot to **all** real
declined-attack decisions across both submissions (~110 for v1 alone) to get a real,
quantified near-tie rate on ladder data for the first time (closing the
`analyze_near_ties`-is-N/A-on-ladder gap noted in Q1's table). If that broader run confirms the
pilot's pattern (most real declines are near-tie-driven, not "correctly declined"), the
recommended **code** fix for a future cycle would be improving `sample_determinization()`'s
opponent-deck prior for real ladder play (a small archetype-aware prior, or an ensemble of a
few plausible decks, instead of the current wrong-by-design "assume our own deck" default) —
since near-tie decisions are exactly where a better hidden-world guess would tip the balance
most. **Expected effect size**: this cycle's n=5 pilot suggests roughly 4/5 real declines were
tie-sensitive rather than genuinely-correct-to-decline, so a materially better prior could
plausibly cut a substantial share of the corrected ~35% decline rate — but this is an estimate
from a small pilot, not a proven bound, which is exactly why the broader measurement should
come before any weight/code change ships.
