# Near-tie/decline hypothesis: full-population measurement — 2026-07-23

Scales the `tools/replay_ladder_decision.py` 5-decision pilot from
`docs/ladder_attack_decline_diagnosis_2026-07-23.md` to v1's full real-decision population,
links declines to real game outcomes, and cross-checks the original local dry-run's "14/32
tied-and-lost" starvation pattern against real ladder data for the first time. **Measurement
only — no agent or weight changes.**

New tooling: `tools/measure_near_tie_hypothesis.py` (resumable replay-and-vote measurement,
checkpoints to `runs/near_tie_measurement/replays.jsonl`) and
`tools/analyze_near_tie_results.py` (read-only analysis pass over that checkpoint). Both call
`agents/search_scorer.py::choose_action()` in-process — no `cg` battle session or subprocess,
confirmed by the prior pilot — so this is inherently light on the Hardware rules; the concern
here was surviving interruption, not spawning simulator workers. **Resumability verified
directly**: ran to completion (117 unique decisions, 20 replays each), then truncated the
checkpoint file to 100/117 records and re-ran — it correctly detected 17 remaining and filled
in exactly those, ending back at 117.

## Headline result

The pilot's finding holds up at scale and sharpens into two distinct, separately-diagnosed
mechanisms:

1. **Cross-kind near-ties (attack vs. setup) driven by single-sample determinization
   variance** — confirmed on the 110-decision DECLINE set: 93.6% flip to attack in at least one
   of 20 local resamples, 89.1% flip in the *majority* of resamples.
2. **Same-kind exact ties (which bench Pokémon to power) that the existing tie-break cannot
   see** — confirmed on the 20-decision STARVATION set: **14/20 (70%) are "tied-and-lost"** on
   real ladder data — a *higher* rate than the original local dry-run's 14/32 (43.8%) — and
   root-caused precisely: the shipped tie-break (`_TIE_BREAK_PRIORITY`) only discriminates by
   **option kind** (attack > evolve > attach > retreat > ...), never by **target** within the
   same kind, so when multiple `attach` options land on functionally-identical reachable states
   (same score to many decimal places — confirmed in the raw trace), it has nothing to break
   the tie with.

Neither mechanism shows evidence of costing games outright: **zero of the 110 declines had a
lethal attack available**, and the intended target Pokémon was still present next real decision
in **110/110** cases.

---

## Item 1 — Scaled replay-and-vote (110 decisions × 20 resamples = 2200 calls)

| | value |
|---|---|
| flip rate (attack chosen ≥1/20 replays) | **103/110 (93.6%)** |
| flip rate (attack chosen in the majority, ≥11/20) | **98/110 (89.1%)** |
| classified near-tie (mean relative margin ≤ 5%) | 40/110 (36.4%) |
| classified confident decline (attack scored meaningfully better) | 56/110 (50.9%) |
| classified "declining was actually better" (chosen scored higher) | 14/110 (12.7%) |

**Margin** = mean, across a decision's 20 replays, of `best_attack_score − ladder_chosen_score`
(only replays where the normal search path ran, `mode=="searched"`, are included — every
decision had at least one such replay; 0/110 unscored). 40/2200 individual replay samples
(1.8%) hit `WIN_SCORE≈1e6` (a search-confirmed lethal/winning line), which dominates the raw
mean — reporting the outlier-excluded distribution instead:

- **Pooled margin, n=2160** (40 win-score outliers excluded): mean=1.52, **median=0.0** (the
  literal median sample has an *exactly* tied score), p10=−0.43, p90=4.50, min=−7.09, max=5.29.
- The median being exactly 0.0, not just "small," reinforces the same-reachable-state-tie
  mechanism found in item 3: many of these aren't "close" in a fuzzy sense, they're identical.

## Item 2 — Outcome linkage

**2a — per-game decline rate, wins vs. losses (v1 only):**

| | attack-legal decisions | declined | rate |
|---|---|---|---|
| 24 wins | 235 | 69 | 29.4% |
| 21 losses | 117 | 41 | 35.0% |

Losses show a somewhat higher decline rate than wins, consistent with declines being a
symptom correlated with harder games rather than a random/uniform behavior — but the gap
(29.4% vs. 35.0%) is modest, not the kind of gap that would make "stop declining" alone look
like a decisive lever.

**2b — did declined turns cost the opportunity or tempo?** For each of the 110 real declines,
walked forward through our next real decision(s) (up to 3):

- **Opportunity preserved** (same opponent-active serial, HP not higher than at decision time,
  at our very next real decision): **110/110**. The declined target was never gone/healed by
  the time we next got to act.
- **Prize-differential change over the following turns**: mean **−0.11**, median **0.0** — no
  systematic tempo cost on average.

**2c — explicit 3-bucket categorization**, criteria stated verbatim:
- **`lethal_declined`**: a legal attack option's base card-data damage (`Attack.damage`, via
  the option's `attackId`) ≥ the opponent active's HP at decision time. **0/110.**
- **`plausibly_correct_setup`**: not lethal, and the chosen option's kind was `attach` or
  `evolve` (i.e. a real step toward the attacker's energy/evolution requirement, not a pass).
  **110/110.**
- **`other/ambiguous`**: neither of the above. **0/110** — matches the earlier finding
  (diagnosis cycle) that v1 never declines into `end`/`retreat`, only into productive setup.

Spot-checked one example by hand (`87506107` decision 5, turn 3, `opp_active_hp=140`): the
*only* `"attack"`-kind option present was **"Ascension," base damage 0** — a 0-damage
utility/evolution-style attack, not a real KO threat. Confirms the lethal-detection logic is
behaving sensibly, and is a useful general caveat: **"attack legal"** in every decline-rate
figure in this and the prior report includes some 0-damage utility attacks whose "decline" is
trivially correct — not new to this cycle, but worth stating once, explicitly.

**Caveat on the lethal-detection proxy**: uses each attack's *base* card-data damage, not
variable/conditional damage encoded only in attack text (coin flips, per-energy bonuses). A
true edge case could be missed by this proxy; none was found in the 110-decision sample, but
this is an approximation, not ground truth.

## Item 3 — Ladder tied-and-lost check (the key cross-check)

Built decision records for the 20-decision STARVATION set (v1 losses only, matching the
original local analysis's own scope) with `score`/`features` backfilled from the **mean across
20 replays per option** (reduces per-call determinization noise before classifying), then ran
`tools/loss_review.py::analyze_energy_routing_detail()` **completely unmodified** — same
function, same `_TIE_EPS_REL=1e-6` exact-tie epsilon that produced the original local 14/32
figure.

```
total classified starvation decisions: 20 (skipped, no scores/features logged: 0)
  outscored, weight-imbalance (energy feature DOES differ, just outweighed): 6
  outscored, horizon/feature-blind (energy feature shows NO difference at all): 0
  tied-and-lost (agent's own tie-break should have caught this): 14
```

**Yes — the pattern appears in real games, at a higher rate than local (14/20 = 70% vs. local's
14/32 = 43.8%).** Reading the raw traces (see `runs/near_tie_measurement/` for full output)
shows exactly *why*: e.g. game `87508450` turn 16.1, three `attach` options targeting three
different bench Pokémon all score **11.4539000...** to full float precision — the 2-ply search
finds these lead to the identical best-reachable position, so they're genuinely
indistinguishable to `evaluate()`. The shipped `_TIE_BREAK_PRIORITY` tie-break only ranks
option **kind** (`attack`=0 < `evolve`=1 < `attach`=2 < ...) — every tied option here is already
the *same* kind (`attach`), so the existing tie-break has literally nothing to discriminate on,
and falls through to option-list order (an engine-ordering artifact, the exact failure mode the
tie-break was originally built to fix — just one level removed, at the *target* level rather
than the *kind* level).

## Item 4 — Sanity check: local's 9.5% is unaffected by the ladder parser bug

Not a new computation — a direct empirical re-check, alongside the already-established code-path
argument (local's `choose_action`'s own `emit()` closure logs synchronously, once per live
agent turn, with no `status` field or step-replay reconstruction anywhere in that path). Ran 6
fresh local games (`tools/eval_arena.py --candidate search_scorer --opponent baseline
--replay-out ...`, mirror-match deck) and scanned the resulting 150 logged decisions for the
exact signature that inflated the ladder numbers (consecutive duplicate turn/options with
`chosen_index: null`): **zero occurrences, and `chosen_index` is never `null` in local data at
all** (every logged decision has a real chosen index by construction). **Confirmed: local and
ladder numbers are apples-to-apples** — the fix changed only the ladder side.

## Item 5 — Ranked v2 recommendation

The measurement reveals **two distinct mechanisms**, each best matched by a different one of
the user's 4 candidates:

1. **Same-kind exact ties (STARVATION set, 14/20 real cases) → extend the tie-break to the
   target level, not just kind — a refinement of candidate (d).** This is **not** the
   already-shipped fix re-applied; the shipped `_TIE_BREAK_PRIORITY` never had a mechanism for
   "same kind, different target" ties at all, so this is a genuinely new, narrow rule: when
   multiple tied `attach`/`energy` options remain after the existing kind-based tie-break,
   prefer the option targeting the deck's designated attacker (`CRUSTLE_ID`) over other bench
   targets. **This is the top pick** — most precisely diagnosed (we know the exact mechanism,
   not just a correlation), touches the highest-severity real pattern found this cycle (70% of
   real starvation losses), and is a small, surgical, easily-reviewable change (one additional
   comparison in `_break_ties()`), not a new search/scoring paradigm.
2. **Cross-kind near-ties (DECLINE set, 93.6%/89.1% flip rate) → more determinization samples
   per decision — candidate (b).** Currently exactly **1** sample per decision
   (`sample_determinization()` called once, before scoring any option, per
   `agents/search_scorer.py::choose_action()`). Measured per-call cost this cycle: mean **26ms**,
   median **19ms**, p90 **53ms**, max **152ms** — even 10 samples/decision (≈530ms at p90)
   is trivial against Kaggle's `runTimeout: 2000s` whole-episode budget and disabled
   `actTimeout` (confirmed 2026-07-23). Averaging scores across N samples directly targets the
   mechanism item 1 measured (a single unlucky hidden-world draw tipping a near-tie), and item
   3's own methodology (averaging 20 replays before classifying) is itself a working
   demonstration of this variance-reduction approach. Second priority — real, well-evidenced,
   but the STARVATION mechanism is both more severe in this data and cheaper to fix.
3. **Opponent-deck prior (candidate a)** — structurally confirmed to exist
   (`sample_determinization()`'s own docstring: *"opp_deck_list defaults to the same list
   (mirror-match assumption)"*) and plausibly compounds #2 (a better-guessed hidden world would
   make each single sample more representative, not just more numerous), but this cycle
   provides no isolated measurement of its own marginal effect — would need a real
   opponent-archetype prior built and A/B'd, out of scope for a measurement-only cycle.
4. **Explicit attack-bias term in `evaluate()` (candidate c)** — **not recommended.** Item 2c
   found 0/110 lethal attacks ever declined and 110/110 declines were genuine setup moves, not
   arbitrary avoidance — an attack-bias term would push toward attacking in cases the data
   shows are frequently correct to decline (12.7% of the DECLINE set scored the setup move as
   *objectively better*, not just tied), risking a regression exactly like the local dry-run's
   already-documented "shallower search" attempt (10% vs. baseline, reverted). A blunt bias term
   would not distinguish "near-tie, resolve toward the pattern with the best long-run record"
   from "declining is correct here" the way a tie-break or sampling fix would.

### Expected effect and 400-game gate for the top pick

**Expected effect**: candidate 1 (target-aware tie-break) directly targets 14/20 real starvation
decisions found tied-and-lost this cycle — most of the `plausibly_correct_setup`/`attach`
share of the 110-decision DECLINE set overlaps with this same starvation phenomenon (Crustle
vs. other-bench-target ties), so a working fix should measurably shrink both the starvation-rate
figure (currently 20/84 legal, 23.8%) and part of the broader attack-decline rate, without
touching the 12.7% of cases where declining was genuinely correct (the fix only ever
discriminates *among already-tied same-kind options*, never overrides a real score difference).
**Verification path** (future cycle, not run this one): implement the target-aware tie-break,
then run `scripts/build_kernel_bakeoff.py`'s 400+-game gate, candidate vs. the currently-shipped
v1 weights, on the real ladder-scale (Kaggle kernel) rather than the local `baseline`/`random`
pair — per `CLAUDE.md`'s own standing policy, no ship without a proven win there. Given the
measured effect is about *reducing wasted setup on tied decisions*, not changing overall
strategy, expect a modest but directionally clear win-rate improvement rather than a dramatic
swing — the 400-game gate exists precisely because this cycle's evidence, while much stronger
than a guess, is still an n=20/110 real-data sample, not a proof.
