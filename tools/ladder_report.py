"""Stage 6 PARSE + AUTOPSY + REPORT for real v1 ladder games (tools/measure.py must have run
first -- this reads runs/measure_state.json for where episodes landed).

Produces a detailed markdown report (overall record, loss taxonomy, top-bucket walkthroughs,
the tied-and-lost/Dwebble check, opponent-strength picture, comparison vs. the local dry-run)
and saves it to docs/. This is a report-only tool -- it never touches agent code or weights.

KNOWN LIMITATIONS (stated here and repeated in the report, not silently papered over):
  - No sigma/confidence-interval field exists anywhere in Kaggle's API; only a point mu.
  - No historical mu time series from the API; runs/mu_history.jsonl (tools/measure.py) is
    our own accumulating record, starting from whenever measure.py was first run with this
    feature -- it is NOT a backfilled complete history.
  - Real per-decision evaluate() scores cannot be recovered for ladder games after the fact:
    faithfully doing so would require replaying the episode through the live cg engine with
    the ORIGINAL shuffle/draw RNG seed, which Kaggle never exposes -- a fresh battle_start()
    reshuffles differently, so the recorded action indices would no longer refer to the same
    options. So "near-tie rate" and the weight-imbalance/horizon-blind split of
    tools/loss_review.py's energy-routing-detail check are LOCAL-ONLY metrics; ladder-side
    analysis of "why" is qualitative, grounded in the real shipped weights
    (runs/tune_run1/winner_weights.json), never a fabricated recomputed score.
  - Episodes matched by team name could in principle come from EITHER of our two active
    submissions (only the latest 2 stay active, so the deprecated `submission_net.tar.gz` was
    still active through v1's whole first day) -- there is no field in the replay identifying
    which of our submissions produced a given game, so this can't be disambiguated from replay
    data alone. Treated as "predominantly v1" since it's the far stronger/most recent, but
    flagged rather than assumed silently.

Usage:
  .venv/Scripts/python tools/ladder_report.py
"""

from __future__ import annotations

import contextlib
import io as io_
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import loss_review  # noqa: E402
from ladder_episode_parser import LadderParseError, parse_episode_file  # noqa: E402
from kaggle_common import OUR_TEAM_NAME, fetch_leaderboard_rows  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEASURE_STATE_PATH = os.path.join(ROOT, "runs", "measure_state.json")
SHIPPED_WEIGHTS_PATH = os.path.join(ROOT, "runs", "tune_run1", "winner_weights.json")
LOCAL_REPLAY_DIR = os.path.join(ROOT, "runs", "autopsy_local")
STRENGTH_THRESHOLD = 50.0  # score-point band around our own mu counted as "similar strength"

CRUSTLE_ID = loss_review.CRUSTLE_ID
DWEBBLE_ID = loss_review.DWEBBLE_ID


# ---------------------------------------------------------------------------
# Loading + parsing
# ---------------------------------------------------------------------------

def load_measure_state() -> dict:
    if not os.path.exists(MEASURE_STATE_PATH):
        sys.exit(f"{MEASURE_STATE_PATH} not found -- run tools/measure.py first.")
    with open(MEASURE_STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_all_episodes(episodes_root: str) -> tuple[list[dict], int, list[tuple[str, str]]]:
    """Returns (games, not_ours_count, parse_failures[(path, reason)])."""
    games, not_ours, failures = [], 0, []
    if not episodes_root or not os.path.isdir(episodes_root):
        return games, not_ours, failures
    for dirpath, _, filenames in os.walk(episodes_root):
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                rec = parse_episode_file(path)
            except LadderParseError as e:
                failures.append((path, e.reason))
                continue
            if rec is None:
                not_ours += 1
                continue
            games.append(rec)
    return games, not_ours, failures


# ---------------------------------------------------------------------------
# Win-condition taxonomy (the real 3 PTCG loss conditions + a timeout/illegal catch-all)
# ---------------------------------------------------------------------------

def classify_ending(game: dict) -> str:
    t = game.get("terminal") or {}
    if t.get("our_status") not in (None, "DONE"):
        return "timeout_or_illegal"
    if t.get("our_deck_count") == 0:
        return "deck_out"
    if t.get("opp_prizes_remaining") == 0 and (t.get("our_prizes_remaining") or 0) > 0:
        return "prize_race_lost"
    if t.get("our_active_present") is False and (t.get("our_bench_count") or 0) == 0:
        return "bench_wipe"
    return "other_attrition"


# ---------------------------------------------------------------------------
# Per-game contributing-factor metrics (board-state only -- no scores needed, so these work
# identically on local and ladder games; near-tie/energy-routing-weight-imbalance stay
# LOCAL-ONLY since they need evaluate() scores -- see module docstring)
# ---------------------------------------------------------------------------

def attack_decline_rate(games: list[dict]) -> tuple[int, int]:
    declined = total = 0
    for game in games:
        for dec in game["decisions"]:
            if not any(o["kind"] == "attack" for o in dec["options"]):
                continue
            total += 1
            chosen = loss_review._chosen_option(dec)
            if chosen is None or chosen["kind"] != "attack":
                declined += 1
    return declined, total


def evolve_decline_rate(games: list[dict]) -> tuple[int, int]:
    declined = total = 0
    for game in games:
        for dec in game["decisions"]:
            if not any(o["kind"] == "evolve" for o in dec["options"]):
                continue
            total += 1
            chosen = loss_review._chosen_option(dec)
            if chosen is None or chosen["kind"] != "evolve":
                declined += 1
    return declined, total


def starvation_events(games: list[dict]) -> list[tuple[dict, dict, bool]]:
    """Returns (game, decision, dwebble_targeted) for every decision where powering an
    unpowered Crustle was a legal energy/attach target but energy went elsewhere."""
    events = []
    for game in games:
        for dec in game["decisions"]:
            energy_opts = [o for o in dec["options"] if o["kind"] in ("energy", "attach")]
            if not energy_opts:
                continue
            unpowered = [m for m in loss_review._our_crustles(dec) if len(m["energies"]) < 3]
            if not unpowered:
                continue
            unpowered_serials = {m["serial"] for m in unpowered}
            targets_crustle = [o for o in energy_opts if o["target_serial"] in unpowered_serials]
            if not targets_crustle:
                continue
            chosen = loss_review._chosen_option(dec)
            if chosen is not None and chosen.get("target_serial") in unpowered_serials:
                continue  # we did power it
            dwebble_targeted = bool(chosen) and chosen.get("target_card_id") == DWEBBLE_ID
            events.append((game, dec, dwebble_targeted))
    return events


def near_tie_rate(games: list[dict]) -> tuple[int, int] | None:
    """None if no decision in these games carries a score (ladder data never does -- see
    module docstring); otherwise (near_ties, scored_decisions), matching loss_review.py's
    analyze_near_ties threshold."""
    total = near = 0
    any_scored = False
    for game in games:
        for dec in game["decisions"]:
            scored = [o for o in dec["options"] if o["score"] is not None]
            if len(scored) < 2:
                continue
            any_scored = True
            total += 1
            ranked = sorted(scored, key=lambda o: o["score"], reverse=True)
            s1, s2 = ranked[0]["score"], ranked[1]["score"]
            denom = max(abs(s1), abs(s2), 1e-9)
            if abs(s1 - s2) / denom <= loss_review.NEAR_TIE_REL_THRESHOLD:
                near += 1
    return (near, total) if any_scored else None


# ---------------------------------------------------------------------------
# Reuse loss_review.py's print-based analyses, capturing their output for the saved report
# ---------------------------------------------------------------------------

def run_captured(fn, *args, **kwargs) -> str:
    buf = io_.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    text = buf.getvalue()
    sys.stdout.write(text)
    return text


# ---------------------------------------------------------------------------

def main() -> None:
    report: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report.append(line)

    state = load_measure_state()
    episodes_root = state.get("our_episodes_dir")
    emit(f"# Ladder autopsy report\n")
    emit(f"measure_state.json: checked_at={state.get('checked_at')}, mu={state.get('mu')}, "
         f"rank={state.get('rank')}, submission_count={state.get('submission_count')}")
    emit(f"episode datasets scanned since {state.get('since_date')}: "
         f"{state.get('days_scanned')} (not yet published: "
         f"{state.get('days_not_yet_published')}, failed: {state.get('days_failed')})")
    emit(f"episodes found per day: {state.get('episodes_by_date')}")
    emit("")

    games, not_ours, failures = parse_all_episodes(episodes_root)
    emit(f"## Parse summary")
    emit(f"parsed OK: {len(games)}  |  not-ours (unexpected if >0): {not_ours}  |  "
         f"failed to parse: {len(failures)}")
    if failures:
        emit("PARSE FAILURES (schema surprises, not silently skipped):")
        for path, reason in failures:
            emit(f"  - {path}: {reason}")
    emit("")

    if not games:
        emit("## Why zero ladder games, despite an active, scored submission")
        emit(f"Our leaderboard score ({state.get('leaderboard_score')}) has already moved away "
             f"from the mu0=600 starting mean across several checks, which means SOME real "
             f"rated games have been played and counted -- but none of them showed up in the "
             f"one published day we could scan ({state.get('episodes_by_date')}). Two "
             f"plausible explanations, not mutually exclusive: (1) each daily dataset appears "
             f"to be a large sample of that day's games across ~5500+ teams, not a literal "
             f"complete record of every rated game played -- with v1 only active for the last "
             f"~6 hours of 2026-07-22 (submitted 17:47), its share of that day's games was "
             f"small and may simply not have been sampled into this particular dump; (2) there "
             f"may be a publication lag between a game affecting the live score and that game's "
             f"replay becoming available in a dataset. Re-running tools/measure.py after "
             f"2026-07-24's dataset publishes (covering the rest of 2026-07-22 and all of "
             f"2026-07-23) is the natural next check.")
        emit("")
        _write_no_ladder_tail(report, emit)
        _write_report(report)
        return

    wins = [g for g in games if g["outcome"] == "candidate_win"]
    losses = [g for g in games if g["outcome"] == "opponent_win"]
    draws = [g for g in games if g["outcome"] == "draw"]
    errored = [g for g in games if (g.get("terminal") or {}).get("our_status") not in
               (None, "DONE")]

    emit(f"## (a) Overall record")
    emit(f"games: {len(games)}  |  wins: {len(wins)}  |  losses: {len(losses)}  |  "
         f"draws: {len(draws)}  |  errored/non-DONE episodes: {len(errored)}")
    emit(f"win rate (draws=0.5): "
         f"{(len(wins) + 0.5 * len(draws)) / len(games):.3f}" if games else "n/a")
    emit("")

    # (b) loss taxonomy
    emit(f"## (b) Loss taxonomy")
    taxonomy = Counter(classify_ending(g) for g in losses)
    if losses:
        for bucket, n in taxonomy.most_common():
            emit(f"  {bucket}: {n} ({n / len(losses):.1%} of losses)")
    else:
        emit("  no losses yet -- nothing to bucket.")
    emit("")

    # contributing-factor stats across all losses (reuses loss_review.py directly)
    emit(f"## Contributing-factor stats across losses (tools/loss_review.py, reused)")
    if losses:
        run_captured(loss_review.analyze_attack_availability, losses)
        run_captured(loss_review.analyze_attacker_starved, losses)
        emit("Energy-routing weight-imbalance/horizon-blind split and near-tie rate: N/A on "
             "ladder data (need evaluate() scores Kaggle never records -- see module "
             "docstring's Known Limitations). See (d)/(e) below for what IS measurable.")
        run_captured(loss_review.analyze_avoidable_kos, losses)
        run_captured(loss_review.analyze_evolve_misplay, losses)
    else:
        emit("  no losses yet.")
    report.append("")

    # (c) top-2-bucket walkthroughs
    emit(f"## (c) Top-2 failure-mode walkthroughs")
    if SHIPPED_WEIGHTS_PATH and os.path.exists(SHIPPED_WEIGHTS_PATH):
        with open(SHIPPED_WEIGHTS_PATH, encoding="utf-8") as f:
            shipped_weights = json.load(f)
        emit(f"(shipped weights for reference, {SHIPPED_WEIGHTS_PATH}: {shipped_weights})")
    top_buckets = [b for b, _ in taxonomy.most_common(2)]
    for bucket in top_buckets:
        emit(f"\n### Bucket: {bucket}")
        bucket_games = [g for g in losses if classify_ending(g) == bucket]
        for g in bucket_games[:3]:
            emit(f"game {g['game']} vs {g.get('opponent_team_name')}: "
                 f"{len(g['decisions'])} logged decisions, terminal={g.get('terminal')}")
            if g["decisions"]:
                mid = g["decisions"][len(g["decisions"]) // 2]
                buf = io_.StringIO()
                with contextlib.redirect_stdout(buf):
                    loss_review.print_example(g, mid, f"mid-game decision in a {bucket} loss")
                text = buf.getvalue()
                sys.stdout.write(text)
                report.append(text.rstrip("\n"))
    emit("")

    # (d) tied-and-lost / Dwebble check
    emit(f"## (d) Attacker-starvation target check (Dwebble pipeline hypothesis)")
    events = starvation_events(losses)
    dwebble_n = sum(1 for _, _, d in events if d)
    emit(f"starvation events in ladder losses: {len(events)}  |  targeted Dwebble "
         f"(pre-evolution, {DWEBBLE_ID}): {dwebble_n}  |  other target: "
         f"{len(events) - dwebble_n}")
    emit("Quantitative tied-vs-weight-imbalance split: NOT recomputable from ladder data (no "
         "evaluate() scores -- see Known Limitations). Qualitative read: the shipped weights "
         "(above) put a large negative weight on `exposed_investment` and a large positive "
         "weight on `we_threaten_ko`/`bench_attacker_advantage_bonus` -- starvation decisions "
         "where those two point in opposite directions (our Crustle is exposed AND we could "
         "threaten a KO some other way) are the board states most likely to have been genuine "
         "near-ties, by the same reasoning the local dry-run's tied_and_lost examples showed.")
    emit("")

    # (e) secondary metrics
    emit(f"## (e) Secondary metrics")
    ad, ad_n = attack_decline_rate(losses)
    ed, ed_n = evolve_decline_rate(losses)
    nt = near_tie_rate(losses)
    emit(f"attack-decline rate (ladder losses): {ad}/{ad_n}" +
         (f" ({ad / ad_n:.1%})" if ad_n else " (n/a)"))
    emit(f"evolve-decline rate (ladder losses): {ed}/{ed_n}" +
         (f" ({ed / ed_n:.1%})" if ed_n else " (n/a)"))
    emit(f"near-tie rate (ladder losses): N/A -- no scores in ladder data" if nt is None else
         f"near-tie rate (ladder losses): {nt[0]}/{nt[1]} ({nt[0] / nt[1]:.1%})")
    emit("")

    # (f) opponent picture
    emit(f"## (f) Opponent picture")
    try:
        lb_rows = fetch_leaderboard_rows()
        score_by_team = {r["TeamName"].strip().lower(): float(r["Score"]) for r in lb_rows
                          if r.get("Score")}
        our_row = next((r for r in lb_rows
                         if r["TeamName"].strip().lower() == OUR_TEAM_NAME.lower()), None)
        our_score = float(our_row["Score"]) if our_row else None
    except Exception as e:
        emit(f"could not fetch leaderboard for opponent-strength join: {e!r}")
        score_by_team, our_score = {}, None

    if our_score is not None:
        emit(f"our current leaderboard score: {our_score}")
        bucket_counts: dict[str, Counter] = {"stronger": Counter(), "similar": Counter(),
                                              "weaker": Counter(), "unknown": Counter()}
        for g in games:
            opp_name = (g.get("opponent_team_name") or "").strip().lower()
            opp_score = score_by_team.get(opp_name)
            if opp_score is None:
                bucket = "unknown"
            elif opp_score > our_score + STRENGTH_THRESHOLD:
                bucket = "stronger"
            elif opp_score < our_score - STRENGTH_THRESHOLD:
                bucket = "weaker"
            else:
                bucket = "similar"
            bucket_counts[bucket][g["outcome"]] += 1
        for bucket, counts in bucket_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
            emit(f"  vs {bucket} opponents: {total} games -- {dict(counts)}")
    else:
        emit("could not resolve our own leaderboard score -- skipping opponent-strength join.")
    emit("")

    # (5) comparison vs local dry-run
    emit(f"## Comparison vs. local dry-run (2026-07-22)")
    local_losses = _load_local_replay_games()
    if local_losses:
        l_ad, l_ad_n = attack_decline_rate(local_losses)
        l_ed, l_ed_n = evolve_decline_rate(local_losses)
        emit(f"local losses analyzed: {len(local_losses)}  |  ladder losses analyzed: "
             f"{len(losses)}")
        emit(f"local attack-decline rate: {l_ad}/{l_ad_n}" +
             (f" ({l_ad / l_ad_n:.1%})" if l_ad_n else ""))
        emit(f"local evolve-decline rate: {l_ed}/{l_ed_n}" +
             (f" ({l_ed / l_ed_n:.1%})" if l_ed_n else ""))
        emit(f"ladder attack-decline rate: {ad}/{ad_n}" + (f" ({ad / ad_n:.1%})" if ad_n else ""))
        emit(f"ladder evolve-decline rate: {ed}/{ed_n}" + (f" ({ed / ed_n:.1%})" if ed_n else ""))
    else:
        emit("runs/autopsy_local/*.jsonl not found this session -- referencing prose numbers "
             "from VERSIONS.md/the 2026-07-22 conversation instead: local vs. baseline was "
             "30% win rate (n=60) with 87.6% near-tie rate and 25.5% evolve-decline rate.")
    emit("")

    # ranked fix candidates
    emit(f"## Ranked fix candidates (evidence-based, NOT implemented this cycle)")
    emit("See report body above for the evidence counts behind each of these; each would be "
         "verified via scripts/build_kernel_bakeoff.py's 400+-game kernel gate before shipping.")

    _write_report(report)


def _load_local_replay_games(losses_only: bool = True) -> list[dict]:
    """Local replay files log every game (win and loss), unlike the ladder pull (which only
    ever pulls our own team's episodes, all outcomes) -- filter to losses by default so
    local-vs-ladder rate comparisons stay apples-to-apples with the ladder `losses` list."""
    games = []
    for name in ("replay_vs_baseline.jsonl", "replay_vs_random.jsonl"):
        path = os.path.join(LOCAL_REPLAY_DIR, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        games.append(json.loads(line))
    if losses_only:
        games = [g for g in games if g["outcome"] == "opponent_win"]
    return games


def _write_no_ladder_tail(report: list[str], emit) -> None:
    """Sections (5) and ranked-fix-candidates, computed from local-only evidence, for the case
    where zero ladder games are available yet."""
    emit(f"## Comparison vs. local dry-run (2026-07-22) -- ladder side not yet available")
    local_losses = _load_local_replay_games()
    if local_losses:
        l_ad, l_ad_n = attack_decline_rate(local_losses)
        l_ed, l_ed_n = evolve_decline_rate(local_losses)
        l_events = starvation_events(local_losses)
        l_nt = near_tie_rate(local_losses)
        emit(f"local losses analyzed: {len(local_losses)}")
        emit(f"local attack-decline rate: {l_ad}/{l_ad_n}" +
             (f" ({l_ad / l_ad_n:.1%})" if l_ad_n else ""))
        emit(f"local evolve-decline rate: {l_ed}/{l_ed_n}" +
             (f" ({l_ed / l_ed_n:.1%})" if l_ed_n else ""))
        emit(f"local starvation events: {len(l_events)} "
             f"(dwebble-targeted: {sum(1 for _, _, d in l_events if d)})")
        emit(f"local near-tie rate: {l_nt[0]}/{l_nt[1]} ({l_nt[0] / l_nt[1]:.1%})"
             if l_nt else "local near-tie rate: n/a")
        emit("ladder side: no games yet this cycle -- see 'Why zero ladder games' above.")
    else:
        emit("runs/autopsy_local/*.jsonl not found this session -- referencing prose numbers "
             "from VERSIONS.md/the 2026-07-22 conversation instead: local vs. baseline was "
             "30% win rate (n=60) with 87.6% near-tie rate and 25.5% evolve-decline rate.")
    emit("")

    emit(f"## Ranked fix candidates (LOCAL evidence only -- ladder confirmation pending)")
    emit("NOT implemented this cycle. Each would be verified via "
         "scripts/build_kernel_bakeoff.py's 400+-game kernel gate before shipping, and ideally "
         "re-confirmed against real ladder losses once they exist (this cycle couldn't -- 0 "
         "ladder games available).")
    emit("1. **Tie-break-on-starvation fix** -- evidence: local dry-run found "
         "14/32 (43.8%) attacker-starvation cases in the 'tied_and_lost' bucket (agent's own "
         "tie-break should have caught these), plus an 87.6% overall near-tie rate. Expected "
         "impact: HIGH if real -- this is the single largest local anomaly across two "
         "diagnosis cycles now. Verify: kernel bake-off of a revised tie-break rule against "
         "shipped v1, 400+ games.")
    emit("2. **Evolve-decline reduction** -- evidence: 25.5% of legal evolve opportunities "
         "declined in local losses (12/47). Expected impact: MEDIUM -- delayed evolution "
         "compounds into the energy/tempo problems already diagnosed. Verify: same kernel "
         "gate, isolate this fix alone to avoid confounding with #1.")
    emit("3. **Re-run this exact report once real ladder losses exist** -- not a code fix, but "
         "the highest-priority NEXT STEP: confirm whether local-only findings (#1, #2) "
         "actually hold against real opponents before spending a FIX cycle on them.")


def _write_report(lines: list[str]) -> None:
    import datetime
    date_str = datetime.date.today().isoformat()
    out_path = os.path.join(ROOT, "docs", f"ladder_autopsy_{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
