"""Stage 3: CMA-ES tuning of agents/search_scorer.py's WEIGHTS dict.

Fitness = win rate through tools/eval_arena.py's harness (per-game subprocess isolation,
crash-safe, unchanged). Noise control, exactly as specified:
  * Every candidate in a generation plays the SAME (opponent, seed) schedule -- common random
    numbers, so differences in win rate reflect the weights, not shuffle/coin-flip luck.
  * Each candidate's 60+ games are 2/3 vs. `baseline` (the strongest rule-based reference in
    this repo) and 1/3 self-relative vs. the CURRENT BEST weights found so far -- frozen for
    the whole generation, updated only between generations (updating mid-generation would
    reintroduce the exact noise the common-seed design removes).
  * Self-relative games need TWO differently-weighted search_scorer instances in the SAME
    process/game -- agents/search_scorer.py's weights parameter (threaded through
    evaluate/choose_action/agent/make_agent this session) makes that possible via
    --candidate-weights/--opponent-weights JSON files.

Fully resumable: runs/tune_<name>/ holds the pickled CMA-ES state, an append-only
generations.jsonl (every candidate's weights + fitness), and best.json (running best). A
restart with the same --name picks up from the last completed generation automatically --
no --resume flag needed, resuming IS the default when state exists.

Once --generations is reached, the same invocation auto-runs finalization: the top 3
candidates (by win rate, across ALL logged generations) each play 300 games vs. baseline +
100 vs. random + 100 vs. the pre-tuning weights, selected by POOLED win rate across all 500 --
not just the baseline slice -- and the full report (validation table, weight deltas from the
hand-set values, behavioral metrics) is written to finalized.json and printed.

Usage:
  .venv/Scripts/python tools/tune_weights.py --name run1 --pop-size 10 --generations 10
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from search_scorer import WEIGHTS  # noqa: E402

from eval_arena import run_one_game, psutil_rss_monitoring_works  # noqa: E402
import loss_review  # noqa: E402

WEIGHT_KEYS = list(WEIGHTS.keys())
DECK = os.path.join(ROOT, "decks", "crustle_wall_deck.csv")
RUNS_DIR = os.path.join(ROOT, "runs")


def _looks_like_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
                or os.path.isdir("/kaggle"))


def _wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _state_dir(name: str) -> str:
    d = os.path.join(RUNS_DIR, f"tune_{name}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_schedule(opponent: str, n: int, seed_base: int) -> list[tuple[str, int, int]]:
    """[(opponent_name, seed, candidate_seat), ...] -- a fixed, reusable schedule."""
    return [(opponent, seed_base + i, i % 2) for i in range(n)]


def _mixed_schedule(n_games: int, seed_base: int) -> list[tuple[str, int, int]]:
    n_baseline = round(n_games * 2 / 3)
    n_self = n_games - n_baseline
    sched = _build_schedule("baseline", n_baseline, seed_base)
    sched += _build_schedule("search_scorer", n_self, seed_base + n_baseline)
    return sched


def _run_batch(candidate_weights_path: str, schedule: list[tuple[str, int, int]],
               self_relative_weights_path: str | None, workers: int, timeout: float,
               rss_cap_mb: float, enforce_rss: bool) -> dict:
    """Plays `schedule` with a fixed candidate weight set. Returns aggregate + per-opponent
    breakdown. Crash attribution counts toward win/loss like any other outcome (candidate-side
    crash = loss, opponent-side = win, unattributed = loss -- conservative)."""
    python_exe = sys.executable

    def _one(item):
        opponent, seed, cseat = item
        opp_weights = self_relative_weights_path if opponent == "search_scorer" else None
        return opponent, run_one_game(
            python_exe, "search_scorer", opponent, DECK, DECK, cseat, seed, timeout,
            rss_cap_mb, enforce_rss, None, seed,
            candidate_weights=candidate_weights_path, opponent_weights=opp_weights,
        )

    per_opponent: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        for opponent, rec in ex.map(_one, schedule):
            bucket = per_opponent.setdefault(opponent, {"wins": 0, "draws": 0, "n": 0,
                                                          "crashes": []})
            bucket["n"] += 1
            if rec["outcome"] == "candidate_win":
                bucket["wins"] += 1
            elif rec["outcome"] == "draw":
                bucket["draws"] += 1
            elif rec["outcome"] == "crash":
                bucket["crashes"].append(rec)
                if rec["crashing_side"] == "opponent":
                    bucket["wins"] += 1
                # candidate-side or unattributed crash: counts as a loss (no increment needed)

    total_wins = sum(b["wins"] for b in per_opponent.values())
    total_draws = sum(b["draws"] for b in per_opponent.values())
    total_n = sum(b["n"] for b in per_opponent.values())
    win_rate = (total_wins + 0.5 * total_draws) / total_n if total_n else 0.0
    return {"win_rate": win_rate, "wins": total_wins, "draws": total_draws, "n": total_n,
            "per_opponent": per_opponent}


def _vector_to_weights(vec) -> dict[str, float]:
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, vec)}


def run_generations(name: str, pop_size: int, generations: int, games_per_candidate: int,
                     workers: int, timeout: float, rss_cap_mb: float, enforce_rss: bool,
                     base_seed: int) -> None:
    state_dir = _state_dir(name)
    pretuning_path = os.path.join(state_dir, "pretuning_weights.json")
    if not os.path.exists(pretuning_path):
        _write_json(pretuning_path, WEIGHTS)

    cma_state_path = os.path.join(state_dir, "cma_state.pkl")
    generations_path = os.path.join(state_dir, "generations.jsonl")
    best_path = os.path.join(state_dir, "best.json")

    if os.path.exists(cma_state_path):
        with open(cma_state_path, "rb") as f:
            es = pickle.load(f)
        start_gen = sum(1 for _ in open(generations_path, encoding="utf-8")) \
            if os.path.exists(generations_path) else 0
        print(f"[tune_weights] resuming '{name}' from generation {start_gen} "
              f"(popsize={es.popsize})")
    else:
        x0 = [WEIGHTS[k] for k in WEIGHT_KEYS]
        stds = [max(abs(WEIGHTS[k]), 0.5) * 0.5 for k in WEIGHT_KEYS]
        opts = {"popsize": pop_size, "CMA_stds": stds, "bounds": [-20, 20],
                "verbose": -9, "seed": base_seed if base_seed else 0}
        es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
        start_gen = 0
        print(f"[tune_weights] starting fresh run '{name}': popsize={es.popsize}, "
              f"{len(WEIGHT_KEYS)} dims")

    if os.path.exists(best_path):
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
    else:
        best = {"weights": dict(WEIGHTS), "win_rate": -1.0, "generation": -1}
        _write_json(best_path, best)

    for gen in range(start_gen, generations):
        stop_reason = es.stop()
        if stop_reason:
            print(f"[tune_weights] CMA-ES stop() at generation {gen}: {stop_reason}")
            break

        t0 = time.time()
        self_relative_path = os.path.join(state_dir, "_self_relative_current.json")
        _write_json(self_relative_path, best["weights"])  # frozen for this whole generation

        schedule = _mixed_schedule(games_per_candidate, base_seed + gen * 1_000_000)
        solutions = es.ask()

        gen_candidates = []
        cand_path = os.path.join(state_dir, "_candidate_current.json")
        for idx, vec in enumerate(solutions):
            weights_dict = _vector_to_weights(vec)
            _write_json(cand_path, weights_dict)
            result = _run_batch(cand_path, schedule, self_relative_path, workers, timeout,
                                 rss_cap_mb, enforce_rss)
            gen_candidates.append({"weights": weights_dict, "win_rate": result["win_rate"],
                                    "per_opponent": {
                                        name_: {"wins": b["wins"], "draws": b["draws"],
                                                "n": b["n"]}
                                        for name_, b in result["per_opponent"].items()}})
            print(f"[tune_weights] gen {gen} candidate {idx + 1}/{len(solutions)}: "
                  f"win_rate={result['win_rate']:.3f} ({result['wins']}/{result['n']})")

        try:
            os.remove(cand_path)
        except OSError:
            pass

        fitnesses = [-c["win_rate"] for c in gen_candidates]
        es.tell(solutions, fitnesses)

        with open(generations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"generation": gen, "candidates": gen_candidates}) + "\n")

        gen_best = max(gen_candidates, key=lambda c: c["win_rate"])
        if gen_best["win_rate"] > best["win_rate"]:
            best = {"weights": gen_best["weights"], "win_rate": gen_best["win_rate"],
                    "generation": gen}
            _write_json(best_path, best)

        with open(cma_state_path, "wb") as f:
            pickle.dump(es, f)

        elapsed = time.time() - t0
        print(f"[tune_weights] generation {gen} done in {elapsed:.0f}s -- "
              f"best so far win_rate={best['win_rate']:.3f} (gen {best['generation']})")

    try:
        os.remove(os.path.join(state_dir, "_self_relative_current.json"))
    except OSError:
        pass


def _validate_candidate(weights: dict, matchup_schedules: dict[str, list], pretuning_path: str,
                         workers: int, timeout: float, rss_cap_mb: float,
                         enforce_rss: bool, tmp_path: str) -> dict:
    _write_json(tmp_path, weights)
    row: dict = {"weights": weights, "matchups": {}}
    total_wins = total_draws = total_n = 0
    for matchup_name, schedule in matchup_schedules.items():
        opp_weights = pretuning_path if matchup_name == "pretuning" else None
        result = _run_batch(tmp_path, schedule, opp_weights, workers, timeout, rss_cap_mb,
                             enforce_rss)
        row["matchups"][matchup_name] = {
            "wins": result["wins"], "draws": result["draws"], "n": result["n"],
            "win_rate": result["win_rate"],
        }
        total_wins += result["wins"]
        total_draws += result["draws"]
        total_n += result["n"]
    row["pooled_win_rate"] = (total_wins + 0.5 * total_draws) / total_n if total_n else 0.0
    row["pooled_wins"] = total_wins
    row["pooled_draws"] = total_draws
    row["pooled_n"] = total_n
    return row


def finalize(name: str, workers: int, timeout: float, rss_cap_mb: float, enforce_rss: bool,
             base_seed: int) -> None:
    state_dir = _state_dir(name)
    finalized_path = os.path.join(state_dir, "finalized.json")
    if os.path.exists(finalized_path):
        print(f"[tune_weights] '{name}' already finalized -- printing saved report.\n")
        with open(finalized_path, encoding="utf-8") as f:
            report = json.load(f)
        _print_report(report)
        return

    generations_path = os.path.join(state_dir, "generations.jsonl")
    pretuning_path = os.path.join(state_dir, "pretuning_weights.json")

    all_candidates = []
    with open(generations_path, encoding="utf-8") as f:
        for line in f:
            gen_data = json.loads(line)
            for c in gen_data["candidates"]:
                all_candidates.append(c)
    if not all_candidates:
        print("[tune_weights] no candidates logged yet -- run generations first.")
        return

    top3 = sorted(all_candidates, key=lambda c: c["win_rate"], reverse=True)[:3]
    print(f"[tune_weights] validating top {len(top3)} candidates "
          f"(by search-phase win rate) on the full slate...")

    matchup_schedules = {
        "baseline": _build_schedule("baseline", 300, base_seed + 90_000_000),
        "random": _build_schedule("random", 100, base_seed + 91_000_000),
        "pretuning": _build_schedule("search_scorer", 100, base_seed + 92_000_000),
    }

    validation = []
    tmp_path = os.path.join(state_dir, "_finalize_candidate.json")
    for i, cand in enumerate(top3):
        print(f"[tune_weights] validating finalist {i + 1}/3 "
              f"(search win_rate={cand['win_rate']:.3f})...")
        row = _validate_candidate(cand["weights"], matchup_schedules, pretuning_path, workers,
                                   timeout, rss_cap_mb, enforce_rss, tmp_path)
        row["rank_in_search"] = i + 1
        row["search_win_rate"] = cand["win_rate"]
        validation.append(row)
        print(f"[tune_weights] finalist {i + 1}: pooled_win_rate={row['pooled_win_rate']:.3f} "
              f"baseline={row['matchups']['baseline']['win_rate']:.3f} "
              f"random={row['matchups']['random']['win_rate']:.3f} "
              f"pretuning={row['matchups']['pretuning']['win_rate']:.3f}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    winner = max(validation, key=lambda r: r["pooled_win_rate"])
    winner_weights_path = os.path.join(state_dir, "winner_weights.json")
    _write_json(winner_weights_path, winner["weights"])

    deltas = sorted(
        ((k, winner["weights"][k] - WEIGHTS[k], WEIGHTS[k], winner["weights"][k])
         for k in WEIGHT_KEYS),
        key=lambda t: -abs(t[1]),
    )

    bw = winner["matchups"]["baseline"]
    p, lo, hi = _wilson_ci(bw["wins"] + 0.5 * bw["draws"], bw["n"])

    # Behavioral metrics for the winner: one replay-logged confirmation batch vs. baseline,
    # reusing tools/loss_review.py's existing (print-based) analyses via stdout capture.
    replay_path = os.path.join(state_dir, "winner_replay.jsonl")
    if os.path.exists(replay_path):
        os.remove(replay_path)
    behavior_schedule = _build_schedule("baseline", 200, base_seed + 93_000_000)
    python_exe = sys.executable
    for idx, (opponent, seed, cseat) in enumerate(behavior_schedule):
        run_one_game(python_exe, "search_scorer", opponent, DECK, DECK, cseat, seed, timeout,
                     rss_cap_mb, enforce_rss, replay_path, idx,
                     candidate_weights=winner_weights_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        losses = loss_review.load_losses(replay_path)
        loss_review.analyze_attack_availability(losses)
        loss_review.analyze_attacker_starved(losses)
    behavior_report = buf.getvalue()

    report = {
        "name": name, "validation": validation, "winner_rank_in_search": winner["rank_in_search"],
        "winner_baseline_win_rate": p, "winner_baseline_ci": [lo, hi],
        "weight_deltas": [{"key": k, "delta": d, "hand_set": h, "tuned": t}
                           for k, d, h, t in deltas],
        "behavior_report": behavior_report,
        "success_bar_met": p >= 0.60 and bw["n"] >= 300,
    }
    _write_json(finalized_path, report)
    _print_report(report)


def _print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"FINAL REPORT: {report['name']}")
    print("=" * 70)
    print(f"\nWinner is search-phase rank #{report['winner_rank_in_search']} of the top 3.")
    p, (lo, hi) = report["winner_baseline_win_rate"], report["winner_baseline_ci"]
    print(f"Win rate vs. baseline: {p:.3f}  (95% CI [{lo:.3f}, {hi:.3f}], n=300)")
    print(f"60%+ bar met: {report['success_bar_met']}")

    print("\nValidation table (all 3 finalists):")
    header = f"{'rank':>4} {'search_wr':>10} {'baseline':>10} {'random':>8} " \
             f"{'pretuning':>10} {'pooled':>8}"
    print(header)
    for row in sorted(report["validation"], key=lambda r: -r["pooled_win_rate"]):
        m = row["matchups"]
        print(f"{row['rank_in_search']:>4} {row['search_win_rate']:>10.3f} "
              f"{m['baseline']['win_rate']:>10.3f} {m['random']['win_rate']:>8.3f} "
              f"{m['pretuning']['win_rate']:>10.3f} {row['pooled_win_rate']:>8.3f}")

    print("\nWeight movement from hand-set values (winner, sorted by |delta|):")
    for d in report["weight_deltas"]:
        print(f"  {d['key']:<32} {d['hand_set']:>7.3f} -> {d['tuned']:>7.3f}  "
              f"(delta {d['delta']:+.3f})")

    print("\nBehavioral metrics for the winner (200 fresh games vs. baseline):")
    print(report["behavior_report"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--games-per-candidate", type=int, default=60)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--rss-cap-mb", type=float, default=2048.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    on_kaggle = _looks_like_kaggle()
    if args.workers is None:
        workers = 4 if on_kaggle else 2
    else:
        workers = args.workers
        if not on_kaggle and workers > 2:
            print(f"WARNING: --workers {workers} on a non-Kaggle machine -- capping to 2. "
                  f"This will still be slow; CMA-ES over {args.pop_size} candidates x "
                  f"{args.games_per_candidate} games/generation is a lot of games.",
                  file=sys.stderr)
            workers = 2
    print(f"[tune_weights] platform={'kaggle' if on_kaggle else 'local'} workers={workers}")

    enforce_rss = psutil_rss_monitoring_works()
    if not enforce_rss:
        print("WARNING: RSS cap self-test failed (see tools/eval_arena.py's module docstring) "
              "-- only --timeout is a real safety net this run.", file=sys.stderr)

    run_generations(args.name, args.pop_size, args.generations, args.games_per_candidate,
                     workers, args.timeout, args.rss_cap_mb, enforce_rss, args.seed)
    finalize(args.name, workers, args.timeout, args.rss_cap_mb, enforce_rss, args.seed)


if __name__ == "__main__":
    main()
