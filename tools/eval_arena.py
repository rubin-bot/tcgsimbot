"""Local evaluation harness: pits --candidate against --opponent over --games games.

Each game runs in its own subprocess (tools/_eval_worker.py) with a wall-clock timeout and an
RSS memory cap, so a native engine crash or a runaway allocation in one game can never take
down this run or the machine -- it's just logged as a loss for whichever side was moving when
it happened, and the run continues. Nothing about game history is held in RAM: each result is
appended to --out (a .jsonl file) as soon as the game finishes, and --resume picks up from
however many lines are already there.

NOTE on resource.setrlimit: that's a POSIX-only stdlib module and doesn't exist on Windows
(this machine's platform, per CLAUDE.md). The RSS cap here is enforced instead by polling the
child's memory via `psutil` on the same tick as the timeout check. If psutil isn't installed,
the timeout is still enforced but the RSS cap is silently unenforced (a warning is printed).

NOTE on psutil reliability: at startup this harness self-tests psutil against a throwaway
subprocess that allocates a known ~150MB before trusting it. This was added because, in the
sandboxed shell used to build this tool, psutil.Process(pid).memory_info() (rss/vms, and even
memory_full_info().private/.uss) reported ~0-4MB for a child subprocess confirmed (via a
print-after-allocate handshake) to be holding a 400MB array -- a silent false sense of safety
rather than a raised error. If the self-test fails, the RSS cap is disabled with a loud warning
and only --timeout remains as the real safety net; this may still work correctly in a plain
terminal outside this sandbox, so it's re-checked (not cached) on every run.

Usage:
  .venv/Scripts/python tools/eval_arena.py --candidate search_scorer --opponent random \
      --games 50 --out arena_vs_random.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tools", "_eval_worker.py")
DEFAULT_DECK = os.path.join(ROOT, "decks", "crustle_wall_deck.csv")

AGENT_NAMES = ("search_scorer", "baseline", "random")
POLL_INTERVAL_S = 0.2


def psutil_rss_monitoring_works() -> bool:
    """Spawn a throwaway subprocess that allocates ~150MB (past a handshake so we know the
    allocation happened before we measure), and check whether psutil reports at least a
    plausible fraction of that back. See the module docstring for why this exists."""
    if psutil is None:
        return False
    probe_code = (
        "buf = [b'x' * (1 << 20) for _ in range(150)]\n"
        "import sys; sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
        "import time; time.sleep(3)\n"
    )
    proc = None
    try:
        proc = subprocess.Popen([sys.executable, "-c", probe_code],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ready_line = proc.stdout.readline()
        if not ready_line:
            return False
        time.sleep(0.3)
        rss_mb = psutil.Process(proc.pid).memory_info().rss / (1024 * 1024)
        return rss_mb > 50.0
    except Exception:
        return False
    finally:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


def _reader_thread(proc: subprocess.Popen, lines: list, lock: threading.Lock) -> None:
    try:
        for raw in proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip("\n")
            with lock:
                lines.append(text)
    except Exception:
        pass


def run_one_game(python_exe: str, candidate: str, opponent: str, candidate_deck: str,
                  opponent_deck: str, candidate_seat: int, seed: int, timeout_s: float,
                  rss_cap_mb: float, enforce_rss: bool, replay_out: str | None,
                  game_index: int, candidate_weights: str | None = None,
                  opponent_weights: str | None = None) -> dict:
    cmd = [
        python_exe, WORKER,
        "--candidate", candidate, "--opponent", opponent,
        "--candidate-deck", candidate_deck, "--opponent-deck", opponent_deck,
        "--candidate-seat", str(candidate_seat), "--seed", str(seed),
        "--game-index", str(game_index),
    ]
    if replay_out is not None:
        cmd += ["--replay-out", replay_out]
    if candidate_weights is not None:
        cmd += ["--candidate-weights", candidate_weights]
    if opponent_weights is not None:
        cmd += ["--opponent-weights", opponent_weights]
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    lines: list = []
    lock = threading.Lock()
    reader = threading.Thread(target=_reader_thread, args=(proc, lines, lock), daemon=True)
    reader.start()

    ps_proc = None
    if psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None
    check_rss = enforce_rss and ps_proc is not None

    peak_rss_mb = 0.0
    killed_reason = None
    while True:
        if proc.poll() is not None:
            break
        elapsed = time.time() - start
        if elapsed > timeout_s:
            killed_reason = "timeout"
            break
        if ps_proc is not None:
            try:
                rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
                peak_rss_mb = max(peak_rss_mb, rss_mb)
                if check_rss and rss_mb > rss_cap_mb:
                    killed_reason = "rss_exceeded"
                    break
            except psutil.NoSuchProcess:
                break
        time.sleep(POLL_INTERVAL_S)

    if killed_reason is not None:
        try:
            if ps_proc is not None:
                ps_proc.kill()
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    reader.join(timeout=5)
    elapsed = round(time.time() - start, 2)

    with lock:
        collected = list(lines)

    result_line = next((l for l in reversed(collected) if l.startswith("RESULT ")), None)
    if killed_reason is None and result_line is not None:
        payload = json.loads(result_line[len("RESULT "):])
        outcome = payload["outcome"]
        crashing_side = payload.get("crashing_side")
        reason = payload.get("reason")
        fallback_counts = payload.get("fallback_counts", {})
    else:
        last_turn_seat = None
        for l in reversed(collected):
            if l.startswith("TURN "):
                last_turn_seat = int(l.split()[1])
                break
        if last_turn_seat is None:
            # Never even saw a TURN marker -- hung/died before the first real decision. No
            # principled way to attribute this; conservatively charge it to the candidate
            # since that's the side under evaluation.
            crashing_side = "candidate"
            reason = killed_reason or "died_with_no_turn_marker"
        else:
            crashing_side = "candidate" if last_turn_seat == candidate_seat else "opponent"
            reason = killed_reason or "died_without_result_line"
        outcome = "crash"
        fallback_counts = {}

    return {
        "candidate": candidate, "opponent": opponent, "candidate_seat": candidate_seat,
        "seed": seed, "outcome": outcome, "crashing_side": crashing_side, "reason": reason,
        "elapsed_s": elapsed, "peak_rss_mb": round(peak_rss_mb, 1),
        "fallback_counts": fallback_counts,
    }


def load_opponent_pool(path: str) -> list[dict]:
    """--opponent-pool spec: a JSON list of {"name", "deck", "weight"} objects (weights need
    not sum to 1 -- normalized by random.choices). Master-study Workstream C3 usage: a ~1/3
    Alakazam-sparring / 1/3 Munkidori-sparring / 1/3 baseline mix matching docs/
    meta_report_2026-07-22.md's real top-100 archetype distribution."""
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)
    for entry in pool:
        if entry["name"] not in AGENT_NAMES:
            sys.exit(f"--opponent-pool entry {entry!r}: name must be one of {AGENT_NAMES}")
    return pool


def _print_pool_summary(candidate: str, records: list) -> None:
    """Like _print_summary, but for --opponent-pool runs: an overall aggregate PLUS a
    per-opponent-identity breakdown (grouped by (opponent, opponent_deck) since a pool can
    contain the same --opponent name piloting different decks, e.g. two search_scorer entries
    for Alakazam vs Munkidori)."""
    import math
    from collections import defaultdict

    def stats(rows):
        n = len(rows)
        wins = sum(1 for r in rows if r["outcome"] == "candidate_win")
        losses = sum(1 for r in rows if r["outcome"] == "opponent_win")
        draws = sum(1 for r in rows if r["outcome"] == "draw")
        crashes = sum(1 for r in rows if r["outcome"] == "crash")
        wr = (wins + 0.5 * draws) / n if n else 0.0
        se = math.sqrt(wr * (1 - wr) / n) if n else 0.0
        ci = (max(0.0, wr - 1.96 * se), min(1.0, wr + 1.96 * se))
        return {"n": n, "wins": wins, "losses": losses, "draws": draws, "crashes": crashes,
                "win_rate": wr, "win_rate_95ci": ci}

    print(f"\n=== {candidate} vs opponent pool: {len(records)} games ===")
    overall = stats(records)
    print(f"OVERALL: {json.dumps(overall, indent=2)}")

    by_group: dict = defaultdict(list)
    for r in records:
        key = f"{r['opponent']}:{os.path.basename(r.get('opponent_deck', '?'))}"
        by_group[key].append(r)
    print("\nper-matchup breakdown:")
    for key, rows in sorted(by_group.items()):
        s = stats(rows)
        print(f"  {key}: {json.dumps(s)}")


def _print_summary(candidate: str, opponent: str, records: list) -> None:
    n = len(records)
    wins = sum(1 for r in records if r["outcome"] == "candidate_win")
    losses = sum(1 for r in records if r["outcome"] == "opponent_win")
    draws = sum(1 for r in records if r["outcome"] == "draw")
    crashes = [r for r in records if r["outcome"] == "crash"]
    cand_crashes = [r for r in crashes if r["crashing_side"] == "candidate"]
    opp_crashes = [r for r in crashes if r["crashing_side"] == "opponent"]
    win_rate = (wins + 0.5 * draws) / n if n else 0.0

    fb_total: dict = {}
    for r in records:
        for k, v in (r.get("fallback_counts") or {}).items():
            fb_total[k] = fb_total.get(k, 0) + v

    print()
    print(f"=== {candidate} vs {opponent}: {n} games ===")
    print(f"wins={wins} losses={losses} draws={draws} crashes={len(crashes)} "
          f"(candidate-side={len(cand_crashes)}, opponent-side={len(opp_crashes)}, "
          f"unattributed={len(crashes) - len(cand_crashes) - len(opp_crashes)})")
    print(f"candidate win rate (draws=0.5): {win_rate:.3f}")
    if crashes:
        print("crash details:")
        for r in crashes:
            print(f"  game {r['game']} seed={r['seed']}: {r['crashing_side']} - {r['reason']}")
    if fb_total:
        print(f"SearchScorer fallback events across {n} games: {fb_total}")
    elif candidate == "search_scorer" or opponent == "search_scorer":
        print("SearchScorer fallback events: none recorded (search never fell back)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="search_scorer", choices=AGENT_NAMES)
    ap.add_argument("--opponent", required=False, choices=AGENT_NAMES,
                     help="single fixed opponent for the whole run. Mutually exclusive with "
                          "--opponent-pool.")
    ap.add_argument("--opponent-pool", default=None,
                     help="path to a JSON list of {name, deck, weight} objects -- picks a "
                          "DIFFERENT opponent per game via a weighted random draw (seeded off "
                          "--seed + game index, reproducible) instead of one fixed --opponent "
                          "for the whole run. Mutually exclusive with --opponent.")
    ap.add_argument("--candidate-deck", default=DEFAULT_DECK)
    ap.add_argument("--opponent-deck", default=DEFAULT_DECK)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed+i")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-game wall-clock seconds")
    ap.add_argument("--rss-cap-mb", type=float, default=2048.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--replay-out", default=None,
                     help="if set (and --candidate search_scorer), each worker appends a full "
                          "per-decision trace for its game to this JSONL path -- see "
                          "tools/loss_review.py.")
    ap.add_argument("--candidate-weights", default=None,
                     help="JSON file of a WEIGHTS-shaped dict for --candidate search_scorer; "
                          "omit to use agents/search_scorer.py's module default.")
    ap.add_argument("--opponent-weights", default=None,
                     help="Same, for --opponent search_scorer -- lets both sides run "
                          "search_scorer with different weight sets in the same game.")
    args = ap.parse_args()

    if bool(args.opponent) == bool(args.opponent_pool):
        sys.exit("Pass exactly one of --opponent or --opponent-pool.")
    opponent_pool = load_opponent_pool(args.opponent_pool) if args.opponent_pool else None

    if args.workers > 2:
        print(f"WARNING: --workers {args.workers} > 2 -- each worker runs a full engine "
              f"subprocess; CLAUDE.md's hardware rules cap concurrent simulator processes at "
              f"2 on this machine. Proceeding anyway, but consider --workers 1 or 2.",
              file=sys.stderr)

    enforce_rss = False
    if psutil is None:
        print("WARNING: psutil not installed -- the RSS cap will NOT be enforced (only the "
              "wall-clock --timeout applies). Install with "
              "`.venv/Scripts/python -m pip install psutil` for the memory cap to apply.",
              file=sys.stderr)
    else:
        enforce_rss = psutil_rss_monitoring_works()
        if not enforce_rss:
            print("WARNING: psutil is installed but failed a self-test (reported <50MB for a "
                  "subprocess confirmed to hold ~150MB) -- memory-query APIs appear "
                  "non-functional in this environment. The RSS cap will NOT be enforced; only "
                  "--timeout is a real safety net this run. peak_rss_mb in the output may still "
                  "be printed but should not be trusted. See the module docstring.",
                  file=sys.stderr)

    completed = 0
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        completed = len(existing)
        if (not opponent_pool and existing
                and (existing[-1]["candidate"] != args.candidate
                     or existing[-1]["opponent"] != args.opponent)):
            print(f"WARNING: {args.out} was recorded for "
                  f"{existing[-1]['candidate']} vs {existing[-1]['opponent']}, not "
                  f"{args.candidate} vs {args.opponent} -- this looks like the wrong file, "
                  f"continuing anyway.", file=sys.stderr)
        print(f"Resuming from {args.out}: {completed} games already recorded.")
    elif not args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"ERROR: {args.out} already exists and is non-empty. Pass --resume to continue "
              f"it, or a different --out to start fresh.", file=sys.stderr)
        sys.exit(1)

    remaining = args.games - completed
    if remaining <= 0:
        print(f"{args.out} already has {completed} >= requested {args.games} games; nothing "
              f"to do.")
        with open(args.out) as f:
            records = [json.loads(l) for l in f if l.strip()]
        if opponent_pool:
            _print_pool_summary(args.candidate, records)
        else:
            _print_summary(args.candidate, args.opponent, records)
        return

    opponent_label = "opponent pool" if opponent_pool else args.opponent
    print(f"Running {remaining} games: {args.candidate} vs {opponent_label} "
          f"(workers={args.workers}, timeout={args.timeout}s, rss_cap={args.rss_cap_mb}MB) "
          f"-> {args.out}")

    game_indices = list(range(completed, args.games))
    write_lock = threading.Lock()

    def _run(i: int) -> dict:
        seed = args.seed + i
        candidate_seat = i % 2
        if opponent_pool:
            # Reseeded per game index (not a shared advancing Random) so the draw is
            # reproducible regardless of worker-thread execution order.
            choice = random.Random(seed).choices(
                opponent_pool, weights=[e["weight"] for e in opponent_pool], k=1)[0]
            opponent_name, opponent_deck = choice["name"], choice["deck"]
        else:
            opponent_name, opponent_deck = args.opponent, args.opponent_deck
        record = run_one_game(
            args.python, args.candidate, opponent_name, args.candidate_deck,
            opponent_deck, candidate_seat, seed, args.timeout, args.rss_cap_mb,
            enforce_rss, args.replay_out, i,
            candidate_weights=args.candidate_weights, opponent_weights=args.opponent_weights,
        )
        record["game"] = i
        record["opponent_deck"] = opponent_deck
        return record

    with open(args.out, "a", encoding="utf-8") as out_f:
        def _handle(record: dict) -> None:
            with write_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
            tag = record["outcome"]
            extra = (f" [crash: {record['crashing_side']} - {record['reason']}]"
                     if tag == "crash" else "")
            print(f"game {record['game']}: seed={record['seed']} "
                  f"candidate_seat={record['candidate_seat']} -> {tag} "
                  f"({record['elapsed_s']}s, {record['peak_rss_mb']}MB){extra}")

        if args.workers <= 1:
            for i in game_indices:
                _handle(_run(i))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                for record in ex.map(_run, game_indices):
                    _handle(record)

    with open(args.out, encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
    if opponent_pool:
        _print_pool_summary(args.candidate, all_records)
    else:
        _print_summary(args.candidate, args.opponent, all_records)


if __name__ == "__main__":
    main()
