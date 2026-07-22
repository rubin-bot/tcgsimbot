"""Verifies a packaged submission tarball the way the actual Kaggle runtime will see it:
size under the cap, extracted to a clean temp directory, and a smoke-test game run using
ONLY the extracted contents -- its own bundled cg/, not this repo's dev copy -- proving the
package is genuinely self-contained rather than accidentally relying on something outside
the tarball.

The smoke test itself runs in a subprocess (its own OS process), so a native crash there
can't take down this script, and a wall-clock timeout catches a hang. RAM is checked
best-effort via psutil with the same honest caveat as tools/eval_arena.py: this sandbox's
psutil cross-process memory queries were confirmed unreliable (near-zero regardless of true
usage) during Stage 2 -- functional correctness (completes N full games, every move legal,
size under 197.7 MiB) is the real bar here, not a RAM number that can't be trusted anyway.

Usage:
  .venv/Scripts/python scripts/verify_submission.py submission_search_scorer.tar.gz
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SUBMISSION_MIB = 197.7

_SMOKE_TEST_SCRIPT = '''
import random
import sys

sys.path.insert(0, {extracted_dir!r})

import main  # the packaged main.py -- must resolve everything from THIS directory only

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return deck
    n = len(obs.select.option)
    k = obs.select.maxCount
    return random.sample(range(n), k)


with open("deck.csv") as f:
    deck = [int(x) for x in f.read().split("\\n")[:60]]

N_GAMES = {n_games}
for seed in range(N_GAMES):
    random.seed(seed)
    agent_seat = seed % 2
    agents = {{agent_seat: main.agent, 1 - agent_seat: random_agent}}
    obs, start_data = battle_start(deck, deck)
    assert obs is not None, (start_data.errorPlayer, start_data.errorType)
    decisions = 0
    for _ in range(4000):
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            battle_finish()
            break
        yi = st["yourIndex"]
        index_list = agents[yi](obs)
        if yi == agent_seat:
            sel = to_observation_class(obs).select
            assert sel.minCount <= len(index_list) <= sel.maxCount, (
                sel.minCount, len(index_list), sel.maxCount)
            assert len(set(index_list)) == len(index_list)
            assert all(0 <= i < len(sel.option) for i in index_list)
            decisions += 1
        obs = battle_select(index_list)
    else:
        raise AssertionError(f"game seed={{seed}} did not finish within step cap")
    print(f"seed={{seed}} agent_seat={{agent_seat}} result={{st['result']}} "
          f"decisions={{decisions}}", flush=True)
print("SMOKE_TEST_OK", flush=True)
'''


def verify(tarball_path: str, n_games: int, timeout_s: float) -> bool:
    if not os.path.exists(tarball_path):
        print(f"FAIL: {tarball_path} does not exist.", file=sys.stderr)
        return False

    size_mib = os.path.getsize(tarball_path) / 1024 / 1024
    print(f"tarball size: {size_mib:.3f} MiB (limit {MAX_SUBMISSION_MIB} MiB)")
    if size_mib > MAX_SUBMISSION_MIB:
        print("FAIL: tarball exceeds the size limit.", file=sys.stderr)
        return False

    extract_dir = tempfile.mkdtemp(prefix="verify_submission_")
    print(f"extracting to {extract_dir} ...")
    with tarfile.open(tarball_path) as tf:
        tf.extractall(extract_dir, filter="data")  # our own just-built tarball, trusted content

    required = ["main.py", "deck.csv", "cg"]
    missing = [f for f in required if not os.path.exists(os.path.join(extract_dir, f))]
    if missing:
        print(f"FAIL: extracted package missing {missing}.", file=sys.stderr)
        return False

    script_path = os.path.join(extract_dir, "_smoke_test_driver.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(_SMOKE_TEST_SCRIPT.format(extracted_dir=extract_dir, n_games=n_games))

    print(f"running {n_games}-game smoke test using ONLY the extracted package "
          f"(own bundled cg/, own deck.csv) ...")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path], cwd=extract_dir, capture_output=True,
            text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL: smoke test did not finish within {timeout_s}s.", file=sys.stderr)
        return False
    elapsed = time.time() - start

    print(proc.stdout)
    if proc.returncode != 0 or "SMOKE_TEST_OK" not in proc.stdout:
        print(f"FAIL: smoke test exited {proc.returncode} after {elapsed:.1f}s.",
              file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return False

    print(f"PASS: {n_games} full games completed via the extracted package alone "
          f"in {elapsed:.1f}s, every agent move legal, size under the cap.")
    print("NOTE: RAM headroom vs. the 12.2 GiB Kaggle limit was NOT independently measured -- "
          "this agent has no numpy/torch dependency and its per-move footprint is small "
          "Python objects only (confirmed during Stage 5 prep), and psutil cross-process "
          "memory queries are confirmed unreliable in this dev sandbox (see "
          "tools/eval_arena.py's module docstring), so a measured number here couldn't be "
          "trusted anyway. Functional correctness is the real verification this provides.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball", nargs="?",
                     default=os.path.join(REPO_ROOT, "submission_search_scorer.tar.gz"))
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    ok = verify(args.tarball, args.games, args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
