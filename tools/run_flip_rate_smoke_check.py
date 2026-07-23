"""v3 causal smoke check (Stage 6, A2): re-runs tools/measure_near_tie_hypothesis.py's
replay-and-vote machinery over the real decline/starvation decision set at a chosen
N_DETERMINIZATIONS, then tools/analyze_near_tie_results.py's flip-rate/margin analysis over the
result -- a cheap, real-decision-level check that the aggregation actually collapses
determinization variance BEFORE spending kernel time on a 400-game gate (the failure mode this
guards against: a fix that looks correct in isolated repro cases but doesn't move the real
population -- exactly what happened to v2's own report).

Sets agents/search_scorer.py's N_DETERMINIZATIONS via monkeypatch (no file edits needed) and
redirects the replay output to a labeled file so multiple N values can be compared side by side
without clobbering each other or the canonical replays.jsonl.

Usage:
  PYTHONIOENCODING=utf-8 python tools/run_flip_rate_smoke_check.py --n 1 --label n1_sanity
  PYTHONIOENCODING=utf-8 python tools/run_flip_rate_smoke_check.py --n 8 --label n8_v3
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

OUT_DIR = os.path.join(ROOT, "runs", "near_tie_measurement")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="N_DETERMINIZATIONS to use")
    ap.add_argument("--label", required=True, help="output file label, e.g. n8_v3")
    args = ap.parse_args()

    import search_scorer
    search_scorer.N_DETERMINIZATIONS = args.n
    print(f"search_scorer.N_DETERMINIZATIONS forced to {args.n}")

    import measure_near_tie_hypothesis as mnth
    replays_path = os.path.join(OUT_DIR, f"replays_{args.label}.jsonl")
    mnth.REPLAYS_PATH = replays_path
    # mnth.main() -> run_replays() is itself resumable (load_done_keys() skips
    # already-completed (game, decision_index) pairs) -- just let it continue.
    if os.path.exists(replays_path):
        print(f"{replays_path} already exists -- resuming (skipping already-done decisions).")
    mnth.main()

    import analyze_near_tie_results as anlz
    anlz.REPLAYS_PATH = replays_path  # keep in sync (name-bound copy, see module docstring)
    buf = io.StringIO()
    with redirect_stdout(buf):
        anlz.main()
    report = buf.getvalue()
    print(report)

    report_path = os.path.join(OUT_DIR, f"flip_rate_report_{args.label}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"N_DETERMINIZATIONS={args.n}\n\n")
        f.write(report)
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
