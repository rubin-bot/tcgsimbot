"""Thin wrapper around the Kaggle CLI submission command for the Simulation category
(pokemon-tcg-ai-battle). `kaggle.exe` is not on PATH on this machine -- invoke via
`python -m kaggle`, per CLAUDE.md's Tooling notes. Submissions are rate-limited to 5/day and
only the latest 2 stay active for scoring -- CLAUDE.md's Submission policy is 1/day so
TrueSkill has time to converge, so this script does not retry or resubmit automatically.

Run scripts/verify_submission.py on the tarball FIRST -- this script doesn't re-verify.

Usage:
  .venv/Scripts/python scripts/submit.py submission_search_scorer.tar.gz -m "v1: SearchScorer"
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys

COMPETITION = "pokemon-tcg-ai-battle"


def _find_kaggle_python() -> list[str]:
    """The `kaggle` package lives in the base Python install (py launcher), not this repo's
    .venv (confirmed: `kaggle` is absent from .venv, present as CLI 2.2.3 under `py`) --
    prefer whichever interpreter actually has it importable, checking the current one first
    so this still works if that ever changes."""
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable]
    if shutil.which("py"):
        probe = subprocess.run(["py", "-c", "import kaggle"], capture_output=True)
        if probe.returncode == 0:
            return ["py"]
    if shutil.which("python"):
        probe = subprocess.run(["python", "-c", "import kaggle"], capture_output=True)
        if probe.returncode == 0:
            return ["python"]
    sys.exit("Could not find a Python interpreter with the `kaggle` package installed "
             "(checked current interpreter, `py`, `python`).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--dry-run", action="store_true",
                     help="print the command without running it")
    args = ap.parse_args()

    python_cmd = _find_kaggle_python()
    cmd = python_cmd + ["-m", "kaggle", "competitions", "submit",
                         "-c", COMPETITION, "-f", args.tarball, "-m", args.message]

    if args.dry_run:
        print(" ".join(cmd))
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
