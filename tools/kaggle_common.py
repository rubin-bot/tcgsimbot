"""Shared Kaggle-CLI plumbing for tools/scripts that need `python -m kaggle`.

The `kaggle` package lives in the base Python install (the `py` launcher), not this repo's
`.venv` (confirmed: absent from `.venv`, present as CLI 2.2.3 under `py`) -- prefer whichever
interpreter actually has it importable, checking the current one first so this still works if
that ever changes. Factored out of scripts/submit.py so tools/measure.py and the kernel-bakeoff
tooling don't each carry their own copy.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

SIMULATION_COMPETITION = "pokemon-tcg-ai-battle"

# This account's leaderboard TeamName (kaggle competitions leaderboard --csv,
# TeamMemberUserNames == "rubinsahota") -- update here if the team is ever renamed.
OUR_TEAM_NAME = "Rubin Sahota"


def find_kaggle_python() -> list[str]:
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


def kaggle_cmd(*args: str) -> list[str]:
    return find_kaggle_python() + ["-m", "kaggle", *args]


def get_kaggle_username() -> str:
    """kernel-metadata.json's "id" field needs "<username>/<slug>" -- read it from the
    authenticated CLI's own config rather than hardcoding it, so this keeps working if the
    account ever changes."""
    result = subprocess.run(kaggle_cmd("config", "view"), capture_output=True, text=True,
                             check=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("Could not find 'username' in `kaggle config view` output:\n" + result.stdout)
