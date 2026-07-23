"""Shared Kaggle-CLI plumbing for tools/scripts that need `python -m kaggle`.

The `kaggle` package lives in the base Python install (the `py` launcher), not this repo's
`.venv` (confirmed: absent from `.venv`, present as CLI 2.2.3 under `py`) -- prefer whichever
interpreter actually has it importable, checking the current one first so this still works if
that ever changes. Factored out of scripts/submit.py so tools/measure.py and the kernel-bakeoff
tooling don't each carry their own copy.
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import zipfile

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


def fetch_leaderboard_rows(competition: str = SIMULATION_COMPETITION,
                            download_dir: str | None = None) -> list[dict]:
    """Downloads the full public leaderboard CSV (a small zip -- ~170KB for this competition's
    ~5500 teams, nothing like the 700MB+ episode dumps) and returns every row as a dict
    (Rank/TeamId/TeamName/LastSubmissionDate/Score/SubmissionCount/TeamMemberUserNames).
    Shared by tools/measure.py (our own row, for rank) and tools/ladder_report.py (every row,
    to look up opponents' scores)."""
    if download_dir is None:
        download_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "runs", "_leaderboard_download")
    os.makedirs(download_dir, exist_ok=True)
    for f in glob.glob(os.path.join(download_dir, "*")):
        os.remove(f)
    subprocess.run(
        kaggle_cmd("competitions", "leaderboard", "-c", competition, "--csv", "--download",
                   "-p", download_dir),
        check=True, capture_output=True,
    )
    zip_paths = glob.glob(os.path.join(download_dir, "*.zip"))
    if not zip_paths:
        return []
    with zipfile.ZipFile(zip_paths[0]) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # utf-8-sig, not utf-8: this CSV ships with a UTF-8 BOM, which otherwise leaks
            # into the first header cell (key becomes '﻿Rank', not 'Rank') and silently
            # breaks any row.get("Rank") lookup -- confirmed against a real download.
            text = io.TextIOWrapper(f, encoding="utf-8-sig")
            rows = list(csv.DictReader(text))
    for f in glob.glob(os.path.join(download_dir, "*")):
        os.remove(f)
    return rows


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
