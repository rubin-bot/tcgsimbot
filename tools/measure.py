"""Stage 6 MEASURE: pulls current ladder mu into VERSIONS.md, and extracts any of our games
from the newest downloadable Kaggle episode-replay dataset.

Two independent jobs, either can fail without blocking the other:

1. mu: `kaggle competitions submissions --csv`, newest row, update VERSIONS.md's most recent
   "## vN" section's "**mu (Kaggle ladder):**" line in place (with an "as of <timestamp>" note
   -- mu moves for hours/days after a submission lands, so a bare number isn't a settled
   result; see VERSIONS.md's v1 entry for why this matters -- it moved 524.6 -> 466.6 between
   two calls made minutes apart on its first day).

2. episodes: daily datasets (kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD) publish
   ~00:00 UTC the day AFTER the games they contain, so today's games are never in today's
   dump -- this walks backward from today until it finds one that exists. There's no
   per-team index (the small ...-episodes-index dataset is just a per-day manifest of
   episode_count/total_bytes, not a team map -- confirmed by inspecting it directly), so this
   downloads the day's full zip (~700MB+ compressed) but avoids decompressing anything it
   doesn't have to: it string-searches each entry's first few KB (info.TeamNames appears near
   the top of every episode JSON, confirmed against a real sample) before deciding to extract
   the full file, so only OUR episodes ever get fully written to disk.

Writes runs/measure_state.json so tools/autopsy.py --source auto knows whether ladder data
exists without re-hitting the API itself.

Usage:
  .venv/Scripts/python tools/measure.py
  .venv/Scripts/python tools/measure.py --skip-episodes   # mu only, faster
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from kaggle_common import OUR_TEAM_NAME, SIMULATION_COMPETITION, kaggle_cmd  # noqa: E402

# Console is cp1252 on this machine (CLAUDE.md's Tooling notes); this file prints "μ" --
# reconfigure rather than crash mid-run on a Windows terminal that hasn't set
# PYTHONIOENCODING=utf-8 itself.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VERSIONS_PATH = os.path.join(ROOT, "VERSIONS.md")
RUNS_DIR = os.path.join(ROOT, "runs")
MEASURE_STATE_PATH = os.path.join(RUNS_DIR, "measure_state.json")
EPISODES_INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_DATASET_PREFIX = "kaggle/pokemon-tcg-ai-battle-episodes-"
MAX_DAYS_BACK = 5
SEARCH_CHUNK_BYTES = 16_384  # info.TeamNames is near the top of every episode JSON (confirmed)


# ---------------------------------------------------------------------------
# 1. mu -> VERSIONS.md
# ---------------------------------------------------------------------------

def fetch_submissions() -> list[dict]:
    cmd = kaggle_cmd("competitions", "submissions", "-c", SIMULATION_COMPETITION, "--csv")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def update_versions_mu(submissions: list[dict]) -> str | None:
    if not submissions:
        print("no submissions returned; skipping VERSIONS.md update.", file=sys.stderr)
        return None
    newest = submissions[0]  # kaggle returns newest-first
    mu = newest["publicScore"]
    checked_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    new_line = (f"**μ (Kaggle ladder):** {mu} (as of {checked_at}, submitted "
                f"{newest['date']} -- may still be settling; re-run tools/measure.py to "
                f"refresh)")

    with open(VERSIONS_PATH, encoding="utf-8") as f:
        text = f.read()

    section_starts = [m.start() for m in re.finditer(r"^## v\d+", text, re.MULTILINE)]
    if not section_starts:
        print("no '## vN' section found in VERSIONS.md; not updating.", file=sys.stderr)
        return mu
    last_start = section_starts[-1]
    section = text[last_start:]

    mu_line_re = re.compile(r"^\*\*μ \(Kaggle ladder\):\*\*.*$", re.MULTILINE)
    if mu_line_re.search(section):
        new_section = mu_line_re.sub(new_line, section, count=1)
    else:
        new_section = section.rstrip("\n") + "\n\n" + new_line + "\n"

    text = text[:last_start] + new_section
    with open(VERSIONS_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"VERSIONS.md updated: {new_line}")
    return mu


# ---------------------------------------------------------------------------
# 2. newest episode dataset -> our episodes only
# ---------------------------------------------------------------------------

def _dataset_exists(slug: str) -> bool:
    probe_dir = os.path.join(RUNS_DIR, "_dataset_probe")
    os.makedirs(probe_dir, exist_ok=True)
    result = subprocess.run(kaggle_cmd("datasets", "metadata", slug, "-p", probe_dir),
                             capture_output=True, text=True)
    return result.returncode == 0


def find_newest_dataset_date() -> str | None:
    today = datetime.date.today()
    for delta in range(MAX_DAYS_BACK):
        d = today - datetime.timedelta(days=delta)
        slug = f"{DAILY_DATASET_PREFIX}{d.isoformat()}"
        if _dataset_exists(slug):
            return d.isoformat()
    return None


def download_and_extract_our_episodes(date_str: str, out_dir: str) -> int:
    slug = f"{DAILY_DATASET_PREFIX}{date_str}"
    download_dir = os.path.join(RUNS_DIR, "_episode_downloads")
    os.makedirs(download_dir, exist_ok=True)
    zip_path = os.path.join(download_dir, f"{slug.split('/')[-1]}.zip")

    if not os.path.exists(zip_path):
        print(f"downloading {slug} (compressed daily dump, can be 700MB+) ...")
        subprocess.run(kaggle_cmd("datasets", "download", "-d", slug, "-p", download_dir),
                        check=True)
    else:
        print(f"{zip_path} already downloaded, reusing.")

    needle = OUR_TEAM_NAME.encode("utf-8")
    os.makedirs(out_dir, exist_ok=True)
    found = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"scanning {len(names)} episodes for team {OUR_TEAM_NAME!r} "
              f"(reading a small header chunk of each, not the full file) ...")
        for i, name in enumerate(names):
            with zf.open(name) as f:
                chunk = f.read(SEARCH_CHUNK_BYTES)
            if needle in chunk:
                zf.extract(name, out_dir)
                found += 1
                print(f"  match: {name}")
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1}/{len(names)} scanned, {found} matches so far")

    # the full daily zip has done its job once scanned; delete it rather than let a 700MB+
    # file per day accumulate under runs/ (gitignored, but still real disk space).
    os.remove(zip_path)
    return found


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-episodes", action="store_true",
                     help="only update mu, skip the (slow, 700MB+) episode-dataset search")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    submissions = fetch_submissions()
    for s in submissions:
        print(f"  {s['date']}  {s['fileName']:<40} status={s['status']:<28} "
              f"publicScore={s['publicScore']}")
    mu = update_versions_mu(submissions)

    state = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mu": mu,
        "newest_dataset_date": None,
        "our_episodes_dir": None,
        "our_episode_count": 0,
    }

    if not args.skip_episodes:
        newest_date = find_newest_dataset_date()
        state["newest_dataset_date"] = newest_date
        if newest_date is None:
            print(f"no episode dataset found in the last {MAX_DAYS_BACK} days -- unexpected, "
                  f"check Kaggle's episode publishing status.", file=sys.stderr)
        else:
            print(f"newest available episode dataset: {newest_date}")
            out_dir = os.path.join(RUNS_DIR, "our_episodes", newest_date)
            count = download_and_extract_our_episodes(newest_date, out_dir)
            state["our_episodes_dir"] = out_dir if count else None
            state["our_episode_count"] = count
            print(f"found {count} of our episodes in the {newest_date} dataset.")
            if count == 0:
                print("0 is expected if our latest submission postdates this dump, or the "
                      "ladder hasn't matched us into many games yet -- not necessarily a bug.")

    with open(MEASURE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"\nwrote {MEASURE_STATE_PATH}")


if __name__ == "__main__":
    main()
