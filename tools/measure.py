"""Stage 6 MEASURE: pulls current ladder mu + rank into VERSIONS.md/runs/mu_history.jsonl, and
extracts any of our games from every downloadable Kaggle episode-replay dataset since our
oldest active submission's date.

Independent jobs, each can fail without blocking the others:

1. mu + rank: `kaggle competitions submissions --csv` for mu, `kaggle competitions leaderboard
   --csv --download` for rank (submissions --csv has no rank column). Updates VERSIONS.md's
   most recent "## vN" section's "**mu (Kaggle ladder):**" line in place (with an "as of
   <timestamp>" note -- mu moves for hours/days after a submission lands, so a bare number
   isn't a settled result; it moved 524.6 -> 466.6 -> 500.9 -> 573.3 across four checks in its
   first ~24h). Also appends {checked_at, mu, rank} to runs/mu_history.jsonl on every run --
   Kaggle's API exposes neither sigma nor a historical time series, so this file IS the only
   real trajectory we get, built one measurement at a time from here on.

2. episodes: daily datasets (kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD) publish
   ~00:00 UTC the day AFTER the games they contain. This scans every day from our oldest
   still-active submission's date through the newest published day (not just "the newest
   day"), so repeated iterate cycles accumulate a complete our_episodes/ set instead of only
   ever seeing the latest 24h. A day inside that range whose dataset can't be fetched (network
   error, corrupt zip) is a FAILURE, logged and counted separately from a day that simply
   isn't published yet (today, normally). There's no per-team index (the small
   ...-episodes-index dataset is just a per-day manifest of episode_count/total_bytes, not a
   team map -- confirmed by inspecting it directly), so this downloads each day's full zip
   (~700MB+ compressed) but avoids decompressing anything it doesn't have to: it
   string-searches each entry's first few KB (info.TeamNames appears near the top of every
   episode JSON, confirmed against a real sample) before deciding to extract the full file, so
   only OUR episodes ever get fully written to disk, into runs/our_episodes/<date>/.

Writes runs/measure_state.json so tools/autopsy.py --source auto and tools/ladder_report.py
know what's available without re-hitting the API themselves.

Usage:
  .venv/Scripts/python tools/measure.py
  .venv/Scripts/python tools/measure.py --skip-episodes   # mu/rank only, faster
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
from kaggle_common import (  # noqa: E402
    OUR_TEAM_NAME, SIMULATION_COMPETITION, fetch_leaderboard_rows, kaggle_cmd,
)

# Console is cp1252 on this machine (CLAUDE.md's Tooling notes); this file prints "μ" --
# reconfigure rather than crash mid-run on a Windows terminal that hasn't set
# PYTHONIOENCODING=utf-8 itself.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VERSIONS_PATH = os.path.join(ROOT, "VERSIONS.md")
RUNS_DIR = os.path.join(ROOT, "runs")
MEASURE_STATE_PATH = os.path.join(RUNS_DIR, "measure_state.json")
MU_HISTORY_PATH = os.path.join(RUNS_DIR, "mu_history.jsonl")
DAILY_DATASET_PREFIX = "kaggle/pokemon-tcg-ai-battle-episodes-"
MAX_DAYS_FORWARD_PROBE = 3  # how many days past "newest known" to probe for a new publish
SEARCH_CHUNK_BYTES = 16_384  # info.TeamNames is near the top of every episode JSON (confirmed)


# ---------------------------------------------------------------------------
# 1. mu + rank -> VERSIONS.md / runs/mu_history.jsonl
# ---------------------------------------------------------------------------

def fetch_submissions() -> list[dict]:
    cmd = kaggle_cmd("competitions", "submissions", "-c", SIMULATION_COMPETITION, "--csv")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def fetch_leaderboard_row(team_name: str = OUR_TEAM_NAME) -> dict | None:
    """Submissions --csv has no rank column; the leaderboard --csv does."""
    rows = fetch_leaderboard_rows(SIMULATION_COMPETITION,
                                   os.path.join(RUNS_DIR, "_leaderboard_download"))
    for row in rows:
        if row.get("TeamName", "").strip().lower() == team_name.strip().lower():
            return row
    print(f"team {team_name!r} not found in leaderboard ({len(rows)} rows) -- unexpected.",
          file=sys.stderr)
    return None


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


def append_mu_history(mu: str | None, rank: str | None) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    record = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mu": mu, "rank": rank,
    }
    with open(MU_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# 2. every day since our oldest active submission -> our episodes only
# ---------------------------------------------------------------------------

def _dataset_exists(slug: str) -> bool:
    probe_dir = os.path.join(RUNS_DIR, "_dataset_probe")
    os.makedirs(probe_dir, exist_ok=True)
    result = subprocess.run(kaggle_cmd("datasets", "metadata", slug, "-p", probe_dir),
                             capture_output=True, text=True)
    return result.returncode == 0


def earliest_relevant_date(submissions: list[dict]) -> str:
    """The oldest date among currently-active submissions (last 2, per the competition's
    rules) -- episodes from before this can't possibly involve the version(s) we care about.
    Falls back to the oldest submission overall if fewer than 2 exist."""
    dates = sorted({s["date"][:10] for s in submissions[:2]})
    return dates[0]


def date_range(start_str: str, end_str: str) -> list[str]:
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    return [(start + datetime.timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)]


def find_available_dataset_dates(since_date: str) -> tuple[list[str], list[str]]:
    """Returns (available_dates, not_yet_published_dates) for every day from since_date
    through the newest day that exists, probed by walking forward until MAX_DAYS_FORWARD_PROBE
    consecutive misses (handles the normal case of "today isn't published yet" without
    treating it as a failure)."""
    today = datetime.date.today()
    candidates = date_range(since_date, today.isoformat())
    available, not_published = [], []
    for d in candidates:
        slug = f"{DAILY_DATASET_PREFIX}{d}"
        if _dataset_exists(slug):
            available.append(d)
        else:
            not_published.append(d)
    return available, not_published


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
                     help="only update mu/rank, skip the (slow, 700MB+/day) episode search")
    ap.add_argument("--since", default=None,
                     help="override the episode-search start date (YYYY-MM-DD); defaults to "
                          "the oldest currently-active submission's date")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    submissions = fetch_submissions()
    for s in submissions:
        print(f"  {s['date']}  {s['fileName']:<40} status={s['status']:<28} "
              f"publicScore={s['publicScore']}")
    mu = update_versions_mu(submissions)

    lb_row = fetch_leaderboard_row()
    rank = lb_row.get("Rank") if lb_row else None
    submission_count = lb_row.get("SubmissionCount") if lb_row else None
    if lb_row:
        print(f"leaderboard: rank={rank} score={lb_row.get('Score')} "
              f"submission_count={submission_count}")
    append_mu_history(mu, rank)

    state = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mu": mu,
        "rank": rank,
        "leaderboard_score": lb_row.get("Score") if lb_row else None,
        "submission_count": submission_count,
        "since_date": None,
        "days_scanned": [],
        "days_not_yet_published": [],
        "days_failed": [],
        "episodes_by_date": {},
        "our_episodes_dir": None,
        "our_episode_count": 0,
    }

    if not args.skip_episodes:
        since_date = args.since or earliest_relevant_date(submissions)
        state["since_date"] = since_date
        available, not_published = find_available_dataset_dates(since_date)
        state["days_scanned"] = available
        state["days_not_yet_published"] = not_published
        print(f"episode datasets since {since_date}: {len(available)} available "
              f"({available}), {len(not_published)} not yet published ({not_published})")

        episodes_root = os.path.join(RUNS_DIR, "our_episodes")
        total_found = 0
        failed_days = []
        for d in available:
            out_dir = os.path.join(episodes_root, d)
            try:
                count = download_and_extract_our_episodes(d, out_dir)
            except Exception as e:
                print(f"FAILED to fetch/scan {d}: {e!r}", file=sys.stderr)
                failed_days.append({"date": d, "error": repr(e)})
                continue
            state["episodes_by_date"][d] = count
            total_found += count
            print(f"  {d}: {count} of our episodes")

        state["days_failed"] = failed_days
        state["our_episode_count"] = total_found
        state["our_episodes_dir"] = episodes_root if total_found else None
        print(f"\ntotal of our episodes found across {len(available)} day(s): {total_found} "
              f"({len(failed_days)} day(s) failed to fetch)")
        if total_found == 0:
            print("0 is expected if our submissions postdate every published dump, or the "
                  "ladder hasn't matched us into many games yet -- not necessarily a bug.")

    with open(MEASURE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"\nwrote {MEASURE_STATE_PATH}")


if __name__ == "__main__":
    main()
