"""Stage 6 MEASURE: pulls current ladder mu + rank into VERSIONS.md/runs/mu_history.jsonl, and
fetches every real episode our active submissions have played.

Independent jobs, each can fail without blocking the others:

1. mu + rank: `kaggle competitions submissions --csv` for mu, `kaggle competitions leaderboard
   --csv --download` for rank (submissions --csv has no rank column). Updates VERSIONS.md's
   most recent "## vN" section's "**mu (Kaggle ladder):**" line in place (with an "as of
   <timestamp>" note -- mu moves for hours/days after a submission lands, so a bare number
   isn't a settled result; it moved 524.6 -> 466.6 -> 500.9 -> 573.3 across four checks in its
   first ~24h). Also appends {checked_at, mu, rank} to runs/mu_history.jsonl on every run --
   Kaggle's API exposes neither sigma nor a historical time series, so this file IS the only
   real trajectory we get, built one measurement at a time from here on.

2. episodes -- PRIMARY method, submission API (added 2026-07-23 audit): `kaggle competitions
   episodes <submission_id> --format json` returns every episode a given submission has played,
   directly and completely -- confirmed against real data (v1, submission 54909461, had ~48
   COMPLETED ladder episodes within its first ~14h, none of which were findable in that day's
   bulk dataset dump; see docs/submission_ladder_audit_2026-07-23.md). Each episode's full
   replay is fetched with `kaggle competitions replay <episode_id> -p <dir>` (skips
   EPISODE_TYPE_VALIDATION -- that's the one self-play sanity game Kaggle runs at submission
   time, not a real ladder game) into runs/our_episodes/<date>/<episode_id>.json, same layout
   tools/autopsy.py already walks.

   FALLBACK method, --also-bulk-scan (off by default): the original approach, downloading each
   day's full episode dataset (kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD, ~700MB+
   compressed) and byte-searching each entry's first 16KB for our team name before extracting a
   match. Confirmed 2026-07-23 that this dataset is a genuine SUBSAMPLE of that day's total
   episodes across ~5500+ teams -- 9/9 of our real, submission-API-confirmed episode IDs from
   2026-07-22 were absent from that day's dump (real 404s, cross-checked against a dataset file
   that does download fine). The byte-search logic itself isn't broken (confirmed against a
   real replay: our team name sits at byte offset 217, well inside the 16KB window) -- the
   dataset it searches just doesn't reliably contain our games. Kept only as a fallback in case
   the submission-API path is ever rate-limited or deprecated.

Writes runs/measure_state.json so tools/autopsy.py --source auto and tools/ladder_report.py
know what's available without re-hitting the API themselves.

Usage:
  .venv/Scripts/python tools/measure.py
  .venv/Scripts/python tools/measure.py --skip-episodes   # mu/rank only, faster
  .venv/Scripts/python tools/measure.py --also-bulk-scan  # also run the fallback dump scan
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
# 2. episodes, PRIMARY: per-submission Kaggle API -- direct, complete, no bulk download
# ---------------------------------------------------------------------------

def fetch_episode_index(submission_id: str) -> list[dict]:
    """`kaggle competitions episodes <id> --format json` -- every episode (validation + ladder)
    that submission has played: {id, createTime, endTime, state, type}. Confirmed 2026-07-23:
    this is a real, complete per-submission index, unlike the bulk daily dataset dump below.
    The CLI prints a JSON array followed by a trailing "Use ... to download a replay" usage
    line on the same stdout stream (confirmed by direct inspection) -- raw_decode() parses just
    the array and ignores that trailing text instead of failing on it as `json.loads` would."""
    cmd = kaggle_cmd("competitions", "episodes", str(submission_id), "--format", "json")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    episodes, _ = json.JSONDecoder().raw_decode(result.stdout.strip())
    return episodes


def download_episode_replay(episode_id: int, out_dir: str) -> str | None:
    """Downloads one episode's full replay JSON, skipping if already present (resumable, per
    Hardware rules). Renames the CLI's `episode-<id>-replay.json` to `<id>.json` to match the
    runs/our_episodes/<date>/<id>.json layout tools/autopsy.py already walks. Returns the final
    path, or None if the download failed (logged, not raised -- one bad episode shouldn't stop
    the rest)."""
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"{episode_id}.json")
    if os.path.exists(final_path):
        return final_path
    cmd = kaggle_cmd("competitions", "replay", str(episode_id), "-p", out_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED to fetch replay for episode {episode_id}: {result.stderr.strip()}",
              file=sys.stderr)
        return None
    downloaded = os.path.join(out_dir, f"episode-{episode_id}-replay.json")
    if not os.path.exists(downloaded):
        print(f"  replay download for {episode_id} reported success but "
              f"{downloaded} is missing.", file=sys.stderr)
        return None
    os.replace(downloaded, final_path)
    return final_path


def fetch_our_episodes_via_submission_api(submissions: list[dict]) -> dict[str, int]:
    """For each of our currently-active submissions (last 2, per the competition's rules),
    fetches its full episode index and downloads every non-validation replay. Returns
    {date: count} across all active submissions combined."""
    episodes_root = os.path.join(RUNS_DIR, "our_episodes")
    counts: dict[str, int] = {}
    for s in submissions[:2]:
        submission_id = s["ref"]
        print(f"fetching episode index for submission {submission_id} ({s['fileName']}) ...")
        try:
            episodes = fetch_episode_index(submission_id)
        except Exception as e:
            print(f"  FAILED to list episodes for submission {submission_id}: {e!r}",
                  file=sys.stderr)
            continue
        # type is e.g. "EpisodeType.EPISODE_TYPE_PUBLIC" / "EpisodeType.EPISODE_TYPE_VALIDATION"
        # (CLI --format json includes the enum class prefix; confirmed by direct inspection).
        ladder_episodes = [e for e in episodes if "VALIDATION" not in (e.get("type") or "")]
        print(f"  {len(episodes)} total ({len(episodes) - len(ladder_episodes)} validation, "
              f"{len(ladder_episodes)} ladder)")
        for ep in ladder_episodes:
            date = ep["createTime"][:10]
            out_dir = os.path.join(episodes_root, date)
            path = download_episode_replay(ep["id"], out_dir)
            if path is not None:
                counts[date] = counts.get(date, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 2b. episodes, FALLBACK (--also-bulk-scan): daily bulk dataset dump
#
# Confirmed 2026-07-23 to be a subsample, not a complete record -- 9/9 of our real episode IDs
# from 2026-07-22 were absent from that day's dump. Kept only in case the submission-API path
# above is ever rate-limited or deprecated.
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
                     help="only update mu/rank, skip fetching episodes entirely")
    ap.add_argument("--also-bulk-scan", action="store_true",
                     help="also run the fallback daily-dataset bulk scan (slow, 700MB+/day, "
                          "confirmed 2026-07-23 to be an unreliable subsample) in addition to "
                          "the primary per-submission API path")
    ap.add_argument("--since", default=None,
                     help="--also-bulk-scan only: override its start date (YYYY-MM-DD); "
                          "defaults to the oldest currently-active submission's date")
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
        "episode_source": None,
        "since_date": None,
        "days_scanned": [],
        "days_not_yet_published": [],
        "days_failed": [],
        "episodes_by_date": {},
        "our_episodes_dir": None,
        "our_episode_count": 0,
    }

    if not args.skip_episodes:
        episodes_root = os.path.join(RUNS_DIR, "our_episodes")
        episodes_by_date: dict[str, int] = {}
        sources = ["submission_api"]

        print("fetching our episodes via the per-submission API (primary method) ...")
        api_counts = fetch_our_episodes_via_submission_api(submissions)
        for d, c in api_counts.items():
            episodes_by_date[d] = episodes_by_date.get(d, 0) + c
        print(f"  {sum(api_counts.values())} episode(s) across {len(api_counts)} day(s) "
              f"via submission API")

        if args.also_bulk_scan:
            sources.append("bulk_scan")
            since_date = args.since or earliest_relevant_date(submissions)
            state["since_date"] = since_date
            available, not_published = find_available_dataset_dates(since_date)
            state["days_scanned"] = available
            state["days_not_yet_published"] = not_published
            print(f"[--also-bulk-scan] episode datasets since {since_date}: "
                  f"{len(available)} available ({available}), {len(not_published)} not yet "
                  f"published ({not_published})")

            failed_days = []
            for d in available:
                out_dir = os.path.join(episodes_root, d)
                try:
                    count = download_and_extract_our_episodes(d, out_dir)
                except Exception as e:
                    print(f"FAILED to fetch/scan {d}: {e!r}", file=sys.stderr)
                    failed_days.append({"date": d, "error": repr(e)})
                    continue
                episodes_by_date[d] = episodes_by_date.get(d, 0) + count
                print(f"  {d}: {count} of our episodes (bulk scan)")
            state["days_failed"] = failed_days

        total_found = sum(episodes_by_date.values())
        state["episode_source"] = "+".join(sources)
        state["episodes_by_date"] = episodes_by_date
        state["our_episode_count"] = total_found
        state["our_episodes_dir"] = episodes_root if total_found else None
        print(f"\ntotal of our episodes found: {total_found} across "
              f"{len(episodes_by_date)} day(s) (source: {state['episode_source']})")
        if total_found == 0:
            print("0 is unexpected from the submission API unless our submissions genuinely "
                  "haven't been matched into any games yet -- worth investigating, not "
                  "assuming benign (see docs/submission_ladder_audit_2026-07-23.md).")

    with open(MEASURE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"\nwrote {MEASURE_STATE_PATH}")


if __name__ == "__main__":
    main()
