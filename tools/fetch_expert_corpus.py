"""Master-study Workstream A2: re-scans an already-probed daily bulk dump (same one
tools/meta_miner.py scanned) and saves the FULL episode JSON -- not just the archetype
signature -- for every episode involving one of the expert/control teams identified by
tools/identify_expert_teams.py.

ID note (important, found the hard way): the bulk dump's episode.get("id") field is a UUID
(e.g. "002af5de-85d3-11f1-889a-0242ac130202"), NOT the numeric id `kaggle competitions replay`
expects (confirmed via docs/submission_ladder_audit_2026-07-23.md's own example,
"87362960.json"). This script never tries to cross-reference that UUID against anything --
it does its OWN independent scan of the zip (same team-name-membership check
tools/meta_miner.py already does) and keys saved files by the ZIP MEMBER'S OWN numeric
filename, exactly matching the runs/our_episodes/<date>/<episode_id>.json convention already
used elsewhere in this repo.

Hardware discipline: one zip at a time, stream-parsed, deleted after a full successful scan;
resumable at the (date) granularity, matching tools/meta_miner.py's own resumability model --
an interrupted run's already-written per-episode JSON files simply get skipped (existence
check) on retry, no separate temp-file dance needed since each output file IS the atomic unit
here (unlike meta_miner.py's single growing jsonl).

Usage:
  .venv/Scripts/python tools/fetch_expert_corpus.py --date 2026-07-22
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kaggle_common import kaggle_cmd  # noqa: E402
import meta_miner  # noqa: E402  (reuses _dataset_exists/DOWNLOAD_DIR/DAILY_DATASET_PREFIX)

EXPERT_TEAMS_PATH = os.path.join(ROOT, "runs", "expert_corpus", "expert_teams.json")
CORPUS_OUT_DIR = os.path.join(ROOT, "runs", "expert_corpus")


def load_team_names() -> tuple[set[str], set[str]]:
    with open(EXPERT_TEAMS_PATH, encoding="utf-8") as f:
        d = json.load(f)
    expert = {m["team_name"] for m in d["expert_teams"]}
    control = {m["team_name"] for m in d["control_teams"]}
    return expert, control


def fetch_for_date(date_str: str, wanted_teams: set[str]) -> dict:
    slug = f"{meta_miner.DAILY_DATASET_PREFIX}{date_str}"
    if not meta_miner._dataset_exists(slug):
        print(f"{date_str}: dataset not published/available, skipping")
        return {"episodes_written": 0, "episodes_skipped_existing": 0}

    os.makedirs(meta_miner.DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(meta_miner.DOWNLOAD_DIR, f"{slug.split('/')[-1]}.zip")
    if not os.path.exists(zip_path):
        print(f"{date_str}: downloading {slug} (can be 700MB+) ...")
        subprocess.run(kaggle_cmd("datasets", "download", "-d", slug, "-p",
                                   meta_miner.DOWNLOAD_DIR), check=True)
    else:
        print(f"{date_str}: {zip_path} already downloaded, reusing.")

    out_dir = os.path.join(CORPUS_OUT_DIR, date_str)
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    skipped_existing = 0
    matched_episodes = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"{date_str}: scanning {len(names)} episodes for {len(wanted_teams)} "
              f"expert/control teams ...")
        for i, name in enumerate(names):
            numeric_id = os.path.splitext(os.path.basename(name))[0]
            out_path = os.path.join(out_dir, f"{numeric_id}.json")
            if os.path.exists(out_path):
                skipped_existing += 1
                continue
            try:
                with zf.open(name) as f:
                    raw_bytes = f.read()
                episode = json.loads(raw_bytes)
            except (json.JSONDecodeError, KeyError):
                continue
            team_names = episode.get("info", {}).get("TeamNames", [])
            if not any(t in wanted_teams for t in team_names):
                continue
            matched_episodes += 1
            with open(out_path, "wb") as out_f:
                out_f.write(raw_bytes)
            written += 1
            if (i + 1) % 1000 == 0:
                print(f"  ...{i + 1}/{len(names)} scanned, {matched_episodes} matched so far")

    os.remove(zip_path)
    print(f"{date_str}: {matched_episodes} matching episodes found, {written} written, "
          f"{skipped_existing} already present from a prior run")
    return {"episodes_written": written, "episodes_skipped_existing": skipped_existing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    expert, control = load_team_names()
    wanted = expert | control
    print(f"{len(expert)} expert teams, {len(control)} control teams, "
          f"{len(wanted)} total team names to match")

    result = fetch_for_date(args.date, wanted)

    out_dir = os.path.join(CORPUS_OUT_DIR, args.date)
    n_files = len([f for f in os.listdir(out_dir) if f.endswith(".json")]) \
        if os.path.isdir(out_dir) else 0
    print(f"\ncorpus directory {out_dir}: {n_files} episode files total")


if __name__ == "__main__":
    main()
