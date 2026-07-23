"""Stage 6 meta-mining (Workstream B): characterizes the ladder's actual deck meta from the
daily bulk episode dumps (kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD), something never
done before this cycle -- prior tooling (tools/measure.py --also-bulk-scan) only ever searched
these dumps for OUR OWN team's episodes. This scans EVERY episode in a dump.

Per-team archetype signature: since decklists aren't published, each team's archetype is
approximated per game as the top-3 most-frequently-seen non-basic-energy card ids across their
own hand + active + bench + discard, taken from the LAST available step of their own observation
stream (the most complete public+own-hand picture available without walking every step -- a
documented simplification, see docs/meta_report_<date>.md's methodology section: this can miss
cards that were drawn and later fully consumed/discarded-then-shuffled-back or that remain in
deck/face-down prize, but captures the great majority of what a team actually played).

Hardware rules (CLAUDE.md): one day's zip at a time, never held fully in RAM (each entry is
opened and parsed individually, discarded after its signature is extracted), deleted after
scanning. Resumable at day granularity via SCAN_STATE_PATH -- a day is only marked done (and its
zip deleted) after a full successful scan; an interrupted day is simply redone from scratch next
run (bounded worst case: one day's re-download).

Usage:
  .venv/Scripts/python tools/meta_miner.py --since 2026-07-17 [--until 2026-07-23] [--max-days N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import zipfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import kaggle_common  # noqa: E402
from kaggle_common import kaggle_cmd, SIMULATION_COMPETITION  # noqa: E402
from sdk_path import ensure_cg_importable  # noqa: E402

ensure_cg_importable()
from cg.api import CardType, all_card_data, to_observation_class  # noqa: E402
from obs import parse_state  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(RUNS_DIR, "meta_mining")
SIGNATURES_PATH = os.path.join(OUT_DIR, "team_game_signatures.jsonl")
SCAN_STATE_PATH = os.path.join(OUT_DIR, "scanned_dates.json")
DOWNLOAD_DIR = os.path.join(RUNS_DIR, "_episode_downloads")
DAILY_DATASET_PREFIX = "kaggle/pokemon-tcg-ai-battle-episodes-"

_CARD_TYPE = {c.cardId: c.cardType for c in all_card_data()}
_ARCHETYPE_TOP_N = 3


# ---------------------------------------------------------------------------
# Date-range / scan-state bookkeeping (resumable at day granularity)
# ---------------------------------------------------------------------------

def load_scan_state() -> dict:
    if not os.path.exists(SCAN_STATE_PATH):
        return {"scanned_dates": []}
    with open(SCAN_STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_scan_state(state: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SCAN_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def date_range(start_str: str, end_str: str) -> list[str]:
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    return [(start + datetime.timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)]


# ---------------------------------------------------------------------------
# Archetype signature extraction
# ---------------------------------------------------------------------------

def _card_ids_from_zone(zone) -> list[int]:
    ids = []
    for item in zone or []:
        cid = getattr(item, "card_id", None)
        if cid is not None:
            ids.append(cid)
    return ids


def archetype_signature(obs_dict: dict) -> list[int] | None:
    """Top-3 most-frequent non-basic-energy card ids seen in this seat's own hand/active/bench/
    discard, from this single observation snapshot. None if the observation has no current
    state (deck-selection phase / not yet started)."""
    obs = to_observation_class(obs_dict)
    if obs.current is None:
        return None
    game_state = parse_state(obs.current)
    you = game_state.you
    ids = []
    ids.extend(_card_ids_from_zone(you.hand))
    ids.extend(_card_ids_from_zone(you.bench))
    ids.extend(_card_ids_from_zone(you.discard))
    if you.active is not None:
        ids.append(you.active.card_id)
    counts = Counter(cid for cid in ids if _CARD_TYPE.get(cid) != CardType.BASIC_ENERGY)
    if not counts:
        return None
    return [cid for cid, _ in counts.most_common(_ARCHETYPE_TOP_N)]


def extract_team_game_signatures(episode: dict) -> list[dict]:
    """One record per seat: {team_name, outcome, archetype, episode_id}. Uses the LAST step
    where that seat's own observation has a non-null `current` (most complete state seen)."""
    team_names = episode.get("info", {}).get("TeamNames", [None, None])
    steps = episode.get("steps", [])
    statuses = episode.get("statuses") or []
    out = []
    for seat in (0, 1):
        best_obs_dict = None
        for step in reversed(steps):
            entry = step[seat] if seat < len(step) else None
            if entry is None:
                continue
            obs_dict = entry.get("observation")
            if obs_dict and obs_dict.get("current") is not None:
                best_obs_dict = obs_dict
                break
        if best_obs_dict is None:
            continue
        sig = archetype_signature(best_obs_dict)
        if sig is None:
            continue
        reward = None
        rewards = episode.get("rewards")
        if rewards and seat < len(rewards):
            reward = rewards[seat]
        out.append({
            "episode_id": episode.get("id"),
            "team_name": team_names[seat] if seat < len(team_names) else None,
            "seat": seat,
            "archetype": sig,
            "reward": reward,
        })
    return out


# ---------------------------------------------------------------------------
# Daily dump download + scan
# ---------------------------------------------------------------------------

def _dataset_exists(slug: str) -> bool:
    import subprocess
    probe_dir = os.path.join(RUNS_DIR, "_dataset_probe")
    os.makedirs(probe_dir, exist_ok=True)
    result = subprocess.run(kaggle_cmd("datasets", "metadata", slug, "-p", probe_dir),
                             capture_output=True, text=True)
    return result.returncode == 0


def scan_one_date(date_str: str) -> int:
    import subprocess
    slug = f"{DAILY_DATASET_PREFIX}{date_str}"
    if not _dataset_exists(slug):
        print(f"{date_str}: dataset not published yet, skipping")
        return 0

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, f"{slug.split('/')[-1]}.zip")
    if not os.path.exists(zip_path):
        print(f"{date_str}: downloading {slug} (can be 700MB+) ...")
        subprocess.run(kaggle_cmd("datasets", "download", "-d", slug, "-p", DOWNLOAD_DIR),
                        check=True)
    else:
        print(f"{date_str}: {zip_path} already downloaded, reusing.")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Write to a per-date temp file first, only appended into SIGNATURES_PATH once the day's
    # scan completes fully -- otherwise a day interrupted partway through (killed process,
    # etc.) and then RESTARTED from scratch (this function doesn't do within-day resume, only
    # day-granularity, per the module docstring) would duplicate the records it already wrote
    # before being killed, since SIGNATURES_PATH itself is append-only across days.
    tmp_path = os.path.join(OUT_DIR, f".tmp_{date_str}.jsonl")
    n_records = 0
    n_episodes = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf, open(tmp_path, "w", encoding="utf-8") as out_f:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"{date_str}: scanning {len(names)} episodes ...")
        for i, name in enumerate(names):
            try:
                with zf.open(name) as f:
                    episode = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue  # malformed/partial entry -- skip, don't fail the whole day
            records = extract_team_game_signatures(episode)
            for rec in records:
                rec["date"] = date_str
                out_f.write(json.dumps(rec) + "\n")
                n_records += 1
            n_episodes += 1
            if (i + 1) % 1000 == 0:
                out_f.flush()
                print(f"  ...{i + 1}/{len(names)} scanned, {n_records} team-game records so far")

    # full day succeeded -- commit the temp file into the real (append-only, cross-day) output,
    # then clean up both the temp file and the day's zip (don't accumulate 700MB+/day).
    with open(tmp_path, encoding="utf-8") as tmp_f, \
            open(SIGNATURES_PATH, "a", encoding="utf-8") as out_f:
        shutil.copyfileobj(tmp_f, out_f)
    os.remove(tmp_path)
    os.remove(zip_path)
    print(f"{date_str}: {n_episodes} episodes scanned, {n_records} team-game signatures written")
    return n_episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="YYYY-MM-DD, first date to scan")
    ap.add_argument("--until", default=None,
                     help="YYYY-MM-DD, last date to scan (default: today)")
    ap.add_argument("--max-days", type=int, default=None,
                     help="cap how many NEW (not-yet-scanned) days to process this run")
    args = ap.parse_args()

    until = args.until or datetime.date.today().isoformat()
    candidates = date_range(args.since, until)

    state = load_scan_state()
    scanned = set(state["scanned_dates"])
    todo = [d for d in candidates if d not in scanned]
    if args.max_days is not None:
        todo = todo[:args.max_days]
    print(f"{len(candidates)} candidate dates, {len(scanned & set(candidates))} already scanned, "
          f"{len(todo)} to process this run")

    for date_str in todo:
        n_episodes = scan_one_date(date_str)
        if n_episodes > 0:
            scanned.add(date_str)
            state["scanned_dates"] = sorted(scanned)
            save_scan_state(state)

    print(f"\nwrote/updated {SIGNATURES_PATH}")
    print(f"scan state: {SCAN_STATE_PATH}")


if __name__ == "__main__":
    main()
