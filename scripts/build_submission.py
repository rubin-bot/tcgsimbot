"""Assembles submission/ from src/ (source of truth) + the cg SDK download + our deck,
then tars it into a submission tarball per the competition's required layout: main.py at
top level, alongside deck.csv (tar -czvf submission.tar.gz * from inside submission/).

Three modes:
  --mode baseline: the rule-based agent only. Output: submission.tar.gz.
  --mode search_scorer (default, v1): agents/search_scorer.py's lookahead+evaluate() agent,
    piloting decks/crustle_wall_deck.csv, falling back to baseline internally on any
    exception (see agents/search_scorer.py's own 3-tier fallback). Torch-free, no pip
    dependencies at all -- see CLAUDE.md's Stage 5 notes. Output:
    submission_search_scorer.tar.gz. Pass --weights PATH to bake in a specific tuned/fixed
    WEIGHTS dict (JSON, agents/search_scorer.py's shape) instead of the module default.
  --mode net: the deprecated trained checkpoint (torch-free NumPy forward + determinized
    MCTS), falling back to the baseline agent on any runtime exception. Output:
    submission_net.tar.gz. Needs torch at BUILD time (to export the checkpoint) but the
    bundle itself stays torch-free.

submission/ and submission*.tar.gz are both gitignored build output -- rerun this script
any time the bundled src/agents modules or the deck change, rather than hand-editing files
under submission/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
AGENTS_DIR = os.path.join(REPO_ROOT, "agents")
SDK_CG_DIR = os.path.join(
    REPO_ROOT, "data", "pokemon-tcg-ai-battle", "sample_submission", "sample_submission", "cg")
DEFAULT_DECK_PATH = os.path.join(REPO_ROOT, "decks", "baseline_deck.csv")
DEFAULT_SEARCH_SCORER_DECK_PATH = os.path.join(REPO_ROOT, "decks", "crustle_wall_deck.csv")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "runs", "run2", "best.pt")
NET_SIMS = 32  # see docs/strategy_snapshot.md: no documented per-move time limit found,
               # chosen as a safety margin (half the current self-play sims setting).

BASELINE_MAIN_PY = "from baseline import agent  # noqa: F401\n"
SEARCH_SCORER_MAIN_PY = "from search_scorer import agent  # noqa: F401\n"

NET_MAIN_PY = f"""from baseline import agent as _baseline_agent, read_deck_csv
from sdk_path import ensure_cg_importable

ensure_cg_importable()

_DECK = read_deck_csv()
_SIMS = {NET_SIMS}

try:
    from net_numpy import NumpyPVNet
    from mcts import search
    from cg.api import all_card_data
    _net = NumpyPVNet.load("model.npz")
    # Real opponents run their own deck, not ours -- sample_determinization defaults the
    # opponent's hidden cards to OUR deck list when no opp_deck_list is given, which is only
    # correct for a mirror match. Live play is never a mirror match, so seed the hidden-world
    # guess from the full card pool instead (a diverse, non-mirror prior) rather than silently
    # assuming they're playing our deck. Basic Energy (ids 1-8) is over-represented in real
    # decks relative to its 8-in-~1100 share of the raw card pool, so weight it up a bit rather
    # than leaving it diluted to near-zero.
    _BASIC_ENERGY_IDS = list(range(1, 9))
    _OPP_POOL = [c.cardId for c in all_card_data()] + _BASIC_ENERGY_IDS * 15
except Exception:
    _net = None


def agent(obs_dict: dict) -> list[int]:
    if _net is not None:
        try:
            out = search(obs_dict, _net, _DECK, sims=_SIMS, temperature=0.0, add_noise=False,
                         opp_deck_list=_OPP_POOL)
            if out is None:
                return list(_DECK)
            _, _, index_list, _ = out
            return index_list
        except Exception:
            pass  # engine/search fault on this decision -- fall back rather than crash the match
    return _baseline_agent(obs_dict)
"""

# Limits from CLAUDE.md / the competition rules.
MAX_SUBMISSION_MIB = 197.7


def build(mode: str, checkpoint: str, deck_path: str, weights_path: str | None = None):
    if not os.path.isdir(SDK_CG_DIR):
        sys.exit(f"cg SDK not found at {SDK_CG_DIR} -- run Phase 0's Kaggle download first.")
    if not os.path.exists(deck_path):
        sys.exit(f"deck not found at {deck_path}.")

    if os.path.isdir(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(os.path.join(SUBMISSION_DIR, "cg"))

    modules = ["sdk_path.py", "obs.py", "baseline.py"]
    if mode == "net":
        modules += ["encode.py", "determinize.py", "mcts.py", "net_numpy.py"]
    elif mode == "search_scorer":
        modules += ["determinize.py"]
    for module in modules:
        shutil.copy2(os.path.join(SRC_DIR, module), os.path.join(SUBMISSION_DIR, module))
    if mode == "search_scorer":
        shutil.copy2(os.path.join(AGENTS_DIR, "search_scorer.py"),
                      os.path.join(SUBMISSION_DIR, "search_scorer.py"))
        if weights_path:
            with open(weights_path, encoding="utf-8") as f:
                override = json.load(f)
            ss_path = os.path.join(SUBMISSION_DIR, "search_scorer.py")
            with open(ss_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n# Baked in at package time from {os.path.basename(weights_path)}\n"
                        f"WEIGHTS.update({json.dumps(override)})\n")

    for name in os.listdir(SDK_CG_DIR):
        src_path = os.path.join(SDK_CG_DIR, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(SUBMISSION_DIR, "cg", name))

    shutil.copy2(deck_path, os.path.join(SUBMISSION_DIR, "deck.csv"))

    if mode == "net":
        if not os.path.exists(checkpoint):
            sys.exit(f"checkpoint not found at {checkpoint}.")
        sys.path.insert(0, SRC_DIR)
        from net import PVNet
        PVNet.load(checkpoint).export_numpy(os.path.join(SUBMISSION_DIR, "model.npz"))
        main_py = NET_MAIN_PY
        tarball_path = os.path.join(REPO_ROOT, "submission_net.tar.gz")
    elif mode == "search_scorer":
        main_py = SEARCH_SCORER_MAIN_PY
        tarball_path = os.path.join(REPO_ROOT, "submission_search_scorer.tar.gz")
    else:
        main_py = BASELINE_MAIN_PY
        tarball_path = os.path.join(REPO_ROOT, "submission.tar.gz")

    with open(os.path.join(SUBMISSION_DIR, "main.py"), "w") as f:
        f.write(main_py)

    if os.path.exists(tarball_path):
        os.remove(tarball_path)
    entries = sorted(os.listdir(SUBMISSION_DIR))
    subprocess.run(["tar", "-czf", tarball_path, *entries], cwd=SUBMISSION_DIR, check=True)

    size_mib = os.path.getsize(tarball_path) / 1024 / 1024
    print(f"Built {tarball_path}: {size_mib:.3f} MiB (limit {MAX_SUBMISSION_MIB} MiB)")
    if size_mib > MAX_SUBMISSION_MIB:
        sys.exit("submission tarball exceeds the size limit!")
    return tarball_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "search_scorer", "net"],
                     default="search_scorer")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--deck", default=None,
                     help="defaults to decks/crustle_wall_deck.csv for search_scorer, "
                          "decks/baseline_deck.csv otherwise.")
    ap.add_argument("--weights", default=None,
                     help="search_scorer mode only: JSON file of a WEIGHTS-shaped dict to "
                          "bake in via WEIGHTS.update(...); omit to ship the module default.")
    args = ap.parse_args()
    deck = args.deck
    if deck is None:
        deck = (DEFAULT_SEARCH_SCORER_DECK_PATH if args.mode == "search_scorer"
                else DEFAULT_DECK_PATH)
    build(args.mode, args.checkpoint, deck, args.weights)
