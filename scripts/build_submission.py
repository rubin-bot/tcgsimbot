"""Assembles submission/ from src/ (source of truth) + the cg SDK download + our deck,
then tars it into a submission tarball per the competition's required layout: main.py at
top level, alongside deck.csv (tar -czvf submission.tar.gz * from inside submission/).

Two modes:
  --mode baseline (default): the rule-based agent only. Output: submission.tar.gz.
  --mode net: the trained checkpoint (torch-free NumPy forward + determinized MCTS),
    falling back to the baseline agent on any runtime exception. Output:
    submission_net.tar.gz. Needs torch at BUILD time (to export the checkpoint) but the
    bundle itself stays torch-free.

submission/ and submission*.tar.gz are both gitignored build output -- rerun this script
any time the bundled src/ modules or decks/baseline_deck.csv change, rather than
hand-editing files under submission/.
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SDK_CG_DIR = os.path.join(
    REPO_ROOT, "data", "pokemon-tcg-ai-battle", "sample_submission", "sample_submission", "cg")
DECK_PATH = os.path.join(REPO_ROOT, "decks", "baseline_deck.csv")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "runs", "run2", "best.pt")
NET_SIMS = 32  # see docs/strategy_snapshot.md: no documented per-move time limit found,
               # chosen as a safety margin (half the current self-play sims setting).

BASELINE_MAIN_PY = "from baseline import agent  # noqa: F401\n"

NET_MAIN_PY = f"""from baseline import agent as _baseline_agent, read_deck_csv

_DECK = read_deck_csv()
_SIMS = {NET_SIMS}

try:
    from net_numpy import NumpyPVNet
    from mcts import search
    _net = NumpyPVNet.load("model.npz")
except Exception:
    _net = None


def agent(obs_dict: dict) -> list[int]:
    if _net is not None:
        try:
            out = search(obs_dict, _net, _DECK, sims=_SIMS, temperature=0.0, add_noise=False)
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


def build(mode: str, checkpoint: str):
    if not os.path.isdir(SDK_CG_DIR):
        sys.exit(f"cg SDK not found at {SDK_CG_DIR} -- run Phase 0's Kaggle download first.")
    if not os.path.exists(DECK_PATH):
        sys.exit(f"deck not found at {DECK_PATH}.")

    if os.path.isdir(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(os.path.join(SUBMISSION_DIR, "cg"))

    modules = ["sdk_path.py", "obs.py", "baseline.py"]
    if mode == "net":
        modules += ["encode.py", "determinize.py", "mcts.py", "net_numpy.py"]
    for module in modules:
        shutil.copy2(os.path.join(SRC_DIR, module), os.path.join(SUBMISSION_DIR, module))

    for name in os.listdir(SDK_CG_DIR):
        src_path = os.path.join(SDK_CG_DIR, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(SUBMISSION_DIR, "cg", name))

    shutil.copy2(DECK_PATH, os.path.join(SUBMISSION_DIR, "deck.csv"))

    if mode == "net":
        if not os.path.exists(checkpoint):
            sys.exit(f"checkpoint not found at {checkpoint}.")
        sys.path.insert(0, SRC_DIR)
        from net import PVNet
        PVNet.load(checkpoint).export_numpy(os.path.join(SUBMISSION_DIR, "model.npz"))
        main_py = NET_MAIN_PY
        tarball_path = os.path.join(REPO_ROOT, "submission_net.tar.gz")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "net"], default="baseline")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()
    build(args.mode, args.checkpoint)
