"""Assembles submission/ from src/ (source of truth) + the cg SDK download + our deck,
then tars it into submission.tar.gz per the competition's required layout: main.py at
top level, alongside deck.csv (tar -czvf submission.tar.gz * from inside submission/).

submission/ and submission.tar.gz are both gitignored build output -- rerun this script
any time src/baseline.py, src/obs.py, or decks/baseline_deck.csv changes, rather than
hand-editing files under submission/.
"""

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
TARBALL_PATH = os.path.join(REPO_ROOT, "submission.tar.gz")

MAIN_PY_CONTENTS = "from baseline import agent  # noqa: F401\n"

# Limits from CLAUDE.md / the competition rules.
MAX_SUBMISSION_MIB = 197.7


def build():
    if not os.path.isdir(SDK_CG_DIR):
        sys.exit(f"cg SDK not found at {SDK_CG_DIR} -- run Phase 0's Kaggle download first.")
    if not os.path.exists(DECK_PATH):
        sys.exit(f"deck not found at {DECK_PATH}.")

    if os.path.isdir(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)
    os.makedirs(os.path.join(SUBMISSION_DIR, "cg"))

    for module in ("sdk_path.py", "obs.py", "baseline.py"):
        shutil.copy2(os.path.join(SRC_DIR, module), os.path.join(SUBMISSION_DIR, module))

    for name in os.listdir(SDK_CG_DIR):
        src_path = os.path.join(SDK_CG_DIR, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(SUBMISSION_DIR, "cg", name))

    shutil.copy2(DECK_PATH, os.path.join(SUBMISSION_DIR, "deck.csv"))

    with open(os.path.join(SUBMISSION_DIR, "main.py"), "w") as f:
        f.write(MAIN_PY_CONTENTS)

    if os.path.exists(TARBALL_PATH):
        os.remove(TARBALL_PATH)
    entries = sorted(os.listdir(SUBMISSION_DIR))
    subprocess.run(["tar", "-czf", TARBALL_PATH, *entries], cwd=SUBMISSION_DIR, check=True)

    size_mib = os.path.getsize(TARBALL_PATH) / 1024 / 1024
    print(f"Built {TARBALL_PATH}: {size_mib:.3f} MiB (limit {MAX_SUBMISSION_MIB} MiB)")
    if size_mib > MAX_SUBMISSION_MIB:
        sys.exit("submission.tar.gz exceeds the size limit!")


if __name__ == "__main__":
    build()
