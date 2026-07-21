"""Locates the cg SDK for local development.

The cg package lives in the gitignored data/ tree (pulled down by the Phase 0 Kaggle
download) everywhere except inside a packaged submission, where it sits directly next to
main.py and needs no path juggling. This helper is for local dev/test scripts only.
"""

import os
import sys

_DEFAULT_SDK_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "pokemon-tcg-ai-battle",
    "sample_submission", "sample_submission",
))


def ensure_cg_importable() -> None:
    """Make `import cg` succeed: use it if already importable, else fall back to the
    local Phase 0 SDK download (override the path with the CG_SDK_PATH env var)."""
    try:
        import cg  # noqa: F401
        return
    except ImportError:
        pass

    sdk_dir = os.environ.get("CG_SDK_PATH", _DEFAULT_SDK_DIR)
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    import cg  # noqa: F401
