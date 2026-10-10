"""Vercel Python function entry point: every /api/* request is rewritten here."""

import sys
from pathlib import Path

# The evolora package lives in src/; make it importable whether or not the build
# installed the project itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolora.web.app import app  # noqa: E402

__all__ = ["app"]
