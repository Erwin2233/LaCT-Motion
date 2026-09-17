"""Paths to resources stored inside this checkout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def project_path(*parts):
    """Resolve a repository-relative resource independently of the caller's cwd."""
    return str(ROOT.joinpath(*parts))
