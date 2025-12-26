"""Git utilities for version tracking."""

import subprocess
from typing import Optional

from loguru import logger

from ticket_triage_ml.utils.paths import get_project_root


def get_git_commit_id(short: bool = True) -> Optional[str]:
    """Get the current git commit ID.

    Args:
        short: If True, return short (7-character) hash.

    Returns:
        Git commit hash string, or None if not in a git repo.
    """
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root(),
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError:
        logger.debug("Not in a git repository or git command failed")
        return None
    except FileNotFoundError:
        logger.debug("Git is not installed")
        return None


def get_git_branch() -> Optional[str]:
    """Get the current git branch name.

    Returns:
        Branch name string, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root(),
        )

        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> Optional[bool]:
    """Check if there are uncommitted changes.

    Returns:
        True if there are uncommitted changes, False if clean,
        None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root(),
        )

        return bool(result.stdout.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
