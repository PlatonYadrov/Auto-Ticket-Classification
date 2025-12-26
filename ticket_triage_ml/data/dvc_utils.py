"""DVC utilities for data versioning and retrieval."""

import subprocess
from pathlib import Path
from typing import List, Optional

from loguru import logger


def dvc_pull(targets: Optional[List[str]] = None, force: bool = False) -> bool:
    """Pull data from DVC remote storage.

    Args:
        targets: Optional list of specific DVC files or directories to pull.
        force: Whether to force overwrite existing files.

    Returns:
        True if pull was successful, False otherwise.
    """
    cmd = ["dvc", "pull"]

    if force:
        cmd.append("--force")

    if targets:
        cmd.extend(targets)

    logger.info(f"Running DVC pull: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"DVC pull stdout: {result.stdout}")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(f"DVC pull failed: {exc.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("DVC is not installed or not in PATH")
        return False


def dvc_add(file_path: Path) -> bool:
    """Add a file to DVC tracking.

    Args:
        file_path: Path to the file to track with DVC.

    Returns:
        True if successful, False otherwise.
    """
    cmd = ["dvc", "add", str(file_path)]
    logger.info(f"Running DVC add: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(f"DVC add failed: {exc.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("DVC is not installed or not in PATH")
        return False


def dvc_push() -> bool:
    """Push data to DVC remote storage.

    Returns:
        True if push was successful, False otherwise.
    """
    cmd = ["dvc", "push"]
    logger.info("Running DVC push")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(f"DVC push failed: {exc.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("DVC is not installed or not in PATH")
        return False


def check_dvc_tracked(file_path: Path) -> bool:
    """Check if a file is tracked by DVC.

    Args:
        file_path: Path to check.

    Returns:
        True if file is DVC-tracked, False otherwise.
    """
    dvc_file = Path(str(file_path) + ".dvc")
    return dvc_file.exists()


def ensure_dvc_initialized(project_root: Path) -> bool:
    """Ensure DVC is initialized in the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        True if DVC is initialized, False otherwise.
    """
    dvc_dir = project_root / ".dvc"

    if dvc_dir.exists():
        return True

    logger.info("Initializing DVC in project")
    cmd = ["dvc", "init"]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=project_root)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"Failed to initialize DVC: {exc}")
        return False
