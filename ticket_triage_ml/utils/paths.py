"""Path utilities for project file management."""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get the project root directory.

    Traverses up from the current file to find the project root,
    identified by the presence of pyproject.toml.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be found.
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Could not find project root (no pyproject.toml found in parent directories)"
    )


def resolve_path(path: Union[str, Path], relative_to: Union[str, Path, None] = None) -> Path:
    """Resolve a path, optionally relative to a base directory.

    Args:
        path: Path to resolve (can be relative or absolute).
        relative_to: Optional base directory for relative paths.
            If None, uses project root.

    Returns:
        Absolute Path object.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if relative_to is None:
        relative_to = get_project_root()
    else:
        relative_to = Path(relative_to)

    return (relative_to / path).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
