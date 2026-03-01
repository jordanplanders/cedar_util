from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional


def ensure_local_file(local_path: Path, master_path: Path) -> Path:
    """Ensure local_path exists by copying from master_path when available."""
    local_path = Path(local_path)
    master_path = Path(master_path)
    if local_path.exists():
        return local_path
    if master_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(master_path, local_path)
    return local_path


def ensure_master_copy(local_path: Path, master_path: Path) -> Path:
    """Ensure master_path exists by copying from local_path when available."""
    local_path = Path(local_path)
    master_path = Path(master_path)
    if master_path.exists():
        return master_path
    if local_path.exists():
        master_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, master_path)
    return master_path


def ensure_both_locations(local_path: Path, master_path: Path) -> tuple[Path, Path]:
    """Ensure both local and master copies exist, preferring local as source."""
    local_path = ensure_local_file(local_path, master_path)
    master_path = ensure_master_copy(local_path, master_path)
    return local_path, master_path

