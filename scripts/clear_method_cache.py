from __future__ import annotations

import shutil
from pathlib import Path

# ==========================================
# CONFIGURATION - BAKE YOUR PARAMS HERE
# ==========================================
DATASET = "imagenet"
METHODS_TO_CLEAR = [
    "U2Net-Saliency",
    "u2net_dino_fusion",
]
INCLUDE_EVALUATION = True  # Set to True to delete evaluation CSVs
DRY_RUN = False  # Set to False to actually delete files


# ==========================================

def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _collect_dirs_named(root: Path, name: str) -> list[Path]:
    """Collect all directories under root whose final path component == name."""
    if not root.exists():
        return []

    found: list[Path] = []
    for p in root.rglob(name):
        if p.is_dir():
            found.append(p)
    return found


def _dedupe_delete_targets(paths: list[Path]) -> list[Path]:
    """
    Keep only the highest-level directory to avoid redundant deletes.
    """
    paths_sorted = sorted(paths, key=lambda p: len(p.parts))
    keep: list[Path] = []
    for p in paths_sorted:
        if any(p2 != p and p.is_relative_to(p2) for p2 in keep):
            continue
        keep.append(p)
    return keep


def main() -> None:
    # This file lives in /scripts, so repo root is one level up.
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results"

    all_targets: list[Path] = []

    for method in METHODS_TO_CLEAR:
        # Collect intermediate caches
        all_targets += _collect_dirs_named(results_root / "heatmaps" / DATASET, method)
        all_targets += _collect_dirs_named(results_root / "occluded" / DATASET, method)

        # Collect evaluation CSVs
        if INCLUDE_EVALUATION:
            all_targets += _collect_dirs_named(results_root / "evaluation" / DATASET, method)

    # Safety and Cleanup
    final_targets = [t for t in all_targets if _is_within(t, results_root) and t.exists()]
    final_targets = _dedupe_delete_targets(sorted(set(final_targets)))

    print(f"--- Bulk Cache Deletion Config ---")
    print(f"Dataset:      {DATASET}")
    print(f"Methods:      {', '.join(METHODS_TO_CLEAR)}")
    print(f"Include Eval: {INCLUDE_EVALUATION}")
    print(f"Dry Run:      {DRY_RUN}")
    print(f"Total Targets found: {len(final_targets)}")

    if not final_targets:
        print("\nNo matching directories found. Nothing to do.")
        return

    print("\nDirectories targeted for removal:")
    for t in final_targets:
        print(f"  - {t}")

    if DRY_RUN:
        print("\n[DRY RUN] No deletion performed.")
        return

    for t in final_targets:
        print(f"Deleting: {t}")
        try:
            shutil.rmtree(t)
        except Exception as e:
            print(f"  [ERROR] Failed to delete {t}: {e}")

    print("\nBulk deletion complete.")


if __name__ == "__main__":
    main()