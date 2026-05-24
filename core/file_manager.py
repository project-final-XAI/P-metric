"""Centralized file path and I/O management for CROSS-XAI experiments.

Handles all file operations including heatmaps, results, and progress tracking.
Intelligently handles Model-Independent (no model) vs Model-Dependent paths.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional


class FileManager:
    """Centralized file and directory management."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.heatmap_dir = self.base_dir / "results" / "heatmaps"
        self.results_dir = self.base_dir / "results" / "evaluation"
        self.analysis_dir = self.base_dir / "results" / "analysis"

    # ==================== Helpers ====================
    @staticmethod
    def _sanitize_category(name: str) -> str:
        """Sanitize a category name for use in filenames."""
        safe = name.replace(" ", "_").replace(",", "").replace("/", "_").replace("\\", "_")
        return "".join(c for c in safe if c.isalnum() or c in "_-")[:30]

    def _build_heatmap_subpath(self, dataset: str, model: Optional[str], method: str, subfolder: str) -> Path:
        """Helper to build the directory path, nesting method under model if model is present."""
        if model is None:
            return self.heatmap_dir / dataset / method / subfolder

        return self.heatmap_dir / dataset / model / method / subfolder

    def _build_heatmap_filename(self, img_id: str, category_name: Optional[str], ext: str) -> str:
        """Helper to build the filename: uses {id}_{category}.{ext} if category exists, else {id}.{ext}."""
        if category_name:
            safe_category = self._sanitize_category(category_name)
            return f"{img_id}_{safe_category}.{ext}"
        return f"{img_id}.{ext}"

    # ==================== Heatmap Paths (Phase 1) ====================

    def get_heatmap_dir(self, dataset: str) -> Path:
        return self.heatmap_dir / dataset

    def get_sorted_heatmap_path(self, dataset: str, model: Optional[str], method: str, img_id: str, category_name: Optional[str] = None) -> Path:
        """Get path to sorted heatmap NPY file."""
        dir_path = self._build_heatmap_subpath(dataset, model, method, "sorted")
        filename = self._build_heatmap_filename(img_id, category_name, "npy")
        return dir_path / filename

    def get_regular_heatmap_path(self, dataset: str, model: Optional[str], method: str, img_id: str, category_name: Optional[str] = None) -> Path:
        """Get path to regular heatmap PNG file."""
        dir_path = self._build_heatmap_subpath(dataset, model, method, "regular")
        filename = self._build_heatmap_filename(img_id, category_name, "png")
        return dir_path / filename

    def scan_sorted_heatmaps(self, dataset: str, model: Optional[str], method: str) -> List[Path]:
        """Scan all sorted heatmap files for a model-method combination."""
        sorted_dir = self._build_heatmap_subpath(dataset, model, method, "sorted")
        if not sorted_dir.exists():
            return []
        return list(sorted_dir.glob("*.npy"))

    # ==================== Occluded Image Paths (Phase 2) ====================

    def get_occluded_dir(self, dataset: str) -> Path:
        return self.base_dir / "results" / "occluded" / dataset

    def get_occluded_image_path(
        self,
        dataset: str,
        model: Optional[str],
        strategy: str,
        method: str,
        level: int,
        img_id: str,
        category_name: Optional[str] = None
    ) -> Path:
        """Get path to occluded image."""
        filename = self._build_heatmap_filename(img_id, category_name, "png")

        if model is None:
            return self.get_occluded_dir(dataset) / method / strategy / str(level) / filename

        # CHANGED: Nested method within model folder for structured tracking
        return self.get_occluded_dir(dataset) / model / method / strategy / str(level) / filename

    def scan_occluded_images(
        self,
        dataset: str,
        model: Optional[str],
        strategy: str,
        method: str,
        level: int
    ) -> List[Path]:
        """Scan all occluded images for a specific combination."""
        if model is None:
            level_dir = self.get_occluded_dir(dataset) / method / strategy / str(level)
        else:
            # CHANGED: Updated directory look-up pattern to check nested structure
            level_dir = self.get_occluded_dir(dataset) / model / method / strategy / str(level)

        if not level_dir.exists():
            return []
        return list(level_dir.glob("*.png"))

    # ==================== Result File Paths ====================

    def get_result_dir(self, dataset: str) -> Path:
        return self.results_dir / dataset

    def get_result_file_path(
        self,
        dataset: str,
        gen_model: Optional[str],
        judge_model: str,
        method: str,
        strategy: str
    ) -> Path:
        """Get path to result CSV file."""
        # CHANGED: Splitting target folder into model / method layout instead of an underscore join
        if gen_model:
            return self.get_result_dir(dataset) / judge_model / gen_model / method / f"{strategy}.csv"
        return self.get_result_dir(dataset) / judge_model / method / f"{strategy}.csv"

    # ==================== I/O Operations ====================

    def ensure_dir_exists(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def save_csv(
        self,
        path: Path,
        data: List[List],
        header: Optional[List[str]] = None,
        append: bool = False
    ) -> None:
        if not data:
            return

        self.ensure_dir_exists(path.parent)
        mode = 'a' if append else 'w'
        write_header = header is not None and (not append or not path.exists())

        try:
            with open(path, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerows(data)
        except Exception as e:
            logging.error(f"Failed to save CSV to {path}: {e}")
            raise

    def load_csv(self, path: Path, skip_header: bool = True) -> List[List]:
        if not path.exists():
            return []

        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                if skip_header:
                    next(reader, None)
                return list(reader)
        except Exception as e:
            logging.error(f"Failed to load CSV from {path}: {e}")
            return []