"""
Centralized file path and I/O management for CROSS-XAI experiments.

Handles all file operations including heatmaps, results, and progress tracking.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional


class FileManager:
    """Centralized file and directory management."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize FileManager with base project directory.
        
        Args:
            base_dir: Root directory of the project
        """
        self.base_dir = Path(base_dir)
        self.heatmap_dir = self.base_dir / "results" / "heatmaps"
        self.results_dir = self.base_dir / "results" / "evaluation"
        self.analysis_dir = self.base_dir / "results" / "analysis"
    
    # ==================== Heatmap Paths ====================
    
    def get_heatmap_dir(self, dataset: str) -> Path:
        """Get directory for dataset heatmaps."""
        return self.heatmap_dir / dataset
    
    def get_heatmap_path(
        self,
        dataset: str,
        model: str,
        method: str,
        img_id: str,
        sorted: bool = False
    ) -> Path:
        """
        Get path to heatmap file.
        
        Args:
            dataset: Dataset name
            model: Model name
            method: Attribution method name
            img_id: Image identifier
            sorted: If True, returns path to sorted indices file
            
        Returns:
            Path to heatmap or sorted indices file
        """
        suffix = "_sorted" if sorted else ""
        filename = f"{model}-{method}-{img_id}{suffix}.npy"
        return self.get_heatmap_dir(dataset) / filename
    
    # ==================== Result File Paths ====================
    
    def get_result_dir(self, dataset: str) -> Path:
        """Get base results directory for dataset."""
        return self.results_dir / dataset
    
    def get_result_file_path(
        self,
        dataset: str,
        gen_model: str,
        judge_model: str,
        method: str,
        strategy: str
    ) -> Path:
        """
        Get path to result CSV file.
        
        Structure: results/evaluation/{dataset}/{gen_model}/{judge_model}/{method}/{strategy}.csv
        
        Args:
            dataset: Dataset name
            gen_model: Generating model name
            judge_model: Judging model name
            method: Attribution method name
            strategy: Fill strategy name
            
        Returns:
            Path to result CSV file
        """
        return (
            self.get_result_dir(dataset) /
            gen_model /
            judge_model /
            method /
            f"{strategy}.csv"
        )
    
    # ==================== Progress Tracking ====================
    
    def get_progress_file_path(self, dataset: str) -> Path:
        """
        Get path to progress tracking file.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Path to .progress.json file
        """
        return self.get_result_dir(dataset) / ".progress.json"
    
    # ==================== I/O Operations ====================
    
    def ensure_dir_exists(self, path: Path) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            path: Directory path to create
        """
        path.mkdir(parents=True, exist_ok=True)
    
    def save_csv(
        self,
        path: Path,
        data: List[List],
        header: Optional[List[str]] = None,
        append: bool = False
    ) -> None:
        """
        Save data to CSV file.
        
        Args:
            path: Path to CSV file
            data: List of rows to write
            header: Optional header row
            append: If True, append to existing file
        """
        if not data:
            return
        
        # Ensure directory exists
        self.ensure_dir_exists(path.parent)
        
        mode = 'a' if append else 'w'
        write_header = header is not None and (not append or not path.exists())
        
        try:
            with open(path, mode, newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerows(data)
        except Exception as e:
            logging.error(f"Failed to save CSV to {path}: {e}")
            raise
    
    def load_csv(self, path: Path, skip_header: bool = True) -> List[List]:
        """
        Load data from CSV file.
        
        Args:
            path: Path to CSV file
            skip_header: If True, skip first row
            
        Returns:
            List of rows from CSV
        """
        if not path.exists():
            return []
        
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                if skip_header:
                    next(reader, None)  # Skip header
                return list(reader)
        except Exception as e:
            logging.error(f"Failed to load CSV from {path}: {e}")
            return []
    
    # ==================== Directory Scanning ====================
    
    def scan_result_files(self, dataset: str) -> List[Path]:
        """
        Scan all result CSV files for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of paths to result CSV files
        """
        result_dir = self.get_result_dir(dataset)
        if not result_dir.exists():
            return []
        
        # Find all CSV files, excluding hidden files like .progress.json
        return [
            f for f in result_dir.rglob("*.csv")
            if not f.name.startswith(".")
        ]
    
    def parse_result_file_path(self, path: Path, dataset: str) -> dict:
        """
        Parse result file path to extract parameters.
        
        Args:
            path: Path to result file
            dataset: Dataset name
            
        Returns:
            Dict with keys: gen_model, judge_model, method, strategy
        """
        # Structure: results/evaluation/{dataset}/{gen_model}/{judge_model}/{method}/{strategy}.csv
        result_dir = self.get_result_dir(dataset)
        
        try:
            relative = path.relative_to(result_dir)
            parts = relative.parts
            
            return {
                'gen_model': parts[0],
                'judge_model': parts[1],
                'method': parts[2],
                'strategy': parts[3].replace('.csv', '')
            }
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse result file path {path}: {e}")
            return {}

