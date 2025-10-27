"""
Fast progress tracking for experiment resumption.

Uses a lightweight JSON index instead of scanning thousands of CSV files.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Set, Tuple, List


class ProgressTracker:
    """
    Tracks completed work for fast experiment resumption.
    
    Maintains a single JSON file with completed work items instead of
    scanning thousands of CSV files, reducing resume time from ~60s to <0.1s.
    """
    
    def __init__(self, file_manager, dataset: str, auto_save_interval: int = 100, auto_save_time: int = 300):
        """
        Initialize progress tracker.
        
        Args:
            file_manager: FileManager instance
            dataset: Dataset name
            auto_save_interval: Save every N completed items (0 to disable)
            auto_save_time: Save every N seconds (0 to disable)
        """
        self.file_manager = file_manager
        self.dataset = dataset
        self.progress_file = file_manager.get_progress_file_path(dataset)
        self.completed: Set[Tuple] = set()
        self._dirty = False
        self.auto_save_interval = auto_save_interval
        self.auto_save_time = auto_save_time
        self._items_since_save = 0
        self._last_save_time = time.time()
        self.load()
    
    def load(self) -> None:
        """Load progress from JSON file."""
        if not self.progress_file.exists():
            logging.info(f"No existing progress file for {self.dataset}")
            return
        
        # First validate the file
        if not self.validate_progress_file():
            logging.warning(f"Progress file for {self.dataset} is corrupted. Starting fresh.")
            self.completed = set()
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                # Convert lists back to tuples for set membership
                self.completed = {tuple(item) for item in data.get('completed', [])}
            logging.info(f"Loaded {len(self.completed)} completed items for {self.dataset}")
        except Exception as e:
            logging.warning(f"Failed to load progress file: {e}. Starting fresh.")
            self.completed = set()
    
    def is_completed(
        self,
        gen_model: str,
        method: str,
        img_id: str,
        judge_model: str,
        strategy: str,
        occlusion_level: float
    ) -> bool:
        """
        Check if a work item is already completed.
        
        Args:
            gen_model: Generating model name
            method: Attribution method name
            img_id: Image identifier
            judge_model: Judging model name
            strategy: Fill strategy name
            occlusion_level: Occlusion percentage
            
        Returns:
            True if work item is completed
        """
        key = (gen_model, method, img_id, judge_model, strategy, float(occlusion_level))
        return key in self.completed
    
    def mark_completed(
        self,
        gen_model: str,
        method: str,
        img_id: str,
        judge_model: str,
        strategy: str,
        occlusion_level: float
    ) -> None:
        """
        Mark a work item as completed.
        
        Args:
            gen_model: Generating model name
            method: Attribution method name
            img_id: Image identifier
            judge_model: Judging model name
            strategy: Fill strategy name
            occlusion_level: Occlusion percentage
        """
        key = (gen_model, method, img_id, judge_model, strategy, float(occlusion_level))
        self.completed.add(key)
        self._dirty = True
        self._items_since_save += 1
        
        # Auto-save if interval or time reached
        current_time = time.time()
        should_save = (
            (self.auto_save_interval > 0 and self._items_since_save >= self.auto_save_interval) or
            (self.auto_save_time > 0 and current_time - self._last_save_time >= self.auto_save_time)
        )
        
        if should_save:
            self.save()
            self._items_since_save = 0
            self._last_save_time = current_time
    
    def mark_batch_completed(self, items: List[Tuple]) -> None:
        """
        Mark multiple work items as completed.
        
        Args:
            items: List of tuples (gen_model, method, img_id, judge_model, strategy, level)
        """
        for item in items:
            self.completed.add(tuple(item))
        self._dirty = True
        self._items_since_save += len(items)
        
        # Auto-save if interval or time reached
        current_time = time.time()
        should_save = (
            (self.auto_save_interval > 0 and self._items_since_save >= self.auto_save_interval) or
            (self.auto_save_time > 0 and current_time - self._last_save_time >= self.auto_save_time)
        )
        
        if should_save:
            self.save()
            self._items_since_save = 0
            self._last_save_time = current_time
    
    def save(self) -> None:
        """Save progress to JSON file (only if modified)."""
        if not self._dirty:
            return
        
        try:
            # Ensure directory exists
            self.file_manager.ensure_dir_exists(self.progress_file.parent)
            
            # Convert tuples to lists for JSON serialization
            data = {
                'dataset': self.dataset,
                'completed': [list(item) for item in self.completed]
            }
            
            # Use atomic write to prevent corruption
            # Write to temporary file first, then rename
            temp_file = self.progress_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(data, f)
                # Force flush to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename (this is atomic on most filesystems)
            temp_file.replace(self.progress_file)
            
            self._dirty = False
            logging.debug(f"Saved progress: {len(self.completed)} items")
        except Exception as e:
            logging.error(f"Failed to save progress file: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
    
    def get_completed_count(self) -> int:
        """Get total number of completed work items."""
        return len(self.completed)
    
    def validate_progress_file(self) -> bool:
        """
        Validate that the progress file is not corrupted.
        
        Returns:
            True if file is valid, False if corrupted
        """
        if not self.progress_file.exists():
            return True  # No file is valid (empty state)
        
        try:
            with open(self.progress_file, 'r') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Progress file is corrupted: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save on exit."""
        self.save()

