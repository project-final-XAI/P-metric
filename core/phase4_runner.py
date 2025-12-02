"""
Phase 4: Analysis and Visualization Runner.

Loads Phase 3 results, calculates metrics (AUC, DROP), and generates visualization plots.
"""

import logging
import pandas as pd

from core.file_manager import FileManager
from evaluation.metrics import calculate_auc, calculate_drop
from visualization.plotter import plot_accuracy_degradation_curves, plot_fill_strategy_comparison


class Phase4Runner:
    """Handles Phase 4: Analysis and visualization of evaluation results."""
    
    def __init__(self, config, file_manager: FileManager):
        """
        Initialize Phase 4 runner.
        
        Args:
            config: Configuration object
            file_manager: File path manager
        """
        self.config = config
        self.file_manager = file_manager
    
    def run(self):
        """Run analysis and visualization on Phase 3 results."""
        try:
            if not self.file_manager.results_dir.exists():
                logging.error(f"Results directory does not exist: {self.file_manager.results_dir}")
                return
            
            datasets = [
                d.name for d in self.file_manager.results_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            
            if not datasets:
                logging.error("No result datasets found")
                return
            
            logging.info(f"Starting Phase 4 - Analyzing: {', '.join(datasets)}")
            
            all_data = self._load_all_results(datasets)
            
            if not all_data:
                logging.warning("No results data found")
                return
            
            df = pd.DataFrame(all_data)
            agg_df = self._calculate_aggregated_accuracy(df)
            metrics_df = self._calculate_metrics(agg_df)
            
            self._save_results(agg_df, metrics_df)
            self._generate_plots(agg_df, datasets)
            
            logging.info(f"Phase 4 complete! Results → {self.file_manager.analysis_dir}")
        except Exception as e:
            logging.error(f"Phase 4 failed: {e}")
            raise
    
    def _load_all_results(self, datasets):
        """Load all result CSV files from Phase 3."""
        all_data = []
        seen_keys = set()  # Track (dataset, gen_model, method, judge_model, strategy, image_id, level) to avoid duplicates
        
        for dataset in datasets:
            result_files = self.file_manager.scan_result_files(dataset)
            
            for result_file in result_files:
                params = self.file_manager.parse_result_file_path(result_file, dataset)
                if not params:
                    continue
                
                rows = self.file_manager.load_csv(result_file, skip_header=True)
                
                for row in rows:
                    if len(row) >= 3:
                        try:
                            image_id = row[0]
                            occlusion_level = float(row[1])
                            is_correct = int(row[2])
                            
                            # Create unique key to detect duplicates
                            unique_key = (
                                dataset,
                                params['gen_model'],
                                params['method'],
                                params['judge_model'],
                                params['strategy'],
                                image_id,
                                occlusion_level
                            )
                            
                            # Skip duplicates (keep first occurrence)
                            if unique_key in seen_keys:
                                logging.debug(f"Skipping duplicate: {unique_key}")
                                continue
                            
                            seen_keys.add(unique_key)
                            
                            all_data.append({
                                'dataset': dataset,
                                'generating_model': params['gen_model'],
                                'attribution_method': params['method'],
                                'judging_model': params['judge_model'],
                                'fill_strategy': params['strategy'],
                                'image_id': image_id,
                                'occlusion_level': occlusion_level,
                                'is_correct': is_correct
                            })
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Skipping corrupted row in {result_file}: {row} - {e}")
                            continue
        
        return all_data
    
    def _calculate_aggregated_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean accuracy aggregated by grouping columns."""
        group_cols = [
            "dataset", "generating_model", "attribution_method",
            "judging_model", "fill_strategy", "occlusion_level"
        ]
        agg_df = df.groupby(group_cols)['is_correct'].mean().reset_index()
        agg_df.rename(columns={'is_correct': 'mean_accuracy'}, inplace=True)
        return agg_df
    
    def _calculate_metrics(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate faithfulness metrics (AUC, DROP) for each curve."""
        metrics_list = []
        curve_group_cols = [
            "dataset", "generating_model", "attribution_method",
            "judging_model", "fill_strategy"
        ]
        
        for name, curve_df in agg_df.groupby(curve_group_cols):
            try:
                dataset, gen_model, method, judge_model, fill_strat = name
                
                baseline_acc = 1.0
                if 0 in curve_df['occlusion_level'].values:
                    baseline_acc = curve_df[curve_df['occlusion_level'] == 0]['mean_accuracy'].iloc[0]
                
                accuracies = curve_df['mean_accuracy'].tolist()
                levels = curve_df['occlusion_level'].tolist()
                
                auc = calculate_auc(accuracies, levels)
                drop75 = calculate_drop(accuracies, levels, initial_accuracy=baseline_acc, drop_level=75)
                
                metrics_list.append({
                    "dataset": dataset,
                    "generating_model": gen_model,
                    "attribution_method": method,
                    "judging_model": judge_model,
                    "fill_strategy": fill_strat,
                    "auc": auc,
                    "drop_at_75": drop75
                })
            except Exception as e:
                logging.warning(f"Error calculating metrics for {name}: {e}")
                continue
        
        return pd.DataFrame(metrics_list)
    
    def _save_results(self, agg_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Save aggregated results and metrics to CSV files."""
        # Ensure analysis directory exists
        self.file_manager.ensure_dir_exists(self.file_manager.analysis_dir)
        
        agg_output_path = self.file_manager.analysis_dir / "aggregated_accuracy_curves.csv"
        metrics_output_path = self.file_manager.analysis_dir / "faithfulness_metrics.csv"
        
        agg_df.to_csv(agg_output_path, index=False)
        metrics_df.to_csv(metrics_output_path, index=False)
    
    def _generate_plots(self, agg_df: pd.DataFrame, datasets):
        """Generate visualization plots for each dataset."""
        for dataset in datasets:
            dataset_df = agg_df[agg_df['dataset'] == dataset].copy()
            if not dataset_df.empty:
                dataset_analysis_dir = self.file_manager.analysis_dir / dataset
                self.file_manager.ensure_dir_exists(dataset_analysis_dir)
                
                plot_accuracy_degradation_curves(
                    dataset_df, output_dir=dataset_analysis_dir
                )
                plot_fill_strategy_comparison(
                    dataset_df, output_dir=dataset_analysis_dir
                )


def main():
    """Simple main function to run Phase 4."""
    import sys
    import logging
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config
    from core.file_manager import FileManager
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    file_manager = FileManager(config.BASE_DIR)
    runner = Phase4Runner(config, file_manager)
    runner.run()


if __name__ == "__main__":
    main()

