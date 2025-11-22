"""
Phase 3: Analysis and Visualization Runner.

Loads Phase 2 results, calculates metrics (AUC, DROP), and generates visualization plots.
"""

import logging
import pandas as pd

from core.file_manager import FileManager
from evaluation.metrics import calculate_auc, calculate_drop
from visualization.plotter import plot_accuracy_degradation_curves, plot_fill_strategy_comparison


class Phase3Runner:
    """Handles Phase 3: Analysis and visualization of evaluation results."""
    
    def __init__(self, config, file_manager: FileManager):
        """
        Initialize Phase 3 runner.
        
        Args:
            config: Configuration object
            file_manager: File path manager
        """
        self.config = config
        self.file_manager = file_manager
    
    def run(self):
        """Run analysis and visualization on Phase 2 results."""
        try:
            # Check if results directory exists
            if not self.file_manager.results_dir.exists():
                logging.error(f"Results directory does not exist: {self.file_manager.results_dir}")
                return
            
            # Find all datasets with results
            datasets = [
                d.name for d in self.file_manager.results_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            
            if not datasets:
                logging.error("No result datasets found")
                return
            
            logging.info(f"Starting Phase 3 - Analyzing: {', '.join(datasets)}")
            
            # Load all results from file structure
            all_data = self._load_all_results(datasets)
            
            if not all_data:
                logging.warning("No results data found")
                return
            
            # Create DataFrame and calculate aggregated accuracy
            df = pd.DataFrame(all_data)
            agg_df = self._calculate_aggregated_accuracy(df)
            
            # Calculate metrics (AUC, DROP)
            metrics_df = self._calculate_metrics(agg_df)
            
            # Save results
            self._save_results(agg_df, metrics_df)
            
            # Generate plots per dataset
            self._generate_plots(agg_df, datasets)
            
            logging.info(f"Phase 3 complete! Results → {self.file_manager.analysis_dir}")
        except Exception as e:
            logging.error(f"Phase 3 failed: {e}")
            raise
    
    def _load_all_results(self, datasets):
        """
        Load all result CSV files from Phase 2.
        
        Args:
            datasets: List of dataset names
            
        Returns:
            List of dictionaries containing result data
        """
        all_data = []
        
        for dataset in datasets:
            result_files = self.file_manager.scan_result_files(dataset)
            
            for result_file in result_files:
                # Parse file path to get parameters
                params = self.file_manager.parse_result_file_path(result_file, dataset)
                if not params:
                    continue
                
                # Load CSV data
                rows = self.file_manager.load_csv(result_file, skip_header=True)
                
                # Add metadata to each row
                for row in rows:
                    if len(row) >= 3:
                        try:
                            all_data.append({
                                'dataset': dataset,
                                'generating_model': params['gen_model'],
                                'attribution_method': params['method'],
                                'judging_model': params['judge_model'],
                                'fill_strategy': params['strategy'],
                                'image_id': row[0],
                                'occlusion_level': float(row[1]),
                                'is_correct': int(row[2])
                            })
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Skipping corrupted row in {result_file}: {row} - {e}")
                            continue
        
        return all_data
    
    def _calculate_aggregated_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean accuracy aggregated by grouping columns.
        
        Args:
            df: DataFrame with result data
            
        Returns:
            DataFrame with aggregated accuracy
        """
        group_cols = [
            "dataset", "generating_model", "attribution_method",
            "judging_model", "fill_strategy", "occlusion_level"
        ]
        agg_df = df.groupby(group_cols)['is_correct'].mean().reset_index()
        agg_df.rename(columns={'is_correct': 'mean_accuracy'}, inplace=True)
        return agg_df
    
    def _calculate_metrics(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate faithfulness metrics (AUC, DROP) for each curve.
        
        Args:
            agg_df: DataFrame with aggregated accuracy
            
        Returns:
            DataFrame with calculated metrics
        """
        metrics_list = []
        curve_group_cols = [
            "dataset", "generating_model", "attribution_method",
            "judging_model", "fill_strategy"
        ]
        
        for name, curve_df in agg_df.groupby(curve_group_cols):
            try:
                dataset, gen_model, method, judge_model, fill_strat = name
                
                # Get baseline accuracy (at 0% occlusion)
                baseline_acc = 1.0
                if 0 in curve_df['occlusion_level'].values:
                    baseline_acc = curve_df[curve_df['occlusion_level'] == 0]['mean_accuracy'].iloc[0]
                
                accuracies = curve_df['mean_accuracy'].tolist()
                levels = curve_df['occlusion_level'].tolist()
                
                # Calculate metrics
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
        """
        Save aggregated results and metrics to CSV files.
        
        Args:
            agg_df: DataFrame with aggregated accuracy
            metrics_df: DataFrame with calculated metrics
        """
        agg_output_path = self.file_manager.analysis_dir / "aggregated_accuracy_curves.csv"
        metrics_output_path = self.file_manager.analysis_dir / "faithfulness_metrics.csv"
        
        agg_df.to_csv(agg_output_path, index=False)
        metrics_df.to_csv(metrics_output_path, index=False)
    
    def _generate_plots(self, agg_df: pd.DataFrame, datasets):
        """
        Generate visualization plots for each dataset.
        
        Args:
            agg_df: DataFrame with aggregated accuracy
            datasets: List of dataset names
        """
        for dataset in datasets:
            dataset_df = agg_df[agg_df['dataset'] == dataset].copy()
            if not dataset_df.empty:
                # Create dataset-specific output directory
                dataset_analysis_dir = self.file_manager.analysis_dir / dataset
                self.file_manager.ensure_dir_exists(dataset_analysis_dir)
                
                plot_accuracy_degradation_curves(
                    dataset_df, output_dir=dataset_analysis_dir
                )
                plot_fill_strategy_comparison(
                    dataset_df, output_dir=dataset_analysis_dir
                )

