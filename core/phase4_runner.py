"""
Phase 4: Analysis and Visualization Runner.

Loads Phase 3 results, calculates metrics (AUC, DROP), and generates visualization plots.
"""
import logging
import pandas as pd
from core.file_manager import FileManager
from evaluation.metrics import calculate_auc, calculate_drop
from visualization.plotter import plot_accuracy_degradation_curves, plot_fill_strategy_comparison
from scripts import create_excel_reports

from typing import Any, List, Dict


class Phase4Runner:
    """Handles Phase 4: Analysis and visualization of evaluation results."""

    def __init__(
            self,
            config,
            gpu_manager: Any = None,
            file_manager: FileManager = None,
            dataset_handler: Any = None,
            model_provider: Any = None
    ):
        """Initialize Phase 4 runner."""
        self.config = config
        # Gracefully handle if file_manager is passed as the 2nd positional argument
        if isinstance(gpu_manager, FileManager):
            self.file_manager = gpu_manager
            self.gpu_manager = None
        else:
            self.gpu_manager = gpu_manager
            self.file_manager = file_manager

        self.dataset_handler = dataset_handler
        self.model_provider = model_provider

    def run(self):
        """Run analysis and visualization on Phase 3 results."""
        try:
            if not self.file_manager.results_dir.exists():
                logging.error(f"Results directory does not exist: {self.file_manager.results_dir}")
                return

            datasets = sorted(
                d.name for d in self.file_manager.results_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            )

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

            # Generate detailed Excel reports (pivot tables + charts).
            try:
                create_excel_reports.main(agg_df=agg_df)
            except Exception as e:
                logging.warning(f"Failed to generate Excel reports: {e}")

            logging.info(f"Phase 4 complete! Results → {self.file_manager.analysis_dir}")
        except Exception as e:
            logging.error(f"Phase 4 failed: {e}")
            raise

    def _load_all_results(self, datasets: List[str]) -> List[Dict[str, Any]]:
        """Load all result CSV files from Phase 3, parsing metadata directly from paths."""
        all_data = []
        seen_keys = set()

        for dataset in datasets:
            dataset_results_dir = self.file_manager.results_dir / dataset
            if not dataset_results_dir.exists():
                logging.warning(f"Results subdirectory for dataset '{dataset}' does not exist.")
                continue

            # Recursively find all CSV files inside the dataset results folder
            result_files = list(dataset_results_dir.rglob("*.csv"))

            for result_file in result_files:
                # --- START DIRECT PATH PARSING ---
                # SWAPPED FIX: Adjusted extraction to handle judge and generator swap
                filename_stem = result_file.stem
                parts = filename_stem.split('-')

                if len(parts) >= 4:
                    # Previous configuration: gen_model = parts[0], judge_model = parts[1]
                    judge_model = parts[0]
                    gen_model = parts[1]
                    method = parts[2]
                    strategy = parts[3]
                else:
                    # Fallback lookup swapped from: dataset / gen_model / judge_model / method / strategy.csv
                    # to: dataset / judge_model / gen_model / method / strategy.csv
                    path_parts = result_file.relative_to(dataset_results_dir).parts
                    if len(path_parts) >= 4:
                        judge_model = path_parts[0]
                        gen_model = path_parts[1]
                        method = path_parts[2]
                        strategy = result_file.stem  # strategy name
                    else:
                        logging.warning(f"Could not parse metadata from result path: {result_file}")
                        continue
                # --- END DIRECT PATH PARSING ---

                rows = self.file_manager.load_csv(result_file, skip_header=True)

                for row in rows:
                    if len(row) >= 3:
                        try:
                            image_id = row[0]
                            occlusion_level = float(row[1])
                            is_correct = int(row[2])
                            is_correct_top5 = int(row[3]) if len(row) >= 4 else is_correct

                            unique_key = (
                                dataset,
                                gen_model,
                                method,
                                judge_model,
                                strategy,
                                image_id,
                                occlusion_level
                            )

                            if unique_key in seen_keys:
                                continue

                            seen_keys.add(unique_key)

                            all_data.append({
                                'dataset': dataset,
                                'generating_model': gen_model,
                                'attribution_method': method,
                                'judging_model': judge_model,
                                'fill_strategy': strategy,
                                'image_id': image_id,
                                'occlusion_level': occlusion_level,
                                'is_correct': is_correct,
                                'is_correct_top5': is_correct_top5
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
        agg_df = df.groupby(group_cols).agg({
            'is_correct': 'mean',
            'is_correct_top5': 'mean'
        }).reset_index()

        agg_df.rename(columns={
            'is_correct': 'mean_accuracy',
            'is_correct_top5': 'mean_accuracy_top5'
        }, inplace=True)
        return agg_df

    def _calculate_metrics(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate faithfulness metrics (AUC, DROP) for each curve with structural sorting validation."""
        metrics_list = []
        curve_group_cols = [
            "dataset", "generating_model", "attribution_method",
            "judging_model", "fill_strategy"
        ]

        for name, curve_df in agg_df.groupby(curve_group_cols):
            try:
                dataset, gen_model, method, judge_model, fill_strat = name

                # CRITICAL FIX: Explicitly sort by occlusion level so the curves don't scramble
                sorted_curve = curve_df.sort_values(by='occlusion_level').copy()

                levels = sorted_curve['occlusion_level'].tolist()
                accuracies = sorted_curve['mean_accuracy'].tolist()
                accuracies_top5 = sorted_curve['mean_accuracy_top5'].tolist()

                if not levels:
                    continue

                # Top-1 metrics calculation
                baseline_acc = 1.0
                if 0 in levels:
                    baseline_acc = sorted_curve[sorted_curve['occlusion_level'] == 0]['mean_accuracy'].iloc[0]
                else:
                    baseline_acc = accuracies[0]  # Fallback to earliest point if 0 is absent

                auc = calculate_auc(accuracies, levels)
                drop75 = calculate_drop(accuracies, levels, initial_accuracy=baseline_acc, drop_level=75)

                # Top-5 metrics calculation
                baseline_acc_top5 = 1.0
                if 0 in levels:
                    baseline_acc_top5 = sorted_curve[sorted_curve['occlusion_level'] == 0]['mean_accuracy_top5'].iloc[0]
                else:
                    baseline_acc_top5 = accuracies_top5[0]

                auc_top5 = calculate_auc(accuracies_top5, levels)
                drop75_top5 = calculate_drop(accuracies_top5, levels, initial_accuracy=baseline_acc_top5, drop_level=75)

                metrics_list.append({
                    "dataset": dataset,
                    "generating_model": gen_model,
                    "attribution_method": method,
                    "judging_model": judge_model,
                    "fill_strategy": fill_strat,
                    "auc": auc,
                    "drop_at_75": drop75,
                    "auc_top5": auc_top5,
                    "drop_at_75_top5": drop75_top5
                })
            except Exception as e:
                logging.warning(f"Error calculating metrics for {name}: {e}", exc_info=True)
                continue

        return pd.DataFrame(metrics_list)

    def _save_results(self, agg_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Save aggregated results and metrics to CSV files."""
        self.file_manager.ensure_dir_exists(self.file_manager.analysis_dir)

        agg_output_path = self.file_manager.analysis_dir / "aggregated_accuracy_curves.csv"
        metrics_output_path = self.file_manager.analysis_dir / "faithfulness_metrics.csv"

        agg_df.to_csv(agg_output_path, index=False)
        metrics_df.to_csv(metrics_output_path, index=False)

    def _generate_plots(self, agg_df: pd.DataFrame, datasets: List[str]):
        """Generate visualization plots for each dataset."""
        for dataset in datasets:
            dataset_df = agg_df[agg_df['dataset'] == dataset].copy()
            if not dataset_df.empty:
                dataset_analysis_dir = self.file_manager.analysis_dir / dataset
                self.file_manager.ensure_dir_exists(dataset_analysis_dir)

                try:
                    plot_accuracy_degradation_curves(
                        dataset_df, output_dir=dataset_analysis_dir
                    )
                    plot_fill_strategy_comparison(
                        dataset_df, output_dir=dataset_analysis_dir
                    )
                except Exception as e:
                    logging.error(f"Failed to generate visualization plots for {dataset}: {e}")

if __name__ == "__main__":
    from core._bootstrap import bootstrap_phase4
    runner = bootstrap_phase4()
    runner.run()