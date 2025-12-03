"""
Main entry point for SSMS vs P-Metric comparison experiment.

Usage:
    python experiment/main.py --quick                    # Quick test (20 images)
    python experiment/main.py --num-images 100           # Custom number of images
    python experiment/main.py                            # Default (300 images)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment.config import ExperimentConfig
from experiment.evaluator import ExperimentEvaluator
from experiment.analysis import load_results, compute_correlations, create_visualizations, print_summary


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SSMS vs P-Metric Comparison Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment/main.py --quick                    # Quick test (20 images)
  python experiment/main.py --num-images 100           # Custom number of images
  python experiment/main.py                            # Default (300 images)
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (20 images)'
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to evaluate (overrides default)'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip statistical analysis and visualization (evaluation only)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create config
    config = ExperimentConfig(quick_mode=args.quick, num_images=args.num_images)
    
    logging.info("="*70)
    logging.info("SSMS vs P-Metric Comparison Experiment")
    logging.info("="*70)
    logging.info(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    logging.info(f"Number of images: {config.num_images}")
    logging.info(f"Dataset: {config.dataset_name}")
    logging.info(f"Generating models: {config.generating_models}")
    logging.info(f"Explainers: {config.explainers}")
    logging.info(f"Judge models: {config.judge_models}")
    logging.info(f"Device: {config.device}")
    logging.info("="*70)
    
    # Run evaluation
    evaluator = ExperimentEvaluator(config)
    evaluator.run()
    
    # Run analysis if not skipped
    if not args.skip_analysis:
        logging.info("\nRunning statistical analysis...")
        
        try:
            # Load results
            df = load_results(config.results_dir)
            
            # Compute correlations
            stats_dict = compute_correlations(df)
            
            # Create visualizations
            create_visualizations(df, config.plots_dir)
            
            # Print summary
            print_summary(stats_dict, df)
            
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info("Experiment complete!")


if __name__ == "__main__":
    main()


