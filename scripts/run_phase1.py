"""
Run Phase 1: Generate Heatmaps.

Usage:
    python scripts/run_phase1.py --dataset imagenet
"""

import argparse
import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1: Generate Heatmaps")
    parser.add_argument('--dataset', default='imagenet', help='Dataset name')
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("PHASE 1: HEATMAP GENERATION")
    logging.info("=" * 60)
    
    try:
        runner = ExperimentRunner(config)
        runner.run_phase_1(args.dataset)
        logging.info("Phase 1 complete!")
    except Exception as e:
        logging.error(f"Phase 1 failed: {e}")
        raise


if __name__ == "__main__":
    main()

