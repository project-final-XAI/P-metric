"""
Run Phase 2: Occlusion Evaluation.

Usage:
    python scripts/run_phase2.py --dataset imagenet
"""

import argparse
import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2: Occlusion Evaluation")
    parser.add_argument('--dataset', default='imagenet', help='Dataset name')
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("PHASE 2: OCCLUSION EVALUATION")
    logging.info("=" * 60)
    
    try:
        runner = ExperimentRunner(config)
        runner.run_phase_2(args.dataset)
        logging.info("Phase 2 complete!")
    except Exception as e:
        logging.error(f"Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    main()

