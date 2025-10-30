"""
Run Full Experiment: All 3 Phases.

Usage:
    python scripts/run_full.py --dataset imagenet
"""

import argparse
import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    parser = argparse.ArgumentParser(description="Run Full CROSS-XAI Experiment")
    parser.add_argument('--dataset', default='SIPaKMeD', help='Dataset name')
    args = parser.parse_args()

    logging.info("CROSS-XAI FULL EXPERIMENT")
    logging.info(f"Dataset: {args.dataset}")
    
    try:
        runner = ExperimentRunner(config)
        
        # Run all phases
        runner.run_phase_1(args.dataset)
        runner.run_phase_2(args.dataset)
        runner.run_phase_3()
        
        logging.info("=" * 60)
        logging.info("FULL EXPERIMENT COMPLETE!")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"Full experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()

