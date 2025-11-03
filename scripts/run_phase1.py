"""
Run Phase 1: Generate Heatmaps.

"""

import argparse
import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    logging.info("=" * 60)
    logging.info("PHASE 1: HEATMAP GENERATION")
    logging.info("=" * 60)
    
    try:
        runner = ExperimentRunner(config)
        runner.run_phase_1()
        logging.info("Phase 1 complete!")
    except Exception as e:
        logging.error(f"Phase 1 failed: {e}")
        raise


if __name__ == "__main__":
    main()

