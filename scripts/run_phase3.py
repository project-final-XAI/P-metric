"""
Run Phase 3: Analysis and Visualization.

Usage:
    python scripts/run_phase3.py
"""

import argparse
import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3: Analysis and Visualization")
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("PHASE 3: ANALYSIS AND VISUALIZATION")
    logging.info("=" * 60)
    
    try:
        runner = ExperimentRunner(config)
        runner.run_phase_3()
        logging.info("Phase 3 complete!")
    except Exception as e:
        logging.error(f"Phase 3 failed: {e}")
        raise


if __name__ == "__main__":
    main()

