"""
Run Phase 3: Analysis and Visualization.
"""


import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    
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

