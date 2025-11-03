"""
Run Phase 2: Occlusion Evaluation.
"""

import logging
from core.experiment_runner import ExperimentRunner
import config


def main():
    
    logging.info("=" * 60)
    logging.info("PHASE 2: OCCLUSION EVALUATION")
    logging.info("=" * 60)
    
    try:
        runner = ExperimentRunner(config)
        runner.run_phase_2()
        logging.info("Phase 2 complete!")
    except Exception as e:
        logging.error(f"Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    main()

