#!/usr/bin/env python3
"""
Script to generate responses using the SafetyProbes defense mechanism.
Usage: python scripts/generate_responses.py --dataset_type [harmful|benign] --start_index <int>
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from defenses.SafetyProbes import SafetyProbe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads the configuration file based on the model_name."""
    config_path = Path(f"{config_path}").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def dict_to_namespace(d):
    """Convert nested dictionary to namespace object for easier access"""
    from types import SimpleNamespace
    
    def _convert(item):
        if isinstance(item, dict):
            return SimpleNamespace(**{k: _convert(v) for k, v in item.items()})
        elif isinstance(item, list):
            return [_convert(i) for i in item]
        else:
            return item
    
    return _convert(d)

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Generate responses using SafetyProbes defense mechanism"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["harmful", "benign"],
        required=True,
        help="Type of dataset to process (harmful or benign)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for processing the dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/safety_probe.yaml",
        help="Path to configuration file (default: configs/safety_probe.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
        config = dict_to_namespace(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Initialize semantic entropy calculator
    logger.info("Initializing Semantic Entropy calculator...")
    safety_probe = SafetyProbe(config)

    # Process the dataset
    logger.info(f"Processing {args.dataset_type} dataset starting from index {args.start_index}")
    
    safety_probe.process_dataset(
        start_index=args.start_index,
        dataset_type=args.dataset_type
    )

    
    return 0


if __name__ == "__main__":
    sys.exit(main())
