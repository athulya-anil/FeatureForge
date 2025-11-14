"""
Create a sample dataset for faster iteration and testing.

This script creates a 1M row stratified sample from the full Criteo dataset.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.spark_utils import create_spark_session, stop_spark_session
from src.data.loader import CriteoDataLoader


def main():
    """Main function to create sample dataset."""
    # Load configuration
    config = Config('config/config.yaml')

    # Setup logging
    logger = setup_logging(level='INFO')

    logger.info("=" * 60)
    logger.info("FeatureForge - Create Sample Dataset")
    logger.info("=" * 60)

    # Create Spark session
    spark = create_spark_session(
        app_name=config['spark']['app_name'],
        master=config['spark']['master'],
        executor_memory=config['spark']['executor_memory'],
        driver_memory=config['spark']['driver_memory']
    )

    try:
        # Initialize loader
        loader = CriteoDataLoader(spark, config)

        # Check if raw data exists
        raw_path = config['data']['raw_path']
        if not os.path.exists(raw_path):
            logger.error(f"Raw data not found: {raw_path}")
            logger.error("Please download the dataset first using:")
            logger.error("  bash scripts/download_data.sh")
            return

        # Load raw data
        logger.info(f"Loading raw data from: {raw_path}")
        df = loader.load_raw_data(raw_path)

        # Create sample
        sample_size = config['data']['sample_size']
        sample_path = config['data']['sample_path']

        logger.info(f"Creating stratified sample of {sample_size:,} rows...")
        sample_df = loader.create_sample(
            df,
            sample_size=sample_size,
            output_path=sample_path,
            stratified=True
        )

        logger.info("=" * 60)
        logger.info("✅ Sample dataset created successfully!")
        logger.info("=" * 60)
        logger.info(f"Sample path: {sample_path}")
        logger.info(f"Sample size: {sample_df.count():,} rows")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run notebooks/02_eda.ipynb for EDA")
        logger.info("  2. Run notebooks/03_baseline_model.ipynb to train model")
        logger.info("=" * 60)

    finally:
        # Stop Spark session
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
