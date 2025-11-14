"""PySpark utility functions for FeatureForge."""

from pyspark.sql import SparkSession
from typing import Dict, Any, Optional
import logging


def create_spark_session(
    app_name: str = "FeatureForge",
    master: str = "local[*]",
    executor_memory: str = "4g",
    driver_memory: str = "4g",
    additional_configs: Optional[Dict[str, Any]] = None
) -> SparkSession:
    """
    Create and configure Spark session.

    Args:
        app_name: Application name
        master: Spark master URL
        executor_memory: Executor memory
        driver_memory: Driver memory
        additional_configs: Additional Spark configurations

    Returns:
        Configured SparkSession
    """
    logger = logging.getLogger(__name__)

    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Add additional configurations
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()

    logger.info(f"Spark session created: {app_name}")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Master: {master}")

    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """
    Stop Spark session.

    Args:
        spark: SparkSession to stop
    """
    logger = logging.getLogger(__name__)

    if spark:
        spark.stop()
        logger.info("Spark session stopped")


def get_dataframe_info(df, name: str = "DataFrame") -> None:
    """
    Print useful information about a PySpark DataFrame.

    Args:
        df: PySpark DataFrame
        name: Name for logging
    """
    logger = logging.getLogger(__name__)

    logger.info(f"=== {name} Info ===")
    logger.info(f"Number of rows: {df.count():,}")
    logger.info(f"Number of columns: {len(df.columns)}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Schema:")
    df.printSchema()
    logger.info("=" * 50)
