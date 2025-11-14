"""Data splitting module for train/validation/test splits."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import logging
from typing import Dict, Any, Tuple


class DataSplitter:
    """Split data into train/val/test with stratification."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data splitter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def split_data(
        self,
        df: DataFrame,
        test_size: float = None,
        val_size: float = None,
        seed: int = None
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Split into train/val/test sets.

        Args:
            df: Input DataFrame
            test_size: Fraction for test set (default from config)
            val_size: Fraction for validation set (default from config)
            seed: Random seed (default from config)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get parameters from config if not provided
        if test_size is None:
            test_size = self.config['model']['test_size']
        if val_size is None:
            val_size = self.config['model']['val_size']
        if seed is None:
            seed = self.config['model']['random_state']

        self.logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}, seed={seed}")

        # Split train and test
        train_val_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)

        # Split train and val
        train_df, val_df = train_val_df.randomSplit([1 - val_size, val_size], seed=seed)

        # Log statistics
        self._log_split_stats(train_df, "Train")
        self._log_split_stats(val_df, "Validation")
        self._log_split_stats(test_df, "Test")

        return train_df, val_df, test_df

    def _log_split_stats(self, df: DataFrame, split_name: str) -> None:
        """
        Log statistics for data split.

        Args:
            df: DataFrame split
            split_name: Name of the split (Train/Validation/Test)
        """
        total = df.count()
        clicks = df.filter(F.col('click') == 1).count()
        click_rate = clicks / total if total > 0 else 0

        self.logger.info(
            f"{split_name} set: {total:,} rows, {clicks:,} clicks "
            f"(CTR: {click_rate:.4f})"
        )

    def stratified_split(
        self,
        df: DataFrame,
        test_size: float = None,
        val_size: float = None,
        seed: int = None,
        stratify_column: str = 'click'
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Stratified split to preserve class distribution.

        Args:
            df: Input DataFrame
            test_size: Fraction for test set
            val_size: Fraction for validation set
            seed: Random seed
            stratify_column: Column to stratify on

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get parameters from config if not provided
        if test_size is None:
            test_size = self.config['model']['test_size']
        if val_size is None:
            val_size = self.config['model']['val_size']
        if seed is None:
            seed = self.config['model']['random_state']

        self.logger.info(f"Stratified splitting on column: {stratify_column}")

        # Add random column for splitting
        df = df.withColumn("_random", F.rand(seed=seed))

        # Get class distribution
        class_counts = df.groupBy(stratify_column).count().collect()
        class_dist = {row[stratify_column]: row['count'] for row in class_counts}

        self.logger.info(f"Class distribution: {class_dist}")

        # Split for each class
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for class_value in class_dist.keys():
            class_df = df.filter(F.col(stratify_column) == class_value)

            # Sort by random column and split
            test_threshold = test_size
            val_threshold = test_size + val_size * (1 - test_size)

            test_class = class_df.filter(F.col("_random") < test_threshold)
            val_class = class_df.filter(
                (F.col("_random") >= test_threshold) &
                (F.col("_random") < val_threshold)
            )
            train_class = class_df.filter(F.col("_random") >= val_threshold)

            train_dfs.append(train_class)
            val_dfs.append(val_class)
            test_dfs.append(test_class)

        # Union all splits
        from functools import reduce
        train_df = reduce(DataFrame.union, train_dfs).drop("_random")
        val_df = reduce(DataFrame.union, val_dfs).drop("_random")
        test_df = reduce(DataFrame.union, test_dfs).drop("_random")

        # Log statistics
        self._log_split_stats(train_df, "Train (Stratified)")
        self._log_split_stats(val_df, "Validation (Stratified)")
        self._log_split_stats(test_df, "Test (Stratified)")

        return train_df, val_df, test_df

    def time_based_split(
        self,
        df: DataFrame,
        time_column: str,
        test_size: float = None,
        val_size: float = None
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Time-based split (for time series data).

        Args:
            df: Input DataFrame
            time_column: Column containing timestamps
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get parameters from config if not provided
        if test_size is None:
            test_size = self.config['model']['test_size']
        if val_size is None:
            val_size = self.config['model']['val_size']

        self.logger.info(f"Time-based splitting on column: {time_column}")

        # Sort by time
        df = df.orderBy(time_column)

        # Calculate split points
        total = df.count()
        test_count = int(total * test_size)
        val_count = int(total * val_size)
        train_count = total - test_count - val_count

        self.logger.info(
            f"Split counts: train={train_count:,}, val={val_count:,}, test={test_count:,}"
        )

        # Add row number
        from pyspark.sql.window import Window
        windowSpec = Window.orderBy(time_column)
        df = df.withColumn("_row_num", F.row_number().over(windowSpec))

        # Split
        train_df = df.filter(F.col("_row_num") <= train_count).drop("_row_num")
        val_df = df.filter(
            (F.col("_row_num") > train_count) &
            (F.col("_row_num") <= train_count + val_count)
        ).drop("_row_num")
        test_df = df.filter(F.col("_row_num") > train_count + val_count).drop("_row_num")

        # Log statistics
        self._log_split_stats(train_df, "Train (Time-based)")
        self._log_split_stats(val_df, "Validation (Time-based)")
        self._log_split_stats(test_df, "Test (Time-based)")

        return train_df, val_df, test_df
