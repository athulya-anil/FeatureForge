"""Data loading module for Criteo CTR dataset."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import logging
from typing import Dict, Any, Optional


class CriteoDataLoader:
    """Load Criteo CTR dataset with PySpark."""

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize data loader.

        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_raw_data(self, path: str) -> DataFrame:
        """
        Load raw Criteo data from TSV file.

        Args:
            path: Path to train.txt file

        Returns:
            PySpark DataFrame
        """
        self.logger.info(f"Loading data from: {path}")

        # Define schema for Criteo dataset
        schema = self._get_criteo_schema()

        # Load data
        df = self.spark.read.csv(
            path,
            sep='\t',
            header=False,
            schema=schema,
            inferSchema=False
        )

        row_count = df.count()
        self.logger.info(f"Loaded {row_count:,} rows")
        self.logger.info(f"Columns: {df.columns}")

        # Basic validation
        self._validate_data(df)

        return df

    def _get_criteo_schema(self) -> StructType:
        """
        Define schema for Criteo dataset.

        Returns:
            StructType schema
        """
        fields = [StructField("click", IntegerType(), True)]

        # Add numerical features I1-I13
        for i in range(1, 14):
            fields.append(StructField(f"I{i}", IntegerType(), True))

        # Add categorical features C1-C26
        for i in range(1, 27):
            fields.append(StructField(f"C{i}", StringType(), True))

        return StructType(fields)

    def _validate_data(self, df: DataFrame) -> None:
        """
        Basic data validation.

        Args:
            df: DataFrame to validate
        """
        total_count = df.count()

        # Check click rate
        click_count = df.filter(F.col('click') == 1).count()
        click_rate = click_count / total_count if total_count > 0 else 0

        self.logger.info(f"Click rate: {click_rate:.4f} ({click_count:,} / {total_count:,})")

        # Check missing values
        self.logger.info("Checking missing values...")
        missing_info = []

        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                null_pct = null_count / total_count * 100
                missing_info.append(f"{col}: {null_pct:.2f}% ({null_count:,} rows)")

        if missing_info:
            self.logger.info("Missing values found:")
            for info in missing_info[:10]:  # Show first 10
                self.logger.info(f"  {info}")
            if len(missing_info) > 10:
                self.logger.info(f"  ... and {len(missing_info) - 10} more columns")
        else:
            self.logger.info("No missing values found")

    def create_sample(
        self,
        df: DataFrame,
        sample_size: int,
        output_path: str,
        stratified: bool = True
    ) -> DataFrame:
        """
        Create sample dataset for testing.

        Args:
            df: Full dataset
            sample_size: Number of rows to sample
            output_path: Where to save sample
            stratified: Whether to use stratified sampling (preserves click rate)

        Returns:
            Sample DataFrame
        """
        self.logger.info(f"Creating sample of {sample_size:,} rows...")

        if stratified:
            # Stratified sampling to preserve click rate
            total_count = df.count()
            click_count = df.filter(F.col('click') == 1).count()
            no_click_count = total_count - click_count

            click_rate = click_count / total_count

            # Calculate sampling fractions
            target_clicks = int(sample_size * click_rate)
            target_no_clicks = sample_size - target_clicks

            click_fraction = min(1.0, target_clicks / click_count) if click_count > 0 else 0
            no_click_fraction = min(1.0, target_no_clicks / no_click_count) if no_click_count > 0 else 0

            self.logger.info(f"Stratified sampling: click_fraction={click_fraction:.4f}, no_click_fraction={no_click_fraction:.4f}")

            sample_df = df.sampleBy('click', fractions={0: no_click_fraction, 1: click_fraction}, seed=42)
        else:
            # Simple random sampling
            fraction = sample_size / df.count()
            sample_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

        # Save as parquet
        self.logger.info(f"Saving sample to: {output_path}")
        sample_df.write.parquet(output_path, mode='overwrite')

        actual_count = sample_df.count()
        self.logger.info(f"Sample created: {actual_count:,} rows")

        # Validate sample
        self._validate_data(sample_df)

        return sample_df

    def load_parquet(self, path: str) -> DataFrame:
        """
        Load data from parquet file.

        Args:
            path: Path to parquet file/directory

        Returns:
            PySpark DataFrame
        """
        self.logger.info(f"Loading parquet from: {path}")
        df = self.spark.read.parquet(path)

        row_count = df.count()
        self.logger.info(f"Loaded {row_count:,} rows from parquet")

        return df

    def save_parquet(self, df: DataFrame, path: str, mode: str = 'overwrite') -> None:
        """
        Save DataFrame as parquet.

        Args:
            df: DataFrame to save
            path: Output path
            mode: Write mode ('overwrite', 'append', 'ignore', 'error')
        """
        self.logger.info(f"Saving {df.count():,} rows to: {path}")
        df.write.parquet(path, mode=mode)
        self.logger.info("Save complete")
