"""Baseline feature engineering for CTR prediction.

This module creates 15-20 baseline features that serve as the CONTROL group
for A/B testing in later phases.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging
from typing import Dict, Any, List


class BaselineFeatureEngineer:
    """
    Create baseline features for CTR prediction.

    These are simple, obvious features that any ML engineer would create.
    They serve as our CONTROL group for A/B testing.

    Feature Groups:
    1. Missing value indicators (13 features for numerical columns)
    2. Count encoding for categorical features (26 features)
    3. Target encoding for top categorical features (5 features)
    4. Original numerical features (13 features)

    Total: ~57 baseline features
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize baseline feature engineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.numerical_cols = config['features']['numerical_cols']
        self.categorical_cols = config['features']['categorical_cols']
        self.target_col = config['features']['target_col']

        # Store encoding mappings for validation/test sets
        self.count_encodings = {}
        self.target_encodings = {}

    def create_features(self, df: DataFrame, is_training: bool = True) -> DataFrame:
        """
        Create all baseline features.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (for fitting encodings)

        Returns:
            DataFrame with baseline features
        """
        self.logger.info("Creating baseline features...")

        # 1. Handle missing values and create indicators
        df = self._handle_missing_values(df)

        # 2. Basic count encoding for categoricals
        df = self._count_encode_categoricals(df, is_training=is_training)

        # 3. Target encoding for top categoricals
        df = self._target_encode_categoricals(df, is_training=is_training)

        # 4. Numerical features are already present (I1-I13)
        # Just ensure they're properly filled

        feature_count = len([c for c in df.columns if c not in [self.target_col]])
        self.logger.info(f"Created {feature_count} total features (including baseline)")

        return df

    def _handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values and create missing indicators.

        For numerical columns:
        - Fill missing with -1
        - Create binary indicator: I{i}_missing (1 if missing, 0 otherwise)

        For categorical columns:
        - Fill missing with "MISSING"

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with imputed values and missing indicators
        """
        self.logger.info("Handling missing values...")

        # Handle numerical columns
        for col in self.numerical_cols:
            # Create missing indicator
            df = df.withColumn(
                f"{col}_missing",
                F.when(F.col(col).isNull(), 1).otherwise(0)
            )

            # Fill missing with -1
            df = df.fillna({col: -1})

        # Handle categorical columns
        for col in self.categorical_cols:
            df = df.fillna({col: "MISSING"})

        return df

    def _count_encode_categoricals(
        self,
        df: DataFrame,
        is_training: bool = True
    ) -> DataFrame:
        """
        Count encoding: Replace category with its frequency.

        Example: If C1="abc" appears 1000 times, C1_count=1000

        This helps the model understand which categories are rare vs common.

        Args:
            df: Input DataFrame
            is_training: Whether to fit the encoding (True) or use existing (False)

        Returns:
            DataFrame with count-encoded features
        """
        self.logger.info("Creating count encoding features...")

        for col in self.categorical_cols:
            if is_training:
                # Calculate counts
                counts = df.groupBy(col).count()
                counts = counts.withColumnRenamed('count', f'{col}_count')

                # Store for validation/test sets
                self.count_encodings[col] = counts

                # Join back
                df = df.join(counts, on=col, how='left')
            else:
                # Use pre-computed encodings
                if col in self.count_encodings:
                    df = df.join(self.count_encodings[col], on=col, how='left')

                    # Fill missing with 0 (for unseen categories)
                    df = df.fillna({f'{col}_count': 0})

        return df

    def _target_encode_categoricals(
        self,
        df: DataFrame,
        is_training: bool = True
    ) -> DataFrame:
        """
        Target encoding: Replace category with mean of target.

        Example: If C1="abc" has avg click rate 0.05, C1_mean_ctr=0.05

        Only apply to top 5 categorical features to avoid overfitting.

        Args:
            df: Input DataFrame
            is_training: Whether to fit the encoding (True) or use existing (False)

        Returns:
            DataFrame with target-encoded features
        """
        self.logger.info("Creating target encoding features...")

        # Get top N categoricals from config
        target_encode_cols = self.config['features']['baseline']['target_encoding_cols']

        for col in target_encode_cols:
            if is_training:
                # Calculate mean click rate per category
                target_means = df.groupBy(col).agg(
                    F.mean(self.target_col).alias(f'{col}_mean_ctr'),
                    F.count(self.target_col).alias(f'{col}_target_count')
                )

                # Store for validation/test sets
                self.target_encodings[col] = target_means

                # Join back
                df = df.join(target_means, on=col, how='left')
            else:
                # Use pre-computed encodings
                if col in self.target_encodings:
                    df = df.join(self.target_encodings[col], on=col, how='left')

                    # Fill missing with global mean (for unseen categories)
                    global_mean = df.agg(F.mean(self.target_col)).collect()[0][0]
                    df = df.fillna({f'{col}_mean_ctr': global_mean})
                    df = df.fillna({f'{col}_target_count': 0})

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names (excluding target).

        Returns:
            List of feature column names
        """
        feature_names = []

        # Original numerical features
        feature_names.extend(self.numerical_cols)

        # Missing indicators
        feature_names.extend([f"{col}_missing" for col in self.numerical_cols])

        # Count encodings
        feature_names.extend([f"{col}_count" for col in self.categorical_cols])

        # Target encodings
        target_encode_cols = self.config['features']['baseline']['target_encoding_cols']
        for col in target_encode_cols:
            feature_names.append(f"{col}_mean_ctr")
            feature_names.append(f"{col}_target_count")

        return feature_names

    def get_feature_summary(self) -> Dict[str, int]:
        """
        Get summary of feature counts by type.

        Returns:
            Dictionary with feature type counts
        """
        target_encode_cols = self.config['features']['baseline']['target_encoding_cols']

        summary = {
            'numerical_features': len(self.numerical_cols),
            'missing_indicators': len(self.numerical_cols),
            'count_encodings': len(self.categorical_cols),
            'target_encodings': len(target_encode_cols) * 2,  # mean_ctr + target_count
            'total': len(self.get_feature_names())
        }

        return summary
