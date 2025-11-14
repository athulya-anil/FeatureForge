"""Feature engineering pipeline orchestrator."""

from pyspark.sql import DataFrame
import logging
from typing import Dict, Any, List
from src.features.base_features import BaselineFeatureEngineer


class FeatureEngine:
    """
    Orchestrate feature engineering pipeline.

    Manages different feature engineering strategies and combines them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize feature engineers
        self.baseline_engineer = BaselineFeatureEngineer(config)

    def create_baseline_features(
        self,
        df: DataFrame,
        is_training: bool = True
    ) -> DataFrame:
        """
        Create baseline features only.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data

        Returns:
            DataFrame with baseline features
        """
        self.logger.info("Creating baseline features via FeatureEngine...")
        return self.baseline_engineer.create_features(df, is_training=is_training)

    def create_experimental_features(
        self,
        df: DataFrame,
        is_training: bool = True
    ) -> DataFrame:
        """
        Create experimental features (Phase 2).

        Args:
            df: Input DataFrame
            is_training: Whether this is training data

        Returns:
            DataFrame with experimental features
        """
        # Placeholder for Phase 2
        self.logger.info("Experimental features not yet implemented (Phase 2)")
        return df

    def get_feature_columns(self, exclude_target: bool = True) -> List[str]:
        """
        Get list of all feature columns.

        Args:
            exclude_target: Whether to exclude target column

        Returns:
            List of feature column names
        """
        features = self.baseline_engineer.get_feature_names()

        if not exclude_target:
            features.append(self.config['features']['target_col'])

        return features

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of all features.

        Returns:
            Dictionary with feature summary information
        """
        baseline_summary = self.baseline_engineer.get_feature_summary()

        summary = {
            'baseline': baseline_summary,
            'experimental': {},  # Placeholder for Phase 2
            'total_features': baseline_summary['total']
        }

        return summary

    def log_feature_summary(self) -> None:
        """Log feature summary information."""
        summary = self.get_feature_summary()

        self.logger.info("=" * 60)
        self.logger.info("FEATURE SUMMARY")
        self.logger.info("=" * 60)

        self.logger.info("Baseline Features:")
        for key, value in summary['baseline'].items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info(f"\nTotal Features: {summary['total_features']}")
        self.logger.info("=" * 60)
