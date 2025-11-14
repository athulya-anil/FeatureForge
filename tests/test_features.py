"""
Unit tests for feature engineering modules.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from src.config import Config
from src.features.base_features import BaselineFeatureEngineer
from src.features.feature_engine import FeatureEngine


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    spark = SparkSession.builder \
        .appName("FeatureForge-Tests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    yield spark

    spark.stop()


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
        'config.yaml'
    )
    return Config(config_path)


@pytest.fixture
def sample_data(spark):
    """Create sample test data."""
    schema = StructType([
        StructField("click", IntegerType(), True),
        StructField("I1", IntegerType(), True),
        StructField("I2", IntegerType(), True),
        StructField("I3", IntegerType(), True),
        StructField("C1", StringType(), True),
        StructField("C2", StringType(), True),
        StructField("C3", StringType(), True),
        StructField("C4", StringType(), True),
        StructField("C5", StringType(), True),
    ])

    data = [
        (0, 1, 2, None, "cat1", "cat2", "cat3", "cat4", "cat5"),
        (1, None, 5, 6, "cat1", "cat2", "cat3", "cat4", "cat5"),
        (0, 7, 8, 9, "cat6", "cat7", "cat8", "cat9", "cat10"),
        (1, 10, None, 12, "cat6", "cat7", "cat8", "cat9", "cat10"),
        (0, 13, 14, 15, None, "cat2", "cat3", "cat4", "cat5"),
    ]

    df = spark.createDataFrame(data, schema)
    return df


class TestBaselineFeatureEngineer:
    """Test baseline feature engineering."""

    def test_missing_value_indicators(self, config, sample_data):
        """Test missing value indicator creation."""
        # Modify config for test
        test_config = config.config.copy()
        test_config['features']['numerical_cols'] = ['I1', 'I2', 'I3']
        test_config['features']['categorical_cols'] = ['C1', 'C2', 'C3', 'C4', 'C5']
        test_config['features']['baseline']['target_encoding_cols'] = ['C1', 'C2']

        engineer = BaselineFeatureEngineer(test_config)

        # Handle missing values
        df_result = engineer._handle_missing_values(sample_data)

        # Check missing indicators exist
        assert 'I1_missing' in df_result.columns
        assert 'I2_missing' in df_result.columns
        assert 'I3_missing' in df_result.columns

        # Check missing values are filled
        assert df_result.filter(F.col('I1').isNull()).count() == 0
        assert df_result.filter(F.col('I2').isNull()).count() == 0
        assert df_result.filter(F.col('I3').isNull()).count() == 0

        # Check indicator values
        result_pd = df_result.toPandas()
        assert result_pd.loc[0, 'I3_missing'] == 1  # Row 0 has I3=None
        assert result_pd.loc[1, 'I1_missing'] == 1  # Row 1 has I1=None
        assert result_pd.loc[0, 'I1_missing'] == 0  # Row 0 has I1=1

    def test_count_encoding(self, config, sample_data):
        """Test count encoding for categorical features."""
        test_config = config.config.copy()
        test_config['features']['numerical_cols'] = ['I1', 'I2', 'I3']
        test_config['features']['categorical_cols'] = ['C1', 'C2', 'C3', 'C4', 'C5']
        test_config['features']['baseline']['target_encoding_cols'] = ['C1', 'C2']

        engineer = BaselineFeatureEngineer(test_config)

        # Fill missing values first
        df_filled = engineer._handle_missing_values(sample_data)

        # Apply count encoding
        df_result = engineer._count_encode_categoricals(df_filled, is_training=True)

        # Check count columns exist
        assert 'C1_count' in df_result.columns
        assert 'C2_count' in df_result.columns

        # Check counts are correct
        result_pd = df_result.toPandas()

        # C1="cat1" appears 2 times
        cat1_rows = result_pd[result_pd['C1'] == 'cat1']
        assert all(cat1_rows['C1_count'] == 2)

        # C1="cat6" appears 2 times
        cat6_rows = result_pd[result_pd['C1'] == 'cat6']
        assert all(cat6_rows['C1_count'] == 2)

    def test_target_encoding(self, config, sample_data):
        """Test target encoding for categorical features."""
        test_config = config.config.copy()
        test_config['features']['numerical_cols'] = ['I1', 'I2', 'I3']
        test_config['features']['categorical_cols'] = ['C1', 'C2', 'C3', 'C4', 'C5']
        test_config['features']['baseline']['target_encoding_cols'] = ['C1', 'C2']
        test_config['features']['target_col'] = 'click'

        engineer = BaselineFeatureEngineer(test_config)

        # Fill missing values first
        df_filled = engineer._handle_missing_values(sample_data)

        # Apply target encoding
        df_result = engineer._target_encode_categoricals(df_filled, is_training=True)

        # Check target encoding columns exist
        assert 'C1_mean_ctr' in df_result.columns
        assert 'C1_target_count' in df_result.columns
        assert 'C2_mean_ctr' in df_result.columns
        assert 'C2_target_count' in df_result.columns

        # Check values are reasonable (between 0 and 1 for CTR)
        result_pd = df_result.toPandas()
        assert all(result_pd['C1_mean_ctr'] >= 0)
        assert all(result_pd['C1_mean_ctr'] <= 1)

    def test_feature_names(self, config):
        """Test feature name generation."""
        test_config = config.config.copy()
        test_config['features']['numerical_cols'] = ['I1', 'I2', 'I3']
        test_config['features']['categorical_cols'] = ['C1', 'C2', 'C3']
        test_config['features']['baseline']['target_encoding_cols'] = ['C1', 'C2']

        engineer = BaselineFeatureEngineer(test_config)

        feature_names = engineer.get_feature_names()

        # Check numerical features
        assert 'I1' in feature_names
        assert 'I2' in feature_names
        assert 'I3' in feature_names

        # Check missing indicators
        assert 'I1_missing' in feature_names
        assert 'I2_missing' in feature_names

        # Check count encodings
        assert 'C1_count' in feature_names
        assert 'C2_count' in feature_names

        # Check target encodings
        assert 'C1_mean_ctr' in feature_names
        assert 'C2_mean_ctr' in feature_names

    def test_feature_summary(self, config):
        """Test feature summary generation."""
        test_config = config.config.copy()
        test_config['features']['numerical_cols'] = ['I1', 'I2', 'I3']
        test_config['features']['categorical_cols'] = ['C1', 'C2', 'C3']
        test_config['features']['baseline']['target_encoding_cols'] = ['C1', 'C2']

        engineer = BaselineFeatureEngineer(test_config)

        summary = engineer.get_feature_summary()

        assert summary['numerical_features'] == 3
        assert summary['missing_indicators'] == 3
        assert summary['count_encodings'] == 3
        assert summary['target_encodings'] == 4  # 2 cols × 2 (mean_ctr + target_count)
        assert 'total' in summary


class TestFeatureEngine:
    """Test feature engine orchestrator."""

    def test_feature_engine_initialization(self, config):
        """Test feature engine initialization."""
        engine = FeatureEngine(config)

        assert engine.config is not None
        assert engine.baseline_engineer is not None

    def test_get_feature_columns(self, config):
        """Test getting feature column names."""
        engine = FeatureEngine(config)

        # Get feature columns (excluding target)
        features = engine.get_feature_columns(exclude_target=True)

        assert len(features) > 0
        assert 'click' not in features

        # Get all columns (including target)
        features_with_target = engine.get_feature_columns(exclude_target=False)

        assert 'click' in features_with_target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
