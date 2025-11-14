"""Model training module with XGBoost and LightGBM support."""

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class XGBoostTrainer:
    """Train XGBoost model with class imbalance handling."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_names = None
        self.best_iteration = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> xgb.Booster:
        """
        Train XGBoost model.

        Args:
            X_train: Training features (pandas DataFrame)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Optional custom parameters (overrides config)

        Returns:
            Trained XGBoost Booster
        """
        self.logger.info("Training XGBoost model...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        self.logger.info(f"Class distribution: {pos_count:,} positive, {neg_count:,} negative")
        self.logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        self.logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        # Get XGBoost parameters
        if params is None:
            params = self.config['model']['xgboost_params'].copy()
        else:
            params = params.copy()

        # Override scale_pos_weight with calculated value
        params['scale_pos_weight'] = scale_pos_weight

        # Extract training-specific params
        n_estimators = params.pop('n_estimators', 100)

        self.logger.info(f"Training with {n_estimators} rounds")
        self.logger.info(f"Parameters: {params}")

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Evaluation list
        evals = [(dtrain, 'train'), (dval, 'val')]

        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=10
        )

        self.best_iteration = self.model.best_iteration
        self.logger.info(f"Training complete. Best iteration: {self.best_iteration}")

        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (probability scores).

        Args:
            X: Features (pandas DataFrame)

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features
            threshold: Classification threshold

        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        importance = self.model.get_score(importance_type=importance_type)

        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.save_model(path)
        self.logger.info(f"Model saved to: {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to model file
        """
        self.model = xgb.Booster()
        self.model.load_model(path)
        self.logger.info(f"Model loaded from: {path}")


class LightGBMTrainer:
    """Train LightGBM model with class imbalance handling."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> lgb.Booster:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Optional custom parameters

        Returns:
            Trained LightGBM Booster
        """
        self.logger.info("Training LightGBM model...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Calculate scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        self.logger.info(f"Class distribution: {pos_count:,} positive, {neg_count:,} negative")
        self.logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        # Get LightGBM parameters
        if params is None:
            params = self.config['model']['lightgbm_params'].copy()
        else:
            params = params.copy()

        params['scale_pos_weight'] = scale_pos_weight

        self.logger.info(f"Parameters: {params}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=10)
            ]
        )

        self.logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.save_model(path)
        self.logger.info(f"Model saved to: {path}")

    def load_model(self, path: str) -> None:
        """Load model."""
        self.model = lgb.Booster(model_file=path)
        self.logger.info(f"Model loaded from: {path}")
