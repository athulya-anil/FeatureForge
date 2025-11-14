"""Model evaluation module with comprehensive metrics and visualizations."""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import os


class ModelEvaluator:
    """Comprehensive model evaluation for CTR prediction."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize model evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities
            dataset_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating model on {dataset_name} set...")

        metrics = {}

        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # AUC metrics
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)

        # Log loss
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Log metrics
        self._log_metrics(metrics, dataset_name)

        return metrics

    def _log_metrics(self, metrics: Dict[str, Any], dataset_name: str) -> None:
        """
        Log all metrics.

        Args:
            metrics: Dictionary of metrics
            dataset_name: Name of dataset
        """
        self.logger.info("=" * 60)
        self.logger.info(f"MODEL EVALUATION RESULTS - {dataset_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        self.logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        self.logger.info(f"AUC-PR:    {metrics['auc_pr']:.4f}")
        self.logger.info(f"Log Loss:  {metrics['log_loss']:.4f}")
        self.logger.info("=" * 60)

        # Log confusion matrix
        cm = metrics['confusion_matrix']
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"  TN: {cm[0][0]:,}  FP: {cm[0][1]:,}")
        self.logger.info(f"  FN: {cm[1][0]:,}  TP: {cm[1][1]:,}")
        self.logger.info("=" * 60)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path: Path to save plot
            normalize: Whether to normalize values
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=['No Click', 'Click'],
            yticklabels=['No Click', 'Click']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to: {save_path}")

        plt.show()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot Precision-Recall curve (important for imbalanced data).

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {auc_pr:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curve saved to: {save_path}")

        plt.show()

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of predicted probabilities by true class.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))

        # Plot distributions
        plt.hist(
            y_pred_proba[y_true == 0],
            bins=50,
            alpha=0.6,
            label='No Click (0)',
            color='blue'
        )
        plt.hist(
            y_pred_proba[y_true == 1],
            bins=50,
            alpha=0.6,
            label='Click (1)',
            color='red'
        )

        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction distribution saved to: {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to plot
            save_path: Path to save plot
        """
        # Get top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to: {save_path}")

        plt.show()

    def create_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        feature_importance: Optional[pd.DataFrame] = None,
        dataset_name: str = "Test",
        save_prefix: str = "model"
    ) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report with all plots.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            feature_importance: Feature importance DataFrame
            dataset_name: Name of dataset
            save_prefix: Prefix for saved files

        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Creating comprehensive evaluation report for {dataset_name}...")

        # Evaluate metrics
        metrics = self.evaluate(y_true, y_pred, y_pred_proba, dataset_name)

        # Create plots
        save_dir = os.path.join(self.output_dir, save_prefix)
        os.makedirs(save_dir, exist_ok=True)

        # Confusion matrix
        cm_path = os.path.join(save_dir, f"{save_prefix}_confusion_matrix.png")
        self.plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)

        # ROC curve
        roc_path = os.path.join(save_dir, f"{save_prefix}_roc_curve.png")
        self.plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)

        # Precision-Recall curve
        pr_path = os.path.join(save_dir, f"{save_prefix}_pr_curve.png")
        self.plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_path)

        # Prediction distribution
        dist_path = os.path.join(save_dir, f"{save_prefix}_prediction_dist.png")
        self.plot_prediction_distribution(y_true, y_pred_proba, save_path=dist_path)

        # Feature importance (if provided)
        if feature_importance is not None:
            fi_path = os.path.join(save_dir, f"{save_prefix}_feature_importance.png")
            self.plot_feature_importance(feature_importance, save_path=fi_path)

            # Save feature importance CSV
            fi_csv = os.path.join(save_dir, f"{save_prefix}_feature_importance.csv")
            feature_importance.to_csv(fi_csv, index=False)
            self.logger.info(f"Feature importance saved to: {fi_csv}")

        # Save metrics to file
        metrics_path = os.path.join(save_dir, f"{save_prefix}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Model Evaluation Report - {dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
            f.write(f"AUC-PR:    {metrics['auc_pr']:.4f}\n")
            f.write(f"Log Loss:  {metrics['log_loss']:.4f}\n\n")

            cm = metrics['confusion_matrix']
            f.write("Confusion Matrix:\n")
            f.write(f"  TN: {cm[0][0]:,}  FP: {cm[0][1]:,}\n")
            f.write(f"  FN: {cm[1][0]:,}  TP: {cm[1][1]:,}\n")

        self.logger.info(f"Metrics saved to: {metrics_path}")
        self.logger.info(f"Evaluation report complete. Files saved to: {save_dir}")

        return metrics

    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model results.

        Args:
            results: Dictionary of {model_name: metrics_dict}
            save_path: Path to save comparison plot

        Returns:
            DataFrame with comparison
        """
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc'],
                'AUC-PR': metrics['auc_pr']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Plot comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            comparison_df.plot(
                x='Model',
                y=metric,
                kind='bar',
                ax=ax,
                legend=False,
                color='skyblue'
            )
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison saved to: {save_path}")

        plt.show()

        return comparison_df
