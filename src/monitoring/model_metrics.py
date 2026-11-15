"""
Model Performance Metrics

Track model-specific performance indicators.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class ModelMetrics:
    """Track model performance metrics."""

    predictions: list[float] = None
    ground_truth: list[float] = None

    def __post_init__(self):
        """Initialize lists if None."""
        if self.predictions is None:
            self.predictions = []
        if self.ground_truth is None:
            self.ground_truth = []

    def add_prediction(self, prediction: float, truth: float = None):
        """
        Record a prediction.

        Args:
            predictions: Model prediction
            truth: Ground truth value (if available)
        """
        self.predictions.append(prediction)
        if truth is not None:
            self.ground_truth.append(truth)


    def calculate_accuracy(self) -> float:
        """
        Calculate prediction accuracy

        Returns:
            Accuracy percentage (0-100)
        """
        if not self.predictions or not self.ground_truth:
            return 0.0

        if len(self.predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth lenghts don't match")

        correct = sum(
            1 for pred, truth in zip(self.predictions, self.ground_truth)
            if abs(pred -truth) < 0.01
        )

        return (correct / len(self.predictions)) * 100


    def get_summary(self) -> dict:
        """
        Get metrics summary

        Returns:
            Dictionary of metrics
        """
        return {
            "total_predictions": len(self.predictions),
            "accuracy": self.calculate_accurary() if self.ground_truth else None,
            "mean_prediction": np.mean(self.predictions) if self.predictions else 0,
            "std_prediction": np.std(self.predictions) if self.predictions else 0
        }

# Global metrics instance
model_metrics = ModelMetrics()

