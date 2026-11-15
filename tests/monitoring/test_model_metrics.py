"""Tests for model metrics"""

import pytest
from src.monitoring.model_metrics import ModelMetrics


def test_model_metrics_initialization():
    """Test metrics initialization."""
    metrics = ModelMetrics()
    assert metrics.predictions == []
    assert metrics.ground_truth == []


def test_add_prediction():
    """Test adding predictions"""
    metrics = ModelMetrics()
    metrics.add_prediction(0.95, 1.0)
    metrics.add_prediction(0.87, 1.0)

    assert len(metrics.predictions) == 2
    assert len(metrics.ground_truth) == 2

def test_calculate_accuracy():
    """Test accuracy calculation."""
    metrics = ModelMetrics()
    metrics.add_prediction(1.0, 1.0)
    metrics.add_prediction(1.0, 1.0)
    metrics.add_prediction(1.0, 1.0)

    accuracy = metrics.calculate_accuracy()
    assert accuracy == pytest.approx(66.67, rel=0.1)

def test_get_summary():
    """Test metrics summary"""
    metrics = ModelMetrics()
    metrics.add_prediction(0.9)
    metrics.add_prediction(0.8)

    summary = metrics.get_summary()
    assert summary["total_predictions"] == 2
    assert "mean_prediction" in summary

