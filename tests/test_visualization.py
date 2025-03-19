import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import altair as alt
from unittest.mock import MagicMock, patch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import sys
import os

# Add the parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization import ModelPerformance


class SimpleModel(nn.Module):
    """A simple model for testing purposes"""

    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


@pytest.fixture
def model():
    """Create a simple model for testing"""
    return SimpleModel(input_dim=5)


@pytest.fixture
def feature_names():
    """Create feature names for testing"""
    return ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]


@pytest.fixture
def device():
    """Return CPU device for testing"""
    return torch.device("cpu")


@pytest.fixture
def model_performance(model, feature_names, device):
    """Create a ModelPerformance instance for testing"""
    return ModelPerformance(model, feature_names, device)


@pytest.fixture
def test_data():
    """Create test data for visualization"""
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 2, size=100)  # Binary labels
    return X, y


@pytest.fixture
def history():
    """Create training history for visualization"""
    # Create mock training history
    history = {
        "train_loss": [0.7, 0.6, 0.5, 0.4, 0.3],
        "eval_metrics": [
            {"loss": 0.65, "accuracy": 0.7, "auc": 0.75},
            {"loss": 0.55, "accuracy": 0.75, "auc": 0.8},
            {"loss": 0.45, "accuracy": 0.8, "auc": 0.82},
            {"loss": 0.35, "accuracy": 0.85, "auc": 0.87},
            {"loss": 0.25, "accuracy": 0.9, "auc": 0.92},
        ],
    }
    return history


def test_initialization(model_performance, model, feature_names, device):
    """Test that ModelPerformance initializes correctly"""
    assert model_performance.model == model
    assert model_performance.feature_names == feature_names
    assert model_performance.device == device


def test_get_predictions(model_performance, test_data):
    """Test the _get_predictions method"""
    X, _ = test_data
    predictions = model_performance._get_predictions(X)
    
    # Check predictions shape and type
    assert predictions.shape == (100, 1)
    assert isinstance(predictions, np.ndarray)
    # Check that predictions are between 0 and 1 (after sigmoid)
    assert np.all((predictions >= 0) & (predictions <= 1))


@patch('mlflow.log_artifact')
@patch('tempfile.NamedTemporaryFile')
def test_log_dashboard(mock_temp_file, mock_log_artifact, model_performance):
    """Test the _log_dashboard method"""
    # Create a mock chart
    mock_chart = MagicMock(spec=alt.Chart)
    mock_chart.to_html.return_value = "<html>Test Chart</html>"
    
    # Mock the temporary file
    mock_file = MagicMock()
    mock_file.name = "/tmp/test_chart.html"
    mock_temp_file.return_value.__enter__.return_value = mock_file
    
    # Call the method
    model_performance._log_dashboard(mock_chart, epoch=5)
    
    # Check if MLflow was called correctly
    mock_log_artifact.assert_called_once_with(
        mock_file.name, "visualization/performance_dashboard_epoch_5"
    )


@patch('shap.DeepExplainer')
def test_get_shap_summary(mock_deep_explainer, model_performance, test_data):
    """Test the _get_shap_summary method"""
    X, _ = test_data
    
    # Mock SHAP explainer
    mock_explainer_instance = MagicMock()
    mock_deep_explainer.return_value = mock_explainer_instance
    mock_explainer_instance.shap_values.return_value = [np.random.randn(100, 5)]
    
    # Call the method
    chart = model_performance._get_shap_summary(X, n_samples=100)
    
    # Verify that the result is an Altair chart
    assert isinstance(chart, alt.Chart)
    
    # Check if the mocked explainer was called
    mock_deep_explainer.assert_called_once()
    mock_explainer_instance.shap_values.assert_called_once()


def test_get_feature_importance(model_performance, test_data):
    """Test the _get_feature_importance method"""
    X, y = test_data
    
    with patch('sklearn.inspection.permutation_importance') as mock_perm_imp:
        # Mock return value for permutation importance
        mock_result = MagicMock()
        mock_result.importances_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_result.importances_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        mock_perm_imp.return_value = mock_result
        
        # Call the method
        chart = model_performance._get_feature_importance(X, y, n_repeats=5)
        
        # Verify that the result is an Altair chart
        assert isinstance(chart, alt.Chart)
        
        # Check if permutation importance was called
        mock_perm_imp.assert_called_once()


def test_get_learning_curves(model_performance, history):
    """Test the _get_learning_curves method"""
    chart = model_performance._get_learning_curves(history)
    
    # Verify that the result is an Altair chart
    assert isinstance(chart, alt.Chart)


def test_get_metric_evolution(model_performance, history):
    """Test the _get_metric_evolution method"""
    chart = model_performance._get_metric_evolution(history)
    
    # Verify that the result is an Altair chart
    assert isinstance(chart, alt.Chart)


def test_get_roc_curve(model_performance):
    """Test the _get_roc_curve method"""
    # Create mock data for ROC curve
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([[0.1], [0.4], [0.35], [0.8]])
    
    chart = model_performance._get_roc_curve(y_true, y_pred.ravel())
    
    # Verify that the result is an Altair chart
    assert isinstance(chart, alt.Chart)


def test_get_pr_curve(model_performance):
    """Test the _get_pr_curve method"""
    # Create mock data for PR curve
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([[0.1], [0.4], [0.35], [0.8]])
    
    chart = model_performance._get_pr_curve(y_true, y_pred.ravel())
    
    # Verify that the result is an Altair chart
    assert isinstance(chart, alt.Chart)


def test_get_prediction_distribution(model_performance, test_data):
    """Test the _get_prediction_distribution method"""
    X, _ = test_data
    
    # Patch _get_predictions to return mock values
    with patch.object(
        model_performance, '_get_predictions', 
        return_value=np.random.rand(100, 1)
    ):
        chart = model_performance._get_prediction_distribution(X)
        
        # Verify that the result is an Altair chart
        assert isinstance(chart, alt.Chart)


@patch('mlflow.log_artifact')
def test_create_performance_dashboard(mock_log_artifact, model_performance, test_data, history):
    """Test the create_performance_dashboard method"""
    X, y = test_data
    
    # Mock all the visualization methods to avoid actual computation
    with patch.multiple(
        model_performance,
        _get_feature_importance=MagicMock(return_value=alt.Chart()),
        _get_shap_summary=MagicMock(return_value=alt.Chart()),
        _get_learning_curves=MagicMock(return_value=alt.Chart()),
        _get_metric_evolution=MagicMock(return_value=alt.Chart()),
        _get_predictions=MagicMock(return_value=np.random.rand(100, 1)),
        _get_roc_curve=MagicMock(return_value=alt.Chart()),
        _get_pr_curve=MagicMock(return_value=alt.Chart()),
        _get_prediction_distribution=MagicMock(return_value=alt.Chart()),
        _log_dashboard=MagicMock()
    ):
        # Call the method
        model_performance.create_performance_dashboard(X, y, history, epoch=5)
        
        # Check that all methods were called
        model_performance._get_feature_importance.assert_called_once()
        model_performance._get_shap_summary.assert_called_once()
        model_performance._get_learning_curves.assert_called_once()
        model_performance._get_metric_evolution.assert_called_once()
        model_performance._get_predictions.assert_called_once()
        model_performance._get_roc_curve.assert_called_once()
        model_performance._get_pr_curve.assert_called_once()
        model_performance._get_prediction_distribution.assert_called_once()
        model_performance._log_dashboard.assert_called_once()
