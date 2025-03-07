import pytest
import torch
import numpy as np
import mlflow
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Mock MLflow functionality for all tests to avoid actual MLflow operations.
    This fixture is automatically used for all tests.
    """
    with patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.log_dict'), \
         patch('mlflow.log_text'):
        yield

@pytest.fixture
def get_device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def set_random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

@pytest.fixture
def create_binary_classification_dataset(input_dim=5, n_samples=100):
    """Create a simple binary classification dataset."""
    X = np.random.randn(n_samples, input_dim)
    # Create simple decision boundary: positive if sum of features is positive
    y = (X.sum(axis=1) > 0).astype(np.float32)
    return X, y
