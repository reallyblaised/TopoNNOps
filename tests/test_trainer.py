import pytest
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add parent directory to path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))
from trainer import Trainer

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_l1_loss(self):
        return torch.tensor(0.01)

# Create mock classes for visualization modules to avoid actual visualization during tests
class MockModelPerformance:
    def __init__(self, *args, **kwargs):
        pass
    
    def create_performance_dashboard(self, *args, **kwargs):
        pass

class MockWeightVisualizer:
    def __init__(self, *args, **kwargs):
        pass
    
    def create_weight_dashboard(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_visualization_modules(monkeypatch):
    """Mock visualization modules to avoid actual visualization during tests."""
    monkeypatch.setattr('trainer.ModelInterpretability', MockModelPerformance)
    monkeypatch.setattr('trainer.WeightVisualizer', MockWeightVisualizer)

@pytest.fixture
def trainer_components():
    """Create components needed for trainer initialization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and training components
    input_dim = 5
    model = SimpleModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    feature_names = [f'feature_{i}' for i in range(input_dim)]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    return {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'feature_names': feature_names,
        'scheduler': scheduler,
        'input_dim': input_dim
    }

@pytest.fixture
def trainer(trainer_components, mock_visualization_modules):
    """Create a trainer instance for testing."""
    return Trainer(
        model=trainer_components['model'],
        optimizer=trainer_components['optimizer'],
        criterion=trainer_components['criterion'],
        device=trainer_components['device'],
        feature_names=trainer_components['feature_names'],
        scheduler=trainer_components['scheduler'],
        grad_clip_val=1.0,
        use_mixed_precision=False
    )

@pytest.fixture
def dummy_data_loaders(trainer_components):
    """Create dummy data loaders for training and validation."""
    input_dim = trainer_components['input_dim']
    batch_size = 4
    
    # Create dummy data
    X_train = torch.randn(batch_size * 3, input_dim)
    y_train = torch.randint(0, 2, (batch_size * 3,), dtype=torch.float32)
    X_val = torch.randn(batch_size * 2, input_dim)
    y_val = torch.randint(0, 2, (batch_size * 2,), dtype=torch.float32)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return {'train': train_loader, 'val': val_loader}

@pytest.fixture
def metrics_config():
    """Create metrics configuration for testing."""
    return OmegaConf.create({
        'compute': {
            'roc_auc': True,
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True
        },
        'threshold': 0.5
    })

def test_trainer_initialization(trainer):
    """Test that trainer initializes correctly."""
    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.model, nn.Module)
    assert trainer.grad_clip_val == 1.0
    assert trainer.use_mixed_precision == False
    assert trainer.scaler is None

def test_train_step(trainer, dummy_data_loaders):
    """Test the training step function."""
    # Get a batch from the training data loader
    X_batch, y_batch = next(iter(dummy_data_loaders['train']))
    
    # Run a training step
    loss, outputs = trainer._train_step(X_batch, y_batch)
    
    # Check that loss and outputs have expected types
    assert isinstance(loss, float)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape[0] == X_batch.shape[0]
    assert outputs.shape[1] == 1  # Binary classification task

def test_evaluate(trainer, dummy_data_loaders, metrics_config):
    """Test the evaluation function."""
    # Run evaluation
    eval_metrics = trainer.evaluate(
        dummy_data_loaders['val'],
        metrics_config
    )
    
    # Check that evaluation returns expected metrics
    assert 'loss' in eval_metrics
    assert 'roc_auc' in eval_metrics
    assert 'accuracy' in eval_metrics
    assert 'precision' in eval_metrics
    assert 'recall' in eval_metrics
    assert 'f1' in eval_metrics

def test_train_with_early_stopping(trainer, dummy_data_loaders, metrics_config, monkeypatch):
    """Test the training loop with early stopping."""
    # Mock _log_epoch_info and _generate_visualizations to avoid actual logging
    monkeypatch.setattr(trainer, '_log_epoch_info', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, '_generate_visualizations', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, '_save_checkpoint', lambda *args, **kwargs: None)
    
    # Run training with early stopping
    history = trainer.train(
        train_loader=dummy_data_loaders['train'],
        val_loader=dummy_data_loaders['val'],
        num_epochs=5,
        metrics_cfg=metrics_config,
        early_stopping_patience=2
    )
    
    # Check that history contains expected keys
    assert 'train_loss' in history
    assert 'eval_metrics' in history
    assert 'epoch_times' in history
    assert 'learning_rates' in history
    
    # Check that lists have expected lengths
    assert len(history['train_loss']) <= 5  # Should be <= num_epochs
    assert len(history['eval_metrics']) <= 5
    assert len(history['epoch_times']) <= 5
    assert len(history['learning_rates']) <= 5

def test_train_with_scheduler(trainer, dummy_data_loaders, metrics_config, monkeypatch):
    """Test that learning rate scheduler is applied correctly."""
    # Mock _log_epoch_info and _generate_visualizations to avoid actual logging
    monkeypatch.setattr(trainer, '_log_epoch_info', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, '_generate_visualizations', lambda *args, **kwargs: None)
    
    # Get initial learning rate
    initial_lr = trainer.scheduler.get_last_lr()[0]
    
    # Run training for 11 epochs to see scheduler effect (step size is 10)
    history = trainer.train(
        train_loader=dummy_data_loaders['train'],
        val_loader=dummy_data_loaders['val'],
        num_epochs=11,
        metrics_cfg=metrics_config,
        early_stopping_patience=None
    )
    
    # Check that learning rate changed after 10 epochs
    assert history['learning_rates'][0] == initial_lr
    assert history['learning_rates'][-1] < initial_lr

def test_mixed_precision(trainer_components, mock_visualization_modules, dummy_data_loaders, metrics_config, monkeypatch):
    """Test training with mixed precision if GPU is available."""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping mixed precision test")
    
    # Create trainer with mixed precision
    trainer_mp = Trainer(
        model=trainer_components['model'],
        optimizer=trainer_components['optimizer'],
        criterion=trainer_components['criterion'],
        device=trainer_components['device'],
        feature_names=trainer_components['feature_names'],
        scheduler=trainer_components['scheduler'],
        grad_clip_val=1.0,
        use_mixed_precision=True
    )
    
    # Mock _log_epoch_info and _generate_visualizations to avoid actual logging
    monkeypatch.setattr(trainer_mp, '_log_epoch_info', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_mp, '_generate_visualizations', lambda *args, **kwargs: None)
    
    # Check that scaler is initialized
    assert trainer_mp.scaler is not None
    
    # Run a single epoch
    history = trainer_mp.train(
        train_loader=dummy_data_loaders['train'],
        val_loader=dummy_data_loaders['val'],
        num_epochs=1,
        metrics_cfg=metrics_config
    )
    
    # Check that history is created
    assert len(history['train_loss']) == 1
