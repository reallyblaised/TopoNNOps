import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))
import main
from models import UnconstrainedNet

@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    return OmegaConf.create({
        "architecture": "unconstrained",
        "model": {
            "input_dim": 5,
            "layer_dims": [10, 5],
            "dropout_rate": 0.1,
            "batch_norm": False,
            "activation_fn": "relu",
            "l1_factor": 0.01,
            "residual": False,
            "init_method": "xavier_uniform"
        },
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5,
            "beta1": 0.9,
            "beta2": 0.999
        },
        "scheduler": {
            "name": "cosine",
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
            "min_lr": 1e-6
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 100,
            "loss_fn": "bce_with_logits",
            "scale_factor": 0.1,
            "ratio": 0.01,
            "grad_clip_val": 1.0,
            "early_stopping_patience": 10,
            "use_mixed_precision": False
        },
        "metrics": {
            "compute": {
                "roc_auc": True,
                "accuracy": True,
                "precision": True,
                "recall": True,
                "f1": True
            },
            "threshold": 0.5
        },
        "paths": {
            "train_data": "/path/to/train_data.pkl",
            "test_data": "/path/to/test_data.pkl"
        },
        "mlflow": {
            "experiment_name": "test_experiment"
        }
    })

def test_get_model(basic_config):
    """Test model creation from configuration."""
    model = main.get_model(basic_config, input_dim=5)
    
    assert isinstance(model, UnconstrainedNet)
    assert hasattr(model, 'layers')
    
    # Verify model structure based on config
    assert len(model.layers) > 0
    
    # Test forward pass
    x = torch.randn(2, 5)
    output = model(x)
    assert output.shape == (2, 1)  # Binary classification output
    
    # Test unsupported architecture
    invalid_config = OmegaConf.create({
        "architecture": "unsupported_architecture",
        "model": {}
    })
    
    with pytest.raises(ValueError, match=r"Unsupported model architecture.*"):
        main.get_model(invalid_config, input_dim=5)

def test_get_optimizer(basic_config):
    """Test optimizer creation from configuration."""
    model = main.get_model(basic_config, input_dim=5)
    optimizer = main.get_optimizer(basic_config, model)
    
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == basic_config.optimizer.lr
    assert optimizer.defaults['weight_decay'] == basic_config.optimizer.weight_decay
    assert optimizer.defaults['betas'] == (basic_config.optimizer.beta1, basic_config.optimizer.beta2)
    
    # Test SGD optimizer
    sgd_config = OmegaConf.create({
        "optimizer": {
            "name": "sgd",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4
        }
    })
    
    optimizer = main.get_optimizer(sgd_config, model)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == sgd_config.optimizer.lr
    assert optimizer.defaults['weight_decay'] == sgd_config.optimizer.weight_decay
    assert optimizer.defaults['momentum'] == sgd_config.optimizer.momentum
    
    # Test unsupported optimizer
    invalid_config = OmegaConf.create({
        "optimizer": {
            "name": "unsupported_optimizer"
        }
    })
    
    with pytest.raises(ValueError, match=r"Unsupported optimizer.*"):
        main.get_optimizer(invalid_config, model)

def test_get_scheduler(basic_config):
    """Test scheduler creation from configuration."""
    model = main.get_model(basic_config, input_dim=5)
    optimizer = main.get_optimizer(basic_config, model)
    scheduler = main.get_scheduler(basic_config, optimizer)
    
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    # Test ReduceLROnPlateau scheduler
    plateau_config = OmegaConf.create({
        "scheduler": {
            "name": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
            "min_lr": 1e-6
        }
    })
    
    scheduler = main.get_scheduler(plateau_config, optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == plateau_config.scheduler.mode
    assert scheduler.factor == plateau_config.scheduler.factor
    assert scheduler.patience == plateau_config.scheduler.patience
    assert scheduler.min_lrs[0] == plateau_config.scheduler.min_lr
    
    # Test no scheduler case
    no_scheduler_config = OmegaConf.create({})
    scheduler = main.get_scheduler(no_scheduler_config, optimizer)
    assert scheduler is None

def test_get_criterion(basic_config):
    """Test loss function creation from configuration."""
    criterion = main.get_criterion(basic_config)
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    
    # Test focal loss
    focal_config = OmegaConf.create({
        "training": {
            "loss_fn": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0
        }
    })
    
    with patch('main.FocalLoss') as mock_focal_loss:
        main.get_criterion(focal_config)
        mock_focal_loss.assert_called_once_with(
            alpha=focal_config.training.focal_alpha,
            gamma=focal_config.training.focal_gamma
        )
    
    # Test unsupported loss function
    invalid_config = OmegaConf.create({
        "training": {
            "loss_fn": "unsupported_loss"
        }
    })
    
    with pytest.raises(ValueError, match=r"Unsupported loss function.*"):
        main.get_criterion(invalid_config)

@patch('main.mlflow')
@patch('main.torch.device')
@patch('main.LHCbMCModule')
@patch('main.Trainer')
def test_main_function(mock_trainer, mock_data_module, mock_device, mock_mlflow, basic_config):
    """Test the main function with mocked components."""
    # Setup mocks
    mock_device_instance = MagicMock()
    mock_device.return_value = mock_device_instance
    
    mock_model = MagicMock()
    mock_model.get_n_features.return_value = 5
    
    mock_data_instance = MagicMock()
    mock_data_instance.feature_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    mock_data_module.return_value = mock_data_instance
    
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance
    mock_trainer_instance.train.return_value = {
        'train_loss': [0.5, 0.4, 0.3],
        'eval_metrics': [{'loss': 0.5, 'roc_auc': 0.7}, {'loss': 0.4, 'roc_auc': 0.8}, {'loss': 0.3, 'roc_auc': 0.9}],
        'epoch_times': [1.0, 1.1, 1.2],
        'learning_rates': [0.001, 0.001, 0.0001]
    }
    
    # Run main with patched hydra.main
    with patch('main.hydra.main') as mock_hydra_main:
        mock_hydra_main.side_effect = lambda config_path, config_name: (lambda func: func(basic_config))
        main.main()
    
    # Verify MLflow was set up correctly
    mock_mlflow.set_experiment.assert_called_once_with(basic_config.mlflow.experiment_name)
    mock_mlflow.start_run.assert_called_once()
    
    # Verify data module was initialized and set up
    mock_data_module.assert_called_once_with(basic_config.paths.train_data, basic_config.paths.test_data)
    mock_data_instance.setup.assert_called_once_with(
        batch_size=basic_config.training.batch_size,
        scale_factor=basic_config.training.scale_factor,
        ratio=basic_config.training.ratio
    )
    
    # Verify trainer was created and train was called
    mock_trainer_instance.train.assert_called_once()
    train_args, train_kwargs = mock_trainer_instance.train.call_args
    assert 'train_loader' in train_kwargs
    assert 'val_loader' in train_kwargs
    assert train_kwargs['num_epochs'] == basic_config.training.num_epochs
    assert 'metrics_cfg' in train_kwargs
    assert train_kwargs['early_stopping_patience'] == basic_config.training.early_stopping_patience
    
    # Verify final metrics were logged
    mock_mlflow.log_metric.assert_any_call('final_loss', 0.3)
    mock_mlflow.log_metric.assert_any_call('final_roc_auc', 0.9)

@patch('main.mlflow')
@patch('main.torch.device')
@patch('main.LHCbMCModule')
def test_main_function_error_handling(mock_data_module, mock_device, mock_mlflow, basic_config, caplog):
    """Test the main function error handling."""
    # Setup mocks to raise an exception
    mock_data_module.side_effect = ValueError("Test error")
    
    # Run main with patched hydra.main expecting exception
    with patch('main.hydra.main') as mock_hydra_main:
        mock_hydra_main.side_effect = lambda config_path, config_name: (lambda func: func(basic_config))
        with pytest.raises(ValueError, match="Test error"):
            main.main()
    
    # Verify proper error logging
    assert "An error occurred" in caplog.text

if __name__ == "__main__":
    pytest.main(["-v", __file__])
