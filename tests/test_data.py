import os
import pandas as pd
import torch
import yaml
from pathlib import Path
import sys
import pytest

# Add parent directory to path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))
from data import LHCbMCModule

@pytest.fixture
def config_data():
    """Create test configuration data."""
    return {
        'features': {
            'TwoBody': {
                'min_PT_final_state_tracks': 1,
                'sum_PT_final_state_tracks': 1
            },
            'ThreeBody': {
                'min_FS_IPCHI2_OWNPV': 1,
                'sum_PT_TRACK12': 1
            }
        }
    }

@pytest.fixture
def config_file(tmp_path, config_data):
    """Create a temporary config file with test features."""
    config_dir = tmp_path / 'config'
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / 'test_features.yml'
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_file

@pytest.fixture
def sample_data():
    """Create sample training and testing data."""
    train_data = pd.DataFrame({
        'min_PT_final_state_tracks': [0.5, 1.2, 0.8, 1.5],
        'sum_PT_final_state_tracks': [2.0, 3.5, 2.2, 4.0],
        'min_FS_IPCHI2_OWNPV': [0.1, 0.2, 0.15, 0.3],
        'sum_PT_TRACK12': [1.8, 2.5, 1.9, 3.0],
        'channel': ['minbias', 'signal', 'minbias', 'signal'],
        'class_label': [0, 1, 0, 1]
    })
    
    test_data = pd.DataFrame({
        'min_PT_final_state_tracks': [0.6, 1.3],
        'sum_PT_final_state_tracks': [2.1, 3.6],
        'min_FS_IPCHI2_OWNPV': [0.12, 0.22],
        'sum_PT_TRACK12': [1.85, 2.6],
        'channel': ['minbias', 'signal'],
        'class_label': [0, 1]
    })
    
    return train_data, test_data

@pytest.fixture
def data_files(tmp_path, sample_data):
    """Create sample data files."""
    train_data, test_data = sample_data
    train_path = tmp_path / 'train_data.pkl'
    test_path = tmp_path / 'test_data.pkl'
    
    train_data.to_pickle(train_path)
    test_data.to_pickle(test_path)
    
    return train_path, test_path

@pytest.fixture
def lhcb_module(monkeypatch, config_file, data_files):
    """Create LHCbMCModule instance with patched config path."""
    train_path, test_path = data_files
    
    # Patch the configuration path method for testing
    monkeypatch.setattr(LHCbMCModule, '_get_features_config_path', lambda filename: config_file)
    
    module = LHCbMCModule(train_path, test_path)
    return module

def test_feature_config(monkeypatch, config_file):
    """Test feature configuration loading."""
    monkeypatch.setattr(LHCbMCModule, '_get_features_config_path', lambda filename: config_file)
    
    two_body_features = LHCbMCModule.feature_config(model="TwoBody", feature_config_file="test_features.yml")
    three_body_features = LHCbMCModule.feature_config(model="ThreeBody", feature_config_file="test_features.yml")
    
    assert len(two_body_features) == 2
    assert len(three_body_features) == 2
    assert 'min_PT_final_state_tracks' in two_body_features
    assert 'min_FS_IPCHI2_OWNPV' in three_body_features

def test_get_n_features(monkeypatch, config_file):
    """Test counting features."""
    monkeypatch.setattr(LHCbMCModule, '_get_features_config_path', lambda filename: config_file)
    
    n_two_body = LHCbMCModule.get_n_features(model="TwoBody")
    n_three_body = LHCbMCModule.get_n_features(model="ThreeBody")
    
    assert n_two_body == 2
    assert n_three_body == 2

def test_process_training_data(lhcb_module, sample_data):
    """Test processing of training data."""
    train_data, _ = sample_data
    processed_data = lhcb_module._process_training_data(train_data, scale_factor=1.0, ratio=0.5)
    
    # Check that we have both minbias and signal samples
    assert (processed_data['channel'] == 'minbias').any()
    assert (processed_data['channel'] == 'signal').any()

def test_setup(lhcb_module):
    """Test setup of data loaders."""
    lhcb_module.setup(batch_size=2, scale_factor=1.0, ratio=0.5)
    
    # Check that loaders are created
    assert lhcb_module.train_loader is not None
    assert lhcb_module.test_loader is not None
    
    # Check input dimension
    assert lhcb_module.input_dim == 2
    
    # Check that we can iterate through the loaders
    X_batch, y_batch = next(iter(lhcb_module.train_loader))
    assert isinstance(X_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)
    assert X_batch.shape[1] == 2  # 2 features
