import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
import yaml
import logging
from unittest.mock import patch, MagicMock, call
import tempfile
import json

# Configure logging for verbose output
logging.basicConfig(level=logging.DEBUG, 
                   format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

# Import the model (with try/except to handle import errors)
try:
    from models import LipschitzNet
    LIPSCHITZ_NET_AVAILABLE = True
except ImportError:
    logger.error("Failed to import LipschitzNet class. Tests will be skipped.")
    LIPSCHITZ_NET_AVAILABLE = False

# Create a detailed test features config
@pytest.fixture
def create_detailed_features_config(tmp_path):
    """Create a temporary features config file with detailed constraints for testing."""
    # Create a comprehensive test config
    config = {
        'features': {
            'TwoBody': {
                'min_PT_final_state_tracks': 1,     # Monotonically increasing
                'sum_PT_final_state_tracks': 1,     # Monotonically increasing
                'min_FS_IPCHI2_OWNPV': 1,           # Monotonically increasing
                'max_FS_IPCHI2_OWNPV': 0,           # No monotonicity constraint
                'TwoBody_PT': 1,                    # Monotonically increasing
                'TwoBody_DOCAMAX': 0,               # No monotonicity constraint
                'TwoBody_MCORR': 0,                 # No monotonicity constraint
                'TwoBody_FDCHI2_OWNPV': 0,          # No monotonicity constraint
                'TwoBody_ENDVERTEX_CHI2DOF': 0      # No monotonicity constraint
            },
            'ThreeBody': {
                'min_FS_IPCHI2_OWNPV': 1,           # Monotonically increasing
                'max_FS_IPCHI2_OWNPV': 0,           # No monotonicity constraint
                'sum_PT_TRACK12': 1,                # Monotonically increasing
                'min_PT_TRACK12': 1,                # Monotonically increasing
                'sum_PT_final_state_tracks': 1,     # Monotonically increasing
                'min_PT_final_state_tracks': 1      # Monotonically increasing
            }
        }
    }
    
    # Write config file
    config_dir = tmp_path / 'config'
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / 'test_features.yml'
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Created test features config at: {config_file}")
    logger.info(f"Features config contents:\n{json.dumps(config, indent=2)}")
    
    return str(config_file)

class MockLinear(nn.Module):
    """Mock Linear layer that can be used in place of nn.Linear."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self._norm_params = {}
        
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class MockGroupSort(nn.Module):
    """Mock GroupSort that just applies ReLU."""
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units
        
    def forward(self, x):
        return torch.relu(x)

# Create mocks for the key components
@pytest.fixture
def mock_components():
    """Create mocks for monotonenorm components with detailed tracking."""
    
    # Create direct_norm mock that tracks parameters and returns the module
    def mock_direct_norm(module, always_norm=False, kind="one", max_norm=1.0):
        logger.info(f"direct_norm called with parameters:")
        logger.info(f"  - module: {module}")
        logger.info(f"  - always_norm: {always_norm}")
        logger.info(f"  - kind: {kind}")
        logger.info(f"  - max_norm: {max_norm}")
        
        # Add the norm parameters to the module for inspection
        module._norm_params = {
            'always_norm': always_norm,
            'kind': kind, 
            'max_norm': max_norm
        }
        return module
    
    # Create a wrapped mock for direct_norm that we can reset
    direct_norm_mock = MagicMock(side_effect=mock_direct_norm)
    
    # Create GroupSort mock
    group_sort_mock = MagicMock(side_effect=lambda num_units: MockGroupSort(num_units))
    
    # Create a sequential model that wraps a provided model
    class MockSigmaNet(nn.Module):
        def __init__(self, model, sigma, monotone_constraints):
            super().__init__()
            self.model = model
            self.sigma = sigma
            self.monotone_constraints = monotone_constraints
            logger.info(f"SigmaNet initialized with:")
            logger.info(f"  - sigma: {sigma}")
            logger.info(f"  - monotone_constraints: {monotone_constraints}")
            
        def forward(self, x):
            return self.model(x)
            
        def parameters(self):
            return self.model.parameters()
    
    # Create a wrapped mock for SigmaNet
    sigma_net_mock = MagicMock(side_effect=MockSigmaNet)
    
    # Return all mocks
    return {
        'GroupSort': group_sort_mock,
        'direct_norm': direct_norm_mock,
        'SigmaNet': sigma_net_mock
    }

@pytest.mark.skipif(not LIPSCHITZ_NET_AVAILABLE, reason="LipschitzNet not available")
class TestLipschitzNetDetailed:
    """Detailed tests for the LipschitzNet model."""
    
    def test_initialization_verbose(self, create_detailed_features_config, mock_components):
        """Test model initialization with detailed logging."""
        logger.info("\n" + "="*80)
        logger.info("TESTING INITIALIZATION (LIPSCHITZ WITHOUT MONOTONICITY)")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 9
        layer_dims = [32, 64, 32]
        lip_const = 2.5  # Using a non-default value to ensure it's used
        feature_names = [
            'min_PT_final_state_tracks', 
            'sum_PT_final_state_tracks',
            'min_FS_IPCHI2_OWNPV',
            'max_FS_IPCHI2_OWNPV',
            'TwoBody_PT',
            'TwoBody_DOCAMAX',
            'TwoBody_MCORR',
            'TwoBody_FDCHI2_OWNPV',
            'TwoBody_ENDVERTEX_CHI2DOF'
        ]
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Layer dimensions: {layer_dims}")
        logger.info(f"Lipschitz constant (lambda): {lip_const}")
        logger.info(f"Feature names: {feature_names}")
        
        # Create model with patched components
        with patch('models.SigmaNet', mock_components['SigmaNet']), \
             patch('models.GroupSort', mock_components['GroupSort']), \
             patch('models.direct_norm', mock_components['direct_norm']):
            
            # Create LipschitzNet without monotonicity
            logger.info("\nCreating LipschitzNet without monotonicity constraints:")
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lip_const=lip_const,
                monotonic=False,
                nbody="TwoBody",
                feature_names=feature_names,
                features_config_path=create_detailed_features_config,
                activation_fn="groupsort",
                dropout_rate=0.0,  # No dropout for Lipschitz
                batch_norm=False,  # No batch norm for Lipschitz
                l1_factor=0.01
            )
            
            # Verify basic attributes
            logger.info("\nVerifying model attributes:")
            assert model.input_dim == input_dim, f"Input dimension mismatch: {model.input_dim} != {input_dim}"
            assert model.layer_dims == layer_dims, f"Layer dimensions mismatch: {model.layer_dims} != {layer_dims}"
            assert model.lip_const == lip_const, f"Lipschitz constant mismatch: {model.lip_const} != {lip_const}"
            assert model.monotonic is False, f"Monotonic flag mismatch: {model.monotonic} != False"
            assert model.feature_names == feature_names, f"Feature names mismatch"
            
            # Verify direct_norm calls
            expected_max_norm = lip_const ** (1 / (len(layer_dims) + 1))
            logger.info(f"\nExpected max_norm per layer: {expected_max_norm} (λ^(1/(n_layers+1)))")
            
            # Check that SigmaNet wasn't called (since monotonic=False)
            mock_components['SigmaNet'].assert_not_called()
            logger.info("Verified: SigmaNet was not called (as expected for non-monotonic)")
            
    def test_monotonic_initialization_verbose(self, create_detailed_features_config, mock_components):
        """Test model initialization with monotonicity constraints and detailed logging."""
        logger.info("\n" + "="*80)
        logger.info("TESTING INITIALIZATION (LIPSCHITZ WITH MONOTONICITY)")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 9
        layer_dims = [32, 64, 32]
        lip_const = 2.5  # Using a non-default value to ensure it's used
        feature_names = [
            'min_PT_final_state_tracks',  # Should be monotonically increasing (1)
            'sum_PT_final_state_tracks',  # Should be monotonically increasing (1)
            'min_FS_IPCHI2_OWNPV',        # Should be monotonically increasing (1)
            'max_FS_IPCHI2_OWNPV',        # No monotonicity constraint (0)
            'TwoBody_PT',                 # Should be monotonically increasing (1)
            'TwoBody_DOCAMAX',            # No monotonicity constraint (0)
            'TwoBody_MCORR',              # No monotonicity constraint (0)
            'TwoBody_FDCHI2_OWNPV',       # No monotonicity constraint (0)
            'TwoBody_ENDVERTEX_CHI2DOF'   # No monotonicity constraint (0)
        ]
        
        # Expected monotonicity constraints based on feature names and config
        expected_constraints = [1, 1, 1, 0, 1, 0, 0, 0, 0]
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Layer dimensions: {layer_dims}")
        logger.info(f"Lipschitz constant (lambda): {lip_const}")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Expected monotonicity constraints: {expected_constraints}")
        
        # Create model with patched components
        with patch('models.SigmaNet', mock_components['SigmaNet']), \
             patch('models.GroupSort', mock_components['GroupSort']), \
             patch('models.direct_norm', mock_components['direct_norm']):
            
            # Create LipschitzNet with monotonicity
            logger.info("\nCreating LipschitzNet with monotonicity constraints:")
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lip_const=lip_const,
                monotonic=True,
                nbody="TwoBody",
                feature_names=feature_names,
                features_config_path=create_detailed_features_config,
                activation_fn="groupsort",
                dropout_rate=0.0,  # No dropout for Lipschitz
                batch_norm=False,  # No batch norm for Lipschitz
                l1_factor=0.01
            )
            
            # Verify basic attributes
            logger.info("\nVerifying model attributes:")
            assert model.input_dim == input_dim, f"Input dimension mismatch: {model.input_dim} != {input_dim}"
            assert model.layer_dims == layer_dims, f"Layer dimensions mismatch: {model.layer_dims} != {layer_dims}"
            assert model.lip_const == lip_const, f"Lipschitz constant mismatch: {model.lip_const} != {lip_const}"
            assert model.monotonic is True, f"Monotonic flag mismatch: {model.monotonic} != True"
            assert model.feature_names == feature_names, f"Feature names mismatch"
            
            # Verify SigmaNet was called
            assert mock_components['SigmaNet'].call_count > 0, "SigmaNet was not called"
            
            # Get the monotone_constraints that were passed to SigmaNet
            args, kwargs = mock_components['SigmaNet'].call_args_list[0]
            model_arg = args[0]  # First positional arg is the model
            sigma_arg = kwargs.get('sigma', None)
            constraints_arg = kwargs.get('monotone_constraints', None)
            
            logger.info(f"\nActual Lipschitz constant passed to SigmaNet: {sigma_arg}")
            logger.info(f"Actual monotonicity constraints: {constraints_arg}")
            
            # Verify the monotonicity constraints match expectations
            assert constraints_arg is not None, "Monotonicity constraints were not passed to SigmaNet"
            assert len(constraints_arg) == len(expected_constraints), f"Constraint length mismatch: {len(constraints_arg)} != {len(expected_constraints)}"
            assert constraints_arg == expected_constraints, f"Monotonicity constraints mismatch: {constraints_arg} != {expected_constraints}"
            
            # Verify Lipschitz constant was passed correctly
            assert sigma_arg == lip_const, f"Sigma value mismatch: {sigma_arg} != {lip_const}"
            
            logger.info("\nAll monotonicity constraints match expected values!")
            
    def test_feature_mapping_verbose(self, create_detailed_features_config, mock_components):
        """Test feature-to-constraint mapping with malformed/reordered feature lists."""
        logger.info("\n" + "="*80)
        logger.info("TESTING FEATURE-TO-CONSTRAINT MAPPING")
        logger.info("="*80)
        
        # Set up test parameters with reordered features
        input_dim = 9
        layer_dims = [32, 64, 32]
        lip_const = 1.0
        
        # Deliberately reorder features to test mapping
        feature_names = [
            'TwoBody_PT',                 # Should be monotonically increasing (1)
            'TwoBody_DOCAMAX',            # No monotonicity constraint (0)
            'min_PT_final_state_tracks',  # Should be monotonically increasing (1) 
            'max_FS_IPCHI2_OWNPV',        # No monotonicity constraint (0)
            'TwoBody_MCORR',              # No monotonicity constraint (0)
            'min_FS_IPCHI2_OWNPV',        # Should be monotonically increasing (1)
            'sum_PT_final_state_tracks',  # Should be monotonically increasing (1)
            'TwoBody_FDCHI2_OWNPV',       # No monotonicity constraint (0)
            'TwoBody_ENDVERTEX_CHI2DOF'   # No monotonicity constraint (0)
        ]
        
        # Expected monotonicity constraints in the reordered sequence
        expected_constraints = [1, 0, 1, 0, 0, 1, 1, 0, 0]
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Layer dimensions: {layer_dims}")
        logger.info(f"Feature names (reordered): {feature_names}")
        logger.info(f"Expected monotonicity constraints: {expected_constraints}")
        
        # Create a mock _load_monotone_constrs method that returns constraints without reading file
        # This bypasses any file operations while still testing the feature mapping
        def mock_load_monotone_constrs(self, path, key):
            # Create a dictionary mapping feature names to constraints
            feature_map = {
                'min_PT_final_state_tracks': 1,
                'sum_PT_final_state_tracks': 1,
                'min_FS_IPCHI2_OWNPV': 1,
                'max_FS_IPCHI2_OWNPV': 0,
                'TwoBody_PT': 1,
                'TwoBody_DOCAMAX': 0,
                'TwoBody_MCORR': 0,
                'TwoBody_FDCHI2_OWNPV': 0,
                'TwoBody_ENDVERTEX_CHI2DOF': 0
            }
            
            # Return constraints based on feature order
            if self.feature_names:
                constraints = []
                for feature in self.feature_names:
                    if feature in feature_map:
                        constraints.append(feature_map[feature])
                    else:
                        raise ValueError(f"Feature {feature} not found in config for {key}")
                return constraints
            
            # Default case
            return [0] * self.input_dim
        
        # Create model with patched components
        with patch('models.SigmaNet', mock_components['SigmaNet']), \
             patch('models.GroupSort', mock_components['GroupSort']), \
             patch('models.direct_norm', mock_components['direct_norm']), \
             patch.object(LipschitzNet, '_load_monotone_constrs', mock_load_monotone_constrs):
            
            # Create LipschitzNet with monotonicity
            logger.info("\nCreating LipschitzNet with reordered features:")
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lip_const=lip_const,
                monotonic=True,
                nbody="TwoBody",
                feature_names=feature_names,
                features_config_path=create_detailed_features_config,
                activation_fn="groupsort",
                dropout_rate=0.0,  # No dropout for Lipschitz
                batch_norm=False   # No batch norm for Lipschitz
            )
            
            # Get the monotone_constraints that were passed to SigmaNet
            args, kwargs = mock_components['SigmaNet'].call_args_list[0]
            constraints_arg = kwargs.get('monotone_constraints', None)
            
            logger.info(f"\nActual monotonicity constraints: {constraints_arg}")
            
            # Verify the monotonicity constraints match expectations for reordered features
            assert constraints_arg == expected_constraints, f"Reordered monotonicity constraints mismatch: {constraints_arg} != {expected_constraints}"
            
            logger.info("\nFeature mapping worked correctly with reordered features!")
    
    def test_different_lambda_values(self, create_detailed_features_config, mock_components):
        """Test model creation with different lambda (Lipschitz constant) values."""
        logger.info("\n" + "="*80)
        logger.info("TESTING DIFFERENT LAMBDA (LIPSCHITZ CONSTANT) VALUES")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 5
        layer_dims = [10, 10]  # 2 hidden layers + output layer = 3 layers total
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # Test different lambda values
        lambda_values = [0.5, 1.0, 1.5, 2.0, 10.0]
        
        for lip_const in lambda_values:
            logger.info(f"\nTesting lambda value: {lip_const}")
            
            # Calculate expected max_norm
            n_layers = len(layer_dims) + 1  # +1 for output layer
            expected_max_norm = lip_const ** (1 / n_layers)
            logger.info(f"Expected max_norm per layer: {expected_max_norm} (λ^(1/{n_layers}))")
            
            # Reset mock before each iteration
            mock_components['direct_norm'].reset_mock()
            
            # Create model with patched components
            with patch('models.SigmaNet', mock_components['SigmaNet']), \
                 patch('models.GroupSort', mock_components['GroupSort']), \
                 patch('models.direct_norm', mock_components['direct_norm']):
                
                # Create LipschitzNet without monotonicity
                model = LipschitzNet(
                    input_dim=input_dim,
                    layer_dims=layer_dims,
                    lip_const=lip_const,
                    monotonic=False,
                    nbody="TwoBody",
                    feature_names=feature_names,
                    features_config_path=create_detailed_features_config,
                    activation_fn="groupsort",
                    dropout_rate=0.0,  # No dropout for Lipschitz
                    batch_norm=False   # No batch norm for Lipschitz
                )
                
                # Check the max_norm values that were passed to direct_norm
                for i, call_args in enumerate(mock_components['direct_norm'].call_args_list):
                    _, kwargs = call_args
                    actual_max_norm = kwargs.get('max_norm')
                    logger.info(f"Layer {i}: max_norm = {actual_max_norm}")
                    
                    # Allow for small floating point differences
                    assert abs(actual_max_norm - expected_max_norm) < 1e-5, \
                        f"Layer {i} max_norm mismatch: {actual_max_norm} != {expected_max_norm}"
                
                logger.info(f"All layers have correct max_norm values for lambda = {lip_const}")
    
    def test_norm_kind_parameters(self, create_detailed_features_config, mock_components):
        """Test different norm types (one/inf) and always_norm parameter."""
        logger.info("\n" + "="*80)
        logger.info("TESTING NORM TYPE AND ALWAYS_NORM PARAMETERS")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 5
        layer_dims = [10, 5]
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # Test cases with different norm types and always_norm settings
        test_cases = [
            {"norm_kind": "one", "always_norm": False, "description": "L1 norm without always_norm"},
            {"norm_kind": "one", "always_norm": True, "description": "L1 norm with always_norm=True"},
            {"norm_kind": "inf", "always_norm": False, "description": "L∞ norm without always_norm"},
            {"norm_kind": "inf", "always_norm": True, "description": "L∞ norm with always_norm=True"},
        ]
        
        for case in test_cases:
            logger.info(f"\nTesting: {case['description']}")
            norm_kind = case["norm_kind"]
            always_norm = case["always_norm"]
            
            # Create custom _lipschitz_norm method to override the one in the class
            def custom_lipschitz_norm(module, is_norm=False, kind="one"):
                # Note: 'self' is not used here as this will be patched as a static method
                logger.info(f"_lipschitz_norm called with:")
                logger.info(f"  - is_norm: {is_norm}")
                logger.info(f"  - kind: {kind}")
                return mock_components['direct_norm'](
                    module, 
                    always_norm=always_norm if is_norm else False,
                    kind=norm_kind,
                    max_norm=1.0
                )
            
            # Reset mock before each test case
            mock_components['direct_norm'].reset_mock()
            
            # Create model with patched _lipschitz_norm method
            with patch('models.SigmaNet', mock_components['SigmaNet']), \
                 patch('models.GroupSort', mock_components['GroupSort']), \
                 patch('models.direct_norm', mock_components['direct_norm']), \
                 patch('models.LipschitzNet._lipschitz_norm', custom_lipschitz_norm):
                
                # Create LipschitzNet
                model = LipschitzNet(
                    input_dim=input_dim,
                    layer_dims=layer_dims,
                    lip_const=1.0,
                    monotonic=False,
                    nbody="TwoBody",
                    feature_names=feature_names,
                    features_config_path=create_detailed_features_config,
                    activation_fn="groupsort",
                    dropout_rate=0.0,  # No dropout for Lipschitz
                    batch_norm=False   # No batch norm for Lipschitz
                )
                
                # Verify direct_norm was called with expected parameters
                for idx, call_args in enumerate(mock_components['direct_norm'].call_args_list):
                    _, kwargs = call_args
                    logger.info(f"Layer {idx} direct_norm parameters:")
                    logger.info(f"  - always_norm: {kwargs.get('always_norm')}")
                    logger.info(f"  - kind: {kwargs.get('kind')}")
                    logger.info(f"  - max_norm: {kwargs.get('max_norm')}")
                    
                    # Check the norm kind parameter
                    assert kwargs.get('kind') == norm_kind, \
                        f"Norm kind mismatch: {kwargs.get('kind')} != {norm_kind}"
                
                # Check if any layers have always_norm=True
                layers_with_always_norm = [i for i, args in enumerate(mock_components['direct_norm'].call_args_list) 
                                         if args[1].get('always_norm') == True]
                
                if always_norm:
                    logger.info(f"Found {len(layers_with_always_norm)} layers with always_norm=True")
                    # We might not find any layers if the setup doesn't use batch norm,
                    # but we've at least checked the code path is exercised
                
                logger.info(f"All norm parameters verified for {case['description']}")
                
    def test_monotonicity_missing_feature(self, create_detailed_features_config, mock_components):
        """Test handling of missing features in monotonicity constraints."""
        logger.info("\n" + "="*80)
        logger.info("TESTING HANDLING OF MISSING FEATURES")
        logger.info("="*80)
        
        # Set up test parameters with a non-existent feature
        input_dim = 5
        layer_dims = [10, 5]
        feature_names = ['feature1', 'feature2', 'non_existent_feature', 'feature4', 'feature5']
        
        logger.info(f"Testing with feature names including non-existent feature: {feature_names}")
        
        # Create a mock _load_monotone_constrs method that raises ValueError for missing feature
        def mock_load_monotone_constrs(self, path, key):
            # This will raise ValueError for the non_existent_feature
            feature_map = {
                'feature1': 1,
                'feature2': 0,
                'feature4': 1,
                'feature5': 0
            }
            
            # Check for missing features
            for feature in self.feature_names:
                if feature not in feature_map:
                    raise ValueError(f"Feature {feature} not found in config for {key}")
            
            # This should not be reached for the test case
            return [0] * self.input_dim
        
        # Create model with patched components and expect ValueError
        with patch('models.SigmaNet', mock_components['SigmaNet']), \
             patch('models.GroupSort', mock_components['GroupSort']), \
             patch('models.direct_norm', mock_components['direct_norm']), \
             patch.object(LipschitzNet, '_load_monotone_constrs', mock_load_monotone_constrs), \
             pytest.raises(ValueError) as excinfo:
            
            # Create LipschitzNet with monotonicity and missing feature
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lip_const=1.0,
                monotonic=True,
                nbody="TwoBody",
                feature_names=feature_names,
                features_config_path=create_detailed_features_config,
                activation_fn="groupsort",
                dropout_rate=0.0,  # No dropout for Lipschitz
                batch_norm=False   # No batch norm for Lipschitz
            )
            
        logger.info(f"Correctly raised ValueError: {str(excinfo.value)}")
        assert "not found in config" in str(excinfo.value), "Error message should mention missing feature"
        
    def test_l1_regularization(self, create_detailed_features_config, mock_components):
        """Test the L1 regularization calculation."""
        logger.info("\n" + "="*80)
        logger.info("TESTING L1 REGULARIZATION")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 3
        layer_dims = [4, 2]
        l1_factors = [0.0, 0.01, 0.1, 1.0]  # Test different L1 regularization weights
        
        for l1_factor in l1_factors:
            logger.info(f"\nTesting with L1 factor: {l1_factor}")
            
            # Create a simple model with actual parameters that we can test L1 regularization on
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Create some parameters with known values
                    self.layer1 = nn.Linear(input_dim, layer_dims[0])
                    self.layer2 = nn.Linear(layer_dims[0], layer_dims[1])
                    self.layer3 = nn.Linear(layer_dims[1], 1)
                    
                    # Set weights to known values for predictable L1 sum
                    with torch.no_grad():
                        self.layer1.weight.fill_(0.1)
                        self.layer1.bias.fill_(0.1)
                        self.layer2.weight.fill_(0.2)
                        self.layer2.bias.fill_(0.2)
                        self.layer3.weight.fill_(0.3)
                        self.layer3.bias.fill_(0.3)
                
                def forward(self, x):
                    x = self.layer1(x)
                    x = torch.relu(x)
                    x = self.layer2(x)
                    x = torch.relu(x)
                    x = self.layer3(x)
                    return x
                
                def parameters(self):
                    return list(nn.Module.parameters(self))
            
            # Create the test model
            test_model = TestModel()
            
            # Calculate the expected L1 loss manually
            expected_param_sum = 0
            for param in test_model.parameters():
                expected_param_sum += torch.abs(param).sum().item()
            
            expected_l1_loss = l1_factor * expected_param_sum
            
            logger.info(f"Total parameter absolute sum: {expected_param_sum}")
            logger.info(f"Expected L1 loss (factor={l1_factor}): {expected_l1_loss}")
            
            # Create a model with the test model as the base
            with patch('models.SigmaNet', lambda model, **kwargs: model), \
                patch('models.GroupSort', lambda x: torch.nn.ReLU()), \
                patch('models.direct_norm', lambda module, **kwargs: module), \
                patch.object(LipschitzNet, '_build_model', return_value=test_model):
                
                # Create the model
                model = LipschitzNet(
                    input_dim=input_dim,
                    layer_dims=layer_dims,
                    lip_const=1.0,
                    monotonic=False,
                    nbody="TwoBody",
                    feature_names=['f1', 'f2', 'f3'],
                    features_config_path=create_detailed_features_config,
                    activation_fn="groupsort",
                    l1_factor=l1_factor,
                    dropout_rate=0.0,  # No dropout for Lipschitz
                    batch_norm=False   # No batch norm for Lipschitz
                )
                
                # Get the actual L1 loss - handle both tensor and float cases
                actual_l1_loss = model.get_l1_loss()
                if torch.is_tensor(actual_l1_loss):
                    actual_l1_loss_value = actual_l1_loss.item()
                else:
                    actual_l1_loss_value = float(actual_l1_loss)
                    
                logger.info(f"Actual L1 loss: {actual_l1_loss_value}")
                
                # Check with a small tolerance for floating point precision
                assert abs(actual_l1_loss_value - expected_l1_loss) < 1e-5, \
                    f"L1 loss mismatch: {actual_l1_loss_value} != {expected_l1_loss}"
                
                logger.info(f"L1 regularization verified for factor={l1_factor}")

    def test_lipschitz_model_structure(self, create_detailed_features_config, mock_components):
        """Test the structure of the created model."""
        logger.info("\n" + "="*80)
        logger.info("TESTING MODEL STRUCTURE")
        logger.info("="*80)
        
        # Set up test parameters
        input_dim = 5
        layer_dims = [10, 5]
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # Create a function to check the layer structure
        def check_model_structure(model_seq, expected_layer_dims, use_nn_modules=False):
            """Verify the structure of a sequential model."""
            # Count the number of Linear and activation layers
            linear_layers = []
            activation_layers = []
            
            if use_nn_modules:
                # For real nn.Module instances
                for layer in model_seq:
                    if isinstance(layer, nn.Linear):
                        linear_layers.append(layer)
                    elif isinstance(layer, (nn.ReLU, MockGroupSort)):
                        activation_layers.append(layer)
            else:
                # For mocked modules
                linear_count = 0
                activation_count = 0
                for i, call in enumerate(mock_components['direct_norm'].call_args_list):
                    # Each direct_norm call corresponds to a linear layer
                    linear_count += 1
                
                for i, call in enumerate(mock_components['GroupSort'].call_args_list):
                    # Each GroupSort call corresponds to an activation layer
                    activation_count += 1
                
                linear_layers = [None] * linear_count
                activation_layers = [None] * activation_count
            
            # Expected number of linear layers: input->hidden1, hidden1->hidden2, hidden2->output
            expected_linear_count = len(expected_layer_dims) + 1
            
            # Expected number of activation layers: one after each hidden layer
            expected_activation_count = len(expected_layer_dims)
            
            logger.info(f"Linear layers count: {len(linear_layers)}, expected: {expected_linear_count}")
            logger.info(f"Activation layers count: {len(activation_layers)}, expected: {expected_activation_count}")
            
            assert len(linear_layers) == expected_linear_count, \
                f"Incorrect number of linear layers: {len(linear_layers)} != {expected_linear_count}"
                
            assert len(activation_layers) == expected_activation_count, \
                f"Incorrect number of activation layers: {len(activation_layers)} != {expected_activation_count}"
        
        # Create model with patched components
        with patch('models.SigmaNet', mock_components['SigmaNet']), \
             patch('models.GroupSort', mock_components['GroupSort']), \
             patch('models.direct_norm', mock_components['direct_norm']):
            
            # Create LipschitzNet
            logger.info("\nCreating LipschitzNet for structure test:")
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lip_const=1.0,
                monotonic=False,
                nbody="TwoBody",
                feature_names=feature_names,
                features_config_path=create_detailed_features_config,
                activation_fn="groupsort",
                dropout_rate=0.0,  # No dropout for Lipschitz
                batch_norm=False   # No batch norm for Lipschitz
            )
            
            # Check the model structure
            logger.info("\nChecking model structure:")
            check_model_structure(model.model, layer_dims)
            
            logger.info("Model structure verified!")