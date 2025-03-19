import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
import yaml
from pathlib import Path
import sys

class UnconstrainedNet(nn.Module):
    """
    Customisable neural network without Lipschitz constraint and residual connections
    encoding monotonicity at the level of the partials of Lip1 learnt functions.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        activation_fn: str = "relu",
        l1_factor: float = 0.0,  # prefactor to L1 regularization addition to loss fn
        residual: bool = False,  # add residual connections
        init_method: str = "xavier_uniform",
    ):
        super().__init__()
        self.layers = self._build_layers(
            input_dim=input_dim,
            layer_dims=layer_dims,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation_fn=activation_fn,
        )
        self.residual = residual
        self.l1_factor = l1_factor
        self._initialize_weights(init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connections"""
        prev_x = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # residual connection warranted only by same-dimensional tensors
            if (
                self.residual
                and isinstance(layer, nn.Linear)
                and x.shape == prev_x.shape
            ):
                x = x + prev_x
            prev_x = x
        return x

    def get_l1_loss(self) -> torch.Tensor:
        """Calculate L1 regularization loss"""
        l1_loss = 0.0
        if self.l1_factor > 0:
            for param in self.parameters():
                l1_loss += torch.abs(param).sum()
        return self.l1_factor * l1_loss

    def _select_activation_fn(self, fn_id: str = "relu") -> nn.Module:
        """Select activation function"""
        activation_fns = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        if fn_id not in activation_fns:
            raise ValueError(f"Unknown activation function: {fn_id}")
        return activation_fns[fn_id]

    def _build_layers(
        self,
        input_dim: int,
        layer_dims: List[int],
        dropout_rate: float,
        batch_norm: bool,
        activation_fn: str,
    ) -> nn.ModuleList:
        """Build neural network layers with advanced options"""
        layers = nn.ModuleList()
        prev_dim = input_dim

        for dim in layer_dims:
            block = nn.ModuleList()

            # Linear layer
            block.append(nn.Linear(prev_dim, dim))

            # Batch normalization
            if batch_norm:
                block.append(nn.BatchNorm1d(dim))

            # Activation function
            block.append(self._select_activation_fn(activation_fn))

            # Dropout
            if dropout_rate > 0:
                block.append(nn.Dropout(dropout_rate))

            # Add block to layers
            layers.extend(block)
            prev_dim = dim

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))

        return layers


    def _initialize_weights(self, init_method: str):
        """Initialize network weights using specified method"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight)
                elif init_method == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {init_method}")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

# Import from monotonenorm if available, otherwise create placeholder imports
try:
    from monotonenorm import SigmaNet, GroupSort, direct_norm
except ImportError:
    print("WARNING: monotonenorm not found, creating placeholder classes")
    # Create placeholder classes/functions for documentation
    class SigmaNet(nn.Module):
        def __init__(self, model, sigma, monotone_constraints):
            super().__init__()
            self.model = model
            self.sigma = sigma
            self.monotone_constraints = monotone_constraints
            raise ImportError("monotonenorm library is required for SigmaNet")
    
    class GroupSort(nn.Module):
        def __init__(self, num_units):
            super().__init__()
            self.num_units = num_units
            raise ImportError("monotonenorm library is required for GroupSort")
    
    def direct_norm(module, always_norm=False, kind="one", max_norm=1.0):
        raise ImportError("monotonenorm library is required for direct_norm")


class LipschitzNet(nn.Module):
    """
    Neural network with Lipschitz constraints.
    
    This class implements a neural network with Lipschitz constraints on the weights,
    optionally with monotonicity constraints on the inputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        lip_const: float = 1.0,
        monotonic: bool = False,
        nbody: str = "TwoBody",
        feature_names: Optional[List[str]] = None,
        features_config_path: str = "config/features.yml",
        activation_fn: str = "groupsort",
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        l1_factor: float = 0.0
    ):
        """
        Initialize a Lipschitz-constrained neural network.
        
        Args:
            input_dim: Dimension of input features
            layer_dims: List of layer dimensions
            lip_const: Lipschitz constant (sigma)
            monotonic: Whether to enforce monotonicity constraints
            nbody: Key in features.yml specifying which feature set to use
            feature_names: Names of input features (must match order in features.yml)
            features_config_path: Path to features configuration file
            activation_fn: Activation function to use (currently only 'groupsort' supported)
            dropout_rate: Dropout rate
            batch_norm: Whether to use batch normalization
            l1_factor: L1 regularization factor
        """
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.lip_const = lip_const
        self.monotonic = monotonic
        self.nbody = nbody
        self.feature_names = feature_names
        self.features_config_path = features_config_path
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l1_factor = l1_factor
        
        # Build the model
        self.model = self._build_model()
    
    def _load_monotone_constrs(self, path: str, key: str) -> List[int]:
        """Load monotonicity constraints from config file."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'features' not in config:
                raise ValueError(f"No 'features' section found in {path}")
            
            if key not in config['features']:
                raise ValueError(f"No '{key}' section found in features configuration")
            
            # Get the feature constraints
            feature_dict = config['features'][key]
            
            # If feature_names is provided, use it to order the constraints
            if self.feature_names:
                monotone_constraints = []
                for feature in self.feature_names:
                    if feature in feature_dict:
                        monotone_constraints.append(int(feature_dict[feature]))
                    else:
                        raise ValueError(f"Feature {feature} not found in config for {key}")
                return monotone_constraints
            
            # Otherwise, just return all constraints in the order from the file
            return list(feature_dict.values())
            
        except Exception as e:
            print(f"Error loading monotonicity constraints: {str(e)}")
            # Return a list of zeros (no monotonicity constraints)
            return [0] * self.input_dim
    
    def _lipschitz_norm(self, module, is_norm=False, kind="one"):
        """Apply Lipschitz normalization to a module."""
        n_layers = len(self.layer_dims) + 1  # +1 for output layer
        
        # Calculate max norm for each layer to ensure overall constraint of lip_const
        max_norm = self.lip_const ** (1 / n_layers)
        
        return direct_norm(
            module,
            always_norm=is_norm,
            kind=kind,
            max_norm=max_norm
        )
    
    def _build_model(self):
        """Build the Lipschitz-constrained model."""
        model_layers = []
        in_features = self.input_dim
        
        # Add hidden layers
        for dim in self.layer_dims:
            # Linear layer with Lipschitz constraint
            model_layers.append(self._lipschitz_norm(nn.Linear(in_features, dim)))
            
            # Activation function
            if self.activation_fn.lower() == 'groupsort':
                model_layers.append(GroupSort(dim // 2))
            else:
                raise ValueError(f"Unsupported activation function: {self.activation_fn}")
            
            # abort if batch_norm or dropout is used
            if self.batch_norm or self.dropout_rate > 0:
                raise ValueError("Batch normalization and dropout are not supported with Lipschitz constraints.")
            
            in_features = dim
        
        # Output layer (binary classification)
        model_layers.append(self._lipschitz_norm(nn.Linear(in_features, 1)))
        
        # Create the sequential model
        model = nn.Sequential(*model_layers)
        
        # If monotonicity is enabled, wrap with SigmaNet
        if self.monotonic:
            # Load monotonicity constraints from config
            monotone_constraints = self._load_monotone_constrs(
                self.features_config_path, self.nbody
            )
            
            # Ensure we have the right number of constraints
            assert len(monotone_constraints) == self.input_dim, \
                f"Number of monotonicity constraints ({len(monotone_constraints)}) " \
                f"does not match input dimension ({self.input_dim})"
            
            # Wrap the model with SigmaNet
            model = SigmaNet(
                model, 
                sigma=self.lip_const, 
                monotone_constraints=monotone_constraints
            )
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_l1_loss(self) -> torch.Tensor:
        """Calculate L1 regularization loss."""
        l1_loss = 0.0
        if self.l1_factor > 0:
            for param in self.parameters():
                l1_loss += torch.abs(param).sum()
        return self.l1_factor * l1_loss