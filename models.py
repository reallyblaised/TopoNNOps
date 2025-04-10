import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
import yaml
from pathlib import Path
import sys
import monotonicnetworks as lmn
from monotonenorm import SigmaNet as sigmanet_legacy
from monotonenorm import GroupSort as groupsort_legacy
from monotonenorm import direct_norm as direct_norm_legacy


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


# Import from monotonicnetworks if available, otherwise create placeholder imports
try:
    import monotonicnetworks as lmn
except ImportError:
    print("WARNING: monotonicnetworks not found, creating placeholder classes")

    # Create placeholder classes/functions for documentation
    class LipschitzLinear(nn.Module):
        def __init__(self, in_features, out_features, kind="one"):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.kind = kind
            raise ImportError(
                "monotonicnetworks library is required for LipschitzLinear"
            )

    class GroupSort(nn.Module):
        def __init__(self, num_units):
            super().__init__()
            self.num_units = num_units
            raise ImportError("monotonicnetworks library is required for GroupSort")

    class MonotonicWrapper(nn.Module):
        def __init__(
            self, lipschitz_module, lipschitz_const=1.0, monotonic_constraints=None
        ):
            super().__init__()
            self.nn = lipschitz_module
            self.lipschitz_const = lipschitz_const
            self.monotonic_constraints = monotonic_constraints
            raise ImportError(
                "monotonicnetworks library is required for MonotonicWrapper"
            )


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
        l1_factor: float = 0.0,
        lip_kind: str = "one",  # Add a new parameter for Lipschitz constraint type
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
            lip_kind: Type of Lipschitz constraint to use:
                      - "one": Use kind="one" for all layers (default)
                      - "inf": Use kind="inf" for all layers
                      - "one-inf": Use kind="one-inf" for first layer, kind="inf" for all following layers
                      - "nominal": Use kind="one-inf" for first layer, kind="inf" for middle layers, and kind="one" for output layer
                      - "default": Same as "one-inf"
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

        self.lip_kind = lip_kind

        # Build the model
        self.model = self._build_model()

    def _load_monotone_constrs(self, path: str, key: str) -> List[int]:
        """Load monotonicity constraints from config file."""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            if "features" not in config:
                raise ValueError(f"No 'features' section found in {path}")

            if key not in config["features"]:
                raise ValueError(f"No '{key}' section found in features configuration")

            # Get the feature constraints
            feature_dict = config["features"][key]

            # If feature_names is provided, use it to order the constraints
            if self.feature_names:
                monotone_constraints = []
                for feature in self.feature_names:
                    if feature in feature_dict:
                        monotone_constraints.append(int(feature_dict[feature]))
                    else:
                        raise ValueError(
                            f"Feature {feature} not found in config for {key}"
                        )
                return monotone_constraints

            # Otherwise, just return all constraints in the order from the file
            return list(feature_dict.values())

        except Exception as e:
            print(f"Error loading monotonicity constraints: {str(e)}")
            # Return a list of zeros (no monotonicity constraints)
            return [0] * self.input_dim

    def _build_model(self):
        """Build the Lipschitz-constrained model using monotonicnetworks."""
        # Check for unsupported options with Lipschitz constraints
        if self.batch_norm or self.dropout_rate > 0:
            raise ValueError(
                "Batch normalization and dropout are not supported with Lipschitz constraints."
            )

        # Check activation function
        if self.activation_fn.lower() != "groupsort":
            raise ValueError(
                f"Unsupported activation function: {self.activation_fn}. Only 'groupsort' is supported."
            )

        # Determine layer kinds based on lip_kind parameter
        if self.lip_kind == "one-inf":
            # First layer uses "one-inf", all others use "inf"
            layer_kinds = ["one-inf"] + ["inf"] * len(self.layer_dims)
        elif self.lip_kind in ["one", "inf"]:
            # All layers use the same kind
            layer_kinds = [self.lip_kind] * (len(self.layer_dims) + 1)
        elif self.lip_kind == "nominal":
            # First layer uses "one-inf", middle layers use "inf", output layer uses "one"
            layer_kinds = ["one-inf"] + ["inf"] * (len(self.layer_dims) - 1) + ["one"]
        elif self.lip_kind == "default":
            # First layer uses "one-inf", all others use "inf"
            layer_kinds = ["one-inf"] + ["inf"] * len(self.layer_dims)
        else:
            raise ValueError(f"Unsupported lip_kind: {self.lip_kind}")

        # Build the sequential model
        layers = []
        in_features = self.input_dim

        # Add hidden layers
        for idx, dim in enumerate(self.layer_dims):
            # Add LipschitzLinear layer with appropriate kind
            layers.append(lmn.LipschitzLinear(in_features, dim, kind=layer_kinds[idx]))

            # Add activation function (GroupSort with dim // 2 groups)
            layers.append(
                lmn.GroupSort(dim // 2)
            )  # NOTE: mathematically, this buys more expressivity

            in_features = dim

        # Add output layer
        layers.append(lmn.LipschitzLinear(in_features, 1, kind=layer_kinds[-1]))

        # Create the sequential model
        model = nn.Sequential(*layers)

        # If monotonicity is enabled, wrap with MonotonicWrapper
        if self.monotonic:
            # Load monotonicity constraints from config
            monotonic_constraints = self._load_monotone_constrs(
                self.features_config_path, self.nbody
            )

            # Ensure we have the right number of constraints
            assert len(monotonic_constraints) == self.input_dim, (
                f"Number of monotonicity constraints ({len(monotonic_constraints)}) "
                f"does not match input dimension ({self.input_dim})"
            )

            # Wrap the model with MonotonicWrapper
            model = lmn.MonotonicWrapper(
                model,
                lipschitz_const=self.lip_const,
                monotonic_constraints=monotonic_constraints,
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

    def print_architecture_details(self):
        """
        Print detailed information about the Lipschitz network architecture.
        Includes Lipschitz constants, GroupSort arguments, and Lipschitz kind.
        """
        print("\n===== LipschitzNet Architecture Details =====")
        print(f"Input dimension: {self.input_dim}")
        print(f"Layer dimensions: {self.layer_dims}")
        print(f"Lipschitz constant: {self.lip_const}")
        print(f"Lipschitz constraint type: {self.lip_kind}")
        print(f"Monotonicity enabled: {self.monotonic}")
        print(f"Activation function: {self.activation_fn}")
        print(f"NBodies setting: {self.nbody}")

        # Always print monotonicity constraints, even if monotonicity is disabled
        # This helps verify what constraints would be applied if monotonicity was enabled
        monotonic_constraints = self._load_monotone_constrs(
            self.features_config_path, self.nbody
        )
        print(f"Monotonicity constraints array: {monotonic_constraints}")

        if self.feature_names:
            print("\nFeature-by-feature monotonicity settings:")
            for i, feature in enumerate(self.feature_names):
                constraint_value = (
                    monotonic_constraints[i] if i < len(monotonic_constraints) else 0
                )
                constraint_desc = (
                    "increasing"
                    if constraint_value == 1
                    else "decreasing" if constraint_value == -1 else "none"
                )
                status = (
                    "ACTIVE" if self.monotonic and constraint_value != 0 else "inactive"
                )
                print(
                    f"  - {feature}: {constraint_desc} ({constraint_value}) - {status}"
                )

        print("\nLayer-by-layer details:")

        # Determine layer kinds based on lip_kind parameter
        if self.lip_kind == "one-inf":
            # First layer uses "one-inf", all others use "inf"
            layer_kinds = ["one-inf"] + ["inf"] * len(self.layer_dims)
        elif self.lip_kind in ["one", "inf"]:
            # All layers use the same kind
            layer_kinds = [self.lip_kind] * (len(self.layer_dims) + 1)
        elif self.lip_kind == "nominal":
            # First layer uses "one-inf", middle layers use "inf", output layer uses "one"
            layer_kinds = ["one-inf"] + ["inf"] * (len(self.layer_dims) - 1) + ["one"]
        elif self.lip_kind == "default":
            # First layer uses "one-inf", all others use "inf"
            layer_kinds = ["one-inf"] + ["inf"] * len(self.layer_dims)
        else:
            layer_kinds = ["unknown"] * (len(self.layer_dims) + 1)

        # Print details for each layer
        in_features = self.input_dim
        for idx, dim in enumerate(self.layer_dims):
            print(f"Layer {idx+1}:")
            print(
                f"  - LipschitzLinear: in={in_features}, out={dim}, kind={layer_kinds[idx]}"
            )
            print(f"  - GroupSort: num_groups={dim // 2}")
            in_features = dim

        # Output layer
        print(f"Output Layer:")
        print(f"  - LipschitzLinear: in={in_features}, out=1, kind={layer_kinds[-1]}")

        # Print wrapper information if monotonicity is enabled
        if self.monotonic:
            print("\nMonotonicity Implementation:")
            print(f"  - Using MonotonicWrapper with lipschitz_const={self.lip_const}")

            # Create a more readable representation of monotonic constraints
            if self.feature_names:
                monotonic_repr = "["
                for i, feature in enumerate(self.feature_names):
                    constraint_value = (
                        monotonic_constraints[i]
                        if i < len(monotonic_constraints)
                        else 0
                    )
                    monotonic_repr += f"{constraint_value},"
                monotonic_repr = monotonic_repr.rstrip(",") + "]"
                print(f"  - monotonic_constraints={monotonic_repr}")
            else:
                print(f"  - monotonic_constraints={monotonic_constraints}")

        print("=============================================\n")


# compare to legacy implementation
class LipschitzLegacyNet(nn.Module):
    """
    Legacy implementation of Lipschitz constrained neural network.

    This class implements the original fixed architecture with 32-64-32 nodes
    following the legacy code structure as closely as possible.
    """

    def __init__(
        self,
        input_dim: int,
        feature_names: List[str],
        layer_dims: List[int] = None,
        lip_const: float = 1.0,
        monotonic: bool = False,
        nbody: str = "TwoBody",
        features_config_path: str = "config/features.yml",
    ):
        """
        Initialize a legacy Lipschitz-constrained neural network.
        """
        super().__init__()
        self.input_dim = input_dim
        self.feature_names = feature_names
        self.layer_dims = layer_dims
        self.lip_const = lip_const
        self.monotonic = monotonic
        self.nbody = nbody
        self.features_config_path = features_config_path

        # Build the legacy model
        self.model = self._build_legacy_model()

    def _build_legacy_model(self):
        """Build the legacy model with fixed architecture"""
        # Flag to indicate if we're using robust (Lipschitz) constraints
        robust = True  # Always true for LipschitzLegacyNet

        # Define lipschitz_norm function as in the original code
        def lipschitz_norm(module, is_norm=False, _kind="one"):
            if not robust:
                print("Running with unconstrained NN")
                return module
            else:
                print("Booked Lip NN")
                print(is_norm, _kind)
                return direct_norm_legacy(
                    module,
                    always_norm=is_norm,
                    kind=_kind,
                    max_norm=float(self.lip_const) ** (1 / depth),
                )

        # Use the layer_dims to build a dynamic model instead of hardcoded dimensions
        layers = []
        prev_dim = len(self.feature_names)

        for dim in self.layer_dims:
            layers.append(lipschitz_norm(nn.Linear(prev_dim, dim)))
            layers.append(groupsort_legacy(dim // 2))
            prev_dim = dim

        # Add output layer
        layers.append(lipschitz_norm(nn.Linear(prev_dim, 1)))

        # Create the sequential model
        model = nn.Sequential(*layers)

        # Apply SigmaNet wrapper if using monotonicity
        if robust:
            print("NOTE: running with monotonicity/robustness in place")
            print(f"Lambda: {self.lip_const}")

            if self.monotonic:
                # Load monotonicity constraints from config file
                _monotone_constraints = self._load_monotone_constrs(
                    self.features_config_path, self.nbody
                )
            else:
                # No monotonicity constraints - all zeros
                _monotone_constraints = list(np.zeros(len(self.feature_names)))
                # Print feature names and constraints
                for i in range(len(self.feature_names)):
                    print(self.feature_names[i], _monotone_constraints[i])

            # Ensure correct number of constraints
            assert len(_monotone_constraints) == len(self.feature_names)

            # Wrap model with SigmaNet for Lipschitz and monotonicity constraints
            model = sigmanet_legacy(
                model, sigma=self.lip_const, monotone_constraints=_monotone_constraints
            )

        print(model)
        return model

    def _load_monotone_constrs(self, path: str, key: str) -> List[int]:
        """Load monotonicity constraints from config file."""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            if "features" not in config:
                raise ValueError(f"No 'features' section found in {path}")

            if key not in config["features"]:
                raise ValueError(f"No '{key}' section found in features configuration")

            # Get the feature constraints
            feature_dict = config["features"][key]

            # If feature_names is provided, use it to order the constraints
            if self.feature_names:
                monotone_constraints = []
                for feature in self.feature_names:
                    if feature in feature_dict:
                        monotone_constraints.append(int(feature_dict[feature]))
                    else:
                        raise ValueError(
                            f"Feature {feature} not found in config for {key}"
                        )
                return monotone_constraints

            # Otherwise, just return all constraints in the order from the file
            return list(feature_dict.values())

        except Exception as e:
            print(f"Error loading monotonicity constraints: {str(e)}")
            # Return a list of zeros (no monotonicity constraints)
            return [0] * self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def get_l1_loss(self) -> torch.Tensor:
        """Return zero L1 regularization loss."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def print_architecture_details(self):
        """Print detailed information about the legacy model architecture."""
        print("\n===== LipschitzLegacyNet Architecture Details =====")
        print(f"Input dimension: {self.input_dim}")
        print(f"Lipschitz constant: {self.lip_const}")
        print(f"Monotonicity enabled: {self.monotonic}")
        print(f"NBodies setting: {self.nbody}")

        # Print monotonicity constraints
        monotonic_constraints = self._load_monotone_constrs(
            self.features_config_path, self.nbody
        )
        print(f"Monotonicity constraints array: {monotonic_constraints}")

        if self.feature_names:
            print("\nFeature-by-feature monotonicity settings:")
            for i, feature in enumerate(self.feature_names):
                constraint_value = (
                    monotonic_constraints[i] if i < len(monotonic_constraints) else 0
                )
                constraint_desc = (
                    "increasing"
                    if constraint_value == 1
                    else "decreasing" if constraint_value == -1 else "none"
                )
                status = (
                    "ACTIVE" if self.monotonic and constraint_value != 0 else "inactive"
                )
                print(
                    f"  - {feature}: {constraint_desc} ({constraint_value}) - {status}"
                )

        print("=============================================\n")
