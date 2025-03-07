import torch
import torch.nn as nn
from typing import List


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
