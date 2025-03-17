import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import mlflow
import json
from torch import nn


class WeightVisualizer:
    """Neural network weight and activation visualization"""

    def __init__(self, model: nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def create_weight_dashboard(self, sample_input: torch.Tensor, epoch: int) -> None:
        """Create and log comprehensive neural network internals visualization dashboard"""
        # Extract layer information
        weights_data = self._get_layer_weights()
        
        # Create figure with subplots - dynamically determine rows based on number of layers
        num_layers = len(weights_data)
        
        # Determine if we have gradients for any layer
        has_gradients = any(self._has_gradients(layer_data['name']) for layer_data in weights_data)
        
        # Calculate number of rows needed
        num_rows = 3  # Default: weights, distributions, activations
        if has_gradients:
            num_rows += 1  # Add a row for gradients
        
        # Create subplot titles
        subplot_titles = []
        # Weight heatmap titles
        subplot_titles.extend([f"Weight Heatmap - Layer {layer_data['name']}" for layer_data in weights_data])
        # Weight distribution titles
        subplot_titles.extend([f"Weight Distribution - Layer {layer_data['name']}" for layer_data in weights_data])
        # Activation title
        subplot_titles.append("Mean Activation Patterns")
        # Add empty titles for remaining cells in activation row
        subplot_titles.extend([""] * (num_layers - 1))
        # Gradient titles if applicable
        if has_gradients:
            subplot_titles.extend([f"Gradient Distribution - Layer {layer_data['name']}" 
                                 for layer_data in weights_data])
        
        # Create subplot specs
        specs = [
            [{"type": "heatmap"} for _ in range(num_layers)],  # Weight heatmaps
            [{"type": "histogram"} for _ in range(num_layers)],  # Weight distributions
            [{"type": "heatmap", "colspan": num_layers}] + [None] * (num_layers - 1)  # Activations
        ]
        
        # Add row for gradients if needed
        if has_gradients:
            specs.append([{"type": "histogram"} for _ in range(num_layers)])  # Gradient distributions
        
        # Create the figure
        fig = make_subplots(
            rows=num_rows, 
            cols=max(num_layers, 1),
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.12,
            horizontal_spacing=0.05
        )

        # Add weight visualizations
        for i, layer_data in enumerate(weights_data):
            col = i + 1  # 1-based indexing for Plotly
            
            # Add weight heatmap
            self._add_weight_heatmap(
                fig, 
                layer_data["weights"], 
                layer_data["name"],
                row=1, 
                col=col, 
                input_labels=self.feature_names if layer_data["name"] == "0" else None
            )
            
            # Add weight distribution
            self._add_weight_distribution(
                fig, 
                layer_data["weights"], 
                layer_data["name"],
                row=2, 
                col=col
            )

        # Add activation visualization
        self._add_activation_patterns(fig, sample_input, row=3, col=1)
        
        # Add gradient visualizations in row 4 if any exist
        if has_gradients:
            for i, layer_data in enumerate(weights_data):
                if self._has_gradients(layer_data['name']):
                    col = i + 1
                    self._add_gradient_visualization(fig, layer_data["name"], row=4, col=col)

        # Update layout
        fig.update_layout(
            height=300 * num_rows,
            width=400 * num_layers,
            title_text=f"Neural Network Weight Analysis - Epoch {epoch}",
            showlegend=False
        )

        # Log to MLflow
        self._log_dashboard(fig, epoch)

    def _get_layer_weights(self) -> List[Dict]:
        """Extract weights from each layer of the model"""
        weights_data = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                bias = module.bias.data.cpu().numpy()

                weights_data.append(
                    {
                        "name": name,
                        "weights": weights,
                        "bias": bias,
                        "shape": weights.shape,
                    }
                )

        return weights_data
    
    def _has_gradients(self, layer_name: str) -> bool:
        """Check if layer has gradients"""
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                return module.weight.grad is not None
        return False

    def _add_weight_heatmap(
        self,
        fig,
        weights: np.ndarray,
        layer_name: str,
        row: int,
        col: int,
        input_labels: Optional[List[str]] = None,
        output_labels: Optional[List[str]] = None,
    ) -> None:
        """Create interactive weight heatmap"""
        # Prepare labels
        if input_labels is None:
            input_labels = [f"Input {i}" for i in range(weights.shape[1])]
        else:
            # Ensure we don't have more labels than inputs
            input_labels = input_labels[:weights.shape[1]]
        
        if output_labels is None:
            output_labels = [f"Neuron {i}" for i in range(weights.shape[0])]
            
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=weights,
                x=input_labels,
                y=output_labels,
                colorscale="RdBu",
                zmid=0,  # Center the color scale at zero
                colorbar=dict(title="Weight Value") if col == 1 else None,
                showscale=col == 1,  # Only show color scale for first column
                hovertemplate="Input: %{x}<br>Neuron: %{y}<br>Weight: %{z:.4f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Input Features", row=row, col=col)
        fig.update_yaxes(title_text="Neurons", row=row, col=col)

    def _add_weight_distribution(self, fig, weights: np.ndarray, layer_name: str, row: int, col: int) -> None:
        """Create weight distribution visualization"""
        # Flatten weights
        flat_weights = weights.flatten()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=flat_weights,
                nbinsx=30,
                marker_color="rgba(50, 168, 82, 0.7)",
                name=f"Layer {layer_name}",
                showlegend=False,
                hovertemplate="Weight: %{x:.4f}<br>Count: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Weight Value", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_activation_patterns(self, fig, sample_input: torch.Tensor, row: int, col: int) -> None:
        """Visualize activation patterns across layers"""
        activations = []
        layer_names = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn))
                layer_names.append(name)

        # Forward pass
        with torch.no_grad():
            self.model(sample_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Create data for heatmap
        z_data = []
        for act in activations:
            act_mean = np.mean(act, axis=0)
            z_data.append(act_mean)
            
        # If we have no activations, return
        if not z_data:
            return
            
        # Convert to numpy array
        z_data = np.array(z_data)
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=[f"Neuron {i}" for i in range(z_data.shape[1])],
                y=[f"Layer {name}" for name in layer_names],
                colorscale="Viridis",
                colorbar=dict(title="Mean Activation"),
                hovertemplate="Layer: %{y}<br>Neuron: %{x}<br>Activation: %{z:.4f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Neuron Index", row=row, col=col)
        fig.update_yaxes(title_text="Layer", row=row, col=col)

    def _add_gradient_visualization(self, fig, layer_name: str, row: int, col: int) -> None:
        """Visualize gradient magnitudes for layer weights"""
        # Get the layer
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    # Get gradient data
                    grad_data = module.weight.grad.cpu().numpy()
                    
                    # Flatten gradients
                    flat_grads = grad_data.flatten()
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=flat_grads,
                            nbinsx=30,
                            marker_color="rgba(255, 127, 14, 0.7)",
                            name=f"Layer {layer_name}",
                            showlegend=False,
                            hovertemplate="Gradient: %{x:.4f}<br>Count: %{y}<extra></extra>"
                        ),
                        row=row, col=col
                    )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Gradient Value", row=row, col=col)
                    fig.update_yaxes(title_text="Count", row=row, col=col)

    def _log_dashboard(self, fig, epoch: int) -> None:
        """Log dashboard to MLflow"""
        # Convert to Vega-Lite spec
        plotly_json = fig.to_json()

        # Create HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Weight Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ width: 100%; margin: 0 auto; }}
                h1 {{ text-align: center; color: #333; }}
            </style>
        </head>
        <body>
            <h1>Neural Network Weight Analysis - Epoch {epoch}</h1>
            <div class="dashboard" id="vis"></div>
            <script>
                var figure = {plotly_json};
                Plotly.newPlot('vis', figure.data, figure.layout);
            </script>
        </body>
        </html>
        """

        # Log to MLflow
        mlflow.log_dict(json.loads(plotly_json), f"internals_visualization_epoch_{epoch}.json")
        mlflow.log_text(html_content, f"internals_visualization_epoch_{epoch}.html")