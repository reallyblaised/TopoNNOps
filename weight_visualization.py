import torch
import numpy as np
import pandas as pd
import altair as alt
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

        # Generate visualizations
        heatmaps = []
        distributions = []
        gradients = []

        for layer_data in weights_data:
            # Create weight heatmap
            input_labels = self.feature_names if layer_data["name"] == "0" else None
            heatmap = self._create_weight_heatmap(
                layer_data["weights"], layer_data["name"], input_labels=input_labels
            )
            heatmaps.append(heatmap)

            # Create weight distribution
            dist = self._create_weight_distribution(
                layer_data["weights"], layer_data["name"]
            )
            distributions.append(dist)

            # Create gradient magnitude visualization
            grad = self._create_gradient_visualization(layer_data["name"])
            if grad:  # Only append if gradients exist
                gradients.append(grad)

        # Create activation visualization
        activation_viz = self._create_activation_patterns(sample_input)

        # Combine visualizations
        dashboard = alt.vconcat(
            alt.hconcat(*heatmaps),
            alt.hconcat(*distributions),
            alt.hconcat(*([activation_viz] + gradients)),
        ).resolve_scale(color="independent")

        # Log to MLflow
        self._log_dashboard(dashboard, epoch)

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

    def _create_weight_heatmap(
        self,
        weights: np.ndarray,
        layer_name: str,
        input_labels: Optional[List[str]] = None,
        output_labels: Optional[List[str]] = None,
    ) -> alt.Chart:
        """Create interactive weight heatmap"""
        # Prepare data
        df = pd.DataFrame(weights)

        if input_labels is None:
            input_labels = [f"Input {i}" for i in range(weights.shape[1])]
        if output_labels is None:
            output_labels = [f"Neuron {i}" for i in range(weights.shape[0])]

        # Melt DataFrame for Altair
        melted_df = df.reset_index().melt(
            id_vars=["index"], var_name="input", value_name="weight"
        )
        melted_df["output"] = melted_df["index"].apply(lambda x: output_labels[x])
        melted_df["input"] = melted_df["input"].apply(lambda x: input_labels[x])

        # Create heatmap
        base = alt.Chart(melted_df).encode(
            x=alt.X("input:N", title="Input Features"),
            y=alt.Y("output:N", title="Neurons"),
        )

        # Main heatmap
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "weight:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="Weight Value",
            ),
            tooltip=["input", "output", "weight"],
        )

        # Add weight values as text
        text = base.mark_text(baseline="middle").encode(
            text=alt.Text("weight:Q", format=".2f"),
            color=alt.condition(
                abs(alt.datum.weight) > 0.5, alt.value("white"), alt.value("black")
            ),
        )

        return (
            (heatmap + text)
            .properties(
                width=400, height=300, title=f"Weight Heatmap - Layer {layer_name}"
            )
            .interactive()
        )

    def _create_weight_distribution(self, weights: np.ndarray, layer_name: str) -> alt.Chart:
        """Create weight distribution visualization"""
        # Flatten weights and create DataFrame
        flat_weights = weights.flatten()
        df = pd.DataFrame({"weight": flat_weights, "layer": f"Layer {layer_name}"})

        # Instead of using transform_density, create a histogram
        chart = (
            alt.Chart(df)
            .mark_bar(opacity=0.5)
            .encode(
                x=alt.X(
                    "weight:Q", 
                    title="Weight Value",
                    bin=alt.Bin(maxbins=30)  # Use binning instead of density transform
                ),
                y=alt.Y(
                    "count():Q", 
                    title="Count"
                ),
                color="layer:N",
                tooltip=["layer", alt.Tooltip("weight:Q", bin=True), "count()"]
            )
            .properties(
                width=400, height=200, title=f"Weight Distribution - Layer {layer_name}"
            )
            .interactive()
        )

        return chart

    def _create_activation_patterns(self, sample_input: torch.Tensor) -> alt.Chart:
        """Visualize activation patterns across layers"""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            self.model(sample_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Create visualization data
        activation_data = []
        for i, act in enumerate(activations):
            act_mean = np.mean(act, axis=0)
            for j, value in enumerate(act_mean):
                activation_data.append(
                    {"layer": f"Layer {i+1}", "neuron": j, "activation": value}
                )

        df = pd.DataFrame(activation_data)

        # Create heatmap
        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("neuron:O", title="Neuron Index"),
                y=alt.Y("layer:N", title="Layer"),
                color=alt.Color(
                    "activation:Q",
                    scale=alt.Scale(scheme="viridis"),
                    title="Mean Activation",
                ),
                tooltip=["layer", "neuron", "activation"],
            )
            .properties(width=600, height=200, title="Mean Activation Patterns")
            .interactive()
        )

        return chart

    def _create_gradient_visualization(self, layer_name: str) -> Optional[alt.Chart]:
        """Visualize gradient magnitudes for layer weights"""
        # Get the layer
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    # Get gradient data
                    grad_data = module.weight.grad.cpu().numpy()

                    # Create DataFrame
                    df = pd.DataFrame(
                        {
                            "gradient": grad_data.flatten(),
                            "layer": f"Layer {layer_name}",
                        }
                    )

                    # Create histogram plot (instead of density)
                    chart = (
                        alt.Chart(df)
                        .mark_bar(opacity=0.5)
                        .encode(
                            x=alt.X(
                                "gradient:Q", 
                                title="Gradient Value",
                                bin=alt.Bin(maxbins=30)
                            ),
                            y=alt.Y(
                                "count():Q", 
                                title="Count"
                            ),
                            color="layer:N",
                            tooltip=["layer", alt.Tooltip("gradient:Q", bin=True), "count()"]
                        )
                        .properties(
                            width=400,
                            height=200,
                            title=f"Gradient Distribution - Layer {layer_name}",
                        )
                        .interactive()
                    )

                    return chart

        return None

    def _log_dashboard(self, dashboard: alt.VConcatChart, epoch: int) -> None:
        """Log dashboard to MLflow"""
        # Convert to Vega-Lite spec
        vega_spec = dashboard.to_dict()

        # Create HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Weight Analysis</title>
            <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ width: 1200px; margin: 0 auto; }}
                h1 {{ text-align: center; color: #333; }}
            </style>
        </head>
        <body>
            <h1>Neural Network Weight Analysis - Epoch {epoch}</h1>
            <div class="dashboard" id="vis"></div>
            <script>
                vegaEmbed('#vis', {json.dumps(vega_spec)});
            </script>
        </body>
        </html>
        """

        # Log to MLflow
        mlflow.log_dict(vega_spec, f"internals_visualization_epoch_{epoch}.json")
        mlflow.log_text(html_content, f"internals_visualization_epoch_{epoch}.html")