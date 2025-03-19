import torch
import numpy as np
import pandas as pd
import altair as alt
import mlflow
from torch import nn
import tempfile
import os
from typing import List, Dict, Optional, Union, Tuple

# Set Altair rendering options
alt.data_transformers.enable('default', max_rows=None)

class WeightVisualizer:
    """Neural network weight and activation visualization using Altair"""

    def __init__(self, model: nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def create_weight_dashboard(self, sample_input: torch.Tensor, epoch: int) -> None:
        """Create and log comprehensive neural network internals visualization dashboard"""
        # Extract layer information
        weights_data = self._get_layer_weights()
        
        # Skip if no layers found
        if not weights_data:
            print("No Linear layers found for visualization")
            return
        
        # Create visualizations
        weight_heatmaps = self._create_weight_heatmaps(weights_data)
        weight_distributions = self._create_weight_distributions(weights_data)
        activation_heatmap = self._create_activation_patterns(sample_input)
        gradient_charts = self._create_gradient_distributions(weights_data)
        
        # Combine charts into a dashboard
        dashboard = self._combine_charts(
            weight_heatmaps, 
            weight_distributions, 
            activation_heatmap,
            gradient_charts,
            epoch
        )
        
        # Log to MLflow
        self._log_dashboard(dashboard, epoch)

    def _get_layer_weights(self) -> List[Dict]:
        """Extract weights from each layer of the model"""
        weights_data = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                bias = module.bias.data.cpu().numpy() if module.bias is not None else None
                
                # Check if gradients are available
                grad = None
                if module.weight.grad is not None:
                    grad = module.weight.grad.cpu().numpy()

                weights_data.append({
                    "name": name,
                    "weights": weights,
                    "bias": bias,
                    "grad": grad,
                    "shape": weights.shape,
                })

        return weights_data
    
    def _create_weight_heatmaps(self, weights_data: List[Dict]) -> List[alt.Chart]:
        """Create weight heatmap visualizations for each layer"""
        heatmaps = []
        
        for layer_data in weights_data:
            # Prepare data for visualization
            weights = layer_data["weights"]
            layer_name = layer_data["name"]
            
            # Create a dataframe for the heatmap
            data = []
            for i in range(weights.shape[0]):  # output neurons
                for j in range(weights.shape[1]):  # input features
                    data.append({
                        'Neuron': f'Neuron {i}',
                        'Input': self.feature_names[j] if j < len(self.feature_names) else f'Input {j}',
                        'Weight': float(weights[i, j])
                    })
            
            df = pd.DataFrame(data)
            
            # Create heatmap
            heatmap = alt.Chart(df).mark_rect().encode(
                x=alt.X('Input:N', title='Input Features'),
                y=alt.Y('Neuron:N', title='Neurons'),
                color=alt.Color('Weight:Q', scale=alt.Scale(scheme='blueorange', domain=[-np.max(np.abs(weights)), np.max(np.abs(weights))])),
                tooltip=['Input:N', 'Neuron:N', 'Weight:Q']
            ).properties(
                width=400,
                height=300,
                title=f'Weight Heatmap - Layer {layer_name}'
            ).interactive()
            
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def _create_weight_distributions(self, weights_data: List[Dict]) -> List[alt.Chart]:
        """Create weight distribution visualizations for each layer"""
        distributions = []
        
        for layer_data in weights_data:
            # Prepare data for visualization
            weights = layer_data["weights"].flatten()
            layer_name = layer_data["name"]
            
            # Create dataframe
            df = pd.DataFrame({
                'Weight': weights,
                'Layer': f'Layer {layer_name}'
            })
            
            # Create histogram
            histogram = alt.Chart(df).mark_bar().encode(
                x=alt.X('Weight:Q', bin=alt.Bin(maxbins=50), title='Weight Value'),
                y=alt.Y('count()', title='Count'),
                tooltip=['Weight:Q', 'count()']
            ).properties(
                width=400,
                height=300,
                title=f'Weight Distribution - Layer {layer_name}'
            ).interactive()
            
            distributions.append(histogram)
        
        return distributions
    
    def _create_activation_patterns(self, sample_input: torch.Tensor) -> alt.Chart:
        """Create activation pattern visualization across layers"""
        activations = []
        layer_names = []

        # Hook function to collect activations
        def hook_fn(module, input, output):
            # Take mean across batch dimension
            mean_activation = output.detach().mean(dim=0).cpu().numpy()
            activations.append(mean_activation)

        # Register hooks for each linear layer
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn))
                layer_names.append(name)

        # Forward pass to collect activations
        with torch.no_grad():
            self.model(sample_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # If no activations collected, return empty chart
        if not activations:
            df = pd.DataFrame({'message': ['No activations collected']})
            return alt.Chart(df).mark_text().encode(text='message')
        
        # Create a dataframe for the activation heatmap
        data = []
        for layer_idx, (act, name) in enumerate(zip(activations, layer_names)):
            for neuron_idx, value in enumerate(act):
                data.append({
                    'Layer': f'Layer {name}',
                    'Neuron': f'Neuron {neuron_idx}',
                    'Activation': float(value)
                })
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('Neuron:N', title='Neuron'),
            y=alt.Y('Layer:N', title='Layer'),
            color=alt.Color('Activation:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Layer:N', 'Neuron:N', 'Activation:Q']
        ).properties(
            width=800,
            height=300,
            title='Mean Activation Patterns'
        ).interactive()
        
        return heatmap
    
    def _create_gradient_distributions(self, weights_data: List[Dict]) -> List[alt.Chart]:
        """Create gradient distribution visualizations for layers with gradients"""
        distributions = []
        
        for layer_data in weights_data:
            # Skip if no gradients
            if layer_data["grad"] is None:
                continue
                
            # Prepare data for visualization
            grads = layer_data["grad"].flatten()
            layer_name = layer_data["name"]
            
            # Create dataframe
            df = pd.DataFrame({
                'Gradient': grads,
                'Layer': f'Layer {layer_name}'
            })
            
            # Create histogram
            histogram = alt.Chart(df).mark_bar().encode(
                x=alt.X('Gradient:Q', bin=alt.Bin(maxbins=50), title='Gradient Value'),
                y=alt.Y('count()', title='Count'),
                tooltip=['Gradient:Q', 'count()']
            ).properties(
                width=400,
                height=300,
                title=f'Gradient Distribution - Layer {layer_name}'
            ).interactive()
            
            distributions.append(histogram)
        
        return distributions
    
    def _combine_charts(
        self, 
        weight_heatmaps: List[alt.Chart], 
        weight_distributions: List[alt.Chart],
        activation_heatmap: alt.Chart,
        gradient_distributions: List[alt.Chart],
        epoch: int
    ) -> alt.VConcatChart:
        """Combine all charts into a dashboard"""
        # Combine weight heatmaps horizontally
        if weight_heatmaps:
            heatmaps_row = alt.hconcat(*weight_heatmaps)
        else:
            heatmaps_row = alt.Chart().mark_text(text="No weight heatmaps available")
        
        # Combine weight distributions horizontally
        if weight_distributions:
            distributions_row = alt.hconcat(*weight_distributions)
        else:
            distributions_row = alt.Chart().mark_text(text="No weight distributions available")
        
        # Create dashboard components
        dashboard_components = [
            heatmaps_row,
            distributions_row,
            activation_heatmap
        ]
        
        # Add gradient distributions if available
        if gradient_distributions:
            gradients_row = alt.hconcat(*gradient_distributions)
            dashboard_components.append(gradients_row)
        
        # Combine all rows vertically
        dashboard = alt.vconcat(
            *dashboard_components
        ).properties(
            title=f"Neural Network Weight Analysis - Epoch {epoch}"
        )
        
        return dashboard
    
    def _log_dashboard(self, dashboard: alt.TopLevelMixin, epoch: int) -> None:
        """Log the Altair dashboard to MLflow"""
        try:
            # Convert to HTML
            html = dashboard.to_html()
            
            # Create a temporary file and save
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                f.write(html.encode('utf-8'))
                temp_path = f.name
            
            # Log to MLflow
            mlflow.log_artifact(temp_path, f"visualization/weight_dashboard_epoch_{epoch}")
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Failed to log weight dashboard: {str(e)}")