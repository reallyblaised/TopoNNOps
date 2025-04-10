import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import monotonicnetworks as lmn
import copy
import json

# Generate the data
def generate_data(num_points=1000):
    x = torch.linspace(0, 1, num_points).view(-1, 1)
    y = x**2
    return x, y

# Define a network with Lipschitz constraints
class LipschitzNN(nn.Module):
    def __init__(self, hidden_size=8, lipschitz_const=2.0):
        super(LipschitzNN, self).__init__()
        
        # Create sequential network with Lipschitz constraints
        self.lip_nn = nn.Sequential(
            lmn.LipschitzLinear(1, hidden_size, lipschitz_const=lipschitz_const, kind="one-inf"),
            lmn.GroupSort(hidden_size//2),
            lmn.LipschitzLinear(hidden_size, hidden_size, lipschitz_const=lipschitz_const, kind="inf"),
            lmn.GroupSort(hidden_size//2),
            lmn.LipschitzLinear(hidden_size, 1, lipschitz_const=lipschitz_const, kind="one"),
        )
        
        # Wrap with monotonicity constraints (first input increasing, no constraints on second)
        self.model = lmn.MonotonicWrapper(self.lip_nn, monotonic_constraints=[1])
    
    def forward(self, x):
        return self.model(x)

# Function to inspect weights
def inspect_weights(model, epoch, after_step=False):
    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch} - {'After' if after_step else 'Before'} optimization step")
    
    # Get weights from the model's forward computation
    with torch.no_grad():
        # Feed a dummy input to trigger forward hooks
        dummy_input = torch.zeros((1, 1)) # scalar input

        _ = model(dummy_input)

        # Now inspect the weights actually used in the forward pass
        print("\nLayer 1 (one-inf constraint):")
        forward_weights_layer1 = model.lip_nn[0].weight.clone()
        print(f"  Shape: {forward_weights_layer1.shape}")
        print(f"  Min: {forward_weights_layer1.min().item():.6f}, Max: {forward_weights_layer1.max().item():.6f}")
        print(f"  Abs Sum (L1 norm along axis=0): {forward_weights_layer1.abs().sum(axis=0)}")
        
        print("\nLayer 2 (inf constraint):")
        forward_weights_layer2 = model.lip_nn[2].weight.clone()
        print(f"  Shape: {forward_weights_layer2.shape}")
        print(f"  Min: {forward_weights_layer2.min().item():.6f}, Max: {forward_weights_layer2.max().item():.6f}")
        print(f"  Abs Sum (L1 norm along axis=1): {forward_weights_layer2.abs().sum(axis=1)}")
        
        print("\nLayer 3 (one constraint):")
        forward_weights_layer3 = model.lip_nn[4].weight.clone()
        print(f"  Shape: {forward_weights_layer3.shape}")
        print(f"  Min: {forward_weights_layer3.min().item():.6f}, Max: {forward_weights_layer3.max().item():.6f}")
        print(f"  Abs Sum (L1 norm along axis=0): {forward_weights_layer3.abs().sum(axis=0)}")
    
    # Get weights from state_dict
    state_dict = model.state_dict()
    
    print("\nState Dict Keys:")
    for key in state_dict.keys():
        print(f"  {key}")
    
    # Find the correct weight keys
    lip_nn_prefix = 'model.nn.'
    weight_keys = [k for k in state_dict.keys() if 'weight' in k and lip_nn_prefix in k]
    
    print(f"\nFound {len(weight_keys)} weight keys in state_dict:")
    for key in weight_keys:
        print(f"  {key}")
    
    # Access the correct weights from state_dict
    layer1_weight_key = lip_nn_prefix + '0.parametrizations.weight.original'
    layer2_weight_key = lip_nn_prefix + '2.parametrizations.weight.original'
    layer3_weight_key = lip_nn_prefix + '4.parametrizations.weight.original'
    
    if not all([layer1_weight_key, layer2_weight_key, layer3_weight_key]):
        print("WARNING: Couldn't find all weight keys!")
        return None
    
    print("\nState Dict Weights:")
    print(f"Layer 1 weight in state_dict shape: {state_dict[layer1_weight_key].shape}")
    print(f"Layer 2 weight in state_dict shape: {state_dict[layer2_weight_key].shape}")
    print(f"Layer 3 weight in state_dict shape: {state_dict[layer3_weight_key].shape}")
    
    # Compare with what's in the state_dict
    try:
        print("\nDifference between forward pass weights and state_dict weights:")
        print(f"Layer 1 max diff: {(forward_weights_layer1 - state_dict[layer1_weight_key]).abs().max().item():.6f}")
        print(f"Layer 2 max diff: {(forward_weights_layer2 - state_dict[layer2_weight_key]).abs().max().item():.6f}")
        print(f"Layer 3 max diff: {(forward_weights_layer3 - state_dict[layer3_weight_key]).abs().max().item():.6f}")
    except ValueError as e:
        print(f"Error comparing weights: {e}")
        print("This suggests that the parametrized weights in the forward pass are different from what's stored in state_dict")
    
    print(f"{'=' * 50}\n")
    
    return {
        'forward': {
            'layer1': forward_weights_layer1.clone().detach(),
            'layer2': forward_weights_layer2.clone().detach(),
            'layer3': forward_weights_layer3.clone().detach(),
        },
        'state_dict': {
            'layer1': state_dict[layer1_weight_key].clone().detach() if layer1_weight_key else None,
            'layer2': state_dict[layer2_weight_key].clone().detach() if layer2_weight_key else None,
            'layer3': state_dict[layer3_weight_key].clone().detach() if layer3_weight_key else None,
        },
        'weight_keys': {
            'layer1': layer1_weight_key,
            'layer2': layer2_weight_key,
            'layer3': layer3_weight_key,
        }
    }

# Function to train the model and inspect weights
def train_model_with_inspection(model, x, y, epochs=10, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    weight_history = []
    
    for epoch in range(epochs):
        # Inspect weights before optimization step      
        weight_info_before = inspect_weights(model, epoch, after_step=False)
        
        # Forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        print(f"Loss: {loss.item():.6f}")
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Inspect weights after optimization step
        weight_info_after = inspect_weights(model, epoch, after_step=True)
        
        weight_history.append({
            'epoch': epoch,
            'before_step': weight_info_before,
            'after_step': weight_info_after,
            'loss': loss.item()
        })
        
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    return weight_history

# Function to export model correctly, applying normalization during export
def export_model(model, filename="lipschitz_model.json", lipschitz_const=2.0, vectorwise=True):
    """
    Export model weights with correct normalization applied.
    
    Args:
        model: The trained Lipschitz neural network
        filename: Output JSON filename
        lipschitz_const: The Lipschitz constant to enforce
        vectorwise: Whether to apply constraints vectorwise (True) or to the entire matrix (False)
    """
    print("\nExporting model with normalized weights...")
    
    # Get the state dictionary
    state_dict = model.state_dict()
    
    # Find all weight keys
    weight_keys = [k for k in state_dict.keys() if 'weight' in k]
    depth = len(weight_keys)
    
    # Sort weight keys to ensure first layer is first, etc.
    weight_keys.sort()  # This assumes keys follow a consistent naming pattern
    
    print(f"Found {depth} weight layers to normalize")
    
    # Create a copy of the state dict to modify
    export_dict = {}
    
    # Process each parameter in the state dictionary
    for k, v in state_dict.items():
        # Convert to CPU tensor if necessary
        v_cpu = v.cpu() if isinstance(v, torch.Tensor) else v
        
        if k in weight_keys:
            # Determine which normalization to apply
            # First layer uses one-inf, others use inf
            layer_index = weight_keys.index(k)
            norm_type = "one-inf" if layer_index == 0 else "inf"
            if layer_index == depth - 1:  # Last layer
                norm_type = "one"  # For scalar output
            
            print(f"Normalizing {k} with {norm_type} norm")
            # Apply the appropriate normalization
            normalized_weight = lmn.get_normed_weights(
                v_cpu,
                kind=norm_type,
                always_norm=False,  # Always normalize to the max_norm
                max_norm=lipschitz_const, # ** (1.0 / depth),  # Each layer gets equal contribution
                vectorwise=vectorwise
            )
            
            # Store the normalized weight
            export_dict[k] = normalized_weight
        else:
            # Store other parameters as-is
            export_dict[k] = v_cpu
    
    # Convert all tensors to lists for JSON serialization
    for k in export_dict.keys():
        if isinstance(export_dict[k], torch.Tensor):
            export_dict[k] = export_dict[k].tolist()
    
    # Add metadata
    export_dict["lipschitz_const"] = lipschitz_const
    export_dict["model_type"] = "lipschitz_nn"
    export_dict["architecture"] = {
        "hidden_size": model.lip_nn[0].weight.shape[0],
        "depth": depth,
        "norm_types": ["one-inf"] + ["inf"] * (depth - 2) + ["one"] if depth > 2 else ["one-inf", "one"]
    }
    
    # Save to JSON
    with open(filename, "w") as f:
        json.dump(export_dict, f)
    
    print(f"Model exported to {filename}")
    return export_dict

# Function to load model correctly, applying normalization during loading
def load_model_normalized(filename, hidden_size=8, lipschitz_const=2.0):
    """
    Load a model from a JSON file, applying the appropriate normalizations.
    
    Args:
        filename: JSON file containing model weights
        hidden_size: Size of hidden layers
        lipschitz_const: Lipschitz constant to enforce
    
    Returns:
        Loaded model with properly normalized weights
    """
    print(f"\nLoading model from {filename}...")

    # Create a new model with the same architecture
    model = LipschitzNN(hidden_size=hidden_size, lipschitz_const=lipschitz_const)
    
    # Load the state dict from JSON
    with open(filename, "r") as f:
        state_dict_data = json.load(f)

    # Extract model metadata if present
    if "architecture" in state_dict_data:
        arch = state_dict_data["architecture"]
        print(f"Found architecture metadata: {arch}")
        hidden_size = arch.get("hidden_size", hidden_size)
        # Could recreate model with detected hidden_size if needed
    
    # Filter out metadata keys
    metadata_keys = ["lipschitz_const", "model_type", "architecture"]
    param_keys = [k for k in state_dict_data.keys() if k not in metadata_keys]
    
    # Create PyTorch state dict
    state_dict = {}
    for k in param_keys:
        if isinstance(state_dict_data[k], list):
            # Convert lists back to tensors
            try:
                if any(isinstance(item, list) for item in state_dict_data[k]):
                    # 2D list
                    state_dict[k] = torch.tensor(state_dict_data[k], dtype=torch.float32)
                else:
                    # 1D list
                    state_dict[k] = torch.tensor(state_dict_data[k], dtype=torch.float32)
            except ValueError as e:
                print(f"Error converting {k} to tensor: {e}")
                print(f"Data: {state_dict_data[k][:5]}...")
        else:
            # Keep as is
            state_dict[k] = state_dict_data[k]
    
    # Load state dict into model
    # This might fail if the keys don't match exactly
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded state dict directly")
    except Exception as e:
        print(f"Error loading state dict directly: {e}")
        print("Trying alternative loading approach...")
        
        # Find weight keys to match by shape
        weight_keys = [k for k in state_dict.keys() if 'weight' in k]
        
        # Find the corresponding layer by matching the shape
        # This assumes the layers have unique shapes
        for k in weight_keys:
            weight = state_dict[k]
            if weight.shape == model.layer1.weight.shape:
                print(f"Matching {k} to layer1.weight")
                model.layer1.weight.data.copy_(weight)
            elif weight.shape == model.layer2.weight.shape:
                print(f"Matching {k} to layer2.weight")
                model.layer2.weight.data.copy_(weight)
            elif weight.shape == model.layer3.weight.shape:
                print(f"Matching {k} to layer3.weight")
                model.layer3.weight.data.copy_(weight)
        
        # Handle bias terms similarly
        bias_keys = [k for k in state_dict.keys() if 'bias' in k]
        for k in bias_keys:
            bias = state_dict[k]
            if bias.shape == model.layer1.bias.shape:
                print(f"Matching {k} to layer1.bias")
                model.layer1.bias.data.copy_(bias)
            elif bias.shape == model.layer2.bias.shape:
                print(f"Matching {k} to layer2.bias")
                model.layer2.bias.data.copy_(bias)
            elif bias.shape == model.layer3.bias.shape:
                print(f"Matching {k} to layer3.bias")
                model.layer3.bias.data.copy_(bias)
    
    print("Model loaded successfully")
    return model

# Function to test our model export and load workflow
def test_export_load(model, x, export_filename="lipschitz_model.json"):
    print("\nTesting model export and load workflow...")
    
    # Get predictions from original model
    with torch.no_grad():
        original_preds = model(x)
    
    # Export model with proper normalization
    export_dict = export_model(model, filename=export_filename)
    
    # Load model with normalization applied
    loaded_model = load_model_normalized(
        export_filename, 
        hidden_size=model.lip_nn[0].weight.shape[0], # assumes all layers have same hidden_size
        lipschitz_const=2.0
    )
    
    # Get predictions from loaded model
    with torch.no_grad():
        loaded_preds = loaded_model(x)
    
    # Compare predictions
    diff = (original_preds - loaded_preds).abs().max().item()
    print(f"Maximum difference in predictions after export/load: {diff:.6f}")
    
    if diff > 1e-5:
        print("WARNING: Significant difference detected after export/load!")
        
        # Additional diagnostics
        print("\nDiagnostics:")
        
        # Check if the Lipschitz constraints are maintained
        with torch.no_grad():
            # Check gradients - create a small perturbation and see how outputs change
            x1 = x.clone()
            x2 = x.clone() + 0.01
            
            y1_orig = model(x1)
            y2_orig = model(x2)
            
            y1_load = loaded_model(x1)
            y2_load = loaded_model(x2)
            
            # Calculate empirical Lipschitz constant
            lip_orig = ((y2_orig - y1_orig).abs() / (x2 - x1).abs()).max().item()
            lip_load = ((y2_load - y1_load).abs() / (x2 - x1).abs()).max().item()
            
            print(f"Empirical Lipschitz constant (original): {lip_orig:.6f}")
            print(f"Empirical Lipschitz constant (loaded): {lip_load:.6f}")
    else:
        print("Export/load successful: predictions match within tolerance.")
    
    return original_preds, loaded_preds, loaded_model

# Main execution
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate data
    x, y = generate_data(1000)
    
    # Create model
    model = LipschitzNN(hidden_size=8, lipschitz_const=2.0)
    
    # Train and inspect
    weight_history = train_model_with_inspection(model, x, y, epochs=5)
    
    # Test export and load workflow (our corrected approach)
    original_preds, loaded_preds, loaded_model = test_export_load(model, x)

    # Plot predictions from both original and loaded model
    plt.figure(figsize=(10, 6))
    plt.scatter(x.numpy(), y.numpy(), alpha=0.3, label='True data')
    plt.plot(x.numpy(), original_preds.numpy(), 'r-', linewidth=2, label='Original model')
    plt.plot(x.numpy(), loaded_preds.numpy(), 'g--', linewidth=2, label='Loaded model')
    plt.legend()
    plt.title('Model Predictions Before and After Export/Load')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('model_predictions_before_after_export_load.png')
    
    # # As an additional verification, test the empirical Lipschitz constant
    x_test = torch.linspace(0, 1, 1000).view(-1, 1)
    with torch.no_grad():
        y_test = model(x_test)
        
    # Calculate numerical gradient
    dx = 0.001
    x_plus = x_test + dx
    with torch.no_grad():
        y_plus = model(x_plus)
    
    numerical_grad = (y_plus - y_test) / dx
    max_grad = numerical_grad.abs().max().item()
    
    print(f"\nEmpirical maximum gradient magnitude: {max_grad:.6f}")
    print(f"Expected Lipschitz constant: 2.0")
    
    # The same test for loaded model
    with torch.no_grad():
        y_test_loaded = loaded_model(x_test)
        y_plus_loaded = loaded_model(x_plus)
    
    numerical_grad_loaded = (y_plus_loaded - y_test_loaded) / dx
    max_grad_loaded = numerical_grad_loaded.abs().max().item()
    
    print(f"Empirical maximum gradient magnitude (loaded model): {max_grad_loaded:.6f}")

if __name__ == "__main__":
    main()