import torch
import json
import monotonicnetworks as lmn
from models import LipschitzNet
from data import LHCbMCModule
from typing import Dict, Optional, List

NBODY_MODEL = "TwoBody"
NN_SCHEMA = "nominal"


def load_into_lipnn(filename, hidden_layer_dims=[128, 128, 128, 128, 128, 128, 128]):
    """
    Load a model with the standard architecture (n layers with Lipschitz constraints).
    
    Args:
        filename: PyTorch checkpoint file (.pt)
        apply_normalization: Whether to explicitly normalize weights post-loading
        hidden_layer_dims: Dimensions for hidden layers
    
    Returns:
        A PyTorch LipschitzNet model with weights properly loaded
    """
    print(f"\nLoading model from PyTorch checkpoint: {filename}")
    
    # Load the state dict
    state_dict = torch.load(filename)

    # Extract latent complexity and lipschitz assignment from model metadata
    try:
        hidden_size = state_dict['model.nn.0.parametrizations.weight.original'].shape[0]
        input_dim = state_dict['model.nn.0.parametrizations.weight.original'].shape[1]
        lipschitz_const = state_dict['model.lipschitz_const'] # NOTE: assume global Lipschitz constant
        monotonic_constraints = state_dict['model.monotonic_constraints']
    except Exception as e:
        raise ValueError(f"Could not extract hidden size and input size from the first layer weights: {e} -- nominally assume same latent complexity")

    # Create new model
    model = LipschitzNet(
        input_dim=input_dim, 
        layer_dims=hidden_layer_dims, 
        lip_const=lipschitz_const.item(),
        monotonic=True if any(monotonic_constraints) else False,
        nbody=NBODY_MODEL,
        feature_names=list(LHCbMCModule.feature_config(model=NBODY_MODEL).keys()),
        lip_kind=NN_SCHEMA
    )

    # Load the weights into the model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded weights into model")
    except Exception as e:
        raise ValueError(f"Error during weight loading: {e}")

    return model


def assign_constrains(constrain_path: str, state_dict: Dict) -> Dict:
    """Pass the constrains from the M-scaling to the state_dict

    Parameters
    ----------
    constrain_path : str
        path to the location of the constraints file
    state_dict
        state_dict of the pytorch model that needs to be modified
    Returns
    -------
    Dict
        modified state_dict model
    """
    variables_constraints = []

    with open(f"{constrain_path}", "r") as file:
        next(file)  # skip the first line as it is an information statement
        for line in file:
            if ":" not in line:  # get the lines with min: value and max: value
                continue
            values = line.replace("\n", "").split(":")[
                -1
            ]  # make stringsplit for values
            values = float(values)  # convert values from string to floats
            variables_constraints.append(values)

    variables_constraints = list(
        make_chunks(variables_constraints, 2)
    )  # chunk it into lists with size 2: (min, max)

    print(f"Assigning constraints: {variables_constraints}")
    state_dict.update({"variables_constraints": variables_constraints})
    state_dict.move_to_end(
        "variables_constraints", last=False
    )  # Need to be at the beginning of the file

    return state_dict


def assign_sigmanet_label(state_dict: Dict) -> Dict:
    """Rename keys according to stack requirements, beware of OrderedDict

    Parameters
    ----------
    state_dict
        state_dict of the pytorch model that needs to be modified
    Returns
    -------
    Dict
        modified state_dict model
    """
    production_dictionary = {} # what gets portd to the LHCb stack


    # Find all weight keys in the state dict that need normalization
    weight_keys = [k for k in state_dict.keys() if '.parametrizations.weight.original' in k]

    # load the global Lipschitz constant
    lipschitz_const = state_dict['model.lipschitz_const']

    # log into production dictionary
    production_dictionary['sigmanet.sigma'] = [lipschitz_const.item()]

    # Apply appropriate normalization to each layer's weights
    for i, key in enumerate(weight_keys):

        # extract the layer number and format with leading zero if single digit - for the stack 
        layer_num_str = (lambda k: f"{int(next(part for part in k.split('.') if part.isdigit())):02d}")(key) # eg 00, 01, 02, ...
    

        # fetch the corresponding bias and assign to the production dictionary; no normalisation required
        production_dictionary[f'sigmanet.nn.{layer_num_str}.bias'] = state_dict[key.replace('.parametrizations.weight.original', '.bias')]

        # verify compatibility with the global lipschitz constant, layer-wise
        assert state_dict[key.replace('.parametrizations.weight.original', '.lipschitz_const')] == lipschitz_const.item(), "Layer-wise Lipschitz constant does not match global Lipschitz constant"

        # Determine normalization type based on layer position -- nominal prescription
        if i == 0:
            norm_type = "one-inf"  # First layer 
        elif i == len(weight_keys) - 1:
            norm_type = "one"      # Last layer
            assert state_dict[key].shape[0] == 1, "Last layer must have 1 output dimension"
        else:
            norm_type = "inf"      # Middle layers

        # Apply normalization using monotonicnetworks library
        normalized_weight = lmn.get_normed_weights(
            state_dict[key],
            kind=norm_type,
            always_norm=False,
            max_norm=lipschitz_const,
            vectorwise=True  # Assuming vectorwise normalization as in original code
        )

        # log the normalised weights into the production dictionary
        production_dictionary[f'sigmanet.nn.{layer_num_str}.parametrizations.weight.original'] = normalized_weight

        print(f"Layer number: {layer_num_str}; index: {i}")
        print(f"Normalization type: {norm_type}")
        print(f"Normalized weight shape: {normalized_weight.shape}")
        print(f"Normalized weight: {normalized_weight}")
        print(f"Bias vector shape: {production_dictionary[f'sigmanet.nn.{layer_num_str}.bias'].shape}")

    return production_dictionary


def export_to_json(state_dict: Dict, output_path: str) -> None:
    """Save the processed state dictionary to JSON

    Parameters
    ----------
    state_dict : Dict
        Processed state dictionary
    output_path : str
        Path to save the JSON file
    """
    # Convert tensors to Python types for JSON serialization
    serializable_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            serializable_dict[key] = value.cpu().tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            # Handle lists of tensors (like for constraints)
            serializable_dict[key] = [v.cpu().tolist() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            serializable_dict[key] = value
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(serializable_dict, f, indent=2)
    
    print(f"Model successfully exported to {output_path}")



if __name__ == "__main__":
    # import os
    # import torch.nn as nn
    # from torch.optim import Adam
    # import matplotlib.pyplot as plt
    
    # # Create a simple dataset (similar to stress_test_lipschitz.py)
    # def generate_dummy_data(lo=0, hi=1, num_points=900):
    #     x = torch.linspace(lo, hi, num_points).reshape(-1, 9)  # 2D input
    #     y = x[:, 0]**2 + x[:, 1]**2  # Simple function
    #     return x, y.reshape(-1, 1)
    
    # print("Step 1: Creating and training a model on dummy data")
    # x, y = generate_dummy_data()
    # input_dim = x.shape[1]
    
    # # Create a LipschitzNet model (don't call _build_model yet)
    # original_model = LipschitzNet(
    #     input_dim=input_dim, 
    #     layer_dims=[8, 8, 8],  # Smaller network for testing
    #     lip_const=2.0,
    #     monotonic=True,
    #     nbody=NBODY_MODEL,
    #     feature_names=list(LHCbMCModule.feature_config(model='TwoBody').keys()), # mock up a two-body model
    #     lip_kind=NN_SCHEMA
    # )
    
    # # This will preserve the model structure with 'model.' prefix
    # built_model = original_model._build_model()
    
    # # Train for a few epochs
    # criterion = nn.MSELoss()
    # optimizer = Adam(built_model.parameters(), lr=0.01)
    
    # for epoch in range(1000):
    #     # Forward pass
    #     y_pred = built_model(x)
    #     loss = criterion(y_pred, y)
        
    #     # Backward and optimize
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # # Save the model state dict - capturing full structure
    # checkpoint_path = "dummy_test_model_checkpoint.pt"
    # torch.save(built_model.state_dict(), checkpoint_path)
    # print(f"\nModel saved to {checkpoint_path}")
    
    # # Print the keys to verify structure
    # saved_dict = built_model.state_dict()
    # print("\nSaved state dict keys:")
    # for key in list(saved_dict.keys())[:5]:  # Just print a few for brevity
    #     print(f"  {key}")
    
    # # Store original predictions
    # with torch.no_grad():
    #     original_preds = built_model(x)
    
    # # Get original weights for comparison
    # original_state_dict = built_model.state_dict()
    
    # print("\nStep 2: Loading the model using load_from_pt")
    # loaded_model = load_from_pt(checkpoint_path, hidden_layer_dims=[8, 8, 8])

    # # Compare the weights and predictions
    # print("\nStep 3: Comparing original and loaded models")
    
    # # Compare predictions
    # with torch.no_grad():
    #     loaded_preds = loaded_model(x)
    
    # pred_diff = (original_preds - loaded_preds).abs().max().item()
    # print(f"Maximum difference in predictions: {pred_diff:.6f}")
    
    # # Compare weights
    # loaded_state_dict = loaded_model.state_dict()
    # print("\nComparing key state dict parameters:")
    
    # print("\nOriginal model keys:")
    # for key in list(original_state_dict.keys())[:5]:
    #     print(f"  {key}")
        
    # print("\nLoaded model keys:")
    # for key in list(loaded_state_dict.keys())[:5]:
    #     print(f"  {key}")
    
    # # Find common keys for comparison
    # original_set = set(original_state_dict.keys())
    # loaded_set = set(loaded_state_dict.keys())
    
    # print(f"\nKeys in original but not in loaded: {len(original_set - loaded_set)}")
    # print(f"Keys in loaded but not in original: {len(loaded_set - original_set)}")
    
    # # Create corresponding key mappings
    # orig_to_loaded = {}
    # for orig_key in original_state_dict.keys():
    #     # Try with and without model prefix
    #     if orig_key in loaded_state_dict:
    #         orig_to_loaded[orig_key] = orig_key
    #     elif "model." + orig_key in loaded_state_dict:
    #         orig_to_loaded[orig_key] = "model." + orig_key
    #     elif orig_key.startswith("model.") and orig_key[6:] in loaded_state_dict:
    #         orig_to_loaded[orig_key] = orig_key[6:]
    
    # print(f"\nFound {len(orig_to_loaded)} matching keys for comparison")
    
    # # Compare weights for matching keys
    # for orig_key, load_key in list(orig_to_loaded.items())[:5]:
    #     weight_diff = (original_state_dict[orig_key] - loaded_state_dict[load_key]).abs().max().item()
    #     print(f"  {orig_key} vs {load_key}: Max diff = {weight_diff:.6f}")
    
    # # Plot predictions for visualization
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(y)), y.numpy(), alpha=0.3, label='True data')
    # plt.plot(range(len(original_preds)), original_preds.numpy(), 'r-', 
    #          linewidth=2, label='Original model')
    # plt.plot(range(len(loaded_preds)), loaded_preds.numpy(), 'g--', 
    #          linewidth=2, label='Loaded model')
    # plt.legend()
    # plt.title('Model Predictions Before and After Export/Load')
    # plt.savefig('model_predictions_comparison.png')
    
    # # Clean up the temporary checkpoint
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     print(f"\nRemoved test checkpoint: {checkpoint_path}")

    # # Production model loading and export - uncomment to test
    print("\nLoading production model:")
    #trained_state_dict = torch.load("/work/submit/blaised/TopoNNOps/mlruns/3/a77b1e292859425c850882df170f1772/artifacts/model_state_dict.pt") # twobody nominal
    trained_state_dict = torch.load("/work/submit/blaised/TopoNNOps/mlruns/5/0048a199fc4b41079083ce201b598d95/artifacts/model_state_dict.pt") # twobody nominal

    # Adjust the labels according to stack requirements
    prod_state_dict = assign_sigmanet_label(state_dict=trained_state_dict)

    # Export the production model to JSON
    export_to_json(state_dict=prod_state_dict, output_path=f"prod_model_{NBODY_MODEL}.json")
    print("Production model exported successfully!")