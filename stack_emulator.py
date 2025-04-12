import numpy as np
import json
from typing import List, Optional, Union
import torch

class SigmaNetEmulator:
    """
    Python implementation that emulates the SigmaNet class from the LHCb C++ code.
    This directly mirrors the implementation in paste.txt.
    """
    def __init__(self, json_file: str):
        """
        Initialize the SigmaNet emulator
        
        Args:
            json_file: Path to the JSON file containing the model weights
        """
        # Network parameters
        self.m_variables = []              # Input variables needed by the classifier
        self.m_input_size = 0              # Number of input features
        self.m_n_layers = 0                # Number of layers (not counting input layer)
        self.m_layer_sizes = []            # Width of each layer including input layer [input_size,...,output_size]
        self.m_monotone_constraints = []   # List of monotone constraints (values: -1, 0, 1)
        self.m_lambda = 0.0                # Upper bound on the weight norm
        self.m_weights = []                # Network weights
        self.m_biases = []                 # Network biases
        self.m_clamp_los = []              # Lower clamp values
        self.m_clamp_his = []              # Upper clamp values
        self.m_size = 128                  # Maximum width of the network
        
        # Load the model
        self._load_model(json_file)
    
    def _load_model(self, json_file: str):
        """
        Load model weights and parameters from the JSON file
        
        Args:
            json_file: Path to the JSON file
        """
        try:
            with open(json_file, 'r') as f:
                self.model_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Couldn't open file: {json_file}. Error: {str(e)}")
        
        print(f"Successfully loaded model from {json_file}")
        
        # Extract model structure
        self._extract_model_structure()
    
    def _extract_model_structure(self):
        """Extract the model structure from the loaded JSON data"""
        # Initialize counters and arrays
        num_layer = 0
        total_number_of_layers = 1
        weights_counter = 0
        bias_counter = 0
        
        # Find the number of layers by counting weight matrices
        weight_keys = [k for k in self.model_data.keys() if "weight" in k and "original" in k]
        self.m_n_layers = len(weight_keys)
        print(f"Found {self.m_n_layers} layers in the model")
        
        # Initialize layer sizes array
        self.m_layer_sizes = [0] * (self.m_n_layers + 1)
        
        # Arrays to store weight and bias sizes for each layer
        weight_sizes = [0] * self.m_n_layers
        bias_sizes = [0] * self.m_n_layers
        
        # Process model data
        for key, value in self.model_data.items():
            if "variables_constraints" in key:
                # Extract clamping values for input variables
                for wl in value:
                    self.m_clamp_los.append(wl[0])
                    self.m_clamp_his.append(wl[1])
                print(f"Found {len(self.m_clamp_los)} clamping constraints")
                
            elif "sigmanet.sigma" in key or "model.lipschitz_const" in key:
                # Extract lambda (Lipschitz constant)
                if isinstance(value, list):
                    self.m_lambda = value[0]
                else:
                    self.m_lambda = value
                print(f"Lipschitz constant (lambda): {self.m_lambda}")
                
            elif "weight" in key and "original" in key:
                # Extract weights for a layer
                layer_weights = []
                for wl in value:
                    layer_weights.extend(wl)  # Flatten the weight matrix
                    weights_counter += len(wl)
                    weight_sizes[num_layer] += len(wl)
                
                self.m_weights.extend(layer_weights)
                num_layer += 1
                if num_layer + 1 > total_number_of_layers:
                    total_number_of_layers += 1
                
            elif "bias" in key:
                # Extract biases for a layer
                if isinstance(value, list):
                    self.m_biases.extend(value)
                    bias_counter += len(value)
                    bias_sizes[num_layer] += len(value)
                
            elif "monotonic_constraints" in key or "model.monotonic_constraints" in key:
                # Extract monotonicity constraints
                if isinstance(value, list):
                    if isinstance(value[0], list):
                        # Handle 2D constraints
                        for v in value:
                            self.m_monotone_constraints.extend(v)
                    else:
                        # Handle 1D constraints
                        self.m_monotone_constraints.extend(value)
                print(f"Monotonicity constraints: {self.m_monotone_constraints}")
        
        # Calculate input size from the constraints or first layer
        if self.m_clamp_los:
            self.m_input_size = len(self.m_clamp_los)
        elif self.m_monotone_constraints:
            self.m_input_size = len(self.m_monotone_constraints)
        else:
            # If no constraints, try to infer from first layer weights
            weights_key = next((k for k in weight_keys if "00." in k or ".0." in k), None)
            if weights_key and self.model_data[weights_key]:
                self.m_input_size = len(self.model_data[weights_key][0])
        
        print(f"Input size: {self.m_input_size}")
        
        # Calculate layer sizes
        for i in range(self.m_n_layers):
            if bias_sizes[i] > 0:
                self.m_layer_sizes[i] = weight_sizes[i] // bias_sizes[i]
        
        # Set output layer size
        self.m_layer_sizes[self.m_n_layers] = 1
        
        print(f"Layer sizes: {self.m_layer_sizes}")
        
        # Validate parameters
        total_parameters = 0
        for layer_idx in range(1, self.m_n_layers + 1):
            n_inputs = self.m_layer_sizes[layer_idx - 1]
            n_outputs = self.m_layer_sizes[layer_idx]
            total_parameters += n_outputs * n_inputs  # Weights
            total_parameters += n_outputs             # Biases
        
        if total_parameters != (weights_counter + bias_counter):
            print(f"Warning: Mismatch between weight_file ({weights_counter + bias_counter} Parameters) "
                  f"and specified architecture ({total_parameters} Parameters)")
        
        # Fill monotonicity constraints if not provided
        if not self.m_monotone_constraints and self.m_input_size > 0:
            # Default to all ones (monotonically increasing)
            self.m_monotone_constraints = [1] * self.m_input_size
    
    def __call__(self, values: Union[List[float], np.ndarray]) -> float:
        """
        Evaluate the network on the given input values
        
        Args:
            values: Input values, should match the expected input size
            
        Returns:
            float: Network output (probability between 0 and 1)
        """
        # Convert input to numpy array if needed
        if isinstance(values, list):
            values = np.array(values)
        
        # Check input dimension
        if len(values) != self.m_input_size:
            raise ValueError(f"Input size mismatch: expected {self.m_input_size}, got {len(values)}")
        
        return self._evaluate(values)
    
    def _evaluate(self, values: np.ndarray) -> float:
        """
        Implementation of the evaluate method from the C++ code
        
        Args:
            values: Input values
            
        Returns:
            float: Network output (probability between 0 and 1)
        """
        # Clone values to avoid modifying the input
        input_values = values.copy()
        
        # Apply clamping if available
        if self.m_clamp_los and len(self.m_clamp_los) == len(input_values):
            for i in range(len(input_values)):
                input_values[i] = np.clip(input_values[i], self.m_clamp_los[i], self.m_clamp_his[i])
        
        # Storage for intermediate values
        storage = np.zeros(2 * self.m_size)
        
        # Helper functions to manage storage
        def input_for(layer):
            assert layer > 0 and layer <= len(self.m_layer_sizes)
            start_idx = self.m_size * ((layer - 1) % 2)
            return storage[start_idx:start_idx + self.m_layer_sizes[layer - 1]]
        
        def output_for(layer):
            return input_for(layer + 1)
        
        # Copy input values to first layer storage
        input_layer = input_for(1)
        for i in range(len(input_values)):
            input_layer[i] = input_values[i]
        
        # Process each layer
        weight_offset = 0
        bias_offset = 0
        
        for layer in range(1, self.m_n_layers + 1):
            input_layer = input_for(layer)
            output_layer = output_for(layer)
            
            # Get layer dimensions
            n_inputs = self.m_layer_sizes[layer - 1]
            n_outputs = self.m_layer_sizes[layer]
            
            # Get weights and biases for this layer
            layer_weights = self.m_weights[weight_offset:weight_offset + n_inputs * n_outputs]
            layer_biases = self.m_biases[bias_offset:bias_offset + n_outputs]
            
            # Update offsets
            weight_offset += n_inputs * n_outputs
            bias_offset += n_outputs
            
            # Compute output = bias + weight * input
            for i in range(n_outputs):
                output_layer[i] = layer_biases[i]
                for j in range(n_inputs):
                    output_layer[i] += layer_weights[i * n_inputs + j] * input_layer[j]
            
            # Apply GroupSort2 activation if not last layer
            if layer != self.m_n_layers:
                self._groupsort2(output_layer[:n_outputs])
        
        # Get final response
        response = output_for(self.m_n_layers)[0]
        
        # Add monotonicity term (lambda * dot(inputs, constraints))
        monotone_term = 0.0
        for i in range(len(values)):
            if i < len(self.m_monotone_constraints):
                monotone_term += values[i] * self.m_monotone_constraints[i]
        
        response += self.m_lambda * monotone_term
        
        # Apply sigmoid activation
        return self._sigmoid(response)
    
    def _groupsort2(self, x: np.ndarray) -> None:
        """
        GroupSort-2 activation function (in-place)
        Sorts each consecutive pair of elements
        
        Args:
            x: Input array to sort in-place
        """
        for i in range(0, len(x) - 1, 2):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function
        
        Args:
            x: Input value
            
        Returns:
            float: Sigmoid(x)
        """
        return 1.0 / (1.0 + np.exp(-x))


# Simple function to test the SigmaNet emulator
def test_sigmanet(json_file: str, input_values: Optional[List[float]] = None) -> float:
    """
    Test the SigmaNet emulator with a model and input values
    
    Args:
        json_file: Path to the JSON model file
        input_values: Input values (randomly generated if None)
        
    Returns:
        float: Model output
    """
    # Create the SigmaNet emulator
    model = SigmaNetEmulator(json_file)
    
    # Generate random input if not provided
    if input_values is None:
        np.random.seed(42)  # For reproducibility
        input_values = np.random.rand(model.m_input_size).tolist()
        print(f"Generated random input: {input_values}")
    
    # Evaluate the model
    output = model(input_values)
    print(f"Model output: {output}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test a model with the LHCb SigmaNet emulator")
    parser.add_argument("-m", "--model", help="Path to the JSON model file", default="prod_model_twobody.json")
    parser.add_argument("-i", "--input", type=float, nargs="+", help="Input values (space-separated)", 
        default=[
            0.0446527, 
            0.24048, 
            0.271239, 
            0.617939, 
            0.243267, 
            0.0357454, 
            0.70625, 
            0.480977, 
            0.853921
        ]
    )
    
    args = parser.parse_args()
    
    # Test the model
    test_sigmanet(args.model, args.input)

    # test read in
    from model_persistence import load_from_pt
    model = load_from_pt("/work/submit/blaised/TopoNNOps/mlruns/3/a77b1e292859425c850882df170f1772/artifacts/model_state_dict.pt", apply_normalization=False)
    output = torch.sigmoid(model(torch.tensor(args.input).unsqueeze(0)))
    print(f"Loaded model from pt output: {output}")

