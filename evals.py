import os
import torch
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

from preprocessing import DataPreprocessor
from data import LHCbMCModule

# Import model classes from models.py
from models import LipschitzNet, UnconstrainedNet

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Class to load a trained model, preprocess test data, perform inference,
    and add predictions to the original dataframe.
    """

    def __init__(
        self,
        config_path: str = "/work/submit/blaised/TopoNNOps/config/config.yaml",
        model_path: str = "/work/submit/blaised/TopoNNOps/mlruns/6/f371fdd7316d472ab435812958f851c7/artifacts/model_state_dict.pth",
        feature_config_file: str = "features.yml",
        model_type: str = "ThreeBody",
        device: str = None,
    ):
        """
        Initialize the evaluator with paths to configuration and model.

        Parameters:
        -----------
        config_path: Path to the YAML configuration file
        model_path: Path to the saved model weights
        feature_config_file: Name of the features configuration file
        model_type: Type of model (TwoBody or ThreeBody)
        device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.config_path = config_path
        self.model_path = model_path
        self.feature_config_file = feature_config_file
        self.model_type = model_type

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load configuration
        self.config = self._load_config()

        # Initialize data module
        self.data_module = self._initialize_data_module()

        # Initialize model
        self.model = None

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _initialize_data_module(self) -> LHCbMCModule:
        """Initialize data module with train and test data paths."""
        train_path = self.config["paths"]["train_data"]
        test_path = self.config["paths"]["test_data"]

        logger.info(f"Initializing data module with test data from: {test_path}")

        data_module = LHCbMCModule(train_data_path=train_path, test_data_path=test_path)

        # Setup the data module to initialize preprocessing
        trigger = self.config.get("trigger", "TwoBody")
        data_module.setup(
            batch_size=self.config["training"]["batch_size"],
            scale_factor=self.config["training"]["training_data_scale_factor"],
            ratio=self.config["training"]["sb_ratio"],
            feature_config_file=self.feature_config_file,
            apply_preprocessing=self.config["training"]["apply_preprocessing"],
            balance_train_sample=self.config["training"]["balance_train_sample"],
            model=trigger,
        )

        return data_module

    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from saved weights."""
        logger.info(f"Loading model from {self.model_path}")

        # Get input features dimension from data module
        input_dim = self.data_module.input_dim
        logger.info(f"Model input dimension: {input_dim}")

        # Determine model architecture from config
        model_config = self.config.get("model", {})
        architecture = model_config.get("architecture", "lipschitz_monotonic")
        hidden_dims = model_config.get("hidden_dims", [128, 128, 128, 128, 128])

        # Create appropriate model based on architecture
        if "lipschitz" in architecture:
            # Determine if monotonicity is required
            monotonic = "monotonic" in architecture

            # Get feature names for monotonicity constraints
            feature_names = self.data_module.feature_cols

            # Create a LipschitzNet model
            model = LipschitzNet(
                input_dim=input_dim,
                layer_dims=hidden_dims,
                lip_const=2.0,  # Default value, should be in config ideally
                monotonic=monotonic,
                nbody=self.model_type,
                feature_names=feature_names,
                features_config_path=self.config.get(
                    "features_config_path", "config/features.yml"
                ),
                activation_fn="groupsort",
                lip_kind="nominal",  # FIXME: make this configurable
            )
            logger.info(f"Created LipschitzNet model with monotonicity={monotonic}")
            logger.info(f"{model.print_architecture_details()}")
        else:
            # Create an UnconstrainedNet model
            model = UnconstrainedNet(
                input_dim=input_dim,
                layer_dims=hidden_dims,
                dropout_rate=0.0,
                batch_norm=False,
                activation_fn="relu",
                l1_factor=0.0,
                residual=False,
                init_method="xavier_uniform",
            )
            logger.info(f"Created UnconstrainedNet model")
        try:
            # Load state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model state dict: {e}")

        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model

    def run_inference(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load test data, preprocess it, run inference, and add predictions to the original dataframe.

        Parameters:
        -----------
        output_path: Optional path to save the resulting dataframe as a pickle file

        Returns:
        --------
        pd.DataFrame with original data and added prediction column
        """
        # 1. Load the model if not already loaded
        if self.model is None:
            self.model = self._load_model()

        # 2. Load the raw test data (unprocessed)
        logger.info("Loading raw test data...")
        test_data_path = self.config["paths"]["test_data"]
        test_df = pd.read_pickle(test_data_path)
        logger.info(f"Loaded test data with {len(test_df)} rows")

        # 3. Get preprocessed test features from data module
        feature_cols = self.data_module.feature_cols
        X_test = self.data_module.X_test

        # sanity check: the text-data length should be the same regarless of preprocessing
        assert len(test_df) == len(
            X_test
        ), "Length of test data and preprocessed data should be the same"

        # 4. Run inference using the model
        logger.info("Running inference...")
        with torch.no_grad():
            X_test_tensor = X_test.to(self.device)
            predictions = self.model(X_test_tensor)

            # Apply sigmoid if the model outputs logits
            predictions = torch.sigmoid(predictions)

            # Move back to CPU and convert to numpy
            predictions_np = predictions.cpu().numpy().flatten()

        logger.info(f"Generated {len(predictions_np)} predictions")

        # 5. Add predictions to the original dataframe
        # Make sure the dataframe has the same number of rows as predictions
        if len(test_df) != len(predictions_np):
            logger.warning(
                f"Number of predictions ({len(predictions_np)}) does not match dataframe size ({len(test_df)})"
            )

        # Add predictions column
        test_df["nn_prediction"] = predictions_np
        logger.info("Added predictions to the dataframe")

        # 6. Optionally save the result
        if output_path:
            test_df.to_pickle(output_path)
            logger.info(f"Saved dataframe with predictions to {output_path}")

        return test_df


def main():
    """Run inference and save results."""
    # Define paths
    config_path = "/work/submit/blaised/TopoNNOps/config/config.yaml"
    model_path = "/work/submit/blaised/TopoNNOps/mlruns/6/f371fdd7316d472ab435812958f851c7/artifacts/model_state_dict.pth"
    output_path = None

    # Create output directory if it doesn't exist
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize evaluator and run inference
    evaluator = ModelEvaluator(
        config_path=config_path, model_path=model_path, model_type="ThreeBody"
    )

    result_df = evaluator.run_inference(output_path=output_path)

    # # Print results statistics
    # logger.info(f"Results summary:")
    # logger.info(f"  - Total samples: {len(result_df)}")
    # # ROC AUC
    # roc_auc = evaluator.data_module.compute_roc_auc(result_df)
    # logger.info(f"  - ROC AUC: {roc_auc:.4f}")
    # # Precision-Recall AUC
    # pr_auc = evaluator.data_module.compute_pr_auc(result_df)
    # logger.info(f"  - Precision-Recall AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
