import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from typing import Union, Dict
import yaml


class LHCbMCModule:
    def __init__(
        self,
        train_data_path: Union[str, Path],
        test_data_path: Union[str, Path],
    ) -> None:
        """Initializes the LHCbMCModule with paths to training and testing data."""
        self.train_path = train_data_path
        self.test_path = test_data_path

    @staticmethod
    def _get_features_config_path(filename: str) -> Path:
        """Find configuration file across possible locations."""
        possible_paths = [
            Path(f"config/{filename}"),  # Relative to working directory
            Path(__file__).parent / f"../config/{filename}",  # Relative to this file
            Path(f"/work/submit/blaised/TopoNNOps/config/{filename}")  # Original absolute path
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find {filename} in any expected location")
    
    @staticmethod
    def feature_config(model: str = "TwoBody", feature_config_file: str = "features.yml") -> dict:
        """Read in the user-defined YAML pipeline configuration."""

        assert model in ["TwoBody", "ThreeBody"], f"Model {model} not supported"

        # read in the user-defined YAML pipeline configuration
        with open(
            LHCbMCModule._get_features_config_path(feature_config_file), "r"
        ) as f:
            config = yaml.safe_load(f)

        # return the feature dict
        return config["features"][model]

    @staticmethod
    def get_n_features(model: str = "TwoBody") -> int:
        """Returns the number of features for the specified model"""
        return len(LHCbMCModule.feature_config(model=model).keys())

    def _process_sb_data(
        self, train_data: pd.DataFrame, scale_factor: float = 1.0, ratio: float = 0.1
    ):
        """Setup with controlled background-to-signal ratio and scale factor"""
        # Split background and signal
        background = train_data[train_data["channel"] == "minbias"]
        signal = train_data[train_data["channel"] != "minbias"]

        # Calculate samples to take
        n_background = int(len(background) * scale_factor)
        if ratio: 
            n_signal = int(
                n_background * ratio
            )  # Maintains minbias:<individual signal channel> ratio
        else:
            n_signal = len(signal)

        # Sample the data
        background_sample = background.sample(n=n_background)
        signal_sample = signal.sample(n=n_signal)

        # Combine and create final dataset
        train_data = pd.concat([background_sample, signal_sample])

        return train_data

    def get_channel_information(self, dataset_type='test'):
        """
        Get channel information for training or test dataset.
        
        Args:
            dataset_type: 'train' or 'test' to specify which dataset
            
        Returns:
            Array of channel labels
        """
        if dataset_type == 'train':
            data = pd.read_pickle(self.train_path)
        else:  # test
            data = pd.read_pickle(self.test_path)
        
        return data["channel"].values

    def setup(
        self, batch_size: int = 128, scale_factor: float = 1.0, ratio: float = 0.1
    ):
        """Setup the DataLoaders for training and testing data."""
        train_data = self._process_sb_data(
            pd.read_pickle(self.train_path),
            scale_factor=scale_factor,
            ratio=ratio,
        )
        # we can simply load the test data owing to reduced size
        test_data = self._process_sb_data(
            pd.read_pickle(self.test_path),
            scale_factor=scale_factor,
            ratio=ratio,
        )

        # Store the channel information
        self.train_channels = train_data["channel"].values
        self.test_channels = test_data["channel"].values
        
        # fetch the features
        self.feature_cols = [feat for feat in self.feature_config().keys()]
        assert len(self.feature_cols) > 0, "No features found in the config file"
        # Check if the feature columns are in the train_data keys
        assert all(
            feat in train_data.columns for feat in self.feature_cols
        ), f"Missing features in training data: {[feat for feat in self.feature_cols if feat not in train_data.columns]}"

        # prepare the relevant tensors
        X_train = torch.tensor(
            train_data[self.feature_cols].values, dtype=torch.float32
        )
        y_train = torch.tensor(train_data["class_label"].values, dtype=torch.float32)
        X_test = torch.tensor(test_data[self.feature_cols].values, dtype=torch.float32)
        y_test = torch.tensor(test_data["class_label"].values, dtype=torch.float32)

        # put together the dataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # create the dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )  # shuffling aids the inclusion of both classes in each batch
        self.input_dim = len(self.feature_cols)
