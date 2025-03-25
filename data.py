import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from typing import Union, Dict
import yaml
from torch.utils.data.distributed import DistributedSampler
from preprocessing import DataPreprocessor


class LHCbMCModule:
    def __init__(
        self,
        train_data_path: Union[str, Path],
        test_data_path: Union[str, Path],
    ) -> None:
        """Initializes the LHCbMCModule with paths to training and testing data."""
        self.train_path = train_data_path
        self.test_path = test_data_path
        self.preprocessor = None  # Will be initialized in setup

    @staticmethod
    def _get_features_config_path(filename: str) -> Path:
        """Find configuration file across possible locations."""
        possible_paths = [
            Path(f"{filename}"),  # Relative to working directory
            Path(f"config/{filename}"),  # Relative to working directory
            Path(__file__).parent / f"../config/{filename}",  # Relative to this file
            Path(__file__).parent / f"../{filename}",  # Relative to this file
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
    def transforms_config(model: str = "TwoBody", feature_config_file: str = "features.yml") -> dict:
        """Read in the transformation configuration for preprocessing."""
        assert model in ["TwoBody", "ThreeBody"], f"Model {model} not supported"

        # read in the user-defined YAML pipeline configuration
        with open(
            LHCbMCModule._get_features_config_path(feature_config_file), "r"
        ) as f:
            config = yaml.safe_load(f)

        # Check if transforms section exists
        transforms = {}
        if "transforms" in config:
            key= "transforms" if "transforms" in config else "preprocessing" # for backwards compatibility
            
            # Fix possible spelling differences (TwoBoody vs TwoBody)
            model_key = model
            if model == "TwoBody" and "TwoBoody" in config[key]:
                model_key = "TwoBoody"
            elif model == "ThreeBody" and "ThreeBoody" in config[key]:
                model_key = "ThreeBoody"
                
            if model_key in config[key]:
                transforms = config[key][model_key]
        
        return transforms

    @staticmethod
    def get_n_features(model: str = "TwoBody") -> int:
        """Returns the number of features for the specified model"""
        return len(LHCbMCModule.feature_config(model=model).keys())

    def _initialize_preprocessor(self, feature_config_file: str = "features.yml", 
                                model: str = "TwoBody") -> None:
        """Initialize the DataPreprocessor with transformation configs"""
        # Get transformation configs
        transforms_dict = self.transforms_config(model=model, feature_config_file=feature_config_file)
        
        # Prepare lists for each transformation type
        gev_vars = []
        log_vars = []
        
        # Parse the transformations
        for feature, transform_type in transforms_dict.items():
            if transform_type == "gev":
                gev_vars.append(feature)
            elif transform_type == "log":
                log_vars.append(feature)
        
        # Create preprocessor with proper configs
        self.preprocessor = DataPreprocessor(
            gev_vars=tuple(gev_vars),
            log_vars=tuple(log_vars),
            background_channel="minbias",
            clip_quantiles=(0.001, 0.999),
            normalize=True
        )

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
        self, batch_size: int = 128, scale_factor: float = 1.0, ratio: float = 0.1,
        feature_config_file: str = "features.yml", apply_preprocessing: bool = True
    ):
        """Setup the DataLoaders for training and testing data with preprocessing."""
        # Load and sample the data
        train_data = self._process_sb_data(
            pd.read_pickle(self.train_path),
            scale_factor=scale_factor,
            ratio=ratio,
        )
        
        # Load the test data with the same sampling
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

        # HACK: store the full datasets (not just the loaders) - to enable access to channel info for efficiency histograms
        self.raw_train_data = train_data.copy()
        self.raw_test_data = test_data.copy()
        
        # Apply preprocessing if requested
        if apply_preprocessing:
            # Initialize the preprocessor with transformation configs
            self._initialize_preprocessor(feature_config_file=feature_config_file)
            
            # Apply preprocessing to train and test data
            # Only preprocess the feature columns
            subset_train = train_data[self.feature_cols + ["class_label", "channel"]]
            subset_test = test_data[self.feature_cols + ["class_label", "channel"]]
            
            # Fit and transform on training data
            processed_train = self.preprocessor.fit_transform(subset_train, balance=True)
            
            # Transform test data using the same fitted preprocessor
            processed_test = self.preprocessor.transform(subset_test)
            
            # Extract the processed feature columns
            X_train = torch.tensor(
                processed_train[self.feature_cols].values, dtype=torch.float32
            )
            X_test = torch.tensor(
                processed_test[self.feature_cols].values, dtype=torch.float32
            )
        else:
            # Use raw unprocessed data
            X_train = torch.tensor(
                train_data[self.feature_cols].values, dtype=torch.float32
            )
            X_test = torch.tensor(
                test_data[self.feature_cols].values, dtype=torch.float32
            )
        
        # Prepare target tensors (always the same regardless of preprocessing)
        y_train = torch.tensor(train_data["class_label"].values, dtype=torch.float32)
        y_test = torch.tensor(test_data["class_label"].values, dtype=torch.float32)
        
        # Store the tensors for direct access
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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
        breakpoint()

    def setup_distributed(
        self,
        batch_size: int = 128,
        scale_factor: float = 1.0,
        ratio: float = 0.1,
        rank: int = 0,
        world_size: int = 1,
        feature_config_file: str = "features.yml",
        apply_preprocessing: bool = True
    ):
        """Setup the DataLoaders for distributed training across multiple GPUs with preprocessing"""
        # Load and sample the data
        train_data = self._process_sb_data(
            pd.read_pickle(self.train_path),
            scale_factor=scale_factor,
            ratio=ratio,
        )
        
        # Load the test data with the same sampling
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

        # HACK: store the full datasets (not just the loaders) - to enable access to channel info for efficiency histograms
        self.raw_train_data = train_data.copy()
        self.raw_test_data = test_data.copy()
        
        # Apply preprocessing if requested
        if apply_preprocessing:
            # Initialize the preprocessor with transformation configs
            self._initialize_preprocessor(feature_config_file=feature_config_file)
            
            # Apply preprocessing to train and test data
            # Only preprocess the feature columns
            subset_train = train_data[self.feature_cols + ["class_label", "channel"]]
            subset_test = test_data[self.feature_cols + ["class_label", "channel"]]
            
            # Master rank does the fitting and shares the preprocessor with other ranks
            if rank == 0:
                # Fit and transform on training data
                processed_train = self.preprocessor.fit_transform(subset_train, balance=True)
                
                # Transform test data using the same fitted preprocessor
                processed_test = self.preprocessor.transform(subset_test)
                
                # Save preprocessor to a temporary file to share with other ranks
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    self.preprocessor.save(f.name)
                    temp_path = f.name
            else:
                # Other ranks wait for master to finish preprocessing
                import time
                time.sleep(1)  # Simple wait to ensure the file is ready
                
                # Load the preprocessor saved by master rank
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    temp_path = f.name
                
                # Wait for the file to be available
                import os
                while not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    time.sleep(1)
                
                # Load the preprocessor
                self.preprocessor = DataPreprocessor.load(temp_path)
                
                # Transform the data using the loaded preprocessor
                processed_train = self.preprocessor.transform(subset_train)
                processed_test = self.preprocessor.transform(subset_test)
            
            # Extract the processed feature columns
            X_train = torch.tensor(
                processed_train[self.feature_cols].values, dtype=torch.float32
            )
            X_test = torch.tensor(
                processed_test[self.feature_cols].values, dtype=torch.float32
            )
            
            # Clean up the temporary file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        else:
            # Use raw unprocessed data
            X_train = torch.tensor(
                train_data[self.feature_cols].values, dtype=torch.float32
            )
            X_test = torch.tensor(
                test_data[self.feature_cols].values, dtype=torch.float32
            )
        
        # Prepare target tensors (always the same regardless of preprocessing)
        y_train = torch.tensor(train_data["class_label"].values, dtype=torch.float32)
        y_test = torch.tensor(test_data["class_label"].values, dtype=torch.float32)
        
        # Store the tensors for direct access
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # put together the dataset
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Set up samplers for distributed training
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        
        # For test set, we need the same samples on all GPUs for proper evaluation
        test_sampler = None
        
        # create the dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            shuffle=False  # Sampler handles shuffling
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            sampler=test_sampler,
            shuffle=True  # Regular shuffle for test set
        )
        
        self.input_dim = len(self.feature_cols)