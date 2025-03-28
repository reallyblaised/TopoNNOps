import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
import numpy as np

class DataPreprocessor:
    """
    A comprehensive preprocessor for HEP data with optimal processing sequence:
    1. Signal balancing
    2. Feature transformations (GeV scaling, log transformation)
    3. Outlier clipping and normalization
    
    Designed for integration with TopoNNOps codebase.
    """
    
    def __init__(
        self,
        gev_vars: Tuple[str, ...] = None,
        log_vars: Tuple[str, ...] = None,
        unchanged_vars: Tuple[str, ...] = None,  # New parameter
        background_channel: str = 'minbias',
        clip_quantiles: Tuple[float, float] = (0.001, 0.999),
        normalize: bool = True
    ):
        """
        Initialize the preprocessor with feature transformation specifications.
        
        Parameters:
        -----------
        gev_vars : tuple of str
            Variables to be converted from MeV to GeV (divide by 1000)
        log_vars : tuple of str
            Variables to be log-transformed
        unchanged_vars : tuple of str
            Variables to include in normalization without any prior transformation
        background_channel : str
            Name of the background channel in the dataset
        clip_quantiles : tuple of (lower, upper)
            Quantiles to use for clipping values
        normalize : bool
            Whether to normalize features to [0,1] range
        """
        self.gev_vars = gev_vars or ()
        self.log_vars = log_vars or ()
        self.unchanged_vars = unchanged_vars or ()  # Store the new parameter
        self.background_channel = background_channel
        self.clip_quantiles = clip_quantiles
        self.normalize = normalize
        
        # Build transformation map
        self.transformations = {}
        for var in self.gev_vars:
            self.transformations[var] = "gev"
        for var in self.log_vars:
            self.transformations[var] = "log"
        for var in self.unchanged_vars:
            self.transformations[var] = "none"  # Add unchanged variables
            
        # Statistics from training data (to be filled during fit)
        self.feature_stats = {}
    
    def fit(self, df: pd.DataFrame, balance_first: bool = False, min_count: Optional[int] = None) -> 'DataPreprocessor':
        """
        Calculate necessary statistics from training data for later transformation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training data
        balance_first : bool
            Whether to balance signal channels before calculating transformation statistics
        min_count : int, optional
            Minimum count to use for all signal channels if balancing
            
        Returns:
        --------
        self : DataPreprocessor
            The fitted preprocessor
        """
        # Optionally balance the dataset first for more representative statistics
        if balance_first:
            fit_df = self.balance_signal(df, min_count=min_count)
        else:
            fit_df = df
            
        # Calculate statistics for each feature to be transformed
        for column, transform_type in self.transformations.items():
            if column not in fit_df.columns:
                continue
                
            # Get the feature values
            values = fit_df[column].values.copy()
            
            # Apply primary transformation
            if transform_type == "gev":
                values = values / 1000.0
            elif transform_type == "log":
                # Find minimum non-zero value to determine appropriate epsilon
                min_positive = np.min(values[values > 0]) if np.any(values > 0) else 1e-6
                epsilon = min(min_positive / 10.0, 1e-6)
                values = np.log(values + epsilon)
                
                # Store epsilon for this feature
                self.feature_stats[f"{column}_epsilon"] = epsilon
            elif transform_type == "none":
                # No transformation needed for unchanged variables
                pass
            
            # Calculate clipping bounds
            lower_bound = np.nanquantile(values, self.clip_quantiles[0])
            upper_bound = np.nanquantile(values, self.clip_quantiles[1])
            
            # Store bounds for this feature
            self.feature_stats[f"{column}_lower"] = lower_bound
            self.feature_stats[f"{column}_upper"] = upper_bound

            # Calculate normalization parameters if needed
            if self.normalize:
                # Apply clipping for calculating normalization bounds
                clipped_values = np.clip(values, lower_bound, upper_bound)
                min_val = np.min(clipped_values)
                max_val = np.max(clipped_values)
                
                # Store normalization parameters
                self.feature_stats[f"{column}_min"] = min_val
                self.feature_stats[f"{column}_max"] = max_val
                
        return self
    
    def transform(self, df: pd.DataFrame, preserve_originals: bool = True) -> pd.DataFrame:
        """
        Apply transformations to the data based on fitted statistics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to transform
        preserve_originals : bool
            Whether to keep original values in '_original' columns
            
        Returns:
        --------
        pandas.DataFrame
            Transformed DataFrame
        """
        # Create copy to work with
        result_df = df.copy()
        
        # Process each feature
        for column, transform_type in self.transformations.items():
            if column not in result_df.columns:
                continue
                
            # Preserve original values if requested
            if preserve_originals:
                result_df[f"{column}_original"] = result_df[column].copy()
            
            # Apply transformation
            if transform_type == "gev":
                result_df[column] = result_df[column] / 1000.0
            elif transform_type == "log":
                epsilon = self.feature_stats.get(f"{column}_epsilon", 1e-6)
                result_df[column] = np.log(result_df[column] + epsilon)
            elif transform_type == "none":
                # No transformation for unchanged variables
                pass
            
            # Apply clipping
            lower = self.feature_stats.get(f"{column}_lower")
            upper = self.feature_stats.get(f"{column}_upper")
            if lower is not None and upper is not None:
                result_df[column] = result_df[column].clip(lower, upper)
            
            # Apply normalization if needed
            if self.normalize:
                min_val = self.feature_stats.get(f"{column}_min")
                max_val = self.feature_stats.get(f"{column}_max")
                if min_val is not None and max_val is not None and max_val > min_val:
                    result_df[column] = (result_df[column] - min_val) / (max_val - min_val)
        
        return result_df
    
    def process(
        self, 
        df: pd.DataFrame, 
        balance: bool = True, 
        min_count: Optional[int] = None,
        preserve_originals: bool = True
    ) -> pd.DataFrame:
        """
        Complete processing pipeline: balance -> transform -> normalize
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to process
        balance : bool
            Whether to balance signal channels
        min_count : int, optional
            Minimum count to use for all signal channels if balancing
        preserve_originals : bool
            Whether to keep original values in '_original' columns
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        # Step 1: Balance signal channels
        if balance:
            processed_df = self.balance_signal(df, min_count=min_count)
        else:
            processed_df = df.copy()
            
        # Step 2: Apply transformations
        processed_df = self.transform(processed_df, preserve_originals=preserve_originals)
        
        return processed_df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        balance: bool = False, 
        min_count: Optional[int] = None,
        preserve_originals: bool = True
    ) -> pd.DataFrame:
        """
        Convenience method to fit and transform in one step,
        following the recommended sequence: balance -> fit -> transform
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training data
        balance : bool
            Whether to balance signal channels
        min_count : int, optional
            Minimum count to use for all signal channels if balancing
        preserve_originals : bool
            Whether to keep original values in '_original' columns
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        # Step 1: Balance for fitting (always balance for fitting parameters)
        balanced_df = self.balance_signal(df, min_count=min_count) if balance else df
        
        # Step 2: Fit on balanced data
        self.fit(balanced_df, balance_first=balance)  # Already balanced
        
        # Step 3: Process using the fitted parameters
        # Either transform the balanced data or re-balance and transform the original
        if balance:
            # Already balanced, just transform
            return self.transform(balanced_df, preserve_originals=preserve_originals)
        else:
            # Transform without additional balancing
            return self.transform(df, preserve_originals=preserve_originals)

    def balance_signal(
        self, 
        df: pd.DataFrame, 
        min_count: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Balance signal channels so they all have equal representation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the data with a 'channel' column
        min_count : int, optional
            Minimum count to use for all signal channels. If None, uses the minimum
            count found in the signal channels
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with balanced signal channels
        """
        # Get the count of each channel
        channel_counts = df['channel'].value_counts()
        
        # Identify signal channels (all except background_channel)
        signal_channels = [ch for ch in channel_counts.index if ch != self.background_channel]
        
        # Determine the target count for each signal channel
        if min_count is None:
            min_count = min(channel_counts[signal_channels])
        
        # Create a new balanced DataFrame
        balanced_df = pd.DataFrame()
        
        # Keep all background channel entries
        if self.background_channel in df['channel'].unique():
            background_df = df[df['channel'] == self.background_channel]
            balanced_df = pd.concat([balanced_df, background_df])
        
        # For each signal channel, take only min_count entries
        for channel in signal_channels:
            channel_df = df[df['channel'] == channel]
            
            # Check if this channel has enough entries
            if len(channel_df) >= min_count:
                # Take min_count random samples
                sampled_df = channel_df.sample(n=min_count, random_state=random_seed)
                balanced_df = pd.concat([balanced_df, sampled_df])
            else:
                # If not enough entries, take all available
                balanced_df = pd.concat([balanced_df, channel_df])
                print(f"Warning: Channel '{channel}' has only {len(channel_df)} entries, less than the target {min_count}")
        
        # Reset index of the final DataFrame
        balanced_df = balanced_df.reset_index(drop=True)
        
        return balanced_df
        
    def save(self, filename: str) -> None:
        """
        Save the preprocessor configuration to a file.
        
        Parameters:
        -----------
        filename : str
            File path to save the configuration
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'gev_vars': self.gev_vars,
                'log_vars': self.log_vars,
                'unchanged_vars': self.unchanged_vars,  # Save the new parameter
                'background_channel': self.background_channel,
                'clip_quantiles': self.clip_quantiles,
                'normalize': self.normalize,
                'transformations': self.transformations,
                'feature_stats': self.feature_stats
            }, f)
    
    @classmethod
    def load(cls, filename: str) -> 'DataPreprocessor':
        """
        Load a preprocessor configuration from a file.
        
        Parameters:
        -----------
        filename : str
            File path to load the configuration from
            
        Returns:
        --------
        DataPreprocessor
            Loaded preprocessor
        """
        import pickle
        with open(filename, 'rb') as f:
            config = pickle.load(f)
            
        # Initialize with basic parameters
        preprocessor = cls(
            gev_vars=config['gev_vars'],
            log_vars=config['log_vars'],
            unchanged_vars=config.get('unchanged_vars', ()),  # Handle loading from older files
            background_channel=config['background_channel'],
            clip_quantiles=config['clip_quantiles'],
            normalize=config['normalize']
        )
        
        # Restore additional attributes
        preprocessor.transformations = config['transformations']
        preprocessor.feature_stats = config['feature_stats']
        
        return preprocessor