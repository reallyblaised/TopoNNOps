import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List, Union
import os
import logging
from datetime import datetime


class EnhancedDataPreprocessor:
    """
    Enhanced version of DataPreprocessor with detailed logging and visualization
    for each preprocessing step.

    This class follows the same processing sequence as the original DataPreprocessor:
    1. Signal balancing
    2. Feature transformations (GeV scaling, log transformation)
    3. Outlier clipping
    4. Normalization
    """

    def __init__(
        self,
        gev_vars: Tuple[str, ...] = None,
        log_vars: Tuple[str, ...] = None,
        tenx_vars: Tuple[str, ...] = None,
        unchanged_vars: Tuple[str, ...] = None,
        background_channel: str = "minbias",
        clip_quantiles: Tuple[float, float] = (0.0001, 0.9999),
        normalize: bool = True,
        log_dir: str = "./preprocessing_logs",
        verbose: bool = True,
    ):
        """
        Initialize the preprocessor with feature transformation specifications and logging.

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
        log_dir : str
            Directory for saving logs and visualizations
        verbose : bool
            Whether to print detailed logs to stdout
        """
        self.gev_vars = gev_vars or ()
        self.log_vars = log_vars or ()
        self.tenx_vars = tenx_vars or ()
        self.unchanged_vars = unchanged_vars or ()
        self.background_channel = background_channel
        self.clip_quantiles = clip_quantiles
        self.normalize = normalize

        # Set up logging
        self.log_dir = log_dir
        self.verbose = verbose
        self.snapshots = {}  # Store data snapshots at different stages

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Set up logger
        self.logger = self._setup_logger()

        # Build transformation map
        self.transformations = {}
        for var in self.gev_vars:
            self.transformations[var] = "gev"
        for var in self.log_vars:
            self.transformations[var] = "log"
        for var in self.tenx_vars:
            self.transformations[var] = "tenx"
        for var in self.unchanged_vars:
            self.transformations[var] = "none"

        # Statistics from training data (to be filled during fit)
        self.feature_stats = {}

        # Log initialization
        self.logger.info("Initialized EnhancedDataPreprocessor:")
        self.logger.info(f"GeV variables: {self.gev_vars}")
        self.logger.info(f"Log variables: {self.log_vars}")
        self.logger.info(f"Tenx variables: {self.tenx_vars}")
        self.logger.info(f"Unchanged variables: {self.unchanged_vars}")
        self.logger.info(f"Clipping quantiles: {self.clip_quantiles}")
        self.logger.info(f"Normalization enabled: {self.normalize}")

    def _setup_logger(self):
        """Set up logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"preprocessing_{timestamp}.log")

        # Create logger
        logger = logging.getLogger(f"preprocessor_{timestamp}")
        logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler with higher level if verbose is False
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def fit(
        self,
        df: pd.DataFrame,
        balance_first: bool = True,
        min_count: Optional[int] = None,
    ) -> "EnhancedDataPreprocessor":
        """
        Calculate necessary statistics from training data with enhanced logging and visualization.

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
        self : EnhancedDataPreprocessor
            The fitted preprocessor
        """
        self.logger.info("Starting fit process...")

        # Store the original data snapshot
        feature_cols = list(self.transformations.keys())
        self.snapshots["raw"] = df[feature_cols].copy()
        self._plot_feature_distributions(
            df[feature_cols], "raw", "Raw Data Distributions"
        )

        # Optionally balance the dataset first for more representative statistics
        if balance_first:
            self.logger.info("Balancing data before fitting...")
            fit_df = self.balance_signal(df, min_count=min_count)

            # Log balancing results
            self.logger.info(
                f"Original data size: {len(df)} -> Balanced data size: {len(fit_df)}"
            )
            if "channel" in df.columns:
                channel_before = df["channel"].value_counts()
                channel_after = fit_df["channel"].value_counts()
                for channel in sorted(
                    set(channel_before.index) | set(channel_after.index)
                ):
                    before = channel_before.get(channel, 0)
                    after = channel_after.get(channel, 0)
                    self.logger.info(f"Channel '{channel}': {before} -> {after}")

            # Save balanced data snapshot
            self.snapshots["balanced"] = fit_df[feature_cols].copy()
            self._plot_feature_distributions(
                fit_df[feature_cols], "balanced", "Balanced Data Distributions"
            )
        else:
            fit_df = df
            self.logger.info("Using unbalanced data for fitting")

        # Calculate statistics for each feature to be transformed
        self.logger.info("Calculating feature statistics...")

        # Create a copy for tracking transformations step by step
        transformed_df = fit_df.copy()

        for column, transform_type in self.transformations.items():
            if column not in fit_df.columns:
                self.logger.warning(f"Column {column} not found in dataset")
                continue

            self.logger.info(
                f"Processing feature '{column}' with transform '{transform_type}'"
            )

            # Get the feature values
            values = fit_df[column].values.copy()
            original_stats = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
            }
            self.logger.info(f"Original stats for '{column}': {original_stats}")

            # Apply primary transformation
            if transform_type == "gev":
                self.logger.info(f"Applying GeV scaling to '{column}'")
                values = values / 1000.0
                transformed_df[column] = transformed_df[column] / 1000.0
            elif transform_type == "log":
                # Find minimum non-zero value to determine appropriate epsilon
                min_positive = (
                    np.min(values[values > 0]) if np.any(values > 0) else 1e-6
                )
                epsilon = min(min_positive / 10.0, 1e-6)
                self.logger.info(
                    f"Applying log transform to '{column}' with epsilon={epsilon}"
                )
                values = np.log(values + epsilon)
                transformed_df[column] = np.log(transformed_df[column] + epsilon)

                # Store epsilon for this feature
                self.feature_stats[f"{column}_epsilon"] = epsilon
            elif transform_type == "tenx":
                self.logger.info(f"Applying 10x scaling to '{column}'")
                values = values * 10.0
                transformed_df[column] = transformed_df[column] * 10.0
            elif transform_type == "none":
                self.logger.info(f"No transformation for '{column}'")

            # Log transformed statistics
            transformed_stats = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
            }
            self.logger.info(
                f"After transform stats for '{column}': {transformed_stats}"
            )

            # Calculate clipping bounds
            lower_bound = np.nanquantile(values, self.clip_quantiles[0])
            upper_bound = np.nanquantile(values, self.clip_quantiles[1])
            self.logger.info(
                f"Clipping bounds for '{column}': lower={lower_bound}, upper={upper_bound}"
            )

            # Store bounds for this feature
            self.feature_stats[f"{column}_lower"] = lower_bound
            self.feature_stats[f"{column}_upper"] = upper_bound

            # Calculate normalization parameters if needed
            if self.normalize:
                # Apply clipping for calculating normalization bounds
                clipped_values = np.clip(values, lower_bound, upper_bound)
                min_val = np.min(clipped_values)
                max_val = np.max(clipped_values)
                self.logger.info(
                    f"Normalization range for '{column}': min={min_val}, max={max_val}"
                )

                # Store normalization parameters
                self.feature_stats[f"{column}_min"] = min_val
                self.feature_stats[f"{column}_max"] = max_val

        # Save transformed data snapshot
        self.snapshots["transformed"] = transformed_df[feature_cols].copy()
        self._plot_feature_distributions(
            transformed_df[feature_cols],
            "transformed",
            "After Initial Transformations (GeV/Log)",
        )

        # Apply clipping to visualization
        clipped_df = transformed_df.copy()
        for column in feature_cols:
            if column in self.feature_stats:
                lower = self.feature_stats.get(f"{column}_lower")
                upper = self.feature_stats.get(f"{column}_upper")
                if lower is not None and upper is not None:
                    clipped_df[column] = clipped_df[column].clip(lower, upper)

        # Save clipped data snapshot
        self.snapshots["clipped"] = clipped_df[feature_cols].copy()
        self._plot_feature_distributions(
            clipped_df[feature_cols], "clipped", "After Clipping (Quantiles)"
        )

        # Create a final visualization after full transformation
        self.logger.info("Fit process completed")
        return self

    def transform(
        self, df: pd.DataFrame, preserve_originals: bool = True
    ) -> pd.DataFrame:
        """
        Apply transformations to the data with enhanced logging.

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
        self.logger.info("Starting transform process...")

        # Create copy to work with
        result_df = df.copy()
        feature_cols = list(self.transformations.keys())

        # Save input snapshot if we don't have the raw data already
        if "raw" not in self.snapshots:
            self.snapshots["raw"] = df[feature_cols].copy()
            self._plot_feature_distributions(
                df[feature_cols], "raw", "Raw Input Data Distributions"
            )

        # Process each feature with logging
        processed_features = 0
        for column, transform_type in self.transformations.items():
            if column not in result_df.columns:
                self.logger.warning(f"Column {column} not found in dataset")
                continue

            processed_features += 1
            self.logger.info(f"Transforming feature '{column}' with '{transform_type}'")

            # Preserve original values if requested
            if preserve_originals:
                result_df[f"{column}_original"] = result_df[column].copy()

            # Record original statistics
            orig_stats = {
                "min": result_df[column].min(),
                "max": result_df[column].max(),
                "mean": result_df[column].mean(),
                "std": result_df[column].std(),
            }
            self.logger.info(f"Original stats for '{column}': {orig_stats}")

            # Apply transformation
            if transform_type == "gev":
                result_df[column] = result_df[column] / 1000.0
                self.logger.info(f"Applied GeV scaling to '{column}'")
            elif transform_type == "log":
                epsilon = self.feature_stats.get(f"{column}_epsilon", 1e-6)
                result_df[column] = np.log(result_df[column] + epsilon)
                self.logger.info(
                    f"Applied log transform to '{column}' with epsilon={epsilon}"
                )
            elif transform_type == "tenx":
                result_df[column] = result_df[column] * 10.0
                self.logger.info(f"Applied 10x scaling to '{column}'")
            elif transform_type == "none":
                self.logger.info(f"No transformation needed for '{column}'")

            # Apply clipping
            lower = self.feature_stats.get(f"{column}_lower")
            upper = self.feature_stats.get(f"{column}_upper")
            if lower is not None and upper is not None:
                before_clip = result_df[column].copy()
                result_df[column] = result_df[column].clip(lower, upper)

                # Calculate how many values were clipped
                lower_clipped = (before_clip < lower).sum()
                upper_clipped = (before_clip > upper).sum()
                perc_clipped = (
                    (lower_clipped + upper_clipped) / len(before_clip)
                ) * 100

                self.logger.info(f"Clipped '{column}' to [{lower}, {upper}]")
                self.logger.info(
                    f"  - Values clipped below: {lower_clipped} ({lower_clipped/len(before_clip)*100:.2f}%)"
                )
                self.logger.info(
                    f"  - Values clipped above: {upper_clipped} ({upper_clipped/len(before_clip)*100:.2f}%)"
                )
                self.logger.info(
                    f"  - Total clipped: {lower_clipped + upper_clipped} ({perc_clipped:.2f}%)"
                )

            # Apply normalization if needed
            if self.normalize:
                min_val = self.feature_stats.get(f"{column}_min")
                max_val = self.feature_stats.get(f"{column}_max")
                if min_val is not None and max_val is not None and max_val > min_val:
                    before_norm = result_df[column].copy()
                    result_df[column] = (result_df[column] - min_val) / (
                        max_val - min_val
                    )
                    self.logger.info(
                        f"Normalized '{column}' from [{min_val}, {max_val}] to [0, 1]"
                    )

                    # Check transformed range
                    actual_min = result_df[column].min()
                    actual_max = result_df[column].max()
                    if actual_min < 0 or actual_max > 1:
                        self.logger.warning(
                            f"  ⚠️ Normalized range outside [0,1]: [{actual_min}, {actual_max}]"
                        )

            # Log transformed statistics
            final_stats = {
                "min": result_df[column].min(),
                "max": result_df[column].max(),
                "mean": result_df[column].mean(),
                "std": result_df[column].std(),
            }
            self.logger.info(f"Final stats for '{column}': {final_stats}")

        # Save a final snapshot
        if processed_features > 0:
            self.snapshots["final"] = result_df[feature_cols].copy()
            self._plot_feature_distributions(
                result_df[feature_cols], "final", "Final Transformed Data"
            )
            self._plot_transformation_stages()
            self._plot_before_after_comparisons()
            self._generate_report()

        self.logger.info(
            f"Transform completed: {processed_features} features processed"
        )
        return result_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        balance: bool = True,
        min_count: Optional[int] = None,
        preserve_originals: bool = True,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step with detailed logging and visualization.

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
        self.logger.info("Starting fit_transform process...")

        # Step 1: Balance for fitting (always balance for fitting parameters)
        if balance:
            self.logger.info("Balancing data for fitting...")
            balanced_df = self.balance_signal(df, min_count=min_count)

            # Log balancing results
            self.logger.info(
                f"Original size: {len(df)} -> Balanced size: {len(balanced_df)}"
            )
            if "channel" in df.columns:
                channel_counts = (
                    pd.DataFrame(
                        {
                            "Original": df["channel"].value_counts(),
                            "Balanced": balanced_df["channel"].value_counts(),
                        }
                    )
                    .fillna(0)
                    .astype(int)
                )

                for channel, row in channel_counts.iterrows():
                    self.logger.info(
                        f"Channel '{channel}': {row['Original']} -> {row['Balanced']}"
                    )

                # Plot channel distributions
                self._plot_channel_distribution(df, balanced_df)
        else:
            balanced_df = df
            self.logger.info("Using unbalanced data (balance=False)")

        # Step 2: Fit on balanced data
        self.logger.info("Fitting preprocessor on data...")
        self.fit(balanced_df, balance_first=False)

        # Step 3: Process using the fitted parameters
        self.logger.info("Applying transformation...")
        if balance:
            # Already balanced, just transform
            return self.transform(balanced_df, preserve_originals=preserve_originals)
        else:
            # Transform without additional balancing
            return self.transform(df, preserve_originals=preserve_originals)

    def balance_signal(
        self, df: pd.DataFrame, min_count: Optional[int] = None, random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Balance signal channels with enhanced logging.

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
        self.logger.info("Balancing signal channels...")

        # Check if channel column exists
        if "channel" not in df.columns:
            self.logger.warning(
                "No 'channel' column found in dataframe, returning original data"
            )
            return df

        # Get the count of each channel
        channel_counts = df["channel"].value_counts()
        self.logger.info("Original channel counts:")
        for channel, count in channel_counts.items():
            self.logger.info(f"  {channel}: {count}")

        # Identify signal channels (all except background_channel)
        signal_channels = [
            ch for ch in channel_counts.index if ch != self.background_channel
        ]
        self.logger.info(
            f"Identified {len(signal_channels)} signal channels: {signal_channels}"
        )

        # Determine the target count for each signal channel
        if min_count is None:
            min_count = min(channel_counts[signal_channels]) if signal_channels else 0
            self.logger.info(f"Using minimum count across signal channels: {min_count}")
        else:
            self.logger.info(f"Using provided minimum count: {min_count}")

        # Create a new balanced DataFrame
        balanced_df = pd.DataFrame()

        # Keep all background channel entries
        if self.background_channel in df["channel"].unique():
            background_df = df[df["channel"] == self.background_channel]
            balanced_df = pd.concat([balanced_df, background_df])
            self.logger.info(
                f"Keeping all {len(background_df)} background ({self.background_channel}) entries"
            )

        # For each signal channel, take only min_count entries
        for channel in signal_channels:
            channel_df = df[df["channel"] == channel]

            # Check if this channel has enough entries
            if len(channel_df) >= min_count:
                # Take min_count random samples
                sampled_df = channel_df.sample(n=min_count, random_state=random_seed)
                balanced_df = pd.concat([balanced_df, sampled_df])
                self.logger.info(
                    f"Channel '{channel}': sampled {min_count} from {len(channel_df)}"
                )
            else:
                # If not enough entries, take all available
                balanced_df = pd.concat([balanced_df, channel_df])
                self.logger.warning(
                    f"Channel '{channel}' has only {len(channel_df)} entries, less than target {min_count}"
                )

        # Reset index of the final DataFrame
        balanced_df = balanced_df.reset_index(drop=True)

        self.logger.info(f"Balanced data size: {len(balanced_df)}")
        self.logger.info("New channel counts:")
        new_counts = balanced_df["channel"].value_counts()
        for channel, count in new_counts.items():
            self.logger.info(f"  {channel}: {count}")

        return balanced_df

    def save(self, filename: str) -> None:
        """Save the preprocessor configuration with enhanced logging."""
        import pickle

        self.logger.info(f"Saving preprocessor configuration to {filename}")

        try:
            with open(filename, "wb") as f:
                pickle.dump(
                    {
                        "gev_vars": self.gev_vars,
                        "log_vars": self.log_vars,
                        "tenx_vars": self.tenx_vars,
                        "unchanged_vars": self.unchanged_vars,
                        "background_channel": self.background_channel,
                        "clip_quantiles": self.clip_quantiles,
                        "normalize": self.normalize,
                        "transformations": self.transformations,
                        "feature_stats": self.feature_stats,
                    },
                    f,
                )
            self.logger.info("Preprocessor saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")

    @classmethod
    def load(
        cls, filename: str, log_dir: str = "./preprocessing_logs"
    ) -> "EnhancedDataPreprocessor":
        """Load a preprocessor configuration with enhanced logging."""
        import pickle

        try:
            with open(filename, "rb") as f:
                config = pickle.load(f)

            # Initialize with basic parameters
            preprocessor = cls(
                gev_vars=config["gev_vars"],
                log_vars=config["log_vars"],
                tenx_vars=config["tenx_vars"],
                unchanged_vars=config.get("unchanged_vars", ()),
                background_channel=config["background_channel"],
                clip_quantiles=config["clip_quantiles"],
                normalize=config["normalize"],
                log_dir=log_dir,
            )

            # Restore additional attributes
            preprocessor.transformations = config["transformations"]
            preprocessor.feature_stats = config["feature_stats"]

            preprocessor.logger.info(f"Loaded preprocessor from {filename}")
            preprocessor.logger.info(
                f"Loaded {len(preprocessor.feature_stats)} feature statistics"
            )

            return preprocessor
        except Exception as e:
            # Create basic logger to report error
            logger = logging.getLogger("preprocessor_loader")
            logger.setLevel(logging.ERROR)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.error(f"Error loading preprocessor from {filename}: {str(e)}")
            raise

    def _plot_feature_distributions(
        self, df: pd.DataFrame, stage: str, title: str
    ) -> None:
        """Plot distributions of all features at a specific stage."""
        if len(df.columns) == 0:
            self.logger.warning(f"No features to plot for stage '{stage}'")
            return

        # Determine grid layout
        n_features = len(df.columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feature in enumerate(df.columns):
            if i < len(axes):
                # Calculate histogram
                sns.histplot(df[feature].dropna(), ax=axes[i], kde=False)
                axes[i].set_title(f"{feature}")

                # Add min/max/mean as text
                stats_text = f"Min: {df[feature].min():.2g}\nMax: {df[feature].max():.2g}\nMean: {df[feature].mean():.2g}"
                axes[i].text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save the figure
        plot_path = os.path.join(self.log_dir, f"distribution_{stage}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        self.logger.info(
            f"Saved feature distributions for '{stage}' stage to {plot_path}"
        )

    def _plot_channel_distribution(
        self, original_df: pd.DataFrame, balanced_df: pd.DataFrame
    ) -> None:
        """Plot channel distribution before and after balancing."""
        if "channel" not in original_df.columns or "channel" not in balanced_df.columns:
            return

        # Get channel counts
        original_counts = original_df["channel"].value_counts().sort_index()
        balanced_counts = balanced_df["channel"].value_counts().sort_index()

        # Combine into a DataFrame
        counts_df = (
            pd.DataFrame({"Original": original_counts, "Balanced": balanced_counts})
            .fillna(0)
            .astype(int)
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        counts_df.plot(kind="bar", ax=ax)
        ax.set_title("Channel Distribution: Original vs. Balanced", fontsize=14)
        ax.set_ylabel("Count")
        ax.set_xlabel("Channel")

        # Add data labels
        for i, p in enumerate(ax.patches):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(
                f"{height:g}",
                (x + width / 2, y + height + 0.1),
                ha="center",
                va="center",
                fontsize=8,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(self.log_dir, "channel_distribution.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        self.logger.info(f"Saved channel distribution plot to {plot_path}")

    def _plot_transformation_stages(self) -> None:
        """Plot feature statistics across different transformation stages."""
        # Get the stages we have
        stages = [
            stage
            for stage in ["raw", "balanced", "transformed", "clipped", "final"]
            if stage in self.snapshots
        ]

        if len(stages) < 2:
            self.logger.warning("Not enough stages to plot transformation progression")
            return

        feature_cols = list(self.transformations.keys())

        for feature in feature_cols:
            if feature not in self.snapshots[stages[0]].columns:
                continue

            # Collect statistics for this feature across stages
            stats = []
            for stage in stages:
                if feature in self.snapshots[stage].columns:
                    feat_data = self.snapshots[stage][feature]
                    stats.append(
                        {
                            "Stage": stage.capitalize(),
                            "Min": feat_data.min(),
                            "Max": feat_data.max(),
                            "Mean": feat_data.mean(),
                            "Median": feat_data.median(),
                            "StdDev": feat_data.std(),
                        }
                    )

            if not stats:
                continue

            stats_df = pd.DataFrame(stats)

            # Plot the statistics
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            # Plot min/max
            axes[0].plot(stats_df["Stage"], stats_df["Min"], "o-", label="Min")
            axes[0].plot(stats_df["Stage"], stats_df["Max"], "s-", label="Max")
            axes[0].set_title(f"{feature} - Min/Max")
            axes[0].set_xticks(range(len(stats_df)))
            axes[0].set_xticklabels(stats_df["Stage"], rotation=45)
            axes[0].legend()

            # Plot mean/median
            axes[1].plot(stats_df["Stage"], stats_df["Mean"], "o-", label="Mean")
            axes[1].plot(stats_df["Stage"], stats_df["Median"], "s-", label="Median")
            axes[1].set_title(f"{feature} - Mean/Median")
            axes[1].set_xticks(range(len(stats_df)))
            axes[1].set_xticklabels(stats_df["Stage"], rotation=45)
            axes[1].legend()

            # Plot standard deviation
            axes[2].plot(stats_df["Stage"], stats_df["StdDev"], "o-")
            axes[2].set_title(f"{feature} - Standard Deviation")
            axes[2].set_xticks(range(len(stats_df)))
            axes[2].set_xticklabels(stats_df["Stage"], rotation=45)

            # Plot range (max - min)
            range_vals = stats_df["Max"] - stats_df["Min"]
            axes[3].plot(stats_df["Stage"], range_vals, "o-")
            axes[3].set_title(f"{feature} - Range")
            axes[3].set_xticks(range(len(stats_df)))
            axes[3].set_xticklabels(stats_df["Stage"], rotation=45)

            plt.suptitle(f"Statistics Changes for {feature}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save the figure
            plot_path = os.path.join(self.log_dir, f"stats_changes_{feature}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

            self.logger.info(
                f"Saved transformation statistics for '{feature}' to {plot_path}"
            )

    def _plot_before_after_comparisons(self) -> None:
        """Create before-after comparison plots for each feature."""
        # Determine which stages to compare
        if "raw" in self.snapshots and "final" in self.snapshots:
            before_stage = "raw"
            after_stage = "final"
        elif "balanced" in self.snapshots and "final" in self.snapshots:
            before_stage = "balanced"
            after_stage = "final"
        else:
            self.logger.warning("Not enough stages to create before-after comparisons")
            return

        feature_cols = list(self.transformations.keys())

        # For each feature, create a before-after plot
        for feature in feature_cols:
            if (
                feature not in self.snapshots[before_stage].columns
                or feature not in self.snapshots[after_stage].columns
            ):
                continue

            before_data = self.snapshots[before_stage][feature]
            after_data = self.snapshots[after_stage][feature]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Before histogram
            sns.histplot(before_data.dropna(), kde=False, ax=axes[0])
            axes[0].set_title(f"{feature} - Before ({before_stage.capitalize()})")

            # Add statistics
            stats_text = f"Min: {before_data.min():.2g}\nMax: {before_data.max():.2g}\nMean: {before_data.mean():.2g}\nStd: {before_data.std():.2g}"
            axes[0].text(
                0.95,
                0.95,
                stats_text,
                transform=axes[0].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            # After histogram
            sns.histplot(after_data.dropna(), kde=False, ax=axes[1])
            axes[1].set_title(f"{feature} - After (Processed)")

            # Add statistics
            stats_text = f"Min: {after_data.min():.2g}\nMax: {after_data.max():.2g}\nMean: {after_data.mean():.2g}\nStd: {after_data.std():.2g}"
            axes[1].text(
                0.95,
                0.95,
                stats_text,
                transform=axes[1].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            plt.suptitle(f"Before vs After Preprocessing - {feature}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save the figure
            plot_path = os.path.join(self.log_dir, f"before_after_{feature}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

            self.logger.info(
                f"Saved before-after comparison for '{feature}' to {plot_path}"
            )

    def _generate_report(self) -> None:
        """Generate a comprehensive markdown report with all preprocessing information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(self.log_dir, "preprocessing_report.md")

        with open(report_path, "w") as f:
            f.write(f"# Preprocessing Validation Report\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- GeV scaling variables: {', '.join(self.gev_vars) or 'None'}\n")
            f.write(
                f"- Log transform variables: {', '.join(self.log_vars) or 'None'}\n"
            )
            f.write(f"- Tenx scaling variables: {', '.join(self.tenx_vars) or 'None'}\n")
            f.write(
                f"- Unchanged variables: {', '.join(self.unchanged_vars) or 'None'}\n"
            )
            f.write(f"- Background channel: {self.background_channel}\n")
            f.write(f"- Clipping quantiles: {self.clip_quantiles}\n")
            f.write(f"- Normalization enabled: {self.normalize}\n\n")

            f.write("## Transformation Parameters\n\n")
            f.write(
                "| Feature | Transform | Epsilon | Lower Bound | Upper Bound | Min Value | Max Value |\n"
            )
            f.write(
                "|---------|-----------|---------|-------------|-------------|-----------|----------|\n"
            )

            for feature in sorted(self.transformations.keys()):
                transform = self.transformations.get(feature, "none")
                epsilon = self.feature_stats.get(f"{feature}_epsilon", "N/A")
                lower = self.feature_stats.get(f"{feature}_lower", "N/A")
                upper = self.feature_stats.get(f"{feature}_upper", "N/A")
                min_val = self.feature_stats.get(f"{feature}_min", "N/A")
                max_val = self.feature_stats.get(f"{feature}_max", "N/A")

                f.write(
                    f"| {feature} | {transform} | {epsilon} | {lower} | {upper} | {min_val} | {max_val} |\n"
                )

            f.write("\n## Visualization Links\n\n")

            # Data distribution links
            f.write("### Distribution Visualizations\n\n")
            stages = ["raw", "balanced", "transformed", "clipped", "final"]
            for stage in stages:
                if os.path.exists(
                    os.path.join(self.log_dir, f"distribution_{stage}.png")
                ):
                    f.write(
                        f"- [{stage.capitalize()} Data Distributions](distribution_{stage}.png)\n"
                    )

            # Feature transformation links
            f.write("\n### Transformation Statistics\n\n")
            for feature in sorted(self.transformations.keys()):
                if os.path.exists(
                    os.path.join(self.log_dir, f"stats_changes_{feature}.png")
                ):
                    f.write(f"- [{feature} Statistics](stats_changes_{feature}.png)\n")

            # Before-after comparison links
            f.write("\n### Before-After Comparisons\n\n")
            for feature in sorted(self.transformations.keys()):
                if os.path.exists(
                    os.path.join(self.log_dir, f"before_after_{feature}.png")
                ):
                    f.write(f"- [{feature} Before-After](before_after_{feature}.png)\n")

            # Add channel distribution if available
            if os.path.exists(os.path.join(self.log_dir, "channel_distribution.png")):
                f.write("\n### Channel Distributions\n\n")
                f.write("- [Channel Distribution](channel_distribution.png)\n")

        self.logger.info(f"Generated preprocessing report at {report_path}")


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
        tenx_vars: Tuple[str, ...] = None,
        unchanged_vars: Tuple[str, ...] = None,  # New parameter
        background_channel: str = "minbias",
        clip_quantiles: Tuple[float, float] = (0.001, 0.999),
        normalize: bool = True,
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
        tenx_vars : tuple of str
            Variables to be scaled by 10
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
        self.tenx_vars = tenx_vars or ()
        self.background_channel = background_channel
        self.clip_quantiles = clip_quantiles
        self.normalize = normalize

        # Build transformation map
        self.transformations = {}
        for var in self.gev_vars:
            self.transformations[var] = "gev"
        for var in self.log_vars:
            self.transformations[var] = "log"
        for var in self.tenx_vars:
            self.transformations[var] = "tenx"
        for var in self.unchanged_vars:
            self.transformations[var] = "none"  # Add unchanged variables

        # Statistics from training data (to be filled during fit)
        self.feature_stats = {}

    def fit(
        self,
        df: pd.DataFrame,
        balance_first: bool = False,
        min_count: Optional[int] = None,
    ) -> "DataPreprocessor":
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
                min_positive = (
                    np.min(values[values > 0]) if np.any(values > 0) else 1e-6
                )
                epsilon = min(min_positive / 10.0, 1e-6)
                values = np.log(values + epsilon)

                # Store epsilon for this feature
                self.feature_stats[f"{column}_epsilon"] = epsilon
            elif transform_type == "tenx":
                values = values * 10.0
            elif transform_type == "none":
                # No transformation needed for unchanged variables
                pass

            # Calculate clipping bounds
            lower_bound = np.nanquantile(values, self.clip_quantiles[0])
            upper_bound = np.nanquantile(values, self.clip_quantiles[1])

            # Store bounds for this feature
            self.feature_stats[f"{column}_lower"] = lower_bound
            self.feature_stats[f"{column}_upper"] = upper_bound

            # Apply clipping for calculating normalization bounds - this assumes that clipping is enacted regardless of whether normalization is applied
            clipped_values = np.clip(values, lower_bound, upper_bound)
            min_val = np.min(clipped_values)
            max_val = np.max(clipped_values)

            # Store normalization parameters
            self.feature_stats[f"{column}_min"] = min_val
            self.feature_stats[f"{column}_max"] = max_val

        return self

    def transform(
        self, df: pd.DataFrame, preserve_originals: bool = True
    ) -> pd.DataFrame:
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
            elif transform_type == "tenx":
                result_df[column] = result_df[column] * 10.0
            elif transform_type == "none":
                # No transformation for unchanged variables
                pass

            # Apply clipping
            lower = self.feature_stats.get(f"{column}_lower")
            upper = self.feature_stats.get(f"{column}_upper")
            if lower is not None and upper is not None:
                result_df[column] = result_df[column].clip(lower, upper)

            # check the new upper and lower bounds of the data effectively match the max and min of the data
            assert np.abs(np.max(result_df[column]) - upper) < 1e-6, "Post-clipping upper bound is not as expected"
            assert np.abs(np.min(result_df[column]) - lower) < 1e-6, "Post-clipping lower bound is not as expected"

            # Apply normalization if needed to the clip extrema, which are now the min and max of the data
            if self.normalize:
                min_val = self.feature_stats.get(f"{column}_lower")
                max_val = self.feature_stats.get(f"{column}_upper")
                if min_val is not None and max_val is not None and max_val > min_val:
                    result_df[column] = (result_df[column] - min_val) / (
                        max_val - min_val
                    )

        return result_df

    def process(
        self,
        df: pd.DataFrame,
        balance: bool = True,
        min_count: Optional[int] = None,
        preserve_originals: bool = True,
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
        processed_df = self.transform(
            processed_df, preserve_originals=preserve_originals
        )

        return processed_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        balance: bool = False,
        min_count: Optional[int] = None,
        preserve_originals: bool = True,
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
        self, df: pd.DataFrame, min_count: Optional[int] = None, random_seed: int = 42
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
        channel_counts = df["channel"].value_counts()

        # Identify signal channels (all except background_channel)
        signal_channels = [
            ch for ch in channel_counts.index if ch != self.background_channel
        ]

        # Determine the target count for each signal channel
        if min_count is None:
            min_count = min(channel_counts[signal_channels])

        # Create a new balanced DataFrame
        balanced_df = pd.DataFrame()

        # Keep all background channel entries
        if self.background_channel in df["channel"].unique():
            background_df = df[df["channel"] == self.background_channel]
            balanced_df = pd.concat([balanced_df, background_df])

        # For each signal channel, take only min_count entries
        for channel in signal_channels:
            channel_df = df[df["channel"] == channel]

            # Check if this channel has enough entries
            if len(channel_df) >= min_count:
                # Take min_count random samples
                sampled_df = channel_df.sample(n=min_count, random_state=random_seed)
                balanced_df = pd.concat([balanced_df, sampled_df])
            else:
                # If not enough entries, take all available
                balanced_df = pd.concat([balanced_df, channel_df])
                print(
                    f"Warning: Channel '{channel}' has only {len(channel_df)} entries, less than the target {min_count}"
                )

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

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "gev_vars": self.gev_vars,
                    "log_vars": self.log_vars,
                    "tenx_vars": self.tenx_vars,
                    "unchanged_vars": self.unchanged_vars,  # Save the new parameter
                    "background_channel": self.background_channel,
                    "clip_quantiles": self.clip_quantiles,
                    "normalize": self.normalize,
                    "transformations": self.transformations,
                    "feature_stats": self.feature_stats,
                },
                f,
            )

    @classmethod
    def load(cls, filename: str) -> "DataPreprocessor":
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

        with open(filename, "rb") as f:
            config = pickle.load(f)

        # Initialize with basic parameters
        preprocessor = cls(
            gev_vars=config["gev_vars"],
            log_vars=config["log_vars"],
            tenx_vars=config["tenx_vars"],
            unchanged_vars=config.get(
                "unchanged_vars", ()
            ),  # Handle loading from older files
            background_channel=config["background_channel"],
            clip_quantiles=config["clip_quantiles"],
            normalize=config["normalize"],
        )

        # Restore additional attributes
        preprocessor.transformations = config["transformations"]
        preprocessor.feature_stats = config["feature_stats"]

        return preprocessor
