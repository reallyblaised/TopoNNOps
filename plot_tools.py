import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Optional, List, Tuple, Dict
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PredictionPlotter:
    """Class for creating plots from model predictions."""
    
    @staticmethod
    def plot_prediction_distributions_by_class(
        df: pd.DataFrame, 
        prediction_col: str = 'nn_prediction',
        class_col: str = 'channel',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 50,
        top_n_classes: Optional[int] = None
    ):
        """
        Plot histograms of predictions for each class on a logarithmic y-scale.
        
        Args:
            df: DataFrame containing predictions and class labels
            prediction_col: Name of the column containing model predictions
            class_col: Name of the column containing class labels
            output_path: Path to save the plot (if None, plot will be displayed)
            figsize: Figure size as (width, height)
            bins: Number of histogram bins
            top_n_classes: If provided, only plot the top N classes by sample count
        """
        # Verify columns exist
        if prediction_col not in df.columns:
            raise ValueError(f"Prediction column '{prediction_col}' not found in DataFrame")
        if class_col not in df.columns:
            raise ValueError(f"Class column '{class_col}' not found in DataFrame")
            
        # Get class counts and optionally filter to top N
        class_counts = df[class_col].value_counts()
        if top_n_classes is not None:
            top_classes = class_counts.head(top_n_classes).index.tolist()
            df_plot = df[df[class_col].isin(top_classes)].copy()
        else:
            df_plot = df.copy()
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each class
        unique_classes = df_plot[class_col].unique()
        for class_name in unique_classes:
            class_data = df_plot[df_plot[class_col] == class_name][prediction_col].values
            if len(class_data) < 10:  # Skip classes with too few samples
                continue
                
            plt.hist(
                class_data, 
                bins=bins, 
                alpha=0.5, 
                label=f"{class_name} (n={len(class_data):,})"
            )
            
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Model Prediction Score')
        plt.ylabel('Count (log scale)')
        plt.title('Distribution of Prediction Scores by Class')
        
        # Add grid and legend
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved prediction distribution plot to {output_path}")
        else:
            plt.show()
            
        plt.close()
    
    @staticmethod
    def plot_signal_background_distributions(
        df: pd.DataFrame,
        prediction_col: str = 'nn_prediction',
        label_col: str = 'Event_Signal',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        bins: int = 50
    ):
        """
        Plot histograms of predictions for signal and background on a logarithmic y-scale.
        
        Args:
            df: DataFrame containing predictions and true labels
            prediction_col: Name of the column containing model predictions
            label_col: Name of the column containing binary labels (0=background, 1=signal)
            output_path: Path to save the plot (if None, plot will be displayed)
            figsize: Figure size as (width, height)
            bins: Number of histogram bins
        """
        # Verify columns exist
        if prediction_col not in df.columns:
            raise ValueError(f"Prediction column '{prediction_col}' not found in DataFrame")
        if label_col not in df.columns:
            # Try alternative label column names
            alt_names = ['label', 'Event_Signal', 'signal']
            for name in alt_names:
                if name in df.columns:
                    label_col = name
                    break
            else:
                raise ValueError(f"No suitable label column found in DataFrame")
                
        # Create figure
        plt.figure(figsize=figsize)
        
        # Get signal and background predictions
        signal_preds = df[df[label_col] == 1][prediction_col].values
        bkg_preds = df[df[label_col] == 0][prediction_col].values
        
        # Plot histograms
        plt.hist(bkg_preds, bins=bins, alpha=0.5, label=f"Background (n={len(bkg_preds):,})", color='blue')
        plt.hist(signal_preds, bins=bins, alpha=0.5, label=f"Signal (n={len(signal_preds):,})", color='red')
        
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Model Prediction Score')
        plt.ylabel('Count (log scale)')
        plt.title('Distribution of Prediction Scores: Signal vs Background')
        
        # Add grid and legend
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved signal vs background plot to {output_path}")
        else:
            plt.show()
            
        plt.close()
    
    @staticmethod
    def plot_enhanced_confusion_matrix(
        df: pd.DataFrame,
        prediction_col: str = 'nn_prediction',
        label_col: str = 'Event_Signal',
        threshold: float = 0.5,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        normalize: bool = False
    ):
        """
        Plot an enhanced confusion matrix with performance metrics.
        
        Args:
            df: DataFrame containing predictions and true labels
            prediction_col: Name of the column containing model predictions
            label_col: Name of the column containing true labels
            threshold: Threshold for converting predictions to binary
            output_path: Path to save the plot (if None, plot will be displayed)
            figsize: Figure size as (width, height)
            normalize: Whether to normalize confusion matrix values
        """
        # Verify columns exist
        if prediction_col not in df.columns:
            raise ValueError(f"Prediction column '{prediction_col}' not found in DataFrame")
        if label_col not in df.columns:
            # Try alternative label column names
            alt_names = ['label', 'Event_Signal', 'signal']
            for name in alt_names:
                if name in df.columns:
                    label_col = name
                    break
            else:
                raise ValueError(f"No suitable label column found in DataFrame")
        
        # Create binary predictions
        y_true = df[label_col].values
        y_pred = (df[prediction_col].values >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=['Background', 'Signal'],
            yticklabels=['Background', 'Signal'],
            cbar=False
        )
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        # For normalized confusion matrix, we need raw counts for some metrics
        if normalize:
            # Get raw counts for metrics calculation
            raw_cm = confusion_matrix(y_true, y_pred)
            tn_raw, fp_raw, fn_raw, tp_raw = raw_cm.ravel()
            total = tn_raw + fp_raw + fn_raw + tp_raw
        else:
            total = tn + fp + fn + tp
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add labels and title
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        plt.title(f'{title} (threshold = {threshold:.2f})')
        
        # Add text with metrics
        metrics_text = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall (TPR): {recall:.4f}\n"
            f"Specificity (TNR): {specificity:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"Total samples: {total:,}"
        )
        
        plt.figtext(0.15, 0.05, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved confusion matrix plot to {output_path}")
        else:
            plt.show()
            
        plt.close()
        
        return cm
    
    @staticmethod
    def plot_multiclass_distributions(
        df: pd.DataFrame,
        prediction_col: str = 'nn_prediction',
        class_col: str = 'channel', 
        signal_col: str = 'Event_Signal',
        bins: int = 50,
        figsize: Tuple[int, int] = (14, 10),
        output_path: Optional[str] = None,
        top_n_classes: int = 7
    ):
        """
        Plot model predictions for different classes, highlighted by signal/background.
        
        Args:
            df: DataFrame containing predictions, class labels, and signal flags
            prediction_col: Name of the column containing model predictions
            class_col: Name of the column containing class labels
            signal_col: Name of the column indicating signal (1) or background (0)
            bins: Number of histogram bins
            figsize: Figure size as (width, height)
            output_path: Path to save the plot (if None, plot will be displayed)
            top_n_classes: Number of top classes to show (by frequency)
        """
        # Verify columns exist
        if prediction_col not in df.columns:
            raise ValueError(f"Prediction column '{prediction_col}' not found in DataFrame")
        if class_col not in df.columns:
            raise ValueError(f"Class column '{class_col}' not found in DataFrame")
        if signal_col not in df.columns:
            # Try alternative signal column names
            alt_names = ['label', 'Event_Signal', 'signal']
            for name in alt_names:
                if name in df.columns:
                    signal_col = name
                    break
            else:
                raise ValueError(f"No suitable signal column found in DataFrame")
        
        # Get top N classes by frequency
        top_classes = df[class_col].value_counts().head(top_n_classes).index.tolist()
        df_plot = df[df[class_col].isin(top_classes)].copy()
        
        # Set up figure with subplots
        fig, axes = plt.subplots(nrows=len(top_classes), figsize=figsize, sharex=True)
        
        # For a single class, axes will not be an array
        if len(top_classes) == 1:
            axes = [axes]
        
        max_count = 0  # Track maximum count for consistent y-axes
        
        # Plot each class
        for i, class_name in enumerate(top_classes):
            class_df = df_plot[df_plot[class_col] == class_name]
            
            # Get signal and background samples
            signal_samples = class_df[class_df[signal_col] == 1][prediction_col].values
            bkg_samples = class_df[class_df[signal_col] == 0][prediction_col].values
            
            # Plot histograms
            if len(bkg_samples) > 0:
                n_bkg, _, _ = axes[i].hist(bkg_samples, bins=bins, alpha=0.5, 
                                         color='blue', label=f'Background (n={len(bkg_samples):,})')
                max_count = max(max_count, n_bkg.max()) if len(n_bkg) > 0 else max_count
                
            if len(signal_samples) > 0:
                n_sig, _, _ = axes[i].hist(signal_samples, bins=bins, alpha=0.5, 
                                        color='red', label=f'Signal (n={len(signal_samples):,})')
                max_count = max(max_count, n_sig.max()) if len(n_sig) > 0 else max_count
            
            # Add legend and title
            axes[i].legend(loc='upper right')
            axes[i].set_title(f'{class_name}')
            axes[i].set_yscale('log')
            axes[i].grid(True, which="both", ls="--", alpha=0.3)
        
        # Set common labels
        fig.text(0.5, 0.01, 'Model Prediction Score', ha='center', va='center', fontsize=12)
        fig.text(0.01, 0.5, 'Count (log scale)', ha='center', va='center', 
                rotation='vertical', fontsize=12)
        
        # Set super title
        fig.suptitle('Prediction Distributions by Class and Signal Status', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.07)
        
        # Save or show the plot
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved multiclass distribution plot to {output_path}")
        else:
            plt.show()
            
        plt.close()


# Function to demonstrate usage
def demo_plots():
    """Demonstrate the plotting functions with sample data."""
    # Create sample data
    np.random.seed(42)
    
    # Sample size
    n = 10000
    
    # Create predictions
    signal_preds = np.random.beta(2, 5, size=n//2)  # Shape: right-skewed for signal
    bkg_preds = np.random.beta(1, 3, size=n//2)  # Shape: more skewed for background
    
    # Create class labels
    classes = ['minbias', 'JpsiPhi', 'Bp_kpJpsi', 'D0pi', 'D0pipi', 'B0_Dstmunu']
    class_probs = [0.6, 0.1, 0.1, 0.1, 0.05, 0.05]  # Probability of each class
    
    # Create dataframe
    df = pd.DataFrame({
        'nn_prediction': np.concatenate([signal_preds, bkg_preds]),
        'Event_Signal': np.concatenate([np.ones(n//2), np.zeros(n//2)]),
        'channel': np.random.choice(classes, size=n, p=class_probs)
    })
    
    # Create output directory
    output_dir = '/work/submit/blaised/TopoNNOps/outputs/demo_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot prediction distributions by class
    PredictionPlotter.plot_prediction_distributions_by_class(
        df, 
        output_path=f'{output_dir}/prediction_distributions.png'
    )
    
    # Plot signal vs background
    PredictionPlotter.plot_signal_background_distributions(
        df,
        output_path=f'{output_dir}/signal_vs_background.png'
    )
    
    # Plot confusion matrix
    PredictionPlotter.plot_enhanced_confusion_matrix(
        df,
        threshold=0.5,
        output_path=f'{output_dir}/confusion_matrix.png'
    )
    
    # Plot normalized confusion matrix
    PredictionPlotter.plot_enhanced_confusion_matrix(
        df,
        threshold=0.5,
        normalize=True,
        output_path=f'{output_dir}/confusion_matrix_normalized.png'
    )
    
    # Plot multiclass distributions
    PredictionPlotter.plot_multiclass_distributions(
        df,
        output_path=f'{output_dir}/multiclass_distributions.png'
    )
    
    print(f"Demo plots saved to {output_dir}")


if __name__ == "__main__":
    demo_plots()
