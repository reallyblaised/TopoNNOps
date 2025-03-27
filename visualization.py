import numpy as np
import pandas as pd
import torch
import altair as alt
import mlflow
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import torch.nn as nn
import tempfile
import os
from typing import Optional

# Set Altair rendering options
alt.data_transformers.enable('default', max_rows=None)

class ModelPerformance:
    """Model visualization with ROC, PR curves and NN response distributions"""

    def __init__(self, model: nn.Module, feature_names: list, device: torch.device):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def create_performance_dashboard(
        self, X_test: np.ndarray, y_test: np.ndarray, 
        history: dict, epoch: int,
        channels: Optional[np.ndarray] = None,
        preprocessor: Optional[object] = None # fetch the preprocessor to invert the [0,1]-normalisation
    ) -> None:
        """
        Create and log performance metrics dashboard with efficiency histograms.
        
        Args:
            X_test: Feature matrix of test data
            y_test: Target labels of test data
            history: Training history dictionary
            epoch: Current epoch number
            channels: Array of channel labels for each sample (optional)
        """
        # Fetch the preprocessor object 
        self.preprocessor = preprocessor

        # Get predictions
        y_test_pred = self._get_predictions(X_test)
        
        # Validate dimensions of our data
        if channels is not None:
            if len(channels) != len(X_test):
                print(f"WARNING: Channel length mismatch. Channels: {len(channels)}, X: {len(X_test)}")
                # Take the smaller length to avoid index errors
                min_len = min(len(channels), len(X_test))
                channels = channels[:min_len]
                X_test = X_test[:min_len]
                y_test = y_test[:min_len]
                y_test_pred = y_test_pred[:min_len]
                print(f"Adjusted to length: {min_len}")
        
        # Create charts
        learning_chart = self._create_learning_curves(history)
        metrics_chart = self._create_metric_evolution(history)
        feature_importance = self._create_feature_importance(X_test, y_test)
        confusion_matrix = self._create_confusion_matrix(y_test, y_test_pred)
        roc_curve_chart = self._create_roc_curve(y_test, y_test_pred)
        pr_curve_chart = self._create_pr_curve(y_test, y_test_pred)
        response_distribution = self._create_response_distribution(y_test, y_test_pred)
        correlation_matrices = self._create_feature_correlation_matrices(X_test, y_test)
        
        # Create standard dashboard components
        dashboard_components = [
            alt.hconcat(learning_chart, metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(feature_importance, confusion_matrix).resolve_scale(color='independent'),
            alt.hconcat(roc_curve_chart, pr_curve_chart).resolve_scale(color='independent'),
            alt.hconcat(response_distribution).resolve_scale(color='independent'),
            correlation_matrices  # Add correlation matrices to dashboard
        ]
        
        # Add efficiency vs PT histogram if channel information is available
        if channels is not None:
            try:
                # Look for TwoBody_PT in feature names
                if 'TwoBody_PT' in self.feature_names:
                    pt_index = self.feature_names.index('TwoBody_PT')
                    pt_values = X_test[:, pt_index]
                    
                    # Add standard fixed-threshold efficiency charts
                    efficiency_chart = self._create_channel_efficiency_vs_pt(
                        X_test, y_test, channels, y_test_pred, pt_values, pt_index
                    )
                    dashboard_components.append(efficiency_chart)
                    
                    # Add dynamic threshold efficiency charts for specific minbias rejection rates
                    dynamic_efficiency_chart = self._create_dynamic_threshold_efficiency_vs_pt(
                        X_test, y_test, channels, y_test_pred, pt_values, pt_index
                    )
                    dashboard_components.append(dynamic_efficiency_chart)
                else:
                    print("TwoBody_PT not found in features, skipping efficiency chart")
            except Exception as e:
                print(f"Error creating efficiency chart: {str(e)}")
                # Add a placeholder instead
                df = pd.DataFrame([{'message': f'Error creating efficiency chart: {str(e)}'}])
                placeholder = alt.Chart(df).mark_text().encode(text='message:N').properties(width=800, height=200)
                dashboard_components.append(placeholder)
        
        # Combine charts into dashboard
        dashboard = alt.vconcat(*dashboard_components).properties(
            title=f"Model Performance Dashboard - Epoch {epoch}"
        )
        
        # Log to MLflow
        self._log_dashboard(dashboard, epoch) 

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return torch.sigmoid(outputs).cpu().numpy()

    def _create_learning_curves(self, history: dict) -> alt.Chart:
        """Generate learning curves visualization"""
        # Create DataFrame
        epochs = range(len(history["train_loss"]))
        df = pd.DataFrame({
            'Epoch': list(epochs),
            'Train Loss': history["train_loss"],
            'Validation Loss': [m["loss"] for m in history["eval_metrics"]]
        })
        
        # Reshape data for Altair
        df_long = pd.melt(
            df, 
            id_vars=['Epoch'], 
            value_vars=['Train Loss', 'Validation Loss'],
            var_name='Set', 
            value_name='Loss'
        )
        
        # Create learning curve visualization
        chart = alt.Chart(df_long).mark_line(point=True).encode(
            x=alt.X('Epoch:Q', title='Epoch'),
            y=alt.Y('Loss:Q', title='Loss Value', scale=alt.Scale(zero=False)),
            color=alt.Color('Set:N', scale=alt.Scale(
                domain=['Train Loss', 'Validation Loss'],
                range=['#1f77b4', '#d62728']
            )),
            tooltip=['Epoch:Q', 'Loss:Q', 'Set:N']
        ).properties(
            width=400,
            height=300,
            title='Learning Curves'
        ).interactive()
        
        return chart

    def _create_metric_evolution(self, history: dict) -> alt.Chart:
        """Generate metric evolution visualization"""
        # Extract metrics excluding loss
        metrics = [key for key in history["eval_metrics"][0].keys() if key != "loss"]
        
        # Create DataFrame
        epochs = range(len(history["eval_metrics"]))
        data = []
        
        for epoch, metrics_dict in zip(epochs, history["eval_metrics"]):
            for metric in metrics:
                data.append({
                    'Epoch': epoch,
                    'Metric': metric.capitalize(),
                    'Value': metrics_dict[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create metric evolution chart
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('Epoch:Q', title='Epoch'),
            y=alt.Y('Value:Q', title='Metric Value'),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Epoch:Q', 'Value:Q', 'Metric:N']
        ).properties(
            width=400,
            height=300,
            title='Validation Metrics Evolution'
        ).interactive()
        
        return chart

    def _create_feature_importance(self, X: np.ndarray, y: np.ndarray) -> alt.Chart:
        """Create a simple feature importance chart based on correlation"""
        importance_data = []
        
        for i, feature_name in enumerate(self.feature_names):
            # Calculate absolute correlation as a simple measure
            corr = np.abs(np.corrcoef(X[:, i], y.ravel())[0, 1])
            importance_data.append({
                'Feature': feature_name,
                'Importance': corr
            })
        
        # Create DataFrame and sort by importance
        df = pd.DataFrame(importance_data)
        df = df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Importance:Q', title='Correlation Importance'),
            y=alt.Y('Feature:N', title='Feature', sort='-x'),
            tooltip=['Feature:N', 'Importance:Q']
        ).properties(
            width=400,
            height=300,
            title='Feature Importance (Correlation)'
        ).interactive()
        
        return chart

    def _create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Create an improved confusion matrix visualization with sanity checks"""
        # Apply threshold for binary prediction
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Ensure arrays are properly flattened
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred_binary.ravel()
        
        # Calculate confusion matrix
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
        tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        
        # # Print debugging info to logs
        # print(f"Confusion Matrix Raw Counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Total number of samples
        total = tn + fp + fn + tp
        
        # Format as DataFrame for visualization with global normalization
        cm_data = [
            {'Predicted': 'Negative', 'Actual': 'Negative', 'Count': int(tn), 'Normalized': float(tn) / float(total)},
            {'Predicted': 'Positive', 'Actual': 'Negative', 'Count': int(fp), 'Normalized': float(fp) / float(total)},
            {'Predicted': 'Negative', 'Actual': 'Positive', 'Count': int(fn), 'Normalized': float(fn) / float(total)},
            {'Predicted': 'Positive', 'Actual': 'Positive', 'Count': int(tp), 'Normalized': float(tp) / float(total)}
        ]
        
        # # Double check that the DataFrame was constructed correctly
        # print("DataFrame construction verification:")
        # for item in cm_data:
        #     print(f"  {item['Actual']} / {item['Predicted']}: {item['Count']}")
        
        df = pd.DataFrame(cm_data)
        
        # Calculate metrics using sklearn functions (Method 1)
        accuracy_sklearn = accuracy_score(y_true_flat, y_pred_flat)
        precision_sklearn = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall_sklearn = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        f1_sklearn = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        
        # print(f"Sklearn metrics: Acc={accuracy_sklearn:.3f}, Prec={precision_sklearn:.3f}, Rec={recall_sklearn:.3f}, F1={f1_sklearn:.3f}")
        
        # Calculate metrics manually from confusion matrix (Method 2)
        accuracy_manual = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
        
        # print(f"Manual metrics: Acc={accuracy_manual:.3f}, Prec={precision_manual:.3f}, Rec={recall_manual:.3f}, F1={f1_manual:.3f}")
        
        # Sanity check - verify metrics match between the two calculation methods
        metrics_match = {
            'accuracy': np.isclose(accuracy_sklearn, accuracy_manual, atol=1e-5),
            'precision': np.isclose(precision_sklearn, precision_manual, atol=1e-5),
            'recall': np.isclose(recall_sklearn, recall_manual, atol=1e-5),
            'f1': np.isclose(f1_sklearn, f1_manual, atol=1e-5)
        }
        
        # Additional sanity checks for confusion matrix consistency
        # These should always be true if the confusion matrix is calculated correctly
        matrix_total_check = (tn + fp + fn + tp) == len(y_true_flat)
        pred_total_check = np.sum(y_pred_flat) == (tp + fp)
        actual_total_check = np.sum(y_true_flat) == (tp + fn)
        
        # # Deep verification of matrix construction
        cross_check = (
            f"Matrix Totals: {tn+fp+fn+tp}, Expected: {len(y_true_flat)}\n"
            f"Predicted Pos: {tp+fp}, Sum(y_pred==1): {np.sum(y_pred_flat)}\n"
            f"Actual Pos: {tp+fn}, Sum(y_true==1): {np.sum(y_true_flat)}"
        )
        # print(cross_check)
        
        # Create heatmap
        base = alt.Chart(df).encode(
            x=alt.X('Predicted:N', title='Predicted Class', sort=['Negative', 'Positive']),
            y=alt.Y('Actual:N', title='Actual Class', sort=['Negative', 'Positive'])
        )
        
        # Create heatmap with global normalization
        heatmap = base.mark_rect().encode(
            color=alt.Color('Normalized:Q', 
                        scale=alt.Scale(scheme='blues'), 
                        title='Fraction of Total')
        )
        
        text = base.mark_text(baseline='middle').encode(
            text='Count:Q',
            color=alt.condition(
                alt.datum.Normalized > 0.3,  # Adjusted threshold for better readability
                alt.value('white'),
                alt.value('black')
            )
        )
        
        # Add overall metrics as text
        metrics_text = alt.Chart(pd.DataFrame([{
            'x': 0.5,
            'y': -0.1,
            'text': f"Accuracy: {accuracy_manual:.3f} | Precision: {precision_manual:.3f} | Recall: {recall_manual:.3f} | F1: {f1_manual:.3f}"
        }])).mark_text(
            align='center',
            baseline='top',
            fontSize=12
        ).encode(
            x=alt.value(200),  # Center in the visualization
            y=alt.value(250),  # Below the confusion matrix
            text='text:N'
        )
        
        # Add false positive rate and true positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        rates_text = alt.Chart(pd.DataFrame([{
            'x': 0.5,
            'y': -0.2,
            'text': f"FPR: {fpr:.3f} | TPR: {tpr:.3f} | Total samples: {total}"
        }])).mark_text(
            align='center',
            baseline='top',
            fontSize=12
        ).encode(
            x=alt.value(200),  # Center in the visualization
            y=alt.value(280),  # Below the previous metrics
            text='text:N'
        )
        
        # Add more detailed calculations
        class_sizes_text = alt.Chart(pd.DataFrame([{
            'x': 0.5,
            'y': -0.25,
            'text': f"Actual Class Sizes - Negative: {int(tn+fp)} | Positive: {int(tp+fn)}"
        }])).mark_text(
            align='center',
            baseline='top',
            fontSize=12
        ).encode(
            x=alt.value(200),
            y=alt.value(310),
            text='text:N'
        )
        
        # Add sanity check information
        sanity_checks = []
        
        # Check if metrics match between sklearn and manual calculation
        for metric, matches in metrics_match.items():
            status = "✓" if matches else "✗"
            sanity_checks.append(f"{metric.capitalize()}: {status}")
        
        # Check if confusion matrix is consistent with data
        matrix_status = "✓" if matrix_total_check else "✗"
        pred_status = "✓" if pred_total_check else "✗" 
        actual_status = "✓" if actual_total_check else "✗"
        
        sanity_text = alt.Chart(pd.DataFrame([{
            'x': 0.5,
            'y': -0.3,
            'text': f"Sanity checks: {', '.join(sanity_checks)} | Matrix consistency: {matrix_status}"
        }])).mark_text(
            align='center',
            baseline='top',
            fontSize=12,
            color='green' if all(metrics_match.values()) and matrix_total_check else 'red'
        ).encode(
            x=alt.value(200),  # Center in the visualization
            y=alt.value(340),  # Below the previous metrics
            text='text:N'
        )
        
        # Combine elements
        matrix = (heatmap + text).properties(
            width=400,
            height=250,
            title='Confusion Matrix (Globally Normalized)'
        )
        
        #return alt.vconcat(matrix, metrics_text, rates_text, class_sizes_text, sanity_text).resolve_scale()
        return alt.vconcat(matrix).resolve_scale()

    def _create_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate ROC curve visualization"""
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred.ravel())
            roc_auc = auc(fpr, tpr)
                       
            # Create chart data manually to ensure alignment
            data = []
            for i in range(len(fpr)):
                thresh_val = thresholds[i-1] if i > 0 and i < len(thresholds) + 1 else 0
                data.append({
                    'False Positive Rate': float(fpr[i]),
                    'True Positive Rate': float(tpr[i]),
                    'Threshold': float(thresh_val)
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create ROC curve
            line = alt.Chart(df).mark_line().encode(
                x=alt.X('False Positive Rate:Q', title='False Positive Rate'),
                y=alt.Y('True Positive Rate:Q', title='True Positive Rate'),
                tooltip=['False Positive Rate:Q', 'True Positive Rate:Q', 'Threshold:Q']
            )
            
            # Add reference line
            reference = alt.Chart(pd.DataFrame({
                'x': [0, 1],
                'y': [0, 1]
            })).mark_line(strokeDash=[4, 4], color='gray').encode(
                x='x',
                y='y'
            )
            
            # Add AUC text annotation
            text = alt.Chart(pd.DataFrame({
                'x': [0.6],
                'y': [0.2],
                'text': [f'AUC = {roc_auc:.3f}']
            })).mark_text(align='left', baseline='middle').encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            # Combine all elements
            chart = (line + reference + text).properties(
                width=400,
                height=300,
                title='ROC Curve'
            ).interactive()
            
            return chart
        
        except Exception as e:
            # Fallback if ROC curve fails
            print(f"Error creating ROC curve: {str(e)}")
            df = pd.DataFrame([{'message': f"Could not compute ROC curve: {str(e)}"}])
            return alt.Chart(df).mark_text(
                align='center',
                baseline='middle'
            ).encode(
                text='message:N'
            ).properties(
                width=400,
                height=300,
                title='ROC Curve (Failed)'
            )

    def _create_pr_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate Precision-Recall curve visualization"""
        try:
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred.ravel())
            avg_precision = average_precision_score(y_true, y_pred.ravel()) 
            
            # Create chart data manually to ensure alignment
            data = []
            for i in range(len(precision)):
                thresh_val = thresholds[i-1] if i > 0 and i < len(thresholds) + 1 else 1.0
                data.append({
                    'Precision': float(precision[i]),
                    'Recall': float(recall[i]),
                    'Threshold': float(thresh_val)
                })
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create PR curve
            line = alt.Chart(df).mark_line().encode(
                x=alt.X('Recall:Q', title='Recall'),
                y=alt.Y('Precision:Q', title='Precision', scale=alt.Scale(zero=False)),
                tooltip=['Recall:Q', 'Precision:Q', 'Threshold:Q']
            )
            
            # Add baseline (class distribution)
            baseline = np.mean(y_true)
            baseline_line = alt.Chart(pd.DataFrame({
                'x': [0, 1],
                'y': [baseline, baseline]
            })).mark_line(strokeDash=[4, 4], color='gray').encode(
                x='x',
                y='y'
            )
            
            # Add AP text annotation
            text = alt.Chart(pd.DataFrame({
                'x': [0.6],
                'y': [0.9],
                'text': [f'AP = {avg_precision:.3f}']
            })).mark_text(align='left', baseline='middle').encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            # Combine all elements
            chart = (line + baseline_line + text).properties(
                width=400,
                height=300,
                title='Precision-Recall Curve'
            ).interactive()
            
            return chart
            
        except Exception as e:
            # Fallback if PR curve fails
            print(f"Error creating PR curve: {str(e)}")
            df = pd.DataFrame([{'message': f"Could not compute PR curve: {str(e)}"}])
            return alt.Chart(df).mark_text(
                align='center',
                baseline='middle'
            ).encode(
                text='message:N'
            ).properties(
                width=400,
                height=300,
                title='PR Curve (Failed)'
            )

    def _create_response_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Create a histogram of NN response for signal and background with log scale and more bins"""
        # Create separate dataframes for signal and background
        signal_df = pd.DataFrame({
            'Response': y_pred.ravel()[y_true.ravel() == 1],
            'Class': 'Signal'
        })
        
        background_df = pd.DataFrame({
            'Response': y_pred.ravel()[y_true.ravel() == 0],
            'Class': 'Background'
        })
        
        # Combine data
        df = pd.concat([signal_df, background_df])
        
        # Create histogram with unstacked bars, specific colors, log scale, and more bins
        chart = alt.Chart(df).mark_bar(opacity=0.6).encode(
            x=alt.X('Response:Q', 
                    bin=alt.Bin(maxbins=100),  
                    title='Neural Network Response [0,1]'),
            y=alt.Y('count()', 
                    title='Count (log scale)', 
                    stack=None,  # No stacking
                    scale=alt.Scale(type='log')),  # Log scale for y-axis
            color=alt.Color('Class:N', scale=alt.Scale(
                domain=['Background', 'Signal'],
                range=['#1f77b4', '#d62728']  # Blue for background, red for signal
            )),
            tooltip=['Class:N', 'count()']
        ).properties(
            width=800,
            height=300,
            title='Neural Network Response Distribution (Log Scale)'
        ).interactive()
        
        return chart

    def _create_channel_efficiency_vs_pt(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                channels: np.ndarray,
                                y_pred: np.ndarray = None,
                                pt_values: np.ndarray = None, 
                                pt_index: int = None) -> alt.Chart:
        """
        Create efficiency histograms in TwoBody_PT for specific signal channels.
        
        Shows efficiency at different cut values with uncertainty bands.
        Also displays minbias rejection rate for each cut value.
        Includes normalized signal PT distribution as shaded background.
        
        Args:
            X: Feature matrix
            y: Target labels
            channels: Array of channel names for each sample
            y_pred: Predicted probabilities (if None, will be computed)
            pt_values: PT values extracted from X (if None, will be extracted using pt_index)
            pt_index: Index of PT feature in X (if None, will be determined from feature_names)
        """        
        # Check if channels array is provided and has correct shape
        if channels is None or len(channels) != len(X):
            # Return informative chart if channels data is incompatible
            return alt.Chart(pd.DataFrame({'message': [f'Channel data shape mismatch: X shape={X.shape}, channels shape={channels.shape if channels is not None else None}']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Get predictions if not provided
        if y_pred is None:
            y_pred = self._get_predictions(X)
            
        # Extract TwoBody_PT values if not provided
        if pt_values is None:
            if pt_index is None:
                if 'TwoBody_PT' not in self.feature_names:
                    # Return empty chart if PT values aren't available
                    return alt.Chart(pd.DataFrame({'message': ['TwoBody_PT not found in features']})).mark_text().encode(
                        text='message:N'
                    ).properties(width=800, height=200)
                
                # Get the index of TwoBody_PT in features
                pt_index = self.feature_names.index('TwoBody_PT')
            
            pt_values = X[:, pt_index]

        # DENORMALIZATION CODE STARTS
        # -----------------------------------------------
        # Check if we have a preprocessor available
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            feature_name = 'TwoBody_PT'
            
            # Check if the preprocessor normalized this feature
            if (hasattr(self.preprocessor, 'normalize') and 
                self.preprocessor.normalize and 
                hasattr(self.preprocessor, 'feature_stats')):
                
                # Get the min/max values used for normalization (stored during fit)
                pt_min = self.preprocessor.feature_stats.get(f'{feature_name}_min')
                pt_max = self.preprocessor.feature_stats.get(f'{feature_name}_max')
                
                if pt_min is not None and pt_max is not None:
                    # Apply inverse transform: normalized_value * (max - min) + min
                    pt_values = pt_values * (pt_max - pt_min) + pt_min
                    
                    # Log the transformation for debugging
                    print(f"Denormalized {feature_name} from [0,1] to [{pt_min:.3f}, {pt_max:.3f}] GeV")
        # -----------------------------------------------
        # DENORMALIZATION CODE ENDS
        breakpoint()
        # Define cut values
        cut_values = [0.75, 0.95, 0.99, 0.995]
        
        # Selected signal channels to analyze
        selected_channels = ['Bs_JpsiPhi', 'Bp_KpJpsi', 'B0_D0pipi', 'B0_DpDm', 'B0_Dstmunu', 'Bm_D0K']
        
        # Define PT bins from 0 to 20 GeV in 1 GeV increments
        pt_bins = np.linspace(0, 20, 21)  # 21 points to get 20 bins of 1.0 GeV each
        
        # Compute minbias rejection for each cut value
        # Ensure all arrays have compatible shapes
        minbias_mask = (channels == 'minbias')
        minbias_total = np.sum(minbias_mask)
        
        minbias_rejection = {}
        
        # Ensure prediction array is flattened to match dimension of mask
        flattened_preds = y_pred.ravel()
        
        # Verify shapes match
        if len(minbias_mask) != len(flattened_preds):
            # Return error message chart when shapes don't match
            return alt.Chart(pd.DataFrame({'message': [f'Shape mismatch: minbias_mask={minbias_mask.shape}, predictions={flattened_preds.shape}']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        for cut in cut_values:
            minbias_rejected = np.sum(minbias_mask & (flattened_preds < cut))
            rejection_rate = minbias_rejected / minbias_total if minbias_total > 0 else 0
            minbias_rejection[cut] = rejection_rate
        
        # Data structure to store efficiency results
        efficiency_data = []
        
        # Dictionary to store overall efficiency per channel per cut
        overall_efficiencies = {}
        
        # Calculate overall efficiencies for each channel and cut value
        for channel in selected_channels:
            channel_mask = (channels == channel)
            channel_total = np.sum(channel_mask)
            
            # Skip if no events in this channel
            if channel_total == 0:
                continue
                
            # Calculate overall efficiency for each cut
            for cut in cut_values:
                if channel not in overall_efficiencies:
                    overall_efficiencies[channel] = {}
                    
                # Count signal events passing the cut
                passing_cut = np.sum(channel_mask & (y_pred.ravel() >= cut))
                
                # Calculate overall efficiency
                if channel_total > 0:
                    overall_eff = passing_cut / channel_total
                    overall_efficiencies[channel][cut] = {
                        'efficiency': overall_eff,
                        'passing': int(passing_cut),
                        'total': int(channel_total)
                    }
        
        # Calculate efficiency for each channel, cut value, and PT bin
        for channel in selected_channels:
            channel_mask = (channels == channel)
            channel_total = np.sum(channel_mask)
            
            # Skip if no events in this channel
            if channel_total == 0:
                continue
                
            # Process each cut value
            for cut in cut_values:
                # Calculate efficiency in each PT bin
                for i in range(len(pt_bins) - 1):
                    pt_min, pt_max = pt_bins[i], pt_bins[i+1]
                    
                    # Select events in this channel and PT bin
                    pt_mask = (pt_values >= pt_min) & (pt_values < pt_max)
                    combined_mask = channel_mask & pt_mask
                    
                    # Skip bins with no events
                    total_events = np.sum(combined_mask)
                    if total_events == 0:
                        continue
                    
                    # Count events passing the cut
                    passing_cut = np.sum(combined_mask & (y_pred.ravel() >= cut))
                    
                    # Calculate efficiency
                    efficiency = passing_cut / total_events
                    
                    # Calculate binomial error
                    # Standard error for binomial proportion: sqrt(p*(1-p)/n)
                    error = np.sqrt((efficiency * (1 - efficiency)) / total_events)
                    
                    # Store results
                    efficiency_data.append({
                        'Channel': channel,
                        'Cut': str(cut),  # Convert to string for proper legend grouping
                        'PT_Bin': (pt_min + pt_max) / 2,  # Bin center for plotting
                        'PT_Min': pt_min,
                        'PT_Max': pt_max,
                        'Efficiency': efficiency,
                        'Error': error,
                        'Efficiency_lower': max(0, efficiency - error),
                        'Efficiency_upper': min(1, efficiency + error),
                        'TotalEvents': total_events,
                        'PassingEvents': passing_cut,
                        'MinbiasRejection': minbias_rejection[cut]
                    })
        
        # Create DataFrame for visualization
        df = pd.DataFrame(efficiency_data)
        
        # Return early if no data is available
        if len(df) == 0:
            return alt.Chart(pd.DataFrame({'message': ['No efficiency data available for selected channels']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Create PT distribution data for all beauty channels combined
        beauty_channels_mask = np.zeros_like(channels, dtype=bool)
        for channel in selected_channels:
            beauty_channels_mask |= (channels == channel)
        
        beauty_pt_values = pt_values[beauty_channels_mask]
        
        # Create histogram data
        hist, bin_edges = np.histogram(beauty_pt_values, bins=pt_bins)
        
        # Normalize the histogram so that the tallest bin is 1.0
        max_bin_height = np.max(hist)
        if max_bin_height > 0:
            hist = hist / max_bin_height
        
        # Create DataFrame for PT distribution
        pt_dist_data = []
        for i in range(len(hist)):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            pt_dist_data.append({
                'PT_Bin': bin_center,
                'Normalized_Height': hist[i]
            })
        
        pt_dist_df = pd.DataFrame(pt_dist_data)
        
        # Create separate charts for each cut value
        cut_charts = []
        
        for cut in cut_values:
            # Filter data for this cut value
            cut_df = df[df['Cut'] == str(cut)]
            
            if len(cut_df) == 0:
                continue
            
            # Create base chart for the PT bins
            base = alt.Chart(pt_dist_df).encode(
                x=alt.X('PT_Bin:Q', scale=alt.Scale(domain=[0, 20]), title='TwoBody_PT [GeV]')
            )
            
            # Add the PT distribution as a gray shaded area
            pt_area = base.mark_area(opacity=0.3, color='gray').encode(
                y='Normalized_Height:Q'
            )
            
            # Create base for efficiency plots
            eff_base = alt.Chart(cut_df).encode(
                x=alt.X('PT_Bin:Q')
            )
            
            # Create uncertainty bands
            bands = eff_base.mark_area(opacity=0.3).encode(
                y=alt.Y('Efficiency_lower:Q'),
                y2=alt.Y2('Efficiency_upper:Q'),
                color=alt.Color('Channel:N', scale=alt.Scale(scheme='category10'))
            )
            
            # Create lines connecting the points
            lines = eff_base.mark_line().encode(
                y=alt.Y('Efficiency:Q', title='Efficiency', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('Channel:N')
            )
            
            # Create points
            points = eff_base.mark_circle(size=50).encode(
                y=alt.Y('Efficiency:Q'),
                color=alt.Color('Channel:N'),
                tooltip=['Channel:N', 'PT_Bin:Q', 'Efficiency:Q', 'Error:Q', 
                        'TotalEvents:Q', 'PassingEvents:Q']
            )
            
            # Add minbias rejection and channel efficiency text
            text_data = []
            
            # First, add minbias rejection
            text_data.append({
                'x': 10,
                'y': 0.08,
                'text': f"Minbias rejection: {minbias_rejection[cut]:.4f}",
                'channel': 'Minbias'
            })
            
            # Then add efficiency for each channel
            y_position = 0.15
            for channel in selected_channels:
                if channel in overall_efficiencies and cut in overall_efficiencies[channel]:
                    eff_info = overall_efficiencies[channel][cut]
                    text_data.append({
                        'x': 10,
                        'y': y_position,
                        'text': f"{channel}: {eff_info['efficiency']:.4f} ({eff_info['passing']}/{eff_info['total']})",
                        'channel': channel
                    })
                    y_position += 0.05
            
            # Create text marks with very small font size
            text_marks = alt.Chart(pd.DataFrame(text_data)).mark_text(
                align='center',
                baseline='middle',
                fontSize=8  # Very small font size
            ).encode(
                x='x:Q',
                y='y:Q',
                text='text:N',
                color=alt.condition(
                    alt.datum.channel == 'Minbias',
                    alt.value('black'),
                    alt.Color('channel:N', scale=alt.Scale(scheme='category10'))
                )
            )
            
            # Combine all elements
            combined_chart = alt.layer(
                pt_area,
                bands,
                lines,
                points,
                text_marks
            ).properties(
                width=400,
                height=300,
                title=f'Channel Efficiency vs PT (Cut = {cut})'
            )
            
            cut_charts.append(combined_chart)
        
        # Combine all cut charts into a single view
        if cut_charts:
            # Create title for overall chart
            title_chart = alt.Chart(pd.DataFrame([{'text': 'Signal Channel Efficiency vs PT with Uncertainty Bands'}])).mark_text(
                fontSize=16,
                font='Arial',
                fontWeight='bold'
            ).encode(
                text='text:N'
            )
            
            # Arrange charts in a 2x2 grid
            if len(cut_charts) >= 4:
                final_chart = alt.vconcat(
                    alt.hconcat(cut_charts[0], cut_charts[1]).resolve_scale(color='shared'),
                    alt.hconcat(cut_charts[2], cut_charts[3]).resolve_scale(color='shared')
                ).resolve_scale(color='shared')
            elif len(cut_charts) == 3:
                final_chart = alt.vconcat(
                    alt.hconcat(cut_charts[0], cut_charts[1]).resolve_scale(color='shared'),
                    cut_charts[2]
                ).resolve_scale(color='shared')
            elif len(cut_charts) == 2:
                final_chart = alt.hconcat(
                    cut_charts[0], cut_charts[1]
                ).resolve_scale(color='shared')
            else:
                final_chart = cut_charts[0]
            
            return alt.vconcat(title_chart, final_chart).resolve_scale(color='shared')
        else:
            return alt.Chart(pd.DataFrame({'message': ['No efficiency data available']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
            
    def _log_dashboard(self, dashboard, epoch: int) -> None:
        """Log the Altair dashboard to MLflow"""
        try:
            # Convert to HTML
            html = dashboard.to_html()
            
            # Create a temporary file and save
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                f.write(html.encode('utf-8'))
                temp_path = f.name
            
            # Log to MLflow
            mlflow.log_artifact(temp_path, f"visualization/performance_dashboard_epoch_{epoch}")
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Failed to log dashboard: {str(e)}")

    def _create_feature_correlation_matrices(self, X: np.ndarray, y: np.ndarray) -> alt.Chart:
        """
        Create feature correlation heatmaps separately for signal and background samples.
        
        Args:
            X: Feature matrix
            y: Target labels (binary)
        
        Returns:
            alt.Chart: Altair chart with two correlation matrices
        """
        # Separate signal and background samples
        X_signal = X[y.ravel() == 1]
        X_background = X[y.ravel() == 0]
        
        # Check if we have enough samples in each class
        if len(X_signal) < 2 or len(X_background) < 2:
            # Return an error message if not enough samples
            message = f"Not enough samples to compute correlations. Signal: {len(X_signal)}, Background: {len(X_background)}"
            return alt.Chart(pd.DataFrame({'message': [message]})).mark_text().encode(
                text='message'
            ).properties(width=800, height=400)
        
        # Compute correlation matrices
        signal_corr = np.corrcoef(X_signal, rowvar=False)
        background_corr = np.corrcoef(X_background, rowvar=False)
        
        # Create DataFrames with feature names
        signal_corr_df = pd.DataFrame(signal_corr, 
                                    columns=self.feature_names, 
                                    index=self.feature_names)
        background_corr_df = pd.DataFrame(background_corr, 
                                        columns=self.feature_names, 
                                        index=self.feature_names)
        
        # Reshape for visualization (long format)
        signal_corr_long = signal_corr_df.reset_index().melt(
            id_vars='index', 
            value_name='correlation', 
            var_name='feature'
        )
        signal_corr_long['type'] = 'Signal'
        
        background_corr_long = background_corr_df.reset_index().melt(
            id_vars='index', 
            value_name='correlation', 
            var_name='feature'
        )
        background_corr_long['type'] = 'Background'
        
        # Combine data
        combined_corr = pd.concat([signal_corr_long, background_corr_long])
        
        # Define color scheme with diverging colors
        # Blue-White-Red palette
        color_scheme = alt.Scale(
            domain=[-1, 0, 1],
            range=['#1f77b4', '#ffffff', '#d62728']  # Blue, White, Red
        )
        
        # Create base heatmap
        base = alt.Chart(combined_corr).encode(
            x=alt.X('feature:N', title=None),
            y=alt.Y('index:N', title=None),
            tooltip=['index:N', 'feature:N', 'correlation:Q', 'type:N']
        )
        
        # Create heatmap with colored cells
        heatmap = base.mark_rect().encode(
            color=alt.Color('correlation:Q', 
                            scale=color_scheme,
                            legend=alt.Legend(title='Correlation'))
        )
        
        # Add correlation values as text
        text = base.mark_text(baseline='middle').encode(
            text=alt.Text('correlation:Q', format='.2f'),
            color=alt.condition(
                abs(alt.datum.correlation) > 0.7,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        # Combine heatmap and text
        matrix = (heatmap + text).properties(
            width=400,
            height=400
        )
        
        # Create faceted view for signal and background without config methods
        faceted_matrix = matrix.facet(
            column=alt.Column('type:N', title=None),
            title='Feature Correlation Matrices'
        )
        
        return faceted_matrix

    def _find_threshold_for_rejection_rate(
        self, 
        y_pred: np.ndarray, 
        channels: np.ndarray, 
        target_rejection_rate: float
    ) -> float:
        """
        Find the exact threshold value that achieves a specified minbias rejection rate.
        
        Args:
            y_pred: Model predictions
            channels: Array of channel labels
            target_rejection_rate: Target minbias rejection rate (e.g., 0.995)
            
        Returns:
            float: Threshold value that achieves the target rejection rate
        """
        # Get minbias samples
        minbias_mask = (channels == 'minbias')
        minbias_preds = y_pred.ravel()[minbias_mask]
        
        if len(minbias_preds) == 0:
            print("Warning: No minbias samples found")
            return 0.5  # Default threshold
        
        # Sort predictions in ascending order
        sorted_preds = np.sort(minbias_preds)
        
        # Find the exact threshold that gives the target rejection rate
        # For 0.995 rejection, we want the value where 99.5% of minbias predictions are below it
        index = int(len(sorted_preds) * target_rejection_rate)
        if index >= len(sorted_preds):
            index = len(sorted_preds) - 1  # Avoid out of bounds
            
        threshold = sorted_preds[index]
        
        # Verify actual rejection rate at this threshold
        actual_rejection = np.mean(minbias_preds <= threshold)
        
        return threshold

    def _create_dynamic_threshold_efficiency_vs_pt(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        channels: np.ndarray, 
        y_pred: np.ndarray,
        pt_values: np.ndarray, 
        pt_index: int
    ) -> alt.Chart:
        """
        Create efficiency histograms in TwoBody_PT for specific signal channels using dynamic thresholds
        that achieve specific minbias rejection rates.
        
        Args:
            X: Feature matrix
            y: Target labels
            channels: Array of channel names for each sample
            y_pred: Predicted probabilities
            pt_values: PT values extracted from X
            pt_index: Index of PT feature in X
        """
        # Check if channels array is provided and has correct shape
        if channels is None or len(channels) != len(X):
            # Return informative chart if channels data is incompatible
            return alt.Chart(pd.DataFrame({'message': [f'Channel data shape mismatch for dynamic thresholds']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Define target rejection rates
        target_rejection_rates = [0.995, 0.999]
        
        # Find thresholds for each target rejection rate
        dynamic_thresholds = {}
        for rate in target_rejection_rates:
            threshold = self._find_threshold_for_rejection_rate(y_pred, channels, rate)
            dynamic_thresholds[rate] = threshold
        
        # Selected signal channels to analyze
        selected_channels = ['Bs_JpsiPhi', 'Bp_KpJpsi', 'B0_D0pipi', 'B0_DpDm', 'B0_Dstmunu', 'Bm_D0K'] 
        
        # Define PT bins from 0 to 20 GeV in 1 GeV increments
        pt_bins = np.linspace(0, 20, 21)  # 21 points to get 20 bins of 1.0 GeV each
        
        # Data structure to store efficiency results
        efficiency_data = []
        
        # Dictionary to store overall efficiency per channel per threshold
        overall_efficiencies = {}
        
        # Calculate overall efficiencies for each channel and dynamic threshold
        for channel in selected_channels:
            channel_mask = (channels == channel)
            channel_total = np.sum(channel_mask)
            
            # Skip if no events in this channel
            if channel_total == 0:
                continue
                
            # Calculate overall efficiency for each threshold
            for rate, threshold in dynamic_thresholds.items():
                rate_key = f"{rate:.3f}"
                if channel not in overall_efficiencies:
                    overall_efficiencies[channel] = {}
                    
                # Count signal events passing the threshold
                passing = np.sum(channel_mask & (y_pred.ravel() >= threshold))
                
                # Calculate overall efficiency
                if channel_total > 0:
                    overall_eff = passing / channel_total
                    overall_efficiencies[channel][rate_key] = {
                        'efficiency': overall_eff,
                        'passing': int(passing),
                        'total': int(channel_total),
                        'threshold': threshold
                    }
        
        # Calculate efficiency for each channel, threshold, and PT bin
        for channel in selected_channels:
            channel_mask = (channels == channel)
            channel_total = np.sum(channel_mask)
            
            # Skip if no events in this channel
            if channel_total == 0:
                continue
                
            # Process each threshold
            for rate, threshold in dynamic_thresholds.items():
                rate_key = f"{rate:.3f}"
                
                # Calculate efficiency in each PT bin
                for i in range(len(pt_bins) - 1):
                    pt_min, pt_max = pt_bins[i], pt_bins[i+1]
                    
                    # Select events in this channel and PT bin
                    pt_mask = (pt_values >= pt_min) & (pt_values < pt_max)
                    combined_mask = channel_mask & pt_mask
                    
                    # Skip bins with no events
                    total_events = np.sum(combined_mask)
                    if total_events == 0:
                        continue
                    
                    # Count events passing the threshold
                    passing = np.sum(combined_mask & (y_pred.ravel() >= threshold))
                    
                    # Calculate efficiency
                    efficiency = passing / total_events
                    
                    # Calculate binomial error
                    error = np.sqrt((efficiency * (1 - efficiency)) / total_events) if total_events > 0 else 0
                    
                    # Store results
                    efficiency_data.append({
                        'Channel': channel,
                        'RejectionRate': rate_key,
                        'Threshold': threshold,
                        'PT_Bin': (pt_min + pt_max) / 2,  # Bin center for plotting
                        'PT_Min': pt_min,
                        'PT_Max': pt_max,
                        'Efficiency': efficiency,
                        'Error': error,
                        'Efficiency_lower': max(0, efficiency - error),
                        'Efficiency_upper': min(1, efficiency + error),
                        'TotalEvents': total_events,
                        'PassingEvents': passing
                    })
        
        # Create DataFrame for visualization
        df = pd.DataFrame(efficiency_data)
        
        # Return early if no data is available
        if len(df) == 0:
            return alt.Chart(pd.DataFrame({'message': ['No dynamic threshold efficiency data available']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Create PT distribution data for all beauty channels combined
        beauty_channels_mask = np.zeros_like(channels, dtype=bool)
        for channel in selected_channels:
            beauty_channels_mask |= (channels == channel)
        
        beauty_pt_values = pt_values[beauty_channels_mask]
        
        # Create histogram data
        hist, bin_edges = np.histogram(beauty_pt_values, bins=pt_bins)
        
        # Normalize the histogram
        max_bin_height = np.max(hist)
        if max_bin_height > 0:
            hist = hist / max_bin_height
        
        # Create DataFrame for PT distribution
        pt_dist_data = []
        for i in range(len(hist)):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            pt_dist_data.append({
                'PT_Bin': bin_center,
                'Normalized_Height': hist[i]
            })
        
        pt_dist_df = pd.DataFrame(pt_dist_data)
        
        # Create separate charts for each rejection rate
        rate_charts = []
        
        for rate in target_rejection_rates:
            rate_key = f"{rate:.3f}"
            threshold = dynamic_thresholds[rate]
            
            # Filter data for this rejection rate
            rate_df = df[df['RejectionRate'] == rate_key]
            
            if len(rate_df) == 0:
                continue
            
            # Create base chart for the PT bins
            base = alt.Chart(pt_dist_df).encode(
                x=alt.X('PT_Bin:Q', scale=alt.Scale(domain=[0, 20]), title='TwoBody_PT [GeV]')
            )
            
            # Add the PT distribution as a gray shaded area
            pt_area = base.mark_area(opacity=0.3, color='gray').encode(
                y='Normalized_Height:Q'
            )
            
            # Create base for efficiency plots
            eff_base = alt.Chart(rate_df).encode(
                x=alt.X('PT_Bin:Q')
            )
            
            # Create uncertainty bands
            bands = eff_base.mark_area(opacity=0.3).encode(
                y=alt.Y('Efficiency_lower:Q'),
                y2=alt.Y2('Efficiency_upper:Q'),
                color=alt.Color('Channel:N', scale=alt.Scale(scheme='category10'))
            )
            
            # Create lines connecting the points
            lines = eff_base.mark_line().encode(
                y=alt.Y('Efficiency:Q', title='Efficiency', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('Channel:N')
            )
            
            # Create points
            points = eff_base.mark_circle(size=50).encode(
                y=alt.Y('Efficiency:Q'),
                color=alt.Color('Channel:N'),
                tooltip=['Channel:N', 'PT_Bin:Q', 'Efficiency:Q', 'Error:Q', 
                        'TotalEvents:Q', 'PassingEvents:Q', 'Threshold:Q']
            )
            
            # Add channel efficiency text and threshold value
            text_data = []
            
            # First, add threshold information
            text_data.append({
                'x': 10,
                'y': 0.08,
                'text': f"Minbias rejection: {rate:.4f} (threshold = {threshold:.6f})",
                'channel': 'Minbias'
            })
            
            # Then add efficiency for each channel
            y_position = 0.15
            for channel in selected_channels:
                if channel in overall_efficiencies and rate_key in overall_efficiencies[channel]:
                    eff_info = overall_efficiencies[channel][rate_key]
                    text_data.append({
                        'x': 10,
                        'y': y_position,
                        'text': f"{channel}: {eff_info['efficiency']:.4f} ({eff_info['passing']}/{eff_info['total']})",
                        'channel': channel
                    })
                    y_position += 0.05
            
            # Create text marks
            text_marks = alt.Chart(pd.DataFrame(text_data)).mark_text(
                align='center',
                baseline='middle',
                fontSize=8
            ).encode(
                x='x:Q',
                y='y:Q',
                text='text:N',
                color=alt.condition(
                    alt.datum.channel == 'Minbias',
                    alt.value('black'),
                    alt.Color('channel:N', scale=alt.Scale(scheme='category10'))
                )
            )
            
            # Combine all elements
            combined_chart = alt.layer(
                pt_area,
                bands,
                lines,
                points,
                text_marks
            ).properties(
                width=400,
                height=300,
                title=f'Channel Efficiency vs PT (Minbias Rejection = {rate:.3f}, Cut = {threshold:.6f})'
            )
            
            rate_charts.append(combined_chart)
        
        # Combine all rejection rate charts into a single view
        if rate_charts:
            # Create title for overall chart
            title_chart = alt.Chart(pd.DataFrame([{'text': 'Signal Channel Efficiency vs PT at Fixed Minbias Rejection Rates'}])).mark_text(
                fontSize=16,
                font='Arial',
                fontWeight='bold'
            ).encode(
                text='text:N'
            )
            
            # Arrange charts horizontally or in a grid
            if len(rate_charts) == 2:
                final_chart = alt.hconcat(
                    rate_charts[0], rate_charts[1]
                ).resolve_scale(color='shared')
            else:
                # Just use whatever charts we have
                final_chart = alt.hconcat(*rate_charts).resolve_scale(color='shared')
            
            return alt.vconcat(title_chart, final_chart).resolve_scale(color='shared')
        else:
            return alt.Chart(pd.DataFrame({'message': ['No dynamic threshold efficiency data available']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)