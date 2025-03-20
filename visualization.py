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
        channels: Optional[np.ndarray] = None
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
        # Get predictions
        y_test_pred = self._get_predictions(X_test)
        
        # Create charts
        learning_chart = self._create_learning_curves(history)
        metrics_chart = self._create_metric_evolution(history)
        feature_importance = self._create_feature_importance(X_test, y_test)
        confusion_matrix = self._create_confusion_matrix(y_test, y_test_pred)
        roc_curve_chart = self._create_roc_curve(y_test, y_test_pred)
        pr_curve_chart = self._create_pr_curve(y_test, y_test_pred)
        response_distribution = self._create_response_distribution(y_test, y_test_pred)
        
        # Create standard dashboard components
        dashboard_components = [
            alt.hconcat(learning_chart, metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(feature_importance, confusion_matrix).resolve_scale(color='independent'),
            alt.hconcat(roc_curve_chart, pr_curve_chart).resolve_scale(color='independent'),
            alt.hconcat(response_distribution).resolve_scale(color='independent')
        ]
        
        # Add efficiency vs PT histogram if channel information is available
        if channels is not None:
            efficiency_chart = self._create_channel_efficiency_vs_pt(X_test, y_test, channels)
            dashboard_components.append(efficiency_chart)
        
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
        """Create a simple confusion matrix visualization"""
        # Apply threshold for binary prediction
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        
        # Format as DataFrame for visualization
        cm_data = [
            {'Predicted': 'Negative', 'Actual': 'Negative', 'Count': int(tn), 'Normalized': float(tn) / float(tn + fp) if (tn + fp) > 0 else 0},
            {'Predicted': 'Positive', 'Actual': 'Negative', 'Count': int(fp), 'Normalized': float(fp) / float(tn + fp) if (tn + fp) > 0 else 0},
            {'Predicted': 'Negative', 'Actual': 'Positive', 'Count': int(fn), 'Normalized': float(fn) / float(fn + tp) if (fn + tp) > 0 else 0},
            {'Predicted': 'Positive', 'Actual': 'Positive', 'Count': int(tp), 'Normalized': float(tp) / float(fn + tp) if (fn + tp) > 0 else 0}
        ]
        
        df = pd.DataFrame(cm_data)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Create heatmap
        base = alt.Chart(df).encode(
            x=alt.X('Predicted:N', title='Predicted Class'),
            y=alt.Y('Actual:N', title='Actual Class')
        )
        
        heatmap = base.mark_rect().encode(
            color=alt.Color('Normalized:Q', scale=alt.Scale(scheme='blues'), title='Fraction of Actual')
        )
        
        text = base.mark_text(baseline='middle').encode(
            text='Count:Q',
            color=alt.condition(
                alt.datum.Normalized > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        # Add metrics as text
        metrics_text = alt.Chart(pd.DataFrame([{
            'x': 0.5,
            'y': -0.1,
            'text': f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
        }])).mark_text(
            align='center',
            baseline='top',
            fontSize=12
        ).encode(
            x=alt.value(200),  # Center in the visualization
            y=alt.value(250),  # Below the confusion matrix
            text='text:N'
        )
        
        # Combine elements
        matrix = (heatmap + text).properties(
            width=400,
            height=250,
            title='Confusion Matrix'
        )
        
        return alt.vconcat(matrix, metrics_text).resolve_scale()

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

    def _create_channel_efficiency_vs_pt(self, 
                                        X: np.ndarray, 
                                        y: np.ndarray, 
                                        channels: np.ndarray) -> alt.Chart:
        """
        Create efficiency histograms in TwoBody_PT for specific signal channels.
        
        Shows efficiency at different cut values (0.75, 0.9, 0.95, 0.99) with binomial errors.
        Also displays minbias rejection rate for each cut value.
        
        Args:
            X: Feature matrix
            y: Target labels
            channels: Array of channel names for each sample
            
        Returns:
            Altair chart with efficiency histograms
        """
        # Check if channels array is provided and has correct shape
        if channels is None or len(channels) != len(X):
            # Return informative chart if channels data is incompatible
            return alt.Chart(pd.DataFrame({'message': [f'Channel data shape mismatch: X shape={X.shape}, channels shape={channels.shape if channels is not None else None}']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Extract TwoBody_PT values if available in feature set
        if 'TwoBody_PT' not in self.feature_names:
            # Return empty chart if PT values aren't available
            return alt.Chart(pd.DataFrame({'message': ['TwoBody_PT not found in features']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Get the index of TwoBody_PT in features
        pt_index = self.feature_names.index('TwoBody_PT')
        pt_values = X[:, pt_index]
        
        # Get model predictions
        y_pred = self._get_predictions(X)
        
        # Define cut values
        cut_values = [0.75, 0.9, 0.95, 0.99]
        
        # Selected signal channels to analyze
        selected_channels = ['Bs_phiphi', 'Bp_KpJpsi', 'B0_Kpi', 'B0_D0pipi']
        
        # Define PT bins from 0 to 20 GeV
        pt_bins = np.linspace(0, 20, 21)  # 1 GeV bins
        
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
        
        # Calculate efficiency for each channel, cut value, and PT bin
        for channel in selected_channels:
            channel_mask = (channels == channel)
            
            # Skip if no events in this channel
            if np.sum(channel_mask) == 0:
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
                        'TotalEvents': total_events,
                        'PassingEvents': passing_cut,
                        'MinbiasRejection': minbias_rejection[cut]
                    })
        
        # Create DataFrame for visualization
        df = pd.DataFrame(efficiency_data)
        
        # Return early if no data is available
        if len(df) == 0:
            return alt.Chart(pd.DataFrame({'message': ['No data available for selected channels']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)
        
        # Create separate charts for each cut value
        cut_charts = []
        
        for cut in cut_values:
            # Filter data for this cut value
            cut_df = df[df['Cut'] == str(cut)]
            
            if len(cut_df) == 0:
                continue
                
            # Create base chart encoding
            base = alt.Chart(cut_df).encode(
                x=alt.X('PT_Bin:Q', title='TwoBody_PT [GeV]', scale=alt.Scale(domain=[0, 20])),
                color=alt.Color('Channel:N', scale=alt.Scale(scheme='category10')),
                tooltip=['Channel:N', 'PT_Bin:Q', 'Efficiency:Q', 'Error:Q', 
                        'TotalEvents:Q', 'PassingEvents:Q']
            )
            
            # Line chart for efficiency
            line = base.mark_line().encode(
                y=alt.Y('Efficiency:Q', title='Efficiency', scale=alt.Scale(domain=[0, 1])),
            )
            
            # Add error bars
            error_bars = base.mark_errorbar().encode(
                y=alt.Y('Efficiency_lower:Q', title=''),
                y2=alt.Y2('Efficiency_upper:Q')
            ).transform_calculate(
                Efficiency_lower="max(0, datum.Efficiency - datum.Error)",
                Efficiency_upper="min(1, datum.Efficiency + datum.Error)"
            )
            
            # Add minbias rejection text
            rejection_text = alt.Chart(pd.DataFrame([{
                'x': 17,  # Position in the right part of the chart
                'y': 0.1,
                'text': f"Minbias rejection: {minbias_rejection[cut]:.4f}"
            }])).mark_text(align='right', fontSize=12).encode(
                x='x:Q',
                y='y:Q',
                text='text:N'
            )
            
            # Combine line chart with error bars and text
            chart = (line + error_bars + rejection_text).properties(
                width=400,
                height=250,
                title=f'Channel Efficiency vs PT (Cut = {cut})'
            )
            
            cut_charts.append(chart)
        
        # Combine all cut charts into a single view
        if cut_charts:
            return alt.vconcat(
                alt.hconcat(cut_charts[0], cut_charts[1]).resolve_scale(color='shared') 
                if len(cut_charts) > 1 else cut_charts[0],
                alt.hconcat(cut_charts[2], cut_charts[3]).resolve_scale(color='shared')
                if len(cut_charts) > 3 else (cut_charts[2] if len(cut_charts) > 2 else alt.Chart()),
            ).resolve_scale(color='shared').properties(
                title='Signal Channel Efficiency vs PT with Binomial Errors'
            )
        else:
            return alt.Chart(pd.DataFrame({'message': ['No efficiency data available']})).mark_text().encode(
                text='message:N'
            ).properties(width=800, height=200)