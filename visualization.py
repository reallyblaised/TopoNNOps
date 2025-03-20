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
from typing import Optional, List

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
        
        # Validate dimensions of our data
        print(f"Visualization data shapes - X: {X_test.shape}, y: {y_test.shape}, preds: {y_test_pred.shape}")
        if channels is not None:
            print(f"Channels array shape: {channels.shape}")
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
        
        # Create standard dashboard components
        dashboard_components = [
            alt.hconcat(learning_chart, metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(feature_importance, confusion_matrix).resolve_scale(color='independent'),
            alt.hconcat(roc_curve_chart, pr_curve_chart).resolve_scale(color='independent'),
            alt.hconcat(response_distribution).resolve_scale(color='independent')
        ]
        
        # Add efficiency vs PT histogram if channel information is available
        if channels is not None:
            try:
                # Look for TwoBody_PT in feature names
                if 'TwoBody_PT' in self.feature_names:
                    pt_index = self.feature_names.index('TwoBody_PT')
                    pt_values = X_test[:, pt_index]
                    efficiency_chart = self._create_channel_efficiency_vs_pt(
                        X_test, y_test, channels, y_test_pred, pt_values, pt_index
                    )
                    dashboard_components.append(efficiency_chart)
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

    # [Other visualization methods remain the same...]

    def _create_channel_efficiency_vs_pt(self, 
                                        X: np.ndarray, 
                                        y: np.ndarray, 
                                        channels: np.ndarray,
                                        y_pred: np.ndarray = None,
                                        pt_values: np.ndarray = None, 
                                        pt_index: int = None) -> alt.Chart:
        """
        Create efficiency histograms in TwoBody_PT for specific signal channels.
        
        Shows efficiency at different cut values (0.75, 0.9, 0.95, 0.99) with binomial errors.
        Also displays minbias rejection rate for each cut value.
        
        Args:
            X: Feature matrix
            y: Target labels
            channels: Array of channel names for each sample
            y_pred: Predicted probabilities (if None, will be computed)
            pt_values: PT values extracted from X (if None, will be extracted using pt_index)
            pt_index: Index of PT feature in X (if None, will be determined from feature_names)
        """
        # Print diagnostic info
        print(f"Creating channel efficiency chart with {len(channels)} channel entries")
        print(f"Unique channels: {np.unique(channels)}")
        
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
        
        # Define cut values
        cut_values = [0.75, 0.9, 0.95, 0.99]
        
        # Selected signal channels to analyze
        selected_channels = ['Bs_phiphi', 'Bp_KpJpsi', 'B0_Kpi', 'B0_D0pipi']
        
        # Define PT bins from 0 to 20 GeV
        pt_bins = np.linspace(0, 20, 21)  # 1 GeV bins
        
        # Compute minbias rejection for each cut value
        minbias_mask = (channels == 'minbias')
        minbias_total = np.sum(minbias_mask)
        print(f"Found {minbias_total} minbias events")
        
        minbias_rejection = {}
        
        # Ensure prediction array is flattened to match dimension of mask
        flattened_preds = y_pred.ravel()
        
        for cut in cut_values:
            minbias_rejected = np.sum(minbias_mask & (flattened_preds < cut))
            rejection_rate = minbias_rejected / minbias_total if minbias_total > 0 else 0
            minbias_rejection[cut] = rejection_rate
            print(f"Cut {cut}: Rejected {minbias_rejected}/{minbias_total} minbias events ({rejection_rate:.4f})")
        
        # Data structure to store efficiency results
        efficiency_data = []
        
        # Calculate efficiency for each channel, cut value, and PT bin
        for channel in selected_channels:
            channel_mask = (channels == channel)
            channel_total = np.sum(channel_mask)
            print(f"Found {channel_total} events for channel {channel}")
            
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