import numpy as np
import pandas as pd
import torch
import altair as alt
import mlflow
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score, 
    precision_recall_fscore_support, f1_score, accuracy_score
)
from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import json
import os
import tempfile


# Set Altair rendering options for better visuals
alt.data_transformers.enable('default', max_rows=None)


class ModelPerformance:
    """Comprehensive model interpretation and visualization for performance metrics using Altair"""

    def __init__(
        self, model: nn.Module, feature_names: List[str], device: torch.device
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def create_performance_dashboard(
        self, X_train: np.ndarray, y_train: np.ndarray, 
        X_test: np.ndarray, y_test: np.ndarray, 
        history: Dict[str, list], epoch: int
    ) -> None:
        """Create and log comprehensive interpretability dashboard with train vs test overlays"""
        # Get predictions for all visualizations
        y_train_pred = self._get_predictions(X_train)
        y_test_pred = self._get_predictions(X_test)
        
        # Create charts
        learning_chart = self._create_learning_curves(history)
        metrics_chart = self._create_metric_evolution(history)
        roc_chart = self._create_roc_curve(y_test, y_test_pred)
        pr_chart = self._create_pr_curve(y_test, y_test_pred)
        distribution_chart = self._create_prediction_distribution(y_train_pred, y_test_pred, y_train, y_test)
        threshold_metrics_chart = self._create_threshold_metrics(y_test, y_test_pred)
        response_curves_chart = self._create_response_curves(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)
        
        # Combine charts into dashboard (3 rows x 2 columns)
        dashboard = alt.vconcat(
            alt.hconcat(learning_chart, metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(roc_chart, pr_chart).resolve_scale(color='independent'),
            alt.hconcat(distribution_chart, threshold_metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(response_curves_chart).resolve_scale(color='independent')
        ).properties(
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

    def _create_learning_curves(self, history: Dict[str, list]) -> alt.Chart:
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

    def _create_metric_evolution(self, history: Dict[str, list]) -> alt.Chart:
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

    def _create_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate ROC curve visualization"""
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred.ravel())
        roc_auc = auc(fpr, tpr)
        
        # Create DataFrame - ensure all arrays have the same length
        # The ROC curve returns one more point in fpr, tpr than there are thresholds
        df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr,
            'Threshold': np.append(thresholds, [thresholds[-1]])  # Duplicate the last threshold instead of using 0
        })
        
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

    def _create_pr_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate Precision-Recall curve visualization"""
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred.ravel())
        avg_precision = average_precision_score(y_true, y_pred.ravel())
        
        # Create DataFrame - ensure lengths match
        # precision_recall_curve returns one more point in precision and recall than there are thresholds
        df = pd.DataFrame({
            'Recall': recall,
            'Precision': precision,
            'Threshold': np.append(thresholds, [1.0])  # Add threshold=1.0 for last point
        })
        
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

    def _create_prediction_distribution(self, y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                                      y_train: np.ndarray, y_test: np.ndarray) -> alt.Chart:
        """Generate prediction distribution visualization"""
        # Create DataFrames for signal and background in train and test sets
        train_sig = pd.DataFrame({
            'Score': y_train_pred.ravel()[y_train.ravel() == 1],
            'Type': 'Signal',
            'Set': 'Train'
        })
        
        train_bkg = pd.DataFrame({
            'Score': y_train_pred.ravel()[y_train.ravel() == 0],
            'Type': 'Background',
            'Set': 'Train'
        })
        
        test_sig = pd.DataFrame({
            'Score': y_test_pred.ravel()[y_test.ravel() == 1],
            'Type': 'Signal',
            'Set': 'Test'
        })
        
        test_bkg = pd.DataFrame({
            'Score': y_test_pred.ravel()[y_test.ravel() == 0],
            'Type': 'Background',
            'Set': 'Test'
        })
        
        # Combine all data
        df = pd.concat([train_sig, train_bkg, test_sig, test_bkg])
        
        # Create histogram
        chart = alt.Chart(df).mark_area(
            opacity=0.6,
            interpolate='step'
        ).encode(
            alt.X('Score:Q', bin=alt.Bin(maxbins=40), title='Prediction Score'),
            alt.Y('count():Q', stack=None, title='Count (Normalized)'),
            alt.Color('Type:N', scale=alt.Scale(
                domain=['Signal', 'Background'],
                range=['#2ca02c', '#d62728']
            )),
            alt.Strokewidth('Set:N', scale=alt.Scale(
                domain=['Train', 'Test'],
                range=[1, 3]
            )),
            alt.Stroke('Set:N'),
            alt.OpacityValue(0.7),
            tooltip=['Type:N', 'Set:N', 'count()']
        ).properties(
            width=400,
            height=300,
            title='Prediction Distribution (Signal vs Background)'
        ).interactive()
        
        return chart

    def _create_threshold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate metrics vs threshold visualization"""
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0.01, 0.99, 30)
        data = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred.ravel() >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_binary, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred_binary)
            
            data.append({
                'Threshold': threshold,
                'Metric': 'F1 Score',
                'Value': f1
            })
            data.append({
                'Threshold': threshold,
                'Metric': 'Precision',
                'Value': precision
            })
            data.append({
                'Threshold': threshold,
                'Metric': 'Recall',
                'Value': recall
            })
            data.append({
                'Threshold': threshold,
                'Metric': 'Accuracy',
                'Value': accuracy
            })
        
        # Calculate ROC AUC (threshold-independent)
        fpr, tpr, _ = roc_curve(y_true, y_pred.ravel())
        roc_auc = auc(fpr, tpr)
        
        for threshold in thresholds:
            data.append({
                'Threshold': threshold,
                'Metric': 'ROC AUC',
                'Value': roc_auc
            })
        
        df = pd.DataFrame(data)
        
        # Create metrics vs threshold visualization
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('Threshold:Q', title='Decision Threshold'),
            y=alt.Y('Value:Q', title='Metric Value'),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Threshold:Q', 'Value:Q', 'Metric:N']
        ).properties(
            width=400,
            height=300,
            title='Performance Metrics vs Threshold'
        ).interactive()
        
        # Find optimal F1 threshold
        f1_df = df[df['Metric'] == 'F1 Score']
        best_f1_idx = f1_df['Value'].idxmax()
        best_threshold = f1_df.iloc[best_f1_idx]['Threshold']
        best_f1 = f1_df.iloc[best_f1_idx]['Value']
        
        # Add marker for optimal F1 score
        optimal_point = alt.Chart(pd.DataFrame({
            'Threshold': [best_threshold],
            'Value': [best_f1],
            'Metric': ['F1 Score']
        })).mark_point(
            size=100,
            shape='circle',
            filled=True,
            color='black'
        ).encode(
            x='Threshold:Q',
            y='Value:Q',
            tooltip=['Threshold:Q', 'Value:Q', alt.value(f'Optimal F1 = {best_f1:.3f}')]
        )
        
        return (chart + optimal_point)

    def _create_response_curves(self, X_train: np.ndarray, y_train: np.ndarray, y_train_pred: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray) -> alt.Chart:
        """Generate response curves showing signal vs background for train and test sets"""
        # Get top features (using a simple correlation method for demonstration)
        feature_importances = []
        for i in range(X_train.shape[1]):
            corr = np.abs(np.corrcoef(X_train[:, i], y_train.ravel())[0, 1])
            feature_importances.append((i, corr))
        
        # Sort by importance
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 3 most important features
        top_features = feature_importances[:3]
        
        # Prepare data for response curves
        data = []
        for feature_idx, importance in top_features:
            feature_name = self.feature_names[feature_idx]
            
            # Training data
            for is_signal in [0, 1]:
                mask = y_train.ravel() == is_signal
                x_values = X_train[mask, feature_idx]
                y_values = y_train_pred.ravel()[mask]
                
                # Create bins for smoother visualization
                bins = np.linspace(np.min(x_values), np.max(x_values), 20)
                digitized = np.digitize(x_values, bins)
                
                for bin_idx in range(1, len(bins)):
                    bin_mask = digitized == bin_idx
                    if np.sum(bin_mask) > 0:
                        bin_x = np.mean(x_values[bin_mask])
                        bin_y = np.mean(y_values[bin_mask])
                        data.append({
                            'Feature': feature_name,
                            'Feature Value': bin_x,
                            'Response': bin_y,
                            'Type': 'Signal' if is_signal else 'Background',
                            'Set': 'Train'
                        })
            
            # Test data
            for is_signal in [0, 1]:
                mask = y_test.ravel() == is_signal
                x_values = X_test[mask, feature_idx]
                y_values = y_test_pred.ravel()[mask]
                
                # Create bins for smoother visualization
                bins = np.linspace(np.min(x_values), np.max(x_values), 20)
                digitized = np.digitize(x_values, bins)
                
                for bin_idx in range(1, len(bins)):
                    bin_mask = digitized == bin_idx
                    if np.sum(bin_mask) > 0:
                        bin_x = np.mean(x_values[bin_mask])
                        bin_y = np.mean(y_values[bin_mask])
                        data.append({
                            'Feature': feature_name,
                            'Feature Value': bin_x,
                            'Response': bin_y,
                            'Type': 'Signal' if is_signal else 'Background',
                            'Set': 'Test'
                        })
        
        df = pd.DataFrame(data)
        
        # Create the response curves chart
        selection = alt.selection_single(
            fields=['Feature'],
            init={'Feature': self.feature_names[top_features[0][0]]},
            bind=alt.binding_select(
                options=[self.feature_names[idx] for idx, _ in top_features],
                name='Feature: '
            )
        )
        
        chart = alt.Chart(df).mark_line(size=3).encode(
            x=alt.X('Feature Value:Q', title='Feature Value'),
            y=alt.Y('Response:Q', title='Model Response', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Signal', 'Background'],
                range=['#2ca02c', '#d62728']
            )),
            strokeDash=alt.StrokeDash('Set:N', scale=alt.Scale(
                domain=['Train', 'Test'],
                range=[[1, 0], [5, 5]]
            )),
            tooltip=['Feature:N', 'Feature Value:Q', 'Response:Q', 'Type:N', 'Set:N']
        ).properties(
            width=800,
            height=300,
            title='Model Response Curves (Signal vs Background)'
        ).add_selection(
            selection
        ).transform_filter(
            selection
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
            
            # Also save a JSON version for programmatic access
            json_spec = json.dumps(dashboard.to_dict())
            mlflow.log_text(json_spec, f"visualization/performance_dashboard_epoch_{epoch}.json")
            
        except Exception as e:
            print(f"Failed to log dashboard: {str(e)}")