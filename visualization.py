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

# Set Altair rendering options
alt.data_transformers.enable('default', max_rows=None)

class ModelPerformance:
    """Model visualization with ROC, PR curves and NN response distributions"""

    def __init__(self, model: nn.Module, feature_names: list, device: torch.device):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def create_performance_dashboard(self, X_test: np.ndarray, y_test: np.ndarray, 
                                    history: dict, epoch: int) -> None:
        """Create and log performance metrics dashboard"""
        # Get predictions
        y_test_pred = self._get_predictions(X_test)
        
        # Create charts
        learning_chart = self._create_learning_curves(history)
        metrics_chart = self._create_metric_evolution(history)
        feature_importance = self._create_feature_importance(X_test, y_test)
        confusion_matrix = self._create_confusion_matrix(y_test, y_test_pred)
        
        # Add ROC and PR curves
        roc_curve_chart = self._create_roc_curve(y_test, y_test_pred)
        pr_curve_chart = self._create_pr_curve(y_test, y_test_pred)
        
        # Add NN response distribution
        response_distribution = self._create_response_distribution(y_test, y_test_pred)
        
        # Combine charts into dashboard (4 rows x 2 columns)
        dashboard = alt.vconcat(
            alt.hconcat(learning_chart, metrics_chart).resolve_scale(color='independent'),
            alt.hconcat(feature_importance, confusion_matrix).resolve_scale(color='independent'),
            alt.hconcat(roc_curve_chart, pr_curve_chart).resolve_scale(color='independent'),
            alt.hconcat(response_distribution).resolve_scale(color='independent')
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