import numpy as np
import pandas as pd
import torch
import altair as alt
from sklearn.inspection import permutation_importance
import mlflow
import json
from typing import List, Dict, Optional
import torch.nn as nn
import shap
from sklearn.metrics import roc_curve, auc


class ModelInterpretability:
    """Comprehensive model interpretation and visualization"""

    def __init__(
        self, model: nn.Module, feature_names: List[str], device: torch.device
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def create_interpretability_dashboard(
        self, X: np.ndarray, y: np.ndarray, history: Dict[str, list], epoch: int
    ) -> None:
        """Create and log comprehensive interpretability dashboard"""
        # Generate all visualizations
        feature_importance = self._get_feature_importance(X, y)
        shap_summary = self._get_shap_summary(X) # NOTE: not essential
        learning_curves = self._get_learning_curves(history)
        metric_evolution = self._get_metric_evolution(history)
        roc_curve = self._get_roc_curve(y, self._get_predictions(X))
        pred_dist = self._get_prediction_distribution(X)

        # Combine charts into a dashboard layout
        dashboard = alt.vconcat(
            alt.hconcat(feature_importance, shap_summary),
            alt.hconcat(learning_curves, metric_evolution),
            alt.hconcat(roc_curve, pred_dist),
        ).resolve_scale(color="independent")

        # Log to MLflow
        self._log_dashboard(dashboard, epoch)

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return torch.sigmoid(outputs).cpu().numpy()

    def _get_feature_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 10
    ) -> alt.Chart:
        """Calculate and visualize permutation feature importance"""
        # Calculate permutation importance
        perm_importance = permutation_importance(
            lambda x: self._get_predictions(x).ravel(),
            X,
            y,
            n_repeats=n_repeats,
            random_state=42,
        )

        # Create DataFrame for visualization
        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": perm_importance.importances_mean,
                "Std": perm_importance.importances_std,
            }
        )

        # Create feature importance chart
        chart = (
            alt.Chart(importance_df)
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Feature Importance"),
                y=alt.Y("Feature:N", sort="-x", title="Feature Name"),
                tooltip=["Feature", "Importance", "Std"],
            )
            .properties(width=400, height=300, title="Feature Importance (Permutation)")
        )

        # Add error bars
        error_bars = (
            alt.Chart(importance_df)
            .mark_errorbar()
            .encode(
                x=alt.X("min:Q", title="Feature Importance"),
                x2="max:Q",
                y=alt.Y("Feature:N", sort="-x"),
            )
            .transform_calculate(
                min="datum.Importance - datum.Std", max="datum.Importance + datum.Std"
            )
        )

        return (chart + error_bars).interactive()

    def _get_shap_summary(self, X: np.ndarray, n_samples: int = 100) -> alt.Chart:
        """Generate SHAP summary plot"""
        # Create background dataset
        background = X[np.random.choice(X.shape[0], n_samples, replace=False)]
        background_tensor = torch.FloatTensor(background).to(self.device)

        # Calculate SHAP values
        explainer = shap.DeepExplainer(self.model, background_tensor)
        X_tensor = torch.FloatTensor(X[:n_samples]).to(self.device)
        shap_values = explainer.shap_values(X_tensor)

        # Create DataFrame for visualization
        shap_df = pd.DataFrame(shap_values[0], columns=self.feature_names).melt()
        shap_df["abs_value"] = abs(shap_df["value"])

        # Create SHAP summary chart
        chart = (
            alt.Chart(shap_df)
            .mark_circle()
            .encode(
                x=alt.X("value:Q", title="SHAP value"),
                y=alt.Y(
                    "variable:N",
                    title="Feature",
                    sort=alt.EncodingSortField(
                        field="abs_value", op="mean", order="descending"
                    ),
                ),
                color=alt.Color(
                    "value:Q", scale=alt.Scale(scheme="redblue"), title="Impact"
                ),
                size=alt.Size("abs_value:Q", title="|SHAP value|"),
                tooltip=["variable", "value", "abs_value"],
            )
            .properties(width=400, height=300, title="SHAP Summary Plot")
            .interactive()
        )

        return chart

    def _get_learning_curves(self, history: Dict[str, list]) -> alt.Chart:
        """Generate interactive learning curves"""
        # Create DataFrame
        epochs = range(len(history["train_loss"]))
        df = pd.DataFrame(
            {
                "Epoch": list(epochs) * 2,
                "Loss": history["train_loss"]
                + [m["loss"] for m in history["eval_metrics"]],
                "Type": ["Train"] * len(epochs) + ["Validation"] * len(epochs),
            }
        )

        # Create chart
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x="Epoch:Q",
                y="Loss:Q",
                color="Type:N",
                tooltip=["Epoch", "Loss", "Type"],
            )
            .properties(width=400, height=300, title="Learning Curves")
            .interactive()
        )

        return chart

    def _get_metric_evolution(self, history: Dict[str, list]) -> alt.Chart:
        """Generate metric evolution plots"""
        # Extract metrics
        metrics = list(history["eval_metrics"][0].keys())
        data = []

        for epoch, metrics_dict in enumerate(history["eval_metrics"]):
            for metric, value in metrics_dict.items():
                if metric != "loss":  # Loss is shown in learning curves
                    data.append({"Epoch": epoch, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        # Create chart
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x="Epoch:Q",
                y="Value:Q",
                color="Metric:N",
                tooltip=["Epoch", "Metric", "Value"],
            )
            .properties(width=400, height=300, title="Metric Evolution")
            .interactive()
        )

        return chart

    def _get_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        """Generate ROC curve visualization"""
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Create DataFrame
        df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

        # Create chart
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("False Positive Rate", title="False Positive Rate"),
                y=alt.Y("True Positive Rate", title="True Positive Rate"),
            )
            .properties(width=400, height=300, title=f"ROC Curve (AUC = {roc_auc:.3f})")
        )

        # Add diagonal reference line
        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[4, 4], color="gray")
            .encode(x="x", y="y")
        )

        return (chart + diagonal).interactive()

    # def _get_feature_importance(
    #     self, X: np.ndarray, y: np.ndarray, n_repeats: int = 10
    # ) -> alt.Chart:
    #     """Calculate and visualize permutation feature importance"""

    #     # Wrapper class to make PyTorch model compatible with sklearn
    #     class ModelWrapper:
    #         def __init__(self, model, device):
    #             self.model = model
    #             self.device = device
    #             self._estimator_type = "classifier"  # Explicitly mark as classifier

    #         def fit(self, X, y):
    #             return self

    #         def predict_proba(self, X):
    #             X_tensor = torch.FloatTensor(X).to(self.device)
    #             self.model.eval()
    #             with torch.no_grad():
    #                 outputs = self.model(X_tensor)
    #                 probs = torch.sigmoid(outputs).cpu().numpy()
    #             return np.column_stack([1 - probs, probs])

    #         def decision_function(self, X):
    #             # Return raw logits for decision function
    #             X_tensor = torch.FloatTensor(X).to(self.device)
    #             self.model.eval()
    #             with torch.no_grad():
    #                 outputs = self.model(X_tensor)
    #                 return outputs.cpu().numpy().ravel()

    #     # Create wrapper instance
    #     wrapped_model = ModelWrapper(self.model, self.device)

    #     # Calculate permutation importance
    #     perm_importance = permutation_importance(
    #         wrapped_model,
    #         X,
    #         y,
    #         scoring="roc_auc",  # Using ROC-AUC as the metric
    #         n_repeats=n_repeats,
    #         n_jobs=-1,
    #         random_state=42,
    #     )

    #     # Create visualization DataFrame
    #     importance_df = pd.DataFrame(
    #         {
    #             "Feature": self.feature_names,
    #             "Importance": perm_importance.importances_mean,
    #             "Std": perm_importance.importances_std,
    #         }
    #     )

    #     # Sort by importance
    #     importance_df = importance_df.sort_values("Importance", ascending=False)

    #     # Create feature importance chart
    #     chart = (
    #         alt.Chart(importance_df)
    #         .mark_bar()
    #         .encode(
    #             x=alt.X("Importance:Q", title="Feature Importance"),
    #             y=alt.Y("Feature:N", sort="-x", title="Feature Name"),
    #             tooltip=["Feature", "Importance", "Std"],
    #         )
    #         .properties(width=400, height=300, title="Feature Importance (Permutation)")
    #     )

    #     # Add error bars
    #     error_bars = (
    #         alt.Chart(importance_df)
    #         .mark_errorbar()
    #         .encode(
    #             x=alt.X("min:Q", title="Feature Importance"),
    #             x2="max:Q",
    #             y=alt.Y("Feature:N", sort="-x"),
    #         )
    #         .transform_calculate(
    #             min="datum.Importance - datum.Std", max="datum.Importance + datum.Std"
    #         )
    #     )

    #     return (chart + error_bars).interactive()
