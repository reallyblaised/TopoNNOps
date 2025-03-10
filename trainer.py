import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import mlflow
from typing import Dict, List, Optional, Tuple
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import time
from visualization import ModelPerformance as ModelInterpretability
from weight_visualization import WeightVisualizer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        feature_names: List[str],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip_val: Optional[float] = None,
        use_mixed_precision: bool = False,
    ):
        """Initialize trainer with model and training components"""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_val = grad_clip_val
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Initialize visualization tools
        self.interpretability = ModelInterpretability(
            model=model, feature_names=feature_names, device=device
        )
        self.weight_viz = WeightVisualizer(model=model, feature_names=feature_names)

    def _train_step(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Single training step with mixed precision support"""
        X, y = X.to(self.device), y.to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Forward pass with optional mixed precision
        with autocast(enabled=self.use_mixed_precision):
            outputs = self.model(X)
            loss = self.criterion(outputs, y.unsqueeze(1))

            # Add L1 regularization if applicable
            if hasattr(self.model, "get_l1_loss"):
                loss += self.model.get_l1_loss()

        # Backward pass with gradient scaling
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            if self.grad_clip_val:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_val
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_val:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_val
                )
            self.optimizer.step()

        return loss.item(), outputs

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        metrics_cfg: DictConfig,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, list]:
        """Complete training loop with comprehensive logging"""
        history = {
            "train_loss": [],
            "eval_metrics": [],
            "learning_rates": [],
            "epoch_times": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        # Get sample batch for visualizations
        X_viz, y_viz = next(iter(val_loader))
        X_viz = X_viz.to(self.device)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            train_losses = []
            all_train_outputs = []
            all_train_targets = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for X, y in progress_bar:
                batch_loss, outputs = self._train_step(X, y)
                train_losses.append(batch_loss)
                all_train_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_train_targets.extend(y.cpu().numpy())

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            eval_metrics = self.evaluate(val_loader, metrics_cfg, epoch)
            epoch_time = time.time() - epoch_start_time

            # Store metrics
            history["train_loss"].append(train_loss)
            history["eval_metrics"].append(eval_metrics)
            history["epoch_times"].append(epoch_time)
            if self.scheduler:
                history["learning_rates"].append(self.scheduler.get_last_lr()[0])

            # Log metrics and visualizations
            self._log_epoch_info(
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                epoch=epoch,
                epoch_time=epoch_time,
                train_outputs=np.array(all_train_outputs),
                train_targets=np.array(all_train_targets),
            )

            # Generate visualizations periodically
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self._generate_visualizations(X_viz, y_viz, history, epoch)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Early stopping check
            if early_stopping_patience:
                if eval_metrics["loss"] < best_val_loss:
                    best_val_loss = eval_metrics["loss"]
                    patience_counter = 0
                    self._save_checkpoint(epoch, eval_metrics)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

        return history

    def evaluate(
        self,
        val_loader: DataLoader,
        metrics_cfg: DictConfig,
        epoch: Optional[int] = None,
    ) -> dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        metrics = {}
        all_preds = []
        all_targets = []
        val_losses = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                with autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y.unsqueeze(1))
                    val_losses.append(loss.item())

                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # Calculate metrics
        metrics["loss"] = np.mean(val_losses)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate performance metrics
        if metrics_cfg.compute.roc_auc:
            metrics["roc_auc"] = roc_auc_score(all_targets, all_preds)

        binary_preds = (all_preds > metrics_cfg.threshold).astype(int)

        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            if metrics_cfg.compute.get(metric_name, False):
                metric_fn = globals()[f"{metric_name}_score"]
                metrics[metric_name] = metric_fn(all_targets, binary_preds)

        return metrics

    def _log_epoch_info(
        self,
        train_loss: float,
        eval_metrics: Dict[str, float],
        epoch: int,
        epoch_time: float,
        train_outputs: np.ndarray,
        train_targets: np.ndarray,
    ) -> None:
        """Log comprehensive epoch information to MLflow"""
        # Log basic metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("epoch_time", epoch_time, step=epoch)

        for metric_name, metric_value in eval_metrics.items():
            mlflow.log_metric(f"eval_{metric_name}", metric_value, step=epoch)

        if self.scheduler:
            mlflow.log_metric(
                "learning_rate", self.scheduler.get_last_lr()[0], step=epoch
            )

        # Log training distribution metrics
        mlflow.log_metric("train_pred_mean", train_outputs.mean(), step=epoch)
        mlflow.log_metric("train_pred_std", train_outputs.std(), step=epoch)

    def _generate_visualizations(
        self,
        X_viz: torch.Tensor,
        y_viz: torch.Tensor,
        history: Dict[str, list],
        epoch: int,
    ) -> None:
        """Generate and log visualizations"""
        self.interpretability.create_performance_dashboard(
            X_viz.cpu().numpy(), y_viz.numpy(), history, epoch
        )
        self.weight_viz.create_weight_dashboard(X_viz, epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        mlflow.log_dict(checkpoint, f"checkpoint_epoch_{epoch}.pth")
