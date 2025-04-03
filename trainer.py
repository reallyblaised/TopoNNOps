import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import mlflow
from typing import Dict, List, Optional, Tuple
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import time
from visualization import ModelPerformance
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
        is_distributed: bool = False,
        is_master: bool = True,
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
        self.is_distributed = is_distributed
        self.is_master = is_master

        # Initialize visualization tools (only on master process if distributed)
        self.performance = ModelPerformance(
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
        with torch.autocast(self.device.type, enabled=self.use_mixed_precision):
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

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Set train loader's sampler epoch for proper shuffling in distributed mode
            if (
                self.is_distributed
                and hasattr(train_loader, "sampler")
                and hasattr(train_loader.sampler, "set_epoch")
            ):
                train_loader.sampler.set_epoch(epoch)

            # Training phase
            self.model.train()
            train_losses = []
            all_train_outputs = []
            all_train_targets = []

            # Only show progress bar on master process to avoid output clutter
            if not self.is_distributed or self.is_master:
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
                train_iter = progress_bar
            else:
                train_iter = train_loader

            for X, y in train_iter:
                batch_loss, outputs = self._train_step(X, y)
                train_losses.append(batch_loss)

                # Collect outputs and targets even in distributed mode for the master process
                # This is used for visualizations later
                if not self.is_distributed or self.is_master:
                    all_train_outputs.extend(
                        torch.sigmoid(outputs).detach().cpu().numpy()
                    )
                    all_train_targets.extend(y.cpu().numpy())
                    # Update progress bar if using one
                    if hasattr(locals(), "progress_bar"):
                        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # Calculate epoch metrics
            train_loss = np.mean(train_losses)

            # Synchronize loss across processes in distributed mode
            if self.is_distributed:
                train_loss_tensor = torch.tensor(train_loss).to(self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = (train_loss_tensor / dist.get_world_size()).item()

            # Evaluate on validation set
            eval_metrics = self.evaluate(val_loader, metrics_cfg, epoch)
            epoch_time = time.time() - epoch_start_time

            # Store metrics and do logging (only on master process to avoid conflicts)
            if not self.is_distributed or self.is_master:
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
                    train_outputs=(
                        np.array(all_train_outputs) if all_train_outputs else None
                    ),
                    train_targets=(
                        np.array(all_train_targets) if all_train_targets else None
                    ),
                )

                # Generate visualizations periodically
                if epoch % 20 == 0 or epoch == num_epochs - 1:
                    self._generate_visualizations(history, epoch)

            # Learning rate scheduling - must happen on all processes
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(eval_metrics["loss"])
                else:
                    self.scheduler.step()

            # Early stopping check - must be synchronized across processes
            if early_stopping_patience:
                # Broadcast val_loss from master to all processes to ensure consistency
                if self.is_distributed:
                    val_loss_tensor = torch.tensor(eval_metrics["loss"]).to(self.device)
                    dist.broadcast(val_loss_tensor, src=0)
                    current_val_loss = val_loss_tensor.item()
                else:
                    current_val_loss = eval_metrics["loss"]

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    if not self.is_distributed or self.is_master:
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
        """Comprehensive model evaluation with distributed support"""
        self.model.eval()
        local_metrics = {}
        all_preds = []
        all_targets = []
        val_losses = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                with torch.autocast(self.device.type, enabled=self.use_mixed_precision):
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y.unsqueeze(1))
                    val_losses.append(loss.item())

                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # Calculate metrics
        local_metrics["loss"] = np.mean(val_losses)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate performance metrics
        if metrics_cfg.compute.get("roc_auc", False):
            local_metrics["roc_auc"] = roc_auc_score(all_targets, all_preds)

        binary_preds = (all_preds > metrics_cfg.threshold).astype(int)

        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            if metrics_cfg.compute.get(metric_name, False):
                metric_fn = globals()[f"{metric_name}_score"]
                local_metrics[metric_name] = metric_fn(all_targets, binary_preds)

        # For distributed training, synchronize metrics across processes
        if self.is_distributed:
            # Create a tensor to hold all metrics
            metrics_keys = list(local_metrics.keys())
            metrics_tensor = torch.zeros(
                len(metrics_keys), dtype=torch.float32, device=self.device
            )

            # Fill the tensor with local metric values
            for i, key in enumerate(metrics_keys):
                metrics_tensor[i] = local_metrics[key]

            # All-reduce to get the average across all processes
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= dist.get_world_size()

            # Update metrics with synchronized values
            for i, key in enumerate(metrics_keys):
                local_metrics[key] = metrics_tensor[i].item()

        return local_metrics

    def _log_epoch_info(
        self,
        train_loss: float,
        eval_metrics: Dict[str, float],
        epoch: int,
        epoch_time: float,
        train_outputs: Optional[np.ndarray],
        train_targets: Optional[np.ndarray],
    ) -> None:
        """Log comprehensive epoch information to MLflow (master process only)"""
        if self.is_distributed and not self.is_master:
            return

        # Log basic metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("epoch_time", epoch_time, step=epoch)

        for metric_name, metric_value in eval_metrics.items():
            mlflow.log_metric(f"eval_{metric_name}", metric_value, step=epoch)

        if self.scheduler:
            mlflow.log_metric(
                "learning_rate", self.scheduler.get_last_lr()[0], step=epoch
            )

        # Log training distribution metrics if available
        if train_outputs is not None and train_targets is not None:
            mlflow.log_metric("train_pred_mean", train_outputs.mean(), step=epoch)
            mlflow.log_metric("train_pred_std", train_outputs.std(), step=epoch)

    def _generate_visualizations(
        self,
        history: Dict[str, list],
        epoch: int,
    ) -> None:
        """Generate and log visualizations using direct access to test dataset (master process only)"""
        if self.is_distributed and not self.is_master:
            return

        # Skip if data module is not available
        if not hasattr(self, "data_module"):
            print("Data module not available for visualization")
            return

        # Check if we have direct access to the test dataset tensors
        if hasattr(self.data_module, "X_test") and hasattr(self.data_module, "y_test"):
            # Direct access to dataset tensors
            X_viz = self.data_module.X_test.to(self.device)
            y_viz = self.data_module.y_test

            # Get channels if available
            channels = (
                self.data_module.test_channels
                if hasattr(self.data_module, "test_channels")
                else None
            )

            try:
                processor = (
                    self.data_module.preprocessor
                    if hasattr(self.data_module, "preprocessor")
                    else None
                )

                # Create performance dashboard
                self.performance.create_performance_dashboard(
                    X_viz.cpu().numpy(),
                    y_viz.numpy(),
                    history,
                    epoch,
                    channels=channels,
                    preprocessor=processor,  # pass to invert [0,1]-scaling in visualization
                )

                # Visualize weights if needed - uncomment this for weight visualizations
                # self.weight_viz.create_weight_dashboard(X_viz, epoch)

                # print(f"Visualizations for epoch {epoch} created successfully")
            except Exception as e:
                print(f"Error creating visualizations: {str(e)}")
        else:
            print("Test dataset tensors not available for visualization")
            return

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint (master process only)"""
        if self.is_distributed and not self.is_master:
            return

        # For distributed training, we need to save the unwrapped model
        if self.is_distributed:
            if hasattr(self.model, "module"):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save as PyTorch file instead of JSON
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        mlflow.log_artifact(checkpoint_path)
