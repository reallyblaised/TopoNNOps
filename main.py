import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import mlflow
from models import UnconstrainedNet
from trainer import Trainer
from data import LHCbMCModule
import logging
from typing import Union


def get_model(cfg: DictConfig, input_dim: int) -> nn.Module:
    """Create model based on configuration"""
    if cfg.architecture == "unconstrained":
        return UnconstrainedNet(**dict(cfg.model))
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.architecture}")


def get_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer with optional scheduling"""
    optimizer_name = cfg.optimizer.get("name", "adam").lower()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(
                cfg.optimizer.get("beta1", 0.9),
                cfg.optimizer.get("beta2", 0.999),
            ),  # match defaults in pytorch
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.get("momentum", 0.9),
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    """Create learning rate scheduler if specified"""
    if not cfg.get("scheduler", None):
        return None

    scheduler_name = cfg.scheduler.name.lower()

    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            min_lr=cfg.scheduler.min_lr,
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.num_epochs
        )
    return None


def get_criterion(cfg: DictConfig) -> nn.Module:
    """Get loss function based on configuration"""
    loss_fn = cfg.training.get("loss_fn", "bce_with_logits").lower()

    if loss_fn == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_fn == "focal":
        from loss import FocalLoss

        return FocalLoss(
            alpha=cfg.training.get("focal_alpha", 1.0),
            gamma=cfg.training.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger = logging.getLogger(__name__)

    # MLflow experiment setup
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log all configurations
        mlflow.log_params(dict(cfg.model))
        mlflow.log_params(dict(cfg.optimizer))
        mlflow.log_params(dict(cfg.training))

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        try:
            # Setup data
            data_module = LHCbMCModule(cfg.paths.train_data, cfg.paths.test_data)
            data_module.setup(
                batch_size=cfg.training.batch_size,
                scale_factor=cfg.training.get(
                    "scale_factor", 0.1
                ),  # how much training data is read in
                ratio=cfg.training.get("ratio", 0.01),  # minbias:signal ratio
            )

            # Create model
            model = get_model(cfg, data_module.get_n_features())
            model = model.to(device)

            # Setup training components
            optimizer = get_optimizer(cfg, model)
            scheduler = get_scheduler(cfg, optimizer)
            criterion = get_criterion(cfg)

            # Create trainer and train
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
                feature_names=data_module.feature_cols,
                grad_clip_val=cfg.training.get("grad_clip_val", None),
                use_mixed_precision=cfg.training.get("use_mixed_precision", False),
            )

            # Train the model
            history = trainer.train(
                train_loader=data_module.train_loader,
                val_loader=data_module.test_loader,
                num_epochs=cfg.training.num_epochs,
                metrics_cfg=cfg.metrics,
                early_stopping_patience=cfg.training.get(
                    "early_stopping_patience", None
                ),
            )

            # Log final metrics
            final_metrics = history["eval_metrics"][-1]
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(f"final_{metric_name}", metric_value)

            logger.info("Training completed successfully!")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise e


if __name__ == "__main__":
    main()
