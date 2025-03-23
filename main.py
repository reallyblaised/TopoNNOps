import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import mlflow
from models import UnconstrainedNet, LipschitzNet
from trainer import Trainer
from data import LHCbMCModule
import logging
from typing import Union, List
from loss import FocalLoss, CombinedFocalBCELoss, WeightedBCELoss


def get_model(cfg: DictConfig, input_dim: int, feature_names: List[str]) -> nn.Module:
    """Create model based on configuration"""
    # Check the selected architecture type
    architecture = cfg.model.identifier.lower()
    
    # For unconstrained models
    if architecture == "unconstrained":
        model_params = dict(cfg.model)
        model_params.pop('identifier', None)  # Removes 'identifier' if it exists as not strictly architectural
        return UnconstrainedNet(
            input_dim=input_dim,
            **model_params
        )
            
    # For Lipschitz-constrained models (with or without monotonicity)
    elif architecture in ["lipschitz", "lipschitz_monotonic"]:
        # Get model configuration
        lip_const = cfg.model.get("lip_const", 1.0)
        nbody = cfg.model.get("nbody", "TwoBody")
        
        # Determine if we use monotonicity constraints - can come from either:
        # 1. The architecture name (lipschitz_monotonic)
        # 2. The model config (monotonic: true)
        monotonic = (architecture == "lipschitz_monotonic" or 
                     cfg.model.get("monotonic", False))
        
        # Handle Lipschitz constraint type
        lip_kind = cfg.model.get("lip_kind", "default")
        
        return LipschitzNet(
            input_dim=input_dim,
            layer_dims=cfg.model.layer_dims,
            lip_const=lip_const,
            monotonic=monotonic,
            nbody=nbody,
            feature_names=feature_names,
            features_config_path=cfg.features_config_path,
            activation_fn=cfg.model.get("activation_fn", "groupsort"),
            dropout_rate=cfg.model.get("dropout_rate", 0.0),
            batch_norm=cfg.model.get("batch_norm", False),
            l1_factor=cfg.model.get("l1_factor", 0.0),
            lip_kind=lip_kind
        )
    
    # Unknown architecture
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")


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
        return FocalLoss(
            alpha=cfg.training.get("focal_alpha", 1.0),
            gamma=cfg.training.get("focal_gamma", 2.0),
        )
    elif loss_fn == "weighted_bce":
        return nn.BCEWithLogitsLoss()
    elif loss_fn == "combined_focal_wbce":
        return CombinedFocalBCELoss(
            alpha=cfg.training.get("focal_alpha", 1.0),
            gamma=cfg.training.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

def log_model_with_metadata(model, data_module, cfg):
    """Log the model state dict to MLflow with detailed feature and class information"""
    try:
        # Extract feature names and constraints from data module
        feature_names = data_module.feature_cols
        
        # Create metadata dictionary
        metadata = {
            # Feature information
            "features": feature_names,
            "feature_count": len(feature_names),
            
            # Class label definitions
            "class_labels": {
                "0": "No monotonicity requirement",
                "1": "Monotonically increasing (at the partials)",
            },
            
            # Feature constraints (if available)
            "feature_constraints": data_module.feature_config(
                model=cfg.get("trigger", "TwoBody"),
                feature_config_file=cfg.get("features_config_path", "features.yaml")
            ),
            
            # Model architecture summary
            "architecture": cfg.model.identifier,
            "layer_dimensions": cfg.model.layer_dims,
            "activation_function": cfg.model.get("activation_fn", "relu"),

            # Lipschitz constraint (if available)
            "lip_const": cfg.model.get("lip_const", None),

            # Training information
            "training_scale_factor": cfg.training.get("training_data_scale_factor", 1.0),
            "signal_background_ratio": cfg.training.get("sb_ratio", 0.1),
            "loss_function": cfg.training.get("loss_fn", "bce_with_logits")
        }
        
        # Log model state dict instead of the full model
        mlflow.log_dict(model.state_dict(), "model_state_dict.pth")
        
        # Log metadata as a separate artifact
        mlflow.log_dict(metadata, "model_metadata.json")
        
        # Other logging code remains the same...
        
        return True
    except Exception as e:
        logging.error(f"Error logging model with metadata: {str(e)}")
        return False

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger = logging.getLogger(__name__)
  
    # MLflow experiment setup
    try:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(cfg.mlflow.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        logger.info(f"Using MLflow experiment: {cfg.mlflow.experiment_name} (ID: {experiment_id})")
    except Exception as e:
        logger.warning(f"MLflow setup failed: {str(e)}. Training will continue without tracking.")

    with mlflow.start_run():
        # Log key configuration parameters
        logger.info(f"Running with architecture: {cfg.model.identifier}")
        logger.info(f"Using model configuration: {cfg.model}")
        logger.info(f"Training configuration: {cfg.training}")

        # Log model architecture to MLflow for dashboarding
        mlflow.log_param("architecture", cfg.model.identifier)
        
        # log the lipschitz normalisations scheme 
        if hasattr(cfg.model, "lip_const"):
            mlflow.log_param("lip_const", cfg.model.lip_const)

        # Log all configurations
        for section in ["model", "optimizer", "training"]:
            if hasattr(cfg, section):
                for key, value in dict(getattr(cfg, section)).items():
                    # Skip complex nested structures
                    if not isinstance(value, (dict, list)) or isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"{section}.{key}", value)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        try:
            # Setup data
            data_module = LHCbMCModule(cfg.paths.train_data, cfg.paths.test_data)
            data_module.setup(
                batch_size=cfg.training.batch_size,
                scale_factor=cfg.training.get("training_data_scale_factor", 1.0),
                ratio=cfg.training.get("sb_ratio", 0.01),
            )
            
            logger.info(f"Input features: {data_module.feature_cols}")
            logger.info(f"Input dimension: {data_module.input_dim}")

            # Create model
            model = get_model(cfg, data_module.input_dim, data_module.feature_cols)
            model = model.to(device)
            logger.info(f"Model created: {model}")

            # granular survery of arch in case of LipschitzNet
            if hasattr(model, 'print_architecture_details'):
                model.print_architecture_details()  

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
            
            # Provide access to the data module for visualization
            trainer.data_module = data_module

            # Train the model
            history = trainer.train(
                train_loader=data_module.train_loader,
                val_loader=data_module.test_loader,
                num_epochs=cfg.training.num_epochs,
                metrics_cfg=cfg.metrics,
                early_stopping_patience=cfg.training.get("early_stopping_patience", None),
            )

            # Log final metrics
            final_metrics = history["eval_metrics"][-1]
            for metric_name, metric_value in final_metrics.items():
                mlflow.log_metric(f"final_{metric_name}", metric_value)

            # Log features and their constraints as individual parameters
            feature_constraints = data_module.feature_config(
                model=cfg.get("trigger", "TwoBody"),
                feature_config_file=cfg.get("features_config_path", "features.yaml")
            )
            for feature_name, constraint_value in feature_constraints.items():
                constraint_type = "monotonic_increasing" if constraint_value == 1 else "no_monotonicity"
                mlflow.log_param(f"feature.{feature_name}", constraint_type)  

            # Log class definitions as parameters
            mlflow.log_param("class.0", "Background (minbias)")
            mlflow.log_param("class.1", "Signal (beauty mesons)")
            mlflow.log_param("constraint.0", "No monotonicity requirement")
            mlflow.log_param("constraint.1", "Monotonically increasing (at the partials)")

            # Log model with detailed metadata
            log_model_with_metadata(model, data_module, cfg)

            logger.info("Training completed successfully!")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise e


if __name__ == "__main__":
    main()