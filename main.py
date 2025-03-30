import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import mlflow
import os
import logging
from typing import Union, List
from models import UnconstrainedNet, LipschitzNet, LipschitzLegacyNet
from trainer import Trainer
from data import LHCbMCModule
from loss import FocalLoss, CombinedFocalBCELoss, WeightedBCELoss


def setup(rank, world_size):
    """Initialize distributed training process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training process group"""
    dist.destroy_process_group()


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
    
    # For legacy Lipschitz model
    elif architecture == "lipschitz_legacy":
        # Extract relevant parameters
        lip_const = cfg.model.get("lip_const", 1.0)
        nbody = cfg.model.get("nbody", "TwoBody")
        monotonic = cfg.model.get("monotonic", False)
        
        return LipschitzLegacyNet(
            input_dim=input_dim,
            feature_names=feature_names,
            lip_const=lip_const,
            monotonic=monotonic,
            nbody=nbody,
            features_config_path=cfg.features_config_path
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
        
        return True
    except Exception as e:
        logging.error(f"Error logging model with metadata: {str(e)}")
        return False


def train_process(rank, world_size, cfg):
    """Training process for each GPU"""
    logger = logging.getLogger(__name__)
    
    # Set up distributed training
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Only the master process handles MLflow tracking
    is_master = rank == 0
    
    # MLflow experiment setup (only on master process)
    if is_master:
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
    
    try:
        # Setup data
        data_module = LHCbMCModule(cfg.paths.train_data, cfg.paths.test_data)
        
        # Set up DataLoaders with DistributedSampler
        data_module.setup_distributed(
            batch_size=cfg.training.batch_size,
            scale_factor=cfg.training.get("training_data_scale_factor", 1.0),
            ratio=cfg.training.get("sb_ratio", 0.01),
            rank=rank,
            world_size=world_size,
            apply_preprocessing=cfg.training.get("apply_preprocessing", True),
            balance_train_sample=cfg.training.get("balance_train_sample", True),
        )
        
        if is_master:
            logger.info(f"Input features: {data_module.feature_cols}")
            logger.info(f"Input dimension: {data_module.input_dim}")

        # Create model
        model = get_model(cfg, data_module.input_dim, data_module.feature_cols)
        model = model.to(device)
        
        # Wrap model with DistributedDataParallel
        ddp_model = DDP(model, device_ids=[rank])
        
        if is_master and hasattr(model, 'print_architecture_details'):
            model.print_architecture_details()

        # Setup training components
        optimizer = get_optimizer(cfg, ddp_model)
        scheduler = get_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg)

        # Create trainer and train
        trainer = Trainer(
            model=ddp_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            feature_names=data_module.feature_cols,
            grad_clip_val=cfg.training.get("grad_clip_val", None),
            use_mixed_precision=cfg.training.get("use_mixed_precision", False),
            is_distributed=True,
            is_master=is_master,
        )
        
        # Provide access to the data module for visualization
        trainer.data_module = data_module

        # Train the model
        if is_master:
            with mlflow.start_run():
                # Log parameters if master process
                for section in ["model", "optimizer", "training"]:
                    if hasattr(cfg, section):
                        for key, value in dict(getattr(cfg, section)).items():
                            # Skip complex nested structures
                            if not isinstance(value, (dict, list)) or isinstance(value, (str, int, float, bool)):
                                mlflow.log_param(f"{section}.{key}", value)
                
                # Add distributed training parameters
                mlflow.log_param("distributed.world_size", world_size)
                
                history = trainer.train(
                    train_loader=data_module.train_loader,
                    val_loader=data_module.test_loader,
                    num_epochs=cfg.training.num_epochs,
                    metrics_cfg=cfg.metrics,
                    early_stopping_patience=cfg.training.get("early_stopping_patience", None),
                )
                
                # Log metrics and model on master process
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

                # Save the model (unwrap from DDP first)
                log_model_with_metadata(ddp_model.module, data_module, cfg)
        else:
            # Non-master processes just train
            trainer.train(
                train_loader=data_module.train_loader,
                val_loader=data_module.test_loader,
                num_epochs=cfg.training.num_epochs,
                metrics_cfg=cfg.metrics,
                early_stopping_patience=cfg.training.get("early_stopping_patience", None),
            )
        
        logger.info(f"Process {rank}: Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Process {rank}: An error occurred: {str(e)}")
        raise e
    finally:
        # Clean up distributed training resources
        cleanup()


def single_gpu_training(cfg: DictConfig) -> None:
    """Original single-GPU training code"""
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
                apply_preprocessing=cfg.training.get("apply_preprocessing", True),
                balance_train_sample=cfg.training.get("balance_train_sample", False),
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


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Check if distributed training is enabled (default to True if multiple GPUs available)
    use_distributed = cfg.get("use_distributed", True)
    
    if world_size > 1 and use_distributed:
        print(f"Distributed training enabled with {world_size} GPUs")
        # Start multiprocessing for distributed training
        mp.spawn(
            train_process, 
            args=(world_size, cfg),
            nprocs=world_size, 
            join=True
        )
    else:
        if world_size > 1 and not use_distributed:
            print(f"Distributed training disabled (use_distributed=False). Using single GPU mode with {world_size} GPUs available.")
        else:
            print(f"Single GPU training (found {world_size} GPU)")
        
        # Use original single-GPU training code
        single_gpu_training(cfg)


if __name__ == "__main__":
    main()