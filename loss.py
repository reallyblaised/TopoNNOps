import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """Binary Cross Entropy with inverse class weighting.

    This loss automatically computes class weights inversely proportional to class frequency,
    helping to address class imbalance problems.

    For binary classification, the positive class weight is:
    w_pos = num_negative_samples / num_positive_samples
    """

    def __init__(self, reduction="mean", eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps  # Small value to prevent division by zero

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate inverse class frequency weights
        num_pos = torch.sum(targets) + self.eps
        num_neg = targets.size(0) - num_pos + self.eps

        # Positive class weight (weight applied to positive samples)
        pos_weight = num_neg / num_pos
        pos_weight_tensor = torch.tensor([pos_weight], device=inputs.device)

        # Use PyTorch's built-in weighted BCE
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight_tensor, reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance by down-weighting easy examples.

    Focal Loss adds a factor (1 - pt)^gamma to standard cross-entropy loss,
    where pt is the probability of the correct class and gamma is a focusing
    parameter. This reduces the impact of easy-to-classify examples and focuses
    training on hard examples.

    Formula:
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha (float, optional): Weighting factor for the rare class. Default: 1.0
        gamma (float, optional): Focusing parameter. Higher gamma gives more weight
                                 to hard examples. Default: 2.0
        reduction (str, optional): Reduction method. Default: "mean"
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Probability of being correct (pt)
        pt = torch.exp(-bce_loss)

        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # "none"
            return focal_loss


class CombinedFocalBCELoss(nn.Module):
    """Combines inverse class weighting with focal loss for severe imbalance.

    This loss function applies both inverse class frequency weighting and focal loss
    modulation, addressing class imbalance from two complementary angles:
    1. Class frequency weighting directly counteracts imbalance in the dataset
    2. Focal loss term focuses on hard examples (often from minority classes)

    Formula:
    L = -[(N_neg/N_pos) * y * (1-p)^gamma * log(p) + (1-y) * p^gamma * log(1-p)]

    Args:
        gamma (float, optional): Focusing parameter. Default: 2.0
        alpha (float, optional): If provided, fixed weight for positive class instead
                                of dynamically computing from class frequencies.
        reduction (str, optional): Reduction method. Default: "mean"
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Fixed weight (optional)
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            # Calculate inverse class frequency for positive class
            num_pos = torch.sum(targets) + self.eps
            num_neg = targets.size(0) - num_pos + self.eps
            pos_weight = num_neg / num_pos
        else:
            pos_weight = self.alpha

        pos_weight_tensor = torch.tensor([pos_weight], device=inputs.device)

        # BCE loss with logits and class weighting
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight_tensor, reduction="none"
        )

        # Add focal term
        pt = torch.exp(-bce_loss)  # probability of being correct
        focal_term = (1 - pt) ** self.gamma

        # Combine
        loss = focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DisCoLoss(nn.Module):
    """
    Distance Correlation loss for decorrelating neural network predictions
    from spectator variables like particle lifetime.
    
    References:
      - https://arxiv.org/abs/2001.05310
    """
    
    def __init__(self, lambda_disco: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.lambda_disco = lambda_disco
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                spectators: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined BCE + DisCo loss.
        
        Args:
            inputs: Model predictions (logits) [batch_size, 1]
            targets: True binary labels [batch_size, 1]
            spectators: Spectator variables for decorrelation [batch_size, n_spectators]
        """
        # Standard BCE loss
        bce = self.bce_loss(inputs, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs).view(-1)
        
        # Initialize total decorrelation term
        disco_term = 0.0
        n_spectators = spectators.shape[1] if len(spectators.shape) > 1 else 1
        
        # Handle multiple spectator variables if provided
        if len(spectators.shape) > 1:
            for i in range(n_spectators):
                disco_term += self._distance_correlation(probs, spectators[:, i])
        else:
            # Single spectator variable
            disco_term = self._distance_correlation(probs, spectators)
            
        # Normalize by number of spectators
        if n_spectators > 1:
            disco_term /= n_spectators
            
        # Combined loss with weighting
        total_loss = bce + self.lambda_disco * disco_term
        
        return total_loss
    
    def _distance_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance correlation between two 1D tensors.
        
        Implementation follows the definition in Székely et al., 
        "Measuring and testing independence by correlation of distances"
        """
        # Ensure inputs are 1D
        x = x.view(-1)
        y = y.view(-1)
        n = x.size(0)
        
        # For numerical stability
        eps = 1e-8
        
        # Compute pairwise Euclidean distances
        x_dists = torch.cdist(x.view(-1, 1), x.view(-1, 1), p=2)
        y_dists = torch.cdist(y.view(-1, 1), y.view(-1, 1), p=2)
        
        # Double centering of distance matrices
        x_dists_centered = self._double_center(x_dists)
        y_dists_centered = self._double_center(y_dists)
        
        # Calculate distance covariance and variances
        dCov_xy = torch.sqrt(torch.maximum(torch.mean(x_dists_centered * y_dists_centered), torch.tensor(eps, device=x.device)))
        dVar_x = torch.sqrt(torch.maximum(torch.mean(x_dists_centered * x_dists_centered), torch.tensor(eps, device=x.device)))
        dVar_y = torch.sqrt(torch.maximum(torch.mean(y_dists_centered * y_dists_centered), torch.tensor(eps, device=x.device)))
        
        # Calculate distance correlation
        dCor = dCov_xy / (dVar_x * dVar_y)
        
        return dCor
    
    def _double_center(self, D: torch.Tensor) -> torch.Tensor:
        """
        Double-center a distance matrix.
        
        Args:
            D: Distance matrix [n, n]
            
        Returns:
            Double-centered matrix
        """
        n = D.shape[0]
        row_means = torch.mean(D, dim=1, keepdim=True)
        col_means = torch.mean(D, dim=0, keepdim=True)
        total_mean = torch.mean(D)
        
        return D - row_means - col_means + total_mean
    

class ConditionalDisCoLoss(nn.Module):
    """
    Distance Correlation loss for decorrelating neural network predictions from 
    spectator variables, conditionally applied based on spectator values.
    
    This implementation calculates DisCo loss only on samples where the spectator
    variable exceeds a specified threshold (e.g., lifetime > 5). This allows for
    targeted decorrelation in specific regions of the spectator distribution.
    
    References:
    - https://arxiv.org/abs/2001.05310 (Original DisCo paper)
    """
    
    def __init__(
        self, 
        lambda_disco: float = 1.0, 
        threshold: float = 5.0, 
        reduction: str = "mean",
        apply_to_all_spectators: bool = False
    ):
        """
        Initialize the conditional DisCo loss.
        
        Args:
            lambda_disco: Weight of the decorrelation term
            threshold: Threshold value for the spectator to apply DisCo loss
            reduction: Loss reduction method ("mean" or "sum")
            apply_to_all_spectators: If True, condition applies only when all 
                                    spectators exceed threshold; if False, 
                                    applies when any spectator exceeds threshold
        """
        super().__init__()
        self.lambda_disco = lambda_disco
        self.threshold = threshold
        self.reduction = reduction
        self.apply_to_all_spectators = apply_to_all_spectators
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        spectators: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined BCE + conditional DisCo loss.
        
        Args:
            inputs: Model predictions (logits) [batch_size, 1]
            targets: True binary labels [batch_size, 1]
            spectators: Spectator variables [batch_size, n_spectators] or [batch_size]
                        (e.g., lifetime values)
        
        Returns:
            Combined loss value
        """
        # Standard BCE loss
        bce = self.bce_loss(inputs, targets)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs).view(-1)
        
        # Create mask for conditioning
        if len(spectators.shape) > 1:
            # Multiple spectator variables
            if self.apply_to_all_spectators:
                # Apply DisCo only when ALL spectators exceed threshold
                condition_mask = torch.all(spectators >= self.threshold, dim=1)
            else:
                # Apply DisCo when ANY spectator exceeds threshold
                condition_mask = torch.any(spectators >= self.threshold, dim=1)
        else:
            # Single spectator variable
            condition_mask = spectators >= self.threshold
        
        # Check if we have any samples meeting the condition
        if not torch.any(condition_mask):
            # No samples meet condition, return only BCE loss
            return bce
        
        # Get the samples that meet the condition
        condition_count = torch.sum(condition_mask)
        filtered_probs = probs[condition_mask]
        
        # Initialize DisCo term
        disco_term = 0.0
        
        # Process each spectator variable
        if len(spectators.shape) > 1:
            # Multiple spectator variables case
            n_spectators = spectators.shape[1]
            
            for i in range(n_spectators):
                filtered_spectator = spectators[condition_mask, i]
                # Calculate DisCo for this spectator
                curr_disco = self._distance_correlation(filtered_probs, filtered_spectator)
                disco_term += curr_disco
                
            # Average across spectators if there are multiple
            disco_term /= n_spectators
        else:
            # Single spectator variable case
            filtered_spectator = spectators[condition_mask]
            disco_term = self._distance_correlation(filtered_probs, filtered_spectator)
        
        # Scale DisCo term by lambda and proportion of samples used
        # This ensures the loss contribution scales appropriately with the 
        # fraction of samples meeting the condition
        scaled_disco = self.lambda_disco * disco_term #* (condition_count / len(probs))
        
        # Combined loss
        total_loss = bce + scaled_disco
        
        return total_loss
    
    def _distance_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance correlation between two 1D tensors.
        
        Args:
            x: First tensor [n_samples] (typically model predictions)
            y: Second tensor [n_samples] (typically spectator values)
            
        Returns:
            Distance correlation value
        """
        # Ensure inputs are 1D
        x = x.view(-1)
        y = y.view(-1)
        
        # For numerical stability
        eps = 1e-8
        n = x.size(0)
        
        # Compute pairwise Euclidean distances
        x_dist = torch.cdist(x.view(-1, 1), x.view(-1, 1), p=2)
        y_dist = torch.cdist(y.view(-1, 1), y.view(-1, 1), p=2)
        
        # Double centering of distance matrices
        x_dist_centered = self._double_center(x_dist)
        y_dist_centered = self._double_center(y_dist)
        
        # Calculate distance covariance and variances
        dCov_xy = torch.sqrt(torch.maximum(
            torch.mean(x_dist_centered * y_dist_centered), 
            torch.tensor(eps, device=x.device)
        ))
        
        dVar_x = torch.sqrt(torch.maximum(
            torch.mean(x_dist_centered * x_dist_centered), 
            torch.tensor(eps, device=x.device)
        ))
        
        dVar_y = torch.sqrt(torch.maximum(
            torch.mean(y_dist_centered * y_dist_centered), 
            torch.tensor(eps, device=x.device)
        ))
        
        # Calculate distance correlation
        dCor = dCov_xy / (dVar_x * dVar_y)
        
        return dCor
    
    def _double_center(self, D: torch.Tensor) -> torch.Tensor:
        """
        Double-center a distance matrix.
        
        Args:
            D: Distance matrix [n, n]
            
        Returns:
            Double-centered matrix
        """
        n = D.size(0)
        
        # Calculate row and column means
        row_mean = torch.mean(D, dim=1, keepdim=True)  # [n, 1]
        col_mean = torch.mean(D, dim=0, keepdim=True)  # [1, n]
        total_mean = torch.mean(D)  # scalar
        
        # Apply double centering
        centered = D - row_mean - col_mean + total_mean
        
        return centered