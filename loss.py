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
