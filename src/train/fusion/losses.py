import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        """
        Args:
            alpha: Class weights (optional)
            gamma: Focusing parameter (higher = more focus on hard examples)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        
        Returns:
            Focal loss value
        """
        num_classes = inputs.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            # One-hot encode targets
            one_hot = F.one_hot(targets, num_classes).float()
            
            # Apply smoothing
            smooth_targets = one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
            
            # Compute log probabilities
            log_probs = F.log_softmax(inputs, dim=1)
            
            # Cross entropy with soft targets
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Compute focal weight: (1 - p_t)^gamma
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()