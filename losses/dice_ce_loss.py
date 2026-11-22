import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceCELoss(nn.Module):
    """
    Sum of Dice Loss and Soft Cross Entropy Loss.
    Loss = DiceLoss + SoftCrossEntropyLoss
    """
    def __init__(self, smooth_factor, ignore_index, mode="multiclass", from_logits=True):
        super(DiceCELoss, self).__init__()

        # Initialize the two losses internally
        self.dice = smp.losses.DiceLoss(mode=mode, from_logits=from_logits, ignore_index=ignore_index)
        self.ce = smp.losses.SoftCrossEntropyLoss(smooth_factor=smooth_factor, ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, H, W) - raw model outputs
            targets: Tensor of shape (N, H, W) or (N, C, H, W) depending on dataset
        """
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets)
        
        return loss_dice + loss_ce