import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()  

    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        ce_loss = self.bce_loss(outputs, targets.float())  

        # Soft IoU loss
        probs = torch.sigmoid(outputs)  
        intersection = (probs * targets).sum(dim=(1, 2))
        union = (probs + targets - probs * targets).sum(dim=(1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()

        lambda_iou = 1 - iou.mean()  
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * lambda_iou * iou_loss

        return total_loss


