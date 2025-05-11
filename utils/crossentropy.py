import torch
import torch.nn.functional as F

def weighted_crossentropy(y_pred, y_true, weights=torch.tensor([1.0, 1.0, 10.0])):
    """
    y_true: one-hot encoded (N, C, H, W)
    y_pred: raw logits (N, C, H, W)
    weights: per-class weight tensor
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match: one-hot y_true and raw y_pred")

    loss = F.cross_entropy(
        y_pred,
        y_true.argmax(dim=1),
        weight=weights.to(y_pred.device),
        reduction='mean'
    )
    return loss
