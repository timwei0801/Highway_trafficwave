# -- coding: utf-8 --
import torch


def masked_mae_loss(pred, label):
    """Masked MAE loss — ignores positions where label == 0.

    Args:
        pred:  [B, N, Q]
        label: [B, N, Q]
    Returns:
        scalar loss
    """
    mask = (label != 0).float()
    mask = mask / mask.mean()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(pred - label) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()
