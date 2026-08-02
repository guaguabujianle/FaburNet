import torch
import torch.nn.functional as F
import torch.nn as nn


# class DiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.smooth = 1e-6

#     def forward(self, y_pred, y_truth):
#         y_pred_f = torch.flatten(y_pred, start_dim=0, end_dim=2)
#         y_truth_f = torch.flatten(y_truth, start_dim=0, end_dim=2)
#         # print(y_pred_f.shape,y_truth_f.shape)
#         dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
#                 y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
#         dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
#                 y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

#         dice1.requires_grad_(True)
#         dice2.requires_grad_(True)
#         return 1 - (dice1 + dice2) / 2


# class CELoss(nn.Module):
#     def __init__(self):
#         super(CELoss, self).__init__()

#     def forward(self, y_pred, y_truth):
#         """
#         :param y_pred:   one-hot
#         :param y_truth:  one-hot
#         :return:
#         """
#         y_pred_f = torch.flatten(y_pred.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

#         target_t = torch.flatten(y_truth, start_dim=0, end_dim=2)
#         loss_ce = F.cross_entropy(y_pred_f, target_t.argmax(dim=1))
#         return loss_ce


# class CEWeightedLoss(nn.Module):
#     def __init__(self):
#         super(CEWeightedLoss, self).__init__()

#     def forward(self, y_pred, y_truth, weight_map):
#         weight_map = weight_map.squeeze(1)  # [B, H, W]
#         if y_truth.dim() == 4 and y_truth.shape[-1] > 1:
#             # B,256,256,3
#             target = y_truth.argmax(dim=-1)  # [B, H, W]
#         else:
#             target = y_truth

#         n_pixels = y_pred.shape[0] * y_pred.shape[2] * y_pred.shape[3]  # B*H*W
#         ce_loss = F.cross_entropy(y_pred, target, reduction='none', ignore_index=0)

#         loss = (ce_loss * weight_map).sum() / n_pixels
#         # loss = (ce_loss * weight_map).sum() / weight_map.sum()

#         return loss


class DiceLoss(nn.Module):
    """Soft multiclass Dice loss for segmentation logits.

    Class 0 is treated as background and is excluded from the Dice average.
    """

    def __init__(self, num_classes=3, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_pred, y_truth):
        if y_pred.ndim != 4:
            raise ValueError("y_pred must have shape [B, C, H, W]")
        if y_pred.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} prediction channels, "
                f"but received {y_pred.shape[1]}"
            )

        # The training loader supplies [B, H, W], but accepting [B, 1, H, W]
        # makes the loss safe to use directly with an unsqueezed mask as well.
        if y_truth.ndim == 4 and y_truth.shape[1] == 1:
            y_truth = y_truth.squeeze(1)
        if y_truth.ndim != 3:
            raise ValueError("y_truth must have shape [B, H, W] or [B, 1, H, W]")

        probabilities = F.softmax(y_pred, dim=1)
        target_onehot = F.one_hot(
            y_truth.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        # Compute Dice for the two foreground classes only.
        probabilities = probabilities[:, 1:]
        target_onehot = target_onehot[:, 1:]
        reduce_dims = (0, 2, 3)

        intersection = (probabilities * target_onehot).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + target_onehot.sum(
            dim=reduce_dims
        )
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class CELoss(nn.Module):
    """Standard multiclass cross-entropy for integer segmentation masks."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_truth):
        if y_truth.ndim == 4 and y_truth.shape[1] == 1:
            y_truth = y_truth.squeeze(1)
        return F.cross_entropy(y_pred, y_truth.long())


class CEWeightedLoss(nn.Module):
    """Pixel-weighted cross-entropy over foreground pixels."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_truth, weight_map):
        if y_truth.ndim == 4 and y_truth.shape[1] == 1:
            y_truth = y_truth.squeeze(1)
        target = y_truth.long()

        if weight_map.ndim == 4 and weight_map.shape[1] == 1:
            weight_map = weight_map.squeeze(1)
        weight_map = weight_map.to(device=y_pred.device, dtype=y_pred.dtype)

        ce_loss = F.cross_entropy(
            y_pred, target, reduction="none", ignore_index=0
        )
        effective_weight = weight_map * (target != 0).to(weight_map.dtype)

        return (ce_loss * effective_weight).sum() / effective_weight.sum().clamp_min(
            1e-6
        )
