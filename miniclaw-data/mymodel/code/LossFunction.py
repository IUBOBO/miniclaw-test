import torch
import torch.nn as nn
import torch.nn.functional as F

# class MultiTaskLoss(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3,n_classes=1):
#         """
#         多任务损失函数
#         :param alpha: 边界损失权重（默认0.7）
#         :param beta: 掩码损失权重（默认0.3）
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#
#         self.boundary_criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
#         self.mask_criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
#
#     def forward(self, pred_boundary, pred_mask, gt_boundary, gt_mask):
#         """
#         计算多任务损失
#         :param pred_boundary: 模型预测的边界 [B,1,H,W]
#         :param pred_mask: 模型预测的掩码 [B,1,H,W]
#         :param gt_boundary: 真实边界标签 [B,1,H,W]（值范围0-1）
#         :param gt_mask: 真实掩码标签 [B,1,H,W]（值范围0-1）
#         :return: 加权总损失
#         """
#         # 边界损失
#         boundary_loss = self.boundary_criterion(pred_boundary, gt_boundary)
#         # 掩码损失
#         mask_loss = self.mask_criterion(pred_mask, gt_mask)
#         # 加权总损失
#         total_loss = self.alpha * boundary_loss + self.beta * mask_loss
#         return {
#             'total_loss': total_loss,
#             'boundary_loss': boundary_loss.item(),
#             'mask_loss': mask_loss.item()
#         }


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, bce_weight=0.5, dice_weight=1):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, gt_mask):
        bce = self.bce_loss(pred_mask, gt_mask)

        pred_prob = torch.sigmoid(pred_mask)
        pred_prob_flat = pred_prob.view(pred_prob.size(0), -1)
        gt_mask_flat = gt_mask.view(gt_mask.size(0), -1)

        intersection = (pred_prob_flat * gt_mask_flat).sum(1)
        dice = 1 - (2. * intersection + self.smooth) / (
            pred_prob_flat.sum(1) + gt_mask_flat.sum(1) + self.smooth
        )
        dice = dice.mean()

        mix_loss = self.bce_weight * bce + self.dice_weight * dice
        return mix_loss

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.boundary_criterion = nn.BCEWithLogitsLoss()
        self.mask_criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=1)

    def forward(self, pred_boundary, pred_mask, gt_boundary, gt_mask):
        boundary_loss = self.boundary_criterion(pred_boundary, gt_boundary)
        mask_loss = self.mask_criterion(pred_mask, gt_mask)

        total_loss = self.alpha * mask_loss + self.beta * boundary_loss

        return {
                'total_loss': total_loss,
                'mask_loss': mask_loss.item(),
                'boundary_loss': boundary_loss.item(),

        }

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class DiceBCELoss(nn.Module):
#     def __init__(self, smooth=1.):
#         super().__init__()
#         self.smooth = smooth
#
#     def forward(self, pred, target):
#
#         bce = F.binary_cross_entropy_with_logits(pred, target)
#         pred_sigmoid = torch.sigmoid(pred)
#         pred_flat = pred_sigmoid.view(-1)
#         target_flat = target.view(-1)
#         intersection = (pred_flat * target_flat).sum()
#         dice = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
#
#         return bce + dice
#
#
#
# class MultiTaskLoss(nn.Module):
#     """
#     双任务损失函数：边界预测 + 掩码分割
#     输入尺寸：(B, 3, 512, 512)
#     输出：
#         pred_boundary: (B, 1, H, W)
#         pred_mask: (B, 1, H, W)
#     """
#     def __init__(self, alpha=0.3, beta=0.7):
#         """
#         :param alpha: 边界损失权重
#         :param beta: 掩码损失权重
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#
#         self.boundary_criterion = nn.BCEWithLogitsLoss()
#         self.mask_criterion = DiceBCELoss()
#
#     def forward(self, pred_boundary, pred_mask, gt_boundary, gt_mask):
#         """
#         :param pred_boundary: 模型预测边界 (B,1,H,W)
#         :param pred_mask: 模型预测掩码 (B,1,H,W)
#         :param gt_boundary: 真实边界标签 (B,1,H,W)
#         :param gt_mask: 真实掩码标签 (B,1,H,W)
#         :return: dict 包含总损失与每项子损失
#         """
#         boundary_loss = self.boundary_criterion(pred_boundary, gt_boundary)
#         mask_loss = self.mask_criterion(pred_mask, gt_mask)
#         total_loss = self.alpha * boundary_loss + self.beta * mask_loss
#
#         return {
#             'total_loss': total_loss,
#             'boundary_loss': boundary_loss.item(),
#             'mask_loss': mask_loss.item()
#         }
